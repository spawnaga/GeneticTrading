import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

class ActorCriticNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, device="cpu"):
        super(ActorCriticNet, self).__init__()
        self.device = torch.device(device)
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ).to(self.device)
        self.policy_head = nn.Linear(hidden_dim, action_dim).to(self.device)
        self.value_head = nn.Linear(hidden_dim, 1).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        base_out = self.base(x)
        policy_logits = self.policy_head(base_out)
        value = self.value_head(base_out)
        return policy_logits, value

    def act(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        state = state.to(self.device)
        with torch.no_grad():
            policy_logits, _ = self.forward(state)
            action = torch.argmax(policy_logits, dim=-1).item()
        return action

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path} at {datetime.datetime.now()}")

    def load_model(self, file_path):
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path, map_location=self.device))
            print(f"Model loaded from {file_path}")
        else:
            print(f"No model found at {file_path}. Using random initialization.")


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) Trainer with a hybrid design.

    This trainer collects long rollouts, computes advantages using Generalized
    Advantage Estimation (GAE), and performs multiple epochs of mini-batch updates.
    """

    def __init__(
            self,
            env,
            input_dim,
            action_dim,
            hidden_dim=64,
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            update_epochs=10,
            rollout_steps=2048,
            batch_size=64,
            device="cpu",
            model_save_path="ppo_actor_critic_model.pth",
            local_rank=0
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.model_save_path = model_save_path
        self.local_rank = local_rank

        self.model = ActorCriticNet(input_dim, hidden_dim, action_dim, device=device)
        self.model.load_model(self.model_save_path)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # ---- New: Set up a learning rate scheduler ----
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        # ---- New: Initial entropy coefficient (will decay over training) ----
        self.entropy_coef = 0.05

    def collect_trajectories(self):
        obs_list = []
        actions_list = []
        rewards_list = []
        logprobs_list = []
        values_list = []
        dones_list = []

        state = self.env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        for _ in range(self.rollout_steps):
            policy_logits, value = self.model(state_tensor)
            dist_ = Categorical(logits=policy_logits)
            action = dist_.sample()
            logprob = dist_.log_prob(action)

            obs_list.append(state_tensor.cpu().numpy()[0])
            actions_list.append(action.item())
            logprobs_list.append(logprob.item())
            values_list.append(value.item())

            next_state, reward, done, _ = self.env.step(action.item())
            rewards_list.append(reward)
            dones_list.append(done)

            if done:
                next_state = self.env.reset()

            state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)

        return (
            np.array(obs_list, dtype=np.float32),
            np.array(actions_list, dtype=np.int32),
            np.array(rewards_list, dtype=np.float32),
            np.array(logprobs_list, dtype=np.float32),
            np.array(values_list, dtype=np.float32),
            np.array(dones_list, dtype=bool)
        )

    def compute_gae(self, rewards, values, dones):
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        # Get next state value by resetting the environment (or use the last state from rollout)
        last_state_tensor = torch.tensor(self.env.reset(), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, next_value = self.model(last_state_tensor)
        next_value = next_value.item()
        values = np.append(values, next_value)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def train_step(self):
        obs, actions, rewards, old_logprobs, values, dones = self.collect_trajectories()
        advantages, returns = self.compute_gae(rewards, values, dones)

        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        old_logprobs_tensor = torch.tensor(old_logprobs, dtype=torch.float32).to(self.device)
        advantage_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        return_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)

        dataset_size = len(obs)
        for epoch in range(self.update_epochs):
            perm = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                if len(batch_idx) == 0:
                    continue

                obs_batch = obs_tensor[batch_idx]
                action_batch = action_tensor[batch_idx]
                old_logprobs_batch = old_logprobs_tensor[batch_idx]
                advantage_batch = advantage_tensor[batch_idx]
                return_batch = return_tensor[batch_idx]

                # Get the new predictions and compute the value loss
                policy_logits, value_est = self.model(obs_batch)
                dist_ = Categorical(logits=policy_logits)
                new_logprobs = dist_.log_prob(action_batch)
                entropy = dist_.entropy().mean()

                ratio = torch.exp(new_logprobs - old_logprobs_batch)
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss with clipping
                value_est = value_est.squeeze()
                value_loss_unclipped = (value_est - return_batch) ** 2

                # Instead of using the full rollout values (values[:-1]), use the mini-batch rollout values:
                old_value_batch = torch.tensor(values, dtype=torch.float32).to(self.device)[batch_idx]
                value_est_clipped = old_value_batch + torch.clamp(value_est - old_value_batch,
                                                                  -self.clip_epsilon, self.clip_epsilon)
                value_loss_clipped = (value_est_clipped - return_batch) ** 2

                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

            # Step the scheduler after each epoch block
            self.scheduler.step()
            self.entropy_coef *= 0.95  # decay entropy coefficient
            print(f"Epoch {epoch + 1}/{self.update_epochs} completed.")

        if self.local_rank == 0:
            if isinstance(self.model, DDP):
                model_to_save = self.model.module
            else:
                model_to_save = self.model
            model_to_save.save_model(self.model_save_path)
        return np.mean(rewards)

    def train(self, total_timesteps, callback=None):
        n_updates = total_timesteps // self.rollout_steps
        print(f"Starting training for {n_updates} updates with {self.rollout_steps} timesteps each.")
        for update in range(n_updates):
            mean_reward = self.train_step()
            print(f"Update {update + 1}/{n_updates}, Mean Reward: {mean_reward:.5f}")
            if callback:
                callback(update, mean_reward, self.model)
        if self.local_rank == 0:
            print(f"Model saved to {self.model_save_path} at {datetime.datetime.now()}")
            print("Training completed.")
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model
