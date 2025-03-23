import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os
import datetime
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
            print(f"No model found at {file_path}. Random init.")


class PPOTrainer:
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
        update_epochs=4,
        rollout_steps=2048,
        batch_size=64,
        device="cpu",
        model_save_path="ppo_model.pth",
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

    def collect_trajectories(self):
        obs = []
        actions = []
        rewards = []
        old_logprobs = []
        values = []
        dones = []

        state = self.env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        for _ in range(self.rollout_steps):
            with torch.no_grad():
                policy_logits, value = self.model(state_tensor)
                dist = Categorical(logits=policy_logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

            obs.append(state_tensor.cpu().numpy()[0])
            actions.append(action.item())
            old_logprobs.append(logprob.item())
            values.append(value.item())

            next_state, reward, done, _ = self.env.step(action.item())
            rewards.append(np.clip(reward, -10, 10))
            dones.append(done)

            if done:
                next_state = self.env.reset()

            state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)

        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(old_logprobs, dtype=np.float32),
            np.array(values, dtype=np.float32),
            np.array(dones, dtype=np.bool_)
        )

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] + self.gamma * (0 if dones[t] else next_value) - values[t]
            else:
                delta = rewards[t] + self.gamma * (0 if dones[t] else values[t + 1]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (0 if dones[t] else gae)
            advantages[t] = gae
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values[:-1]
        return advantages, returns

    def train_step(self):
        obs, actions, rewards, old_logprobs, values, dones = self.collect_trajectories()

        last_state = self.env.reset() if dones[-1] else obs[-1]  # minor fix
        last_state_tensor = torch.tensor(last_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, next_value = self.model(last_state_tensor)
        values = np.append(values, next_value.item())

        advantages, returns = self.compute_gae(rewards, values, dones, next_value.item())

        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        old_logprobs_tensor = torch.tensor(old_logprobs, dtype=torch.float32).to(self.device)
        advantage_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        return_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)

        dataset_size = len(obs)
        for _ in range(self.update_epochs):
            perm = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                batch_indices = perm[start:start + self.batch_size]
                if len(batch_indices) == 0:
                    continue

                obs_batch = obs_tensor[batch_indices]
                action_batch = action_tensor[batch_indices]
                old_logprobs_batch = old_logprobs_tensor[batch_indices]
                advantage_batch = advantage_tensor[batch_indices]
                return_batch = return_tensor[batch_indices]

                policy_logits, value_est = self.model(obs_batch)
                dist = Categorical(logits=policy_logits)
                new_logprobs = dist.log_prob(action_batch)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logprobs - old_logprobs_batch)
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(value_est.squeeze(), return_batch)
                # loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                loss = policy_loss + 0.5 * value_loss - 0.05 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

        if self.local_rank == 0:
            model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
            model_to_save.save_model(self.model_save_path)

        return np.mean(rewards)

    def train(self, total_timesteps, callback=None):
        n_updates = total_timesteps // self.rollout_steps
        for update in range(n_updates):
            mean_reward = self.train_step()
            if self.local_rank == 0:
                print(f"Update {update + 1}/{n_updates}, Mean Reward: {mean_reward:.5f}")
            if callback:
                callback(update, mean_reward, self.model)
        if self.local_rank == 0:
            print(f"Model saved to {self.model_save_path} at {datetime.datetime.now()}")
            print("Training completed.")
        return self.model.module if isinstance(self.model, DDP) else self.model
