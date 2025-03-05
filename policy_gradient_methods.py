import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class ActorCriticNet(nn.Module):
    """
    Shared network for actor + critic (PPO style).
    This network takes the state as input and outputs:
    - `policy_logits`: The logits for the action probabilities (used for selecting actions).
    - `value`: The state value estimate (used for advantage estimation and critic loss).
    """

    def __init__(self, input_dim, hidden_dim, action_dim, device="cpu"):
        super(ActorCriticNet, self).__init__()
        self.device = device  # Store the device (CPU/GPU)

        # Shared base network
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ).to(self.device)  # Ensure model is on the correct device

        # Policy head (outputs logits for action selection)
        self.policy_head = nn.Linear(hidden_dim, action_dim).to(self.device)
        # Value head (outputs a single value estimate)
        self.value_head = nn.Linear(hidden_dim, 1).to(self.device)

    def forward(self, x):
        """
        Forward pass for both policy and value networks.
        - Ensures input `x` is on the correct device.
        - Returns `policy_logits` and `value` as outputs.
        """
        x = x.to(self.device)  # Move input tensor to correct device
        base_out = self.base(x)
        policy_logits = self.policy_head(base_out)
        value = self.value_head(base_out)
        return policy_logits, value

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path, map_location=self.device))
            print(f"ActorCriticNet model loaded from {file_path}")
        else:
            print(f"No ActorCriticNet model found at {file_path}. Starting fresh.")


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) Trainer
    - This class handles training an actor-critic network using PPO updates.
    - Implements:
      - Rollout collection
      - Generalized Advantage Estimation (GAE)
      - PPO policy and value loss updates
    """

    def __init__(self, env, input_dim, action_dim, hidden_dim=64, lr=3e-4, gamma=0.99, clip_epsilon=0.2,
                 update_epochs=4, rollout_steps=2048, device="cpu", model_save_path="ppo_model.pth"):
        """
        Initializes PPOTrainer with the given hyperparameters.
        - `env`: Trading environment
        - `input_dim`: Observation space dimension
        - `action_dim`: Number of actions
        - `hidden_dim`: Hidden layer size
        - `lr`: Learning rate for Adam optimizer
        - `gamma`: Discount factor
        - `clip_epsilon`: PPO clipping parameter
        - `update_epochs`: Number of PPO update epochs per rollout
        - `rollout_steps`: Number of environment steps per batch
        - `device`: Device (CPU/GPU)
        """
        self.env = env
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.rollout_steps = rollout_steps
        self.device = device  # Store device (CPU/GPU)
        self.model_save_path = model_save_path

        # Initialize actor-critic network
        self.model = ActorCriticNet(input_dim, hidden_dim, action_dim, device=device)
        try:
            self.model.load_model(self.model_save_path)
        except FileNotFoundError:
            print(f"No ActorCriticNet model found at {self.model_save_path}. Starting fresh.")

        # Define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def collect_trajectories(self):
        """
        Runs the policy in the environment for `rollout_steps`.
        Collects:
        - Observations
        - Actions
        - Log probabilities of actions
        - Rewards
        - Value estimates
        - Done flags
        """
        obs_list = []
        action_list = []
        logprob_list = []
        reward_list = []
        value_list = []
        done_list = []

        obs = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  # Move obs to model's device

        for _ in range(self.rollout_steps):
            policy_logits, value = self.model(obs)  # obs is already on device

            # Sample an action from policy
            action_dist = torch.distributions.Categorical(logits=policy_logits)
            action = action_dist.sample()
            logprob = action_dist.log_prob(action)

            obs_list.append(obs.cpu().numpy())  # Convert back to NumPy to store
            action_list.append(action.item())
            logprob_list.append(logprob.item())
            value_list.append(value.item())

            # Step environment
            obs, reward, done, _ = self.env.step(action.item())
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  # Ensure new state is on device

            # Normalize rewards to prevent numerical instability
            reward = np.clip(reward, -1, 1)  # Keep rewards between -1 and 1
            reward_list.append(reward)
            done_list.append(done)

            if done:
                obs = self.env.reset()
                obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(
                    self.device)  # Ensure reset state is on device

        # Convert to NumPy arrays
        obs_list = np.array(obs_list, dtype=np.float32)
        action_list = np.array(action_list, dtype=np.int32)
        reward_list = np.array(reward_list, dtype=np.float32)
        logprob_list = np.array(logprob_list, dtype=np.float32)
        value_list = np.array(value_list, dtype=np.float32)
        done_list = np.array(done_list, dtype=np.bool_)

        return obs_list, action_list, reward_list, logprob_list, value_list, done_list

    def compute_advantages(self, rewards, values, dones, next_value, gamma=0.99, lam=0.95):
        """
        Computes Generalized Advantage Estimation (GAE).1`
        - Uses `gamma` (discount factor) and `lam` (GAE factor) for smooth updates.
        - Normalizes advantages for stability.
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * (0 if dones[t] else values[t + 1] if t < len(rewards) - 1 else next_value) - \
                    values[t]
            gae = delta + gamma * lam * (0 if dones[t] else gae)
            advantages[t] = gae

        # Normalize advantages to improve training stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = advantages + values[:-1]
        return advantages, returns

    def train_step(self):
        """
        Performs one PPO training step:
        - Collects a batch of rollouts
        - Computes advantages
        - Updates policy and value networks using PPO loss
        """
        obs_list, action_list, reward_list, old_logprob_list, value_list, done_list = self.collect_trajectories()

        # Compute advantages
        last_state = self.env.current_state()
        last_state_tensor = torch.tensor(last_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, next_value = self.model(last_state_tensor)

        # Compute advantages
        # pad value_list with next_value for convenience
        value_list_pad = np.concatenate([value_list, [next_value.item()]])
        advantages, returns = self.compute_advantages(reward_list, value_list_pad, done_list, next_value.item())

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        obs_tensor = torch.tensor(obs_list, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor(action_list, dtype=torch.long).to(self.device)
        old_logprobs_tensor = torch.tensor(old_logprob_list, dtype=torch.float32).to(self.device)
        advantage_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        return_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # PPO update
        for _ in range(self.update_epochs):
            policy_logits, value_est = self.model(obs_tensor)
            dist = torch.distributions.Categorical(logits=policy_logits)
            new_logprobs = dist.log_prob(action_tensor)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logprobs - old_logprobs_tensor)
            surr1 = ratio * advantage_tensor
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage_tensor

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(value_est.squeeze(-1), return_tensor.unsqueeze(-1))  # ✅ Fix shape issue
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Return some stats
        mean_reward = np.mean(reward_list)
        return mean_reward

    def train(self, total_timesteps):
        """
        Train PPO for total_timesteps.
        We gather rollout_steps each iteration, so number of updates = total_timesteps/rollout_steps
        """
        n_updates = int(total_timesteps // self.rollout_steps)
        for update in range(n_updates):
            mean_reward = self.train_step()
            print(f"Update {update}, mean reward = {mean_reward:.5f}")
            self.model.save_model(self.model_save_path)

        return self.model
