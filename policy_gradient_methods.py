import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCriticNet(nn.Module):
    """
    Shared network for actor + critic (PPO style).
    """
    def __init__(self, input_dim, hidden_dim, action_dim, device="cpu"):
        super(ActorCriticNet, self).__init__()
        self.device = device  # ✅ Store the device

        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ).to(self.device)  # ✅ Ensure model is on the correct device

        self.policy_head = nn.Linear(hidden_dim, action_dim).to(self.device)
        self.value_head = nn.Linear(hidden_dim, 1).to(self.device)

    def forward(self, x):
        x = x.to(self.device)  # ✅ Ensure input is on the correct device
        base_out = self.base(x)
        policy_logits = self.policy_head(base_out)
        value = self.value_head(base_out)
        return policy_logits, value

class PPOTrainer:
    def __init__(self, env, input_dim, action_dim, hidden_dim=64, lr=3e-4, gamma=0.99, clip_epsilon=0.2,
                 update_epochs=4, rollout_steps=2048, device="cpu"):
        """
        PPO Training class
        """
        self.env = env
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.rollout_steps = rollout_steps
        self.device = device  # ✅ Store device correctly

        # ✅ Pass `device` to `ActorCriticNet`
        self.model = ActorCriticNet(input_dim, hidden_dim, action_dim, device=device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def collect_trajectories(self):
        """
        Run the policy in the environment for rollout_steps.
        Record states, actions, rewards, etc.
        """
        obs_list = []
        action_list = []
        logprob_list = []
        reward_list = []
        value_list = []
        done_list = []

        obs = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  # ✅ Move obs to model's device

        for _ in range(self.rollout_steps):
            policy_logits, value = self.model(obs)  # ✅ obs is already on device

            # Sample an action from policy
            action_dist = torch.distributions.Categorical(logits=policy_logits)
            action = action_dist.sample()
            logprob = action_dist.log_prob(action)

            obs_list.append(obs.cpu().numpy())  # ✅ Convert back to NumPy to store
            action_list.append(action.item())
            logprob_list.append(logprob.item())
            value_list.append(value.item())

            # Step environment
            obs, reward, done, _ = self.env.step(action.item())
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  # ✅ Ensure new state is on device
            reward_list.append(reward)
            done_list.append(done)

            if done:
                obs = self.env.reset()
                obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  # ✅ Ensure reset state is on device

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
        GAE-lambda advantage computation (Generalized Advantage Estimation).
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * (0 if dones[t] else values[t+1] if t < len(rewards)-1 else next_value) - values[t]
            gae = delta + gamma * lam * (0 if dones[t] else gae)
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns

    def train_step(self):
        obs_list, action_list, reward_list, old_logprob_list, value_list, done_list = self.collect_trajectories()
        # Next value
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
            # Shuffle the indices for mini-batch updates if desired
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
            print(f"Update {update}, mean reward = {mean_reward:.2f}")

        return self.model
