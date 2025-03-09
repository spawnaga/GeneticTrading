import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os
import datetime


class ActorCriticNet(nn.Module):
    """
    A shared neural network for the actor and critic in Proximal Policy Optimization (PPO).
    - Input: State (observation from the environment).
    - Outputs:
      - `policy_logits`: Logits for action probabilities (actor).
      - `value`: State value estimate (critic).
    """

    def __init__(self, input_dim, hidden_dim, action_dim, device="cpu"):
        """
        Initialize the ActorCriticNet.

        Args:
            input_dim (int): Dimension of the input observation space.
            hidden_dim (int): Number of units in hidden layers.
            action_dim (int): Number of possible discrete actions.
            device (str or torch.device): Device to run the model on ('cpu' or 'cuda').
        """
        super(ActorCriticNet, self).__init__()
        self.device = torch.device(device) if isinstance(device, str) else device

        # Shared base network with two hidden layers for feature extraction
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ).to(self.device)

        # Actor head: Outputs logits for action probabilities
        self.policy_head = nn.Linear(hidden_dim, action_dim).to(self.device)
        # Critic head: Outputs state value estimate
        self.value_head = nn.Linear(hidden_dim, 1).to(self.device)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input state tensor of shape (batch_size, input_dim).

        Returns:
            tuple: (policy_logits, value) where:
                - policy_logits (torch.Tensor): Logits for action probabilities.
                - value (torch.Tensor): Estimated state value.
        """
        x = x.to(self.device)
        base_out = self.base(x)
        policy_logits = self.policy_head(base_out)
        value = self.value_head(base_out)
        return policy_logits, value

    def act(self, state):
        """
        Select an action based on the state during evaluation (deterministic).

        Args:
            state (np.ndarray or torch.Tensor): Input state.

        Returns:
            int: Chosen action index.
        """
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        state = state.to(self.device)
        with torch.no_grad():
            policy_logits, _ = self.forward(state)
            action = torch.argmax(policy_logits, dim=-1).item()
        return action

    def save_model(self, file_path):
        """
        Save the model's state dictionary to a file.

        Args:
            file_path (str): Path where the model will be saved.
        """
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path} at {datetime.datetime.now()}")

    def load_model(self, file_path):
        """
        Load the model's state dictionary from a file if it exists.

        Args:
            file_path (str): Path to the saved model file.
        """
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path, map_location=self.device))
            print(f"Model loaded from {file_path}")
        else:
            print(f"No model found at {file_path}. Initializing with random weights.")


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) Trainer.
    - Manages training of an actor-critic network using PPO.
    - Features:
      - Efficient trajectory collection with NumPy arrays.
      - Generalized Advantage Estimation (GAE) for advantage computation.
      - Mini-batch PPO updates with gradient clipping for stability.
      - Distributed training support (only rank 0 saves models).
      - Progress reporting for integration with main.py.
    """

    def __init__(self, env, input_dim, action_dim, hidden_dim=64, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, update_epochs=4, rollout_steps=2048, batch_size=64, device="cpu",
                 model_save_path="ppo_model.pth", local_rank=0):
        """
        Initialize the PPOTrainer with hyperparameters.

        Args:
            env: The environment (e.g., TradingEnvironment).
            input_dim (int): Dimension of the observation space.
            action_dim (int): Number of discrete actions.
            hidden_dim (int): Size of hidden layers in the network.
            lr (float): Learning rate for the Adam optimizer.
            gamma (float): Discount factor for future rewards.
            gae_lambda (float): GAE smoothing parameter.
            clip_epsilon (float): Clipping parameter for PPO policy update.
            update_epochs (int): Number of epochs per PPO update.
            rollout_steps (int): Number of steps to collect per rollout.
            batch_size (int): Size of mini-batches for PPO updates.
            device (str or torch.device): Device for computation ('cpu' or 'cuda').
            model_save_path (str): Path to save the trained model.
            local_rank (int): Local rank of the process in distributed training.
        """
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model_save_path = model_save_path
        self.local_rank = local_rank

        # Initialize the actor-critic network
        self.model = ActorCriticNet(input_dim, hidden_dim, action_dim, device=self.device)
        self.model.load_model(self.model_save_path)  # Load existing model if available

        # Optimizer for training
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def collect_trajectories(self):
        """
        Collect trajectories by interacting with the environment.

        Returns:
            tuple: (obs, actions, rewards, old_logprobs, values, dones) as NumPy arrays.
        """
        obs = []
        actions = []
        rewards = []
        old_logprobs = []
        values = []
        dones = []

        state = self.env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        for _ in range(self.rollout_steps):
            # Get policy and value from the model
            with torch.no_grad():
                policy_logits, value = self.model(state_tensor)
                action_dist = Categorical(logits=policy_logits)
                action = action_dist.sample()
                logprob = action_dist.log_prob(action)

            # Store trajectory data
            obs.append(state_tensor.cpu().numpy()[0])
            actions.append(action.item())
            old_logprobs.append(logprob.item())
            values.append(value.item())

            # Step the environment
            next_state, reward, done, _ = self.env.step(action.item())
            rewards.append(np.clip(reward, -10, 10))  # Clip rewards for stability in trading
            dones.append(done)

            # Prepare next state
            state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            if done:
                state_tensor = torch.tensor(self.env.reset(), dtype=torch.float32).unsqueeze(0).to(self.device)

        # Convert lists to NumPy arrays
        return (np.array(obs, dtype=np.float32),
                np.array(actions, dtype=np.int32),
                np.array(rewards, dtype=np.float32),
                np.array(old_logprobs, dtype=np.float32),
                np.array(values, dtype=np.float32),
                np.array(dones, dtype=np.bool_))

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE) and returns.

        Args:
            rewards (np.ndarray): Array of rewards.
            values (np.ndarray): Array of value estimates (including next_value at the end).
            dones (np.ndarray): Array of done flags.
            next_value (float): Value estimate of the final state.

        Returns:
            tuple: (advantages, returns) as NumPy arrays.
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        # Compute GAE backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] + self.gamma * (0 if dones[t] else next_value) - values[t]
            else:
                delta = rewards[t] + self.gamma * (0 if dones[t] else values[t + 1]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (0 if dones[t] else gae)
            advantages[t] = gae

        # Normalize advantages for training stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values[:-1]  # Exclude next_value
        return advantages, returns

    def train_step(self):
        """
        Perform one PPO training step:
        - Collect trajectories.
        - Compute advantages and returns.
        - Update the network using mini-batch PPO updates.

        Returns:
            float: Mean reward of the rollout for logging.
        """
        # Collect rollout data
        obs, actions, rewards, old_logprobs, values, dones = self.collect_trajectories()

        # Get the value of the final state
        last_state = self.env.reset() if dones[-1] else self.env.current_state()
        last_state_tensor = torch.tensor(last_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, next_value = self.model(last_state_tensor)
        values = np.append(values, next_value.item())

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_value.item())

        # Convert data to tensors
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        old_logprobs_tensor = torch.tensor(old_logprobs, dtype=torch.float32).to(self.device)
        advantage_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        return_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # Mini-batch PPO updates
        dataset_size = len(obs)
        indices = torch.arange(dataset_size)

        for _ in range(self.update_epochs):
            perm = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                batch_indices = perm[start:start + self.batch_size]
                if len(batch_indices) == 0:
                    continue

                # Extract mini-batch
                obs_batch = obs_tensor[batch_indices]
                action_batch = action_tensor[batch_indices]
                old_logprobs_batch = old_logprobs_tensor[batch_indices]
                advantage_batch = advantage_tensor[batch_indices]
                return_batch = return_tensor[batch_indices]

                # Forward pass
                policy_logits, value_est = self.model(obs_batch)
                dist = Categorical(logits=policy_logits)
                new_logprobs = dist.log_prob(action_batch)
                entropy = dist.entropy().mean()

                # PPO policy loss
                ratio = torch.exp(new_logprobs - old_logprobs_batch)
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(value_est.squeeze(), return_batch)

                # Total loss with entropy bonus
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)  # Gradient clipping
                self.optimizer.step()

        # Save model only on rank 0
        if self.local_rank == 0:
            self.model.save_model(self.model_save_path)

        return np.mean(rewards)

    def train(self, total_timesteps, callback=None):
        """
        Train the PPO agent for a specified number of timesteps.

        Args:
            total_timesteps (int): Total number of environment steps to train for.
            callback (callable, optional): Function to call after each update (e.g., for logging or early stopping).

        Returns:
            ActorCriticNet: The trained model.
        """
        n_updates = total_timesteps // self.rollout_steps
        for update in range(n_updates):
            mean_reward = self.train_step()
            if self.local_rank == 0:
                print(f"Update {update + 1}/{n_updates}, Mean Reward: {mean_reward:.5f}")
            if callback is not None:
                callback(update, mean_reward, self.model)  # Report progress to main.py
        if self.local_rank == 0:
            print(f"Model saved to {self.model_save_path} at {datetime.datetime.now()}")
            print("Training completed.")
        return self.model

    def current_state(self):
        """
        Get the current state of the environment (for compatibility with TradingEnvironment).

        Returns:
            np.ndarray: Current state.
        """
        return self.env.current_state() if hasattr(self.env, 'current_state') else self.env.reset()
