import numpy as np
import gym
from gym import spaces


class TradingEnvironment(gym.Env):
    """
    Trading Environment for reinforcement learning, simulating trading of NQ futures.

    Observations:
        - Scaled features vector (e.g., open, high, low, close, volume, return, ma_10).
    Actions:
        - Discrete: [0=hold, 1=long, 2=short].
    Reward:
        - Profit/Loss (PnL) from mark-to-market position changes.
    """

    def __init__(self, df, initial_balance=100000.0, max_position=1.0):
        """
        Initializes the trading environment.

        Args:
            df (pd.DataFrame): DataFrame containing scaled columns plus 'date_time'.
            initial_balance (float): Starting capital in dollars.
            max_position (float): Maximum allowable position (contracts/shares).
        """
        # Initialize the parent class (gym.Env) for Gym compatibility
        super(TradingEnvironment, self).__init__()

        # Store the DataFrame and reset its index for consistent access
        self.df = df.reset_index(drop=True)

        # Define the maximum step (last valid index in the DataFrame)
        self.max_step = len(self.df) - 1

        # Initialize state variables
        self.current_step = 0
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.position = 0  # 0=hold, 1=long, -1=short
        self.shares_held = 0  # Retained from original, unused here but available for expansion
        self.done = False

        # Define feature columns (exclude non-numeric 'date_time')
        self.feature_cols = [col for col in df.columns if col != 'date_time']
        self.observation_dim = len(self.feature_cols)

        # Define Gym observation space
        # Assumes features are scaled to [-1, 1]; adjust bounds if scaling differs
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.observation_dim,),
            dtype=np.float32
        )

        # Define Gym action space: 0=hold, 1=long, 2=short
        self.action_space = spaces.Discrete(3)

        # Track the last price for PnL calculations
        self.last_price = self._get_close_price(self.current_step)

        # Track balance history for monitoring
        self.balance_history = [self.current_balance]

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
            np.array: Initial observation vector.
        """
        # Reset all state variables to their initial values
        self.current_step = 0
        self.current_balance = self.initial_balance
        self.position = 0
        self.shares_held = 0
        self.done = False
        self.last_price = self._get_close_price(self.current_step)
        self.balance_history = [self.current_balance]

        # Return the initial observation
        return self._get_observation(self.current_step)

    def step(self, action):
        """
        Executes one time step in the environment.

        Args:
            action (int): Action to take (0=hold, 1=long, 2=short).

        Returns:
            tuple: (observation (np.array), reward (float), done (bool), info (dict)).
        """
        # If the episode is done, return the current observation with zero reward
        if self.done:
            return self._get_observation(self.current_step), 0.0, True, {}

        # Get the current price for PnL calculation
        current_price = self._get_close_price(self.current_step)
        reward = 0.0

        # Calculate reward (PnL) based on the current position
        if self.position == 1:  # Long position
            reward = current_price - self.last_price
        elif self.position == -1:  # Short position
            reward = self.last_price - current_price

        # Update the balance with the reward
        self.current_balance += reward

        # Apply the new action
        if action == 1:
            self.position = 1  # Go long
        elif action == 2:
            self.position = -1  # Go short
        else:
            self.position = 0  # Hold/flat

        # Update the last price for the next step's PnL calculation
        self.last_price = current_price

        # Advance to the next step and check if the episode is done
        self.current_step += 1
        if self.current_step >= self.max_step:
            self.done = True
            obs = np.zeros(self.observation_dim, dtype=np.float32)  # Safe final state
        else:
            obs = self._get_observation(self.current_step)

        # Record the updated balance
        self.balance_history.append(self.current_balance)

        # Return the step results: observation, reward, done flag, and empty info dict
        return obs, reward, self.done, {}

    def _get_observation(self, step):
        """
        Retrieves the observation at a given step.

        Args:
            step (int): Current index in the DataFrame.

        Returns:
            np.array: Feature vector for the current step.
        """
        # Extract the feature vector and ensure it's a NumPy array of float32
        obs = self.df.iloc[step][self.feature_cols].values.astype(np.float32)

        # Handle potential GPU tensors (e.g., from cudf) if present
        if hasattr(obs, 'get'):
            obs = obs.get()

        return obs

    def _get_close_price(self, step):
        """
        Retrieves the closing price at a given step.

        Args:
            step (int): Current index in the DataFrame.

        Returns:
            float: Closing price.
        """
        # Get the close price and convert to float
        price = self.df.iloc[step]["close"]

        # Handle potential Series objects (e.g., from pandas operations)
        if hasattr(price, 'iloc'):
            price = price.iloc[0]

        return float(price)

    def current_state(self):
        """
        Returns the current observation.

        Returns:
            np.array: Current feature vector, or zeros if done.
        """
        # Return zeros if the episode is done, otherwise get the current observation
        if self.done:
            return np.zeros(self.observation_dim, dtype=np.float32)
        return self._get_observation(self.current_step)

    def render(self, mode='human'):
        """
        Renders the environment by printing the current trading state.

        Args:
            mode (str): Rendering mode, defaults to 'human' (console output).
        """
        # Get the current position as a string for readability
        position_str = {0: "Hold", 1: "Long", -1: "Short"}[self.position]

        # Get the current price and timestamp
        current_price = self._get_close_price(self.current_step)
        timestamp = self.df.iloc[self.current_step]["date_time"]

        # Print the current state with step, balance, position, price, and timestamp
        print(f"Step: {self.current_step} | "
              f"Balance: {self.current_balance:.2f} | "
              f"Position: {position_str} | "
              f"Price: {current_price:.2f} | "
              f"Timestamp: {timestamp}")