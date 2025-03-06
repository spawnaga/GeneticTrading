import numpy as np


class TradingEnvironment:
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
        self.df = df.reset_index(drop=True)  # Ensure consistent indexing
        self.max_step = len(self.df) - 1  # Last valid index
        self.current_step = 0
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.position = 0  # 0=hold, 1=long, -1=short
        self.shares_held = 0  # Placeholder for future position sizing
        self.done = False

        # Define observation space (exclude timestamp)
        self.feature_cols = [col for col in df.columns if col != 'date_time']
        self.observation_dim = len(self.feature_cols)
        self.action_space = 3  # Discrete actions: hold, long, short

        # Initial price for PnL tracking
        self.last_price = self._get_close_price(self.current_step)

        # Track balance over time
        self.balance_history = [self.current_balance]

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
            np.array: Initial observation vector.
        """
        self.current_step = 0
        self.current_balance = self.initial_balance
        self.position = 0
        self.shares_held = 0
        self.done = False
        self.last_price = self._get_close_price(self.current_step)
        self.balance_history = [self.current_balance]
        return self._get_observation(self.current_step)

    def step(self, action):
        """
        Executes one time step in the environment.

        Args:
            action (int): Action to take (0=hold, 1=long, 2=short).

        Returns:
            tuple: (observation (np.array), reward (float), done (bool), info (dict)).
        """
        if self.done:
            return self._get_observation(self.current_step), 0.0, True, {}

        current_price = self._get_close_price(self.current_step)
        reward = 0.0

        # Calculate PnL based on current position
        if self.position == 1:  # Long position
            reward = current_price - self.last_price
        elif self.position == -1:  # Short position
            reward = self.last_price - current_price

        # Update balance with PnL
        self.current_balance += reward

        # Apply the new action
        if action == 1:
            self.position = 1  # Go long
        elif action == 2:
            self.position = -1  # Go short
        else:
            self.position = 0  # Hold/flat

        self.last_price = current_price

        # Move to next step
        self.current_step += 1
        if self.current_step >= self.max_step:
            self.done = True
            obs = np.zeros(self.observation_dim, dtype=np.float32)  # Safe final state
        else:
            obs = self._get_observation(self.current_step)

        self.balance_history.append(self.current_balance)

        return obs, reward, self.done, {}

    def _get_observation(self, step):
        """
        Retrieves the observation at a given step.

        Args:
            step (int): Current index in the DataFrame.

        Returns:
            np.array: Feature vector for the current step.
        """
        obs = self.df.iloc[step][self.feature_cols].values.astype(np.float32)
        if hasattr(obs, 'get'):  # Handle potential GPU tensors
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
        price = self.df.iloc[step]["close"]
        if hasattr(price, 'iloc'):  # Handle potential Series objects
            price = price.iloc[0]
        return float(price)

    def current_state(self):
        """
        Returns the current observation.

        Returns:
            np.array: Current feature vector, or zeros if done.
        """
        if self.done:
            return np.zeros(self.observation_dim, dtype=np.float32)
        return self._get_observation(self.current_step)
