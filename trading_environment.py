import numpy as np


class TradingEnvironment:
    """
    Trading Environment stepping through time-series data for reinforcement learning.

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
            initial_balance (float): Starting capital.
            max_position (float): Maximum allowable position (contracts/shares).
        """
        self.df = df.reset_index(drop=True)
        self.max_step = len(self.df) - 1  # last index
        self.current_step = 0
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.position = 0  # +1=long, -1=short, 0=flat
        self.shares_held = 0
        self.done = False

        # Define observation columns excluding the timestamp
        self.feature_cols = [col for col in df.columns if col != 'date_time']
        self.observation_dim = len(self.feature_cols)
        self.action_space = 3  # hold, long, short

        # Track initial price for PnL calculations
        self.last_price = self._get_close_price(self.current_step)

        # Balance history for performance tracking
        self.balance_history = [self.current_balance]

    def reset(self):
        """
        Resets environment to initial state.

        Returns:
            np.array: Initial observation.
        """
        self.current_step = 0
        self.current_balance = self.initial_balance
        self.position = 0
        self.done = False
        self.shares_held = 0
        self.last_price = self._get_close_price(self.current_step)
        self.balance_history = [self.current_balance]

        return self._get_observation(self.current_step)

    def step(self, action):
        """
        Executes a single time step within the environment.

        Args:
            action (int): Action to take (0=hold, 1=long, 2=short).

        Returns:
            tuple: observation (np.array), reward (float), done (bool), info (dict)
        """
        if self.done:
            # Return current state if episode has ended
            return self._get_observation(self.current_step), 0.0, True, {}

        current_price = self._get_close_price(self.current_step)
        reward = 0.0

        # Calculate PnL from existing position
        if self.position == 1:  # Long
            reward = current_price - self.last_price
        elif self.position == -1:  # Short
            reward = self.last_price - current_price

        # Update balance
        self.current_balance += reward

        # Execute new action
        if action == 1:
            self.position = 1
        elif action == 2:
            self.position = -1
        else:
            self.position = 0

        self.last_price = current_price

        # Advance to next step
        self.current_step += 1
        if self.current_step >= self.max_step:
            self.done = True
            obs = np.zeros(self.observation_dim, dtype=np.float32)  # safe final obs
        else:
            obs = self._get_observation(self.current_step)

        self.balance_history.append(self.current_balance)

        # A small reward shaping can be done to avoid too large drawdown or encourage stable growth
        # But for now, let's keep it at pure PnL
        return obs, reward, self.done, {}

    def _get_observation(self, step):
        """
        Retrieves observation features at a specific step.

        Args:
            step (int): Index in the data.

        Returns:
            np.array: Observation array.
        """
        obs = self.df.iloc[step][self.feature_cols].values.astype(np.float32)
        if hasattr(obs, 'get'):
            obs = obs.get()  # Convert from GPU to CPU explicitly if needed
        return obs

    def _get_close_price(self, step):
        """
        Retrieves the close price at a given step.

        Args:
            step (int): Index in the data.

        Returns:
            float: Close price.
        """
        price = self.df.iloc[step]["close"]
        if hasattr(price, 'iloc'):
            price = price.iloc[0]
        return float(price)

    def current_state(self):
        """
        Returns current state observation.

        Returns:
            np.array: Current observation.
        """
        if self.done:
            return np.zeros(self.observation_dim, dtype=np.float32)
        return self._get_observation(self.current_step)
