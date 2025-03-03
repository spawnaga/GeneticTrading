import numpy as np
import torch

class TradingEnvironment:
    """
    A simple environment that steps through the time-series data.
    Observations: A vector of scaled features (open, high, low, close, volume, return, ma_10, etc.)
    Action space: 3 actions [0: hold, 1: long, 2: short].
    Reward: The PnL from the position changes or Mark-to-Market profit.
    """
    def __init__(self, df, initial_balance=100000.0, max_position=1.0):
        """
        df: pandas DataFrame with scaled columns + date_time
        initial_balance: starting capital
        max_position: we might restrict position to 1 contract or partial shares in real usage
        """
        self.df = df.reset_index(drop=True)
        self.max_step = len(self.df) - 1  # last index
        self.current_step = 0
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.position = 0  # +1 for long, -1 for short, 0 for flat
        self.shares_held = 0
        self.done = False
        # Extract the columns for state
        self.feature_cols = [col for col in df.columns if col not in ['date_time']]
        self.observation_dim = len(self.feature_cols)
        self.action_space = 3  # hold, buy, sell

        # Store the initial price to track PnL
        self.last_price = self._get_close_price(self.current_step)

        self.balance_history = []
        self.balance_history.append(self.current_balance)

    def reset(self):
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
        action: 0=hold, 1=long, 2=short
        """
        if self.done:
            return (self._get_observation(self.current_step), 0, True, {})

        # Get current price
        current_price = self._get_close_price(self.current_step)

        reward = 0.0

        # Mark-to-market PnL from previous step
        # If position was +1, PnL is (current_price - last_price)
        # If position was -1, PnL is (last_price - current_price)
        if self.position == 1:
            reward = current_price - self.last_price
        elif self.position == -1:
            reward = self.last_price - current_price

        # Update balance by that PnL
        self.current_balance += reward

        # Now process the new action
        if action == 1:  # go long
            self.position = 1
        elif action == 2:  # go short
            self.position = -1
        else:  # hold
            pass

        self.last_price = current_price

        # Move to the next step
        self.current_step += 1
        if self.current_step >= self.max_step:
            self.done = True

        obs = self._get_observation(self.current_step)
        self.balance_history.append(self.current_balance)

        # A small reward shaping can be done to avoid too large drawdown or encourage stable growth
        # But for now, let's keep it at pure PnL
        return (obs, reward, self.done, {})

    def _get_observation(self, step):
        row = self.df.iloc[step][self.feature_cols].values
        return row.astype(np.float32)

    def _get_close_price(self, step):
        return self.df.iloc[step]['close']

    def current_state(self):
        return self._get_observation(self.current_step)
