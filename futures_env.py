# futures_env.py

import math
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from uuid import uuid4


def round_to_nearest_increment(value: float, increment: float) -> float:
    """
    Round `value` to the nearest multiple of `increment`.
    """
    return round(value / increment) * increment


def monotonicity(series):
    """
    Compute the average sign of sequential differences: +1 for up, -1 for down.
    """
    dirs = []
    for i in range(1, len(series)):
        diff = series[i] - series[i - 1]
        dirs.append(1 if diff > 0 else -1 if diff < 0 else 0)
    return float(np.mean(dirs)) if dirs else 0.0


class TimeSeriesState:
    """Single time-step state holding immutable OHLC data and features."""

    def __init__(self, ts, open_price, close_price, features):
        self.ts = ts
        self.open_price = open_price
        self.close_price = close_price
        # always keep this at length-7 base features
        self.features = np.array(features, dtype=np.float32)


class FuturesEnv(gym.Env):
    """
    Reinforcement‐learning environment simulating single‐contract futures trading.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 states,
                 value_per_tick,
                 tick_size,
                 fill_probability=1.0,
                 long_values=None,
                 long_probabilities=None,
                 short_values=None,
                 short_probabilities=None,
                 execution_cost_per_order=0.0,
                 contracts_per_trade=1,
                 margin_rate=0.0,
                 bid_ask_spread=0.0,
                 add_current_position_to_state=False,
                 log_dir="./logs/futures_env"):
        super().__init__()

        # Core parameters
        self.states = states
        self.limit = len(states)
        self.value_per_tick = value_per_tick
        self.tick_size = tick_size
        self.fill_probability = fill_probability
        self.execution_cost_per_order = execution_cost_per_order
        self.contracts_per_trade = contracts_per_trade
        self.margin_rate = margin_rate
        self.bid_ask_spread = bid_ask_spread
        self.add_current_position_to_state = add_current_position_to_state
        self.log_dir = log_dir

        # Slippage distributions
        self.long_values = long_values
        self.long_probabilities = long_probabilities
        self.short_values = short_values
        self.short_probabilities = short_probabilities
        self.can_random = all([long_values, long_probabilities, short_values, short_probabilities])

        # Internal state
        self.current_index = 0
        self.done = False
        self.current_position = 0       # -1=short, 0=flat, 1=long
        self.last_position = 0
        self.entry_time = None
        self.entry_price = None
        self.exit_time = None
        self.exit_price = None
        self.partial_reward_sum = 0.0
        self.total_reward = 0.0
        self.orders = []
        self.trades = []
        self.episode = 0
        self.last_ts = None

        # Setup directories
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # Define action & observation spaces
        self.action_space = spaces.Discrete(3)
        base_dim = len(states[0].features)  # will now be >> 7
        extras = 3 if add_current_position_to_state else 0
        obs_dim = base_dim + extras                  # 7 or 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

    def reset(self):
        """
        Begin a new episode. Return initial observation of fixed length.
        """
        # Reset all internal trackers
        self.done = False
        self.current_index = 0
        self.current_position = 0
        self.last_position = 0
        self.entry_time = None
        self.entry_price = None
        self.exit_time = None
        self.exit_price = None
        self.partial_reward_sum = 0.0
        self.total_reward = 0.0
        self.orders.clear()
        self.trades.clear()
        self.episode += 1
        self.last_ts = self.states[0].ts if self.limit > 0 else None

        if self.limit == 0:
            # no data case
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        # build a fresh obs from base features + zeros for extras
        base = self.states[0].features  # length=7
        if self.add_current_position_to_state:
            extras = np.zeros(3, dtype=np.float32)
            return np.concatenate([base, extras])
        else:
            return base.copy()

    def step(self, action):
        """
        Execute action & advance one time-step.
        Actions: 0=Buy, 1=Hold, 2=Sell
        Returns (obs, reward, done, info) with obs always fixed-length.
        """
        if self.done:
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, {}

        # current state for reward & trade logic
        current_state = self.states[self.current_index]
        next_state = (self.states[self.current_index + 1]
                      if self.current_index + 1 < self.limit
                      else current_state)
        self.current_index += 1

        # Handle BUY/SELL actions at the next bar's open
        if action == 0:
            self._handle_buy(next_state)
        elif action == 2:
            self._handle_sell(next_state)

        # compute reward from this time-step
        reward = self._get_reward(current_state)
        info = {
            "message": f"Pos {self.current_position}",
            "total_profit": self.total_reward,
            "timestamp": current_state.ts
        }

        # Build next observation fresh:
        if self.current_index < self.limit:
            base = self.states[self.current_index].features
        else:
            # end of episode
            self.done = True
            base = np.zeros(len(self.states[0].features), dtype=np.float32)

        if self.add_current_position_to_state:
            entry_ts = self.entry_time.timestamp() if self.entry_time else 0.0
            exit_ts  = self.exit_time.timestamp()  if self.exit_time  else 0.0
            extras = np.array([self.current_position, entry_ts, exit_ts], dtype=np.float32)
            obs = np.concatenate([base, extras])
        else:
            obs = base.copy()

        return obs, reward, self.done, info

    def _handle_buy(self, state):
        """
        Open/close long positions.
        """
        if self.current_position == -1:
            self._close_short(state)
        elif self.current_position == 0:
            self.partial_reward_sum = 0.0
            filled_price = self._simulate_fill(state.open_price, 1)
            self.entry_time = state.ts
            self.entry_price = filled_price
            self.orders.append([str(uuid4()), str(state.ts), filled_price, 1])
            self.current_position = 1

    def _handle_sell(self, state):
        """
        Open/close short positions.
        """
        if self.current_position == 1:
            self._close_long(state)
        elif self.current_position == 0:
            self.partial_reward_sum = 0.0
            filled_price = self._simulate_fill(state.open_price, -1)
            self.entry_time = state.ts
            self.entry_price = filled_price
            self.orders.append([str(uuid4()), str(state.ts), filled_price, -1])
            self.current_position = -1

    def _simulate_fill(self, price: float, trade_type: int) -> float:
        """
        Simulate slippage, half‐spread cost, and fill probability.

        Parameters:
        - price:        The mid‐market price at which the order is placed.
        - trade_type:   +1 for a buy/long, -1 for a sell/short.

        Returns:
        - The executed price after slippage and spread adjustments.
          If the order “does not fill” (based on fill_probability), returns
          the mid‐price unchanged.
        """
        # 1) Decide whether the order actually fills
        if np.random.rand() > self.fill_probability:
            # No fill: assume execution at mid‐market price (no slippage/spread)
            return price

        # 2) Sample slippage if custom distributions are provided
        slippage = 0.0
        if self.can_random:
            if trade_type == 1:
                slippage = np.random.choice(self.long_values, p=self.long_probabilities)
            else:
                slippage = np.random.choice(self.short_values, p=self.short_probabilities)

        # 3) Apply half the bid‐ask spread in the direction of the trade
        spread_adj = (self.bid_ask_spread / 2.0) * (1 if trade_type == 1 else -1)

        # 4) Combine into a raw execution price
        raw_price = price + slippage + spread_adj

        # 5) Round to the nearest tick increment
        return round_to_nearest_increment(raw_price, self.tick_size)


    def _get_reward(self, state):
        """
        Compute combined partial & realized PnL, minus margin & execution costs.
        """
        net = 0.0
        # unrealized
        if self.current_position != 0:
            diff = (state.close_price - self.entry_price) if self.current_position == 1 else (self.entry_price - state.close_price)
            unrealized = (diff / self.tick_size) * self.value_per_tick * self.contracts_per_trade
        else:
            unrealized = 0.0

        # logistic partial reward
        scale, k = 100.0, 0.1
        partial = (2.0 / (1.0 + np.exp(-k * (unrealized/scale))) - 1.0) * 0.05
        self.partial_reward_sum += partial
        net += partial

        # realized on close
        if self.current_position == 0 and self.last_position != 0:
            price_diff = self.exit_price - self.entry_price
            if self.last_position == -1:
                price_diff = -price_diff
            ticks = price_diff / self.tick_size
            gross = ticks * self.value_per_tick * self.contracts_per_trade
            cost  = 2 * self.execution_cost_per_order * self.contracts_per_trade
            net += (gross - cost)

        # margin cost over time
        if self.last_ts and self.current_position != 0:
            delta = (state.ts - self.last_ts).total_seconds()
            year_secs = 365.25 * 24 * 3600
            margin_cost = (self.margin_rate
                           * abs(self.current_position)
                           * self.contracts_per_trade
                           * state.close_price
                           * (delta / year_secs))
            net -= margin_cost

        self.total_reward += net
        self.last_position = self.current_position
        self.last_ts = state.ts
        return net

    def _close_long(self, state):
        """
        Close an existing long position.
        """
        self.exit_time = state.ts
        self.exit_price = self._simulate_fill(state.open_price, -1)
        self.orders.append([str(uuid4()), str(state.ts), self.exit_price, -1])
        self.current_position = 0
        self._record_trade("long")

    def _close_short(self, state):
        """
        Close an existing short position.
        """
        self.exit_time = state.ts
        self.exit_price = self._simulate_fill(state.open_price, 1)
        self.orders.append([str(uuid4()), str(state.ts), self.exit_price, 1])
        self.current_position = 0
        self._record_trade("short")

    def _record_trade(self, trade_type):
        """
        Log a completed trade.
        """
        duration = ((self.exit_time - self.entry_time).total_seconds()
                    if self.entry_time else 0.0)
        profit = self.total_reward
        self.trades.append([
            str(uuid4()), trade_type,
            self.entry_price, self.exit_price,
            profit, duration, self.partial_reward_sum
        ])

    def render(self, mode='human'):
        """
        Save diagnostic plots and metrics.
        """
        orders_df, trades_df = self.get_episode_data()
        if trades_df.empty:
            print("No trades to render.")
            return self.total_reward, {}

        # duration histogram
        fig1 = plt.figure()
        trades_df['duration'].hist(bins='auto', edgecolor='black')
        plt.title(f"Trade Duration - Ep {self.episode}")
        plt.xlabel("Seconds"); plt.ylabel("Count")
        fig1.savefig(f"{self.log_dir}/duration_ep{self.episode}.png")
        plt.close(fig1)

        # profit histogram
        fig2 = plt.figure()
        trades_df['profit'].hist(bins='auto', edgecolor='black')
        plt.title(f"Profit Distribution - Ep {self.episode}")
        plt.xlabel("Profit"); plt.ylabel("Count")
        fig2.savefig(f"{self.log_dir}/profit_ep{self.episode}.png")
        plt.close(fig2)

        metrics = self.generate_episode_metrics()
        return self.total_reward, metrics

    def get_episode_data(self):
        """
        Return DataFrames for orders & trades.
        """
        orders_cols = ["order_id", "timestamp", "price", "type"]
        trades_cols = ["trade_id", "trade_type", "entry_price", "exit_price",
                       "profit", "duration", "partial_sum"]
        orders_df = (pd.DataFrame(self.orders, columns=orders_cols)
                     if self.orders else pd.DataFrame(columns=orders_cols))
        trades_df = (pd.DataFrame(self.trades, columns=trades_cols)
                     if self.trades else pd.DataFrame(columns=trades_cols))
        return orders_df, trades_df

    def generate_episode_metrics(self, as_file=True):
        """
        Compute aggregate metrics and optionally save to JSON.
        """
        _, trades_df = self.get_episode_data()
        if trades_df.empty:
            return {}

        running = trades_df['profit'].cumsum().tolist()
        tnp     = trades_df['profit'].sum()
        wins    = trades_df[trades_df['profit'] >= 0]
        losses  = trades_df[trades_df['profit'] < 0]
        win_pct = (len(wins) / len(trades_df) * 100) if len(trades_df) else 0.0
        profit_factor = (wins['profit'].sum() / -losses['profit'].sum()
                         if not losses.empty else float('inf'))
        intraday_low = min(running) if running else 0.0

        metrics = {
            "total_net_profit": tnp,
            "win_percentage": win_pct,
            "profit_factor": profit_factor,
            "intraday_low": intraday_low,
            "monotonicity": monotonicity(running),
            "running_equity": running
        }

        if as_file:
            Path(self.log_dir + "/metrics").mkdir(parents=True, exist_ok=True)
            with open(f"{self.log_dir}/metrics/ep{self.episode}.json", "w") as f:
                json.dump(metrics, f, indent=4)

        return metrics
