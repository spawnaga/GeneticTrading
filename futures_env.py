# futures_env.py

import json
import math
from pathlib import Path
from typing import List, Sequence
from uuid import uuid4

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces


# Just placeholders if you had them in "utils.py":
def round_to_nearest_increment(value, increment):
    return round(value / increment) * increment


def monotonicity(series):
    # Example placeholder, define properly if needed
    return 0


class TimeSeriesState:
    """
    Example minimal TimeSeriesState class.
    You can adapt it to hold whatever features you want.
    """

    def __init__(self, ts, price, features):
        self.ts = ts
        self.price = price
        # 'features' can be anything you want the agent to see (list, np.array, etc.)
        # Must be numeric so we can feed it to the agent
        self.features = np.array(features, dtype=np.float32)

    def set_current_position(self, position, entry_time, exit_time):
        # Optional: if you want to add position info to features
        pass


class FuturesEnv(gym.Env):
    """
    A futures-trading environment with discrete actions: 0=Buy, 1=Hold, 2=Sell.
    """
    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            states: Sequence[TimeSeriesState],
            value_per_tick: float,
            tick_size: float,
            fill_probability: float = 1.0,
            long_values: List[float] = None,
            long_probabilities: List[float] = None,
            short_values: List[float] = None,
            short_probabilities: List[float] = None,
            execution_cost_per_order=0.0,
            add_current_position_to_state: bool = False,
            log_dir: str = "./logs/futures_env"
    ):
        super().__init__()

        # Core environment data
        self.states = states
        self.limit = len(self.states)
        self.value_per_tick = value_per_tick
        self.tick_size = tick_size
        self.long_values = long_values
        self.long_probabilities = long_probabilities
        self.short_values = short_values
        self.short_probabilities = short_probabilities
        self.fill_probability = fill_probability
        self.execution_cost_per_order = execution_cost_per_order
        self.add_current_position_to_state = add_current_position_to_state
        self.log_dir = log_dir

        self.can_generate_random_fills = all([
            self.long_values, self.long_probabilities,
            self.short_values, self.short_probabilities
        ])

        # Internal states
        self.current_index = 0
        self.done = False
        self.current_price = None

        # Position-tracking
        self.current_position = 0
        self.last_position = 0
        self.entry_time = None
        self.entry_id = None
        self.entry_price = None
        self.exit_time = None
        self.exit_id = None
        self.exit_price = None

        # Episode bookkeeping
        self.total_reward = 0
        self.total_net_profit = 0
        self.orders = []
        self.trades = []
        self.episode = 0
        self.feature_data = []

        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # --- GYM COMPAT: define action_space and observation_space ---
        # 3 discrete actions: 0=Buy, 1=Hold, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Example observation space: shape = # of features in one TimeSeriesState
        if len(self.states) > 0:
            obs_dim = len(self.states[0].features)
        else:
            obs_dim = 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def reset(self):
        self.done = False
        self.current_index = 0
        self.current_price = None
        self.current_position = 0
        self.last_position = 0
        self.entry_time = None
        self.entry_id = None
        self.entry_price = None
        self.exit_time = None
        self.exit_id = None
        self.exit_price = None
        self.total_reward = 0
        self.total_net_profit = 0
        self.orders = []
        self.trades = []
        self.feature_data = []
        self.episode += 1

        if self.limit == 0:
            self.done = True
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        # Return the first observation's features
        return self.states[0].features

    def step(self, action: int):
        """
        Standard Gym step:
        Returns: (obs, reward, done, info)
        """
        if self.done:
            # If done, return dummy
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                0.0,
                True,
                {}
            )

        # Move forward in our states
        _s, s = self._get_next_state()
        if self.done:
            # If the env is done after getting next state, we produce final step
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                0.0,
                True,
                {}
            )

        # Evaluate the chosen action (0=buy,1=hold,2=sell)
        if action == 0:
            # Buy
            if self.current_position == 1:
                # We are already long => hold
                reward = self.get_reward(s)
                info = {"message": "Long->Long: hold", "total_profit": self.total_reward}
            elif self.current_position == 0:
                # Open new long
                fill_decision = np.random.choice(
                    a=[0, 1], size=1,
                    p=[1 - self.fill_probability, self.fill_probability]
                )[0]
                if fill_decision == 1:
                    self.buy(s)
                    reward = self.get_reward(s)
                    info = {
                        "message": f"Opened long at {self.entry_price}",
                        "total_profit": self.total_reward
                    }
                else:
                    reward = 0.0
                    info = {
                        "message": "Long was not filled",
                        "total_profit": self.total_reward
                    }

            else:  # current_position == -1
                # Close short => go flat
                self.buy(s)
                reward = self.get_reward(s)
                net_profit = reward
                info = {
                    "message": f"Closed short => {net_profit}",
                    "total_profit": self.total_reward
                }
                self._close_position(reward, net_profit)

        elif action == 1:
            # Hold
            reward = self.get_reward(s)
            info = {
                "message": "No action (hold)",
                "total_profit": self.total_reward
            }

        elif action == 2:
            # Sell
            if self.current_position == -1:
                # Already short => hold
                reward = self.get_reward(s)
                info = {"message": "Short->Short: hold", "total_profit": self.total_reward}
            elif self.current_position == 0:
                # Open new short
                fill_decision = np.random.choice(
                    a=[0, 1], size=1,
                    p=[1 - self.fill_probability, self.fill_probability]
                )[0]
                if fill_decision == 1:
                    self.sell(s)
                    reward = self.get_reward(s)
                    info = {
                        "message": f"Opened short at {self.entry_price}",
                        "total_profit": self.total_reward
                    }
                else:
                    reward = 0.0
                    info = {
                        "message": "Short was not filled",
                        "total_profit": self.total_reward
                    }
            else:  # current_position == 1
                # Close long => go flat
                self.sell(s)
                reward = self.get_reward(s)
                net_profit = reward
                info = {
                    "message": f"Closed long => {net_profit}",
                    "total_profit": self.total_reward
                }
                self._close_position(reward, net_profit)

        # Next observation is _s.features
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        if _s:
            obs = _s.features
        return obs, reward, self.done, info

    # --- Order logic ---
    def buy(self, state):
        # Creates a buy order
        if self.current_position == 1:
            pass
        elif self.current_position == -1:
            self.last_position = -1
            self.current_position = 0
            self.exit_price = self.generate_random_fill_differential(state.price, 1)
            self.exit_time = state.ts
            self.exit_id = str(uuid4())
            self.orders.append([
                self.exit_id, str(state.ts), self.exit_price, 1, state
            ])
        elif self.current_position == 0:
            self.last_position = 0
            self.current_position = 1
            self.entry_price = self.generate_random_fill_differential(state.price, 1)
            self.entry_time = state.ts
            self.entry_id = str(uuid4())
            self.orders.append([
                self.entry_id, str(state.ts), self.entry_price, 1, state
            ])

    def sell(self, state):
        # Creates a sell order
        if self.current_position == -1:
            pass
        elif self.current_position == 1:
            self.last_position = 1
            self.current_position = 0
            self.exit_price = self.generate_random_fill_differential(state.price, -1)
            self.exit_time = state.ts
            self.exit_id = str(uuid4())
            self.orders.append([
                self.exit_id, str(state.ts), self.exit_price, -1, state
            ])
        elif self.current_position == 0:
            self.last_position = 0
            self.current_position = -1
            self.entry_price = self.generate_random_fill_differential(state.price, -1)
            self.entry_time = state.ts
            self.entry_id = str(uuid4())
            self.orders.append([
                self.entry_id, str(state.ts), self.entry_price, -1, state
            ])

    # --- Reward logic ---
    def get_reward(self, state):
        """
        Modified to provide partial 'unrealized' reward each step
        plus your usual reward upon close.
        """
        net_profit = 0.0

        # 1) A small fraction of unrealized PnL while the position is open
        #    so the agent sees if it's currently "winning" or "losing".
        if self.current_position == 1:
            # Long position: unrealized = (current price - entry price)
            unrealized = (state.price - self.entry_price) / self.tick_size * self.value_per_tick
            net_profit += 0.1 * unrealized  # 1% of unrealized each step (tune as needed)
        elif self.current_position == -1:
            # Short position: unrealized = (entry price - current price)
            unrealized = (self.entry_price - state.price) / self.tick_size * self.value_per_tick
            net_profit += 0.1 * unrealized

        # 2) The final "close" logic remains the same
        if any([
            (self.current_position == 0 and self.last_position == 0),
            (self.current_position == 1 and self.last_position == 0),
            (self.current_position == -1 and self.last_position == 0)
        ]):
            # no reward if no close and no position
            return net_profit  # we still return partial reward if we are in a position
        else:
            # If we moved from 1->0 or -1->0, compute the final trade profit
            if self.current_position == 0 and self.last_position == 1:
                # closed a long
                diff = round((self.exit_price - self.entry_price), 2)
                n_ticks = math.ceil(diff / self.tick_size)
                gross_profit = n_ticks * self.value_per_tick
                close_reward = gross_profit - (2 * self.execution_cost_per_order)
                net_profit += close_reward
            elif self.current_position == 0 and self.last_position == -1:
                # closed a short
                diff = round((self.exit_price - self.entry_price), 2)
                n_ticks = math.ceil(diff / self.tick_size) * -1
                gross_profit = n_ticks * self.value_per_tick
                close_reward = gross_profit - (2 * self.execution_cost_per_order)
                net_profit += close_reward

        self.total_reward += net_profit
        return net_profit

    def _close_position(self, reward, net_profit):
        duration = 0.0
        if self.exit_time and self.entry_time:
            duration = (self.exit_time - self.entry_time).total_seconds()
        trade_id = str(uuid4())
        if self.current_position == 0 and self.last_position == -1:
            self.trades.append([
                trade_id, "short", self.entry_id, self.exit_id,
                net_profit, reward, duration
            ])
        elif self.current_position == 0 and self.last_position == 1:
            self.trades.append([
                trade_id, "long", self.entry_id, self.exit_id,
                net_profit, reward, duration
            ])

        self.current_position = 0
        self.last_position = 0
        self.entry_time = None
        self.entry_id = None
        self.entry_price = None
        self.exit_time = None
        self.exit_id = None
        self.exit_price = None

    # --- State iteration ---
    def _get_next_state(self):
        current_state = self.states[self.current_index]
        self.current_index += 1
        if self.current_index <= self.limit - 1:
            next_state = self.states[self.current_index]
            self.current_price = current_state.price
            if current_state.ts > next_state.ts:
                raise Exception("Time ordering error in states.")
            # Optionally embed current_position into next_state
            if self.add_current_position_to_state:
                next_state.set_current_position(
                    self.current_position, self.entry_time, self.exit_time
                )
            return (next_state, current_state)
        else:
            self.done = True
            return (None, current_state)

    # --- Fill slippage logic ---
    def generate_random_fill_differential(self, intent_price: float, trade_type: int) -> float:
        if not self.can_generate_random_fills:
            return intent_price
        else:
            if trade_type == 1:  # buy
                diff = np.random.choice(
                    a=self.long_values, size=1, p=self.long_probabilities
                )[0]
                return round_to_nearest_increment(intent_price + diff, self.tick_size)
            else:  # -1 => sell
                diff = np.random.choice(
                    a=self.short_values, size=1, p=self.short_probabilities
                )[0]
                return round_to_nearest_increment(intent_price + diff, self.tick_size)

    # --- Rendering & metrics ---
    def render(self, mode='human'):
        """
        Basic usage: we just call this at the end to produce
        distributions, metrics, etc. (the function from your snippet).
        """
        self._generate_episode_graphs()
        metrics = self.generate_episode_metrics()
        return self.total_reward, metrics

    def _generate_episode_graphs(self):
        if len(self.trades) == 0:
            return
        _, trades = self.get_episode_data()
        durations = trades["duration"]
        plt.hist(durations, bins=20)
        Path(f"{self.log_dir}/img").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{self.log_dir}/img/duration_distribution_episode_{self.episode}.png")
        plt.clf()

        short_df = trades[trades["trade_type"] == "short"]
        long_df = trades[trades["trade_type"] == "long"]

        plt.hist(long_df["profit"], bins=20, alpha=0.5, label="long")
        plt.hist(short_df["profit"], bins=20, alpha=0.5, label="short")
        plt.savefig(f"{self.log_dir}/img/profit_loss_distribution_episode_{self.episode}.png")
        plt.clf()

    def get_episode_data(self):
        order_cols = ["order_id", "timestamp", "price", "type"]
        if len(self.orders) == 0:
            order_df = pd.DataFrame(columns=order_cols)
        else:
            raw_orders = [o[:4] for o in self.orders]
            order_df = pd.DataFrame(raw_orders, columns=order_cols)

        trade_cols = [
            "trade_id", "trade_type", "entry_order_id", "exit_order_id",
            "profit", "reward", "duration"
        ]
        trade_df = pd.DataFrame(self.trades, columns=trade_cols)
        return order_df, trade_df

    def generate_episode_metrics(self, as_file=True):
        orders, trades = self.get_episode_data()
        if len(trades) == 0:
            return {}

        all_trades = trades
        running_pnl = []
        pnl_val = 0
        for i, row in all_trades.iterrows():
            pnl_val += row["profit"]
            running_pnl.append(pnl_val)

        # Example: some basic stats
        tnp = sum(all_trades["profit"])
        wins = all_trades[all_trades["profit"] >= 0]
        losses = all_trades[all_trades["profit"] < 0]
        total_trades = len(all_trades)
        if len(wins) > 0:
            win_pl = sum(wins["profit"])
        else:
            win_pl = 0
        if len(losses) > 0:
            loss_pl = sum(losses["profit"])
        else:
            loss_pl = 0
        win_percentage = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        profit_factor = abs(win_pl) / abs(loss_pl) if loss_pl != 0 else 999.0
        intraday_low = np.min(running_pnl) if len(running_pnl) > 0 else 0

        metrics = {
            "tnp": tnp,
            "win_percentage": win_percentage,
            "profit_factor": profit_factor,
            "intraday_low": intraday_low,
            "pnl_monotonicity": monotonicity(running_pnl),
            "running_pl": running_pnl
        }
        if as_file:
            Path(f"{self.log_dir}/metrics").mkdir(parents=True, exist_ok=True)
            with open(f"{self.log_dir}/metrics/metrics_episode_{str(self.episode)}.json", "w+") as f:
                json.dump(metrics, f, indent=4)
        return metrics
