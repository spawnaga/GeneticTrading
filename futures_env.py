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

from utils import round_to_nearest_increment, monotonicity, cleanup_old_logs


class TimeSeriesState:
    """Single time-step state holding immutable OHLCV data and features."""

    def __init__(
        self,
        ts,
        open_price,
        high_price=None,
        low_price=None,
        close_price=None,
        volume=None,
        features=None,
    ):
        self.ts = ts
        self.open_price = open_price
        self.high_price = high_price
        self.low_price = low_price
        self.close_price = close_price
        self.volume = volume
        self.features = (
            np.array(features, dtype=np.float32) if features is not None else None
        )


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
                 log_dir="./logs/futures_env",
                 market_hours=(9, 16),  # Trading hours (9 AM to 4 PM)
                 weekend_trading=False,
                 gap_penalty=0.02,  # Penalty for overnight gaps
                 liquidity_impact=True):
        super().__init__()

        # Core parameters
        self.states = states
        self.limit = len(states)
        self.value_per_tick = value_per_tick
        self.tick_size = tick_size
        if self.value_per_tick <= 0 or self.tick_size <= 0:
            raise ValueError("value_per_tick and tick_size must be positive")
        self.fill_probability = fill_probability
        self.execution_cost_per_order = execution_cost_per_order
        self.contracts_per_trade = contracts_per_trade
        self.margin_rate = margin_rate
        self.bid_ask_spread = bid_ask_spread
        self.add_current_position_to_state = add_current_position_to_state
        self.log_dir = log_dir
        self.market_hours = market_hours
        self.weekend_trading = weekend_trading
        self.gap_penalty = gap_penalty
        self.liquidity_impact = liquidity_impact

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
        self.balance = 0.0               # realized PnL minus costs
        self.entry_cost = 0.0            # commission paid at entry
        self.total_reward = 0.0          # running equity (balance + unrealized)
        self.orders = []
        self.trades = []
        self.episode = 0
        self.last_ts = None
        self.last_price = None           # Track price for reward calculation
        self.last_action_was_close = False  # Track if last action closed a position

        # Setup directories
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # Define action & observation spaces
        self.action_space = spaces.Discrete(3)
        if not states:
            raise ValueError("States list cannot be empty")
        base_dim = len(states[0].features)
        extras = 3 if add_current_position_to_state else 0
        obs_dim = base_dim + extras
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Begin a new episode. Return initial observation of fixed length.
        """
        super().reset(seed=seed)

        # Reset all internal trackers
        self.done = False
        start_idx = 0
        if self.limit > 1:
            start_idx = np.random.randint(0, self.limit - 1)
        self.current_index = start_idx
        self.current_position = 0
        self.last_position = 0
        self.entry_time = None
        self.entry_price = None
        self.exit_time = None
        self.exit_price = None
        self.balance = 0.0
        self.entry_cost = 0.0
        self.total_reward = 0.0
        self.orders.clear()
        self.trades.clear()
        self.episode += 1
        self.last_ts = self.states[start_idx].ts if self.limit > 0 else None
        self.last_price = self.states[start_idx].close_price if self.limit > 0 else None
        self.last_action_was_close = False

        if self.limit == 0:
            # no data case
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, {}

        # build a fresh obs from base features + zeros for extras
        base = self.states[start_idx].features
        if self.add_current_position_to_state:
            extras = np.zeros(3, dtype=np.float32)
            obs = np.concatenate([base, extras])
        else:
            obs = base.copy()

        return obs, {}

    def step(self, action):
        """
        Take action in environment and return (obs, reward, done, info).
        """
        if self.done or self.current_index >= self.limit:
            return self._get_observation(), 0.0, True, {}

        # Validate action
        if not isinstance(action, (int, np.integer)) or action < 0 or action >= 3:
            action = 0  # Default to hold for invalid actions

        # Track state before action
        prev_balance = self.balance
        prev_total_reward = self.total_reward

        # Process action
        if action == 0:  # hold
            pass
        elif action == 1:  # buy/long
            if self.current_position <= 0:
                self._execute_trade(1)
        elif action == 2:  # sell/short  
            if self.current_position >= 0:
                self._execute_trade(-1)

        # Move to next state
        self.current_index += 1
        if self.current_index >= self.limit:
            self.done = True

        # Calculate reward with NaN protection
        try:
            reward = self._get_reward(
                self.states[min(self.current_index - 1, self.limit - 1)]
            )

            # Ensure reward is finite
            if not np.isfinite(reward):
                reward = 0.0

        except Exception as e:
            reward = 0.0

        # Get updated observation and info with NaN protection
        obs = self._get_observation()
        info = self._get_info()

        # Final NaN check on all outputs
        if not np.all(np.isfinite(obs)):
            obs = np.zeros_like(obs)

        if not np.isfinite(reward):
            reward = 0.0

        return obs, reward, self.done, info


    def _handle_buy(self, state):
        """
        Open/close long positions.
        """
        self.last_action_was_close = False
        if self.current_position == -1:
            self._close_short(state)
            self.last_action_was_close = True
        elif self.current_position == 0:
            filled_price = self._simulate_fill(state.open_price, 1)
            self.entry_time = state.ts
            self.entry_price = filled_price
            self.entry_cost = self.execution_cost_per_order * self.contracts_per_trade
            self.balance -= self.entry_cost
            self.orders.append([str(uuid4()), str(state.ts), filled_price, 1])
            self.current_position = 1

    def _handle_sell(self, state, high_price=None, low_price=None, volume=None):
        """
        Open/close short positions.
        """
        self.last_action_was_close = False
        if self.current_position == 1:
            self._close_long(state)
            self.last_action_was_close = True
        elif self.current_position == 0:
            filled_price = self._simulate_fill(state.open_price, -1)
            self.entry_time = state.ts
            self.entry_price = filled_price
            self.entry_cost = self.execution_cost_per_order * self.contracts_per_trade
            self.balance -= self.entry_cost
            self.orders.append([str(uuid4()), str(state.ts), filled_price, -1])
            self.current_position = -1

    def _is_market_open(self, timestamp):
        """Check if market is open based on timestamp"""
        if not self.weekend_trading and timestamp.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        hour = timestamp.hour
        return self.market_hours[0] <= hour < self.market_hours[1]

    def _detect_gap(self, current_state, previous_state):
        """Detect price gaps between sessions"""
        if previous_state is None:
            return 0.0

        # Check for time gap (overnight, weekend)
        time_diff = (current_state.ts - previous_state.ts).total_seconds()
        if time_diff > 3600:  # More than 1 hour gap
            price_change = abs(current_state.open_price - previous_state.close_price) / previous_state.close_price
            return price_change
        return 0.0

    def _adjust_for_liquidity(self, price, volume, trade_type):
        """Adjust execution price based on market liquidity"""
        if not self.liquidity_impact or volume is None:
            return price

        # Lower volume = higher slippage
        liquidity_factor = 1.0 / (1.0 + volume / 10000.0)  # Normalize volume
        slippage_adjustment = liquidity_factor * 0.001 * abs(trade_type)  # 0.1% max slippage

        return price + (slippage_adjustment * trade_type)

    def _simulate_fill(
        self,
        price: float,
        high_price: float | None = None,
        low_price: float | None = None,
        volume: float | None = None,
        size: int = 1
    ) -> float:
        """
        Simulate execution price with slippage and spread adjustments.
        High/low bounds are used for clipping, while volume modulates
        slippage magnitude (lower volume -> more slippage).
        """
        # Validate input parameters
        if not np.isfinite(price) or price <= 0:
            price = 100.0  # Default fallback price

        if high_price is not None and not np.isfinite(high_price):
            high_price = None

        if low_price is not None and not np.isfinite(low_price):
            low_price = None

        if volume is not None and not np.isfinite(volume):
            volume = None

        # 1) Decide whether the order actually fills
        current_state = self.states[self.current_index] if self.current_index < len(self.states) else None

        # Reduce fill probability during low liquidity periods
        adjusted_fill_prob = self.fill_probability
        if volume is not None and volume < 1000:  # Low volume threshold
            adjusted_fill_prob *= 0.8

        if np.random.rand() > adjusted_fill_prob:
            return price

        # 2) Apply liquidity-based price adjustment
        price = self._adjust_for_liquidity(price, volume, size)

        # 3) Sample slippage if custom distributions are provided
        slippage = 0.0
        if self.can_random:
            if size == 1:
                slippage = np.random.choice(self.long_values, p=self.long_probabilities)
            else:
                slippage = np.random.choice(self.short_values, p=self.short_probabilities)

        # 4) Add volatility-based slippage during high volatility periods
        if high_price is not None and low_price is not None:
            volatility = (high_price - low_price) / price
            if volatility > 0.02:  # High volatility threshold (2%)
                volatility_slippage = volatility * 0.5 * np.random.uniform(-1, 1)
                slippage += volatility_slippage
        # 3) Use high/low range to derive a dynamic spread when available
        dynamic_spread = self.bid_ask_spread
        if high_price is not None and low_price is not None:
            dynamic_spread = max(dynamic_spread, high_price - low_price)

        # Apply half the spread in the direction of the trade
        spread_adj = (dynamic_spread / 2.0) * (1 if size == 1 else -1)

        # 4) Volume-based scaling of slippage
        volume_scale = 1.0
        if volume is not None and volume > 0:
            volume_scale += 1.0 / (volume + 1e-8)

        # 5) Combine all components
        raw_price = price + (slippage + spread_adj) * volume_scale

        # 6) Check for NaN or infinite values
        if not np.isfinite(raw_price):
            raw_price = price  # Fallback to original price

        # 7) Clip to high/low range if provided
        if high_price is not None and low_price is not None:
            raw_price = np.clip(raw_price, low_price, high_price)

        # 8) Round to the nearest tick increment
        return round_to_nearest_increment(raw_price, self.tick_size)

    def _get_reward(self, state):
        """
        Compute the step reward based on P&L change + position penalties.
        """
        # Validate input state
        if state is None or not hasattr(state, 'close_price'):
            return 0.0

        current_price = state.close_price
        if not np.isfinite(current_price) or current_price <= 0:
            return 0.0

        # Update unrealized P&L based on current market price
        if self.current_position != 0 and self.entry_price is not None:
            if np.isfinite(self.entry_price) and self.entry_price > 0:
                price_diff = (current_price - self.entry_price) * self.current_position
                unrealized_pnl = price_diff * self.value_per_tick / self.tick_size

                # Check for NaN/infinite unrealized PnL
                if not np.isfinite(unrealized_pnl):
                    unrealized_pnl = 0.0
            else:
                unrealized_pnl = 0.0
        else:
            unrealized_pnl = 0.0

        # Ensure balance is finite
        if not np.isfinite(self.balance):
            self.balance = 0.0

        # Total equity = realized balance + unrealized P&L
        current_total = self.balance + unrealized_pnl

        # Check for NaN/infinite total
        if not np.isfinite(current_total):
            current_total = 0.0

        # Ensure total_reward is finite
        if not np.isfinite(self.total_reward):
            self.total_reward = 0.0

        # Reward is the change in total equity from last step
        reward = current_total - self.total_reward

        # Check for NaN/infinite reward
        if not np.isfinite(reward):
            reward = 0.0

        self.total_reward = current_total

        # Add small position holding cost to encourage trading
        if self.current_position != 0:
            holding_cost = abs(self.current_position) * 0.01  # Small holding cost
            if np.isfinite(holding_cost):
                reward -= holding_cost

        # Final reward validation
        if not np.isfinite(reward):
            reward = 0.0

        # Clamp reward to reasonable range
        reward = np.clip(reward, -1000.0, 1000.0)

        return reward

    def _close_long(self, state):
        """
        Close an existing long position.
        """
        self.exit_time = state.ts
        self.exit_price = self._simulate_fill(state.open_price, -1)
        price_diff = self.exit_price - self.entry_price
        ticks = price_diff / self.tick_size
        pnl = ticks * self.value_per_tick * self.contracts_per_trade
        exit_cost = self.execution_cost_per_order * self.contracts_per_trade
        trade_profit = pnl - self.entry_cost - exit_cost
        self.balance += trade_profit
        self.orders.append([str(uuid4()), str(state.ts), self.exit_price, -1])
        self.current_position = 0
        self._record_trade("long", trade_profit)
        self.entry_cost = 0.0

    def _close_short(self, state):
        """
        Close an existing short position.
        """
        self.exit_time = state.ts
        self.exit_price = self._simulate_fill(state.open_price, 1)
        price_diff = self.entry_price - self.exit_price
        ticks = price_diff / self.tick_size
        pnl = ticks * self.value_per_tick * self.contracts_per_trade
        exit_cost = self.execution_cost_per_order * self.contracts_per_trade
        trade_profit = pnl - self.entry_cost - exit_cost
        self.balance += trade_profit
        self.orders.append([str(uuid4()), str(state.ts), self.exit_price, 1])
        self.current_position = 0
        self._record_trade("short", trade_profit)
        self.entry_cost = 0.0

    def _record_trade(self, trade_type, profit):
        """
        Log a completed trade.
        """
        duration = ((self.exit_time - self.entry_time).total_seconds()
                    if self.entry_time else 0.0)
        self.trades.append([
            str(uuid4()), trade_type,
            self.entry_price, self.exit_price,
            profit, duration
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

        # Cleanup old logs every 10 episodes
        if self.episode % 10 == 0:
            cleanup_old_logs(self.log_dir, max_episodes=10)

        metrics = self.generate_episode_metrics()
        return self.total_reward, metrics

    def get_episode_data(self):
        """
        Return DataFrames for orders & trades.
        """
        orders_cols = ["order_id", "timestamp", "price", "type"]
        trades_cols = ["trade_id", "trade_type", "entry_price", "exit_price",
                       "profit", "duration"]
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

    def _execute_trade(self, target_position):
        """
        Execute a trade to move from current_position to target_position.
        """
        if target_position == self.current_position:
            return

        current_state = (
            self.states[self.current_index] 
            if self.current_index < len(self.states) 
            else self.states[-1]
        )

        trade_size = target_position - self.current_position
        fill_price = self._simulate_fill(
            price=current_state.close_price,
            high_price=getattr(current_state, 'high_price', None),
            low_price=getattr(current_state, 'low_price', None),
            volume=getattr(current_state, 'volume', None),
            size=trade_size
        )

        # Validate fill price
        if not np.isfinite(fill_price) or fill_price <= 0:
            fill_price = current_state.close_price
            if not np.isfinite(fill_price) or fill_price <= 0:
                return  # Skip trade if price is invalid

        # Calculate trade cost
        trade_cost = abs(trade_size) * self.execution_cost_per_order
        if not np.isfinite(trade_cost):
            trade_cost = 0.0

        # Update position and costs
        if self.current_position == 0:
            # Opening new position
            self.entry_price = fill_price
            self.entry_time = current_state.ts
            self.entry_cost = trade_cost
        elif target_position == 0:
            # Closing position
            if self.entry_price is not None and np.isfinite(self.entry_price):
                position_pnl = (
                    (fill_price - self.entry_price) * self.current_position
                    * self.value_per_tick / self.tick_size
                )

                # Validate PnL calculation
                if np.isfinite(position_pnl):
                    self.balance += position_pnl - self.entry_cost - trade_cost

                    # Ensure balance remains finite
                    if not np.isfinite(self.balance):
                        self.balance = 0.0
            else:
                position_pnl = 0.0

            self.exit_price = fill_price
            self.exit_time = current_state.ts

            # Record trade
            self.trades.append({
                'entry_time': self.entry_time,
                'exit_time': self.exit_time,
                'entry_price': self.entry_price,
                'exit_price': fill_price,
                'position_size': self.current_position,
                'pnl': position_pnl if 'position_pnl' in locals() else 0,
                'total_cost': self.entry_cost + trade_cost
            })

            # Reset position tracking
            self.entry_price = None
            self.entry_time = None
            self.entry_cost = 0.0
        else:
            # Position reversal - close old, open new
            if self.entry_price is not None and np.isfinite(self.entry_price):
                position_pnl = (
                    (fill_price - self.entry_price) * self.current_position
                    * self.value_per_tick / self.tick_size
                )

                # Validate PnL calculation
                if np.isfinite(position_pnl):
                    self.balance += position_pnl - self.entry_cost - trade_cost

                    # Ensure balance remains finite
                    if not np.isfinite(self.balance):
                        self.balance = 0.0

            self.entry_price = fill_price
            self.entry_time = current_state.ts
            self.entry_cost = trade_cost

        # Record the order
        self.orders.append({
            'time': current_state.ts,
            'action': 'buy' if trade_size > 0 else 'sell',
            'size': abs(trade_size),
            'price': fill_price,
            'cost': trade_cost
        })

        self.last_position = self.current_position
        self.current_position = target_position