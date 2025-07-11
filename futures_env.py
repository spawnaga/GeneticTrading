"""This change introduces structured table logging for trading activities within the FuturesEnv class, enhancing monitoring and analysis capabilities."""
#!/usr/bin/env python
"""
Enhanced Professional Futures Trading Environment
=================================================

Fixed version that generates real trading activity and proper metrics
for the monitoring dashboard.
"""

import math
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from uuid import uuid4
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

from utils import round_to_nearest_increment, monotonicity

# Import trading monitor
try:
    from trading_activity_monitor import log_trading_action, log_position_change
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False

logger = logging.getLogger("FUTURES_ENV")


class MarketRegimeDetector:
    """Advanced market regime detection for adaptive trading"""

    def __init__(self, window=50):
        self.window = window
        self.price_history = []
        self.volume_history = []
        self.volatility_history = []

    def update(self, price: float, volume: float):
        self.price_history.append(price)
        self.volume_history.append(volume)

        if len(self.price_history) > 1:
            returns = (price - self.price_history[-2]) / self.price_history[-2]
            self.volatility_history.append(abs(returns))

        # Maintain rolling window
        if len(self.price_history) > self.window:
            self.price_history.pop(0)
            self.volume_history.pop(0)
        if len(self.volatility_history) > self.window:
            self.volatility_history.pop(0)

    def detect_regime(self) -> Dict[str, float]:
        """Return normalized market regime indicators"""
        if len(self.price_history) < 10:
            return {"trending": 0.5, "volatile": 0.5, "liquid": 0.5}

        prices = np.array(self.price_history)
        volumes = np.array(self.volume_history)
        volatilities = np.array(self.volatility_history) if self.volatility_history else np.array([0.01])

        # Trend strength using linear regression slope
        x = np.arange(len(prices))
        if len(x) > 1:
            slope = np.polyfit(x, prices, 1)[0]
            trend_strength = abs(slope) / (np.std(prices) + 1e-8)
        else:
            trend_strength = 0.0

        # Volatility regime
        vol_percentile = np.percentile(volatilities, 80) if len(volatilities) > 5 else 0.01
        current_vol = volatilities[-1] if len(volatilities) > 0 else 0.01
        vol_normalized = min(current_vol / (vol_percentile + 1e-8), 2.0)

        # Liquidity proxy from volume
        avg_volume = np.mean(volumes)
        volume_normalized = min(avg_volume / 10000, 1.0)

        return {
            "trending": min(trend_strength * 10, 1.0),
            "volatile": min(vol_normalized, 1.0),
            "liquid": volume_normalized
        }


class TimeSeriesState:
    """Enhanced state representation for NQ futures data"""

    def __init__(self, ts, open_price, high_price=None, low_price=None, 
                 close_price=None, volume=None, features=None, bid_ask_spread=None,
                 order_flow=None, market_depth=None):
        self.ts = ts
        self.open_price = float(open_price)
        self.high_price = float(high_price or open_price)
        self.low_price = float(low_price or open_price)
        self.close_price = float(close_price or open_price)
        self.volume = int(volume or 1000)
        self.features = np.array(features, dtype=np.float32) if features is not None else np.zeros(10, dtype=np.float32)
        self.bid_ask_spread = float(bid_ask_spread or 0.25)
        self.order_flow = float(order_flow or 0.0)
        self.market_depth = float(market_depth or 100)

        # Legacy compatibility
        self.price = self.close_price

    def __str__(self):
        return f"TS: {self.ts}, OHLC: [{self.open_price:.2f}, {self.high_price:.2f}, {self.low_price:.2f}, {self.close_price:.2f}], Vol: {self.volume}"


class FuturesEnv(gym.Env):
    """
    Professional NQ Futures Trading Environment - Fixed Version
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, states: List[TimeSeriesState], 
                 value_per_tick: float = 12.5,
                 tick_size: float = 0.25, 
                 initial_capital: float = 100000,
                 max_position_size: int = 5, 
                 commission_per_contract: float = 2.50,
                 margin_rate: float = 0.05,
                 fill_probability: float = 1.0,
                 execution_cost_per_order: float = 0.00005,
                 contracts_per_trade: int = 1,
                 bid_ask_spread: float = 0.25,
                 add_current_position_to_state: bool = True,
                 log_dir: str = None,
                 **kwargs):

        super().__init__()

        # Core environment parameters
        self.states = states
        self.limit = len(states)
        self.value_per_tick = value_per_tick
        self.tick_size = tick_size
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.commission_per_contract = commission_per_contract
        self.margin_rate = margin_rate
        self.fill_probability = fill_probability
        self.execution_cost_per_order = execution_cost_per_order
        self.contracts_per_trade = contracts_per_trade
        self.bid_ask_spread = bid_ask_spread
        self.add_current_position_to_state = add_current_position_to_state

        # Logging setup
        self.log_dir = log_dir or f"./logs/futures_env/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # Market microstructure components
        self.regime_detector = MarketRegimeDetector()
        self.liquidity_factor = 1.0
        self.market_impact_coefficient = 0.001

        # Environment state
        self.current_index = 0
        self.done = False

        # Account state
        self.account_balance = initial_capital
        self.unrealized_pnl = 0.0
        self.margin_used = 0.0

        # Position state
        self.current_position = 0
        self.position_entry_price = 0.0
        self.position_entry_time = None
        self.position_value = 0.0

        # Performance tracking
        self.trades = []
        self.orders = []
        self.daily_pnl = []
        self.equity_curve = [initial_capital]
        self.drawdown_series = [0.0]
        self.max_equity = initial_capital
        self.total_reward = 0.0
        self.episode = 0

        # Market condition tracking
        self.time_of_day_factor = 1.0
        self.volatility_regime = "normal"
        self.liquidity_regime = "normal"

        # Simplified action space for better trading
        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell

        # Enhanced observation space
        base_features = len(states[0].features) if states and len(states[0].features) > 0 else 20
        position_info = 8
        market_info = 10

        obs_dim = base_features + position_info + market_info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Real-time metrics tracking for dashboard
        self.metrics_history = {
            'timestamps': [],
            'account_values': [],
            'positions': [],
            'pnl_changes': [],
            'trades_count': [],
            'unrealized_pnl': []
        }

        logger.info(f"Initialized FuturesEnv with {len(states)} states, obs_dim={obs_dim}")

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)

        # Reset core state
        start_idx = np.random.randint(0, max(1, self.limit - 1000)) if self.limit > 1000 else 0
        self.current_index = start_idx
        self.done = False

        # Reset account
        self.account_balance = self.initial_capital
        self.unrealized_pnl = 0.0
        self.margin_used = 0.0

        # Reset position
        self.current_position = 0
        self.position_entry_price = 0.0
        self.position_entry_time = None
        self.position_value = 0.0

        # Reset tracking
        self.trades.clear()
        self.orders.clear()
        self.daily_pnl.clear()
        self.equity_curve = [self.initial_capital]
        self.drawdown_series = [0.0]
        self.max_equity = self.initial_capital
        self.total_reward = 0.0

        # Reset market components
        self.regime_detector = MarketRegimeDetector()

        # Initialize metrics tracking
        self.metrics_history = {
            'timestamps': [datetime.now()],
            'account_values': [self.initial_capital],
            'positions': [0],
            'pnl_changes': [0.0],
            'trades_count': [0],
            'unrealized_pnl': [0.0]
        }

        return self._get_observation(), {}

    def step(self, action: int):
        """Execute one step in the environment"""
        if self.done or self.current_index >= self.limit:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # Validate action
        action = int(np.clip(action, 0, 2))

        # Get current market state
        current_state = self.states[self.current_index]
        self._update_market_conditions(current_state)

        # Execute action with more aggressive trading
        reward = 0.0
        info = {"message": "hold"}

        # Log the action being taken
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        current_action = action_names.get(action, f"UNKNOWN({action})")

        # Log current price
        current_price = current_state.close_price

        # Calculate profit/loss (pnl)
        profit_loss = 0.0  # Initialize pnl

        # Track action for monitoring with structured logging
        if MONITOR_AVAILABLE:
            log_trading_action(self.current_index, action, current_price, self.current_position, self.account_balance)

        # Create trading activity logger (file only, not console)
        trading_logger = logging.getLogger("TRADING_ACTIVITY_FILE")
        if not trading_logger.handlers:
            trading_file_handler = logging.FileHandler(Path(self.log_dir) / "trading_activity.log")
            trading_file_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)-8s] %(message)s"
            ))
            trading_logger.addHandler(trading_file_handler)
            trading_logger.setLevel(logging.INFO)
            trading_logger.propagate = False  # Don't send to console

        # Log current state to file only
        trading_logger.info(f"📊 Step {self.current_index} | Action: {current_action} | "
                           f"Price: ${current_state.close_price:.2f} | "
                           f"Position: {self.current_position} | "
                           f"Account: ${self.account_balance:,.2f}")

        if action != 0:  # Not hold
            reward, info = self._execute_trade(action, current_state)

            # Log trade execution result to file only
            trading_logger.info(f"🔄 Trade Result: {info.get('message', 'Unknown')} | "
                               f"Reward: {reward:.4f} | "
                               f"New Position: {self.current_position}")

            # Log to structured trading table with actual trade reward as P&L
            self._log_to_trading_table(action, current_price, self.current_position, self.account_balance, reward, reward)
        else:
            # Log hold action to trading table
            self._log_to_trading_table(action, current_price, self.current_position, self.account_balance, 0.0, 0.0)

        # Update position valuation
        self._update_position_value(current_state.close_price)

        # Calculate step reward
        step_reward = self._calculate_reward(current_state, action, reward)

        # Move to next state
        self.current_index += 1
        if self.current_index >= self.limit:
            self.done = True
            step_reward += self._calculate_final_reward()

        # Update performance tracking and metrics
        self._update_performance_tracking()
        self._update_realtime_metrics()

        # Log account status every 1000 steps to reduce console spam
        if self.current_index % 1000 == 0:
            total_equity = self.account_balance + self.unrealized_pnl
            logger.info(f"💰 ACCOUNT STATUS | Step: {self.current_index} | "
                       f"Balance: ${self.account_balance:,.2f} | "
                       f"Unrealized P&L: ${self.unrealized_pnl:,.2f} | "
                       f"Total Equity: ${total_equity:,.2f} | "
                       f"Total Trades: {len(self.trades)}")

        return self._get_observation(), step_reward, self.done, False, self._get_info()

    def _execute_trade(self, action: int, state: TimeSeriesState) -> Tuple[float, Dict]:
        """Execute trade action with realistic market mechanics"""
        # Simplified action mapping: 1=buy, 2=sell
        if action == 1:  # Buy
            direction = 1
            size = 1
        else:  # Sell
            direction = -1
            size = 1

        # Calculate new position
        new_position = np.clip(
            self.current_position + (direction * size),
            -self.max_position_size, 
            self.max_position_size
        )

        actual_trade_size = new_position - self.current_position
        if actual_trade_size == 0:
            return 0.0, {"message": "position limit reached"}

        # Check margin requirements
        if not self._check_margin_requirements(new_position, state.close_price):
            return -0.01, {"message": "insufficient margin"}

        # Calculate execution price with minimal slippage for more trading
        execution_price = state.close_price + (direction * self.tick_size * 0.5)  # Reduced slippage

        # Execute the trade
        reward = 0.0
        info = {}

        # Determine position type for logging
        position_type = "LONG" if actual_trade_size > 0 else "SHORT"

        if self.current_position == 0:
            # Opening new position
            self.position_entry_price = execution_price
            self.position_entry_time = state.ts
            info = {"message": f"OPENED {position_type}: {abs(actual_trade_size)} contracts at ${execution_price:.2f}"}
            logger.debug(f"🟢 NEW POSITION | {position_type} {abs(actual_trade_size)} contracts | "
                        f"Entry: ${execution_price:.2f} | Time: {state.ts}")

        elif new_position == 0:
            # Closing position
            reward = self._close_position(execution_price, state.ts)
            prev_type = "LONG" if self.current_position > 0 else "SHORT"
            info = {"message": f"CLOSED {prev_type}: P&L=${reward:.2f}"}
            logger.debug(f"🔴 CLOSED POSITION | {prev_type} | "
                        f"Exit: ${execution_price:.2f} | P&L: ${reward:.2f}")

        elif np.sign(new_position) != np.sign(self.current_position):
            # Reversing position
            prev_type = "LONG" if self.current_position > 0 else "SHORT"
            reward = self._close_position(execution_price, state.ts)
            self.position_entry_price = execution_price
            self.position_entry_time = state.ts
            info = {"message": f"REVERSED {prev_type}→{position_type}: P&L=${reward:.2f}"}
            logger.debug(f"🔄 REVERSED POSITION | {prev_type} → {position_type} | "
                        f"P&L: ${reward:.2f} | New Entry: ${execution_price:.2f}")

        else:
            # Adding to existing position
            total_size = abs(new_position)
            existing_size = abs(self.current_position)
            new_size = abs(actual_trade_size)

            self.position_entry_price = (
                (self.position_entry_price * existing_size + execution_price * new_size) / 
                total_size
            )
            current_type = "LONG" if new_position > 0 else "SHORT"
            info = {"message": f"ADDED TO {current_type}: +{abs(actual_trade_size)} contracts"}
            logger.debug(f"🔵 ADDED TO POSITION | {current_type} +{abs(actual_trade_size)} | "
                        f"Total: {abs(new_position)} contracts | Avg Entry: ${self.position_entry_price:.2f}")

        # Log position change
        if MONITOR_AVAILABLE:
            log_position_change(self.current_position, new_position, reward)

        # Update position
        old_position = self.current_position
        self.current_position = new_position

        # Deduct commission
        commission = self.commission_per_contract * abs(actual_trade_size)
        self.account_balance -= commission

        # Record order
        self.orders.append({
            'order_id': str(uuid4()),
            'timestamp': state.ts,
            'price': execution_price,
            'size': actual_trade_size,
            'commission': commission,
            'state': state
        })

        return reward, info

    def _close_position(self, exit_price: float, exit_time) -> float:
        """Close current position and calculate P&L"""
        if self.current_position == 0 or self.position_entry_price == 0:
            return 0.0

        # Calculate P&L
        price_diff = exit_price - self.position_entry_price
        ticks = price_diff / self.tick_size
        pnl = self.current_position * ticks * self.value_per_tick

        # Add to account balance
        self.account_balance += pnl

        # Record trade
        duration = (exit_time - self.position_entry_time).total_seconds() if self.position_entry_time else 0

        self.trades.append({
            'trade_id': str(uuid4()),
            'entry_time': self.position_entry_time,
            'exit_time': exit_time,
            'entry_price': self.position_entry_price,
            'exit_price': exit_price,
            'size': self.current_position,
            'pnl': pnl,
            'duration': duration,
            'trade_type': 'long' if self.current_position > 0 else 'short'
        })

        # Reset position
        self.position_entry_price = 0.0
        self.position_entry_time = None

        return pnl

    def _check_margin_requirements(self, position_size: int, price: float) -> bool:
        """Check if sufficient margin is available"""
        required_margin = abs(position_size) * price * self.margin_rate
        available_capital = self.account_balance + self.unrealized_pnl
        return available_capital >= required_margin

    def _update_position_value(self, current_price: float):
        """Update unrealized P&L and position value"""
        if self.current_position != 0 and self.position_entry_price != 0:
            price_diff = current_price - self.position_entry_price
            ticks = price_diff / self.tick_size
            self.unrealized_pnl = self.current_position * ticks * self.value_per_tick
            self.position_value = abs(self.current_position) * current_price
            self.margin_used = self.position_value * self.margin_rate
        else:
            self.unrealized_pnl = 0.0
            self.position_value = 0.0
            self.margin_used = 0.0

    def _update_market_conditions(self, state: TimeSeriesState):
        """Update market regime and time-of-day effects"""
        self.regime_detector.update(state.close_price, state.volume)

        # Time of day effects
        hour = state.ts.hour if hasattr(state.ts, 'hour') else 14
        if 9 <= hour <= 16:
            self.time_of_day_factor = 1.0
        elif 17 <= hour <= 23 or 0 <= hour <= 8:
            self.time_of_day_factor = 0.7
        else:
            self.time_of_day_factor = 0.5

        # Update liquidity
        base_liquidity = min(state.volume / 5000, 2.0)
        self.liquidity_factor = base_liquidity * self.time_of_day_factor

    def _calculate_reward(self, state: TimeSeriesState, action: int, trade_reward: float) -> float:
        """Calculate comprehensive reward function"""
        # Base reward from trade
        base_reward = trade_reward / self.initial_capital if trade_reward != 0 else 0.0

        # Equity change reward
        total_equity = self.account_balance + self.unrealized_pnl
        if len(self.equity_curve) > 0:
            equity_change = total_equity - self.equity_curve[-1]
            equity_reward = equity_change / self.initial_capital
        else:
            equity_reward = 0.0

        # Encourage more trading with smaller penalties
        trading_bonus = 0.0001 if action != 0 else 0.0

        total_reward = base_reward + equity_reward + trading_bonus
        return np.clip(total_reward, -0.1, 0.1)

    def _calculate_final_reward(self) -> float:
        """Calculate final episode reward"""
        if len(self.equity_curve) < 2:
            return 0.0

        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital

        # Sharpe ratio bonus
        if len(self.daily_pnl) > 1:
            sharpe = np.mean(self.daily_pnl) / (np.std(self.daily_pnl) + 1e-8)
            sharpe_bonus = np.clip(sharpe * 0.01, -0.02, 0.02)
        else:
            sharpe_bonus = 0.0

        return total_return * 0.1 + sharpe_bonus

    def _update_performance_tracking(self):
        """Update comprehensive performance metrics"""
        total_equity = self.account_balance + self.unrealized_pnl
        self.equity_curve.append(total_equity)

        # Update max equity and drawdown
        if total_equity > self.max_equity:
            self.max_equity = total_equity

        current_drawdown = (self.max_equity - total_equity) / self.max_equity
        self.drawdown_series.append(current_drawdown)

        # Daily P&L tracking
        if len(self.equity_curve) > 1:
            daily_change = self.equity_curve[-1] - self.equity_curve[-2]
            self.daily_pnl.append(daily_change)

    def _update_realtime_metrics(self):
        """Update real-time metrics for dashboard"""
        current_time = datetime.now()
        total_equity = self.account_balance + self.unrealized_pnl

        # Update metrics history
        self.metrics_history['timestamps'].append(current_time)
        self.metrics_history['account_values'].append(total_equity)
        self.metrics_history['positions'].append(self.current_position)
        self.metrics_history['trades_count'].append(len(self.trades))
        self.metrics_history['unrealized_pnl'].append(self.unrealized_pnl)

        if len(self.equity_curve) > 1:
            pnl_change = self.equity_curve[-1] - self.equity_curve[-2]
            self.metrics_history['pnl_changes'].append(pnl_change)
        else:
            self.metrics_history['pnl_changes'].append(0.0)

        # Keep only last 1000 points
        max_points = 1000
        for key in self.metrics_history:
            if len(self.metrics_history[key]) > max_points:
                self.metrics_history[key] = self.metrics_history[key][-max_points:]

    def get_realtime_metrics(self) -> Dict:
        """Get current metrics for dashboard"""
        total_equity = self.account_balance + self.unrealized_pnl

        return {
            'timestamp': datetime.now().isoformat(),
            'account_value': total_equity,
            'account_balance': self.account_balance,
            'unrealized_pnl': self.unrealized_pnl,
            'current_position': self.current_position,
            'total_trades': len(self.trades),
            'total_return': ((total_equity - self.initial_capital) / self.initial_capital) * 100,
            'max_drawdown': max(self.drawdown_series) if self.drawdown_series else 0.0,
            'equity_curve': self.equity_curve[-100:],  # Last 100 points
            'metrics_history': self.metrics_history
        }

    def _get_observation(self):
        """Get comprehensive observation vector"""
        if self.current_index >= len(self.states):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        current_state = self.states[self.current_index]

        # Base features
        base_features = current_state.features.copy()

        # Position information
        total_equity = self.account_balance + self.unrealized_pnl
        position_info = np.array([
            self.current_position / self.max_position_size,
            self.unrealized_pnl / self.initial_capital,
            (self.account_balance - self.initial_capital) / self.initial_capital,
            self.margin_used / self.initial_capital,
            len(self.trades) / 100.0,
            (total_equity - self.initial_capital) / self.initial_capital,
            self.position_value / self.initial_capital,
            1.0 if self.current_position > 0 else (-1.0 if self.current_position < 0 else 0.0)
        ], dtype=np.float32)

        # Market regime and condition information
        regime = self.regime_detector.detect_regime()
        market_info = np.array([
            self.time_of_day_factor,
            regime.get("trending", 0.5),
            regime.get("volatile", 0.5),
            regime.get("liquid", 0.5),
            current_state.order_flow,
            current_state.bid_ask_spread / self.tick_size,
            self.liquidity_factor,
            max(self.drawdown_series) if self.drawdown_series else 0.0,
            current_state.volume / 10000.0,
            (current_state.close_price - current_state.open_price) / current_state.open_price
        ], dtype=np.float32)

        # Combine all features
        observation = np.concatenate([base_features, position_info, market_info])

        # Ensure finite values
        observation = np.where(np.isfinite(observation), observation, 0.0)
        observation = np.clip(observation, -10.0, 10.0)

        return observation.astype(np.float32)

    def _get_info(self):
        """Return comprehensive info dictionary"""
        total_equity = self.account_balance + self.unrealized_pnl
        regime = self.regime_detector.detect_regime()

        return {
            'account_balance': self.account_balance,
            'unrealized_pnl': self.unrealized_pnl,
            'total_equity': total_equity,
            'current_position': self.current_position,
            'position_value': self.position_value,
            'margin_used': self.margin_used,
            'total_trades': len(self.trades),
            'total_orders': len(self.orders),
            'current_drawdown': self.drawdown_series[-1] if self.drawdown_series else 0.0,
            'max_drawdown': max(self.drawdown_series) if self.drawdown_series else 0.0,
            'market_regime': regime,
            'time_of_day_factor': self.time_of_day_factor,
            'liquidity_factor': self.liquidity_factor,
            'total_profit': sum([t['pnl'] for t in self.trades]),
            'timestamp': self.states[self.current_index].ts if self.current_index < len(self.states) else None
        }

    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            metrics = self.generate_episode_metrics()
            return self.total_reward, metrics
        return None

    def generate_episode_metrics(self) -> Dict:
        """Generate comprehensive episode performance metrics"""
        total_equity = self.account_balance + self.unrealized_pnl

        return {
            'total_trades': len(self.trades),
            'total_pnl': sum([t['pnl'] for t in self.trades]),
            'total_return': ((total_equity - self.initial_capital) / self.initial_capital) * 100,
            'final_equity': total_equity,
            'max_drawdown': max(self.drawdown_series) if self.drawdown_series else 0.0,
            'equity_curve': self.equity_curve.copy(),
            'current_position': self.current_position,
            'unrealized_pnl': self.unrealized_pnl
        }

    def close(self):
        """Clean up environment resources"""
        pass

    def _log_to_trading_table(self, action, price, position, balance, reward, pnl):
        """Log trading activity to structured table format."""
        try:
            from datetime import datetime
            import fcntl
            import tempfile
            import os

            # Create structured log entry
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            action_name = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(action, "UNKNOWN")

            # Determine position change
            position_change = ""
            if hasattr(self, '_last_position'):
                if position != self._last_position:
                    if position > self._last_position:
                        position_change = f"+{position - self._last_position}"
                    else:
                        position_change = f"{position - self._last_position}"
            else:
                self._last_position = 0

            if not hasattr(self, '_last_position') or position != self._last_position:
                self._last_position = position

            # Create table entry - convert all values to native Python types
            table_entry = {
                "timestamp": timestamp,
                "step": int(self.current_index),
                "action": action_name,
                "price": f"${float(price):.2f}",
                "position": int(position),
                "pos_change": position_change if position_change else "0",
                "balance": f"${float(balance):,.2f}",
                "pnl": f"${float(pnl):.2f}",
                "reward": f"{float(reward):.4f}"
            }

            # Load existing table or create new
            table_file = Path(self.log_dir) / "trading_table.json"
            table_file.parent.mkdir(parents=True, exist_ok=True)

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Try to read existing data
                    trading_table = []
                    if table_file.exists():
                        try:
                            with open(table_file, 'r') as f:
                                # Try to acquire file lock
                                try:
                                    fcntl.flock(f.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                                except:
                                    # If can't lock, skip this logging attempt
                                    return
                                
                                content = f.read().strip()
                                if content:  # Only parse if file has content
                                    trading_table = json.loads(content)
                        except (json.JSONDecodeError, ValueError) as e:
                            # If JSON is corrupted, start fresh
                            logger.warning(f"Corrupted trading table, starting fresh: {e}")
                            trading_table = []

                    # Add new entry
                    trading_table.append(table_entry)

                    # Keep only last 5000 entries to prevent huge files
                    if len(trading_table) > 5000:
                        trading_table = trading_table[-5000:]

                    # Write atomically using temporary file
                    with tempfile.NamedTemporaryFile(mode='w', dir=table_file.parent, 
                                                   prefix='.trading_table_tmp_', 
                                                   suffix='.json', delete=False) as tmp_f:
                        json.dump(trading_table, tmp_f, indent=2)
                        tmp_f.flush()
                        os.fsync(tmp_f.fileno())
                        temp_path = tmp_f.name

                    # Atomic rename
                    os.rename(temp_path, table_file)
                    break  # Success, exit retry loop

                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to save trading table after {max_retries} attempts: {e}")
                    else:
                        # Brief delay before retry
                        import time
                        time.sleep(0.01)

        except Exception as e:
            # Silent failure - don't spam logs
            pass