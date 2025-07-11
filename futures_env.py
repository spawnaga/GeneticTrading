#!/usr/bin/env python
"""
Enhanced Futures Trading Environment with Market Microstructure
===============================================================

Realistic NQ futures trading simulation with:
- Market impact modeling
- Liquidity dynamics
- Slippage based on order size and volatility
- Time-of-day effects
- Market regime detection
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
from typing import Dict, List, Optional, Tuple

from utils import round_to_nearest_increment, monotonicity, cleanup_old_logs

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """Detect market conditions: trending, ranging, volatile, calm"""

    def __init__(self, window=50):
        self.window = window
        self.price_history = []
        self.volume_history = []

    def update(self, price: float, volume: float):
        self.price_history.append(price)
        self.volume_history.append(volume)

        if len(self.price_history) > self.window:
            self.price_history.pop(0)
            self.volume_history.pop(0)

    def detect_regime(self) -> Dict[str, float]:
        """Return market regime indicators"""
        if len(self.price_history) < self.window:
            return {"trending": 0.5, "volatile": 0.5, "liquid": 0.5}

        prices = np.array(self.price_history)
        volumes = np.array(self.volume_history)

        # Trend strength
        returns = np.diff(prices) / prices[:-1]
        trend_strength = abs(np.mean(returns)) / (np.std(returns) + 1e-8)

        # Volatility
        volatility = np.std(returns) * np.sqrt(252)
        vol_normalized = min(volatility / 0.3, 1.0)  # Normalize to 30% annual vol

        # Liquidity proxy
        avg_volume = np.mean(volumes)
        liquidity = min(avg_volume / 10000, 1.0)  # Normalize volume

        return {
            "trending": min(trend_strength, 1.0),
            "volatile": vol_normalized,
            "liquid": liquidity
        }


class TimeSeriesState:
    """Enhanced state with market microstructure data."""

    def __init__(self, ts, open_price, high_price=None, low_price=None, 
                 close_price=None, volume=None, features=None, bid_ask_spread=None,
                 order_flow=None, market_depth=None):
        self.ts = ts
        self.open_price = open_price
        self.high_price = high_price or open_price
        self.low_price = low_price or open_price
        self.close_price = close_price or open_price
        self.volume = volume or 1000
        self.features = np.array(features, dtype=np.float32) if features is not None else None
        self.bid_ask_spread = bid_ask_spread or 0.25
        self.order_flow = order_flow or 0.0  # Net buying/selling pressure
        self.market_depth = market_depth or 100  # Liquidity at best bid/ask


class FuturesEnv(gym.Env):
    """
    Enhanced futures trading environment with realistic market microstructure.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, states, value_per_tick=12.5, tick_size=0.25, 
                 initial_capital=100000, max_position_size=10, 
                 commission_per_contract=2.50, **kwargs):
        super().__init__()

        self.states = states
        self.limit = len(states)
        self.value_per_tick = value_per_tick
        self.tick_size = tick_size
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.commission_per_contract = commission_per_contract

        # Market microstructure components
        self.regime_detector = MarketRegimeDetector()
        self.liquidity_factor = 1.0
        self.market_impact_coefficient = 0.001

        # Enhanced state tracking
        self.current_index = 0
        self.done = False
        self.account_balance = initial_capital
        self.unrealized_pnl = 0.0
        self.current_position = 0  # -max_position_size to +max_position_size
        self.position_entry_price = 0.0
        self.position_entry_time = None

        # Performance tracking
        self.trades = []
        self.daily_pnl = []
        self.equity_curve = [initial_capital]
        self.drawdown_series = []
        self.sharpe_window = []
        self.max_equity = initial_capital

        # Market condition adaptation
        self.time_of_day_factor = 1.0
        self.volatility_regime = "normal"
        self.liquidity_regime = "normal"

        # Action and observation spaces
        self.action_space = spaces.Discrete(7)  # 0=hold, 1-3=buy(1,2,3), 4-6=sell(1,2,3)

        # Enhanced observation space
        base_features = len(states[0].features) if states and states[0].features is not None else 10
        position_info = 5  # position, unrealized_pnl, account_balance, regime indicators
        market_info = 8   # time_of_day, volatility, liquidity, order_flow, etc.

        obs_dim = base_features + position_info + market_info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Reset environment with enhanced initialization."""
        super().reset(seed=seed)

        # Reset core state
        self.current_index = np.random.randint(0, max(1, self.limit - 1000)) if self.limit > 1000 else 0
        self.done = False
        self.account_balance = self.initial_capital
        self.unrealized_pnl = 0.0
        self.current_position = 0
        self.position_entry_price = 0.0
        self.position_entry_time = None

        # Reset tracking
        self.trades.clear()
        self.daily_pnl.clear()
        self.equity_curve = [self.initial_capital]
        self.drawdown_series = [0.0]
        self.sharpe_window.clear()
        self.max_equity = self.initial_capital

        # Reset market components
        self.regime_detector = MarketRegimeDetector()

        return self._get_observation(), {}

    def step(self, action):
        """Enhanced step function with realistic trading mechanics."""
        if self.done or self.current_index >= self.limit:
            return self._get_observation(), 0.0, True, {}

        # Validate and process action
        action = int(np.clip(action, 0, 6))

        # Get current market state
        current_state = self.states[self.current_index]
        self._update_market_conditions(current_state)

        # Execute trade
        reward = 0.0
        if action != 0:  # Not hold
            reward = self._execute_enhanced_trade(action, current_state)

        # Update position valuation
        self._update_position_value(current_state.close_price)

        # Calculate step reward
        step_reward = self._calculate_enhanced_reward(current_state, action)

        # Move to next state
        self.current_index += 1
        if self.current_index >= self.limit:
            self.done = True
            step_reward += self._calculate_final_reward()

        # Update tracking
        self._update_performance_tracking()

        return self._get_observation(), step_reward, self.done, self._get_info()

    def _update_market_conditions(self, state: TimeSeriesState):
        """Update market regime and conditions."""
        self.regime_detector.update(state.close_price, state.volume)

        # Time of day effects (NQ is most liquid during US hours)
        hour = state.ts.hour if hasattr(state.ts, 'hour') else 14
        if 9 <= hour <= 16:  # US market hours
            self.time_of_day_factor = 1.0
        elif 17 <= hour <= 23 or 0 <= hour <= 8:  # After hours
            self.time_of_day_factor = 0.7
        else:
            self.time_of_day_factor = 0.5

        # Update liquidity based on volume and time
        base_liquidity = min(state.volume / 5000, 2.0)
        self.liquidity_factor = base_liquidity * self.time_of_day_factor

    def _execute_enhanced_trade(self, action: int, state: TimeSeriesState) -> float:
        """Execute trade with realistic market impact and slippage."""
        # Decode action: 1-3 = buy 1-3 contracts, 4-6 = sell 1-3 contracts
        if 1 <= action <= 3:
            direction = 1
            size = action
        else:
            direction = -1
            size = action - 3

        new_position = np.clip(
            self.current_position + (direction * size),
            -self.max_position_size, 
            self.max_position_size
        )

        actual_trade_size = new_position - self.current_position
        if actual_trade_size == 0:
            return 0.0

        # Calculate execution price with market impact
        execution_price = self._calculate_execution_price(
            state, actual_trade_size
        )

        # Handle position changes
        reward = 0.0
        if self.current_position == 0:
            # Opening new position
            self.position_entry_price = execution_price
            self.position_entry_time = state.ts
        elif new_position == 0:
            # Closing position
            reward = self._close_position(execution_price, state.ts)
        elif np.sign(new_position) != np.sign(self.current_position):
            # Reversing position
            reward = self._close_position(execution_price, state.ts)
            self.position_entry_price = execution_price
            self.position_entry_time = state.ts
        else:
            # Adding to existing position
            # Update average entry price
            total_size = abs(new_position)
            existing_size = abs(self.current_position)
            new_size = abs(actual_trade_size)

            self.position_entry_price = (
                (self.position_entry_price * existing_size + execution_price * new_size) / 
                total_size
            )

        # Update position
        self.current_position = new_position

        # Deduct commission
        commission = self.commission_per_contract * abs(actual_trade_size)
        self.account_balance -= commission

        return reward - commission / self.initial_capital  # Normalize commission impact

    def _calculate_execution_price(self, state: TimeSeriesState, trade_size: int) -> float:
        """Calculate realistic execution price with slippage and market impact."""
        base_price = state.close_price

        # Market impact based on trade size and liquidity
        impact_factor = abs(trade_size) / self.max_position_size
        liquidity_adjustment = 1.0 / max(self.liquidity_factor, 0.1)
        market_impact = self.market_impact_coefficient * impact_factor * liquidity_adjustment

        # Add bid-ask spread
        spread_cost = state.bid_ask_spread / 2

        # Volatility-based slippage
        regime = self.regime_detector.detect_regime()
        volatility_slippage = regime.get("volatile", 0.5) * 0.5 * self.tick_size

        # Total slippage
        direction = np.sign(trade_size)
        total_slippage = (market_impact + spread_cost + volatility_slippage) * self.tick_size

        execution_price = base_price + (direction * total_slippage)

        # Ensure price is within reasonable bounds
        if state.high_price and state.low_price:
            execution_price = np.clip(execution_price, state.low_price, state.high_price)

        return round_to_nearest_increment(execution_price, self.tick_size)

    def _close_position(self, exit_price: float, exit_time) -> float:
        """Close current position and calculate P&L."""
        if self.current_position == 0 or self.position_entry_price == 0:
            return 0.0

        # Calculate P&L
        price_diff = exit_price - self.position_entry_price
        ticks = price_diff / self.tick_size
        pnl = self.current_position * ticks * self.value_per_tick

        # Add to account balance
        self.account_balance += pnl

        # Record trade
        self.trades.append({
            'entry_time': self.position_entry_time,
            'exit_time': exit_time,
            'entry_price': self.position_entry_price,
            'exit_price': exit_price,
            'size': self.current_position,
            'pnl': pnl,
            'duration': (exit_time - self.position_entry_time).total_seconds() if self.position_entry_time else 0
        })

        return pnl / self.initial_capital  # Normalized reward

    def _update_position_value(self, current_price: float):
        """Update unrealized P&L for open position."""
        if self.current_position != 0 and self.position_entry_price != 0:
            price_diff = current_price - self.position_entry_price
            ticks = price_diff / self.tick_size
            self.unrealized_pnl = self.current_position * ticks * self.value_per_tick
        else:
            self.unrealized_pnl = 0.0

    def _calculate_enhanced_reward(self, state: TimeSeriesState, action: int) -> float:
        """Calculate sophisticated reward function."""
        # Base reward: change in total equity
        total_equity = self.account_balance + self.unrealized_pnl
        equity_change = total_equity - self.equity_curve[-1] if self.equity_curve else 0.0
        base_reward = equity_change / self.initial_capital

        # Risk-adjusted returns
        regime = self.regime_detector.detect_regime()

        # Penalty for excessive position size in volatile markets
        position_penalty = 0.0
        if regime.get("volatile", 0.5) > 0.7:
            position_ratio = abs(self.current_position) / self.max_position_size
            position_penalty = -0.001 * position_ratio * regime["volatile"]

        # Reward for trading with the trend
        trend_reward = 0.0
        if action != 0 and len(self.equity_curve) > 10:
            recent_trend = np.polyfit(range(10), self.equity_curve[-10:], 1)[0]
            if action in [1, 2, 3] and recent_trend > 0:  # Buy with uptrend
                trend_reward = 0.0005 * regime.get("trending", 0.5)
            elif action in [4, 5, 6] and recent_trend < 0:  # Sell with downtrend
                trend_reward = 0.0005 * regime.get("trending", 0.5)

        # Liquidity reward (trading when liquid)
        liquidity_reward = 0.0001 * regime.get("liquid", 0.5) if action != 0 else 0.0

        # Drawdown penalty
        drawdown_penalty = 0.0
        current_drawdown = (self.max_equity - total_equity) / self.max_equity
        if current_drawdown > 0.1:  # More than 10% drawdown
            drawdown_penalty = -0.01 * current_drawdown

        total_reward = (base_reward + trend_reward + liquidity_reward + 
                       position_penalty + drawdown_penalty)

        return np.clip(total_reward, -0.1, 0.1)  # Prevent extreme rewards

    def _calculate_final_reward(self) -> float:
        """Calculate final episode reward based on overall performance."""
        if len(self.equity_curve) < 2:
            return 0.0

        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital

        # Sharpe ratio calculation
        if len(self.daily_pnl) > 1:
            sharpe = np.mean(self.daily_pnl) / (np.std(self.daily_pnl) + 1e-8)
            sharpe_bonus = np.clip(sharpe * 0.1, -0.05, 0.05)
        else:
            sharpe_bonus = 0.0

        # Maximum drawdown penalty
        max_dd = max(self.drawdown_series) if self.drawdown_series else 0.0
        drawdown_penalty = -max_dd * 0.1

        return total_return + sharpe_bonus + drawdown_penalty

    def _update_performance_tracking(self):
        """Update performance metrics."""
        total_equity = self.account_balance + self.unrealized_pnl
        self.equity_curve.append(total_equity)

        # Update max equity and drawdown
        if total_equity > self.max_equity:
            self.max_equity = total_equity

        current_drawdown = (self.max_equity - total_equity) / self.max_equity
        self.drawdown_series.append(current_drawdown)

        # Daily P&L (simplified - assuming each step is a day)
        if len(self.equity_curve) > 1:
            daily_change = self.equity_curve[-1] - self.equity_curve[-2]
            self.daily_pnl.append(daily_change / self.initial_capital)

    def _get_observation(self):
        """Get enhanced observation with market regime information."""
        if self.current_index >= len(self.states):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        current_state = self.states[self.current_index]

        # Base features
        base_features = current_state.features if current_state.features is not None else np.zeros(10)
        base_features = np.array(base_features, dtype=np.float32)

        # Position information
        position_info = np.array([
            self.current_position / self.max_position_size,  # Normalized position
            self.unrealized_pnl / self.initial_capital,      # Normalized unrealized P&L
            (self.account_balance - self.initial_capital) / self.initial_capital,  # Account performance
            len(self.trades) / 100.0,                       # Trade count (normalized)
            (self.equity_curve[-1] - self.initial_capital) / self.initial_capital  # Total performance
        ], dtype=np.float32)

        # Market regime information
        regime = self.regime_detector.detect_regime()
        market_info = np.array([
            self.time_of_day_factor,
            regime.get("trending", 0.5),
            regime.get("volatile", 0.5),
            regime.get("liquid", 0.5),
            current_state.order_flow if current_state.order_flow else 0.0,
            current_state.bid_ask_spread / self.tick_size,
            self.liquidity_factor,
            max(self.drawdown_series) if self.drawdown_series else 0.0
        ], dtype=np.float32)

        # Combine all features
        observation = np.concatenate([base_features, position_info, market_info])

        # Ensure finite values
        observation = np.where(np.isfinite(observation), observation, 0.0)
        observation = np.clip(observation, -10.0, 10.0)

        return observation.astype(np.float32)

    def _get_info(self):
        """Return comprehensive info dictionary."""
        total_equity = self.account_balance + self.unrealized_pnl
        regime = self.regime_detector.detect_regime()

        return {
            'account_balance': self.account_balance,
            'unrealized_pnl': self.unrealized_pnl,
            'total_equity': total_equity,
            'current_position': self.current_position,
            'total_trades': len(self.trades),
            'current_drawdown': self.drawdown_series[-1] if self.drawdown_series else 0.0,
            'max_drawdown': max(self.drawdown_series) if self.drawdown_series else 0.0,
            'market_regime': regime,
            'time_of_day_factor': self.time_of_day_factor,
            'liquidity_factor': self.liquidity_factor
        }

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        if len(self.equity_curve) < 2:
            return {}

        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital

        # Calculate metrics
        if len(self.daily_pnl) > 1:
            sharpe = np.mean(self.daily_pnl) / (np.std(self.daily_pnl) + 1e-8) * np.sqrt(252)
            max_dd = max(self.drawdown_series) if self.drawdown_series else 0.0
        else:
            sharpe = 0.0
            max_dd = 0.0

        # Trade statistics
        if self.trades:
            profitable_trades = sum(1 for trade in self.trades if trade['pnl'] > 0)
            win_rate = profitable_trades / len(self.trades)
            avg_trade = np.mean([trade['pnl'] for trade in self.trades])
        else:
            win_rate = 0.0
            avg_trade = 0.0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_trade_pnl': avg_trade,
            'final_equity': self.equity_curve[-1]
        }