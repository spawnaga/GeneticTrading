
#!/usr/bin/env python
"""
Enhanced Professional Futures Trading Environment
=================================================

Highly optimized trading environment specifically designed for NQ futures
with 1-minute OHLCV data. Features realistic market microstructure,
advanced position management, and comprehensive performance tracking.
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

from utils import round_to_nearest_increment, monotonicity, cleanup_old_logs

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
    Professional NQ Futures Trading Environment
    
    Designed specifically for NQ futures with 1-minute OHLCV data.
    Features realistic market microstructure, commission modeling,
    and comprehensive performance tracking.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, states: List[TimeSeriesState], 
                 value_per_tick: float = 12.5,
                 tick_size: float = 0.25, 
                 initial_capital: float = 100000,
                 max_position_size: int = 10, 
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
        self.current_position = 0  # -max_position_size to +max_position_size
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

        # Action space: 0=hold, 1-3=buy(1,2,3 contracts), 4-6=sell(1,2,3 contracts)
        self.action_space = spaces.Discrete(7)

        # Enhanced observation space
        base_features = len(states[0].features) if states and len(states[0].features) > 0 else 20
        position_info = 8  # position, unrealized_pnl, account_balance, margin, etc.
        market_info = 10   # time_of_day, volatility, liquidity, regime indicators, etc.
        
        obs_dim = base_features + position_info + market_info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

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

        return self._get_observation(), {}

    def step(self, action: int):
        """Execute one step in the environment"""
        if self.done or self.current_index >= self.limit:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # Validate action
        action = int(np.clip(action, 0, 6))
        
        # Get current market state
        current_state = self.states[self.current_index]
        self._update_market_conditions(current_state)

        # Execute action
        reward = 0.0
        info = {"message": "hold"}
        
        if action != 0:  # Not hold
            reward, info = self._execute_trade(action, current_state)

        # Update position valuation
        self._update_position_value(current_state.close_price)

        # Calculate step reward
        step_reward = self._calculate_reward(current_state, action, reward)

        # Move to next state
        self.current_index += 1
        if self.current_index >= self.limit:
            self.done = True
            step_reward += self._calculate_final_reward()

        # Update performance tracking
        self._update_performance_tracking()

        return self._get_observation(), step_reward, self.done, False, self._get_info()

    def _execute_trade(self, action: int, state: TimeSeriesState) -> Tuple[float, Dict]:
        """Execute trade action with realistic market mechanics"""
        # Decode action: 1-3 = buy 1-3 contracts, 4-6 = sell 1-3 contracts
        if 1 <= action <= 3:
            direction = 1
            size = action
        else:
            direction = -1
            size = action - 3

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

        # Calculate execution price with slippage
        execution_price = self._calculate_execution_price(state, actual_trade_size)

        # Execute the trade
        reward = 0.0
        info = {}
        
        if self.current_position == 0:
            # Opening new position
            self.position_entry_price = execution_price
            self.position_entry_time = state.ts
            info = {"message": f"opened position: {actual_trade_size} contracts at {execution_price:.2f}"}
            
        elif new_position == 0:
            # Closing position
            reward = self._close_position(execution_price, state.ts)
            info = {"message": f"closed position: pnl={reward:.2f}"}
            
        elif np.sign(new_position) != np.sign(self.current_position):
            # Reversing position
            reward = self._close_position(execution_price, state.ts)
            self.position_entry_price = execution_price
            self.position_entry_time = state.ts
            info = {"message": f"reversed position: pnl={reward:.2f}"}
            
        else:
            # Adding to existing position (average price)
            total_size = abs(new_position)
            existing_size = abs(self.current_position)
            new_size = abs(actual_trade_size)
            
            self.position_entry_price = (
                (self.position_entry_price * existing_size + execution_price * new_size) / 
                total_size
            )
            info = {"message": f"added to position: {actual_trade_size} contracts"}

        # Update position
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

    def _calculate_execution_price(self, state: TimeSeriesState, trade_size: int) -> float:
        """Calculate realistic execution price with market impact"""
        base_price = state.close_price

        # Market impact based on trade size and liquidity
        impact_factor = abs(trade_size) / self.max_position_size
        liquidity_adjustment = 1.0 / max(self.liquidity_factor, 0.1)
        market_impact = self.market_impact_coefficient * impact_factor * liquidity_adjustment

        # Bid-ask spread cost
        spread_cost = state.bid_ask_spread / 2

        # Volatility-based slippage
        regime = self.regime_detector.detect_regime()
        volatility_slippage = regime.get("volatile", 0.5) * 0.5

        # Total slippage
        direction = np.sign(trade_size)
        total_slippage = (market_impact + spread_cost + volatility_slippage) * self.tick_size

        execution_price = base_price + (direction * total_slippage)

        # Ensure price is within OHLC bounds
        execution_price = np.clip(execution_price, state.low_price, state.high_price)

        return round_to_nearest_increment(execution_price, self.tick_size)

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

        # Time of day effects (NQ is most liquid during US hours)
        hour = state.ts.hour if hasattr(state.ts, 'hour') else 14
        if 9 <= hour <= 16:  # US market hours
            self.time_of_day_factor = 1.0
        elif 17 <= hour <= 23 or 0 <= hour <= 8:  # After hours
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

        # Market regime adjustments
        regime = self.regime_detector.detect_regime()
        
        # Penalty for excessive position in volatile markets
        position_penalty = 0.0
        if regime.get("volatile", 0.5) > 0.7:
            position_ratio = abs(self.current_position) / self.max_position_size
            position_penalty = -0.001 * position_ratio * regime["volatile"]

        # Reward for trading with liquidity
        liquidity_bonus = 0.0001 * regime.get("liquid", 0.5) if action != 0 else 0.0

        # Drawdown penalty
        drawdown_penalty = 0.0
        current_drawdown = (self.max_equity - total_equity) / self.max_equity
        if current_drawdown > 0.05:  # More than 5% drawdown
            drawdown_penalty = -0.005 * current_drawdown

        total_reward = base_reward + equity_reward + liquidity_bonus + position_penalty + drawdown_penalty
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

        # Maximum drawdown penalty
        max_dd = max(self.drawdown_series) if self.drawdown_series else 0.0
        drawdown_penalty = -max_dd * 0.05

        return total_return * 0.1 + sharpe_bonus + drawdown_penalty

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

    def _get_observation(self):
        """Get comprehensive observation vector"""
        if self.current_index >= len(self.states):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        current_state = self.states[self.current_index]

        # Base features (technical indicators, OHLCV, etc.)
        base_features = current_state.features.copy()

        # Position information
        total_equity = self.account_balance + self.unrealized_pnl
        position_info = np.array([
            self.current_position / self.max_position_size,  # Normalized position
            self.unrealized_pnl / self.initial_capital,      # Normalized unrealized P&L
            (self.account_balance - self.initial_capital) / self.initial_capital,  # Account performance
            self.margin_used / self.initial_capital,         # Margin utilization
            len(self.trades) / 100.0,                       # Trade count (normalized)
            (total_equity - self.initial_capital) / self.initial_capital,  # Total performance
            self.position_value / self.initial_capital,      # Position value
            1.0 if self.current_position > 0 else (-1.0 if self.current_position < 0 else 0.0)  # Position direction
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
            current_state.volume / 10000.0,  # Normalized volume
            (current_state.close_price - current_state.open_price) / current_state.open_price  # Intrabar return
        ], dtype=np.float32)

        # Combine all features
        observation = np.concatenate([base_features, position_info, market_info])

        # Ensure finite values and reasonable bounds
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
        """Render environment state and generate performance reports"""
        if mode == 'human':
            self._generate_episode_reports()
            metrics = self.generate_episode_metrics()
            return self.total_reward, metrics
        return None

    def generate_episode_metrics(self) -> Dict:
        """Generate comprehensive episode performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'final_equity': self.equity_curve[-1] if self.equity_curve else self.initial_capital
            }

        # Trade analysis
        trade_pnls = [t['pnl'] for t in self.trades]
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl <= 0]

        # Basic metrics
        total_pnl = sum(trade_pnls)
        win_rate = len(winning_trades) / len(trade_pnls) if trade_pnls else 0.0
        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean(losing_trades) if losing_trades else 0.0
        profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf')

        # Performance metrics
        final_equity = self.equity_curve[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        max_drawdown = max(self.drawdown_series) if self.drawdown_series else 0.0

        # Sharpe ratio
        if len(self.daily_pnl) > 1:
            sharpe = np.mean(self.daily_pnl) / (np.std(self.daily_pnl) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Trade duration analysis
        durations = [t['duration'] for t in self.trades]
        avg_duration = np.mean(durations) if durations else 0.0

        metrics = {
            'total_trades': len(self.trades),
            'total_pnl': total_pnl,
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_win': max(trade_pnls) if trade_pnls else 0.0,
            'max_loss': min(trade_pnls) if trade_pnls else 0.0,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'avg_trade_duration': avg_duration,
            'final_equity': final_equity,
            'long_trades': len([t for t in self.trades if t['trade_type'] == 'long']),
            'short_trades': len([t for t in self.trades if t['trade_type'] == 'short']),
            'pnl_monotonicity': monotonicity([t['pnl'] for t in self.trades]) if self.trades else 0.0,
            'equity_curve': self.equity_curve.copy(),
            'drawdown_series': self.drawdown_series.copy()
        }

        # Save metrics to file
        metrics_dir = Path(self.log_dir) / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        
        with open(metrics_dir / f"ep{self.episode}_metrics.json", "w") as f:
            # Convert numpy types for JSON serialization
            json_metrics = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                          for k, v in metrics.items() if k not in ['equity_curve', 'drawdown_series']}
            json.dump(json_metrics, f, indent=2)

        return metrics

    def _generate_episode_reports(self):
        """Generate visual reports and analysis"""
        if not self.trades:
            return

        # Create output directories
        img_dir = Path(self.log_dir) / "img"
        img_dir.mkdir(exist_ok=True)

        try:
            # Trade duration distribution
            durations = [t['duration'] for t in self.trades]
            plt.figure(figsize=(10, 6))
            plt.hist(durations, bins=20, alpha=0.7, edgecolor='black')
            plt.title('Trade Duration Distribution')
            plt.xlabel('Duration (seconds)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(img_dir / f"duration_ep{self.episode}.png", dpi=150, bbox_inches='tight')
            plt.close()

            # P&L distribution
            pnls = [t['pnl'] for t in self.trades]
            long_pnls = [t['pnl'] for t in self.trades if t['trade_type'] == 'long']
            short_pnls = [t['pnl'] for t in self.trades if t['trade_type'] == 'short']

            plt.figure(figsize=(12, 6))
            bins = np.linspace(min(pnls), max(pnls), 20)
            
            if long_pnls:
                plt.hist(long_pnls, bins=bins, alpha=0.6, label='Long Trades', color='green')
            if short_pnls:
                plt.hist(short_pnls, bins=bins, alpha=0.6, label='Short Trades', color='red')
                
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
            plt.title('P&L Distribution by Trade Type')
            plt.xlabel('P&L ($)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(img_dir / f"pnl_distribution_ep{self.episode}.png", dpi=150, bbox_inches='tight')
            plt.close()

            # Equity curve
            plt.figure(figsize=(14, 8))
            time_steps = range(len(self.equity_curve))
            plt.plot(time_steps, self.equity_curve, linewidth=2, color='blue', label='Equity Curve')
            plt.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
            plt.title('Equity Curve Evolution')
            plt.xlabel('Time Steps')
            plt.ylabel('Equity ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(img_dir / f"equity_curve_ep{self.episode}.png", dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.warning(f"Failed to generate episode reports: {e}")

    def get_performance_summary(self) -> Dict:
        """Get concise performance summary"""
        return self.generate_episode_metrics()

    def close(self):
        """Clean up environment resources"""
        try:
            cleanup_old_logs(self.log_dir, max_episodes=5)
        except Exception as e:
            logger.warning(f"Failed to cleanup logs: {e}")
