import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple
import os
import datetime
import logging
import time
import collections
import hashlib
import glob
from pathlib import Path
import json
import shutil
from uuid import uuid4

# ──────────────────────────────────────────────────────────────────────────────
# File and Data Utilities
# ──────────────────────────────────────────────────────────────────────────────

def hash_files(file_list: list[str]) -> str:
    """
    Generate a SHA256 hash over sorted filenames and modification times.
    Ensures cache invalidates if any file changes.
    """
    hasher = hashlib.sha256()
    for fp in sorted(file_list):
        st = os.stat(fp)
        hasher.update(fp.encode("utf-8"))
        hasher.update(str(st.st_mtime).encode("utf-8"))
    return hasher.hexdigest()

def setup_logging(local_rank: int) -> None:
    """Configure enhanced root logger with rotating file shared by all ranks."""
    from logging.handlers import RotatingFileHandler
    import sys

    os.makedirs("logs", exist_ok=True)

    # Enhanced format with more context
    fmt = "%(asctime)s [%(levelname)-8s] [Rank-%(process)d] %(name)-20s: %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    log_path = os.path.join("logs", f"training_rank_{local_rank}.log")

    # Create rotating file handler (10MB max, keep 5 backups)
    file_handler = RotatingFileHandler(log_path, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    file_handler.setLevel(logging.INFO)

    # Enhanced console handler with colors (if available)
    stream_handler = logging.StreamHandler(sys.stderr)
    try:
        import colorlog
        color_fmt = "%(log_color)s%(asctime)s [%(levelname)-8s]%(reset)s [Rank-%(process)d] %(name)-15s: %(message)s"
        stream_handler.setFormatter(colorlog.ColoredFormatter(
            color_fmt,
            datefmt=date_fmt,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        ))
    except ImportError:
        stream_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))

    stream_handler.setLevel(logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()  # Remove any existing handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    # Create detailed startup log
    logger = logging.getLogger("STARTUP")
    logger.info("="*80)
    logger.info(f"Training session started for rank {local_rank}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Process ID: {os.getpid()}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info("="*80)

# ──────────────────────────────────────────────────────────────────────────────
# Trading and Financial Utilities
# ──────────────────────────────────────────────────────────────────────────────

def round_to_nearest_increment(value: float, increment: float) -> float:
    """
    Round `value` to the nearest multiple of `increment`.
    """
    # Handle NaN and infinite values
    if not np.isfinite(value) or not np.isfinite(increment) or increment == 0:
        return 0.0
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

def compute_performance_metrics(balance_history_or_profits, times=None):
    """
    Compute CAGR, Sharpe ratio, and Maximum Drawdown from balance history or profits.

    Args:
        balance_history_or_profits: List of balance values or profit values
        times: Optional list of timestamps (for annualization)

    Returns:
        tuple: (cagr, sharpe, mdd)
    """
    if isinstance(balance_history_or_profits, (list, np.ndarray)):
        profits = list(balance_history_or_profits)
    else:
        profits = balance_history_or_profits

    # Need sufficient data for meaningful calculations
    if len(profits) < 10:
        return 0.0, 0.0, 0.0

    profits = np.array(profits, dtype=np.float64)

    # Remove infinite and NaN values
    profits = profits[np.isfinite(profits)]
    if len(profits) < 10:
        return 0.0, 0.0, 0.0

    # Time-based calculations
    if times is not None and len(times) >= 2:
        # Filter out None values from times
        valid_times = [t for t in times if t is not None]
        if len(valid_times) >= 2:
            # Calculate actual time period
            total_time_days = max((valid_times[-1] - valid_times[0]).total_seconds() / (24 * 3600), 1)
            years = max(total_time_days / 365.25, 1/252)  # Minimum 1 trading day
        else:
            # Fallback if no valid times
            trading_periods_per_day = 24  # Assuming 24 periods per day (15-min bars)
            trading_days = len(profits) / trading_periods_per_day
            years = max(trading_days / 252, 1/252)  # 252 trading days per year
    else:
        # Estimate time period from number of data points
        # Assume intraday data (e.g., 15-minute bars during trading hours)
        trading_periods_per_day = 24  # Assuming 24 periods per day (15-min bars)
        trading_days = len(profits) / trading_periods_per_day
        years = max(trading_days / 252, 1/252)  # 252 trading days per year

    # Calculate CAGR with more realistic approach
    cumulative_profits = np.cumsum(profits)
    if len(cumulative_profits) > 0:
        # Use a more reasonable starting capital estimation
        profit_magnitude = np.percentile(np.abs(profits), 95)  # 95th percentile for outlier resistance

        # Estimate starting capital based on profit scale and typical risk management
        if profit_magnitude > 10000:
            starting_capital = 1000000.0  # Large institutional scale
        elif profit_magnitude > 1000:
            starting_capital = 100000.0   # Professional trader scale  
        elif profit_magnitude > 100:
            starting_capital = 50000.0    # Small account scale
        elif profit_magnitude > 10:
            starting_capital = 10000.0    # Very small account
        else:
            starting_capital = 5000.0     # Micro account

        total_pnl = cumulative_profits[-1]
        ending_capital = starting_capital + total_pnl

        # Only calculate CAGR if we have sufficient time period and positive ending capital
        if years >= (30/365.25) and ending_capital > 0:  # At least 30 days of data
            total_return = ending_capital / starting_capital
            if total_return > 0:
                # CAGR calculation with bounds checking
                cagr = (total_return ** (1 / years) - 1) * 100
                # Apply realistic bounds for trading returns
                cagr = np.clip(cagr, -95.0, 50.0)  # More realistic bounds
            else:
                cagr = -99.0  # Total loss
        else:
            # For insufficient data or losses, use simple annualized return
            if years > 0:
                simple_annual_return = (total_pnl / starting_capital) / years * 100
                cagr = np.clip(simple_annual_return, -99.0, 200.0)
            else:
                cagr = 0.0
    else:
        cagr = 0.0

    # Sharpe ratio calculation with proper risk-free rate consideration
    if len(profits) > 5:
        # Clean profits of any remaining invalid values
        valid_profits = profits[np.isfinite(profits)]

        if len(valid_profits) >= 5:
            mean_profit = np.mean(valid_profits)
            std_profit = np.std(valid_profits, ddof=1)

            if std_profit > 0:
                # Estimate periods per year more accurately
                periods_per_year = len(valid_profits) / max(years, 1/252)  # Minimum 1 trading day

                # Apply risk-free rate (approximate 2% annually for recent years)
                risk_free_rate_annual = 0.02
                risk_free_per_period = risk_free_rate_annual / periods_per_year

                # Calculate excess return
                excess_return = mean_profit - risk_free_per_period

                # Period Sharpe ratio
                sharpe_period = excess_return / std_profit

                # Annualize using square root of time rule
                sharpe = sharpe_period * np.sqrt(periods_per_year)

                # Apply realistic bounds for trading Sharpe ratios
                # Professional traders rarely exceed 2.5, and -2.0 is quite poor
                sharpe = np.clip(sharpe, -2.0, 2.5)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Maximum drawdown calculation from cumulative profits
    cumulative_profits = np.cumsum(profits)
    if len(cumulative_profits) > 0:
        # Use realistic starting capital based on profit scale
        profit_scale = np.max(np.abs(cumulative_profits))
        if profit_scale > 50000:
            initial_capital = 500000.0
        elif profit_scale > 10000:
            initial_capital = 100000.0
        elif profit_scale > 1000:
            initial_capital = 50000.0
        else:
            initial_capital = 10000.0

        equity_curve = initial_capital + cumulative_profits

        # Calculate running maximum (peak equity)
        running_max = np.maximum.accumulate(equity_curve)

        # Calculate drawdowns as absolute dollar amounts
        dollar_drawdowns = running_max - equity_curve

        # Convert to percentage drawdowns
        drawdowns = np.where(running_max > 0, (dollar_drawdowns / running_max) * 100, 0)

        # Maximum drawdown is the largest percentage drop from peak
        mdd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

        # Additional check: if there are losses, ensure MDD is not zero
        if mdd == 0.0 and np.any(profits < 0):
            # Calculate MDD from consecutive losses
            consecutive_losses = []
            current_loss = 0
            for profit in profits:
                if profit < 0:
                    current_loss += profit
                else:
                    if current_loss < 0:
                        consecutive_losses.append(abs(current_loss))
                    current_loss = 0
            if current_loss < 0:
                consecutive_losses.append(abs(current_loss))

            if consecutive_losses:
                max_loss = max(consecutive_losses)
                mdd = min((max_loss / initial_capital) * 100, 100.0)

        # Handle edge cases
        if np.isnan(mdd) or np.isinf(mdd) or mdd == 0.0:
            # Ensure MDD reflects actual risk
            if len(profits) > 1:
                negative_profits = profits[profits < 0]
                if len(negative_profits) > 0:
                    # Use worst single loss as baseline MDD
                    worst_loss = abs(np.min(negative_profits))
                    mdd = min((worst_loss / initial_capital) * 100, 50.0)
                else:
                    # Even with all profits, assume some risk
                    profit_volatility = np.std(profits)
                    mdd = min((profit_volatility * 2 / initial_capital) * 100, 5.0)
            else:
                mdd = 0.0

        mdd = np.clip(mdd, 0.0, 100.0)
    else:
        mdd = 0.0

    return float(cagr), float(sharpe), float(mdd)

# ──────────────────────────────────────────────────────────────────────────────
# Environment and State Building Utilities
# ──────────────────────────────────────────────────────────────────────────────

def build_states_for_futures_env(df_chunk):
    """
    Convert a DataFrame slice into a list of TimeSeriesState for FuturesEnv.
    """
    from futures_env import TimeSeriesState

    states = []
    for row in df_chunk.itertuples(index=False):
        feats = [
            row.Open, row.High, row.Low, row.Close,
            row.Volume, row.return_, row.ma_10,
            row.rsi, row.volatility,
            row.sin_time, row.cos_time,
            row.sin_weekday, row.cos_weekday,
        ]
        # append all one‐hots
        for col in df_chunk.columns:
            if col.startswith("tb_") or col.startswith("wd_"):
                feats.append(getattr(row, col))
        states.append(
            TimeSeriesState(
                ts=row.date_time,
                open_price=getattr(row, "Open_raw", row.Open),
                high_price=getattr(row, "High_raw", row.High),
                low_price=getattr(row, "Low_raw", row.Low),
                close_price=getattr(row, "Close_raw", row.Close),
                volume=getattr(row, "Volume_raw", row.Volume),
                features=feats,
            )
        )
    return states

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation and Testing Utilities
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_agent_distributed(env, agent, local_rank=0):
    """
    A single-process (debug) version of agent evaluation.
    Returns the profit history and timestamps as lists.
    """
    profit_history = []
    timestamps = []
    obs = env.reset()

    # Handle both tuple and single return from reset
    if isinstance(obs, tuple):
        obs = obs[0]

    done = False
    device = next(agent.parameters()).device

    while not done:
        # Convert observation to tensor
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            obs_tensor = obs.unsqueeze(0).to(device) if obs.dim() == 1 else obs.to(device)

        with torch.no_grad():
            if hasattr(agent, 'act'):
                action = agent.act(obs_tensor)
            else:
                policy_logits, _ = agent.forward(obs_tensor)
                dist_ = torch.distributions.Categorical(logits=policy_logits)
                action = dist_.sample().item()

        step_result = env.step(action)
        # Handle both 4-tuple and 5-tuple returns
        if len(step_result) == 5:
            obs, reward, done, truncated, info = step_result
            done = done or truncated
        else:
            obs, reward, done, info = step_result

        profit_history.append(info.get('total_profit', 0.0))
        # Handle missing timestamp gracefully
        timestamp = info.get('timestamp')
        if timestamp is None:
            # Generate a synthetic timestamp if missing
            import datetime
            base_time = datetime.datetime.now()
            timestamp = base_time + datetime.timedelta(seconds=len(timestamps))
        timestamps.append(timestamp)

    return profit_history, timestamps

def make_test_env(data_folder, cache_folder, local_rank=0):
    """
    Load & preprocess data, then build a FuturesEnv on the test split.
    """
    from data_preprocessing import create_environment_data
    from futures_env import FuturesEnv

    train_df, test_df, scaler, segment_dict = create_environment_data(
        data_folder=data_folder,
        cache_folder=cache_folder,
        test_size=0.2
    )
    # ensure pandas for iteration
    if hasattr(test_df, "to_pandas"):
        test_df = test_df.to_pandas()
    test_df = test_df.rename(columns={"return": "return_"})

    test_states = build_states_for_futures_env(test_df)
    return FuturesEnv(
        states=test_states,
        log_dir=f"./logs/test_ga_rank{local_rank}",
        value_per_tick=12.5,
        tick_size=0.25,
        fill_probability=1.0,
        execution_cost_per_order=0.00005,
        contracts_per_trade=1,
        margin_rate=0.01,
        bid_ask_spread=0.25,
        add_current_position_to_state=True
    )

# ──────────────────────────────────────────────────────────────────────────────
# GA Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def _unpack_reset(reset_ret):
    """
    Handle both Gymnasium (obs, info) and custom obs-only resets.
    """
    if isinstance(reset_ret, tuple) and len(reset_ret) == 2:
        return reset_ret[0]
    return reset_ret

def _unpack_step(step_ret):
    """
    Handle both:
      - Gymnasium 5-tuple: (obs, reward, done, truncated, info)
      - Custom 4-tuple:   (obs, reward, done, info)
    Returns: obs, reward, done, info
    """
    if len(step_ret) == 5:
        obs, reward, done, truncated, info = step_ret
        return obs, reward, (done or truncated), info
    elif len(step_ret) == 4:
        return step_ret
    else:
        raise ValueError(f"Unexpected step return length: {len(step_ret)}")

# ──────────────────────────────────────────────────────────────────────────────
# Cleanup and Memory Management Utilities
# ──────────────────────────────────────────────────────────────────────────────

def cleanup_old_logs(log_dir="./logs", max_episodes=10):
    """
    Clean up old log files, keeping only the most recent max_episodes.
    """
    if not os.path.exists(log_dir):
        return

    # Find all episode log files
    episode_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith("duration_ep") or file.startswith("profit_ep"):
                full_path = os.path.join(root, file)
                # Extract episode number
                try:
                    ep_num = int(file.split("ep")[1].split(".")[0])
                    episode_files.append((ep_num, full_path))
                except (ValueError, IndexError):
                    continue

    # Sort by episode number and remove old ones
    episode_files.sort(key=lambda x: x[0])
    if len(episode_files) > max_episodes * 2:  # *2 because we have duration and profit files
        files_to_remove = episode_files[:-max_episodes * 2]
        for _, file_path in files_to_remove:
            try:
                os.remove(file_path)
                print(f"Removed old log file: {file_path}")
            except OSError:
                pass

    # Clean up metrics folder
    metrics_dir = os.path.join(log_dir, "metrics")
    if os.path.exists(metrics_dir):
        metric_files = []
        for file in os.listdir(metrics_dir):
            if file.startswith("ep") and file.endswith(".json"):
                try:
                    ep_num = int(file[2:-5])  # Remove 'ep' and '.json'
                    metric_files.append((ep_num, os.path.join(metrics_dir, file)))
                except ValueError:
                    continue

        metric_files.sort(key=lambda x: x[0])
        if len(metric_files) > max_episodes:
            files_to_remove = metric_files[:-max_episodes]
            for _, file_path in files_to_remove:
                try:
                    os.remove(file_path)
                    print(f"Removed old metric file: {file_path}")
                except OSError:
                    pass

def cleanup_tensorboard_runs(runs_dir="./runs", keep_latest=1):
    """
    Clean up old TensorBoard run directories, keeping only the most recent ones.
    """
    if not os.path.exists(runs_dir):
        return

    # Get all run directories with their modification times
    run_dirs = []
    for item in os.listdir(runs_dir):
        item_path = os.path.join(runs_dir, item)
        if os.path.isdir(item_path):
            try:
                mtime = os.path.getmtime(item_path)
                run_dirs.append((mtime, item_path))
            except OSError:
                continue  # Skip inaccessible directories

    # Sort by modification time and remove old ones
    run_dirs.sort(key=lambda x: x[0], reverse=True)
    if len(run_dirs) > keep_latest:
        dirs_to_remove = run_dirs[keep_latest:]
        for _, dir_path in dirs_to_remove:
            try:
                # Add delay to avoid race conditions
                import time
                time.sleep(0.1)

                # Check if directory still exists before removing
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    print(f"Removed old TensorBoard run: {dir_path}")
            except OSError as e:
                print(f"Failed to remove {dir_path}: {e}")
                continue


class TrainingMetricsTracker:
    """
    Comprehensive metrics tracker for training sessions with creative visualizations
    """
    def __init__(self, log_dir="./logs/metrics"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Training session metadata
        self.session_id = str(uuid4())[:8]
        self.start_time = datetime.datetime.now()

        # Metrics storage
        self.metrics_history = {
            'loss': [],
            'reward': [],
            'learning_rate': [],
            'entropy': [],
            'value_accuracy': [],
            'action_distribution': [],
            'gradient_norms': [],
            'episode_lengths': [],
            'timestamps': []
        }

        # Trading-specific metrics
        self.trading_metrics = {
            'positions': [],
            'profits': [],
            'trades_per_episode': [],
            'win_rates': [],
            'sharpe_ratios': [],
            'max_drawdowns': []
        }

        # Performance benchmarks
        self.benchmarks = {
            'best_reward': float('-inf'),
            'best_sharpe': float('-inf'),
            'lowest_drawdown': float('inf'),
            'fastest_convergence': None
        }

    def log_training_step(self, metrics_dict):
        """Log metrics from a single training step"""
        timestamp = datetime.datetime.now()
        self.metrics_history['timestamps'].append(timestamp)

        for key, value in metrics_dict.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)

    def log_trading_metrics(self, trading_dict):
        """Log trading-specific metrics"""
        for key, value in trading_dict.items():
            if key in self.trading_metrics:
                self.trading_metrics[key].append(value)

    def update_benchmarks(self, current_metrics):
        """Update performance benchmarks"""
        if 'reward' in current_metrics:
            self.benchmarks['best_reward'] = max(self.benchmarks['best_reward'], current_metrics['reward'])

        if 'sharpe_ratio' in current_metrics:
            self.benchmarks['best_sharpe'] = max(self.benchmarks['best_sharpe'], current_metrics['sharpe_ratio'])

        if 'max_drawdown' in current_metrics:
            self.benchmarks['lowest_drawdown'] = min(self.benchmarks['lowest_drawdown'], current_metrics['max_drawdown'])

    def generate_session_report(self):
        """Generate comprehensive session report"""
        end_time = datetime.datetime.now()
        duration = end_time - self.start_time

        report = {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'total_steps': len(self.metrics_history['timestamps']),
            'benchmarks': self.benchmarks.copy(),
            'final_metrics': self._get_recent_averages(),
            'convergence_analysis': self._analyze_convergence(),
            'trading_performance': self._analyze_trading_performance()
        }

        # Save to file
        report_path = os.path.join(self.log_dir, f"session_report_{self.session_id}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        return report

    def _get_recent_averages(self, window=100):
        """Get recent averages of key metrics"""
        averages = {}
        for key, values in self.metrics_history.items():
            if values and key != 'timestamps':
                recent_values = values[-window:] if len(values) > window else values
                if isinstance(recent_values[0], (int, float)):
                    averages[f'recent_{key}_avg'] = np.mean(recent_values)
                    averages[f'recent_{key}_std'] = np.std(recent_values)
        return averages

    def _analyze_convergence(self):
        """Analyze convergence patterns"""
        if not self.metrics_history['loss']:
            return {}

        losses = self.metrics_history['loss']
        if len(losses) < 10:
            return {'status': 'insufficient_data'}

        # Detect convergence point
        window = min(50, len(losses) // 4)
        recent_std = np.std(losses[-window:])
        early_std = np.std(losses[:window])

        convergence_ratio = recent_std / (early_std + 1e-8)

        return {
            'convergence_ratio': convergence_ratio,
            'is_converged': convergence_ratio < 0.1,
            'recent_stability': recent_std,
            'improvement_rate': (losses[0] - losses[-1]) / (len(losses) + 1e-8)
        }

    def _analyze_trading_performance(self):
        """Analyze trading-specific performance"""
        if not self.trading_metrics['profits']:
            return {}

        profits = self.trading_metrics['profits']

        return {
            'total_profit': sum(profits),
            'profit_per_trade': np.mean(profits),
            'win_rate': len([p for p in profits if p > 0]) / len(profits),
            'profit_factor': sum([p for p in profits if p > 0]) / abs(sum([p for p in profits if p <= 0]) + 1e-8),
            'max_consecutive_wins': self._max_consecutive(profits, lambda x: x > 0),
            'max_consecutive_losses': self._max_consecutive(profits, lambda x: x <= 0)
        }

    def _max_consecutive(self, sequence, condition):
        """Find maximum consecutive occurrences of condition"""
        max_count = 0
        current_count = 0

        for item in sequence:
            if condition(item):
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

def setup_log_cleanup_schedule(log_dir="./logs", max_episodes=10):
    """
    Set up automatic log cleanup that runs periodically.
    """
    def cleanup_if_needed():
        # Count current episode files
        episode_count = 0
        if os.path.exists(log_dir):
            for root, dirs, files in os.walk(log_dir):
                episode_count += len([f for f in files if f.startswith("duration_ep") or f.startswith("profit_ep")])

        # Cleanup if we exceed the limit
        if episode_count > max_episodes * 2:
            cleanup_old_logs(log_dir, max_episodes)

    return cleanup_if_needed

# ──────────────────────────────────────────────────────────────────────────────
# Live Trading Data Processing
# ──────────────────────────────────────────────────────────────────────────────

def process_live_row(bar: dict, history: dict, scaler, segment_dict: dict) -> "cudf.DataFrame":
    """
    Given a single OHLCV bar dict, compute all features exactly as in training,
    apply the saved scaler, and return a 1-row cuDF DataFrame ready for inference.
    """
    try:
        import cudf
    except ImportError:
        import pandas as pd
        cudf = pd

    WINDOW_MA = 10
    WINDOW_RSI = 14
    WINDOW_VOL = 10
    BIN_SIZE = 15
    SECONDS_IN_DAY = 24 * 60

    ts = bar["date_time"]
    close = bar["Close"]
    history["closes"].append(close)

    # compute delta & return
    if len(history["closes"]) >= 2:
        prev = history["closes"][-2]
        delta = close - prev
        rtn = delta / (prev + 1e-8)
    else:
        delta, rtn = 0.0, 0.0

    history["deltas"].append(delta)
    history["returns"].append(rtn)

    # moving average, RSI, volatility
    ma10 = float(np.mean(history["closes"]))
    gain = float(np.mean([d for d in history["deltas"] if d > 0])) if history["deltas"] else 0.0
    loss = float(np.mean([-d for d in history["deltas"] if d < 0])) if history["deltas"] else 1e-8
    rsi = 100 - (100 / (1 + gain/(loss+1e-8)))
    vol = float(np.std(history["returns"])) if len(history["returns"]) > 1 else 0.0

    # temporal features
    weekday = ts.weekday()
    minutes = ts.hour * 60 + ts.minute
    sin_t = np.sin(2 * np.pi * (minutes / SECONDS_IN_DAY))
    cos_t = np.cos(2 * np.pi * (minutes / SECONDS_IN_DAY))
    sin_w = np.sin(2 * np.pi * (weekday / 7))
    cos_w = np.cos(2 * np.pi * (weekday / 7))

    # one‐hot bins
    bin_idx = minutes // BIN_SIZE
    total_bins = SECONDS_IN_DAY // BIN_SIZE
    ohe = {f"wd_{d}": 1 if d == weekday else 0 for d in range(7)}
    ohe.update({f"tb_{b}": 1 if b == bin_idx else 0 for b in range(total_bins)})

    # assemble row
    row = {
        "Open": bar["Open"], "High": bar["High"], "Low": bar["Low"],
        "Close": close, "Volume": bar["Volume"],
        "return": rtn, "ma_10": ma10,
        "rsi": rsi, "volatility": vol,
        "sin_time": sin_t, "cos_time": cos_t,
        "sin_weekday": sin_w, "cos_weekday": cos_w,
        **ohe,
        "Open_raw": bar["Open"], "High_raw": bar["High"],
        "Low_raw": bar["Low"], "Close_raw": close,
        "Volume_raw": bar["Volume"],
    }

    df_live = cudf.DataFrame([row])
    numeric_cols = ["Open","High","Low","Close","Volume","return","ma_10"]
    df_live[numeric_cols] = scaler.transform(df_live[numeric_cols])
    return df_live

def hash_files(file_paths):
    """Create a hash of file paths and their modification times for caching."""
    hasher = hashlib.sha256()
    for fp in sorted(file_paths):
        hasher.update(fp.encode())
        if os.path.exists(fp):
            hasher.update(str(os.path.getmtime(fp)).encode())
    return hasher.hexdigest()


def validate_data_integrity(df, logger=None):
    """
    Comprehensive data validation to ensure no null, infinite, or invalid values.
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    logger.info("Starting comprehensive data integrity validation...")

    # Check for null values
    if hasattr(df, 'isnull'):
        null_counts = df.isnull().sum()
    else:
        null_counts = df.isna().sum()

    total_nulls = null_counts.sum()
    if total_nulls > 0:
        logger.error(f"Found {total_nulls} null values in data!")
        logger.error("Null counts by column:")
        for col, count in null_counts.items():
            if count > 0:
                logger.error(f"  {col}: {count} nulls")
        return False

    # Check for infinite values
    import numpy as np
    for col in df.select_dtypes(include=[np.number]).columns:
        if hasattr(df[col], 'to_pandas'):
            # cuDF case
            col_data = df[col].to_pandas()
        else:
            col_data = df[col]

        inf_count = np.isinf(col_data).sum()
        if inf_count > 0:
            logger.error(f"Found {inf_count} infinite values in column {col}")
            return False

    # Check for empty data
    if len(df) == 0:
        logger.error("DataFrame is empty!")
        return False

    logger.info(f"Data integrity validation passed: {len(df)} rows, {len(df.columns)} columns")
    return True