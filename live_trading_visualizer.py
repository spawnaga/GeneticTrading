
#!/usr/bin/env python
"""
Live Trading Visualization Dashboard
====================================

Real-time visualization of trading performance, market conditions,
and learning progress for the hybrid GA-PPO system.
"""

import os
import time
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class LiveTradingVisualizer:
    """Real-time visualization dashboard for trading system."""
    
    def __init__(self, update_interval: int = 5, max_history: int = 1000):
        self.update_interval = update_interval
        self.max_history = max_history
        
        # Data storage
        self.performance_data = {
            'timestamps': [],
            'equity_curve': [],
            'daily_pnl': [],
            'positions': [],
            'trades': [],
            'market_regimes': [],
            'algorithm_used': [],
            'sharpe_ratio': [],
            'drawdown': [],
            'win_rate': []
        }
        
        # Real-time metrics
        self.current_metrics = {
            'total_return': 0.0,
            'current_position': 0,
            'unrealized_pnl': 0.0,
            'account_value': 100000.0,
            'trades_today': 0,
            'algorithm': 'GA',
            'market_regime': 'normal',
            'confidence': 0.5
        }
        
        # Setup visualization
        self.fig = None
        self.axes = {}
        self.animation = None
        self.is_running = False
        
        # Colors and styling
        plt.style.use('dark_background')
        self.colors = {
            'profit': '#00ff88',
            'loss': '#ff4444',
            'neutral': '#888888',
            'ga': '#ff6b35',
            'ppo': '#4ecdc4',
            'hybrid': '#45b7d1'
        }
        
    def start_visualization(self, port: int = 5000):
        """Start the live visualization dashboard."""
        self.is_running = True
        
        # Create the main dashboard
        self._setup_dashboard()
        
        # Start data update thread
        update_thread = threading.Thread(target=self._data_update_loop)
        update_thread.daemon = True
        update_thread.start()
        
        # Start the animation
        self.animation = FuncAnimation(
            self.fig, self._update_plots, interval=self.update_interval * 1000,
            blit=False, cache_frame_data=False
        )
        
        logger.info(f"🚀 Live dashboard started - updating every {self.update_interval}s")
        plt.show()
        
    def stop_visualization(self):
        """Stop the visualization."""
        self.is_running = False
        if self.animation:
            self.animation.event_source.stop()
            
    def _setup_dashboard(self):
        """Setup the main dashboard layout."""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle('🚀 Live Trading Performance Dashboard', 
                         fontsize=16, fontweight='bold', color='white')
        
        # Define grid layout
        gs = self.fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Main equity curve (top row, spans 3 columns)
        self.axes['equity'] = self.fig.add_subplot(gs[0, :3])
        
        # Current metrics (top right)
        self.axes['metrics'] = self.fig.add_subplot(gs[0, 3])
        
        # Daily P&L (second row left)
        self.axes['daily_pnl'] = self.fig.add_subplot(gs[1, :2])
        
        # Position tracking (second row right)
        self.axes['positions'] = self.fig.add_subplot(gs[1, 2:])
        
        # Algorithm performance (third row left)
        self.axes['algorithm'] = self.fig.add_subplot(gs[2, :2])
        
        # Market regime (third row right)
        self.axes['regime'] = self.fig.add_subplot(gs[2, 2:])
        
        # Risk metrics (bottom row left)
        self.axes['risk'] = self.fig.add_subplot(gs[3, :2])
        
        # Trade distribution (bottom row right)
        self.axes['trades'] = self.fig.add_subplot(gs[3, 2:])
        
        # Style all axes
        for ax in self.axes.values():
            ax.set_facecolor('#0d1117')
            ax.grid(True, alpha=0.2, color='white')
            ax.tick_params(colors='white')
            
    def _update_plots(self, frame):
        """Update all plots with latest data."""
        try:
            self._update_equity_curve()
            self._update_current_metrics()
            self._update_daily_pnl()
            self._update_positions()
            self._update_algorithm_performance()
            self._update_market_regime()
            self._update_risk_metrics()
            self._update_trade_distribution()
            
        except Exception as e:
            logger.error(f"Error updating plots: {e}")
            
        return []  # Required for blit=False
        
    def _update_equity_curve(self):
        """Update the main equity curve plot."""
        ax = self.axes['equity']
        ax.clear()
        
        if not self.performance_data['equity_curve']:
            ax.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center',
                   transform=ax.transAxes, color='white', fontsize=12)
            return
            
        equity = self.performance_data['equity_curve']
        timestamps = self.performance_data['timestamps']
        
        # Main equity line
        ax.plot(timestamps, equity, color=self.colors['profit'], linewidth=2.5, alpha=0.9)
        
        # Fill area
        initial_value = equity[0] if equity else 100000
        ax.fill_between(timestamps, equity, initial_value, 
                       where=np.array(equity) >= initial_value,
                       color=self.colors['profit'], alpha=0.2)
        ax.fill_between(timestamps, equity, initial_value, 
                       where=np.array(equity) < initial_value,
                       color=self.colors['loss'], alpha=0.2)
        
        # Highlight recent performance
        if len(equity) > 10:
            recent_x = timestamps[-10:]
            recent_y = equity[-10:]
            ax.plot(recent_x, recent_y, color='yellow', linewidth=3, alpha=0.8)
            
        ax.set_title('💰 Account Equity Curve', color='white', fontweight='bold')
        ax.set_xlabel('Time', color='white')
        ax.set_ylabel('Account Value ($)', color='white')
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Add current value annotation
        if equity:
            current_value = equity[-1]
            initial_value = equity[0]
            pnl_pct = ((current_value - initial_value) / initial_value) * 100
            ax.annotate(f'${current_value:,.0f} ({pnl_pct:+.1f}%)',
                       xy=(timestamps[-1], current_value),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       fontsize=10, fontweight='bold', color='black')
                       
    def _update_current_metrics(self):
        """Update current metrics display."""
        ax = self.axes['metrics']
        ax.clear()
        ax.set_facecolor('#0d1117')
        
        metrics = self.current_metrics
        
        # Create metrics text
        metrics_text = f"""
📊 CURRENT STATUS
{'='*20}

💵 Account Value: ${metrics['account_value']:,.0f}
📈 Total Return: {metrics['total_return']:+.2f}%
🎯 Position: {metrics['current_position']}
💰 Unrealized P&L: ${metrics['unrealized_pnl']:+,.0f}

🔄 Algorithm: {metrics['algorithm']}
🌊 Market: {metrics['market_regime']}
🎲 Confidence: {metrics['confidence']:.1%}

📈 Trades Today: {metrics['trades_today']}
⏰ Last Update: {datetime.now().strftime('%H:%M:%S')}
"""
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', color='white',
               fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
    def _update_daily_pnl(self):
        """Update daily P&L chart."""
        ax = self.axes['daily_pnl']
        ax.clear()
        
        if not self.performance_data['daily_pnl']:
            return
            
        daily_pnl = self.performance_data['daily_pnl'][-30:]  # Last 30 days
        colors = [self.colors['profit'] if pnl >= 0 else self.colors['loss'] for pnl in daily_pnl]
        
        bars = ax.bar(range(len(daily_pnl)), daily_pnl, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        ax.axhline(y=0, color='white', linestyle='-', alpha=0.8, linewidth=1)
        
        # Add value labels on significant bars
        for i, (bar, value) in enumerate(zip(bars, daily_pnl)):
            if abs(value) > max(abs(np.array(daily_pnl))) * 0.3:
                ax.text(bar.get_x() + bar.get_width()/2., value, f'${value:.0f}',
                       ha='center', va='bottom' if value >= 0 else 'top', 
                       fontsize=8, color='white')
                       
        ax.set_title('📅 Daily P&L (Last 30 Days)', color='white', fontweight='bold')
        ax.set_xlabel('Days Ago', color='white')
        ax.set_ylabel('Daily P&L ($)', color='white')
        
    def _update_positions(self):
        """Update position tracking."""
        ax = self.axes['positions']
        ax.clear()
        
        if not self.performance_data['positions']:
            return
            
        positions = self.performance_data['positions'][-100:]  # Last 100 observations
        timestamps = self.performance_data['timestamps'][-100:]
        
        # Create position step plot
        ax.step(timestamps, positions, where='post', linewidth=2, color=self.colors['hybrid'])
        ax.fill_between(timestamps, positions, 0, step='post', alpha=0.3, color=self.colors['hybrid'])
        
        # Highlight position changes
        for i in range(1, len(positions)):
            if positions[i] != positions[i-1]:
                ax.axvline(x=timestamps[i], color='yellow', alpha=0.6, linestyle='--')
                
        ax.set_title('📊 Position Tracking', color='white', fontweight='bold')
        ax.set_xlabel('Time', color='white')
        ax.set_ylabel('Position Size', color='white')
        ax.set_ylim(-10, 10)  # Assuming max position size of 10
        
    def _update_algorithm_performance(self):
        """Update algorithm performance comparison."""
        ax = self.axes['algorithm']
        ax.clear()
        
        algorithms = self.performance_data['algorithm_used']
        if not algorithms:
            return
            
        # Count algorithm usage
        unique_algs, counts = np.unique(algorithms, return_counts=True)
        
        # Create pie chart
        colors_list = [self.colors.get(alg, '#888888') for alg in unique_algs]
        wedges, texts, autotexts = ax.pie(counts, labels=unique_algs, autopct='%1.1f%%',
                                         colors=colors_list, textprops={'color': 'white'})
        
        ax.set_title('🤖 Algorithm Usage', color='white', fontweight='bold')
        
    def _update_market_regime(self):
        """Update market regime visualization."""
        ax = self.axes['regime']
        ax.clear()
        
        regimes = self.performance_data['market_regimes']
        if not regimes:
            return
            
        # Count regime occurrences
        unique_regimes, counts = np.unique(regimes, return_counts=True)
        
        # Create horizontal bar chart
        bars = ax.barh(unique_regimes, counts, color=self.colors['neutral'], alpha=0.7)
        
        # Color code by regime type
        regime_colors = {
            'trending_stable': self.colors['profit'],
            'high_volatility': self.colors['loss'],
            'normal': self.colors['neutral'],
            'sideways': '#ffaa00'
        }
        
        for bar, regime in zip(bars, unique_regimes):
            bar.set_color(regime_colors.get(regime, self.colors['neutral']))
            
        ax.set_title('🌊 Market Regime Distribution', color='white', fontweight='bold')
        ax.set_xlabel('Frequency', color='white')
        
    def _update_risk_metrics(self):
        """Update risk metrics display."""
        ax = self.axes['risk']
        ax.clear()
        
        if len(self.performance_data['sharpe_ratio']) < 2:
            return
            
        # Plot Sharpe ratio over time
        sharpe = self.performance_data['sharpe_ratio']
        drawdown = self.performance_data['drawdown']
        
        ax2 = ax.twinx()
        
        # Sharpe ratio
        line1 = ax.plot(sharpe, color=self.colors['profit'], linewidth=2, label='Sharpe Ratio')
        ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Good (1.0)')
        ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='Excellent (2.0)')
        
        # Drawdown
        line2 = ax2.plot(drawdown, color=self.colors['loss'], linewidth=2, label='Drawdown', alpha=0.7)
        ax2.fill_between(range(len(drawdown)), drawdown, 0, color=self.colors['loss'], alpha=0.2)
        
        ax.set_title('⚠️ Risk Metrics', color='white', fontweight='bold')
        ax.set_xlabel('Time', color='white')
        ax.set_ylabel('Sharpe Ratio', color='white')
        ax2.set_ylabel('Drawdown (%)', color='white')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
    def _update_trade_distribution(self):
        """Update trade outcome distribution."""
        ax = self.axes['trades']
        ax.clear()
        
        trades = self.performance_data['trades']
        if not trades:
            return
            
        # Extract trade P&L
        trade_pnl = [trade.get('pnl', 0) for trade in trades[-50:]]  # Last 50 trades
        
        # Create histogram
        bins = np.linspace(min(trade_pnl), max(trade_pnl), 20)
        n, bins, patches = ax.hist(trade_pnl, bins=bins, alpha=0.7, edgecolor='white')
        
        # Color bars based on profit/loss
        for patch, bin_center in zip(patches, (bins[:-1] + bins[1:]) / 2):
            if bin_center >= 0:
                patch.set_facecolor(self.colors['profit'])
            else:
                patch.set_facecolor(self.colors['loss'])
                
        ax.axvline(x=0, color='white', linestyle='-', alpha=0.8, linewidth=2)
        ax.set_title('📈 Trade P&L Distribution', color='white', fontweight='bold')
        ax.set_xlabel('Trade P&L ($)', color='white')
        ax.set_ylabel('Frequency', color='white')
        
    def _data_update_loop(self):
        """Background thread to update data from log files."""
        while self.is_running:
            try:
                self._load_latest_data()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in data update loop: {e}")
                time.sleep(self.update_interval)
                
    def _load_latest_data(self):
        """Load latest data from log files or database."""
        # This would connect to your actual data source
        # For now, simulate with random data
        
        current_time = datetime.now()
        
        # Simulate data updates
        if not self.performance_data['timestamps']:
            # Initialize
            self.performance_data['timestamps'] = [current_time]
            self.performance_data['equity_curve'] = [100000.0]
            self.performance_data['daily_pnl'] = [0.0]
            self.performance_data['positions'] = [0]
            self.performance_data['algorithm_used'] = ['GA']
            self.performance_data['market_regimes'] = ['normal']
        else:
            # Add new data point
            last_equity = self.performance_data['equity_curve'][-1]
            change = np.random.normal(0, 1000)  # Random change
            new_equity = max(10000, last_equity + change)  # Prevent going too low
            
            self.performance_data['timestamps'].append(current_time)
            self.performance_data['equity_curve'].append(new_equity)
            self.performance_data['daily_pnl'].append(change)
            self.performance_data['positions'].append(np.random.randint(-3, 4))
            
            # Randomly switch algorithms
            if np.random.random() < 0.1:  # 10% chance to switch
                alg = np.random.choice(['GA', 'PPO', 'Hybrid'])
                self.performance_data['algorithm_used'].append(alg)
            else:
                self.performance_data['algorithm_used'].append(
                    self.performance_data['algorithm_used'][-1]
                )
                
            # Random market regime
            regime = np.random.choice(['normal', 'trending_stable', 'high_volatility', 'sideways'])
            self.performance_data['market_regimes'].append(regime)
            
        # Update current metrics
        if self.performance_data['equity_curve']:
            current_equity = self.performance_data['equity_curve'][-1]
            initial_equity = self.performance_data['equity_curve'][0]
            
            self.current_metrics.update({
                'account_value': current_equity,
                'total_return': ((current_equity - initial_equity) / initial_equity) * 100,
                'current_position': self.performance_data['positions'][-1] if self.performance_data['positions'] else 0,
                'unrealized_pnl': np.random.normal(0, 500),  # Simulate unrealized P&L
                'algorithm': self.performance_data['algorithm_used'][-1] if self.performance_data['algorithm_used'] else 'GA',
                'market_regime': self.performance_data['market_regimes'][-1] if self.performance_data['market_regimes'] else 'normal',
                'trades_today': len([t for t in self.performance_data['timestamps'] 
                                   if t.date() == current_time.date()]) if self.performance_data['timestamps'] else 0
            })
            
        # Calculate rolling metrics
        self._calculate_rolling_metrics()
        
        # Trim data to max_history
        for key in self.performance_data:
            if len(self.performance_data[key]) > self.max_history:
                self.performance_data[key] = self.performance_data[key][-self.max_history:]
                
    def _calculate_rolling_metrics(self):
        """Calculate rolling Sharpe ratio and drawdown."""
        equity = self.performance_data['equity_curve']
        
        if len(equity) < 30:
            self.performance_data['sharpe_ratio'] = [0.0] * len(equity)
            self.performance_data['drawdown'] = [0.0] * len(equity)
            self.performance_data['win_rate'] = [0.5] * len(equity)
            return
            
        # Calculate rolling Sharpe ratio
        returns = np.diff(equity) / np.array(equity[:-1])
        sharpe_ratios = []
        drawdowns = []
        
        for i in range(30, len(returns) + 1):
            window_returns = returns[i-30:i]
            mean_return = np.mean(window_returns)
            std_return = np.std(window_returns)
            if std_return > 0:
                sharpe = (mean_return / std_return) * np.sqrt(252)
            else:
                sharpe = 0.0
            sharpe_ratios.append(sharpe)
            
            # Calculate drawdown
            window_equity = equity[i-30:i+1]
            peak = np.maximum.accumulate(window_equity)
            drawdown = ((peak - window_equity) / peak * 100)[-1]
            drawdowns.append(drawdown)
            
        # Pad with zeros for early periods
        self.performance_data['sharpe_ratio'] = [0.0] * 29 + sharpe_ratios
        self.performance_data['drawdown'] = [0.0] * 29 + drawdowns
        
        # Calculate win rate from trades
        trades = self.performance_data['trades']
        if trades:
            recent_trades = trades[-20:]  # Last 20 trades
            wins = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
            win_rate = wins / len(recent_trades)
        else:
            win_rate = 0.5
            
        # Update win rate history
        if not hasattr(self, '_win_rate_history'):
            self._win_rate_history = []
        self._win_rate_history.append(win_rate)
        self.performance_data['win_rate'] = self._win_rate_history[-len(equity):]
        
    def export_data(self, filepath: str):
        """Export current data to file."""
        export_data = {
            'performance_data': self.performance_data,
            'current_metrics': self.current_metrics,
            'export_time': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
            
        logger.info(f"Data exported to {filepath}")


def main():
    """Run the live trading visualizer."""
    visualizer = LiveTradingVisualizer(update_interval=2)
    
    try:
        visualizer.start_visualization()
    except KeyboardInterrupt:
        logger.info("Stopping visualization...")
        visualizer.stop_visualization()


if __name__ == "__main__":
    main()
