
#!/usr/bin/env python
"""
Trading Performance Dashboard for TensorBoard
============================================

Provides comprehensive trading metrics visualization and real-time monitoring.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger("DASHBOARD")

class TradingDashboard:
    """Enhanced trading dashboard with comprehensive metrics."""
    
    def __init__(self, tb_writer=None):
        self.tb_writer = tb_writer
        self.metrics_history = {
            'account_values': [],
            'daily_profits': [],
            'daily_losses': [],
            'drawdowns': [],
            'sharpe_ratios': [],
            'cagr_values': [],
            'timestamps': [],
            'positions': [],
            'trades': []
        }
        
        # Performance tracking
        self.starting_capital = 100000.0
        self.peak_value = self.starting_capital
        self.max_drawdown = 0.0
        
    def update_metrics(self, account_value, profit_loss, position=0, timestamp=None):
        """Update all trading metrics with comprehensive logging."""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Log significant changes
        if len(self.metrics_history['account_values']) > 0:
            prev_value = self.metrics_history['account_values'][-1]
            change_pct = ((account_value - prev_value) / prev_value) * 100
            if abs(change_pct) > 1.0:  # Log changes > 1%
                logger.info(f"📈 Account value: ${account_value:,.2f} ({change_pct:+.2f}%)")
            
        self.metrics_history['timestamps'].append(timestamp)
        self.metrics_history['account_values'].append(account_value)
        self.metrics_history['positions'].append(position)
        
        # Update peak value and drawdown
        if account_value > self.peak_value:
            self.peak_value = account_value
            
        current_drawdown = ((self.peak_value - account_value) / self.peak_value) * 100
        self.metrics_history['drawdowns'].append(current_drawdown)
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Categorize profit/loss
        if profit_loss > 0:
            self.metrics_history['daily_profits'].append(profit_loss)
            self.metrics_history['daily_losses'].append(0)
        else:
            self.metrics_history['daily_profits'].append(0)
            self.metrics_history['daily_losses'].append(abs(profit_loss))
            
        # Calculate rolling metrics
        self._calculate_rolling_metrics()
        
    def _calculate_rolling_metrics(self):
        """Calculate rolling Sharpe ratio and CAGR with logging."""
        if len(self.metrics_history['account_values']) < 10:
            return
            
        logger.debug("🔄 Calculating rolling performance metrics...")
            
        # Calculate returns
        values = self.metrics_history['account_values']
        returns = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                daily_return = (values[i] - values[i-1]) / values[i-1]
                returns.append(daily_return)
                
        if len(returns) > 1:
            # Sharpe ratio (annualized)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                sharpe = (mean_return / std_return) * np.sqrt(252)
            else:
                sharpe = 0
            self.metrics_history['sharpe_ratios'].append(sharpe)
            
            # CAGR
            total_return = values[-1] / values[0]
            periods = len(values) / 252  # Convert to years
            if periods > 0 and total_return > 0:
                cagr = (total_return ** (1/periods) - 1) * 100
            else:
                cagr = 0
            self.metrics_history['cagr_values'].append(cagr)
            
    def log_to_tensorboard(self, global_step):
        """Log all metrics to TensorBoard."""
        if not self.tb_writer or len(self.metrics_history['account_values']) == 0:
            return
            
        latest_idx = -1
        
        # Core metrics
        self.tb_writer.add_scalar("Trading/AccountValue", 
                                 self.metrics_history['account_values'][latest_idx], global_step)
        
        if len(self.metrics_history['daily_profits']) > 0:
            self.tb_writer.add_scalar("Trading/DailyProfit", 
                                     self.metrics_history['daily_profits'][latest_idx], global_step)
            self.tb_writer.add_scalar("Trading/DailyLoss", 
                                     self.metrics_history['daily_losses'][latest_idx], global_step)
                                     
        # Drawdown metrics
        self.tb_writer.add_scalar("Trading/CurrentDrawdown", 
                                 self.metrics_history['drawdowns'][latest_idx], global_step)
        self.tb_writer.add_scalar("Trading/MaxDrawdown", self.max_drawdown, global_step)
        
        # Performance ratios
        if len(self.metrics_history['sharpe_ratios']) > 0:
            self.tb_writer.add_scalar("Trading/SharpeRatio", 
                                     self.metrics_history['sharpe_ratios'][latest_idx], global_step)
                                     
        if len(self.metrics_history['cagr_values']) > 0:
            self.tb_writer.add_scalar("Trading/CAGR", 
                                     self.metrics_history['cagr_values'][latest_idx], global_step)
                                     
        # Risk metrics
        if len(self.metrics_history['account_values']) >= 30:
            recent_values = self.metrics_history['account_values'][-30:]
            volatility = np.std(recent_values) / np.mean(recent_values) * 100
            self.tb_writer.add_scalar("Trading/Volatility30D", volatility, global_step)
            
        # Generate comprehensive charts
        self._create_comprehensive_charts(global_step)
        
    def _create_comprehensive_charts(self, global_step):
        """Create comprehensive trading charts."""
        
        # 1. Account Value Evolution
        if len(self.metrics_history['account_values']) > 10:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            timestamps = self.metrics_history['timestamps']
            values = self.metrics_history['account_values']
            
            # Main account value line
            ax.plot(timestamps, values, 'b-', linewidth=2.5, label='Account Value', alpha=0.8)
            
            # Starting capital line
            ax.axhline(y=self.starting_capital, color='gray', linestyle='--', 
                      alpha=0.7, label=f'Starting Capital (${self.starting_capital:,.0f})')
            
            # Peak value line
            ax.axhline(y=self.peak_value, color='green', linestyle='--', 
                      alpha=0.7, label=f'Peak Value (${self.peak_value:,.0f})')
            
            # Fill area between current and starting value
            ax.fill_between(timestamps, values, self.starting_capital, 
                           where=np.array(values) >= self.starting_capital, 
                           color='green', alpha=0.2, interpolate=True, label='Gains')
            ax.fill_between(timestamps, values, self.starting_capital, 
                           where=np.array(values) < self.starting_capital, 
                           color='red', alpha=0.2, interpolate=True, label='Losses')
            
            ax.set_title("Account Value Evolution", fontsize=16, fontweight='bold')
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Account Value ($)", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Format y-axis as currency
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            self.tb_writer.add_figure("TradingCharts/AccountValue", fig, global_step)
            plt.close(fig)
            
        # 2. Daily P&L Chart
        if len(self.metrics_history['daily_profits']) > 5:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Daily P&L Bar Chart
            daily_pnl = np.array(self.metrics_history['daily_profits']) - np.array(self.metrics_history['daily_losses'])
            colors = ['green' if pnl >= 0 else 'red' for pnl in daily_pnl]
            
            bars = ax1.bar(range(len(daily_pnl)), daily_pnl, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            ax1.set_title("Daily Profits and Losses", fontsize=14, fontweight='bold')
            ax1.set_xlabel("Trading Day")
            ax1.set_ylabel("Daily P&L ($)")
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, daily_pnl)):
                if abs(value) > max(abs(daily_pnl)) * 0.1:  # Only label significant values
                    ax1.text(bar.get_x() + bar.get_width()/2., value, f'${value:.0f}',
                            ha='center', va='bottom' if value >= 0 else 'top', fontsize=8)
            
            # Cumulative P&L
            cumulative_pnl = np.cumsum(daily_pnl)
            ax2.plot(range(len(cumulative_pnl)), cumulative_pnl, 'b-', linewidth=2.5, marker='o', markersize=3)
            ax2.fill_between(range(len(cumulative_pnl)), cumulative_pnl, 0, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8)
            ax2.set_title("Cumulative P&L", fontsize=14, fontweight='bold')
            ax2.set_xlabel("Trading Day")
            ax2.set_ylabel("Cumulative P&L ($)")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            self.tb_writer.add_figure("TradingCharts/DailyPnL", fig, global_step)
            plt.close(fig)
            
        # 3. Drawdown Analysis
        if len(self.metrics_history['drawdowns']) > 5:
            fig, ax = plt.subplots(figsize=(14, 6))
            
            drawdowns = self.metrics_history['drawdowns']
            timestamps = self.metrics_history['timestamps']
            
            # Drawdown area chart
            ax.fill_between(timestamps, drawdowns, 0, color='red', alpha=0.4, label='Drawdown')
            ax.plot(timestamps, drawdowns, 'r-', linewidth=2)
            
            # Max drawdown line
            ax.axhline(y=self.max_drawdown, color='darkred', linestyle='--', 
                      label=f'Max Drawdown ({self.max_drawdown:.2f}%)')
            
            ax.set_title("Drawdown Analysis", fontsize=16, fontweight='bold')
            ax.set_xlabel("Time")
            ax.set_ylabel("Drawdown (%)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()  # Drawdowns should go down
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            self.tb_writer.add_figure("TradingCharts/Drawdown", fig, global_step)
            plt.close(fig)
            
        # 4. Performance Metrics Dashboard
        if len(self.metrics_history['sharpe_ratios']) > 0 and len(self.metrics_history['cagr_values']) > 0:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Sharpe Ratio Evolution
            sharpe_data = self.metrics_history['sharpe_ratios']
            ax1.plot(range(len(sharpe_data)), sharpe_data, 'g-', linewidth=2.5, marker='o', markersize=4)
            ax1.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Good (1.0)')
            ax1.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='Excellent (2.0)')
            ax1.set_title("Sharpe Ratio Evolution", fontsize=14, fontweight='bold')
            ax1.set_ylabel("Sharpe Ratio")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # CAGR Evolution
            cagr_data = self.metrics_history['cagr_values']
            ax2.plot(range(len(cagr_data)), cagr_data, 'orange', linewidth=2.5, marker='s', markersize=4)
            ax2.axhline(y=10, color='green', linestyle='--', alpha=0.7, label='Target (10%)')
            ax2.set_title("CAGR Evolution", fontsize=14, fontweight='bold')
            ax2.set_ylabel("CAGR (%)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Monthly Returns Heatmap
            if len(self.metrics_history['daily_profits']) >= 20:
                monthly_data = []
                month_labels = []
                profits = self.metrics_history['daily_profits']
                losses = self.metrics_history['daily_losses']
                
                for i in range(0, len(profits), 20):  # Approximate monthly grouping
                    month_profits = sum(profits[i:i+20])
                    month_losses = sum(losses[i:i+20])
                    monthly_return = month_profits - month_losses
                    monthly_data.append(monthly_return)
                    month_labels.append(f'M{len(monthly_data)}')
                
                colors = ['green' if ret >= 0 else 'red' for ret in monthly_data]
                bars = ax3.bar(month_labels, monthly_data, color=colors, alpha=0.7)
                ax3.set_title("Monthly Returns", fontsize=14, fontweight='bold')
                ax3.set_ylabel("Monthly P&L ($)")
                ax3.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, monthly_data):
                    ax3.text(bar.get_x() + bar.get_width()/2., value, f'${value:.0f}',
                            ha='center', va='bottom' if value >= 0 else 'top', fontsize=10)
            
            # Risk-Return Scatter
            if len(self.metrics_history['account_values']) >= 30:
                returns_data = []
                for i in range(1, len(self.metrics_history['account_values'])):
                    prev_val = self.metrics_history['account_values'][i-1]
                    curr_val = self.metrics_history['account_values'][i]
                    if prev_val > 0:
                        daily_return = (curr_val - prev_val) / prev_val
                        returns_data.append(daily_return)
                
                if len(returns_data) > 1:
                    mean_return = np.mean(returns_data) * 252  # Annualized
                    volatility = np.std(returns_data) * np.sqrt(252)  # Annualized
                    
                    ax4.scatter([volatility*100], [mean_return*100], s=200, c='blue', alpha=0.8, marker='o')
                    ax4.set_xlabel("Volatility (% Annualized)")
                    ax4.set_ylabel("Return (% Annualized)")
                    ax4.set_title("Risk-Return Profile", fontsize=14, fontweight='bold')
                    ax4.grid(True, alpha=0.3)
                    
                    # Add quadrant lines
                    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                    ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
                    
                    # Add text annotation
                    ax4.annotate(f'Return: {mean_return*100:.1f}%\nVolatility: {volatility*100:.1f}%', 
                                xy=(volatility*100, mean_return*100), xytext=(10, 10),
                                textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='yellow', alpha=0.7), fontsize=10)
            
            plt.tight_layout()
            self.tb_writer.add_figure("TradingCharts/PerformanceDashboard", fig, global_step)
            plt.close(fig)

    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        if len(self.metrics_history['account_values']) == 0:
            return "No trading data available."
            
        current_value = self.metrics_history['account_values'][-1]
        total_return = ((current_value - self.starting_capital) / self.starting_capital) * 100
        
        # Calculate additional metrics
        win_rate = 0
        profit_factor = 0
        if len(self.metrics_history['daily_profits']) > 0:
            winning_days = sum(1 for p in self.metrics_history['daily_profits'] if p > 0)
            total_days = len(self.metrics_history['daily_profits'])
            win_rate = (winning_days / total_days) * 100 if total_days > 0 else 0
            
            total_profits = sum(self.metrics_history['daily_profits'])
            total_losses = sum(self.metrics_history['daily_losses'])
            profit_factor = total_profits / total_losses if total_losses > 0 else float('inf')
        
        current_sharpe = self.metrics_history['sharpe_ratios'][-1] if self.metrics_history['sharpe_ratios'] else 0
        current_cagr = self.metrics_history['cagr_values'][-1] if self.metrics_history['cagr_values'] else 0
        
        report = f"""
🚀 TRADING PERFORMANCE SUMMARY
{'='*50}
💰 Account Performance:
   • Starting Capital: ${self.starting_capital:,.2f}
   • Current Value: ${current_value:,.2f}
   • Total Return: {total_return:+.2f}%
   • Peak Value: ${self.peak_value:,.2f}
   
📈 Performance Metrics:
   • CAGR: {current_cagr:.2f}%
   • Sharpe Ratio: {current_sharpe:.2f}
   • Maximum Drawdown: {self.max_drawdown:.2f}%
   
📊 Trading Statistics:
   • Win Rate: {win_rate:.1f}%
   • Profit Factor: {profit_factor:.2f}
   • Total Trading Days: {len(self.metrics_history['daily_profits'])}
   
⚠️ Risk Assessment:
   • Current Drawdown: {self.metrics_history['drawdowns'][-1]:.2f}%
   • Risk Level: {'HIGH' if self.max_drawdown > 20 else 'MODERATE' if self.max_drawdown > 10 else 'LOW'}
        """
        
        return report
