
#!/usr/bin/env python
"""
Real-time Trading Activity Monitor
=================================

Monitor trading actions and positions in real-time.
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque

logger = logging.getLogger("TRADING_MONITOR")

class TradingActivityMonitor:
    """Monitor and log real-time trading activity."""
    
    def __init__(self, log_file="./logs/trading_activity.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Activity tracking
        self.action_counts = defaultdict(int)
        self.recent_actions = deque(maxlen=50)
        self.position_changes = deque(maxlen=20)
        self.profit_loss_history = deque(maxlen=100)
        
        # Setup dedicated file logger
        self.setup_file_logging()
        
    def setup_file_logging(self):
        """Setup dedicated file logging for trading activity."""
        from logging.handlers import RotatingFileHandler
        
        # Create trading activity logger
        self.trading_logger = logging.getLogger("TRADING_ACTIVITY")
        self.trading_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.trading_logger.handlers.clear()
        
        # File handler for trading activity
        file_handler = RotatingFileHandler(
            self.log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        ))
        
        self.trading_logger.addHandler(file_handler)
        self.trading_logger.info("🚀 Trading Activity Monitor Started")
        
    def log_action(self, step, action, price, position, account_balance):
        """Log a trading action."""
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        action_name = action_names.get(action, f"UNK({action})")
        
        self.action_counts[action_name] += 1
        self.recent_actions.append({
            'step': step,
            'action': action_name,
            'price': price,
            'position': position,
            'balance': account_balance,
            'timestamp': datetime.now()
        })
        
        # Log to file
        self.trading_logger.info(
            f"Step {step:6d} | {action_name:4s} | "
            f"Price: ${price:8.2f} | "
            f"Pos: {position:3d} | "
            f"Balance: ${account_balance:10,.2f}"
        )
        
    def log_position_change(self, old_position, new_position, pnl=0.0):
        """Log position changes."""
        change = {
            'old_position': old_position,
            'new_position': new_position,
            'change': new_position - old_position,
            'pnl': pnl,
            'timestamp': datetime.now()
        }
        
        self.position_changes.append(change)
        
        if pnl != 0.0:
            self.profit_loss_history.append(pnl)
            
        # Log significant position changes
        if old_position != new_position:
            direction = "LONG" if new_position > 0 else "SHORT" if new_position < 0 else "FLAT"
            self.trading_logger.info(
                f"📍 POSITION CHANGE | {old_position} → {new_position} ({direction}) | "
                f"P&L: ${pnl:.2f}"
            )
            
    def get_activity_summary(self):
        """Get current trading activity summary."""
        total_actions = sum(self.action_counts.values())
        
        if total_actions == 0:
            return "No trading activity yet"
            
        # Calculate percentages
        action_pct = {
            action: (count / total_actions) * 100 
            for action, count in self.action_counts.items()
        }
        
        # Recent activity
        recent_trades = len([a for a in self.recent_actions if a['action'] != 'HOLD'])
        
        # P&L summary
        total_pnl = sum(self.profit_loss_history) if self.profit_loss_history else 0.0
        avg_pnl = total_pnl / len(self.profit_loss_history) if self.profit_loss_history else 0.0
        
        summary = f"""
🎯 TRADING ACTIVITY SUMMARY
{'='*40}
📊 Action Distribution:
   • HOLD: {action_pct.get('HOLD', 0):.1f}% ({self.action_counts['HOLD']} times)
   • BUY:  {action_pct.get('BUY', 0):.1f}% ({self.action_counts['BUY']} times)
   • SELL: {action_pct.get('SELL', 0):.1f}% ({self.action_counts['SELL']} times)
   
🔄 Recent Activity (last 50 actions):
   • Non-HOLD actions: {recent_trades}
   • Trading frequency: {(recent_trades/50)*100:.1f}%
   
💰 P&L Performance:
   • Total P&L: ${total_pnl:.2f}
   • Average trade P&L: ${avg_pnl:.2f}
   • Total trades: {len(self.profit_loss_history)}
   
📈 Current Status:
   • Last action: {self.recent_actions[-1]['action'] if self.recent_actions else 'None'}
   • Current position: {self.recent_actions[-1]['position'] if self.recent_actions else 'Unknown'}
"""
        return summary
        
    def print_live_summary(self, interval=30):
        """Print live summary every interval seconds."""
        last_print = 0
        
        while True:
            current_time = time.time()
            if current_time - last_print >= interval:
                print(self.get_activity_summary())
                last_print = current_time
            time.sleep(1)

# Global monitor instance
trading_monitor = TradingActivityMonitor()

def log_trading_action(step, action, price, position, account_balance):
    """Convenience function to log trading actions."""
    trading_monitor.log_action(step, action, price, position, account_balance)

def log_position_change(old_position, new_position, pnl=0.0):
    """Convenience function to log position changes."""
    trading_monitor.log_position_change(old_position, new_position, pnl)
