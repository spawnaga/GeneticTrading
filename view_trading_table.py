
#!/usr/bin/env python
"""
Trading Table Viewer
===================

View recent trading activities in a formatted table.
"""

import json
import time
from pathlib import Path
from datetime import datetime

def view_trading_table(num_rows=20):
    """Display recent trading activities in a table format."""
    table_file = Path("./logs/trading_table.json")
    
    if not table_file.exists():
        print("❌ No trading table data found. Start training to generate data.")
        return
        
    try:
        with open(table_file, 'r') as f:
            trading_data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading trading data: {e}")
        return
        
    if not trading_data:
        print("📊 No trading activities recorded yet.")
        return
        
    # Get recent entries
    recent_data = trading_data[-num_rows:] if len(trading_data) > num_rows else trading_data
    
    print(f"\n🚀 RECENT TRADING ACTIVITIES (Last {len(recent_data)} entries)")
    print("=" * 120)
    
    # Header
    print(f"{'Time':<10} {'Step':<6} {'Action':<6} {'Price':<10} {'Pos':<5} {'Change':<8} {'Balance':<12} {'P&L':<10} {'Reward':<10} {'Status':<8}")
    print("-" * 120)
    
    # Data rows
    for row in recent_data:
        # Color coding for terminal
        action_color = ""
        status_color = ""
        reset_color = "\033[0m"
        
        if row['action'] == 'BUY':
            action_color = "\033[92m"  # Green
        elif row['action'] == 'SELL':
            action_color = "\033[91m"  # Red
        elif row['action'] == 'HOLD':
            action_color = "\033[93m"  # Yellow
            
        if row['status'] == 'PROFIT':
            status_color = "\033[92m"  # Green
        elif row['status'] == 'LOSS':
            status_color = "\033[91m"  # Red
            
        print(f"{row['timestamp'].split(' ')[1]:<10} "
              f"{row['step']:<6} "
              f"{action_color}{row['action']:<6}{reset_color} "
              f"{row['price']:<10} "
              f"{row['position']:<5} "
              f"{row['pos_change']:<8} "
              f"{row['balance']:<12} "
              f"{status_color}{row['pnl']:<10}{reset_color} "
              f"{row['reward']:<10} "
              f"{status_color}{row['status']:<8}{reset_color}")
              
    print("-" * 120)
    
    # Summary stats
    total_trades = len([r for r in recent_data if r['action'] != 'HOLD'])
    profitable_trades = len([r for r in recent_data if r['status'] == 'PROFIT'])
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    print(f"\n📊 SUMMARY:")
    print(f"   • Total Entries: {len(recent_data)}")
    print(f"   • Active Trades: {total_trades}")
    print(f"   • Profitable Trades: {profitable_trades}")
    print(f"   • Win Rate: {win_rate:.1f}%")
    
    if recent_data:
        print(f"   • Current Balance: {recent_data[-1]['balance']}")
        print(f"   • Current Position: {recent_data[-1]['position']}")
        print(f"   • Last Update: {recent_data[-1]['timestamp']}")

def monitor_trading_table(refresh_interval=10):
    """Monitor trading table with auto-refresh."""
    print("🔄 Starting trading table monitor (Ctrl+C to stop)")
    
    try:
        while True:
            # Clear screen
            print("\033[2J\033[H", end="")
            
            view_trading_table(30)
            print(f"\n⏱️  Auto-refreshing every {refresh_interval} seconds...")
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n👋 Trading table monitor stopped.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        monitor_trading_table()
    else:
        view_trading_table()
        print("\nTip: Run 'python view_trading_table.py monitor' for auto-refresh mode")
