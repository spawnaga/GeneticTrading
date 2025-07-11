#!/usr/bin/env python
"""
Comprehensive Trading Dashboard
==============================
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime

def get_trading_stats():
    """Get trading statistics from the table."""
    table_file = Path("./logs/trading_table.json")

    if not table_file.exists():
        return {}

    try:
        with open(table_file, 'r') as f:
            data = json.load(f)

        if not data:
            return {}

        # Calculate stats
        total_trades = len([entry for entry in data if entry['action'] != 'HOLD'])
        buy_trades = len([entry for entry in data if entry['action'] == 'BUY'])
        sell_trades = len([entry for entry in data if entry['action'] == 'SELL'])
        hold_actions = len([entry for entry in data if entry['action'] == 'HOLD'])

        # Get P&L stats
        profitable_trades = len([entry for entry in data if entry['status'] == 'PROFIT'])
        losing_trades = len([entry for entry in data if entry['status'] == 'LOSS'])

        # Get latest info
        latest = data[-1]

        return {
            'total_entries': len(data),
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'hold_actions': hold_actions,
            'profitable_trades': profitable_trades,
            'losing_trades': losing_trades,
            'latest_step': latest['step'],
            'latest_price': latest['price'],
            'latest_position': latest['position'],
            'latest_balance': latest['balance'],
            'win_rate': (profitable_trades / max(total_trades, 1)) * 100
        }

    except Exception as e:
        print(f"Error calculating stats: {e}")
        return {}

def show_dashboard():
    """Show the comprehensive trading dashboard."""

    # Clear screen
    print("\033[2J\033[H", end="")

    print("🎯 COMPREHENSIVE TRADING DASHBOARD")
    print("=" * 80)
    print(f"📅 Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Get statistics
    stats = get_trading_stats()

    if not stats:
        print("❌ No trading data available")
        print("💡 Start training to see dashboard data")
        return

    # Display stats
    print("📊 TRADING STATISTICS")
    print("-" * 40)
    print(f"Total Table Entries: {stats['total_entries']:,}")
    print(f"Total Trades:        {stats['total_trades']:,}")
    print(f"  • Buy Trades:      {stats['buy_trades']:,}")
    print(f"  • Sell Trades:     {stats['sell_trades']:,}")
    print(f"  • Hold Actions:    {stats['hold_actions']:,}")
    print()
    print(f"Profitable Trades:   {stats['profitable_trades']:,}")
    print(f"Losing Trades:       {stats['losing_trades']:,}")
    print(f"Win Rate:            {stats['win_rate']:.1f}%")
    print()
    print(f"Latest Step:         {stats['latest_step']:,}")
    print(f"Latest Price:        {stats['latest_price']}")
    print(f"Latest Position:     {stats['latest_position']}")
    print(f"Latest Balance:      {stats['latest_balance']}")
    print()

    # Show recent trading table
    print("📋 RECENT TRADING ACTIVITY")
    print("-" * 80)

    # Import and use the view function
    from view_trading_table import view_trading_table
    view_trading_table(15)

def monitor_dashboard(refresh_interval=15):
    """Monitor the dashboard with auto-refresh."""

    try:
        while True:
            show_dashboard()
            print(f"\n🔄 Auto-refreshing every {refresh_interval} seconds... (Ctrl+C to stop)")
            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n👋 Dashboard monitoring stopped.")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        monitor_dashboard()
    else:
        show_dashboard()
        print("\nTip: Run 'python trading_dashboard.py monitor' for live monitoring")