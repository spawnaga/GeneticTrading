#!/usr/bin/env python
"""
View Trading Table
==================
"""

import json
import time
from pathlib import Path

def view_trading_table(num_entries=20):
    """View recent entries from the trading table."""

    table_file = Path("./logs/trading_table.json")

    print("🎯 STRUCTURED TRADING TABLE")
    print("=" * 80)

    if not table_file.exists():
        print("❌ Trading table file does not exist")
        print("💡 The table will be created when trading starts")
        return

    try:
        with open(table_file, 'r') as f:
            trading_data = json.load(f)

        if not trading_data:
            print("📊 Trading table is empty")
            return

        print(f"✅ Found {len(trading_data)} trading entries")
        print()

        # Show recent entries
        recent_entries = trading_data[-num_entries:] if len(trading_data) > num_entries else trading_data

        # Table header
        print(f"{'Time':<19} {'Step':<8} {'Action':<6} {'Price':<12} {'Pos':<4} {'P&L':<12} {'Status':<8}")
        print("-" * 80)

        for entry in recent_entries:
            time_str = entry['timestamp']
            step = str(entry['step'])
            action = entry['action']
            price = entry['price']
            position = str(entry['position'])
            pnl = entry['pnl']
            status = entry['status']

            # Color coding for actions
            if action == "BUY":
                action_display = f"🟢{action}"
            elif action == "SELL":
                action_display = f"🔴{action}"
            else:
                action_display = f"⚪{action}"

            print(f"{time_str:<19} {step:<8} {action_display:<8} {price:<12} {position:<4} {pnl:<12} {status:<8}")

        print("-" * 80)
        print(f"Last updated: {trading_data[-1]['timestamp']}")

    except Exception as e:
        print(f"❌ Error reading trading table: {e}")

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
        print("\nTip: Run 'python view_trading_table.py monitor' for live monitoring")