
#!/usr/bin/env python
"""
Real-time Trading Activity Monitor Script
========================================

Run this script to monitor trading activity in real-time.
Usage: python monitor_trading_activity.py
"""

import time
import os
from pathlib import Path
import json
from datetime import datetime

def tail_trading_log(log_file="./logs/trading_activity.log"):
    """Tail the trading activity log file."""
    log_path = Path(log_file)
    
    if not log_path.exists():
        print(f"⏳ Waiting for trading log: {log_path}")
        while not log_path.exists():
            time.sleep(1)
    
    print(f"📊 Monitoring trading activity: {log_path}")
    print("=" * 80)
    
    # Follow the log file
    with open(log_path, 'r') as f:
        # Go to end of file
        f.seek(0, 2)
        
        while True:
            line = f.readline()
            if line:
                # Parse and format the line
                if "Step" in line and "|" in line:
                    parts = line.strip().split("|")
                    if len(parts) >= 4:
                        timestamp = parts[0].split("]")[0].split("[")[1] if "]" in parts[0] else ""
                        step_action = parts[1].strip()
                        price = parts[2].strip()
                        position = parts[3].strip()
                        balance = parts[4].strip() if len(parts) > 4 else ""
                        
                        # Color code actions
                        if "BUY" in step_action:
                            action_color = "🟢"
                        elif "SELL" in step_action:
                            action_color = "🔴"
                        else:
                            action_color = "⚪"
                            
                        print(f"{action_color} {timestamp} | {step_action} | {price} | {position} | {balance}")
                elif "POSITION CHANGE" in line:
                    print(f"🔄 {line.strip()}")
                elif "NEW POSITION" in line:
                    print(f"🟢 {line.strip()}")
                elif "CLOSED POSITION" in line:
                    print(f"🔴 {line.strip()}")
                elif "REVERSED POSITION" in line:
                    print(f"🔄 {line.strip()}")
            else:
                time.sleep(0.1)

def monitor_training_logs():
    """Monitor the main training logs for trading activity."""
    logs_dir = Path("./logs")
    
    print("🔍 Searching for training logs...")
    
    # Look for rank logs
    rank_logs = list(logs_dir.glob("trading_system_rank_*.log"))
    
    if not rank_logs:
        print("⏳ No training logs found. Waiting...")
        return
    
    print(f"📋 Found {len(rank_logs)} training log files")
    
    # Monitor the first rank log
    main_log = rank_logs[0]
    print(f"📊 Monitoring: {main_log}")
    print("=" * 80)
    
    with open(main_log, 'r') as f:
        # Go to end of file
        f.seek(0, 2)
        
        while True:
            line = f.readline()
            if line:
                # Filter for trading-related logs
                if any(keyword in line for keyword in [
                    "📊 Step", "🟢 NEW POSITION", "🔴 CLOSED POSITION", 
                    "🔄 REVERSED POSITION", "🔵 ADDED TO POSITION", 
                    "💰 ACCOUNT STATUS"
                ]):
                    # Extract timestamp
                    timestamp = ""
                    if "[" in line and "]" in line:
                        timestamp = line.split("]")[0].split("[")[-1]
                    
                    # Clean up the line
                    clean_line = line.split(":")[-1].strip()
                    print(f"[{timestamp}] {clean_line}")
            else:
                time.sleep(0.1)

def show_menu():
    """Show monitoring options menu."""
    print("\n🎯 Trading Activity Monitor")
    print("=" * 40)
    print("1. Monitor dedicated trading activity log")
    print("2. Monitor main training logs")
    print("3. Show current activity summary")
    print("4. Exit")
    print("=" * 40)
    
    choice = input("Select option (1-4): ").strip()
    return choice

def main():
    """Main monitoring function."""
    print("🚀 Trading Activity Monitor Started")
    
    while True:
        choice = show_menu()
        
        if choice == "1":
            try:
                tail_trading_log()
            except KeyboardInterrupt:
                print("\n⏸️  Monitoring paused")
                continue
                
        elif choice == "2":
            try:
                monitor_training_logs()
            except KeyboardInterrupt:
                print("\n⏸️  Monitoring paused")
                continue
                
        elif choice == "3":
            try:
                from trading_activity_monitor import trading_monitor
                print(trading_monitor.get_activity_summary())
            except ImportError:
                print("❌ Trading monitor not available")
            input("\nPress Enter to continue...")
            
        elif choice == "4":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
