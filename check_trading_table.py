
#!/usr/bin/env python
"""
Check Trading Table Status
=========================
"""

import json
from pathlib import Path

def check_trading_table():
    """Check if trading table exists and show recent entries."""
    
    table_file = Path("./logs/trading_table.json")
    
    print("🔍 Checking trading table status...")
    print(f"📁 Looking for: {table_file.absolute()}")
    
    if not table_file.exists():
        print("❌ Trading table file does not exist yet")
        print("💡 The table will be created when trading starts")
        
        # Check if logs directory exists
        logs_dir = Path("./logs")
        if logs_dir.exists():
            print(f"📂 Logs directory exists with {len(list(logs_dir.iterdir()))} files:")
            for file in logs_dir.iterdir():
                print(f"  - {file.name}")
        else:
            print("📂 Logs directory does not exist")
        return
    
    try:
        with open(table_file, 'r') as f:
            trading_data = json.load(f)
            
        print(f"✅ Trading table found with {len(trading_data)} entries")
        
        if trading_data:
            print("\n📊 Recent trading activities:")
            print("-" * 80)
            print(f"{'Time':<10} {'Action':<6} {'Price':<12} {'Position':<8} {'P&L':<12} {'Status':<8}")
            print("-" * 80)
            
            # Show last 10 entries
            for entry in trading_data[-10:]:
                time_short = entry['timestamp'].split(' ')[1] if ' ' in entry['timestamp'] else entry['timestamp']
                print(f"{time_short:<10} {entry['action']:<6} {entry['price']:<12} {entry['position']:<8} {entry['pnl']:<12} {entry['status']:<8}")
        else:
            print("📊 Trading table is empty")
            
    except Exception as e:
        print(f"❌ Error reading trading table: {e}")

if __name__ == "__main__":
    check_trading_table()
