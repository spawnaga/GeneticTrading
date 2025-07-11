
#!/usr/bin/env python
"""
Test Trading Table Creation
==========================
"""

import json
from pathlib import Path
from datetime import datetime

def test_trading_table_creation():
    """Test if we can create a trading table entry."""
    
    # Create a sample table entry
    table_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "step": 1,
        "action": "BUY",
        "price": "$17000.00",
        "position": 1,
        "pos_change": "+1",
        "balance": "$100,000.00",
        "pnl": "+$500.00",
        "reward": "0.0050",
        "status": "PROFIT"
    }
    
    # Save to trading table file
    table_file = Path("./logs/trading_table.json")
    table_file.parent.mkdir(exist_ok=True)
    
    try:
        # Load existing data or create new
        if table_file.exists():
            with open(table_file, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []
            
        # Add test entry
        existing_data.append(table_entry)
        
        # Save updated data
        with open(table_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
            
        print(f"✅ Successfully created trading table with {len(existing_data)} entries")
        print(f"📁 File location: {table_file.absolute()}")
        
        # Show the content
        print("\n📊 Table content:")
        for entry in existing_data[-5:]:  # Show last 5 entries
            print(f"  {entry['timestamp']} | {entry['action']} | {entry['price']} | {entry['pnl']}")
            
    except Exception as e:
        print(f"❌ Error creating trading table: {e}")

if __name__ == "__main__":
    test_trading_table_creation()
