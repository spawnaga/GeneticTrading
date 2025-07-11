
#!/usr/bin/env python
"""
NQ Data Format Converter
========================
Convert NQ.txt to the expected format for the trading system
"""

import pandas as pd
import os
from pathlib import Path

def convert_nq_format():
    """Convert NQ.txt to standard format"""
    
    data_dir = Path('./data_txt')
    
    # Find NQ files
    nq_files = list(data_dir.glob('*NQ*.txt')) + list(data_dir.glob('*nq*.txt'))
    
    if not nq_files:
        print("❌ No NQ files found")
        return
        
    for file_path in nq_files:
        print(f"🔄 Converting {file_path.name}...")
        
        try:
            # Read the NQ file - assume first format from your sample
            df = pd.read_csv(
                file_path,
                names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'],
                header=None,
                parse_dates=['DateTime']
            )
            
            print(f"✅ Loaded {len(df)} rows")
            print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
            
            # Save in correct format
            output_file = data_dir / f"processed_{file_path.name}"
            df.to_csv(output_file, index=False, header=False)
            print(f"✅ Saved to {output_file.name}")
            
            # Show sample
            print("Sample data:")
            print(df.head())
            
        except Exception as e:
            print(f"❌ Error converting {file_path.name}: {e}")

if __name__ == "__main__":
    convert_nq_format()
