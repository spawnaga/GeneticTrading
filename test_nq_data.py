
#!/usr/bin/env python
"""
Test NQ Data Loading
===================
Quick test to verify NQ.txt file format and loading
"""

import pandas as pd
import os
from pathlib import Path

def test_nq_data():
    """Test loading NQ.txt file"""
    
    # Look for NQ.txt file
    data_files = []
    for pattern in ['NQ.txt', 'nq.txt', '*.txt']:
        files = list(Path('./data_txt').glob(pattern))
        data_files.extend(files)
    
    if not data_files:
        print("❌ No NQ.txt files found in ./data_txt/")
        print("Available files:")
        for f in Path('./data_txt').glob('*'):
            print(f"  - {f.name}")
        return
        
    for file_path in data_files[:3]:  # Test first 3 files
        print(f"\n📁 Testing file: {file_path.name}")
        
        try:
            # Test reading with headers
            df = pd.read_csv(file_path, nrows=10)
            print(f"✅ Headers detected: {list(df.columns)}")
            print(f"📊 Shape: {df.shape}")
            print("Sample data:")
            print(df.head(3))
            
            # Test datetime parsing
            if df.columns[0] in df.columns:
                first_col = df.columns[0]
                df[first_col] = pd.to_datetime(df[first_col], errors='coerce')
                print(f"✅ Datetime parsing successful")
                print(f"Date range: {df[first_col].min()} to {df[first_col].max()}")
            
        except Exception as e:
            print(f"❌ Error reading {file_path.name}: {e}")
            
            # Try without headers
            try:
                df = pd.read_csv(
                    file_path, 
                    names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'],
                    header=None,
                    nrows=10
                )
                print(f"✅ No-header format works: {df.shape}")
                print(df.head(3))
            except Exception as e2:
                print(f"❌ Also failed without headers: {e2}")

if __name__ == "__main__":
    test_nq_data()
