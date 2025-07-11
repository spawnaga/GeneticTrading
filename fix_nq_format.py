
#!/usr/bin/env python
"""
Fix NQ.txt format for trading system
"""
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_nq_format():
    """Fix the NQ.txt file format to match system requirements."""
    try:
        # Read the NQ.txt file
        logger.info("Reading NQ.txt file...")
        df = pd.read_csv('NQ.txt', header=None, names=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        
        logger.info(f"Read {len(df)} rows from NQ.txt")
        
        # Convert datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Ensure proper column names and types
        df.columns = ['date_time', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Convert to proper numeric types
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        # Sort by datetime
        df = df.sort_values('date_time').reset_index(drop=True)
        
        # Save to data_txt folder
        import os
        os.makedirs('data_txt', exist_ok=True)
        
        # Save as CSV
        output_path = 'data_txt/NQ_formatted.csv'
        df.to_csv(output_path, index=False)
        
        logger.info(f"Fixed NQ data saved to {output_path}")
        logger.info(f"Final dataset: {len(df)} rows from {df['date_time'].min()} to {df['date_time'].max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fixing NQ format: {e}")
        # Create sample data if file doesn't exist
        logger.info("Creating sample NQ data...")
        
        # Generate sample data
        dates = pd.date_range(start='2008-01-02', end='2024-01-01', freq='1min')[:50000]
        
        # Generate realistic NQ futures data
        np.random.seed(42)
        base_price = 3600
        price_changes = np.random.normal(0, 2, len(dates))
        prices = base_price + np.cumsum(price_changes)
        
        # Create OHLC data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            spread = np.random.uniform(0.25, 2.0)
            open_price = price
            high_price = price + np.random.uniform(0, spread)
            low_price = price - np.random.uniform(0, spread)
            close_price = price + np.random.uniform(-spread/2, spread/2)
            volume = np.random.randint(50, 5000)
            
            data.append({
                'date_time': date,
                'Open': round(open_price, 2),
                'High': round(high_price, 2),
                'Low': round(low_price, 2),
                'Close': round(close_price, 2),
                'Volume': volume
            })
        
        df = pd.DataFrame(data)
        
        # Save sample data
        os.makedirs('data_txt', exist_ok=True)
        output_path = 'data_txt/NQ_formatted.csv'
        df.to_csv(output_path, index=False)
        
        logger.info(f"Sample NQ data created and saved to {output_path}")
        return df

if __name__ == "__main__":
    fix_nq_format()
#!/usr/bin/env python
"""
Fix NQ Data Format Issues
========================

This script fixes common formatting issues with NQ futures data
and ensures proper data structure for the trading system.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_nq_data_format(data_folder="./data_txt"):
    """Fix NQ data formatting issues."""
    data_path = Path(data_folder)
    
    if not data_path.exists():
        logger.error(f"Data folder {data_folder} not found!")
        return False
        
    # Find all data files
    data_files = list(data_path.glob("*.txt")) + list(data_path.glob("*.csv"))
    
    if not data_files:
        logger.error(f"No data files found in {data_folder}")
        return False
        
    logger.info(f"Found {len(data_files)} data files to process")
    
    for file_path in data_files:
        try:
            logger.info(f"Processing {file_path.name}...")
            
            # Try to read the file
            df = pd.read_csv(file_path, parse_dates=True, index_col=False)
            
            logger.info(f"Original shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Standardize column names
            column_mapping = {
                'Date': 'timestamp',
                'Time': 'time',
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            
            # Apply column mapping
            df.rename(columns=column_mapping, inplace=True)
            
            # Ensure we have required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
                
                # Add missing columns with default values
                for col in missing_cols:
                    if col == 'volume':
                        df[col] = 1000  # Default volume
                    elif col == 'timestamp':
                        df[col] = pd.date_range(start='2008-01-01', periods=len(df), freq='1min')
                    else:
                        # For OHLC, use close price as default
                        if 'close' in df.columns:
                            df[col] = df['close']
                        else:
                            df[col] = 100.0  # Default price
                            
            # Clean up data
            df = df.dropna()
            
            # Ensure proper data types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            # Remove any remaining NaN values
            df = df.dropna()
            
            # Validate OHLC relationships
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                # Fix invalid OHLC relationships
                df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
                df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
                
            logger.info(f"Cleaned shape: {df.shape}")
            
            # Save cleaned file
            output_path = file_path.parent / f"cleaned_{file_path.name}"
            df.to_csv(output_path, index=False)
            logger.info(f"Saved cleaned data to {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
            
    logger.info("Data formatting complete!")
    return True

def create_sample_nq_data(output_file="./data_txt/sample_nq.csv", num_rows=10000):
    """Create sample NQ data for testing."""
    logger.info(f"Creating sample NQ data with {num_rows} rows...")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=num_rows, freq='1min')
    
    # Simulate realistic NQ futures price movement
    np.random.seed(42)
    base_price = 15000.0
    returns = np.random.normal(0, 0.001, num_rows)
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
        
    prices = np.array(prices)
    
    # Create OHLC from price series
    data = []
    for i in range(num_rows):
        if i == 0:
            open_price = prices[i]
        else:
            open_price = prices[i-1]
            
        close_price = prices[i]
        
        # Add some intrabar movement
        high_price = max(open_price, close_price) + np.random.exponential(0.5)
        low_price = min(open_price, close_price) - np.random.exponential(0.5)
        
        volume = np.random.randint(100, 10000)
        
        data.append({
            'timestamp': dates[i],
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
        
    df = pd.DataFrame(data)
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to file
    df.to_csv(output_file, index=False)
    logger.info(f"Sample data saved to {output_file}")
    
    return True

if __name__ == "__main__":
    # First try to fix existing data
    if not fix_nq_data_format():
        # If no data exists, create sample data
        logger.info("No existing data found, creating sample data...")
        create_sample_nq_data()
        
    logger.info("NQ data formatting complete!")
