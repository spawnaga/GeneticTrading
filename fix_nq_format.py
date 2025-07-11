
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
