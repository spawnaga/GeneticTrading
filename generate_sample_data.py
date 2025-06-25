
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_sample_ohlcv_data(n_rows=10000, start_price=100.0):
    """Generate realistic OHLCV data with trends and patterns"""
    np.random.seed(42)
    
    dates = []
    opens, highs, lows, closes, volumes = [], [], [], [], []
    
    current_price = start_price
    start_date = datetime(2023, 1, 1)
    
    # Add trend and cyclical components
    trend = np.linspace(0, 0.05, n_rows)  # 5% upward trend over time
    cycle = 0.02 * np.sin(2 * np.pi * np.arange(n_rows) / 200)  # Cyclical pattern
    
    for i in range(n_rows):
        # Generate realistic price movement with trend and cycles
        base_volatility = 0.001 + 0.001 * np.sin(2 * np.pi * i / 100)  # Variable volatility
        trend_component = trend[i] * start_price / n_rows
        cycle_component = cycle[i] * current_price
        random_component = np.random.normal(0, base_volatility) * current_price
        
        change = trend_component + cycle_component + random_component
        
        open_price = current_price
        close_price = current_price + change
        
        # More realistic high/low with intraday volatility
        intraday_range = abs(np.random.normal(0, 0.002)) * current_price
        high_price = max(open_price, close_price) + intraday_range * 0.7
        low_price = min(open_price, close_price) - intraday_range * 0.7
        
        # Volume with some correlation to price movements
        base_volume = 5000
        volatility_volume = int(abs(change) / current_price * 50000)
        volume = max(1000, base_volume + volatility_volume + np.random.randint(-1000, 1000))
        
        dates.append(start_date + timedelta(minutes=i))
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        volumes.append(volume)
        
        current_price = close_price
    
    df = pd.DataFrame({
        'DateTime': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    })
    
    return df

if __name__ == "__main__":
    # Generate sample data
    data = generate_sample_ohlcv_data(10000)  # 10k rows for better training
    data.to_csv('./data_txt/large_sample_data.txt', index=False)
    print(f"Generated {len(data)} rows of sample data")
    
    # Also create a smaller test dataset
    test_data = generate_sample_ohlcv_data(2000, start_price=105.0)
    test_data.to_csv('./data_txt/test_data.txt', index=False)
    print(f"Generated {len(test_data)} rows of test data")
