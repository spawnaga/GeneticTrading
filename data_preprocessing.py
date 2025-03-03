import glob
import os

import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data_from_text_files(data_folder):
    """
    Load all the .txt files in `data_folder` (stock/futures data).
    Each file should be a CSV with columns: date_time, open, high, low, close, volume.
    Return a concatenated DataFrame sorted by date_time.
    """
    all_files = glob.glob(os.path.join(data_folder, '*.txt'))
    dfs = []
    for file in all_files:
        df = pd.read_csv(file,
                         names=['date_time', 'open', 'high', 'low', 'close', 'volume'],
                         parse_dates=['date_time'],
                         header=None)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values(by='date_time').reset_index(drop=True)
    return combined_df


def feature_engineering(df):
    """
    Add additional features like returns, moving averages, RSI, etc., if desired.
    For simplicity, let's add a simple return column and scaled volume.
    """
    # Compute a simple return
    df['return'] = df['close'].pct_change().fillna(0)

    # Example technical indicator: moving average
    df['ma_10'] = df['close'].rolling(window=10).mean().fillna(method='bfill')

    # RSI, Bollinger, or other features can be added similarly
    # ...
    # For now, let's keep it minimal

    return df


def scale_and_split(df):
    """
    Scale numeric columns using StandardScaler.
    Split into train and test sets (e.g., 80/20).
    Return: train_data, test_data as Pandas DataFrames or NumPy arrays
    """
    # We'll drop the date_time here for the sake of model input
    # But let's keep it in a separate column for reference
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'return', 'ma_10']
    # We also keep the next minute's close price or return as a label
    # but in RL, we typically don't do supervised label => We'll just treat price as part of state

    # Drop any rows with NaN due to rolling computations
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)

    # Scale
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Split
    train_size = int(0.8 * len(df))
    train_data = df.iloc[:train_size].copy()
    test_data = df.iloc[train_size:].copy()

    return train_data, test_data, scaler


def create_environment_data(data_folder, max_rows=1000):
    """
    Load, preprocess, and return a limited dataset for testing.
    max_rows: Number of rows to limit for faster testing.
    """
    df = load_data_from_text_files(data_folder)
    df = feature_engineering(df)

    # ✅ Limit dataset to first `max_rows` rows
    df = df.iloc[:max_rows].copy()

    train_data, test_data, scaler = scale_and_split(df)
    return train_data, test_data, scaler

