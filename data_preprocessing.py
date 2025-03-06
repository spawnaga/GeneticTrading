import glob
import os
import hashlib
import cudf  # GPU-accelerated DataFrame
import pandas as pd
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor


def hash_files(file_list):
    """
    Generate a unique hash based on filenames and modification timestamps.
    Used to identify unique combinations of files for caching.
    """
    hasher = hashlib.sha256()
    for file in sorted(file_list):
        file_stats = os.stat(file)
        hasher.update(file.encode('utf-8'))
        hasher.update(str(file_stats.st_mtime).encode('utf-8'))
    return hasher.hexdigest()


def load_single_file_gpu(file):
    """
    Load a single CSV file into a GPU DataFrame using cudf.
    """
    df = cudf.read_csv(
        file,
        names=['date_time', 'open', 'high', 'low', 'close', 'volume'],
        parse_dates=['date_time'],
        header=None
    )
    return df


def load_data_from_text_files_gpu(data_folder, cache_dir='./cached_data'):
    """
    Load all `.txt` and `.csv` files from `data_folder` using GPU acceleration.
    Checks for cached data using file hashes to avoid redundant processing.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Look for both .txt and .csv files
    all_files = glob.glob(os.path.join(data_folder, '*.txt')) + glob.glob(os.path.join(data_folder, '*.csv'))
    if not all_files:
        raise ValueError(f"No .txt or .csv files found in {data_folder}")

    files_hash = hash_files(all_files)
    cached_file = os.path.join(cache_dir, f'combined_data_{files_hash}.parquet')

    if os.path.exists(cached_file):
        print(f"Cached file found: {cached_file}. Loading directly.")
        combined_df = cudf.read_parquet(cached_file)
    else:
        print(f"No cached file found. Loading {len(all_files)} files using GPU acceleration...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            dfs = list(executor.map(load_single_file_gpu, all_files))
        combined_df = cudf.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values('date_time').reset_index(drop=True)
        combined_df.to_parquet(cached_file)
        print(f"Saved combined GPU DataFrame to {cached_file}")

    return combined_df


def feature_engineering_gpu(df):
    """
    Perform GPU-based feature engineering:
    - Simple returns
    - Moving average (10-minute window)
    - RSI (Relative Strength Index)
    - Volatility (rolling standard deviation of returns)
    """
    # Ensure 'close' is float64 for precise calculations
    df['close'] = df['close'].astype('float64')

    # Simple returns
    df['return'] = df['close'].pct_change().fillna(0)

    # Moving average (10-minute window)
    df['ma_10'] = df['close'].rolling(window=10, min_periods=1).mean()
    df['ma_10'] = df['ma_10'].bfill()  # Backfill to handle initial NaNs

    # RSI (14-period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta).where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].bfill()

    # Volatility (10-period rolling standard deviation of returns)
    df['volatility'] = df['return'].rolling(window=10).std().bfill()

    return df


def scale_and_split_gpu(df):
    """
    Scale numeric columns using StandardScaler (on CPU) and split into train/test sets.
    """
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'return', 'ma_10', 'rsi', 'volatility']

    # Check for missing columns
    missing_cols = [col for col in numeric_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

    # Drop rows with NaNs in numeric columns
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)

    # Convert to CPU pandas DataFrame for scaling
    df_cpu = df.to_pandas()

    # Scale numeric columns
    scaler = StandardScaler()
    df_cpu[numeric_cols] = scaler.fit_transform(df_cpu[numeric_cols].astype('float64'))

    # Split into train and test sets (80-20 split)
    train_size = int(0.8 * len(df_cpu))
    train_data = df_cpu.iloc[:train_size].copy()
    test_data = df_cpu.iloc[train_size:].copy()

    return train_data, test_data, scaler


def create_environment_data(data_folder, max_rows=None, use_gpu=True, cache_folder='./cached_data'):
    """
    High-performance pipeline for loading, preprocessing, scaling, and splitting data.
    - `max_rows`: Optional limit for faster testing. Set to None for no limit.
    - `use_gpu`: Boolean flag to utilize GPU acceleration.
    """
    if use_gpu:
        df = load_data_from_text_files_gpu(data_folder, cache_folder)
        df = feature_engineering_gpu(df)
    else:
        df = load_data_from_text_files(data_folder)
        df = feature_engineering(df)

    if max_rows is not None:
        df = df.iloc[:max_rows]

    train_data, test_data, scaler = scale_and_split_gpu(df)

    return train_data, test_data, scaler


# CPU-based functions for non-GPU usage

def load_data_from_text_files(data_folder):
    """
    Load all .txt files using pandas (CPU-based).
    """
    all_files = glob.glob(os.path.join(data_folder, '*.txt'))
    if not all_files:
        raise ValueError(f"No .txt files found in {data_folder}")
    dfs = [pd.read_csv(file, names=['date_time', 'open', 'high', 'low', 'close', 'volume'],
                       parse_dates=['date_time'], header=None) for file in all_files]
    combined_df = pd.concat(dfs, ignore_index=True).sort_values(by='date_time').reset_index(drop=True)
    return combined_df


def feature_engineering(df):
    """
    CPU-based feature engineering.
    """
    df['return'] = df['close'].pct_change().fillna(0)
    df['ma_10'] = df['close'].rolling(window=10).mean().bfill()

    # RSI (14-period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta).where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].bfill()

    # Volatility
    df['volatility'] = df['return'].rolling(window=10).std().bfill()

    return df


def scale_and_split(df):
    """
    CPU-based scaling and splitting.
    """
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'return', 'ma_10', 'rsi', 'volatility']
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    train_size = int(0.8 * len(df))
    train_data = df.iloc[:train_size].copy()
    test_data = df.iloc[train_size:].copy()
    return train_data, test_data, scaler
