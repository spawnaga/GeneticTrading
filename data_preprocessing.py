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
    Used for identifying unique combinations of files.
    """
    hasher = hashlib.sha256()
    for file in sorted(file_list):
        file_stats = os.stat(file)
        hasher.update(file.encode('utf-8'))
        hasher.update(str(file_stats.st_mtime).encode('utf-8'))
    return hasher.hexdigest()


def load_single_file_gpu(file):
    """
    Load a single CSV file into a GPU DataFrame.
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
    Load all `.txt` files from `data_folder` using GPU acceleration.
    Checks if a cached preprocessed version exists based on file hashes.
    """
    os.makedirs(cache_dir, exist_ok=True)

    all_files = glob.glob(os.path.join(data_folder, '*.txt'))
    files_hash = hash_files(all_files)
    cached_file = os.path.join(cache_dir, f'combined_data_{files_hash}.parquet')

    # Check if cached file exists
    if os.path.exists(cached_file):
        print(f"Cached file found: {cached_file}. Loading directly.")
        combined_df = cudf.read_parquet(cached_file)
    else:
        print(f"No cached file found. Loading {len(all_files)} files using GPU acceleration...")

        with ThreadPoolExecutor(max_workers=8) as executor:
            dfs = list(executor.map(load_single_file_gpu, all_files))

        combined_df = cudf.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values('date_time').reset_index(drop=True)

        # Save to cache
        combined_df.to_parquet(cached_file)
        print(f"Saved combined GPU DataFrame to {cached_file}")

    return combined_df


def feature_engineering_gpu(df):
    """
    Perform GPU-based feature engineering:
    - Simple returns
    - Moving average (10-minute window)
    """
    df['return'] = df['close'].pct_change().fillna(0)
    df['ma_10'] = df['close'].rolling(window=10).mean().bfill()
    return df


def scale_and_split_gpu(df):
    """
    Scale numeric columns using StandardScaler (CPU required), then split into train and test sets.
    """
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'return', 'ma_10']

    # Drop NaNs from GPU DataFrame
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)

    # Convert GPU DataFrame to CPU pandas DataFrame for scaling
    df_cpu = df.to_pandas()

    scaler = StandardScaler()
    df_cpu[numeric_cols] = scaler.fit_transform(df_cpu[numeric_cols])

    train_size = int(0.8 * len(df_cpu))
    train_data = df_cpu.iloc[:train_size].copy()
    test_data = df_cpu.iloc[train_size:].copy()

    return train_data, test_data, scaler


def create_environment_data(data_folder, max_rows=None, use_gpu=True, cache_folder='./cached_data'):
    """
    High-performance loading, preprocessing, scaling, and splitting pipeline.
    - `max_rows`: Optional limit for faster testing. Set to None for no limit.
    - `use_gpu`: Boolean flag to utilize GPU acceleration.
    """
    if use_gpu:
        df = load_data_from_text_files_gpu(data_folder)
        df = feature_engineering_gpu(df)
    else:
        df = load_data_from_text_files(data_folder)
        df = feature_engineering(df)

    if max_rows is not None:
        df = df.iloc[:max_rows]

    train_data, test_data, scaler = scale_and_split_gpu(df)

    return train_data, test_data, scaler


# Original CPU functions (kept for backward compatibility or non-GPU usage)

def load_data_from_text_files(data_folder):
    """
    Load all .txt files using pandas (CPU-based).
    """
    all_files = glob.glob(os.path.join(data_folder, '*.txt'))
    dfs = []
    for file in all_files:
        df = pd.read_csv(
            file,
            names=['date_time', 'open', 'high', 'low', 'close', 'volume'],
            parse_dates=['date_time'],
            header=None
        )
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values(by='date_time').reset_index(drop=True)
    return combined_df


def feature_engineering(df):
    """
    CPU-based feature engineering.
    """
    df['return'] = df['close'].pct_change().fillna(0)
    df['ma_10'] = df['close'].rolling(window=10).mean().bfill()
    return df


def scale_and_split(df):
    """
    CPU-based scaling and splitting.
    """
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'return', 'ma_10']
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    train_size = int(0.8 * len(df))
    train_data = df.iloc[:train_size].copy()
    test_data = df.iloc[train_size:].copy()
    return train_data, test_data, scaler
