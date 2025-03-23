import glob
import joblib
import os
import hashlib
import cudf
import numpy as np
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
        names=['date_time', 'Open', 'High', 'Low', 'Close', 'Volume'],
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
    GPU-based feature engineering robust for all securities.
    Creates segments based on weekday and 15-minute intervals.
    """
    df['return'] = df['Close'].pct_change().fillna(0).astype('float64')
    df['ma_10'] = df['Close'].rolling(window=10, min_periods=1).mean().bfill()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta).where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = (100 - (100 / (1 + rs))).bfill()

    df['volatility'] = df['return'].rolling(window=10).std().bfill()

    segment_dict = {}
    segment_counter = 0

    segments = []

    for dt in df['date_time'].to_pandas():
        weekday = dt.weekday()
        minutes_since_midnight = dt.hour * 60 + dt.minute
        minutes_quantized = (minutes_since_midnight // 15) * 15
        key = (weekday, minutes_quantized)

        if key not in segment_dict:
            segment_dict[key] = segment_counter
            segment_counter += 1

        segments.append(segment_dict[key])

    df['day_time_segment'] = cudf.Series(segments, dtype=np.int32)

    return df


def scale_and_split_gpu(df):
    """
    Scale numeric columns using StandardScaler (CPU required), then split into train and test sets.
    """
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'return', 'ma_10']

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


def load_data_from_text_files(data_folder):
    """
    Load all .txt files using pandas (CPU-based).
    """
    all_files = glob.glob(os.path.join(data_folder, '*.txt'))
    dfs = []
    for file in all_files:
        df = pd.read_csv(
            file,
            names=['date_time', 'Open', 'High', 'Low', 'Close', 'Volume'],
            parse_dates=['date_time'],
            header=None
        )
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values(by='date_time').reset_index(drop=True)
    return combined_df


def feature_engineering(df):
    """
    CPU-based feature engineering with robust handling of trading segments.
    Segments are calculated based purely on weekday and 15-minute intervals,
    allowing robustness across securities (stocks, forex, cryptos, futures, etc.).
    This approach handles holidays by naturally skipping segments when no data is present.
    """

    # Sort by date_time to ensure chronological order
    df = df.sort_values('date_time').reset_index(drop=True)

    # Calculate returns
    df['return'] = df['Close'].pct_change().fillna(0)

    # Calculate 10-period moving average
    df['ma_10'] = df['Close'].rolling(window=10, min_periods=1).mean().bfill()

    # Define segments by combining weekday and 15-minute intervals into a unique identifier
    # Create a dictionary mapping each (weekday, time) combination to a unique segment
    segment_dict = {}
    segment_counter = 0

    segments = []

    for dt in df['date_time']:
        weekday = dt.weekday()
        minutes_since_midnight = dt.hour * 60 + dt.minute

        # Quantize minutes to nearest 15-minute segment
        minutes_quantized = (minutes_since_midnight // 15) * 15

        key = (weekday, minutes_quantized)

        # Assign a unique segment ID if not already assigned
        if key not in segment_dict:
            segment_dict[key] = segment_counter
            segment_counter += 1

        # Append segment ID
        segments.append(segment_dict[key])

    # Add segments to DataFrame
    df['day_time_segment'] = segments

    # This approach naturally skips segments for holidays (no data present)

    return df


def scale_and_split(df):
    """
    CPU-based scaling and splitting.
    """
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'return', 'ma_10']
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)

    scaler = StandardScaler()
    # Save scaler
    joblib.dump(scaler, 'scaler.save')

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    train_size = int(0.8 * len(df))
    train_data = df.iloc[:train_size].copy()
    test_data = df.iloc[train_size:].copy()
    return train_data, test_data, scaler


def create_environment_data(data_folder, max_rows=None, use_gpu=True, cache_folder='./cached_data'):
    """
    High-performance loading, preprocessing, scaling, and splitting pipeline.

    Args:
        data_folder (str): Directory containing raw .txt data files.
        max_rows (int, optional): Number of rows to keep from the end of the dataset.
                                  If None, use all data. If positive, take the last max_rows.
        use_gpu (bool): Flag to enable GPU acceleration for data loading and processing.
        cache_folder (str): Directory to store cached preprocessed data.

    Returns:
        tuple: (train_data, test_data, scaler) as pandas DataFrames and a fitted scaler.
    """
    if use_gpu:
        df = load_data_from_text_files_gpu(data_folder, cache_dir=cache_folder)
        df = feature_engineering_gpu(df)
        train_data, test_data, scaler = scale_and_split_gpu(df)
    else:
        df = load_data_from_text_files(data_folder)
        df = feature_engineering(df)
        train_data, test_data, scaler = scale_and_split(df)

    # Apply row limiting if max_rows is specified (take last max_rows rows)
    if max_rows is not None:
        if max_rows > 0:
            train_data = train_data.iloc[-max_rows:].copy()  # Last max_rows of train
            test_data = test_data.iloc[-max_rows:].copy()  # Last max_rows of test
        else:
            raise ValueError(f"max_rows must be positive or None, got {max_rows}")

    return train_data, test_data, scaler


def process_live_row(new_row, scaler, segment_dict):
    """
    Processes a new incoming 1-minute OHLCV row using the existing scaler and segment dictionary.

    Parameters:
    - new_row (dict or DataFrame): New data row with keys ['date_time', 'Open', 'High', 'Low', 'Close', 'Volume'].
    - scaler (StandardScaler): Previously fitted scaler object.
    - segment_dict (dict): Previously created dictionary of day-time segments.

    Returns:
    - DataFrame: Processed and standardized row ready for prediction or further analysis.
    """

    if isinstance(new_row, dict):
        new_row = pd.DataFrame([new_row])

    # Ensure date_time is a datetime type
    new_row['date_time'] = pd.to_datetime(new_row['date_time'])

    # Feature engineering
    new_row['return'] = 0  # Cannot compute pct_change for single row
    new_row['ma_10'] = new_row['Close']  # Approximate as current Close

    # Compute day_time_segment
    weekday = new_row['date_time'].iloc[0].weekday()
    minutes_since_midnight = new_row['date_time'].iloc[0].hour * 60 + new_row['date_time'].iloc[0].minute
    minutes_quantized = (minutes_since_midnight // 15) * 15
    key = (weekday, minutes_quantized)

    # Assign existing segment or new one
    if key not in segment_dict:
        segment_dict[key] = max(segment_dict.values()) + 1

    new_row['day_time_segment'] = segment_dict[key]

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'return', 'ma_10']
    new_row[numeric_cols] = scaler.transform(new_row[numeric_cols])

    return new_row
