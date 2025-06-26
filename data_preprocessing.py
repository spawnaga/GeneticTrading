"""
High-Performance OHLCV Data Pipeline
------------------------------------

This module provides a single unified GPU-first pipeline that:
  1. Loads raw CSV/.txt files into cuDF DataFrames and caches as partitioned Parquet.
  2. Performs fully-vectorized feature engineering (returns, MA, RSI, volatility).
  3. Encodes temporal features cyclically and with one-hot (no Python loops).
  4. Scales numeric features on-GPU via cuML’s StandardScaler.
  5. Splits into train/test sets on-GPU.
  6. Persists scaler and segment mapping for reproducible inference.
  7. Uses logging everywhere (no print statements).
  8. Handles arbitrarily large datasets in chunks if needed.

"""

import glob
import os
import logging
import warnings

# Setup logger first before using it
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

try:
    import torch
    # Check for both GPU availability and CUDA driver compatibility
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            # Test basic CUDA operations to ensure driver compatibility
            torch.cuda.current_device()
            import cudf
            import cupy as cp
            # Test basic cupy operation
            test_array = cp.array([1, 2, 3])
            _ = cp.sum(test_array)
            HAS_CUDF = True
            logger.info(f"GPU processing enabled with {torch.cuda.device_count()} GPUs")
        except Exception as cuda_err:
            logger.warning(f"CUDA driver incompatible or insufficient: {cuda_err}")
            raise ImportError("CUDA driver issues detected")
    else:
        raise ImportError("No GPUs available")
except (ImportError, AttributeError, RuntimeError, Exception):
    cudf = pd
    cp = np
    HAS_CUDF = False
    logger.info("Using CPU fallback for data processing")

# Handle cuML imports separately with better error handling
try:
    from cuml.preprocessing import StandardScaler as CuStandardScaler
    from cuml.model_selection import train_test_split
    HAS_CUML = True
except (ImportError, AttributeError, RuntimeError) as e:
    # Handle various CUDA-related import errors
    if "cuda" in str(e).lower() or "numba" in str(e).lower():
        warnings.warn(f"cuML import failed due to CUDA/numba issues ({e}); using CPU fallback.")
    else:
        warnings.warn(f"cuML import failed ({e}); using CPU fallback.")
    from sklearn.preprocessing import StandardScaler as CuStandardScaler
    from sklearn.model_selection import train_test_split
    HAS_CUML = False

from utils import hash_files

# ──────────────────────────────────────────────────────────────────────────────
# Logger setup (moved to top of file)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Constants & Defaults
# ──────────────────────────────────────────────────────────────────────────────
NUMPY_RANDOM_SEED = 42
WINDOW_RETURNS     = 1   # for pct_change
WINDOW_MA          = 10
WINDOW_RSI         = 14
WINDOW_VOL          = 10
BIN_SIZE_MINUTES   = 15
SECONDS_IN_DAY     = 24 * 60
WEEKDAYS           = 7

# ──────────────────────────────────────────────────────────────────────────────
# Data Loading & Caching
# ──────────────────────────────────────────────────────────────────────────────

def load_and_cache_data(
    data_folder: str,
    cache_dir: str = "./cached_data",
    pattern: str = "*.txt",
    read_chunk_size: int = 500000  # Read files in 500K row chunks
) -> cudf.DataFrame:
    """
    Load all text/CSV files from `data_folder` into a single cuDF DataFrame.
    If a parquet cache exists (based on file-hash), load from cache instead.
    Uses chunked reading for memory efficiency with large files.
    """
    os.makedirs(cache_dir, exist_ok=True)
    files = glob.glob(os.path.join(data_folder, pattern))
    if not files:
        logger.error("No files found in %s matching %s", data_folder, pattern)
        raise FileNotFoundError(f"No files in {data_folder}/{pattern}")
    cache_hash = hash_files(files)
    cache_file = os.path.join(cache_dir, f"combined_{cache_hash}.parquet")

    if os.path.exists(cache_file):
        logger.info("Loading cached dataset from %s", cache_file)
        try:
            if HAS_CUDF:
                return cudf.read_parquet(cache_file)
            else:
                return pd.read_parquet(cache_file)
        except Exception as e:
            logging.warning(f"Failed to load cached data with GPU backend: {e}, falling back to pandas")
            return pd.read_parquet(cache_file)

    logger.info("Scanning %d files for raw data (using chunked reading)...", len(files))

    # Process files in chunks to handle large datasets
    all_chunks = []

    for fp in files:
        logger.info(f"Processing file: {os.path.basename(fp)}")

        # Read file (cuDF doesn't support chunksize, so read directly)
        try:
            # Read the file and properly handle the comma-separated format with many columns
            try:
                # First, read a sample to detect the actual number of columns
                sample_df = pd.read_csv(fp, nrows=5, header=None)
                num_cols = len(sample_df.columns)
                logger.info(f"Detected {num_cols} columns in {os.path.basename(fp)}")

                # Create column names - first 6 are OHLCV data, rest are features
                column_names = ["date_time", "Open", "High", "Low", "Close", "Volume"]
                if num_cols > 6:
                    # Add feature column names for the additional columns
                    feature_cols = [f"feature_{i}" for i in range(6, num_cols)]
                    column_names.extend(feature_cols)

                if HAS_CUDF:
                    # cuDF reading with proper null handling
                    file_df = cudf.read_csv(
                        fp,
                        names=column_names,
                        header=None,
                        na_values=['', ' ', 'null', 'NULL', 'nan', 'NaN'],
                        keep_default_na=True
                    )
                    # Convert datetime column with cuDF-compatible method
                    try:
                        # First convert to pandas for datetime parsing, then back to cuDF
                        if HAS_CUDF and hasattr(file_df, 'to_pandas'):
                            temp_df = file_df.to_pandas()
                            temp_df["date_time"] = pd.to_datetime(temp_df["date_time"], errors='coerce')
                            file_df = cudf.DataFrame.from_pandas(temp_df)
                        else:
                            file_df["date_time"] = cudf.to_datetime(file_df["date_time"])
                    except Exception as e:
                        logger.warning(f"cuDF datetime conversion failed: {e}, using pandas fallback")
                        # Complete fallback to pandas
                        if hasattr(file_df, 'to_pandas'):
                            file_df = file_df.to_pandas()
                        file_df["date_time"] = pd.to_datetime(file_df["date_time"], errors='coerce')
                else:
                    # Use pandas with better null handling
                    file_df = pd.read_csv(
                        fp,
                        names=column_names,
                        header=None,
                        na_values=['', ' ', 'null', 'NULL', 'nan', 'NaN'],
                        keep_default_na=True,
                        low_memory=False
                    )
                    # Convert datetime column
                    file_df["date_time"] = pd.to_datetime(file_df["date_time"], errors='coerce')

                # Keep only the basic OHLCV columns for consistency
                basic_cols = ["date_time", "Open", "High", "Low", "Close", "Volume"]
                if all(col in file_df.columns for col in basic_cols):
                    file_df = file_df[basic_cols].copy()
                else:
                    logger.warning(f"Missing required columns in {fp}, skipping file")
                    continue

            except Exception as e:
                logger.warning(f"Error reading {fp} with auto-detection: {e}")
                # Fallback to basic 6-column reading
                try:
                    if HAS_CUDF:
                        file_df = cudf.read_csv(
                            fp,
                            names=["date_time", "Open", "High", "Low", "Close", "Volume"],
                            header=None,
                            usecols=[0,1,2,3,4,5],  # Use only first 6 columns
                            na_values=['', ' ', 'null', 'NULL', 'nan', 'NaN']
                        )
                        file_df["date_time"] = cudf.to_datetime(file_df["date_time"])
                    else:
                        file_df = pd.read_csv(
                            fp,
                            names=["date_time", "Open", "High", "Low", "Close", "Volume"],
                            header=None,
                            usecols=[0,1,2,3,4,5],  # Use only first 6 columns
                            na_values=['', ' ', 'null', 'NULL', 'nan', 'NaN'],
                            low_memory=False
                        )
                        file_df["date_time"] = pd.to_datetime(file_df["date_time"], errors='coerce')
                except Exception as e2:
                    logger.error(f"Fallback reading also failed for {fp}: {e2}")
                    continue

            # Drop rows with invalid dates
            file_df = file_df.dropna(subset=["date_time"])
            if len(file_df) > 0:
                all_chunks.append(file_df)

        except Exception as e:
            logger.warning(f"Failed to read {fp}: {e}")
            # Try with pandas as fallback
            try:
                logging.info(f"Trying pandas fallback for {os.path.basename(fp)}")
                fallback_chunks = read_file_chunked_pandas(fp)
                all_chunks.extend(fallback_chunks)
            except Exception as e2:
                logger.error(f"Pandas fallback also failed for {fp}: {e2}")
                continue

    logger.info("Combining all file chunks...")
    if not all_chunks:
        logger.error("No valid data chunks found!")
        raise ValueError("No valid data found in input files")

    try:
        if HAS_CUDF and len(all_chunks) > 0:
            df = cudf.concat(all_chunks, ignore_index=True)
        else:
            df = pd.concat(all_chunks, ignore_index=True)
    except Exception as e:
        logging.warning(f"Failed to combine chunks with GPU backend: {e}, using pandas")
        pandas_chunks = []
        for chunk in all_chunks:
            if hasattr(chunk, 'to_pandas'):
                pandas_chunks.append(chunk.to_pandas())
            else:
                pandas_chunks.append(chunk)
        df = pd.concat(pandas_chunks, ignore_index=True)

    # Ensure date_time column is properly typed before sorting (already named correctly)
    if HAS_CUDF and hasattr(df, 'to_pandas') and len(df) > 0:
        try:
            df["date_time"] = cudf.to_datetime(df["date_time"])
        except Exception as e:
            logger.warning(f"cuDF datetime conversion failed: {e}, converting to pandas")
            df = df.to_pandas()
            df["date_time"] = pd.to_datetime(df["date_time"], errors='coerce')
    else:
        df["date_time"] = pd.to_datetime(df["date_time"], errors='coerce')

    # Drop any remaining invalid dates
    df = df.dropna(subset=["date_time"])

    if len(df) == 0:
        logger.error("No valid data remaining after date parsing!")
        raise ValueError("No valid data remaining after date parsing")

    df = df.sort_values("date_time").reset_index(drop=True)

    # Convert numeric columns to proper dtypes before saving to Parquet
    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_columns:
        if col in df.columns:
            if HAS_CUDF:
                df[col] = cudf.to_numeric(df[col], errors='coerce')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows where numeric conversion failed
    df = df.dropna(subset=numeric_columns)

    # Additional null checking and cleaning
    logger.info("Performing comprehensive null data cleaning...")

    # Replace any remaining infinite values
    if HAS_CUDF:
        import cupy as cp
        df = df.replace([cp.inf, -cp.inf], None)
    else:
        df = df.replace([np.inf, -np.inf], np.nan)

    # Fill any remaining nulls with appropriate defaults
    for col in numeric_columns:
        if col == "Volume":
            df[col] = df[col].fillna(1000)  # Default volume
        else:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Final null check
    if HAS_CUDF:
        null_counts = df.isnull().sum()
    else:
        null_counts = df.isna().sum()

    total_nulls = null_counts.sum()
    if total_nulls > 0:
        logger.warning(f"Found {total_nulls} null values, performing final cleaning")
        df = df.fillna(0)

    # Validate final data
    df = df.dropna()

    if len(df) == 0:
        logger.error("No valid data remaining after comprehensive cleaning!")
        raise ValueError("No valid data remaining after comprehensive cleaning")

    logger.info(f"Final clean data: {len(df)} rows with no null values")

    logger.info("Caching combined data...")
    try:
        df.to_parquet(cache_file)
    except Exception as e:
        logging.warning(f"Failed to cache data: {e}")
    logger.info("Cached combined data to %s", cache_file)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Feature Engineering (GPU)
# ──────────────────────────────────────────────────────────────────────────────

def feature_engineering_gpu(
    df,
    bin_size_min: int = BIN_SIZE_MINUTES,
    window_ma: int = WINDOW_MA,
    window_rsi: int = WINDOW_RSI,
    window_vol: int = WINDOW_VOL,
) -> tuple:
    """
    Vectorized GPU feature engineering:
      - pct-returns, moving average, RSI, volatility
      - cyclical sin/cos for minute-of-day & weekday
      - one-hot weekday (7 cols) & time_bin cols
    Returns modified df and the segment_dict for inference reuse.
    """
    # Comprehensive null and infinite value cleaning
    logger.info("Starting comprehensive data validation and cleaning...")

    # First, handle all numeric columns comprehensively
    numeric_base_cols = ["Open", "High", "Low", "Close", "Volume"]

    for col in numeric_base_cols:
        if col in df.columns:
            # Replace infinites with NaN
            if HAS_CUDF:
                df[col] = df[col].replace([float('inf'), float('-inf')], None)
            else:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

            # Forward fill, then backward fill, then use median
            if HAS_CUDF:
                df[col] = df[col].ffill().bfill()
            else:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            if df[col].isna().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    # If median is also NaN, use a reasonable default
                    if col == "Volume":
                        median_val = 1000
                    else:
                        median_val = 100.0  # Default price
                df[col] = df[col].fillna(median_val)

    # Ensure Close prices are specifically validated
    if HAS_CUDF:
        df["Close"] = df["Close"].ffill().bfill()
        df["Close"] = df["Close"].replace([float('inf'), float('-inf')], None).fillna(df["Close"].median())
    else:
        df["Close"] = df["Close"].fillna(method='ffill').fillna(method='bfill')
        df["Close"] = df["Close"].replace([np.inf, -np.inf], np.nan).fillna(df["Close"].median())

    # Final check for Close column
    if df["Close"].isna().any():
        logger.warning("Close column still has NaN values, using default price")
        df["Close"] = df["Close"].fillna(100.0)

    # Calculate returns with robust NaN handling
    df["return"] = df["Close"].pct_change(periods=WINDOW_RETURNS).fillna(0)
    if HAS_CUDF:
        df["return"] = df["return"].replace([float('inf'), float('-inf')], 0)
    else:
        df["return"] = df["return"].replace([np.inf, -np.inf], 0)
    df["return"] = df["return"].clip(-0.5, 0.5)  # Clamp extreme returns

    # Moving average with robust handling
    df["ma_10"] = df["Close"].rolling(window=window_ma, min_periods=1).mean()
    df["ma_10"] = df["ma_10"].fillna(df["Close"]).bfill()

    # RSI calculation with division by zero protection
    delta = df["Close"].diff().fillna(0)
    gain = delta.where(delta > 0, 0).rolling(window=window_rsi, min_periods=1).mean()
    loss = (-delta).where(delta < 0, 0).rolling(window=window_rsi, min_periods=1).mean()

    # Prevent division by zero in RSI calculation
    loss = loss.where(loss > 1e-10, 1e-10)  # Minimum threshold
    rs = gain / loss
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50.0).clip(0, 100)

    # Volatility with robust handling
    df["volatility"] = df["return"].rolling(window=window_vol, min_periods=1).std()
    df["volatility"] = df["volatility"].fillna(0.01).clip(0, 1.0)  # Reasonable bounds

    dt       = df["date_time"]
    minutes  = (dt.dt.hour * 60 + dt.dt.minute).astype("int32")
    weekday  = dt.dt.weekday.astype("int32")

    # Always use CPU for trigonometric operations to avoid CUDA driver issues
    try:
        # Convert cuDF/pandas series to numpy arrays for consistent CPU processing
        if HAS_CUDF and hasattr(minutes, 'to_pandas'):
            minutes_np = minutes.to_pandas().values
            weekday_np = weekday.to_pandas().values
        else:
            minutes_np = minutes.values if hasattr(minutes, 'values') else np.array(minutes)
            weekday_np = weekday.values if hasattr(weekday, 'values') else np.array(weekday)

        # Ensure arrays are valid
        minutes_np = np.nan_to_num(minutes_np, nan=0.0)
        weekday_np = np.nan_to_num(weekday_np, nan=0.0)

        # Perform trigonometric operations on CPU
        sin_time_vals = np.sin(2 * np.pi * (minutes_np / SECONDS_IN_DAY))
        cos_time_vals = np.cos(2 * np.pi * (minutes_np / SECONDS_IN_DAY))
        sin_weekday_vals = np.sin(2 * np.pi * (weekday_np / WEEKDAYS))
        cos_weekday_vals = np.cos(2 * np.pi * (weekday_np / WEEKDAYS))

        # Ensure no NaN values in trigonometric results
        sin_time_vals = np.nan_to_num(sin_time_vals, nan=0.0)
        cos_time_vals = np.nan_to_num(cos_time_vals, nan=1.0)
        sin_weekday_vals = np.nan_to_num(sin_weekday_vals, nan=0.0)
        cos_weekday_vals = np.nan_to_num(cos_weekday_vals, nan=1.0)

        # Assign back to dataframe
        df["sin_time"] = sin_time_vals
        df["cos_time"] = cos_time_vals
        df["sin_weekday"] = sin_weekday_vals
        df["cos_weekday"] = cos_weekday_vals

    except Exception as e:
        logger.warning(f"Trigonometric operations failed: {e}, using fallback")
        # Final fallback with basic values
        df["sin_time"] = 0.0
        df["cos_time"] = 1.0
        df["sin_weekday"] = 0.0
        df["cos_weekday"] = 1.0

    total_bins = SECONDS_IN_DAY // bin_size_min
    time_bin   = (minutes // bin_size_min).astype("int32")
    segment_dict = {int((d * total_bins + t)): (d * total_bins + t)
                    for d in range(WEEKDAYS) for t in range(total_bins)}

    if HAS_CUDF:
        ohe_wd = cudf.get_dummies(weekday, prefix="wd")
        ohe_tb = cudf.get_dummies(time_bin, prefix="tb")
        df = cudf.concat([df, ohe_wd, ohe_tb], axis=1)
    else:
        ohe_wd = pd.get_dummies(weekday, prefix="wd")
        ohe_tb = pd.get_dummies(time_bin, prefix="tb")
        df = pd.concat([df, ohe_wd, ohe_tb], axis=1)

    # Final comprehensive null check and cleaning
    logger.info("Performing final null validation...")

    # Check for any remaining null values
    if HAS_CUDF:
        null_counts = df.isnull().sum()
    else:
        null_counts = df.isna().sum()

    total_nulls = null_counts.sum()
    if total_nulls > 0:
        logger.warning(f"Found {total_nulls} null values after feature engineering, cleaning...")

        # Fill remaining nulls with appropriate defaults
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                if 'return' in col.lower():
                    df[col] = df[col].fillna(0.0)
                elif 'rsi' in col.lower():
                    df[col] = df[col].fillna(50.0)
                elif 'volatility' in col.lower():
                    df[col] = df[col].fillna(0.01)
                elif any(x in col.lower() for x in ['sin', 'cos']):
                    if 'sin' in col.lower():
                        df[col] = df[col].fillna(0.0)
                    else:
                        df[col] = df[col].fillna(1.0)
                else:
                    df[col] = df[col].fillna(0.0)

    # Drop any rows that still have nulls
    df = df.dropna()

    logger.info(f"Feature engineering completed: {len(df)} rows, no null values")

    return df, segment_dict

# ──────────────────────────────────────────────────────────────────────────────
# Scaling & Splitting (GPU)
# ──────────────────────────────────────────────────────────────────────────────

def scale_and_split_gpu(
    df,
    numeric_cols: list = None,
    test_size: float = 0.2,
    random_state: int = NUMPY_RANDOM_SEED
):
    """
    On-GPU standard scaling via cuML + train/test split.
    Returns train_df, test_df, fitted_scaler.
    """
    if numeric_cols is None:
        numeric_cols = ["Open","High","Low","Close","Volume","return","ma_10"]

    logger.info("Starting scaling with null validation...")

    # Comprehensive null checking before scaling
    initial_count = len(df)
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    logger.info(f"Removed {initial_count - len(df)} rows with nulls in numeric columns")

    if len(df) == 0:
        raise ValueError("No data remaining after null removal")

    # Additional validation for infinite values
    for col in numeric_cols:
        if col in df.columns:
            if HAS_CUDF:
                df[col] = df[col].replace([float('inf'), float('-inf')], None).dropna()
            else:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).dropna()

    # Final null check
    df = df.dropna().reset_index(drop=True)
    logger.info(f"Final scaling dataset: {len(df)} rows")

    raw_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in raw_cols:
        df[f"{col}_raw"] = df[col].astype("float64")

    X = df[numeric_cols]
    scaler = CuStandardScaler()
    X_scaled = scaler.fit_transform(X)
    df[numeric_cols] = X_scaled

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    return train_df, test_df, scaler

# ──────────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────────

def create_environment_data(
    data_folder: str,
    max_rows: int | None = None,
    test_size: float = 0.2,
    cache_folder: str = './cached_data',
    chunk_size: int = 500000,  # Process 500K rows at a time
    stream_processing: bool = False  # Enable for very large datasets
):
    """
    Args:
      data_folder: path to raw .txt files
      max_rows: take only the last N rows
      test_size: fraction of data to reserve for the test split
      cache_folder: where to store caches
      chunk_size: number of rows to process at once for memory efficiency
    Returns:
      train_df, test_df, scaler, segment_dict
    """
    # Enable streaming for very large datasets
    if max_rows and max_rows > 10_000_000:
        stream_processing = True
        logger.info(f"Enabling streaming processing for {max_rows} rows")

    # For streaming, we process data in smaller batches to avoid memory issues
    if stream_processing:
        return _create_streaming_environment_data(
            data_folder, max_rows, test_size, cache_folder, chunk_size
        )

    df = load_and_cache_data(data_folder, cache_folder)
    total_rows = len(df)

    if max_rows is not None and max_rows > 0:
        df = df.iloc[-max_rows:].reset_index(drop=True)
        total_rows = len(df)

    logger.info(f"Processing {total_rows} rows in chunks of {chunk_size}")

    # Process in chunks to avoid memory issues
    if total_rows > chunk_size:
        processed_chunks = []
        scaler = None

        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = df.iloc[start_idx:end_idx].copy()

            logger.info(f"Processing chunk {start_idx//chunk_size + 1}/{(total_rows + chunk_size - 1)//chunk_size}")

            chunk, segment_dict = feature_engineering_gpu(chunk)

            # Fit scaler on first chunk, transform on subsequent chunks
            if scaler is None:
                # Just fit the scaler without splitting (we'll split later)
                numeric_cols = ["Open","High","Low","Close","Volume","return","ma_10"]
                chunk = chunk.dropna(subset=numeric_cols).reset_index(drop=True)

                raw_cols = ["Open", "High", "Low", "Close", "Volume"]
                for col in raw_cols:
                    chunk[f"{col}_raw"] = chunk[col].astype("float64")

                X = chunk[numeric_cols]
                scaler = CuStandardScaler()
                X_scaled = scaler.fit_transform(X)
                chunk[numeric_cols] = X_scaled
            else:
                numeric_cols = ["Open","High","Low","Close","Volume","return","ma_10"]
                chunk = chunk.dropna(subset=numeric_cols).reset_index(drop=True)

                raw_cols = ["Open", "High", "Low", "Close", "Volume"]
                for col in raw_cols:
                    chunk[f"{col}_raw"] = chunk[col].astype("float64")

                X = chunk[numeric_cols]
                X_scaled = scaler.transform(X)
                chunk[numeric_cols] = X_scaled

            processed_chunks.append(chunk)

            # Clear memory
            del chunk

        # Combine all chunks
        df = cudf.concat(processed_chunks, ignore_index=True)
        del processed_chunks

        # Final train/test split
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    else:
        df, segment_dict = feature_engineering_gpu(df)
        train_df, test_df, scaler = scale_and_split_gpu(df, test_size=test_size)

    return train_df, test_df, scaler, segment_dict
# Add dummy implementations for read_file_chunked and read_file_chunked_pandas
def read_file_chunked(file_path):
    """Dummy function to simulate chunked file reading."""
    import pandas as pd
    import cudf
    try:
        df = cudf.read_csv(file_path)
        return [df]
    except:
        df = pd.read_csv(file_path)
        return [cudf.DataFrame.from_pandas(df)]



def _create_streaming_environment_data(
    data_folder: str,
    max_rows: int,
    test_size: float,
    cache_folder: str,
    chunk_size: int
):
    """
    Streaming data processing for massive datasets (100M+ rows)
    Processes data in chunks and saves intermediate results to avoid memory issues
    """
    logger.info("Starting streaming data processing for massive dataset")

    # Create temporary directory for streaming chunks
    import tempfile
    temp_dir = os.path.join(cache_folder, "streaming_chunks")
    os.makedirs(temp_dir, exist_ok=True)

    # Process data in streaming chunks
    processed_chunks = []
    scaler = None
    segment_dict = None

    # Load data in smaller chunks to avoid memory explosion
    streaming_chunk_size = min(chunk_size, 100_000)  # Even smaller chunks for streaming

    try:
        files = glob.glob(os.path.join(data_folder, "*.txt"))
        if not files:
            raise FileNotFoundError(f"No data files found in {data_folder}")

        total_processed = 0

        for file_idx, file_path in enumerate(files):
            logger.info(f"Streaming processing file {file_idx + 1}/{len(files)}: {os.path.basename(file_path)}")

            # Read file in chunks using pandas (more memory efficient than cuDF for streaming)
            chunk_reader = pd.read_csv(
                file_path,
                names=["date_time", "Open", "High", "Low", "Close", "Volume"],
                header=None,
                chunksize=streaming_chunk_size
            )

            for chunk_idx, raw_chunk in enumerate(chunk_reader):
                if max_rows and total_processed >= max_rows:
                    break

                try:
                    # Basic data cleaning
                    raw_chunk["date_time"] = pd.to_datetime(raw_chunk["date_time"], errors='coerce')
                    raw_chunk = raw_chunk.dropna(subset=["date_time"])

                    if len(raw_chunk) == 0:
                        continue

                    # Convert to cuDF if available for faster processing
                    if HAS_CUDF:
                        try:
                            chunk = cudf.DataFrame.from_pandas(raw_chunk)
                        except Exception:
                            chunk = raw_chunk
                    else:
                        chunk = raw_chunk

                    # Feature engineering
                    chunk, seg_dict = feature_engineering_gpu(chunk)
                    if segment_dict is None:
                        segment_dict = seg_dict

                    # Scaling - fit on first chunk, transform on rest
                    numeric_cols = ["Open","High","Low","Close","Volume","return","ma_10"]
                    chunk = chunk.dropna(subset=numeric_cols).reset_index(drop=True)

                    if len(chunk) == 0:
                        continue

                    # Add raw columns
                    raw_cols = ["Open", "High", "Low", "Close", "Volume"]
                    for col in raw_cols:
                        if col in chunk.columns:
                            chunk[f"{col}_raw"] = chunk[col].astype("float64")

                    if scaler is None:
                        # Fit scaler on first chunk
                        scaler = CuStandardScaler()
                        X = chunk[numeric_cols]
                        X_scaled = scaler.fit_transform(X)
                        chunk[numeric_cols] = X_scaled
                        logger.info("Fitted scaler on first data chunk")
                    else:
                        # Transform subsequent chunks
                        X = chunk[numeric_cols]
                        X_scaled = scaler.transform(X)
                        chunk[numeric_cols] = X_scaled

                    # Save chunk to temporary file
                    chunk_file = os.path.join(temp_dir, f"chunk_{file_idx}_{chunk_idx}.parquet")

                    # Convert back to pandas for saving if needed
                    if hasattr(chunk, 'to_pandas'):
                        chunk.to_pandas().to_parquet(chunk_file, index=False)
                    else:
                        chunk.to_parquet(chunk_file, index=False)

                    processed_chunks.append(chunk_file)
                    total_processed += len(chunk)

                    if chunk_idx % 10 == 0:
                        logger.info(f"Processed {total_processed} rows, saved {len(processed_chunks)} chunks")

                    # Clear memory
                    del chunk, raw_chunk

                except Exception as e:
                    logger.warning(f"Error processing chunk {chunk_idx} from file {file_path}: {e}")
                    continue

                if max_rows and total_processed >= max_rows:
                    break

            if max_rows and total_processed >= max_rows:
                break

        logger.info(f"Streaming processing complete: {total_processed} rows in {len(processed_chunks)} chunks")

        # Now combine chunks and split into train/test
        logger.info("Combining processed chunks for train/test split...")

        # Determine split point
        split_idx = int(len(processed_chunks) * (1 - test_size))
        train_chunk_files = processed_chunks[:split_idx]
        test_chunk_files = processed_chunks[split_idx:]

        # Combine train chunks
        if train_chunk_files:
            train_dfs = []
            for chunk_file in train_chunk_files:
                df_chunk = pd.read_parquet(chunk_file)
                train_dfs.append(df_chunk)

            train_df = pd.concat(train_dfs, ignore_index=True)
            del train_dfs
        else:
            raise ValueError("No training data chunks available")

        # Combine test chunks
        if test_chunk_files:
            test_dfs = []
            for chunk_file in test_chunk_files:
                df_chunk = pd.read_parquet(chunk_file)
                test_dfs.append(df_chunk)

            test_df = pd.concat(test_dfs, ignore_index=True)
            del test_dfs
        else:
            raise ValueError("No test data chunks available")

        logger.info(f"Final streaming result: {len(train_df)} train, {len(test_df)} test rows")

        # Clean up temporary files
        try:
            import shutil
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary streaming files")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")

        return train_df, test_df, scaler, segment_dict

    except Exception as e:
        logger.error(f"Streaming processing failed: {e}")
        # Clean up on failure
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass
        raise


def read_file_chunked_pandas(file_path):
    """Pandas fallback for chunked file reading with proper column names."""
    import pandas as pd
    chunks = []
    for chunk in pd.read_csv(
        file_path, 
        chunksize=10000,
        names=["date_time", "Open", "High", "Low", "Close", "Volume"],
        header=None
    ):
        # Convert datetime column directly since we set the name explicitly
        chunk["date_time"] = pd.to_datetime(chunk["date_time"], errors='coerce')
        chunk = chunk.dropna(subset=["date_time"])
        if len(chunk) > 0:
            chunks.append(chunk)
    return chunks
# ─── FEATURE ENGINEERING ───────────────────────────────────────────────────

def feature_engineering(df, has_cudf):
    logging.info("Computing technical indicators...")

    # Fill any NaN values in close prices first
    df["close"] = df["close"].fillna(method='ffill').fillna(method='bfill').fillna(0)

    # moving averages
    df["ma_5"]  = df["close"].rolling(5, min_periods=1).mean()
    df["ma_20"] = df["close"].rolling(20, min_periods=1).mean()

    # RSI with better NaN handling
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()

    # Prevent division by zero
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Fill any remaining NaN values in RSI
    df["rsi"] = df["rsi"].fillna(50.0)  # Neutral RSI value

    # bollinger bands
    bb_period = 20
    bb_std = 2
    df["bb_mid"] = df["close"].rolling(bb_period).mean()
    bb_std_val = df["close"].rolling(bb_period).std()
    df["bb_upper"] = df["bb_mid"] + (bb_std_val * bb_std)
    df["bb_lower"] = df["bb_mid"] - (bb_std_val * bb_std)

    # volatility
    df["volatility"] = df["close"].pct_change().rolling(10).std()

    # volume indicators (if volume exists)
    if "volume" in df.columns:
        df["volume_ma"] = df["volume"].rolling(10).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]
    else:
        df["volume_ma"] = 1000
        df["volume_ratio"] = 1.0

    # returns & lagged features
    df["return"] = df["close"].pct_change()
    df["return_lag1"] = df["return"].shift(1)
    df["return_lag2"] = df["return"].shift(2)

    # ─── NaN CLEANING ───────────────────────────────────────────────────────────
    logging.info("Cleaning NaN values...")

    # Replace infinite values with NaN first
    if has_cudf:
        import cupy as cp
        df = df.replace([cp.inf, -cp.inf], None)
    else:
        df = df.replace([np.inf, -np.inf], np.nan)

    # Forward fill then backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')

    # If still NaN, fill with reasonable defaults
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == 'rsi':
            df[col] = df[col].fillna(50.0)  # Neutral RSI
        elif 'return' in col:
            df[col] = df[col].fillna(0.0)   # Zero returns
        elif 'ratio' in col:
            df[col] = df[col].fillna(1.0)   # Neutral ratios
        elif 'price' in col or col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].fillna(df[col].median())  # Median price
        else:
            df[col] = df[col].fillna(0.0)   # Default to zero

    # Verify no NaN values remain
    if has_cudf:
        nan_counts = df.isnull().sum()
    else:
        nan_counts = df.isna().sum()

    total_nans = nan_counts.sum()
    if total_nans > 0:
        logging.warning(f"Still have {total_nans} NaN values after cleaning")
        # Final aggressive cleaning
        df = df.fillna(0.0)

    logging.info("NaN cleaning completed")
    return df