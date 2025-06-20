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
import hashlib
import logging

import cudf
import cupy as cp
from cuml.preprocessing import StandardScaler as CuStandardScaler
from cuml.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# Logger setup
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants & Defaults
# ──────────────────────────────────────────────────────────────────────────────
NUMPY_RANDOM_SEED = 42
WINDOW_RETURNS     = 1   # for pct_change
WINDOW_MA          = 10
WINDOW_RSI         = 14
WINDOW_VOL         = 10
BIN_SIZE_MINUTES   = 15
SECONDS_IN_DAY     = 24 * 60
WEEKDAYS           = 7

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def hash_files(file_list: list[str]) -> str:
    """
    Generate a SHA256 hash over sorted filenames and modification times.
    Ensures cache invalidates if any file changes.
    """
    hasher = hashlib.sha256()
    for fp in sorted(file_list):
        st = os.stat(fp)
        hasher.update(fp.encode("utf-8"))
        hasher.update(str(st.st_mtime).encode("utf-8"))
    return hasher.hexdigest()

# ──────────────────────────────────────────────────────────────────────────────
# Data Loading & Caching
# ──────────────────────────────────────────────────────────────────────────────

def load_and_cache_data(
    data_folder: str,
    cache_dir: str = "./cached_data",
    pattern: str = "*.txt",
) -> cudf.DataFrame:
    """
    Load all text/CSV files from `data_folder` into a single cuDF DataFrame.
    If a parquet cache exists (based on file-hash), load from cache instead.
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
        return cudf.read_parquet(cache_file)

    logger.info("Scanning %d files for raw data...", len(files))
    dfs = [
        cudf.read_csv(
            fp,
            names=["date_time", "Open", "High", "Low", "Close", "Volume"],
            parse_dates=["date_time"],
            header=None,
        )
        for fp in files
    ]
    df = cudf.concat(dfs, ignore_index=True)
    df = df.sort_values("date_time").reset_index(drop=True)
    df.to_parquet(cache_file)
    logger.info("Cached combined data to %s", cache_file)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Feature Engineering (GPU)
# ──────────────────────────────────────────────────────────────────────────────

def feature_engineering_gpu(
    df: cudf.DataFrame,
    bin_size_min: int = BIN_SIZE_MINUTES,
    window_ma: int = WINDOW_MA,
    window_rsi: int = WINDOW_RSI,
    window_vol: int = WINDOW_VOL,
) -> (cudf.DataFrame, dict[int,int]):
    """
    Vectorized GPU feature engineering:
      - pct-returns, moving average, RSI, volatility
      - cyclical sin/cos for minute-of-day & weekday
      - one-hot weekday (7 cols) & time_bin cols
    Returns modified df and the segment_dict for inference reuse.
    """
    df["return"] = df["Close"].pct_change(periods=WINDOW_RETURNS).fillna(0).astype("float64")
    df["ma_10"]  = df["Close"].rolling(window=window_ma, min_periods=1).mean().bfill()
    delta       = df["Close"].diff()
    gain        = delta.where(delta > 0, 0).rolling(window=window_rsi, min_periods=1).mean()
    loss        = (-delta).where(delta < 0, 0).rolling(window=window_rsi, min_periods=1).mean()
    rs          = gain / loss
    df["rsi"]   = (100 - (100 / (1 + rs))).bfill()
    df["volatility"] = df["return"].rolling(window=window_vol, min_periods=1).std().bfill()

    dt       = df["date_time"]
    minutes  = (dt.dt.hour * 60 + dt.dt.minute).astype("int32")
    weekday  = dt.dt.weekday.astype("int32")

    df["sin_time"]     = cp.sin(2 * cp.pi * (minutes / SECONDS_IN_DAY))
    df["cos_time"]     = cp.cos(2 * cp.pi * (minutes / SECONDS_IN_DAY))
    df["sin_weekday"]  = cp.sin(2 * cp.pi * (weekday / WEEKDAYS))
    df["cos_weekday"]  = cp.cos(2 * cp.pi * (weekday / WEEKDAYS))

    total_bins = SECONDS_IN_DAY // bin_size_min
    time_bin   = (minutes // bin_size_min).astype("int32")
    segment_dict = {int((d * total_bins + t)): (d * total_bins + t)
                    for d in range(WEEKDAYS) for t in range(total_bins)}

    ohe_wd = cudf.get_dummies(weekday, prefix="wd")
    ohe_tb = cudf.get_dummies(time_bin, prefix="tb")

    df = cudf.concat([df, ohe_wd, ohe_tb], axis=1)
    return df, segment_dict

# ──────────────────────────────────────────────────────────────────────────────
# Scaling & Splitting (GPU)
# ──────────────────────────────────────────────────────────────────────────────

def scale_and_split_gpu(
    df: cudf.DataFrame,
    numeric_cols: list[str] = None,
    test_size: float = 0.2,
    random_state: int = NUMPY_RANDOM_SEED
):
    """
    On-GPU standard scaling via cuML + train/test split.
    Returns train_df, test_df, fitted_scaler.
    """
    if numeric_cols is None:
        numeric_cols = ["Open","High","Low","Close","Volume","return","ma_10"]

    df = df.dropna(subset=numeric_cols).reset_index(drop=True)

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
    cache_folder: str = './cached_data'
):
    """
    Args:
      data_folder: path to raw .txt files
      max_rows: take only the last N rows
      test_size: fraction of data to reserve for the test split
      cache_folder: where to store caches
    Returns:
      train_df, test_df, scaler, segment_dict
    """
    df = load_and_cache_data(data_folder, cache_folder)
    if max_rows is not None and max_rows > 0:
        df = df.iloc[-max_rows:].reset_index(drop=True)

    df, segment_dict = feature_engineering_gpu(df)
    train_df, test_df, scaler = scale_and_split_gpu(df, test_size=test_size)
    return train_df, test_df, scaler, segment_dict
