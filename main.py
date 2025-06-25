
#!/usr/bin/env python
import os
import argparse
import logging
from logging.handlers import RotatingFileHandler
import sys
import warnings
import pickle
import collections
import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

logging.getLogger("torch.distributed").setLevel(logging.WARNING)
logging.getLogger("torch.distributed").setLevel(logging.INFO)

# ─── TRY GPU DATAFRAME SUPPORT ─────────────────────────────────────────────────
has_cudf = False
cudf = None

# Check if we should even try GPU libraries
try_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0

if try_gpu:
    try:
        # Test CUDA functionality before importing GPU libraries
        torch.cuda.current_device()
        import cudf
        import cupy as cp
        # Test basic cupy operation to ensure driver compatibility
        test_array = cp.array([1, 2, 3])
        _ = cp.sum(test_array)
        has_cudf = True
        logging.info(f"cudf loaded successfully with {torch.cuda.device_count()} GPUs available")
    except (ImportError, AttributeError, RuntimeError, Exception) as e:
        # Handle various CUDA-related import errors
        if any(keyword in str(e).lower() for keyword in ["cuda", "numba", "driver", "insufficient"]):
            warnings.warn(f"cudf import failed due to CUDA/driver issues ({e}); falling back to pandas.")
        else:
            warnings.warn(f"cudf import failed ({e}); falling back to pandas.")
        has_cudf = False

if not has_cudf:
    import pandas as pd
    sys.modules['cudf'] = pd
    cudf = pd

# ─── OPTIONAL: load cuml.StandardScaler so scaler.transform() is type‐checked ──
try:
    from cuml.preprocessing import StandardScaler as CuStandardScaler
except (ImportError, AttributeError, RuntimeError) as e:
    # Handle various CUDA-related import errors
    if "cuda" in str(e).lower() or "numba" in str(e).lower():
        warnings.warn(f"cuML import failed due to CUDA/numba issues ({e}); using CPU fallback.")
    else:
        warnings.warn(f"cuML import failed ({e}); using CPU fallback.")
    CuStandardScaler = None

# ─── LOCAL IMPORTS ─────────────────────────────────────────────────────────────
from data_preprocessing import create_environment_data
from ga_policy_evolution import run_ga_evolution, PolicyNetwork
from policy_gradient_methods import PPOTrainer, ActorCriticNet
from utils import (
    evaluate_agent_distributed, compute_performance_metrics, 
    setup_logging, build_states_for_futures_env, process_live_row
)
from futures_env import FuturesEnv, TimeSeriesState

# ──────────────────────────────────────────────────────────────────────────────
# GLOBALS FOR LIVE‐INFERENCE BUFFERING & FEATURE SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
WINDOW_MA      = 10
WINDOW_RSI     = 14
WINDOW_VOL     = 10
BIN_SIZE       = 15
SECONDS_IN_DAY = 24 * 60

history = {
    "closes":  collections.deque(maxlen=WINDOW_MA),
    "deltas":  collections.deque(maxlen=WINDOW_RSI),
    "returns": collections.deque(maxlen=WINDOW_VOL),
}

scaler = None
segment_dict: dict[int,int] = {}


def parse_arguments():
    """Parse command line arguments for professional configuration."""
    parser = argparse.ArgumentParser(
        description="Professional Trading System with GA and PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data configuration
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--data-folder', type=str, default='./data_txt',
                           help='Path to folder containing OHLCV data files')
    data_group.add_argument('--cache-folder', type=str, default='./cached_data',
                           help='Path to folder for caching processed data')
    data_group.add_argument('--data-percentage', type=float, default=1.0,
                           help='Percentage of data to use (0.0-1.0)')
    data_group.add_argument('--max-rows', type=int, default=0,
                           help='Maximum number of rows to process (0 for all)')
    data_group.add_argument('--chunk-size', type=int, default=500000,
                           help='Chunk size for processing large datasets')
    data_group.add_argument('--streaming-threshold', type=int, default=10000000,
                           help='Row threshold to enable streaming processing')
    data_group.add_argument('--memory-limit-gb', type=float, default=8.0,
                           help='Memory limit in GB for processing')
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--models-dir', type=str, default='./models',
                            help='Directory to save/load models')
    model_group.add_argument('--ga-model-name', type=str, default='ga_policy_model.pth',
                            help='Filename for GA model')
    model_group.add_argument('--ppo-model-name', type=str, default='ppo_model.pth',
                            help='Filename for PPO model')
    model_group.add_argument('--checkpoint-interval', type=int, default=10,
                            help='Save checkpoint every N updates')
    model_group.add_argument('--backup-count', type=int, default=5,
                            help='Number of model backups to keep')
    
    # Training configuration
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--training-mode', type=str, default='adaptive',
                            choices=['adaptive', 'ga_only', 'ppo_only', 'sequential'],
                            help='Training mode: adaptive switching, GA only, PPO only, or sequential GA then PPO')
    train_group.add_argument('--adaptive-iterations', type=int, default=20,
                            help='Number of adaptive training iterations')
    train_group.add_argument('--stagnation-threshold', type=int, default=5,
                            help='Switch to GA after N iterations without improvement')
    train_group.add_argument('--poor-performance-threshold', type=int, default=3,
                            help='Switch to GA after N iterations of poor performance')
    train_group.add_argument('--total-steps', type=int, default=1000000,
                            help='Total training steps (for non-adaptive modes)')
    train_group.add_argument('--ga-population', type=int, default=80,
                            help='GA population size')
    train_group.add_argument('--ga-generations', type=int, default=100,
                            help='GA number of generations')
    train_group.add_argument('--ppo-lr', type=float, default=3e-4,
                            help='PPO learning rate')
    train_group.add_argument('--ppo-batch-size', type=int, default=64,
                            help='PPO batch size')
    train_group.add_argument('--eval-interval', type=int, default=10,
                            help='Evaluation interval')
    
    # Distributed training
    dist_group = parser.add_argument_group('Distributed Training')
    dist_group.add_argument('--max-train-per-rank', type=int, default=100000,
                           help='Maximum training samples per rank')
    dist_group.add_argument('--max-test-per-rank', type=int, default=20000,
                           help='Maximum test samples per rank')
    dist_group.add_argument('--nccl-timeout', type=int, default=1800000,
                           help='NCCL timeout in milliseconds')
    
    # Environment configuration
    env_group = parser.add_argument_group('Environment Configuration')
    env_group.add_argument('--value-per-tick', type=float, default=12.5,
                          help='Value per tick for trading environment')
    env_group.add_argument('--tick-size', type=float, default=0.25,
                          help='Tick size for trading environment')
    env_group.add_argument('--commission', type=float, default=0.0005,
                          help='Commission per trade')
    env_group.add_argument('--margin-rate', type=float, default=0.01,
                          help='Margin rate')
    
    # Logging and output
    log_group = parser.add_argument_group('Logging Configuration')
    log_group.add_argument('--log-dir', type=str, default='./logs',
                          help='Directory for log files')
    log_group.add_argument('--log-level', type=str, default='INFO',
                          choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                          help='Logging level')
    log_group.add_argument('--tensorboard-dir', type=str, default='./runs',
                          help='Directory for TensorBoard logs')
    
    return parser.parse_args()


def setup_model_directories(args):
    """Setup model directories with proper structure."""
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different model types
    subdirs = ['ga_models', 'ppo_models', 'checkpoints', 'backups']
    for subdir in subdirs:
        (models_dir / subdir).mkdir(exist_ok=True)
    
    return {
        'base_dir': models_dir,
        'ga_dir': models_dir / 'ga_models',
        'ppo_dir': models_dir / 'ppo_models',
        'checkpoint_dir': models_dir / 'checkpoints',
        'backup_dir': models_dir / 'backups'
    }


def save_model_with_backup(model, save_path, backup_dir, backup_count=5):
    """Save model with automatic backup rotation."""
    save_path = Path(save_path)
    backup_dir = Path(backup_dir)
    
    # Create backup if model already exists
    if save_path.exists():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{save_path.stem}_{timestamp}.pth"
        backup_path = backup_dir / backup_name
        
        # Copy existing model to backup
        import shutil
        shutil.copy2(save_path, backup_path)
        
        # Remove old backups (keep only backup_count)
        backup_files = sorted(backup_dir.glob(f"{save_path.stem}_*.pth"))
        if len(backup_files) > backup_count:
            for old_backup in backup_files[:-backup_count]:
                old_backup.unlink()
    
    # Save new model
    if hasattr(model, 'save_model'):
        model.save_model(str(save_path))
    else:
        torch.save(model.state_dict(), save_path)
    
    logging.info(f"Model saved to {save_path} with backup")


def load_model_safely(model, model_path, model_name="model"):
    """Load model with error handling and fallback."""
    model_path = Path(model_path)
    
    if not model_path.exists():
        logging.info(f"No {model_name} found at {model_path}, starting from scratch")
        return False
    
    try:
        if hasattr(model, 'load_model'):
            model.load_model(str(model_path))
        else:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        logging.info(f"Successfully loaded {model_name} from {model_path}")
        return True
    except Exception as e:
        logging.warning(f"Failed to load {model_name} from {model_path}: {e}")
        return False


def main():
    """Enhanced main function with professional argument handling."""
    args = parse_arguments()
    
    # ─── DYNAMIC NCCL TIMEOUT & 4 GPU OPTIMIZATION ─────────────────────────────
    os.environ["NCCL_TIMEOUT"] = str(args.nccl_timeout)
    
    # Optimize for 4 GPU NVLink setup
    if torch.cuda.device_count() >= 4:
        os.environ["NCCL_P2P_DISABLE"] = "0"  # Enable P2P for NVLink
        os.environ["NCCL_NET_GDR_DISABLE"] = "0"  # Enable GPU Direct RDMA
        os.environ["NCCL_TREE_THRESHOLD"] = "0"  # Use ring for small data
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Ensure all GPUs visible
        os.environ["NCCL_DEBUG"] = "INFO"  # Enable NCCL debugging
        logging.info("Enabled NVLink optimizations for 4 GPU setup")
        
        # Check if we're in distributed mode
        if "LOCAL_RANK" not in os.environ:
            logging.warning("Not running in distributed mode! Use 'torchrun --nproc_per_node=4' to utilize all GPUs")
            logging.warning("Current single-process mode will only use 1 GPU")
        else:
            logging.info(f"Running in distributed mode with {torch.cuda.device_count()} GPUs")

    # torchrun / torch.distributed sets LOCAL_RANK
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Only set CUDA device if CUDA is available, has GPUs, and drivers are compatible
    use_cuda = False
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            # Test CUDA functionality
            torch.cuda.current_device()
            torch.cuda.set_device(local_rank)
            backend = "nccl"
            use_cuda = True
            logging.info(f"Using NCCL backend for GPU {local_rank}/{torch.cuda.device_count()}")
        except Exception as e:
            logging.warning(f"CUDA device initialization failed: {e}")
            backend = "gloo"
            logging.info("Falling back to Gloo backend for CPU training")
    else:
        backend = "gloo"
        logging.info("Using Gloo backend for CPU training")
    
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")

    # init process group with matching timeout
    try:
        if "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=datetime.timedelta(milliseconds=args.nccl_timeout)
            )
            world_size = dist.get_world_size()
            logging.info(f"Successfully initialized distributed training with {world_size} processes")
        else:
            raise RuntimeError("Distributed environment variables not set")
    except (RuntimeError, ValueError) as e:
        logging.warning(f"Distributed training not available ({e}), running in single-process mode")
        logging.warning("To use all 4 GPUs, run with: torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12355 main.py")
        world_size = 1
        local_rank = 0

    # Setup directories
    model_dirs = setup_model_directories(args)
    Path(args.cache_folder).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tensorboard_dir).mkdir(parents=True, exist_ok=True)

    # setup logging & report NCCL only once
    setup_logging(local_rank)
    if local_rank == 0:
        logging.info(f"NCCL_TIMEOUT = {args.nccl_timeout} ms")
        logging.info(f"Using {args.data_percentage*100:.1f}% of available data")
        logging.info(f"Models will be saved to: {args.models_dir}")
    
    # Device already set above
    logging.info(f"Rank {local_rank}/{world_size} starting on {device} (has_cudf={has_cudf})")

    # ─── DATA PREP ─────────────────────────────────────────────────────────────
    train_path = os.path.join(args.cache_folder, "train_data.parquet")
    test_path  = os.path.join(args.cache_folder, "test_data.parquet")

    # Handle large dataset optimization
    max_rows = args.max_rows if args.max_rows > 0 else None
    
    # Log actual parameters being used
    logging.info(f"Data loading parameters: max_rows={max_rows}, data_percentage={args.data_percentage}")
    
    # Only apply percentage if max_rows is set
    if max_rows and max_rows > 0 and args.data_percentage < 1.0:
        max_rows = int(max_rows * args.data_percentage)
        logging.info(f"Adjusted max_rows to {max_rows} based on data_percentage")
    
    # Intelligent chunk sizing based on data size
    if max_rows:
        if max_rows <= 1000:
            args.chunk_size = min(args.chunk_size, max_rows // 2)
        elif max_rows > 10_000_000:  # Large datasets
            args.chunk_size = 1_000_000  # 1M row chunks
            logging.info(f"Large dataset detected ({max_rows} rows), using 1M row chunks")
        elif max_rows > 1_000_000:   # Medium datasets  
            args.chunk_size = 500_000   # 500K row chunks
    else:
        # No row limit - assume very large dataset
        args.chunk_size = 1_000_000
        logging.info("No row limit specified, using 1M row chunks for memory efficiency")

    if local_rank == 0:
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            logging.info(f"Processing data with max_rows={max_rows}, chunk_size={args.chunk_size}")
            train_df, test_df, sc, seg = create_environment_data(
                data_folder=args.data_folder,
                max_rows=max_rows,
                test_size=0.2,
                cache_folder=args.cache_folder,
                chunk_size=args.chunk_size
            )

            logging.info("Saving processed data to compressed Parquet...")
            train_df.to_parquet(train_path, index=False, compression='snappy')
            test_df.to_parquet(test_path, index=False, compression='snappy')

            with open(os.path.join(args.cache_folder, "scaler.pkl"), "wb") as f:
                pickle.dump(sc, f)
            with open(os.path.join(args.cache_folder, "segment_dict.pkl"), "wb") as f:
                pickle.dump(seg, f)
            logging.info("Data cached to Parquet and artifacts saved.")
        else:
            logging.info("Parquet cache found; skipping preprocessing.")

    # robust barrier
    if world_size > 1:
        try:
            if torch.cuda.is_available():
                dist.barrier(device_ids=[local_rank])
            else:
                dist.barrier()
        except Exception as e:
            logging.warning(f"Barrier failed ({e}); continuing without synchronization")

    # load cached data
    if has_cudf:
        full_train = cudf.read_parquet(train_path)
        full_test  = cudf.read_parquet(test_path)
    else:
        import pandas as pd
        full_train = pd.read_parquet(train_path)
        full_test  = pd.read_parquet(test_path)

    with open(os.path.join(args.cache_folder, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(args.cache_folder, "segment_dict.pkl"), "rb") as f:
        segment_dict = pickle.load(f)

    # ─── SHARD FOR DDP ──────────────────────────────────────────────────────────
    n_train, n_test = len(full_train), len(full_test)
    logging.info(f"Total data: {n_train} train, {n_test} test rows")

    # For small datasets, ensure each GPU gets meaningful data
    if n_train <= 1000:
        per_t = max(n_train // world_size, 50)  # Minimum 50 rows per GPU
        per_v = max(n_test // world_size, 10)   # Minimum 10 rows per GPU
    else:
        per_t = min(n_train // world_size, args.max_train_per_rank)
        per_v = min(n_test // world_size, args.max_test_per_rank)

    # Use sampling for very large datasets
    if n_train > world_size * args.max_train_per_rank:
        train_indices = np.linspace(0, n_train-1, per_t * world_size, dtype=int)
        s_t, e_t = local_rank*per_t, (local_rank+1)*per_t
        train_slice = full_train.iloc[train_indices[s_t:e_t]]
        logging.info(f"Rank {local_rank}: Sampled {len(train_slice)} train rows from {n_train} total")
    else:
        s_t, e_t = local_rank*per_t, (local_rank+1)*per_t if local_rank<world_size-1 else n_train
        train_slice = full_train.iloc[s_t:e_t]

    if n_test > world_size * args.max_test_per_rank:
        test_indices = np.linspace(0, n_test-1, per_v * world_size, dtype=int)
        s_v, e_v = local_rank*per_v, (local_rank+1)*per_v
        test_slice = full_test.iloc[test_indices[s_v:e_v]]
        logging.info(f"Rank {local_rank}: Sampled {len(test_slice)} test rows from {n_test} total")
    else:
        s_v, e_v = local_rank*per_v, (local_rank+1)*per_v if local_rank<world_size-1 else n_test
        test_slice = full_test.iloc[s_v:e_v]

    # convert to pandas for .itertuples
    if has_cudf:
        train_pd = train_slice.to_pandas()
        test_pd  = test_slice.to_pandas()
    else:
        train_pd = train_slice.copy()
        test_pd  = test_slice.copy()

    train_pd.rename(columns={"return":"return_"}, inplace=True)
    test_pd .rename(columns={"return":"return_"}, inplace=True)

    train_states = build_states_for_futures_env(train_pd)
    test_states  = build_states_for_futures_env(test_pd)

    # ─── ENVIRONMENTS ──────────────────────────────────────────────────────────
    base_kwargs = {
        "value_per_tick": args.value_per_tick, 
        "tick_size": args.tick_size, 
        "fill_probability": 1.0,
        "execution_cost_per_order": args.commission * 0.1,  # Reduce training costs
        "contracts_per_trade": 1,
        "margin_rate": args.margin_rate * 0.1,  # Reduce margin costs
        "bid_ask_spread": args.tick_size * 0.5,  # Reduce spread
        "add_current_position_to_state": True
    }
    train_env = FuturesEnv(states=train_states,
                           log_dir=f"{args.log_dir}/train_rank{local_rank}",
                           **base_kwargs)
    base_kwargs["execution_cost_per_order"] = args.commission * 0.1  # Lower test commission
    test_env  = FuturesEnv(states=test_states,
                           log_dir=f"{args.log_dir}/test_rank{local_rank}",
                           **base_kwargs)

    # ─── ADAPTIVE TRAINING (GA + PPO with intelligent switching) ──────────────
    current_input_dim = int(np.prod(train_env.observation_space.shape))
    
    if local_rank == 0:
        from adaptive_trainer import AdaptiveTrainer
        from email_notifications import TrainingNotificationSystem
        
        # Setup email notifications
        email_manager = TrainingNotificationSystem()
        try:
            email_manager.start_monitoring()
            logging.info("Email notifications enabled - reports every 6 hours")
        except Exception as e:
            logging.warning(f"Failed to setup email notifications: {e}")
            email_manager = None
        
        # Initialize adaptive trainer with distributed settings
        adaptive_trainer = AdaptiveTrainer(
            train_env=train_env,
            test_env=test_env,
            input_dim=current_input_dim,
            action_dim=train_env.action_space.n,
            device=str(device),
            models_dir=str(model_dirs['base_dir']),
            local_rank=local_rank,
            world_size=world_size,
            use_distributed=(world_size > 1)
        )
        
        # Run adaptive training with smaller iterations to avoid timeouts
        training_log = adaptive_trainer.adaptive_train(
            max_iterations=10,  # Reduced to avoid timeouts
            evaluation_interval=1
        )
        
        # Get the best agent from adaptive training
        best_agent = adaptive_trainer.get_best_agent()
        
        # Log training summary
        logging.info("=== Adaptive Training Summary ===")
        for i, (method, perf, reason) in enumerate(zip(
            training_log['methods'],
            training_log['performances'], 
            training_log['switch_reasons']
        )):
            logging.info(f"Iteration {i+1}: {method} -> Performance: {perf:.4f}, Switch: {reason}")
        
        # Save the best model with backup
        best_model_path = model_dirs['base_dir'] / "best_adaptive_model.pth"
        save_model_with_backup(best_agent, best_model_path, model_dirs['backup_dir'], args.backup_count)
        
        # Final evaluation of the best agent
        final_profits, final_times = evaluate_agent_distributed(test_env, best_agent, local_rank)
        final_cagr, final_sharpe, final_mdd = compute_performance_metrics(final_profits, final_times)
        logging.info(f"Final Best Agent → CAGR: {final_cagr:.4f}, Sharpe: {final_sharpe:.4f}, MDD: {final_mdd:.4f}")
        
        # Signal completion to other ranks
        completion_flag = model_dirs['base_dir'] / "training_complete.flag"
        completion_flag.touch()
        logging.info("Training completion flag created")
        
    else:
        # Non-rank 0 processes just wait with regular heartbeat
        import time
        heartbeat_count = 0
        while True:
            try:
                # Send heartbeat every 5 minutes to prevent timeout
                if heartbeat_count % 60 == 0:  # Every 5 minutes (60 * 5 seconds)
                    logging.info(f"Rank {local_rank} heartbeat - waiting for adaptive training")
                time.sleep(5)
                heartbeat_count += 1
                
                # Check if training is complete (simple file-based check)
                if (model_dirs['base_dir'] / "training_complete.flag").exists():
                    logging.info(f"Rank {local_rank} detected training completion")
                    break
                    
                # Safety timeout - exit after 4 hours
                if heartbeat_count > 2880:  # 4 hours
                    logging.warning(f"Rank {local_rank} timeout - exiting after 4 hours")
                    break
                    
            except KeyboardInterrupt:
                logging.info(f"Rank {local_rank} interrupted")
                break
            except Exception as e:
                logging.warning(f"Rank {local_rank} error in wait loop: {e}")
                break
    
    if world_size > 1:
        if torch.cuda.is_available():
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
