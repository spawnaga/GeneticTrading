
#!/usr/bin/env python
"""
Professional Training Runner Script
===================================

This script provides easy-to-use commands for running the trading system
with various configurations and distributed training options.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_single_gpu():
    """Run training on single GPU/CPU."""
    cmd = [
        sys.executable, "main.py",
        "--data-percentage", "0.1",  # Use 10% of data for testing
        "--max-rows", "5000",
        "--models-dir", "./models",
        "--checkpoint-interval", "5",
        "--backup-count", "3"
    ]
    subprocess.run(cmd)


def run_distributed():
    """Run distributed training across multiple GPUs."""
    cmd = [
        "torchrun",
        "--nproc_per_node=4",  # Use all 4 GPUs
        "--nnodes=1",
        "--node_rank=0",
        "--master_addr=127.0.0.1",
        "--master_port=12355",
        "main.py",
        "--data-percentage", "0.8",  # Use 80% of data for 4 GPUs
        "--max-rows", "200000",
        "--models-dir", "./models",
        "--total-steps", "2000000",
        "--ga-population", "160",  # Scale population with GPU count
        "--ga-generations", "200",
        "--nccl-timeout", "3600000"  # 1 hour timeout for stability
    ]
    
    # Set environment variables for 4 GPU setup
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    env["NCCL_TIMEOUT"] = "3600000"
    
    subprocess.run(cmd, env=env)


def run_full_training():
    """Run full training with all data."""
    cmd = [
        sys.executable, "main.py",
        "--data-percentage", "1.0",  # Use all data
        "--max-rows", "0",  # Use all available rows
        "--models-dir", "./models/production",
        "--total-steps", "2000000",
        "--ga-population", "100",
        "--ga-generations", "200",
        "--checkpoint-interval", "10",
        "--backup-count", "10"
    ]
    subprocess.run(cmd)


def run_quick_test():
    """Run quick test with minimal data."""
    cmd = [
        sys.executable, "main.py",
        "--data-percentage", "0.01",  # Use 1% of data
        "--max-rows", "1000",
        "--models-dir", "./models/test",
        "--total-steps", "10000",
        "--ga-population", "10",
        "--ga-generations", "5",
        "--eval-interval", "2"
    ]
    subprocess.run(cmd)


def run_4gpu_performance():
    """Run high-performance training optimized for 4x 3090 GPUs."""
    cmd = [
        "torchrun",
        "--nproc_per_node=4",
        "--nnodes=1", 
        "--node_rank=0",
        "--master_addr=127.0.0.1",
        "--master_port=12355",
        "main.py",
        "--data-percentage", "1.0",  # Use all available data
        "--max-rows", "0",  # No row limit
        "--models-dir", "./models/4gpu_production",
        "--total-steps", "5000000",  # Extended training
        "--ga-population", "320",  # Large population for 4 GPUs
        "--ga-generations", "500",
        "--ppo-batch-size", "256",  # Larger batches for 4 GPUs
        "--checkpoint-interval", "25",
        "--backup-count", "20",
        "--nccl-timeout", "7200000"  # 2 hour timeout
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    env["NCCL_TIMEOUT"] = "7200000"
    env["NCCL_DEBUG"] = "INFO"  # For debugging if needed
    
    subprocess.run(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description="Professional Trading System Runner")
    parser.add_argument("mode", choices=["single", "distributed", "full", "test", "4gpu"],
                       help="Training mode to run")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    Path("./models").mkdir(exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)
    Path("./cached_data").mkdir(exist_ok=True)
    
    if args.mode == "single":
        run_single_gpu()
    elif args.mode == "distributed":
        run_distributed()
    elif args.mode == "full":
        run_full_training()
    elif args.mode == "test":
        run_quick_test()
    elif args.mode == "4gpu":
        run_4gpu_performance()


if __name__ == "__main__":
    main()
