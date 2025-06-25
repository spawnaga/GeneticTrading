
#!/usr/bin/env python
"""
4 GPU Training Script - Optimized for 1000 rows
===============================================

This script runs the trading system on 4 GPUs with NVLink,
optimized for small datasets (1000 rows).
"""

import os
import sys
import subprocess
from pathlib import Path


def setup_4gpu_environment():
    """Setup environment variables for 4 GPU training."""
    env = os.environ.copy()
    
    # GPU Configuration
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    env["NCCL_TIMEOUT"] = "3600000"  # 1 hour timeout
    env["NCCL_DEBUG"] = "INFO"
    
    # NVLink optimization
    env["NCCL_P2P_DISABLE"] = "0"  # Enable P2P for NVLink
    env["NCCL_NET_GDR_DISABLE"] = "0"  # Enable GPU Direct RDMA
    env["NCCL_TREE_THRESHOLD"] = "0"  # Use ring algorithm for small data
    
    # Memory optimization for small datasets
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    return env


def run_4gpu_1000rows():
    """Run training on 4 GPUs with 1000 rows."""
    
    # Ensure directories exist
    Path("./models/4gpu_1000rows").mkdir(parents=True, exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)
    Path("./cached_data").mkdir(exist_ok=True)
    Path("./runs").mkdir(exist_ok=True)
    
    # Setup environment
    env = setup_4gpu_environment()
    
    cmd = [
        "torchrun",
        "--nproc_per_node=4",  # 4 GPUs
        "--nnodes=1",          # Single node
        "--node_rank=0",
        "--master_addr=127.0.0.1",
        "--master_port=12355",
        "main.py",
        
        # Data configuration for 1000 rows
        "--max-rows", "1000",
        "--data-percentage", "1.0",  # Use all 1000 rows
        "--chunk-size", "250",       # Small chunks for 4 GPUs
        
        # Model configuration
        "--models-dir", "./models/4gpu_1000rows",
        "--checkpoint-interval", "2",
        "--backup-count", "3",
        
        # Training configuration optimized for small data
        "--training-mode", "adaptive",
        "--adaptive-iterations", "10",  # Fewer iterations for small data
        "--ga-population", "40",        # Smaller population
        "--ga-generations", "20",       # Fewer generations
        "--ppo-batch-size", "16",       # Small batches
        "--ppo-lr", "1e-3",            # Higher learning rate for small data
        
        # Distributed training
        "--max-train-per-rank", "200",  # 200 rows per GPU (800 total train)
        "--max-test-per-rank", "50",    # 50 rows per GPU (200 total test)
        "--nccl-timeout", "3600000",
        
        # Environment settings
        "--value-per-tick", "12.5",
        "--tick-size", "0.25",
        "--commission", "0.0001",  # Lower commission for training
        "--margin-rate", "0.005",  # Lower margin for training
        
        # Logging
        "--log-dir", "./logs/4gpu_1000rows",
        "--tensorboard-dir", "./runs/4gpu_1000rows"
    ]
    
    print("Starting 4 GPU training with 1000 rows...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Environment variables set for NVLink optimization")
    
    result = subprocess.run(cmd, env=env)
    
    if result.returncode == 0:
        print("✅ Training completed successfully!")
    else:
        print(f"❌ Training failed with exit code {result.returncode}")
    
    return result.returncode


if __name__ == "__main__":
    exit_code = run_4gpu_1000rows()
    sys.exit(exit_code)
