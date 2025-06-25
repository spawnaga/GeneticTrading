
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


def check_cuda_environment():
    """Check CUDA environment and return status."""
    import subprocess
    try:
        # Check if nvidia-smi works
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA driver detected")
            return True
        else:
            print("❌ NVIDIA driver not working")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        return False

def setup_4gpu_environment():
    """Setup environment variables for 4 GPU training."""
    env = os.environ.copy()
    
    # Check CUDA availability first
    cuda_available = check_cuda_environment()
    
    if cuda_available:
        # GPU Configuration
        env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        env["NCCL_TIMEOUT"] = "3600000"  # 1 hour timeout
        env["NCCL_DEBUG"] = "INFO"
        
        # NVLink optimization
        env["NCCL_P2P_DISABLE"] = "0"  # Enable P2P for NVLink
        env["NCCL_NET_GDR_DISABLE"] = "0"  # Enable GPU Direct RDMA
        env["NCCL_TREE_THRESHOLD"] = "0"  # Use ring algorithm for small data
        
        # Memory optimization for small datasets
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
        
        # Fix CUDA library path issues
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"
        
        # Set NUMBA CUDA driver path
        env["NUMBA_CUDA_DRIVER"] = "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
    else:
        print("⚠️  Running in CPU-only mode")
        env["CUDA_VISIBLE_DEVICES"] = ""
    
    return env, cuda_available


def run_4gpu_1000rows():
    """Run training on 4 GPUs with 1000 rows."""
    
    # Ensure directories exist
    Path("./models/4gpu_1000rows").mkdir(parents=True, exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)
    Path("./cached_data").mkdir(exist_ok=True)
    Path("./runs").mkdir(exist_ok=True)
    Path("./data_txt").mkdir(exist_ok=True)
    
    # Setup environment and check CUDA
    env, cuda_available = setup_4gpu_environment()
    
    # Adjust command based on CUDA availability
    if cuda_available:
        cmd = [
            "torchrun",
            "--nproc_per_node=4",  # 4 GPUs
            "--nnodes=1",          # Single node
            "--node_rank=0",
            "--master_addr=127.0.0.1",
            "--master_port=12355",
            "main.py"
        ]
        print("🚀 Starting 4 GPU training with 1000 rows...")
    else:
        # Fallback to single process
        cmd = ["python", "main.py"]
        print("🚀 Starting CPU training with 1000 rows...")
    
    cmd.extend([
        # Data configuration for 1000 rows
        "--max-rows", "1000",
        "--data-percentage", "1.0",  # Use all 1000 rows
        "--chunk-size", "250",       # Small chunks
        
        # Model configuration
        "--models-dir", "./models/4gpu_1000rows",
        "--checkpoint-interval", "5",
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
    ])
    
    print(f"Command: {' '.join(cmd)}")
    if cuda_available:
        print("Environment variables set for NVLink optimization")
    
    # Check if data file exists
    data_file = "./data_txt/NQ_full_1min_continuous_absolute_adjusted.txt"
    if not os.path.exists(data_file):
        print(f"⚠️  Data file not found: {data_file}")
        print("Creating sample data file...")
        os.makedirs("./data_txt", exist_ok=True)
        # Create a minimal sample data file
        sample_data = """2023-01-01 00:00:00,15000.0,15100.0,14900.0,15050.0,1000
2023-01-01 00:01:00,15050.0,15150.0,14950.0,15100.0,1100
2023-01-01 00:02:00,15100.0,15200.0,15000.0,15150.0,1200"""
        with open(data_file, 'w') as f:
            f.write(sample_data)
        print(f"✅ Created sample data file: {data_file}")
    
    result = subprocess.run(cmd, env=env)
    
    if result.returncode == 0:
        print("✅ Training completed successfully!")
    else:
        print(f"❌ Training failed with exit code {result.returncode}")
    
    return result.returncode


if __name__ == "__main__":
    exit_code = run_4gpu_1000rows()
    sys.exit(exit_code)
