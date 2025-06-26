
#!/usr/bin/env python
"""
Local GPU Training Script
========================

Optimized for running on local machines with single or multiple GPUs.
"""

import os
import sys
import subprocess
import torch
from pathlib import Path


def check_gpu_availability():
    """Check GPU availability and return configuration."""
    if not torch.cuda.is_available():
        print("❌ CUDA not available - will run on CPU")
        return False, 0
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ Found {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {gpu_name} ({memory:.1f} GB)")
    
    return True, gpu_count


def setup_environment():
    """Setup environment variables for optimal training."""
    env = os.environ.copy()
    
    # Force GPU usage
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        env["CUDA_LAUNCH_BLOCKING"] = "0"  # Allow async operations
        env["NCCL_DEBUG"] = "INFO"
        print(f"🔥 Forcing GPU usage with {gpu_count} GPUs")
    else:
        print("⚠️ No GPUs available, using CPU")
    
    # Memory optimizations
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    
    # Keep cuDF enabled for GPU acceleration
    if torch.cuda.is_available():
        env.pop("CUDF_BACKEND", None)
        env.pop("RAPIDS_NO_INITIALIZE", None)
    
    return env


def run_single_gpu_training():
    """Run training on single GPU with optimized settings."""
    print("🚀 Starting Single GPU Training...")
    
    # Ensure directories exist
    Path("./models/local_gpu").mkdir(parents=True, exist_ok=True)
    Path("./logs/local_gpu").mkdir(parents=True, exist_ok=True)
    Path("./runs/local_gpu").mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "main.py",
        
        # Data configuration - reduced for faster training
        "--max-rows", "5000",
        "--data-percentage", "1.0",
        "--chunk-size", "1000",
        
        # Model configuration
        "--models-dir", "./models/local_gpu",
        "--checkpoint-interval", "5",
        "--backup-count", "3",
        
        # Training configuration - optimized for single GPU
        "--training-mode", "adaptive",
        "--adaptive-iterations", "5",  # Reduced iterations
        "--stagnation-threshold", "3",
        "--poor-performance-threshold", "2",
        
        # GA settings - smaller for faster convergence
        "--ga-population", "20",
        "--ga-generations", "10",
        
        # PPO settings - optimized for single GPU
        "--ppo-lr", "1e-3",
        "--ppo-batch-size", "32",
        
        # Distributed settings (single process)
        "--max-train-per-rank", "4000",
        "--max-test-per-rank", "1000",
        
        # Environment settings
        "--value-per-tick", "12.5",
        "--tick-size", "0.25",
        "--commission", "0.0001",
        "--margin-rate", "0.005",
        
        # Logging
        "--log-dir", "./logs/local_gpu",
        "--tensorboard-dir", "./runs/local_gpu",
        "--log-level", "INFO"
    ]
    
    env = setup_environment()
    
    print(f"Command: {' '.join(cmd)}")
    print("Starting training...")
    
    try:
        result = subprocess.run(cmd, env=env, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return 1


def run_multi_gpu_training(gpu_count):
    """Run training on multiple GPUs with torchrun."""
    print(f"🚀 Starting Multi-GPU Training on {gpu_count} GPUs...")
    
    # Ensure directories exist
    Path("./models/multi_gpu").mkdir(parents=True, exist_ok=True)
    Path("./logs/multi_gpu").mkdir(parents=True, exist_ok=True)
    Path("./runs/multi_gpu").mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "torchrun",
        f"--nproc_per_node={gpu_count}",
        "--nnodes=1",
        "--node_rank=0",
        "--master_addr=127.0.0.1",
        "--master_port=12356",  # Different port to avoid conflicts
        "main.py",
        
        # Data configuration
        "--max-rows", "10000",
        "--data-percentage", "1.0",
        "--chunk-size", "2000",
        
        # Model configuration
        "--models-dir", "./models/multi_gpu",
        "--checkpoint-interval", "10",
        "--backup-count", "5",
        
        # Training configuration
        "--training-mode", "adaptive",
        "--adaptive-iterations", "8",
        "--stagnation-threshold", "3",
        "--poor-performance-threshold", "2",
        
        # GA settings - scaled for multiple GPUs
        "--ga-population", str(20 * gpu_count),
        "--ga-generations", "15",
        
        # PPO settings
        "--ppo-lr", "5e-4",
        "--ppo-batch-size", str(16 * gpu_count),
        
        # Distributed settings
        "--max-train-per-rank", str(2000 * gpu_count),
        "--max-test-per-rank", str(500 * gpu_count),
        "--nccl-timeout", "1800000",  # 30 minutes
        
        # Environment settings
        "--value-per-tick", "12.5",
        "--tick-size", "0.25",
        "--commission", "0.0001",
        "--margin-rate", "0.005",
        
        # Logging
        "--log-dir", "./logs/multi_gpu",
        "--tensorboard-dir", "./runs/multi_gpu",
        "--log-level", "INFO"
    ]
    
    env = setup_environment()
    
    # Additional multi-GPU environment variables
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))
    env["NCCL_TIMEOUT"] = "1800000"
    env["NCCL_DEBUG"] = "WARN"
    
    print(f"Command: {' '.join(cmd)}")
    print("Starting multi-GPU training...")
    
    try:
        result = subprocess.run(cmd, env=env, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return 1


def main():
    """Main function to run appropriate training configuration."""
    print("🔍 Checking system configuration...")
    
    has_gpu, gpu_count = check_gpu_availability()
    
    if not has_gpu:
        print("Running CPU-only training...")
        return run_single_gpu_training()
    
    print(f"\nSelect training mode:")
    print(f"1. Single GPU training (recommended for testing)")
    print(f"2. Multi-GPU training ({gpu_count} GPUs)")
    print(f"3. CPU-only training")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            return run_single_gpu_training()
        elif choice == "2" and gpu_count > 1:
            return run_multi_gpu_training(gpu_count)
        elif choice == "2" and gpu_count == 1:
            print("Only 1 GPU available, running single GPU training...")
            return run_single_gpu_training()
        elif choice == "3":
            # Disable CUDA for CPU-only training
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            return run_single_gpu_training()
        else:
            print("Invalid choice, defaulting to single GPU training...")
            return run_single_gpu_training()
            
    except KeyboardInterrupt:
        print("\n⚠️ Cancelled by user")
        return 1


if __name__ == "__main__":
    exit_code = main()
    if exit_code == 0:
        print("\n✅ Training completed successfully!")
    else:
        print("\n❌ Training failed or was interrupted")
    sys.exit(exit_code)
