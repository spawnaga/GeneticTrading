
#!/usr/bin/env python
"""
Simple Local Training Script
===========================

Runs training without distributed setup to avoid torchrun issues.
"""

import os
import sys
import torch
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_environment():
    """Setup environment for local training."""
    # Disable distributed training
    if "LOCAL_RANK" in os.environ:
        del os.environ["LOCAL_RANK"]
    if "WORLD_SIZE" in os.environ:
        del os.environ["WORLD_SIZE"]
    if "RANK" in os.environ:
        del os.environ["RANK"]
    
    # Force CPU fallback for problematic libraries
    os.environ["CUDF_BACKEND"] = "cpu"
    os.environ["RAPIDS_NO_INITIALIZE"] = "1"
    
    # Optimize for single process
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available with {torch.cuda.device_count()} GPU(s)")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    else:
        print("⚠️ CUDA not available, running on CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    """Run simple local training."""
    setup_environment()
    
    # Ensure directories exist
    Path("./models/local_simple").mkdir(parents=True, exist_ok=True)
    Path("./logs/local_simple").mkdir(parents=True, exist_ok=True)
    Path("./runs/local_simple").mkdir(parents=True, exist_ok=True)
    
    # Import and run main directly (no subprocess)
    try:
        from main import main as main_func
        
        # Override sys.argv to pass arguments
        sys.argv = [
            "main.py",
            "--max-rows", "2000",  # Small dataset for testing
            "--data-percentage", "1.0",
            "--models-dir", "./models/local_simple",
            "--training-mode", "adaptive",
            "--adaptive-iterations", "3",  # Very few iterations
            "--ga-population", "10",  # Small population
            "--ga-generations", "5",   # Few generations
            "--ppo-batch-size", "16",  # Small batch size
            "--log-dir", "./logs/local_simple",
            "--tensorboard-dir", "./runs/local_simple",
            "--log-level", "INFO"
        ]
        
        print("🚀 Starting simple local training...")
        print("📊 Using 2000 rows with 3 adaptive iterations")
        print("🔬 Small GA population (10) and few generations (5)")
        
        main_func()
        print("✅ Training completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
