
#!/usr/bin/env python
"""
Simple Trading System Runner
===========================

Convenience script to run the trading system with various configurations.
All actual logic is in main.py.
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_system(config="default"):
    """Run the trading system with specified configuration."""
    
    # Base command
    base_cmd = ["python", "main.py"]
    
    # Configuration presets
    configs = {
        "default": [
            "--data-percentage", "0.1",
            "--max-rows", "1000", 
            "--training-mode", "adaptive",
            "--adaptive-iterations", "5"
        ],
        "full": [
            "--data-percentage", "1.0",
            "--max-rows", "0",
            "--training-mode", "adaptive", 
            "--adaptive-iterations", "20"
        ],
        "4gpu": [
            "--data-percentage", "1.0",
            "--max-rows", "0",
            "--training-mode", "adaptive",
            "--adaptive-iterations", "20"
        ],
        "test": [
            "--data-percentage", "0.01",
            "--max-rows", "100",
            "--training-mode", "adaptive",
            "--adaptive-iterations", "2"
        ]
    }
    
    # Build command
    if config in configs:
        cmd = base_cmd + configs[config]
    else:
        cmd = base_cmd
    
    # Special handling for 4GPU
    if config == "4gpu":
        cmd = [
            "torchrun", "--nproc_per_node=4", "--nnodes=1", 
            "--node_rank=0", "--master_addr=127.0.0.1", "--master_port=12355"
        ] + cmd[1:]  # Remove "python" since torchrun handles it
    
    print(f"🚀 Running trading system with config: {config}")
    print(f"📝 Command: {' '.join(cmd)}")
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Trading system completed successfully")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Trading system failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        return 130

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the trading system")
    parser.add_argument("--config", default="default", 
                       choices=["default", "full", "4gpu", "test"],
                       help="Configuration preset to use")
    
    args = parser.parse_args()
    exit_code = run_system(args.config)
    sys.exit(exit_code)
