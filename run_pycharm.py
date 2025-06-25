
#!/usr/bin/env python3
"""
PyCharm Run Configuration
========================

Simple script to run the trading system in PyCharm with optimal settings.
Just run this file directly in PyCharm.
"""

import sys
import os
from pathlib import Path

def setup_pycharm_environment():
    """Setup environment for PyCharm execution."""
    # Ensure all directories exist
    dirs = ['./models', './logs', './cached_data', './runs', './data_txt']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✓ Environment setup complete")

def run_quick_test():
    """Run quick test optimized for PyCharm."""
    print("🚀 Starting PyCharm Quick Test...")
    
    # Import main after environment setup
    from main import main
    
    # Set arguments for quick testing
    sys.argv = [
        'main.py',
        '--data-percentage', '0.02',    # Use 2% of data
        '--max-rows', '1000',           # Limit to 1000 rows
        '--models-dir', './models/pycharm',
        '--total-steps', '2000',        # Quick training
        '--ga-population', '8',         # Small population
        '--ga-generations', '3',        # Few generations
        '--adaptive-iterations', '5',   # Fewer adaptive iterations
        '--eval-interval', '1',         # Frequent evaluation
        '--log-level', 'INFO'
    ]
    
    # Run main function
    main()

def main():
    """Main entry point for PyCharm."""
    print("🤖 Trading System - PyCharm Mode")
    print("=" * 40)
    
    setup_pycharm_environment()
    
    try:
        run_quick_test()
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
