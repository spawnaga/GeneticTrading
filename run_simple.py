#!/usr/bin/env python3
"""
Simple Trading System Launcher
==============================

Easy-to-use script for running the trading system locally.
Perfect for PyCharm or any IDE.
"""

import os
import sys
import logging
from pathlib import Path

def setup_environment():
    """Setup basic environment for local development."""
    # Create necessary directories
    dirs_to_create = [
        './models',
        './logs', 
        './cached_data',
        './runs'
    ]

    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def run_quick_test():
    """Run a quick test with minimal data for development."""
    print("🚀 Starting Quick Test Mode...")
    print("📊 Using minimal data for fast iteration")

    # Import main after environment setup
    from main import main

    # Override sys.argv to simulate command line arguments
    sys.argv = [
        'main.py',
        '--data-percentage', '0.01',  # Use 1% of data
        '--max-rows', '500',          # Limit to 500 rows
        '--models-dir', './models/test',
        '--total-steps', '1000',      # Quick training
        '--ga-population', '10',      # Small population
        '--ga-generations', '5',      # Few generations
        '--eval-interval', '1',       # Frequent evaluation
        '--log-level', 'INFO'
    ]

    # Run the main function
    main()

def run_development():
    """Run development mode with all available data."""
    print("🔧 Starting Development Mode...")
    print("📊 Using all available data")

    from main import main

    sys.argv = [
        'main.py',
        '--data-percentage', '1.0',   # Use all data
        '--max-rows', '0',            # Use all rows (0 = no limit)
        '--models-dir', './models/dev',
        '--adaptive-iterations', '20', # Use adaptive training instead
        '--ga-population', '50',      # Larger population for real data
        '--ga-generations', '50',     # More generations
        '--eval-interval', '1',       # Evaluate every iteration
        '--log-level', 'INFO'
    ]

    main()

def run_full_training():
    """Run full training with all available data."""
    print("🚀 Starting Full Training Mode...")
    print("📊 Using 100% of available data")

    from main import main

    sys.argv = [
        'main.py',
        '--data-percentage', '1.0',   # Use all data
        '--max-rows', '0',            # No row limit
        '--models-dir', './models/production',
        '--adaptive-iterations', '20', # Full adaptive training
        '--ga-population', '80',      # Large population
        '--ga-generations', '100',    # More generations
        '--eval-interval', '5',
        '--log-level', 'INFO'
    ]

    print("📈 Full training mode - this will take much longer")
    print("📊 Monitor progress in ./logs/ and ./runs/ directories")
    main()

def main():
    """Main launcher with mode selection."""
    print("🤖 Trading System Launcher")
    print("=" * 50)

    # Setup environment
    setup_environment()

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("Available modes:")
        print("  test - Quick test with minimal data (default)")
        print("  dev  - Development mode with 10% data")
        print("  full - Full training with 100% data")
        print()
        print("Usage: python run_simple.py [test|dev|full]")
        mode = input("Select mode (test/dev/full) [test]: ").strip().lower()
    if not mode:
        mode = "test"

    try:
        if mode == 'test':
            print("🧪 Starting Test Mode...")
            print("📊 Using minimal data for quick testing")
            print("📈 Monitoring enabled - check ./logs/training_metrics.json for progress")
            run_quick_test()
        elif mode == 'dev':
            print("🔧 Starting Development Mode...")
            print("📊 Using all available data with adaptive training")
            print("📈 Monitoring enabled - check ./logs/training_metrics.json for progress")
            print("⚡ Adaptive GA+PPO training with all your data")
            run_development()
        elif mode == 'full':
            print("🚀 Starting Full Training Mode...")
            print("📊 Using 100% of available data")
            run_full_training()

        print("\n✅ Training completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()