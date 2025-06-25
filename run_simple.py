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
    """Run development mode with 10% of data."""
    print("🔧 Starting Development Mode...")
    print("📊 Using 10% of data for development")

    from main import main

    sys.argv = [
        'main.py',
        '--data-percentage', '0.1',   # Use 10% of data
        '--max-rows', '5000',         # 5K rows
        '--models-dir', './models/dev',
        '--total-steps', '50000',     # Moderate training
        '--ga-population', '20',      # Reasonable population
        '--ga-generations', '20',     # Moderate generations
        '--eval-interval', '5',
        '--log-level', 'INFO'
    ]

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
        print("\nAvailable modes:")
        print("  test - Quick test with minimal data (default)")
        print("  dev  - Development mode with 10% data")
        print("\nUsage: python run_simple.py [test|dev]")
        mode = input("\nSelect mode (test/dev) [test]: ").lower() or 'test'

    try:
        if mode == 'dev':
            run_development()
        else:
            run_quick_test()

        print("\n✅ Training completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()