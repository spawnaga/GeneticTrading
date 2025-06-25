
#!/usr/bin/env python3
"""
Minimal test script to verify the system works with tiny data
"""
import logging
from main import main
import sys
import os

# Set minimal arguments for testing
sys.argv = [
    'test_minimal.py',
    '--data-percentage', '1.0', 
    '--max-rows', '100',  # Very small for quick testing
    '--adaptive-iterations', '3',  # Just 3 iterations
    '--ga-population', '10',  # Small population
    '--ga-generations', '5',  # Few generations
    '--ppo-batch-size', '8',  # Small batch
    '--eval-interval', '1'  # Evaluate every iteration
]

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./cached_data', exist_ok=True)
    
    print("🧪 Running minimal test...")
    print("📊 Using 100 rows max, 3 adaptive iterations")
    print("⚡ Small GA population (10) and generations (5)")
    
    main()
