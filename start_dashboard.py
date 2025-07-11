
#!/usr/bin/env python
"""
Complete Dashboard & TensorBoard Startup Script
"""
import subprocess
import sys
import threading
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_tensorboard():
    """Start TensorBoard in background."""
    logger.info("🔧 Starting TensorBoard...")
    try:
        subprocess.run([sys.executable, "start_tensorboard.py", "--port", "6006"])
    except Exception as e:
        logger.error(f"TensorBoard failed: {e}")

def start_dashboard():
    """Start the main dashboard."""
    logger.info("🚀 Starting Trading Dashboard...")
    try:
        subprocess.run([sys.executable, "standalone_dashboard.py"])
    except KeyboardInterrupt:
        logger.info("\n🛑 Dashboard stopped")

def start_both():
    """Start both TensorBoard and dashboard."""
    print("🎯 NQ Trading System - Complete Dashboard")
    print("=" * 50)
    print("Starting TensorBoard on port 6006...")
    print("Starting Trading Dashboard on port 5000...")
    print("=" * 50)
    
    # Start TensorBoard in background thread
    tb_thread = threading.Thread(target=start_tensorboard, daemon=True)
    tb_thread.start()
    
    # Give TensorBoard time to start
    time.sleep(3)
    
    # Start main dashboard (this will block)
    start_dashboard()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensorboard-only", action="store_true", help="Start only TensorBoard")
    parser.add_argument("--dashboard-only", action="store_true", help="Start only dashboard")
    
    args = parser.parse_args()
    
    if args.tensorboard_only:
        start_tensorboard()
    elif args.dashboard_only:
        start_dashboard()
    else:
        start_both()
#!/usr/bin/env python
"""
Start Dashboard with TensorBoard Integration
==========================================

This script starts both the trading dashboard and TensorBoard
for comprehensive monitoring of the training process.
"""

import subprocess
import threading
import time
import logging
import sys
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_tensorboard(log_dir="./runs", port=6006):
    """Start TensorBoard server."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting TensorBoard on port {port}...")
    logger.info(f"TensorBoard log directory: {log_path}")
    
    cmd = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir", str(log_path),
        "--port", str(port),
        "--host", "0.0.0.0",
        "--reload_interval", "5"
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        logger.info(f"TensorBoard started successfully!")
        logger.info(f"🔗 TensorBoard URL: http://0.0.0.0:{port}")
        return process
    except Exception as e:
        logger.error(f"Failed to start TensorBoard: {e}")
        return None

def start_dashboard(port=5000):
    """Start the trading dashboard."""
    logger.info(f"Starting trading dashboard on port {port}...")
    
    try:
        from standalone_dashboard import ComprehensiveTradingDashboard
        dashboard = ComprehensiveTradingDashboard(port=port)
        dashboard.start_server()
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")

def create_sample_tensorboard_data():
    """Create sample TensorBoard data for testing."""
    try:
        import numpy as np
        import torch
        from torch.utils.tensorboard import SummaryWriter
        
        logger.info("Creating sample TensorBoard data...")
        
        writer = SummaryWriter('./runs/sample_training')
        
        # Simulate training data
        for i in range(100):
            # GA metrics
            ga_fitness = 0.5 + 0.3 * (i / 100) + 0.1 * np.random.random()
            writer.add_scalar('GA/Fitness', ga_fitness, i)
            
            # PPO metrics
            ppo_reward = 0.3 + 0.4 * (i / 100) + 0.15 * np.random.random()
            writer.add_scalar('PPO/Reward', ppo_reward, i)
            
            # Trading metrics
            account_value = 100000 + 1000 * i + 500 * np.random.random()
            writer.add_scalar('Trading/AccountValue', account_value, i)
            
            # Performance metrics
            sharpe_ratio = 0.5 + 0.5 * (i / 100) + 0.2 * np.random.random()
            writer.add_scalar('Performance/SharpeRatio', sharpe_ratio, i)
            
        writer.close()
        logger.info("Sample TensorBoard data created successfully!")
        
    except ImportError:
        logger.warning("TensorBoard not available, skipping sample data creation")
    except Exception as e:
        logger.error(f"Error creating sample TensorBoard data: {e}")

def main():
    """Main function to start all services."""
    logger.info("🚀 Starting comprehensive trading monitoring system...")
    
    # Create sample data if needed
    try:
        import numpy as np
        create_sample_tensorboard_data()
    except ImportError:
        logger.warning("NumPy not available, skipping sample data creation")
    
    # Start TensorBoard in background
    tb_process = start_tensorboard()
    
    if tb_process:
        # Give TensorBoard time to start
        time.sleep(3)
        
        # Check if TensorBoard is running
        if tb_process.poll() is None:
            logger.info("✅ TensorBoard is running")
        else:
            logger.warning("⚠️ TensorBoard may have failed to start")
    
    # Start dashboard (this will block)
    logger.info("Starting main dashboard...")
    try:
        start_dashboard()
    except KeyboardInterrupt:
        logger.info("Shutting down services...")
        if tb_process and tb_process.poll() is None:
            tb_process.terminate()
            logger.info("TensorBoard terminated")

if __name__ == "__main__":
    main()
