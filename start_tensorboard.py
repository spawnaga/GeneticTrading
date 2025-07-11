
#!/usr/bin/env python
"""
TensorBoard Launcher for NQ Trading System
==========================================

Simple script to start TensorBoard and make it accessible in Replit.
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_tensorboard(log_dir="./runs", port=6006, host="0.0.0.0"):
    """Start TensorBoard server accessible in Replit."""
    
    # Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create some sample data if directory is empty
    if not any(log_path.iterdir()):
        logger.info("No TensorBoard data found, creating sample data...")
        create_sample_data(log_path)
    
    logger.info(f"Starting TensorBoard...")
    logger.info(f"Log directory: {log_path.absolute()}")
    logger.info(f"Host: {host}, Port: {port}")
    
    # TensorBoard command
    cmd = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir", str(log_path),
        "--port", str(port),
        "--host", host,
        "--reload_interval", "10",
        "--bind_all"  # Important for Replit accessibility
    ]
    
    try:
        logger.info("Starting TensorBoard server...")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Start TensorBoard
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor startup
        startup_timeout = 30
        start_time = time.time()
        
        logger.info(f"🚀 TensorBoard starting on http://{host}:{port}")
        logger.info("Waiting for TensorBoard to start...")
        
        while time.time() - start_time < startup_timeout:
            if process.poll() is not None:
                # Process ended
                stdout, stderr = process.communicate()
                logger.error(f"TensorBoard failed to start: {stdout}")
                return None
                
            time.sleep(1)
            logger.info(".")
        
        logger.info("✅ TensorBoard should now be accessible!")
        logger.info(f"🔗 Open: http://{host}:{port}")
        logger.info("📊 TensorBoard will automatically detect new training runs")
        logger.info("Press Ctrl+C to stop TensorBoard")
        
        # Keep the process running and show output
        try:
            for line in process.stdout:
                print(f"[TB] {line.strip()}")
        except KeyboardInterrupt:
            logger.info("Stopping TensorBoard...")
            process.terminate()
            process.wait()
            logger.info("TensorBoard stopped")
            
        return process
        
    except FileNotFoundError:
        logger.error("TensorBoard not installed. Installing...")
        install_tensorboard()
        return start_tensorboard(log_dir, port, host)
    except Exception as e:
        logger.error(f"Failed to start TensorBoard: {e}")
        return None

def install_tensorboard():
    """Install TensorBoard if not available."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])
        logger.info("TensorBoard installed successfully!")
    except Exception as e:
        logger.error(f"Failed to install TensorBoard: {e}")

def create_sample_data(log_dir):
    """Create sample TensorBoard data for testing."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        import numpy as np
        
        logger.info("Creating sample training data...")
        
        # Create sample GA data
        ga_writer = SummaryWriter(log_dir / "ga_experiment")
        for i in range(50):
            fitness = 0.3 + 0.4 * (i / 50) + 0.1 * np.random.random()
            ga_writer.add_scalar('GA/Fitness/Average', fitness, i)
            ga_writer.add_scalar('GA/Fitness/Max', fitness + 0.1, i)
            ga_writer.add_scalar('GA/Population/Diversity', 0.5 + 0.2 * np.random.random(), i)
        ga_writer.close()
        
        # Create sample PPO data
        ppo_writer = SummaryWriter(log_dir / "ppo_experiment")
        for i in range(100):
            reward = 0.2 + 0.5 * (i / 100) + 0.15 * np.random.random()
            loss = 0.5 - 0.3 * (i / 100) + 0.1 * np.random.random()
            ppo_writer.add_scalar('PPO/Reward', reward, i)
            ppo_writer.add_scalar('PPO/Loss', loss, i)
            ppo_writer.add_scalar('PPO/Learning_Rate', 3e-4, i)
        ppo_writer.close()
        
        # Create sample trading data
        trading_writer = SummaryWriter(log_dir / "trading_metrics")
        account_value = 100000
        for i in range(200):
            daily_return = 0.001 * np.random.randn()
            account_value *= (1 + daily_return)
            
            trading_writer.add_scalar('Trading/AccountValue', account_value, i)
            trading_writer.add_scalar('Trading/DailyReturn', daily_return * 100, i)
            
            if i > 20:
                returns = np.random.randn(20) * 0.01
                sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
                trading_writer.add_scalar('Trading/SharpeRatio', sharpe, i)
        trading_writer.close()
        
        logger.info("Sample data created successfully!")
        
    except ImportError:
        logger.warning("Could not create sample data - tensorboard not available")

def check_existing_runs():
    """Check for existing training runs."""
    runs_dir = Path("./runs")
    if runs_dir.exists():
        subdirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        if subdirs:
            logger.info(f"Found {len(subdirs)} existing training runs:")
            for subdir in subdirs:
                logger.info(f"  - {subdir.name}")
            return True
    return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start TensorBoard for NQ Trading System")
    parser.add_argument("--logdir", default="./runs", help="TensorBoard log directory")
    parser.add_argument("--port", type=int, default=6006, help="Port for TensorBoard")
    parser.add_argument("--host", default="0.0.0.0", help="Host for TensorBoard")
    
    args = parser.parse_args()
    
    print("🎯 NQ Trading System - TensorBoard Launcher")
    print("=" * 50)
    
    # Check for existing runs
    if not check_existing_runs():
        logger.info("No existing training runs found")
        logger.info("Sample data will be created for demonstration")
    
    # Start TensorBoard
    start_tensorboard(args.logdir, args.port, args.host)
