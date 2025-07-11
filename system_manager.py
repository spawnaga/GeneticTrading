
#!/usr/bin/env python
"""
Complete System Startup Script
=============================

Starts TensorBoard, training, and dashboard in the correct order
with proper monitoring and real-time updates.
"""

import subprocess
import threading
import time
import logging
import sys
import os
from pathlib import Path
import signal
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemManager:
    """Manages the complete trading system startup and monitoring."""

    def __init__(self, log_dir="./logs/futures_env", dashboard_port=8080):
        self.processes = {}
        self.running = True
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.dashboard_port = dashboard_port

    def start_tensorboard(self):
        """Start TensorBoard first."""
        logger.info("🚀 Starting TensorBoard...")

        runs_dir = Path("./runs")
        runs_dir.mkdir(exist_ok=True)

        cmd = [
            sys.executable, "-m", "tensorboard.main",
            "--logdir", "./runs",
            "--port", "6006",
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
            self.processes['tensorboard'] = process
            logger.info("✅ TensorBoard started on http://0.0.0.0:6006")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start TensorBoard: {e}")
            return False

    def start_training(self):
        """Start the training process."""
        logger.info("🧠 Starting training process...")

        cmd = [
            "torchrun", "--nproc_per_node=4", "--nnodes=1",
            "--node_rank=0", "--master_addr=127.0.0.1", "--master_port=12356",
            "main.py",
            "--data-folder", "./data_txt",
            "--max-rows", "0",
            "--data-percentage", "1.0",
            "--adaptive-iterations", "20",
            "--log-level", "INFO",
            "--training-mode", "adaptive",
            "--max-train-per-rank", "5000000",
            "--max-test-per-rank", "1000000",
            "--log-dir", str(self.log_dir)  # Pass log_dir to main.py
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            self.processes['training'] = process
            logger.info("✅ Training started")

            def monitor_training():
                while self.running and process.poll() is None:
                    line = process.stdout.readline()
                    if line:
                        print(f"[TRAINING] {line.strip()}")
                        if "Performance:" in line or "CAGR:" in line or "Sharpe:" in line:
                            logger.info(f"📊 {line.strip()}")

            threading.Thread(target=monitor_training, daemon=True).start()
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start training: {e}")
            return False

    def start_dashboard(self):
        """Start the monitoring dashboard."""
        logger.info("📊 Starting dashboard...")

        cmd = [sys.executable, "simple_trading_monitor.py", "--port", str(self.dashboard_port)]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            self.processes['dashboard'] = process
            logger.info(f"✅ Dashboard started on http://0.0.0.0:{self.dashboard_port}")

            def monitor_dashboard():
                while self.running and process.poll() is None:
                    line = process.stdout.readline()
                    if line:
                        if "Running on" in line or "Dashboard" in line:
                            logger.info(f"[DASHBOARD] {line.strip()}")
                        elif "ERROR" in line:
                            logger.warning(f"[DASHBOARD] {line.strip()}")

            threading.Thread(target=monitor_dashboard, daemon=True).start()
            
            # Wait a moment for dashboard to start
            time.sleep(2)
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start dashboard: {e}")
            return False

    def wait_for_tensorboard(self, max_wait=10):
        """Wait for TensorBoard to be ready."""
        logger.info("⏳ Waiting for TensorBoard to initialize...")
        for i in range(max_wait):
            try:
                response = requests.get("http://localhost:6006", timeout=1)
                if response.status_code == 200:
                    logger.info("✅ TensorBoard is ready!")
                    return True
            except:
                pass
            if i < 5:  # Only show first 5 attempts
                logger.info(f"⏳ Waiting for TensorBoard... ({i+1}/{max_wait})")
            time.sleep(1)
        logger.info("⚠️ TensorBoard starting in background, continuing...")
        return True

    def start_complete_system(self):
        """Start the complete system in proper order."""
        logger.info("🚀 Starting Complete NQ Trading System")
        logger.info("=" * 60)

        if not self.start_tensorboard():
            return False

        self.wait_for_tensorboard()

        if not self.start_training():
            return False

        logger.info("⏳ Waiting for training to initialize...")
        time.sleep(10)

        if not self.start_dashboard():
            return False

        logger.info("🎉 System startup complete!")
        logger.info(f"📊 Dashboard: http://0.0.0.0:{self.dashboard_port}")
        logger.info("📈 TensorBoard: http://0.0.0.0:6006")
        logger.info("🧠 Training: Active")
        logger.info("=" * 60)
        return True

    def monitor_system(self):
        """Monitor all processes and provide status updates."""
        while self.running:
            try:
                status = {name: "🟢 Running" if process.poll() is None else "🔴 Stopped"
                          for name, process in self.processes.items()}
                logger.info("📊 System Status:")
                for name, state in status.items():
                    logger.info(f"   {name.capitalize()}: {state}")
                time.sleep(30)
            except KeyboardInterrupt:
                logger.info("🛑 Shutting down system...")
                self.shutdown()
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(5)

    def shutdown(self):
        """Gracefully shutdown all processes."""
        self.running = False
        logger.info("🛑 Shutting down all processes...")
        for name, process in self.processes.items():
            if process.poll() is None:
                logger.info(f"Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {name}...")
                    process.kill()
        logger.info("✅ System shutdown complete")

def main():
    """Main function with command line argument support."""
    import argparse
    parser = argparse.ArgumentParser(description="Complete Trading System Manager")
    parser.add_argument("--log-dir", default="./logs/futures_env", 
                       help="Directory for logging (default: ./logs/futures_env)")
    parser.add_argument("--dashboard-port", type=int, default=8080,
                       help="Port for dashboard (default: 8080)")
    args = parser.parse_args()
    
    system = SystemManager(log_dir=args.log_dir, dashboard_port=args.dashboard_port)
    
    signal.signal(signal.SIGINT, lambda sig, frame: (system.shutdown(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda sig, frame: (system.shutdown(), sys.exit(0)))
    
    try:
        if system.start_complete_system():
            system.monitor_system()
        else:
            logger.error("❌ Failed to start system")
            return 1
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        system.shutdown()
        return 1
    return 0

if __name__ == "__main__":
    main()
