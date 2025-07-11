
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemManager:
    """Manages the complete trading system startup and monitoring."""
    
    def __init__(self):
        self.processes = {}
        self.running = True
        
    def start_tensorboard(self):
        """Start TensorBoard first."""
        logger.info("🚀 Starting TensorBoard...")
        
        # Ensure directories exist
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
            sys.executable, "main.py",
            "--data-folder", "./data_txt",
            "--max-rows", "50000",
            "--adaptive-iterations", "10",
            "--log-level", "INFO",
            "--training-mode", "adaptive"
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
            
            # Monitor training output in background
            def monitor_training():
                while self.running and process.poll() is None:
                    line = process.stdout.readline()
                    if line:
                        print(f"[TRAINING] {line.strip()}")
                        
                        # Log key metrics
                        if "Performance:" in line:
                            logger.info(f"📊 {line.strip()}")
                        elif "CAGR:" in line:
                            logger.info(f"💰 {line.strip()}")
                        elif "Sharpe:" in line:
                            logger.info(f"📈 {line.strip()}")
            
            threading.Thread(target=monitor_training, daemon=True).start()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start training: {e}")
            return False
    
    def start_dashboard(self):
        """Start the monitoring dashboard."""
        logger.info("📊 Starting dashboard...")
        
        cmd = [sys.executable, "standalone_dashboard.py"]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            self.processes['dashboard'] = process
            logger.info("✅ Dashboard started on http://0.0.0.0:5000")
            
            # Monitor dashboard output
            def monitor_dashboard():
                while self.running and process.poll() is None:
                    line = process.stdout.readline()
                    if line and "ERROR" in line:
                        logger.warning(f"[DASHBOARD] {line.strip()}")
            
            threading.Thread(target=monitor_dashboard, daemon=True).start()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start dashboard: {e}")
            return False
    
    def wait_for_tensorboard(self, max_wait=30):
        """Wait for TensorBoard to be ready."""
        import requests
        
        for i in range(max_wait):
            try:
                response = requests.get("http://localhost:6006", timeout=2)
                if response.status_code == 200:
                    logger.info("✅ TensorBoard is ready!")
                    return True
            except:
                pass
            
            logger.info(f"⏳ Waiting for TensorBoard... ({i+1}/{max_wait})")
            time.sleep(1)
        
        logger.warning("⚠️ TensorBoard may not be fully ready")
        return False
    
    def start_complete_system(self):
        """Start the complete system in proper order."""
        logger.info("🚀 Starting Complete NQ Trading System")
        logger.info("=" * 60)
        
        # Step 1: Start TensorBoard
        if not self.start_tensorboard():
            return False
        
        # Step 2: Wait for TensorBoard to be ready
        self.wait_for_tensorboard()
        
        # Step 3: Start training
        if not self.start_training():
            return False
        
        # Step 4: Wait a bit for training to generate initial data
        logger.info("⏳ Waiting for training to initialize...")
        time.sleep(10)
        
        # Step 5: Start dashboard
        if not self.start_dashboard():
            return False
        
        logger.info("🎉 System startup complete!")
        logger.info("📊 Dashboard: http://0.0.0.0:5000")
        logger.info("📈 TensorBoard: http://0.0.0.0:6006")
        logger.info("🧠 Training: Active")
        logger.info("=" * 60)
        
        return True
    
    def monitor_system(self):
        """Monitor all processes and provide status updates."""
        while self.running:
            try:
                status = {}
                for name, process in self.processes.items():
                    if process.poll() is None:
                        status[name] = "🟢 Running"
                    else:
                        status[name] = "🔴 Stopped"
                
                # Print status every 30 seconds
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
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {name}...")
                    process.kill()
        
        logger.info("✅ System shutdown complete")

def main():
    """Main function."""
    system = SystemManager()
    
    # Setup signal handling
    def signal_handler(sig, frame):
        system.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the complete system
        if system.start_complete_system():
            # Monitor the system
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
