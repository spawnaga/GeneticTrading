
#!/usr/bin/env python
"""
Standalone Training Monitor
==========================

Run this script in a separate terminal to monitor training progress in real-time.
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime

def monitor_training(log_dir="./logs", refresh_interval=10):
    """Monitor training progress in real-time."""
    metrics_file = Path(log_dir) / "training_metrics.json"
    
    print("🔍 Training Monitor Started")
    print(f"📁 Watching: {metrics_file}")
    print(f"🔄 Refresh interval: {refresh_interval} seconds")
    print("=" * 60)
    
    last_iteration = 0
    
    while True:
        try:
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    
                history = data.get('performance_history', [])
                warnings = data.get('warning_flags', [])
                
                if history and len(history) > last_iteration:
                    # Show new iterations
                    for entry in history[last_iteration:]:
                        iteration = entry['iteration']
                        method = entry['method']
                        performance = entry['performance']
                        training_time = entry['training_time']
                        metrics = entry['metrics']
                        
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                              f"Iter {iteration:2d} | {method:3s} | "
                              f"Perf: {performance:6.4f} | "
                              f"Time: {training_time/60:4.1f}m | "
                              f"CAGR: {metrics.get('cagr', 0):5.1f}%")
                    
                    last_iteration = len(history)
                    
                    # Show warnings if any
                    if len(warnings) > 0:
                        print(f"⚠️  {len(warnings)} warnings detected!")
                        for warning in warnings[-3:]:  # Show last 3 warnings
                            print(f"   └─ {warning}")
                            
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for training to start...")
                
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No valid metrics found yet...")
            
        except KeyboardInterrupt:
            print("\n👋 Monitor stopped by user")
            break
            
        time.sleep(refresh_interval)

if __name__ == "__main__":
    monitor_training()
