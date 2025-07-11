
#!/usr/bin/env python
"""
Quick Dashboard Startup Script
"""
import subprocess
import sys
import threading
import time

def start_dashboard():
    """Start the dashboard server."""
    print("🚀 Starting NQ Trading Dashboard...")
    try:
        # Start the standalone dashboard
        subprocess.run([sys.executable, "standalone_dashboard.py"])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped")

if __name__ == "__main__":
    start_dashboard()
