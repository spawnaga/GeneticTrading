
#!/usr/bin/env python
"""
Run Trading Dashboard
====================

Starts the web dashboard at http://0.0.0.0:5000
All trading logs will be displayed in the web interface only.
"""

import os
import sys
import subprocess
import signal
import time

def start_dashboard():
    """Start the dashboard server"""
    print("🚀 Starting Trading Dashboard...")
    print("📊 Dashboard will be available at: http://0.0.0.0:5000")
    print("🔕 All trading output will be shown in the web interface only")
    print("\n" + "="*60)
    
    try:
        # Start the dashboard server
        process = subprocess.Popen([sys.executable, "dashboard_server.py"])
        
        # Wait a moment for server to start
        time.sleep(3)
        
        print("✅ Dashboard server is running!")
        print("🌐 Open http://0.0.0.0:5000 in your browser")
        print("📈 Trading logs will appear in the web dashboard")
        print("\nPress Ctrl+C to stop the dashboard")
        
        # Keep running
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping dashboard server...")
        process.terminate()
        print("✅ Dashboard stopped")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

if __name__ == "__main__":
    start_dashboard()
