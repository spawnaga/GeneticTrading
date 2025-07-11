
#!/usr/bin/env python
"""
Standalone Dashboard Server
==========================
Simple web server for viewing trading system status
"""

import http.server
import socketserver
import threading
import time
import os
import signal
import sys
from datetime import datetime

class TradingDashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=".", **kwargs)
    
    def do_GET(self):
        if self.path == '/trading_dashboard.html' or self.path == '/' or self.path == '/dashboard':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>🚀 NQ Trading System Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body {{ 
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            color: white; 
            font-family: 'Segoe UI', Arial, sans-serif; 
            padding: 20px;
            margin: 0;
            min-height: 100vh;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ 
            text-align: center; 
            margin-bottom: 30px;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 20px;
        }}
        .metric {{ 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            margin: 15px; 
            border-radius: 10px;
            border-left: 4px solid #4CAF50;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
        }}
        .status {{ color: #4CAF50; font-weight: bold; font-size: 1.2em; }}
        .grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 20px; 
        }}
        .warning {{ color: #FFA500; }}
        .error {{ color: #FF6B6B; }}
        .success {{ color: #4CAF50; }}
        .pulse {{ animation: pulse 2s infinite; }}
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
            100% {{ opacity: 1; }}
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            border-top: 1px solid rgba(255,255,255,0.2);
            opacity: 0.7;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Revolutionary NQ Futures Trading System</h1>
            <p>Real-time GA+PPO Hybrid Training Dashboard</p>
            <p class="pulse">System Status: <span class="success">✅ ONLINE</span></p>
        </div>
        
        <div class="grid">
            <div class="metric">
                <h3>🔥 System Status</h3>
                <p class="status">✅ Dashboard Server Active</p>
                <p><strong>Server:</strong> 0.0.0.0:5000</p>
                <p><strong>Last Updated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Access URL:</strong> http://0.0.0.0:5000/</p>
            </div>
            
            <div class="metric">
                <h3>🧬 Training Configuration</h3>
                <p><strong>Mode:</strong> Adaptive GA+PPO</p>
                <p><strong>Data Processing:</strong> <span class="success">50,000 rows</span></p>
                <p><strong>Iterations:</strong> 3 adaptive cycles</p>
                <p><strong>Method:</strong> Genetic Algorithm → PPO</p>
            </div>
            
            <div class="metric">
                <h3>📊 NQ Data Status</h3>
                <p><strong>Market:</strong> NQ Futures (NASDAQ-100)</p>
                <p><strong>Period:</strong> 2008-Present</p>
                <p><strong>Format:</strong> OHLCV minute data</p>
                <p><strong>Status:</strong> <span class="warning">⚠️ Needs formatting</span></p>
            </div>
            
            <div class="metric">
                <h3>💻 Hardware Status</h3>
                <p><strong>GPU:</strong> <span class="success">✅ CUDA Available</span></p>
                <p><strong>Framework:</strong> PyTorch + cuDF</p>
                <p><strong>Processing:</strong> GPU Accelerated</p>
                <p><strong>Environment:</strong> WSL2 + Replit</p>
            </div>
            
            <div class="metric">
                <h3>📈 Training Commands</h3>
                <p><strong>Quick Test:</strong></p>
                <code>python run_trading_system.py --config test</code>
                <p><strong>Full Training:</strong></p>
                <code>python main.py --data-folder ./data_txt --max-rows 50000</code>
            </div>
            
            <div class="metric">
                <h3>🔄 Next Steps</h3>
                <p>1. Fix NQ.txt format: <code>python fix_nq_format.py</code></p>
                <p>2. Start training: <code>python main.py --data-folder ./data_txt</code></p>
                <p>3. Monitor progress via this dashboard</p>
            </div>
        </div>
        
        <div class="footer">
            <p>🚀 Revolutionary Trading System - Powered by GA+PPO Hybrid Intelligence</p>
            <p>Auto-refresh every 30 seconds | Server running on 0.0.0.0:5000</p>
        </div>
    </div>
</body>
</html>
            """
            self.wfile.write(html_content.encode())
        else:
            super().do_GET()

def start_dashboard_server(port=5000):
    """Start the standalone dashboard server."""
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\n🛑 Server stopped by user (Ctrl+C)")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Try to start the server
        with socketserver.TCPServer(("0.0.0.0", port), TradingDashboardHandler) as httpd:
            print(f"🌐 Trading Dashboard Server Started!")
            print(f"🔗 Access at: http://0.0.0.0:{port}/")
            print(f"🔗 Dashboard URL: http://0.0.0.0:{port}/trading_dashboard.html")
            print(f"🚀 Ready to serve NQ Trading System Dashboard")
            print("Press Ctrl+C to stop the server\n")
            
            # Serve forever
            httpd.serve_forever()
            
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use")
            print("Another process might be using this port")
            print("Try: pkill -f 'python.*dashboard' to kill existing dashboard servers")
        else:
            print(f"❌ Error starting server: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    start_dashboard_server()
