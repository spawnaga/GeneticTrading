
#!/usr/bin/env python
"""
Real Trading Dashboard Server - Fixed Version
============================================

Reads actual trading logs and displays them in web dashboard.
No console output - everything goes to http://0.0.0.0:5000
"""

import json
import time
import threading
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Web framework
from flask import Flask, render_template_string, jsonify

# Disable Flask console output
import os
import sys
os.environ['WERKZEUG_RUN_MAIN'] = 'true'

# Setup logging to file only
log_dir = Path("./logs")
log_dir.mkdir(exist_ok=True)

# Configure logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'dashboard.log'),
    ]
)

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Suppress Flask startup messages
cli = sys.modules['flask.cli']
cli.show_server_banner = lambda *x: None

# HTML template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>🎯 Live Trading Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { 
            background-color: #0d1117; 
            color: white; 
            font-family: 'Arial', sans-serif; 
            margin: 0; 
            padding: 20px; 
        }
        .header { 
            text-align: center; 
            margin-bottom: 30px; 
            background: linear-gradient(135deg, #1e3a8a, #3b82f6);
            padding: 20px;
            border-radius: 10px;
        }
        .status-badge {
            background: #10b981;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin: 10px;
        }
        .metrics-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin-bottom: 30px; 
        }
        .metric-card { 
            background: linear-gradient(135deg, #1f2937, #374151); 
            padding: 20px; 
            border-radius: 10px; 
            border: 1px solid #4b5563; 
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metric-card h3 {
            margin-top: 0;
            color: #60a5fa;
        }
        .chart-container { 
            background: linear-gradient(135deg, #1f2937, #374151); 
            padding: 20px; 
            border-radius: 10px; 
            margin-bottom: 20px; 
            border: 1px solid #4b5563;
        }
        .status-good { color: #10b981; font-weight: bold; }
        .status-warning { color: #f59e0b; font-weight: bold; }
        .status-error { color: #ef4444; font-weight: bold; }
        .live-indicator {
            width: 10px;
            height: 10px;
            background: #10b981;
            border-radius: 50%;
            display: inline-block;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .log-container {
            background: #111827;
            border: 1px solid #374151;
            border-radius: 8px;
            padding: 15px;
            height: 400px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 12px;
        }
        .log-entry {
            margin: 2px 0;
            padding: 2px 5px;
            border-radius: 3px;
        }
        .log-buy { background-color: rgba(16, 185, 129, 0.1); color: #10b981; }
        .log-sell { background-color: rgba(239, 68, 68, 0.1); color: #ef4444; }
        .log-hold { background-color: rgba(107, 114, 128, 0.1); color: #9ca3af; }
        .big-number {
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Live NQ Futures Trading Dashboard</h1>
        <div class="status-badge">
            <span class="live-indicator"></span> LIVE MONITORING
        </div>
        <p>Real-time GA + PPO adaptive training system</p>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <h3>💰 Account Performance</h3>
            <div class="big-number" id="account-value">$100,000</div>
            <div id="total-return">Return: <span class="status-good">+0.00%</span></div>
            <div id="total-trades">Total Trades: 0</div>
        </div>
        
        <div class="metric-card">
            <h3>📊 Current Position</h3>
            <div class="big-number" id="current-position">0</div>
            <div id="position-type">Position: FLAT</div>
            <div id="unrealized-pnl">Unrealized: $0.00</div>
        </div>
        
        <div class="metric-card">
            <h3>🤖 AI Algorithm</h3>
            <div class="big-number" id="current-algorithm">GA</div>
            <div id="algorithm-performance">Performance: Monitoring...</div>
            <div id="last-update">Updated: <span id="update-time">--:--:--</span></div>
        </div>
        
        <div class="metric-card">
            <h3>⚡ System Health</h3>
            <div class="big-number status-good" id="system-health">94.2%</div>
            <div>Status: <span class="status-good">ACTIVE</span></div>
            <div id="active-sessions">Sessions: 1</div>
        </div>
    </div>
    
    <div class="chart-container">
        <h3>📈 Account Value Evolution</h3>
        <div id="equity-chart" style="height: 400px;"></div>
    </div>
    
    <div class="chart-container">
        <h3>📋 Live Trading Activity</h3>
        <div class="log-container" id="trading-logs">
            <div class="log-entry">Waiting for trading data...</div>
        </div>
    </div>

    <script>
        let equityData = {
            x: [],
            y: [],
            type: 'scatter',
            mode: 'lines',
            name: 'Account Value',
            line: { color: '#10b981', width: 3 },
            fill: 'tonexty',
            fillcolor: 'rgba(16, 185, 129, 0.1)'
        };

        function updateDashboard() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    // Update metrics
                    document.getElementById('account-value').textContent = 
                        `$${data.account_value.toLocaleString()}`;
                    document.getElementById('total-return').innerHTML = 
                        `Return: <span class="${data.total_return >= 0 ? 'status-good' : 'status-error'}">${data.total_return >= 0 ? '+' : ''}${data.total_return.toFixed(2)}%</span>`;
                    document.getElementById('total-trades').textContent = 
                        `Total Trades: ${data.total_trades}`;
                    
                    document.getElementById('current-position').textContent = data.current_position;
                    document.getElementById('position-type').textContent = 
                        `Position: ${data.current_position == 0 ? 'FLAT' : data.current_position > 0 ? 'LONG' : 'SHORT'}`;
                    document.getElementById('unrealized-pnl').innerHTML = 
                        `Unrealized: <span class="${data.unrealized_pnl >= 0 ? 'status-good' : 'status-error'}">$${data.unrealized_pnl.toLocaleString()}</span>`;
                    
                    document.getElementById('current-algorithm').textContent = data.algorithm;
                    document.getElementById('update-time').textContent = new Date().toLocaleTimeString();

                    // Update chart
                    if (data.equity_history && data.equity_history.length > 0) {
                        equityData.x = data.timestamps;
                        equityData.y = data.equity_history;
                        
                        Plotly.newPlot('equity-chart', [equityData], {
                            title: 'Real-time Account Performance',
                            paper_bgcolor: 'rgba(0,0,0,0)',
                            plot_bgcolor: 'rgba(0,0,0,0)',
                            font: { color: 'white', family: 'Arial' },
                            xaxis: { 
                                gridcolor: '#374151', 
                                title: 'Time',
                                color: 'white'
                            },
                            yaxis: { 
                                gridcolor: '#374151', 
                                title: 'Account Value ($)',
                                color: 'white',
                                tickformat: '$,.0f'
                            },
                            margin: { t: 50, r: 20, b: 50, l: 80 }
                        });
                    }
                })
                .catch(error => {
                    console.error('Error fetching data:', error);
                    document.getElementById('update-time').innerHTML = 
                        '<span class="status-error">Connection Error</span>';
                });
                
            // Update trading logs
            fetch('/api/trading-logs')
                .then(response => response.json())
                .then(data => {
                    updateTradingLogs(data);
                })
                .catch(error => {
                    console.error('Error fetching logs:', error);
                });
        }
        
        function updateTradingLogs(logs) {
            const logsContainer = document.getElementById('trading-logs');
            
            if (!logs || logs.length === 0) {
                logsContainer.innerHTML = '<div class="log-entry">No trading activity detected...</div>';
                return;
            }
            
            // Show last 50 entries
            const recentLogs = logs.slice(-50);
            
            logsContainer.innerHTML = recentLogs.map(log => {
                let className = 'log-entry';
                if (log.includes('BUY')) className += ' log-buy';
                else if (log.includes('SELL')) className += ' log-sell';
                else if (log.includes('HOLD')) className += ' log-hold';
                
                return `<div class="${className}">${log}</div>`;
            }).join('');
            
            // Auto-scroll to bottom
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }

        // Update every 2 seconds
        setInterval(updateDashboard, 2000);
        updateDashboard(); // Initial load
    </script>
</body>
</html>
"""

class TradingDashboardServer:
    def __init__(self, port=5000):
        self.port = port
        self.log_dir = Path("./logs")
        self.current_metrics = {
            'account_value': 100000,
            'total_return': 0.0,
            'unrealized_pnl': 0.0,
            'current_position': 0,
            'total_trades': 0,
            'algorithm': 'GA',
            'equity_history': [100000],
            'timestamps': [datetime.now().isoformat()],
        }

    def start_server(self):
        """Start the dashboard server with no console output"""
        @app.route('/')
        def dashboard():
            return render_template_string(DASHBOARD_HTML)

        @app.route('/api/metrics')
        def get_metrics():
            return jsonify(self._get_current_metrics())
            
        @app.route('/api/trading-logs')
        def get_trading_logs():
            return jsonify(self._get_trading_logs())

        # Start server in a separate thread with no output
        def run_server():
            app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False)

        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
        
        logger.info(f"Dashboard server started at http://0.0.0.0:{self.port}")
        return server_thread

    def _get_current_metrics(self):
        """Get current metrics from log files"""
        try:
            # Read trading table
            trading_table_file = self.log_dir / "trading_table.json"
            if trading_table_file.exists():
                with open(trading_table_file, 'r') as f:
                    trading_data = json.load(f)
                
                if trading_data:
                    latest = trading_data[-1]
                    # Extract real metrics
                    self.current_metrics.update({
                        'account_value': float(latest.get('balance', 100000)),
                        'current_position': int(latest.get('position', 0)),
                        'total_trades': len([t for t in trading_data if t.get('action') in ['BUY', 'SELL']]),
                        'total_return': ((float(latest.get('balance', 100000)) - 100000) / 100000) * 100,
                    })
                    
                    # Build equity history
                    if len(trading_data) > len(self.current_metrics['equity_history']):
                        self.current_metrics['equity_history'] = [float(t.get('balance', 100000)) for t in trading_data[-100:]]
                        self.current_metrics['timestamps'] = [t.get('timestamp', datetime.now().isoformat()) for t in trading_data[-100:]]

        except Exception as e:
            logger.debug(f"Could not read trading data: {e}")

        return self.current_metrics

    def _get_trading_logs(self):
        """Get recent trading logs from files"""
        logs = []
        try:
            # Read from trading activity log
            activity_log = self.log_dir / "trading_activity_rank_0.log"
            if activity_log.exists():
                with open(activity_log, 'r') as f:
                    lines = f.readlines()
                    # Get last 100 lines
                    logs = [line.strip() for line in lines[-100:] if line.strip()]
        except Exception as e:
            logger.debug(f"Could not read activity logs: {e}")
            
        return logs


if __name__ == "__main__":
    # Suppress all console output
    import warnings
    warnings.filterwarnings("ignore")
    
    server = TradingDashboardServer()
    server.start_server()
    
    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Dashboard server stopped")
