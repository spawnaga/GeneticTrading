
#!/usr/bin/env python
"""
Simple Trading Activity Monitor
==============================
A lightweight web dashboard to monitor trading activity in real-time.
"""

import json
import time
from pathlib import Path
from flask import Flask, render_template_string, jsonify
import threading

app = Flask(__name__)

# HTML template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Trading Activity Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
        .header { background: #333; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .stats { display: flex; gap: 20px; margin-bottom: 20px; }
        .stat-card { background: #2a2a2a; padding: 15px; border-radius: 8px; flex: 1; }
        .stat-value { font-size: 24px; font-weight: bold; color: #4CAF50; }
        .table-container { background: #2a2a2a; border-radius: 8px; overflow: hidden; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #444; }
        th { background: #333; font-weight: bold; }
        .buy { color: #4CAF50; }
        .sell { color: #f44336; }
        .hold { color: #FFC107; }
        .refresh-btn { background: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        .refresh-btn:hover { background: #45a049; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Trading Activity Monitor</h1>
        <button class="refresh-btn" onclick="refreshData()">Refresh</button>
        <span id="last-update"></span>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <div>Total Trades</div>
            <div class="stat-value" id="total-trades">0</div>
        </div>
        <div class="stat-card">
            <div>Current Position</div>
            <div class="stat-value" id="current-position">0</div>
        </div>
        <div class="stat-card">
            <div>Account Balance</div>
            <div class="stat-value" id="account-balance">$0.00</div>
        </div>
        <div class="stat-card">
            <div>Total P&L</div>
            <div class="stat-value" id="total-pnl">$0.00</div>
        </div>
    </div>
    
    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Step</th>
                    <th>Action</th>
                    <th>Price</th>
                    <th>Position</th>
                    <th>Change</th>
                    <th>Balance</th>
                    <th>P&L</th>
                </tr>
            </thead>
            <tbody id="trades-table">
                <tr><td colspan="8">Loading...</td></tr>
            </tbody>
        </table>
    </div>

    <script>
        function refreshData() {
            fetch('/api/trading-data')
                .then(response => response.json())
                .then(data => {
                    updateStats(data.stats);
                    updateTable(data.trades);
                    document.getElementById('last-update').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
                })
                .catch(error => console.error('Error:', error));
        }
        
        function updateStats(stats) {
            document.getElementById('total-trades').textContent = stats.total_trades;
            document.getElementById('current-position').textContent = stats.current_position;
            document.getElementById('account-balance').textContent = stats.account_balance;
            document.getElementById('total-pnl').textContent = stats.total_pnl;
        }
        
        function updateTable(trades) {
            const tbody = document.getElementById('trades-table');
            if (trades.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8">No trading data available</td></tr>';
                return;
            }
            
            tbody.innerHTML = trades.slice(-50).reverse().map(trade => `
                <tr>
                    <td>${trade.timestamp.split(' ')[1]}</td>
                    <td>${trade.step}</td>
                    <td class="${trade.action.toLowerCase()}">${trade.action}</td>
                    <td>${trade.price}</td>
                    <td>${trade.position}</td>
                    <td>${trade.pos_change}</td>
                    <td>${trade.balance}</td>
                    <td>${trade.pnl}</td>
                </tr>
            `).join('');
        }
        
        // Auto-refresh every 2 seconds
        setInterval(refreshData, 2000);
        
        // Initial load
        refreshData();
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/trading-data')
def get_trading_data():
    """Get latest trading data from JSON files."""
    try:
        # Look for trading table files
        logs_dir = Path('./logs')
        trading_files = list(logs_dir.glob('**/trading_table.json'))
        
        all_trades = []
        stats = {
            'total_trades': 0,
            'current_position': 0,
            'account_balance': '$0.00',
            'total_pnl': '$0.00'
        }
        
        for file_path in trading_files:
            try:
                with open(file_path, 'r') as f:
                    trades = json.load(f)
                    all_trades.extend(trades)
            except (json.JSONDecodeError, FileNotFoundError):
                continue
        
        if all_trades:
            # Sort by timestamp
            all_trades.sort(key=lambda x: x.get('timestamp', ''))
            
            # Calculate stats from latest trade
            latest_trade = all_trades[-1]
            stats = {
                'total_trades': len([t for t in all_trades if t.get('action') != 'HOLD']),
                'current_position': latest_trade.get('position', 0),
                'account_balance': latest_trade.get('balance', '$0.00'),
                'total_pnl': latest_trade.get('pnl', '$0.00')
            }
        
        return jsonify({
            'trades': all_trades,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({
            'trades': [],
            'stats': {
                'total_trades': 0,
                'current_position': 0,
                'account_balance': '$0.00',
                'total_pnl': '$0.00'
            },
            'error': str(e)
        })

def run_monitor():
    """Run the trading monitor on port 8080."""
    app.run(host='0.0.0.0', port=8080, debug=False)

if __name__ == '__main__':
    print("🚀 Starting Trading Activity Monitor...")
    print("📊 Dashboard: http://0.0.0.0:8080")
    print("🔄 Auto-refreshes every 2 seconds")
    run_monitor()
