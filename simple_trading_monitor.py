
#!/usr/bin/env python
"""
Enhanced Trading Activity Monitor
================================
A comprehensive web dashboard to monitor trading activity with dual table support.
"""

import json
import time
from pathlib import Path
from flask import Flask, render_template_string, jsonify
import threading

app = Flask(__name__)

# Enhanced HTML template with dual table support
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Trading Activity Monitor</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
        .header { background: #333; padding: 20px; border-radius: 8px; margin-bottom: 20px; text-align: center; }
        .stats { display: flex; gap: 20px; margin-bottom: 20px; }
        .stat-card { background: #2a2a2a; padding: 15px; border-radius: 8px; flex: 1; text-align: center; }
        .stat-value { font-size: 24px; font-weight: bold; color: #4CAF50; }
        .chart-container { background: #2a2a2a; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .table-container { background: #2a2a2a; border-radius: 8px; overflow: hidden; margin-bottom: 20px; }
        .table-tabs { background: #333; display: flex; }
        .tab-button { background: none; color: #fff; border: none; padding: 10px 20px; cursor: pointer; }
        .tab-button.active { background: #4CAF50; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #444; font-size: 12px; }
        th { background: #333; font-weight: bold; }
        .buy { color: #4CAF50; }
        .sell { color: #f44336; }
        .hold { color: #FFC107; }
        .profit { background-color: rgba(76, 175, 80, 0.1); }
        .loss { background-color: rgba(244, 67, 54, 0.1); }
        .refresh-btn { background: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin: 5px; }
        .refresh-btn:hover { background: #45a049; }
        .live-indicator { width: 8px; height: 8px; background: #4CAF50; border-radius: 50%; display: inline-block; animation: pulse 1s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Enhanced Trading Activity Monitor</h1>
        <div>
            <span class="live-indicator"></span> LIVE MONITORING
            <button class="refresh-btn" onclick="refreshData()">Refresh</button>
            <span id="last-update"></span>
        </div>
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
        <div class="stat-card">
            <div>Win Rate</div>
            <div class="stat-value" id="win-rate">0%</div>
        </div>
    </div>
    
    <div class="chart-container">
        <h3>📈 Account Equity Chart</h3>
        <div id="equity-chart" style="height: 300px;"></div>
    </div>
    
    <div class="chart-container">
        <h3>📊 P&L Distribution</h3>
        <div id="pnl-chart" style="height: 300px;"></div>
    </div>
    
    <div class="table-container">
        <div class="table-tabs">
            <button class="tab-button active" onclick="showTab('trading')">Trading Activity</button>
            <button class="tab-button" onclick="showTab('trades')">Trade Outcomes</button>
        </div>
        
        <div id="trading-tab" class="tab-content active">
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
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="trading-table">
                    <tr><td colspan="9">Loading...</td></tr>
                </tbody>
            </table>
        </div>
        
        <div id="trades-tab" class="tab-content">
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Step</th>
                        <th>Action</th>
                        <th>Price</th>
                        <th>Position</th>
                        <th>Type</th>
                        <th>Change</th>
                        <th>Balance</th>
                        <th>P&L</th>
                        <th>Equity</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="trades-table">
                    <tr><td colspan="11">Loading...</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab-button').forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
        }
        
        function refreshData() {
            fetch('/api/trading-data')
                .then(response => response.json())
                .then(data => {
                    updateStats(data.stats);
                    updateTradingTable(data.trading_data);
                    updateTradesTable(data.trades_data);
                    updateCharts(data);
                    document.getElementById('last-update').textContent = 'Updated: ' + new Date().toLocaleTimeString();
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('last-update').innerHTML = '<span style="color: #f44336;">Connection Error</span>';
                });
        }
        
        function updateStats(stats) {
            document.getElementById('total-trades').textContent = stats.total_trades;
            document.getElementById('current-position').textContent = stats.current_position;
            document.getElementById('account-balance').textContent = stats.account_balance;
            document.getElementById('total-pnl').textContent = stats.total_pnl;
            document.getElementById('win-rate').textContent = stats.win_rate + '%';
        }
        
        function updateTradingTable(data) {
            const tbody = document.getElementById('trading-table');
            if (!data || data.length === 0) {
                tbody.innerHTML = '<tr><td colspan="9">No trading data available</td></tr>';
                return;
            }
            
            tbody.innerHTML = data.slice(-100).reverse().map(trade => {
                const statusClass = trade.status === 'PROFIT' ? 'profit' : trade.status === 'LOSS' ? 'loss' : '';
                return `
                <tr class="${statusClass}">
                    <td>${trade.timestamp.split(' ')[1]}</td>
                    <td>${trade.step}</td>
                    <td class="${trade.action.toLowerCase()}">${trade.action}</td>
                    <td>$${trade.price.toFixed(2)}</td>
                    <td>${trade.position}</td>
                    <td>${trade.pos_change}</td>
                    <td>$${trade.balance.toLocaleString()}</td>
                    <td>$${trade.pnl.toFixed(2)}</td>
                    <td>${trade.status}</td>
                </tr>
            `;}).join('');
        }
        
        function updateTradesTable(data) {
            const tbody = document.getElementById('trades-table');
            if (!data || data.length === 0) {
                tbody.innerHTML = '<tr><td colspan="11">No trades data available</td></tr>';
                return;
            }
            
            tbody.innerHTML = data.slice(-100).reverse().map(trade => {
                const statusClass = trade.status === 'PROFIT' ? 'profit' : trade.status === 'LOSS' ? 'loss' : '';
                return `
                <tr class="${statusClass}">
                    <td>${trade.timestamp.split(' ')[1]}</td>
                    <td>${trade.step}</td>
                    <td class="${trade.action.toLowerCase()}">${trade.action}</td>
                    <td>$${trade.price.toFixed(2)}</td>
                    <td>${trade.position}</td>
                    <td>${trade.position_type}</td>
                    <td>${trade.position_change}</td>
                    <td>$${trade.balance.toLocaleString()}</td>
                    <td>$${trade.pnl.toFixed(2)}</td>
                    <td>$${trade.account_equity.toLocaleString()}</td>
                    <td>${trade.status}</td>
                </tr>
            `;}).join('');
        }
        
        function updateCharts(data) {
            // Update equity chart
            if (data.trading_data && data.trading_data.length > 0) {
                const equityData = data.trading_data.map((trade, index) => ({
                    x: index,
                    y: trade.balance
                }));
                
                Plotly.newPlot('equity-chart', [{
                    x: equityData.map(d => d.x),
                    y: equityData.map(d => d.y),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Account Balance',
                    line: { color: '#4CAF50', width: 2 }
                }], {
                    title: 'Account Balance Over Time',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: 'white' },
                    xaxis: { gridcolor: '#444', title: 'Trade Number' },
                    yaxis: { gridcolor: '#444', title: 'Balance ($)' },
                    margin: { t: 40, r: 20, b: 40, l: 60 }
                });
            }
            
            // Update P&L distribution chart
            if (data.trades_data && data.trades_data.length > 0) {
                const pnlData = data.trades_data
                    .filter(trade => trade.pnl !== 0)
                    .map(trade => trade.pnl);
                
                if (pnlData.length > 0) {
                    Plotly.newPlot('pnl-chart', [{
                        x: pnlData,
                        type: 'histogram',
                        name: 'P&L Distribution',
                        marker: { color: '#4CAF50', opacity: 0.7 }
                    }], {
                        title: 'P&L Distribution',
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: 'white' },
                        xaxis: { gridcolor: '#444', title: 'P&L ($)' },
                        yaxis: { gridcolor: '#444', title: 'Frequency' },
                        margin: { t: 40, r: 20, b: 40, l: 60 }
                    });
                }
            }
        }
        
        // Auto-refresh every 3 seconds
        setInterval(refreshData, 3000);
        
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
    """Get latest trading data from both JSON files."""
    try:
        # Look for both trading table files
        logs_dir = Path('./logs')
        trading_files = list(logs_dir.glob('**/trading_table.json'))
        trades_files = list(logs_dir.glob('**/trades_table.json'))
        
        trading_data = []
        trades_data = []
        
        # Read trading activity data
        for file_path in trading_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    trading_data.extend(data)
            except (json.JSONDecodeError, FileNotFoundError):
                continue
        
        # Read trades data
        for file_path in trades_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    trades_data.extend(data)
            except (json.JSONDecodeError, FileNotFoundError):
                continue
        
        # Calculate comprehensive stats
        stats = {
            'total_trades': 0,
            'current_position': 0,
            'account_balance': '$100,000.00',
            'total_pnl': '$0.00',
            'win_rate': '0'
        }
        
        if trading_data:
            # Sort by timestamp
            trading_data.sort(key=lambda x: x.get('timestamp', ''))
            latest_trade = trading_data[-1]
            
            # Calculate stats
            actual_trades = [t for t in trading_data if t.get('action') != 'HOLD']
            profitable_trades = [t for t in trading_data if t.get('status') == 'PROFIT']
            
            stats = {
                'total_trades': len(actual_trades),
                'current_position': latest_trade.get('position', 0),
                'account_balance': f"${latest_trade.get('balance', 100000):,.2f}",
                'total_pnl': f"${sum([t.get('pnl', 0) for t in trading_data]):,.2f}",
                'win_rate': f"{(len(profitable_trades) / max(len(actual_trades), 1) * 100):.1f}"
            }
        
        if trades_data:
            trades_data.sort(key=lambda x: x.get('timestamp', ''))
        
        return jsonify({
            'trading_data': trading_data,
            'trades_data': trades_data,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({
            'trading_data': [],
            'trades_data': [],
            'stats': {
                'total_trades': 0,
                'current_position': 0,
                'account_balance': '$0.00',
                'total_pnl': '$0.00',
                'win_rate': '0'
            },
            'error': str(e)
        })

def run_monitor():
    """Run the enhanced trading monitor on port 8080."""
    app.run(host='0.0.0.0', port=8080, debug=False)

if __name__ == '__main__':
    print("🚀 Starting Enhanced Trading Activity Monitor...")
    print("📊 Dashboard: http://0.0.0.0:8080")
    print("🔄 Auto-refreshes every 3 seconds")
    print("📋 Dual table view: Trading Activity + Trade Outcomes")
    run_monitor()
