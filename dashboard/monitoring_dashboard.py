"""
Real-Time Monitoring Dashboard - Phase 5.2
Web-based dashboard for ML Market Maker system variables.

Author: Antigravity
Version: 1.0.0

Usage:
    python dashboard/monitoring_dashboard.py

Access at: http://localhost:5555
"""

import json
import time
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging

# Dashboard HTML Template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>ML Market Maker Dashboard</title>
    <meta charset="UTF-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { 
            text-align: center; 
            margin-bottom: 30px;
            font-size: 28px;
            color: #00d4ff;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
        }
        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        .card h2 {
            font-size: 14px;
            color: #888;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        .metric:last-child { border-bottom: none; }
        .metric-label { color: #aaa; font-size: 13px; }
        .metric-value { 
            font-weight: 600; 
            font-size: 15px;
            font-family: 'Consolas', monospace;
        }
        .positive { color: #00ff88; }
        .negative { color: #ff4444; }
        .warning { color: #ffaa00; }
        .neutral { color: #00d4ff; }
        .status-ok { color: #00ff88; }
        .status-error { color: #ff4444; }
        .big-number {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            padding: 20px 0;
        }
        .alert {
            background: rgba(255, 68, 68, 0.2);
            border: 1px solid #ff4444;
            border-radius: 8px;
            padding: 10px 15px;
            margin-bottom: 15px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        .progress-bar {
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            margin-top: 5px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        .timestamp { 
            text-align: center; 
            color: #666; 
            margin-top: 20px;
            font-size: 12px;
        }
        .regime-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
        .regime-low_vol { background: #2196F3; }
        .regime-high_vol { background: #f44336; }
        .regime-trend_up { background: #4CAF50; }
        .regime-trend_down { background: #ff9800; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 ML Market Maker Dashboard</h1>
        
        <div id="alerts"></div>
        
        <div class="grid">
            <!-- Performance -->
            <div class="card">
                <h2>💰 Performance</h2>
                <div class="big-number" id="total_pnl">$0.00</div>
                <div class="metric">
                    <span class="metric-label">Realized PnL</span>
                    <span class="metric-value" id="realized_pnl">$0.00</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Unrealized PnL</span>
                    <span class="metric-value" id="unrealized_pnl">$0.00</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Sharpe Ratio</span>
                    <span class="metric-value neutral" id="sharpe">0.00</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Trades</span>
                    <span class="metric-value" id="trades">0</span>
                </div>
            </div>
            
            <!-- Risk -->
            <div class="card">
                <h2>⚖️ Risk Metrics</h2>
                <div class="metric">
                    <span class="metric-label">Inventory</span>
                    <span class="metric-value" id="inventory">$0</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="inv_bar" style="width: 50%; background: #00d4ff;"></div>
                </div>
                <div class="metric" style="margin-top: 10px;">
                    <span class="metric-label">Max Drawdown</span>
                    <span class="metric-value warning" id="max_dd">0.00%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Position Limit</span>
                    <span class="metric-value" id="pos_limit">$5,000</span>
                </div>
            </div>
            
            <!-- Regime -->
            <div class="card">
                <h2>🎯 Market Regime</h2>
                <div style="text-align: center; padding: 15px 0;">
                    <span class="regime-badge" id="regime_badge">unknown</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Confidence</span>
                    <span class="metric-value neutral" id="regime_conf">0%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Liquidation Alert</span>
                    <span class="metric-value status-ok" id="liq_alert">✓ No</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Liq Severity</span>
                    <span class="metric-value" id="liq_severity">0.00</span>
                </div>
            </div>
            
            <!-- Orders -->
            <div class="card">
                <h2>📈 Order Parameters</h2>
                <div class="metric">
                    <span class="metric-label">Current Spread</span>
                    <span class="metric-value neutral" id="spread">0 bps</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Bandit Spread</span>
                    <span class="metric-value" id="bandit_spread">0 bps</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Bid Size</span>
                    <span class="metric-value positive" id="bid_size">$0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Ask Size</span>
                    <span class="metric-value negative" id="ask_size">$0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Fill Rate</span>
                    <span class="metric-value" id="fill_rate">0%</span>
                </div>
            </div>
            
            <!-- Adverse Selection -->
            <div class="card">
                <h2>🛡️ Adverse Selection</h2>
                <div class="metric">
                    <span class="metric-label">AS Probability</span>
                    <span class="metric-value" id="as_prob">0%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="as_bar" style="width: 0%; background: #ff4444;"></div>
                </div>
                <div class="metric" style="margin-top: 10px;">
                    <span class="metric-label">Spread Adjustment</span>
                    <span class="metric-value warning" id="as_adj">+0 bps</span>
                </div>
                <div class="metric">
                    <span class="metric-label">AS Trades Detected</span>
                    <span class="metric-value" id="as_trades">0</span>
                </div>
            </div>
            
            <!-- Funding -->
            <div class="card">
                <h2>💱 Funding & Bias</h2>
                <div class="metric">
                    <span class="metric-label">Funding Prediction</span>
                    <span class="metric-value" id="funding_pred">0.00%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Inventory Bias</span>
                    <span class="metric-value neutral" id="inv_bias">0.00</span>
                </div>
            </div>
            
            <!-- ML Models -->
            <div class="card">
                <h2>🤖 ML Model Status</h2>
                <div class="metric">
                    <span class="metric-label">Regime Model</span>
                    <span class="metric-value" id="model_regime">✗</span>
                </div>
                <div class="metric">
                    <span class="metric-label">AS Model</span>
                    <span class="metric-value" id="model_as">✗</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Fill Model</span>
                    <span class="metric-value" id="model_fill">✗</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Drift Detection</span>
                    <span class="metric-value" id="drift">✓</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Online Learning</span>
                    <span class="metric-value" id="online">Active</span>
                </div>
            </div>
            
            <!-- System -->
            <div class="card">
                <h2>⏱️ System Status</h2>
                <div class="metric">
                    <span class="metric-label">Uptime</span>
                    <span class="metric-value" id="uptime">0m</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Cycles</span>
                    <span class="metric-value" id="cycles">0</span>
                </div>
            </div>
        </div>
        
        <div class="timestamp" id="last_update">Last Update: -</div>
    </div>
    
    <script>
        function fetchMetrics() {
            fetch('/api/metrics')
                .then(r => r.json())
                .then(data => updateDashboard(data))
                .catch(e => console.error('Fetch error:', e));
        }
        
        function updateDashboard(m) {
            // Performance
            document.getElementById('total_pnl').textContent = m.total_pnl || '$0.00';
            document.getElementById('total_pnl').className = 'big-number ' + 
                (parseFloat(m.total_pnl?.replace('$','')) >= 0 ? 'positive' : 'negative');
            document.getElementById('realized_pnl').textContent = m.realized_pnl || '$0.00';
            document.getElementById('unrealized_pnl').textContent = m.unrealized_pnl || '$0.00';
            document.getElementById('sharpe').textContent = m.sharpe_ratio || '0.00';
            document.getElementById('trades').textContent = m.trades || '0';
            
            // Risk
            document.getElementById('inventory').textContent = m.inventory || '$0';
            document.getElementById('max_dd').textContent = m.max_drawdown || '0%';
            
            // Inventory bar
            const invRatio = parseFloat(m.inventory_ratio?.replace('%','')) || 0;
            const invBar = document.getElementById('inv_bar');
            invBar.style.width = Math.abs(invRatio) + '%';
            invBar.style.background = invRatio > 0 ? '#00ff88' : '#ff4444';
            
            // Regime
            const regime = m.regime || 'unknown';
            const regimeBadge = document.getElementById('regime_badge');
            regimeBadge.textContent = regime;
            regimeBadge.className = 'regime-badge regime-' + regime.toLowerCase();
            document.getElementById('regime_conf').textContent = m.regime_confidence || '0%';
            document.getElementById('liq_alert').innerHTML = m.liq_alert || '✓ No';
            document.getElementById('liq_alert').className = 'metric-value ' + 
                (m.liq_alert?.includes('YES') ? 'status-error' : 'status-ok');
            
            // Orders
            document.getElementById('spread').textContent = m.spread || '0 bps';
            document.getElementById('bandit_spread').textContent = m.bandit_spread || '0 bps';
            document.getElementById('bid_size').textContent = m.bid_size || '$0';
            document.getElementById('ask_size').textContent = m.ask_size || '$0';
            document.getElementById('fill_rate').textContent = m.fill_rate || '0%';
            
            // AS
            const asProb = parseFloat(m.as_prob?.replace('%','')) || 0;
            document.getElementById('as_prob').textContent = m.as_prob || '0%';
            document.getElementById('as_bar').style.width = asProb + '%';
            document.getElementById('as_adj').textContent = m.as_spread_adj || '+0 bps';
            document.getElementById('as_trades').textContent = m.as_trades || '0';
            
            // Funding
            document.getElementById('funding_pred').textContent = m.funding_pred || '0.00%';
            document.getElementById('inv_bias').textContent = m.inv_bias || '0.00';
            
            // Models
            document.getElementById('model_regime').innerHTML = m.regime_fitted === '✓' ? 
                '<span class="status-ok">✓ Fitted</span>' : '<span class="status-error">✗ Not Fitted</span>';
            document.getElementById('model_as').innerHTML = m.as_fitted === '✓' ? 
                '<span class="status-ok">✓ Fitted</span>' : '<span class="status-error">✗ Not Fitted</span>';
            document.getElementById('model_fill').innerHTML = m.fill_fitted === '✓' ? 
                '<span class="status-ok">✓ Fitted</span>' : '<span class="status-error">✗ Not Fitted</span>';
            document.getElementById('drift').innerHTML = m.drift === '⚠️' ? 
                '<span class="status-error">⚠️ Detected</span>' : '<span class="status-ok">✓ OK</span>';
            
            // System
            document.getElementById('uptime').textContent = m.uptime || '0m';
            document.getElementById('cycles').textContent = m.cycles || '0';
            document.getElementById('last_update').textContent = 'Last Update: ' + (m.last_update || '-');
            
            // Alerts
            const alerts = document.getElementById('alerts');
            alerts.innerHTML = '';
            if (m.liq_alert?.includes('YES')) {
                alerts.innerHTML += '<div class="alert">⚠️ LIQUIDATION CASCADE DETECTED</div>';
            }
            if (asProb > 70) {
                alerts.innerHTML += '<div class="alert">⚠️ HIGH ADVERSE SELECTION RISK (' + m.as_prob + ')</div>';
            }
        }
        
        // Initial fetch and auto-refresh every 2 seconds
        fetchMetrics();
        setInterval(fetchMetrics, 2000);
    </script>
</body>
</html>
"""

# Global metrics storage
current_metrics = {}
metrics_lock = threading.Lock()


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for dashboard requests."""
    
    def log_message(self, format, *args):
        pass  # Suppress default logging
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        
        elif self.path == '/api/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            with metrics_lock:
                self.wfile.write(json.dumps(current_metrics).encode())
        
        else:
            self.send_response(404)
            self.end_headers()


def update_metrics(metrics_dict: dict):
    """Update global metrics for dashboard."""
    global current_metrics
    with metrics_lock:
        current_metrics = metrics_dict


def run_dashboard(port: int = 5555):
    """Run dashboard server."""
    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    print(f"📊 Dashboard running at http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    # Demo with mock data
    import random
    
    def generate_mock_metrics():
        pnl = random.uniform(-500, 800)
        inv = random.uniform(-3000, 3000)
        return {
            'total_pnl': f"${pnl:.2f}",
            'realized_pnl': f"${pnl*0.6:.2f}",
            'unrealized_pnl': f"${pnl*0.4:.2f}",
            'sharpe_ratio': f"{random.uniform(1.5, 3.5):.2f}",
            'trades': str(random.randint(50, 200)),
            'inventory': f"${inv:.0f}",
            'inventory_ratio': f"{inv/5000*100:.1f}%",
            'max_drawdown': f"{random.uniform(0.1, 1.5):.2f}%",
            'regime': random.choice(['low_vol', 'high_vol', 'trend_up', 'trend_down']),
            'regime_confidence': f"{random.uniform(60, 95):.1f}%",
            'liq_alert': random.choice(['✓ No', '✓ No', '✓ No', '⚠️ YES']),
            'spread': f"{random.uniform(5, 15):.1f} bps",
            'bandit_spread': f"{random.choice([5, 8, 10, 15])} bps",
            'bid_size': f"${random.randint(120, 200)}",
            'ask_size': f"${random.randint(150, 250)}",
            'fill_rate': f"{random.uniform(40, 80):.1f}%",
            'as_prob': f"{random.uniform(5, 40):.1f}%",
            'as_spread_adj': f"+{random.randint(0, 5)} bps",
            'as_trades': str(random.randint(0, 20)),
            'funding_pred': f"{random.uniform(-0.01, 0.03):.4f}%",
            'inv_bias': f"{random.uniform(-0.5, 0.5):+.2f}",
            'regime_fitted': '✓',
            'as_fitted': random.choice(['✓', '✗']),
            'fill_fitted': '✗',
            'drift': random.choice(['✓', '⚠️']),
            'uptime': f"{random.randint(10, 300)}m",
            'cycles': str(random.randint(100, 5000)),
            'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # Start mock data generator
    def mock_updater():
        while True:
            update_metrics(generate_mock_metrics())
            time.sleep(2)
    
    threading.Thread(target=mock_updater, daemon=True).start()
    
    # Run dashboard
    print("Starting ML Market Maker Dashboard...")
    print("Open http://localhost:5555 in your browser")
    run_dashboard(5555)
