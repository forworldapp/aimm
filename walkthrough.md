# Deployment Walkthrough - v0.2.1

## 1. Parameter Optimization
We successfully reverted to the 1-year backtest verified parameters and applied the new high-performance optimization results.

### Optimized Values
- **High Volatility**: Gamma 2.6, Kappa 199 (Defensive/Sniper Strategy)
- **Low Volatility**: Gamma 2.67, Kappa 1315
- **Trend Up**: Gamma 0.79
- **Trend Down**: Gamma 0.96

## 2. System Restart
- **Bot**: Restarted `main.py` successfully.
- **Dashboard**: Restarted `dashboard/monitoring_dashboard.py`.

## 3. Dashboard Access
The dashboard has been rolled back to the **Streamlit version** (Port 8501).

**URL:** [http://localhost:8501](http://localhost:8501)

> [!NOTE]
> The dashboard runs as a separate process from the bot.
> If it stops updating, ensure the `streamlit run dashboard.py` command is running.

**New Features:**
- Added **Risk & Adverse Selection** section (Sharpe, AS Prob, Inventory Bias) backported from V4 dashboard.
