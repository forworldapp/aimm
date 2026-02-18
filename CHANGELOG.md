# Changelog

All notable changes to the AIMM (AI Market Maker) project will be documented in this file.

## [v7.0.0] - 2026-02-18
### 🚀 LSTM Production Release
- **Full LSTM Integration**: Activated `MLPredictor` (v2) for live trading.
- **Hybrid Regime**: 
    - **LSTM**: Short-term directional skew (Trend/Range).
    - **HMM**: Long-term market state classification (Low/High Vol, Trend).
- **Dashboard Upgrade**: 
    - Added **ML Status (LSTM v2)** section (Probability, Target Inventory).
    - Separated **Legacy HMM** metrics for clear dual-model monitoring.

### 🐛 Bug Fixes & Stabilization
- **A&S Model**: Fixed `optimal_spread` variable shadowing causing calculation errors.
- **Risk Management**: Increased `max_drawdown_pct` to 50% to prevent premature stops.
- **Order Flow**: Fixed `KeyError: 'amount'` in `OrderFlowAnalyzer` (added `size` fallback).
- **Status Sync**: Fixed missing `hmm_regime` in dashboard JSON output.
- **Persistence**: Fixed `AttributeError` in `_save_status` for `GrvtExchange`.

## [v6.0.0] - 2026-02-03
### 🤖 RL Agent Implementation
- **Gymnasium Environment**: Custom `MarketMakingEnv` with 8-dim observation, 3-dim action
- **PPO Training**: CPU-friendly training with stable-baselines3
- **RLAgentWrapper**: Integration wrapper for MarketMaker
- **Status**: Model trained (50k steps), NOT ENABLED by default (needs GPU training for production)

### 📚 Documentation
- Added `docs/ML_MODULES.md` - Comprehensive guide for all ML modules
- Updated CHANGELOG with all v5.x features

## [v5.4.0] - 2026-02-03
### ⚡ Execution Algorithms
- **TWAP Executor**: Time-Weighted Average Price order slicing
- **VWAP Executor**: Volume-weighted execution
- **Adaptive Executor**: Dynamic strategy selection based on volatility
- **Backtest Result**: Saves $93.67/year in slippage (90% reduction) ✅

## [v5.3.0] - 2026-02-03
### 🔗 Cross-Asset Hedging (DISABLED)
- **CorrelationAnalyzer**: BTC/ETH rolling correlation
- **CrossAssetHedger**: Beta-based directional hedge
- **Backtest Result**: -$5,406 loss → ROLLED BACK ❌

## [v5.2.0] - 2026-02-03
### 🔬 Microstructure Signals (DISABLED)
- **VPIN**: Volume-Synchronized Probability of Informed Trading
- **TradeArrivalAnalyzer**: Trade frequency analysis
- **VolumeClock**: Volume-based time measurement
- **Backtest Result**: Adverse -3%, PnL -$826 → ROLLED BACK ❌

## [v5.1.0] - 2026-02-03
### 💰 Funding Rate Arbitrage ⭐
- **FundingRateMonitor**: Tracks funding rates, calculates annualized APR
- **FundingIntegratedMM**: Adjusts bid/ask sizing based on funding direction
- **Freeze Logic**: Holds orders before funding settlement
- **Backtest Result**: +$890/year improvement ✅

## [v5.0.0] - 2026-02-02
### 📊 Order Flow Analysis
- **OrderFlowAnalyzer**: Bid/ask imbalance detection
- **Spread/Size Adjustment**: Widens spread in adverse conditions
- **Backtest Result**: Adverse selection -0.3% (risk trade-off) ✅

## [v4.0.2] - 2026-02-01
### Added
- **Emergency Liquidation**: Circuit breaker now triggers an immediate **Market Close** of all positions when `max_loss_usd` ($200) is breached. Matches empirical "Toxic Inventory" analysis findings.

## [v4.0.1] - 2026-01-30
### 🛡️ Circuit Breaker Optimization (Risk Management)
- **Parameters**: Lowered `max_loss_usd` from $350 to **$200** ("Golden Zone")
- **Analysis**: Conducted sensitivity scan on Oct 2025 crash data.
    - $200 limit prevents ~50% more losses compared to $350.
    - False positive rate increases slightly but prevents major blowups.
    - Documentation added: `docs/analysis_circuit_breaker_v4.md`

## [v4.0.0] - 2026-01-27
### MVP Release - ML Enhanced Strategy
- **New Feature**: Integrated LightGBM Volatility & Direction Models
  - Volatility Model: Predicts 15m range to adjust spread/size (Aggressive mode)
  - Direction Model: Predicts 15m trend to skew grid layers
- **Performance**: Validated improvement over baseline (Sharpe +61%, PnL +68%)
- **Dashboard**: Added "v4.0 ML Insights" section with real-time predictions
- **Strategy**: Added `ml/strategy_v4.py` and updated `market_maker.py` integration

---

## [v3.8.1-hmm-baseline] - 2026-01-27

### 🎯 Baseline Tag
This tag marks the stable HMM-based strategy before LightGBM integration.

### Features
- **HMM Regime Detection**: 4-state Hidden Markov Model (low_vol, high_vol, trend_up, trend_down)
- **Avellaneda-Stoikov Grid MM**: Dynamic γ/κ blending based on ML regime probabilities
- **Paper Trading**: Binance data feed with simulated execution
- **Dashboard**: Real-time Streamlit dashboard with 5-second auto-refresh

### Performance (12-month Backtest)
- **Total PnL**: +$718.65 (HMM) vs -$725.80 (Baseline)
- **HMM Advantage**: +$1,444.45 improvement
- **Best Months**: April (+$412), October (+$389)

### Configuration
```yaml
strategy:
  order_size_usd: 200
  grid_layers: 7
  ml_regime_enabled: true
  regime_model_type: hmm
risk:
  max_position_usd: 5000
  max_loss_usd: 350
```

### Files
- `ml/hmm_regime_detector.py` - HMM implementation
- `ml/train_hmm.py` - Training script (1-year hourly data)
- `data/regime_model_hmm.pkl` - Trained model
- `backtest/hmm_comparison.py` - Comparison backtester

---

## [v3.8.0] - 2026-01-20

### Initial Release
- Grid Market Making with Bollinger Band regime detection
- RSI filter for overbought/oversold conditions
- Basic paper trading support
