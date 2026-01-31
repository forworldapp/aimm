# Changelog

All notable changes to the AIMM (AI Market Maker) project will be documented in this file.

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
