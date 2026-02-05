# v4.0 ML Strategy Guide

## Overview
The v4.0 strategy integrates advanced Machine Learning models to optimize market making parameters in real-time. It replaces static rules with dynamic predictions based on market volatility and direction.

## Components

### 1. Volatility Prediction Model (LightGBM)
- **Target**: `range_15m` (High-Low range over next 15 minutes)
- **Performance**: R² = 0.50 (Explains 50% of volatility variance)
- **Logic (AGGRESSIVE Mode)**:
    - **High Volatility (Top 30%)**: 
        - **Spread**: 0.8x (Tighter) - Capture more volume when market moves
        - **Size**: 1.3x (Larger) - Capitalize on high probability of fills
    - **Low Volatility (Bottom 30%)**:
        - **Spread**: 1.3x (Wider) - Protect against adverse selection in thin markets
        - **Size**: 0.8x (Smaller) - Conserve capital
- **Rationale**: Backtesting showed that aggressive behavior in high volatility significantly outperforms conservative approaches (Sharpe Ratio +61%).

### 2. Direction Prediction Model (LightGBM)
- **Target**: Binary (UP vs DOWN) over next 15 minutes
- **Performance**: Accuracy ~53% (Slight edge over random)
- **Logic**:
    - **UP Signal**: Shift grid UP (Bid layers +1, Ask layers -1), Increase Bid Size (1.15x)
    - **DOWN Signal**: Shift grid DOWN (Bid layers -1, Ask layers +1), Increase Ask Size (1.15x)
- **Threshold**: Confidence > 52%

## Performance (Backtest)
Based on 1-year historical data (526,000 minutes):

| Metric | v3.8 Baseline | v4.0 ML Strategy | Improvement |
|--------|---------------|------------------|-------------|
| **Total PnL** | $8,679 | **$14,567** | **+68%** |
| **Sharpe Ratio** | 14.55 | **23.48** | **+61%** |
| **Max Drawdown** | 0.68% | **0.42%** | **-38%** (Risk Reduced) |

## Configuration
Controlled via `ml/strategy_v4.py` and `config.yaml`.

```yaml
strategy:
  ml:
    enabled: true
    mode: "aggressive"
    vol_high_percentile: 70
    vol_low_percentile: 30
```

## Maintenance
- Models are located in `data/` directory.
- Retrain models monthly using `ml/train_volatility.py` and `ml/train_improved.py`.
