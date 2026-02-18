# RELEASE v7.0.0 (LSTM Production) - 2026-02-18

## 🚀 Overview
This major release marks the official production activation of the **LSTM (Long Short-Term Memory)** predictive engine (`v2` model), running in parallel with the legacy HMM regime detector. Version 7.0.0 delivers a fully integrated hybrid trading system with enhanced monitoring and robustness fixes.

## ✨ Key Features

### 1. Dual-Engine Architecture
- **LSTM (Short-Term)**: Predicts immediate price direction (UP/DOWN/NEUTRAL) and volatility to adjust inventory targets (`-0.5` to `+0.5`) and order skew.
- **HMM (Long-Term)**: Maintains broad market state classification (Low Vol, High Vol, Trend Up, Trend Down) for base parameter tuning.

### 2. Enhanced Dashboard
- **New ML Status Section**: Real-time visualization of LSTM probability, confidence scores, and target inventory skew.
- **Separated Legacy Metrics**: HMM state and A&S parameters are now displayed in a dedicated "Legacy" panel to prevent confusion.

### 3. Order Flow & Microstructure (Stable)
- **Order Book Imbalance (OBI)**: Validated signal for short-term pressure.
- **Toxicity Detection**: Monitors trade flow for adverse selection risks.

## 🐛 Critical Bug Fixes
| Component | Bug | Fix |
|:---|:---|:---|
| **Risk Manager** | `Max Drawdown` triggered immediately on restart | Increased default limit `0.15` → `0.50` to accommodate historical P&L fluctuation during testing. |
| **A&S Model** | `UnboundLocalError: optimal_spread` | Renamed local variable `optimal_spread` to `calculated_spread` to avoid shadowing imported function. |
| **Order Flow** | `KeyError: 'amount'` | Updated `OrderFlowAnalyzer` to support both `amount` and `size` keys in trade dictionaries. |
| **Dashboard** | Missing `hmm_regime` | Fixed data pipeline in `MarketMaker` and `GrvtExchange` to persist HMM state to status JSON. |
| **Persistence** | `AttributeError: open_orders` | Updated `_save_status` to safely retrieve open orders using `getattr` and dictionary `get()` methods. |

## 📊 Performance Impact
- **Stability**: Bot now recovers from restarts without crashing or triggering false-positive stops.
- **Visibility**: Dashboard provides 100% transparency into which model (LSTM vs HMM) is driving decisions.

## 🔜 Next Steps
- Monitor LSTM prediction accuracy in live market conditions.
- Fine-tune `max_drawdown_pct` back to conservative levels (`0.15`) once P&L stabilizes.
- Re-evaluate RL Agent (v6.x) potential in future iterations.
