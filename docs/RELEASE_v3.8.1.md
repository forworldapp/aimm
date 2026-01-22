# AIMM v3.8.1 Release Notes

**Release Date**: 2026-01-21

## 📋 Summary

Minor fixes and improvements for v3.8.0 baseline strategy after 12-month backtest validation.

---

## 🔧 Changes

### Paper Trading Stability
- **Fixed**: Paper exchange now uses Binance Futures directly for orderbook data
  - Removed dependency on `grvt-pysdk` for paper trading mode
  - Symbol conversion: `BTC_USDT_Perp` → `BTC/USDT:USDT`

### Regime Detection
- **Fixed**: Trading now starts immediately in `neutral` mode
  - Previous: Bot waited for regime detection before placing orders
  - Now: Grid trading active while ML data accumulates

### Dashboard Updates
- **Updated**: ML status section now shows "HMM (v3.8.0)" instead of "GMM (v3.0)"
- **Fixed**: Grid order details display correctly

### Risk Management
- **Updated**: Circuit breaker threshold `max_loss_usd: $50 → $350`
  - Based on 12-month analysis showing max unrealized loss of $327

---

## 📊 Backtest Results (12-Month)

| Metric | Value |
|--------|-------|
| Total PnL | +$8,460 |
| Monthly Win Rate | 91.7% (11/12) |
| Sideways Performance | +$8,908 |
| Max Unrealized Loss | $327 |

---

## ⚙️ Configuration

```yaml
strategy:
  regime_model_type: hmm
  volatility_adaptation:
    enabled: false  # v3.8.0 baseline

risk:
  max_loss_usd: 350.0  # Circuit breaker
  max_drawdown_pct: 0.05
  max_position_usd: 5000.0
```

---

## 📁 Files Modified

- `core/paper_exchange.py` - Binance direct connection
- `strategies/market_maker.py` - Neutral mode trading
- `dashboard.py` - HMM v3.8.0 display
- `config.yaml` - Circuit breaker threshold
- `docs/PAPER_TO_LIVE_GUIDE.md` - Deployment guide
