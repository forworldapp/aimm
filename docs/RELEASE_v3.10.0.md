# AIMM v3.10.0 Release Notes

## Release Date: 2026-01-20

## Theme: Hybrid Volatility Strategy & Long-Term Backtest Validation

---

## 🆕 New Features

### 1. Hybrid Volatility Mode (v3.10.0)
- **Dynamic mode switching** based on real-time volatility
- **Low volatility mode**: Wider spreads (1.2x), 70% quote skip, min profit threshold
- **High volatility mode**: Normal v3.8.0 aggressive strategy

### 2. Stable Backtesting Tools
New memory-safe backtesting tools added:

| Tool | Description |
|------|-------------|
| `minute_backtest_chunked.py` | 12-month 1-min backtest (chunk processing) |
| `stable_backtest.py` | 90-day hourly backtest with regime analysis |
| `quarterly_backtest.py` | 1-year daily backtest with quarterly breakdown |
| `analyze_volatility.py` | 1-year volatility regime distribution analysis |

---

## 📊 Backtest Results (12-Month, 1-Minute Data)

### Overall Performance
| Metric | Value |
|--------|-------|
| **Total PnL** | **+$8,460** |
| **Profitable Months** | 11/12 (91.7%) |
| **Total Fills** | 206,947 |

### Performance by Market Regime
| Regime | PnL | Status |
|--------|-----|--------|
| **SIDEWAYS** | +$8,908 | ✅ Most Profitable |
| DOWN | +$80 | ✅ Profitable |
| UP | -$528 | ⚠️ Minor Loss |

### 1-Year Volatility Distribution
| Regime | Time % | Duration Avg |
|--------|--------|--------------|
| Low Volatility | 64.8% | 54 min |
| Medium Volatility | 22.1% | - |
| High Volatility | 13.1% | - |

---

## 📝 Configuration Changes

### New Hybrid Config (config.yaml)
```yaml
volatility_adaptation:
  enabled: true
  mode: hybrid
  thresholds:
    low: 0.0005    # σ < 0.05%
    medium: 0.0008  # 0.05% ≤ σ < 0.08%
  low_vol_mode:
    spread_multiplier: 1.2
    quote_skip_prob: 0.7
    min_profit_threshold: 0.0005
  high_vol_mode:
    spread_multiplier: 1.0
    quote_skip_prob: 0.0
    min_profit_threshold: 0.0
```

---

## 🔧 Code Changes

### strategies/market_maker.py
- Added hybrid mode logic in `_calculate_dynamic_spread()`
- Updated `cycle()` for dynamic quote skipping based on volatility regime
- New attributes: `_vol_regime`, `_vol_spread_mult`, `_vol_skip_prob`

---

## 📈 Key Findings

1. **v3.8.0 baseline is profitable** - +$8,460 over 12 months
2. **Sideways markets are most profitable** - Strategy captures spreads consistently
3. **Hybrid mode (v3.10.0) underperformed** in 3-period tests (+$64 vs +$306)
4. **Recommendation**: Use v3.8.0 baseline, hybrid mode available as option

---

## 🔄 Version History

| Version | Highlight |
|---------|-----------|
| v3.10.0 | Hybrid volatility strategy + backtest validation |
| v3.9.1 | Quote pause in low volatility |
| v3.9.0 | Volatility-adaptive spread |
| v3.8.0 | Baseline aggressive strategy |

---

## 📋 Files Changed

### Modified
- `config.yaml` - Added hybrid volatility config
- `strategies/market_maker.py` - Hybrid mode implementation

### Added
- `backtest/minute_backtest_chunked.py` - 12-month stable backtest
- `backtest/stable_backtest.py` - 90-day regime analysis
- `backtest/quarterly_backtest.py` - Quarterly breakdown
- `backtest/analyze_volatility.py` - Volatility distribution
- `backtest/run_1year_comparison.py` - Strategy comparison

---

**Recommendation**: Start with v3.8.0 baseline (volatility_adaptation.enabled: false)
