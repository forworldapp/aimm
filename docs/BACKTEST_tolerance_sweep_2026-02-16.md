# Backtest: Tolerance & Grid Spacing Sweep
**Date:** 2026-02-16  
**Scripts:** `backtest/sweep_tolerance.py` (hourly), `backtest/sweep_tolerance_1m.py` (1-minute)

## Problem

During a steady uptrend ($68,452 → $69,700, +1.8%), sell orders were never filled.  
The bot re-placed orders at higher prices every ~2.5s cycle because `price_tolerance` was too tight (0.15%).

---

## Test 1: Hourly Backtest (1000h BTC)

**Data:** `data/btc_hourly_1000.csv` | **Combinations:** 36

| Tol% | PnL$ | Fills | Cancels |
|------|------|-------|---------|
| 0.10 | **+78.86** | 28 | 457 |
| 0.50 | +73.77 | 51 | 216 |
| 0.80 | +68.61 | **56** | **137** |

⚠️ **Misleading**: Hourly candles simulate fills using full high/low range, so even 0.1% tolerance gets fills. This does NOT reflect live 2.5s cycle behavior.

---

## Test 2: 1-Minute Backtest (10,000 candles, ~7 days) ← THE TRUTH

**Data:** `data/btcusdt_1m_1year.csv` (last 10K rows) | **Combinations:** 24

| Rank | Tol% | Spc% | PnL$ | PnL% | Sharpe | MaxDD% | Fills | Cancels | Fill/Cancel |
|------|------|------|------|------|--------|--------|-------|---------|-------------|
| 1 | **1.00** | **0.30** | **+28.50** | +0.28 | **12.68** | -0.08 | 35 | **99** | **0.35** |
| 2 | **0.80** | 0.10 | +23.92 | +0.24 | 9.79 | -0.08 | **43** | 157 | 0.27 |
| 3 | 0.80 | 0.15 | +23.92 | +0.24 | 9.79 | -0.08 | **43** | 157 | 0.27 |
| 4 | 0.80 | 0.20 | +23.92 | +0.24 | 9.79 | -0.08 | **43** | 157 | 0.27 |
| 5 | 1.00 | 0.10 | +22.91 | +0.23 | 7.58 | -0.12 | 36 | 104 | 0.35 |
| 8 | 0.80 | 0.30 | +15.99 | +0.16 | 3.83 | -0.15 | 40 | 142 | 0.28 |
| 9 | 0.50 | 0.10 | +9.17 | +0.09 | 3.71 | -0.11 | 18 | 271 | 0.07 |
| 12 | 0.30 | 0.30 | +8.95 | +0.09 | 7.43 | -0.04 | 4 | 541 | 0.01 |
| 20 | **0.10** | 0.10 | **+2.69** | +0.03 | 8.26 | -0.02 | **2** | **2833** | **0.0007** |
| 24 | 0.50 | 0.30 | -3.68 | -0.04 | -1.05 | -0.21 | 15 | 243 | 0.06 |

### Key Findings (1-Minute)

1. **0.1% tolerance = nearly useless**: Only 2 fills out of 2,833 cancel cycles (0.07% efficiency)
2. **0.8% tolerance = sweet spot for fills**: 43 fills (most), $23.92 PnL, 157 cancels
3. **1.0% + 0.30% spacing = best overall**: $28.50 PnL, 12.68 Sharpe, 35 fills, only 99 cancels
4. **Spacing matters less than tolerance**: Within same tolerance, spacing differences are small
5. **0.3% spacing can hurt at low tolerance**: 0.5%+0.3% produced the only negative PnL (-$3.68)

### Why 1-Minute ≠ Hourly

| Metric | 0.1% tol (hourly) | 0.1% tol (1-min) | Ratio |
|--------|-------------------|-------------------|-------|
| PnL | +$78.86 | +$2.69 | **29x worse** |
| Fills | 28 | 2 | **14x worse** |
| Cancels | 457 | 2,833 | **6x worse** |

Hourly candles check fills against 60 minutes of price range → almost any order placed near current price will fill.
1-minute candles check against 1 minute of range → orders must survive re-placement to fill.

---

## Final Parameter Decision

| Regime | Old Tolerance | New Tolerance | Old Spacing | New Spacing |
|--------|---------------|---------------|-------------|-------------|
| low_vol | 0.1% | **0.5%** | 0.10% | **0.15%** |
| trend_up | 0.15% | **0.8%** | 0.15% | **0.20%** |
| trend_down | 0.15% | **0.8%** | 0.15% | **0.20%** |
| high_vol | 0.2% | **1.0%** | 0.20% | **0.30%** |
| fallback | 0.1% | **0.8%** | — | — |

**Rationale:**
- **0.8% for trends** (current default scenario): Best fill rate (43) with strong PnL ($23.92)
- **1.0% for high_vol**: Extreme volatility benefits from maximum order stability
- **0.5% for low_vol**: Calm markets allow tighter tracking without missing fills

## Files Modified

- `ml/hmm_regime_detector.py` — REGIME_PARAMS updated
- `ml/regime_detector.py` — REGIME_PARAMS updated  
- `strategies/market_maker.py` — Default fallback tolerance → 0.8%

## Raw Data

- `data/sweep_tolerance_results.csv` (hourly, 36 rows)
- `data/sweep_tolerance_1m_results.csv` (1-minute, 24 rows)

## How to Re-run

```bash
# Hourly (~2 min)
python backtest/sweep_tolerance.py

# 1-minute (~12 min, more realistic)
python backtest/sweep_tolerance_1m.py
```
