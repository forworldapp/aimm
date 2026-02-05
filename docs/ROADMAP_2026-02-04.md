# AIMM Bot Improvement Roadmap

> **Last Updated:** 2026-02-04
> **Current Version:** v5.2.1 (feature/ml-v4 branch)
> **Status:** Paper Trading with Rank 1 Optimal Parameters

---

## Current State Summary

| Module | Status | Parameters |
|--------|--------|------------|
| v4.0 LightGBM | ✅ Enabled | volatility + direction models |
| v5.0 Order Flow | ✅ Enabled | OBI threshold=0.3 |
| v5.1 Funding Rate | ✅ Enabled | freeze_before=30min |
| v5.2 Microstructure | ✅ Enabled | **Rank 1: threshold=0.50** |
| v5.3 Cross-Asset | ❌ Disabled | PnL -$5,406 (rollback) |
| v5.4 Execution Algo | ✅ Enabled | TWAP/VWAP |
| v6.0 RL Agent | ❌ Disabled | Needs GPU training |

---

## 📌 Short-term (1-2 Weeks)

### 1. Paper Trading Monitoring 🔴
- **Priority:** Critical
- **Duration:** 7 days
- **Goal:** Validate Rank 1 parameters in real market
- **Metrics:** PnL, Sharpe, Max Drawdown, Kill Switch status
- **Action:** Monitor dashboard at http://localhost:8501

### 2. Create PR and Merge to Main 🔴
- **Priority:** Critical
- **Branch:** `feature/ml-v4` → `main`
- **Files Changed:** 51 files
- **Review:** Use `docs/PR_DESCRIPTION.md` template

### 3. Telegram Notifications 🟠
- **Priority:** High
- **Alerts:**
  - Circuit Breaker triggered
  - Kill Switch module disabled
  - Daily PnL summary
- **Implementation:** Create `core/notifier.py`

### 4. Cross-Asset Hedging Retuning 🟠
- **Priority:** High
- **Issue:** Used synthetic ETH data → -$5,406 loss
- **Fix:** Use real ETH/USDT market data
- **Backtest:** Re-run with actual correlation data

---

## 📌 Mid-term (1 Month)

### 5. Live Trading Transition 🟡
- **Prerequisites:**
  - [ ] 7 days Paper Trading stable
  - [ ] PnL positive or breakeven
  - [ ] No Kill Switch triggers
- **Initial Capital:** $500-1000
- **Checklist:** See `docs/PAPER_TRADING_GUIDE.md`

### 6. Multi-Symbol Expansion 🟡
- **Symbols:** ETH_USDT_Perp, SOL_USDT_Perp
- **Changes:**
  - Update `config.yaml` for multi-symbol
  - Separate risk limits per symbol
  - Parallel MarketMaker instances

### 7. RL Agent GPU Training 🟡
- **Current:** 50k timesteps (CPU)
- **Target:** 1M+ timesteps (GPU)
- **Expected:** Self-tuning parameters
- **Script:** `rl/train_extended.py`

### 8. Model Drift Detection 🟡
- **Module:** `ml/drift_detector.py` (exists, needs integration)
- **Alerts:** Performance degradation warning
- **Threshold:** Sharpe < 1.0, Win Rate < 45%

---

## 📌 Long-term (Quarterly)

### 9. On-chain Data Integration
- Whale wallet tracking
- Exchange inflow/outflow
- Liquidation heatmaps

### 10. News/Social Sentiment
- Twitter/X crypto sentiment
- News headline analysis
- Fear & Greed Index

### 11. Portfolio-level Risk Management
- Multi-bot coordination
- Cross-strategy correlation
- Unified risk dashboard

### 12. Monthly Model Retraining
- LightGBM volatility/direction
- HMM regime detector
- Automated pipeline

---

## 🚀 Immediate Actions

```
Week 1:
  ✅ Apply Rank 1 params (threshold=0.50)
  ✅ Push backup to GitHub
  🔄 Paper Trading monitoring (in progress)

Week 2:
  □ Create PR (feature/ml-v4 → main)
  □ Telegram notifications
  □ Analyze Paper Trading results
```

---

## References

- [ML_MODULES.md](file:///C:/Users/camel/.gemini/antigravity/scratch/aimm/docs/ML_MODULES.md) - Module documentation
- [TUNING_GUIDE.md](file:///C:/Users/camel/.gemini/antigravity/scratch/aimm/docs/TUNING_GUIDE.md) - Parameter tuning
- [PAPER_TRADING_GUIDE.md](file:///C:/Users/camel/.gemini/antigravity/scratch/aimm/docs/PAPER_TRADING_GUIDE.md) - Paper trading setup
- [CHANGELOG.md](file:///C:/Users/camel/.gemini/antigravity/scratch/aimm/CHANGELOG.md) - Version history
