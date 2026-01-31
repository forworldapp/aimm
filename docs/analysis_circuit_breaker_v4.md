# 🛡️ Circuit Breaker Optimization Analysis (v4.0.1)

**Date**: 2026-01-30
**Subject**: Determining Optimal `max_loss_usd` Threshold for Market Maker Bot
**Context**: Post-incident analysis using "October 2025 Crash" simulation data.

## 1. Executive Summary
- **Decision**: Lower limit from **$350** to **$200**.
- **Reasoning**: Precision testing revealed that specific "loss steps" exist due to market gaps. The $200 threshold provides the same "True Positive" protection as lower settings ($150) but minimizes "False Positives" better than tight limits, while saving **~50% more capital** than the previous $350 setting.

## 2. Sensitivity Scan Results
Simulated against Oct 1, 2025 crash event (Step 166 trigger).

| Threshold | Actual Loss | Outcome |
| :--- | :--- | :--- |
| **$150** | ~$230 | Triggered by gap. No extra protection vs $200. |
| **$200** | **~$230** | **✅ OPTIMAL (Golden Zone)**. Identical protection to $150. |
| **$250** | ~$345 | Missed the first drop. Loss increased by +$115. |
| **$300** | ~$345 | Redundant with $250. |
| **$350** | **~$462** | **❌ Previous Setting**. Late response caused +$232 extra loss. |

## 3. False Positive Analysis (Long-term)
Simulated over full month (44,000 candles).

- **$200 Limit**: 
    - 17 Triggers Total
    - 12 **True Saves** (Prevented further crash)
    - 5 **False Alarms** (Price recovered)
- **$350 Limit**:
    - 11 Triggers Total
    - 8 True Saves
    - 3 False Alarms

**Trade-off**: Shifting to $200 causes ~2 extra false stops per month but prevents ~4 major crashes that would otherwise breach the higher limit. The capital preservation utility outweighs the operational cost of restarting the bot.

## 4. Configuration Change
File: `config.yaml`
```yaml
risk:
  max_loss_usd: 200.0  # limit updated from 350.0
```

## 5. Next Steps for Future Agents
- Monitor live false positive rate. If > 2 per week in calm markets, consider moving to $250.
- Do **NOT** raise above $300 without re-running `backtest/run_breaker_scan.py` on new crash data.
