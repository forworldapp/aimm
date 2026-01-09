# GRVT Market Maker Strategy v2.3.0

## 1. Core Concept: "Anchor & Defense"
This strategy aims to capture grid profits in a directional market while prioritizing capital preservation through strict inventory controls. Unlike traditional market makers that purely seek volume, this bot behaves like a **Mean Reversion Swing Trader**.

---

## 2. Market Regime Detection (Bollinger Filter)

| Regime | Condition | Behavior |
|--------|-----------|----------|
| **Neutral** | 20% ≤ BB% ≤ 80% | **Dual Grid** (Buy & Sell) - Accumulate spread profit |
| **Buy Signal** | BB% < 20% (Oversold) | **Buy Only** - Catch falling knife safely |
| **Sell Signal** | BB% > 80% (Overbought) | **Sell Only** - Fade the rally |

---

## 3. Order Logic

### Grid Structure
- **Layers**: 5 layers standard
- **Spacing**: Dynamic based on Volatility (ATR)
- **Order Size**: Configurable (Default $100 per order)

### Smart Order Management
- **Update Frequency**: Every 1-3 seconds
- **Tolerance**: Only moves orders if price shifts > 0.1% (Prevents spam)

---

## 4. Risk Management System (The "Defense")

### A. Entry Anchor (Anti-Pyramiding)
Prevents "Adding to Winners" (Pyramiding) which raises average price.
- **Long**: Can only buy **BELOW** entry price (Averaging Down allowed).
- **Short**: Can only sell **ABOVE** entry price.
- **Result**: Maintains a highly favorable Break-Even Price.

### B. Inventory Skew
Adjusts quotes to balance inventory.
- **Too much Long**: Lowers both Bid/Ask → Harder to buy, easier to sell.
- **Too much Short**: Raises both Bid/Ask → Harder to sell, easier to buy.

### C. Max Position Limit (Hard Cap)
- **Limit**: **$500 USD** (Configurable)
- **Action**: Blocks new orders in the direction of the limit.
- **Purpose**: Prevents "Falling Knife" accumulation in extreme trends.

### D. Circuit Breaker (Kill Switch)
- **Trigger**: Unrealized Loss > **$50 USD** (Configurable)
- **Action**: **Cancels ALL orders & STOPS BOT**.
- **Purpose**: Catastrophic loss prevention.

---

## 5. Profit Tracking (FIFO)
- **Exchange PnL**: Standard average price PnL.
- **Grid Profit**: Tracks individual grid pair profit (FIFO basis).
    - Buy @ 90,000 / Buy @ 89,000 / Sell @ 89,500
    - Realized PnL: Loss (Avg 89.5k vs Sell 89.5k)
    - Grid Profit: +$500 profit (Matched against 89k buy)

---

## 6. Configuration (config.yaml)

```yaml
marketing:
  mode: live
risk:
  max_position_usd: 500.0
  max_loss_usd: 50.0  # Circuit Breaker
strategy:
  entry_anchor_mode: true
  trend_strategy: bollinger
```
