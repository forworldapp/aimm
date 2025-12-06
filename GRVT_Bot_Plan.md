# GRVT Trading Bot Development Plan & Strategy

## 1. Project Overview
**Objective**: Migrate from a Bitget/TradingView webhook-based bot to a high-performance, Python-native trading bot on **GRVT Exchange**.
**Primary Goal**: Capitalize on maker fee rebates and spread capture through Market Making (MM) or Grid Trading strategies.
**Tech Stack**:
- **Language**: Python (AsyncIO)
- **SDK**: `grvt-pysdk` (Official GRVT Python SDK)
- **Connection**: WebSocket (JSON RPC) for real-time data and order management.
- **Infrastructure**: Local Development -> Cloud Deployment (AWS Tokyo recommended for proximity to GRVT servers).

---

## 2. Strategy Concepts

### A. Pure Market Making (Primary Goal)
*Best for: High liquidity, stable or mean-reverting markets, earning maker rebates.*

**Core Logic**:
1.  **Quote Generation**: Continuously place Limit Buy (Bid) and Limit Sell (Ask) orders around the current Mid-Price.
    *   `Bid Price = Mid_Price - Spread - Skew`
    *   `Ask Price = Mid_Price + Spread - Skew`
2.  **Spread Management**:
    *   **Fixed Spread**: Constant % distance (e.g., 0.05%). Simple but risky in high volatility.
    *   **Dynamic Spread**: Widen spread when volatility (ATR or StdDev) increases to protect against toxic flow.
3.  **Inventory Risk Management (Skewing)**:
    *   Adjust quotes to maintain a neutral inventory (target 0 position).
    *   *Example*: If holding too much Long position, lower both Bid and Ask prices. This makes your Ask closer to market (easier to sell) and Bid further away (harder to buy).

**Pros**:
- Consistent income from Spread + Fee Rebates.
- High trade frequency.

**Cons**:
- **Adverse Selection (Toxic Flow)**: Buying just before a crash or selling just before a pump.
- **Latency Sensitivity**: Requires fast reaction times to cancel stale quotes.

### B. Grid Trading (Fallback/Hybrid)
*Best for: Ranging markets (Neutral) or Trend Following (Directional), lower latency requirements.*

**Core Logic**:
1.  **Grid Setup**: Define a price range (Min to Max) and number of grids.
2.  **Execution**:
    *   **Neutral**: Place buy orders below current price, sell orders above. As price moves, fill orders and place the opposite closing order.
    *   **Directional (Long/Short)**: Similar to Neutral but only takes trades in one direction or skews the grid density.
3.  **Difference from MM**: Grid orders are usually static until filled, whereas MM orders are constantly moved with the mid-price.

**Pros**:
- Less sensitive to latency (orders sit in the book).
- Proven profitability in sideways markets.

**Cons**:
- **Impermanent Loss**: If price leaves the grid range, you are left with a bag (if Long) or short exposure (if Short).

---

## 3. Technical Architecture

### Directory Structure
```
grvt_bot/
├── main.py                 # Entry point
├── config.py               # API Keys, Parameters (Env Vars)
├── core/
│   ├── grvt_client.py      # Wrapper around grvt-pysdk
│   ├── websocket_manager.py# Handles WS connection & Reconnection
│   └── order_manager.py    # Tracks open orders, fills, and PnL
├── strategies/
│   ├── base_strategy.py    # Abstract base class
│   ├── market_maker.py     # MM Logic
│   └── grid_strategy.py    # Grid Logic
└── utils/
    ├── calculations.py     # Indicators (ATR, Volatility)
    └── logger.py           # Logging setup
```

### Key Components
1.  **GRVT Client (`grvt-pysdk`)**:
    - Uses `CCXT`-compatible classes for familiarity.
    - Handles L2 signing (Private Key) automatically.
2.  **WebSocket Loop**:
    - Subscribe to `orderbook` (L2 data) for price feeds.
    - Subscribe to `orders` and `positions` for account updates.
3.  **AsyncIO Event Loop**:
    - Python's `asyncio` is crucial for handling WS messages without blocking strategy logic.

---

## 4. Development Roadmap

### Phase 1: Environment & Connectivity (Current Step)
- [ ] Install `grvt-pysdk`.
- [ ] Configure Environment Variables (`GRVT_API_KEY`, `GRVT_PRIVATE_KEY`, etc.).
- [ ] Create a simple script to:
    - Connect to Testnet.
    - Fetch Account Balance.
    - Place a dummy Limit Order.

### Phase 2: Core Infrastructure
- [ ] Implement `WebSocketManager` to handle data streams.
- [ ] Build `OrderManager` to track local state of orders (avoiding API rate limits).

### Phase 3: Strategy Implementation (MVP)
- [ ] Implement **Simple Market Maker**:
    - Place 1 Bid and 1 Ask at fixed % from mid-price.
    - Cancel & Replace when price moves > X%.
- [ ] Implement **Inventory Skew**:
    - Adjust prices based on held position.

### Phase 4: Testing & Optimization
- [ ] **Paper Trading**: Run on GRVT Testnet for 24-48 hours.
- [ ] **Latency Check**: Measure time from "Price Update" -> "Order Placed".
- [ ] **Risk Controls**: Add "Kill Switch" (Max Drawdown, Max Position Size).

### Phase 5: Live Deployment
- [ ] Deploy to AWS Tokyo (ap-northeast-1) for minimal latency.
- [ ] Start with minimum size.

---

## 5. Next Steps for User
1.  **Install SDK**: `pip install grvt-pysdk`
2.  **API Keys**: Generate API Key & Private Key from GRVT Web UI (Testnet first).
3.  **Hello World**: Run a test script to verify connection.
