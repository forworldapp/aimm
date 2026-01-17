# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a high-performance market making bot for GRVT Exchange (cryptocurrency derivatives). The bot implements the **Avellaneda-Stoikov optimal market making model** with ML-based regime detection, technical indicator filters (Bollinger Bands, RSI, ADX), and sophisticated inventory management.

The system operates in two modes:
- **Paper Trading**: Simulates execution using real GRVT mainnet orderbook data
- **Live Trading**: Executes real orders on GRVT Exchange

## Running the Bot

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure credentials in .env
GRVT_API_KEY=your_api_key
GRVT_PRIVATE_KEY=your_private_key
GRVT_TRADING_ACCOUNT_ID=your_account_id
GRVT_ENV=prod  # or testnet
```

### Start Bot and Dashboard
```bash
# Terminal 1: Start the trading bot
python main.py

# Terminal 2: Start the Streamlit dashboard
streamlit run dashboard.py
# or with custom port:
python -m streamlit run dashboard.py --server.port 8503
```

### Key Configuration
Edit `config.yaml` for strategy parameters:
- `exchange.mode`: `paper` or `live`
- `strategy.spread_pct`: Base spread (default 0.005 = 0.5%)
- `strategy.grid_layers`: Number of grid levels (default 7)
- `strategy.order_size_usd`: Order size per layer ($200 default)
- `risk.max_position_usd`: Maximum position size ($5000 default)
- `risk.max_loss_usd`: Circuit breaker trigger ($50 default)

## Architecture

### Core Components

**Exchange Layer** (`core/`)
- `exchange_interface.py`: Abstract interface defining exchange operations
- `grvt_exchange.py`: Live GRVT Exchange implementation using `grvt-pysdk`
- `paper_exchange.py`: Paper trading simulator (fetches real orderbook, simulates fills locally)
- `risk_manager.py`: Position limits, drawdown monitoring, inventory skew calculations
- `config.py`: YAML + environment variable configuration system

**Strategy Layer** (`strategies/`)
- `market_maker.py`: Main market making strategy (V1.4.1)
- `base_strategy.py`: Abstract strategy interface
- `filters.py`: Technical indicator filters (RSI, Bollinger, ADX, ATR, Choppiness)

**Data Flow**
1. `main.py` loads config and initializes exchange (paper or live based on `config.yaml`)
2. Exchange connects and starts async monitoring loops
3. `MarketMaker` strategy runs the main cycle every `refresh_interval` seconds
4. Status is saved to `data/paper_status_{symbol}.json` for dashboard consumption
5. Dashboard reads JSON files and displays real-time charts

### Critical State Files
All located in `data/` directory:
- `paper_status_{symbol}.json`: Current bot state (position, balance, orders, regime)
- `pnl_history_{symbol}.csv`: Equity curve time series
- `trade_history_{symbol}.csv`: Filled trades log with FIFO grid profit calculation
- `command.json`: Dashboard control commands (start/stop/reload)

## Strategy Logic

### Avellaneda-Stoikov Model
The bot uses the academic optimal market making framework:

**Optimal Spread Formula**: `δ = γ×σ²×(T-t) + (2/γ)×ln(1+γ/κ)`
- γ (gamma): Risk aversion parameter (higher = wider spreads, default 1.0)
- κ (kappa): Market liquidity estimate (higher = tighter spreads, default 1000)
- σ (sigma): Realized volatility from log returns (20-tick rolling window)

**Reservation Price**: `r = s - q×γ×σ²×(T-t)`
- Shifts reference price based on inventory position
- Long position (q > 0) → lowers prices to encourage selling
- Short position (q < 0) → raises prices to encourage buying
- Quotes are placed at `r ± δ/2`

See `market_maker.py:318-498` for implementation.

### Regime Detection & Signal Latching
The strategy detects market regimes using selected filters (configurable via `strategy.trend_strategy` in config):
- **Bollinger Bands**: Mean reversion at band edges (default)
- **RSI**: Overbought/oversold conditions (auxiliary filter)
- **ADX/ATR/Choppiness**: Alternative trend/volatility indicators

**Signal Latch Mechanism** (`market_maker.py:598-630`):
- When a `buy_signal` or `sell_signal` is detected, it's "latched" (remembered)
- Latch persists even if signal indicator returns to neutral
- **Release Conditions**:
  - Buy latch: RSI > 40 (confirmation of reversal)
  - Sell latch: RSI < 60
  - Position flat (qty = 0)
- **Purpose**: Prevents premature loss-cutting when signals briefly fade but risky conditions persist

### Entry Anchor Mode
When `entry_anchor_mode: true` in config (`market_maker.py:682-703`):
- **Neutral Regime**: 0% loss tolerance (profit-only exits)
- **Signal Regime**: 0.5% loss tolerance (allows cutting bad positions)
- Prevents "buying high" (limits buy orders below entry price when long)
- Enables DCA (dollar-cost averaging) by allowing buys below entry

### Smart Order Management
Orders are only updated when prices deviate beyond `_ml_price_tolerance` (default 0.1%):
```python
# market_maker.py:805-847
if prices_match(existing_orders, new_orders, PRICE_TOLERANCE):
    # Skip update - no action needed
else:
    # Cancel and replace all orders
```
This reduces API calls and avoids unnecessary order churn.

### Grid Layers & Signal Boost
- Base grid: `grid_layers` orders on each side, spaced by `ml_grid_spacing` (ML-adjusted or 0.12% default)
- **Signal Boost** (`market_maker.py:759-772`): When buy/sell signal detected, adds 2 extra aggressive orders very close to mid price (0.05%, 0.1% away) to maximize fill probability

## ML Integration (Optional)

### Regime Detector
If `strategy.ml_regime_enabled: true` and `ml/regime_detector.py` is available:
- Loads GMM (Gaussian Mixture Model) from `data/regime_model.pkl`
- Predicts market regime probabilities (trending/ranging/volatile)
- Blends γ/κ parameters based on regime probabilities
- See `market_maker.py:342-396`

### Adaptive Parameter Tuner
If `strategy.adaptive_tuning_enabled: true` and `ml/adaptive_tuner.py` is available:
- Online learning from recent PnL performance
- Fine-tunes γ/κ dynamically based on win rate
- See `market_maker.py:399-406`

## Paper Trading Execution Model

`PaperGrvtExchange` (`core/paper_exchange.py`) simulates realistic fills:

**Fill Logic** (`_check_paper_fills`, line 139-188):
- **Cross**: Immediate fill if order price crosses spread (buy ≥ ask, sell ≤ bid)
- **Touch**: 10% fill probability if order touches best bid/ask
- Prevents overly optimistic fill rates

**PnL Calculation** (`_execute_paper_trade`, line 206-379):
- Distinguishes "increase" (opening/adding) vs "reduce" (closing) trades
- FIFO (First-In-First-Out) queue for grid profit calculation
- Maker rebate: 0.01% (1 bps) per filled order
- Saves each trade to CSV with realized PnL and grid profit columns

**Persistence**: On restart, loads last state from `paper_status_{symbol}.json` to preserve balance and position.

## Working with the Code

### Adding a New Technical Filter
1. Create new filter class in `strategies/filters.py` inheriting base structure
2. Implement `analyze(df)` method returning regime string
3. Register in `MarketMaker._initialize_filter()` switch statement
4. Set `strategy.trend_strategy: your_filter_name` in config

### Modifying Risk Parameters
All risk limits are in `config.yaml` under `risk:` section:
- `max_position_usd`: Hard position limit (blocks orders when exceeded)
- `max_loss_usd`: Circuit breaker (stops bot on unrealized loss threshold)
- `max_drawdown_pct`: Closes all positions if equity drops X% from peak
- `inventory_skew_factor`: Aggressiveness of inventory-based price skewing

### Dashboard Control Commands
Dashboard writes to `data/command.json`, bot reads in `check_command()`:
- `start`: Resume trading
- `stop`: Pause (cancel all orders)
- `stop_close`: Pause + market close position
- `reload_config`: Hot reload `config.yaml` without restart
- `restart`: Full bot restart (reloads exchange connection)
- `shutdown`: Graceful exit

### Debugging Fill Issues
- Check `logs/bot.log` for "PAPER TRADE" entries
- Verify orderbook connectivity: look for "Error fetching orderbook" warnings
- Examine `data/trade_history_{symbol}.csv` for actual fill prices vs order prices
- Monitor `mid_price` field in `paper_status_{symbol}.json` to ensure live data feed

### Testing Strategy Changes
1. Set `exchange.mode: paper` in config
2. Optionally reduce `risk.max_position_usd` for faster testing
3. Watch `data/pnl_history_{symbol}.csv` for equity curve
4. Use `strategy.refresh_interval: 1` for faster cycles
5. Monitor dashboard at http://localhost:8501

## Important Constraints

### GRVT Exchange Specifics
- **Minimum lot size**: 0.001 BTC for BTC_USDT_Perp
- **Tick size**: 0.1 USD (prices must be rounded to nearest $0.10)
- Orders must be POST_ONLY for maker rebates (strategy enforces this)
- SDK uses L2 signing (private key required for all operations)

### Singleton Pattern
`main.py` uses socket lock on port 45433 to prevent multiple bot instances. Only one instance can run at a time.

### File-based IPC
Dashboard and bot communicate via JSON files (not HTTP/WebSocket). This is intentional for simplicity but means:
- Dashboard may show stale data if bot crashes
- No direct "request-response" communication
- All state must be serialized to JSON

### Windows Compatibility
Direct file writes used instead of atomic `os.replace()` to avoid PermissionError on Windows. See `paper_exchange.py:419-423`.

## Common Modifications

**Change trading symbol**: Edit `exchange.symbol` in `config.yaml` (e.g., `ETH_USDT_Perp`)

**Disable ML features**: Set `ml_regime_enabled: false` and `adaptive_tuning_enabled: false`

**Switch to live trading**:
1. Set `exchange.mode: live` in config
2. Ensure `.env` has valid GRVT credentials
3. Start with small `order_size_usd` for safety

**Adjust aggressiveness**:
- Tighter spreads: Lower `gamma` (more risk) or increase `kappa` (assume more liquidity)
- Wider spreads: Increase `gamma` or lower `kappa`
- More layers: Increase `grid_layers`
- Bigger orders: Increase `order_size_usd`
