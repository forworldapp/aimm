# Walkthrough - Setup aimm workspace

The `aimm` workspace has been set up for Paper Trading, with a complete virtual environment and dependencies. Several critical bugs in the Paper Trading implementation and Dashboard integration have been resolved.

## Setup Steps
1. **Clone & Configure**
   - Cloned `aimm` repository.
   - Configured remote to `forworldapp/aimm`.
   - Set mode to `paper` in `config.yaml`.

2. **Environment**
   - Created `venv` and installed requirements.
   - Removed bundled library folders to fix import conflicts.

## Bug Fixes & Improvements

### 1. Paper Exchange Order Matching
- **Issue**: Paper Exchange returns orders in a simpler format than the Live API, causing the `MarketMaker` to fail to recognize existing orders and constantly cancel/re-place them.
- **Fix**: Modified `MarketMaker.cycle` to support both Paper (flat dict) and Live (nested `legs`) order formats.

### 2. Strategy "Waiting" State
- **Issue**: Bot would not place orders initially because it was waiting for 20 candles of Bilinger Band data.
- **Fix**: Explicitly allowed grid order placement in `waiting` and `neutral` regimes, only applying signal restrictions after sufficient data is collected.

### 3. Position Limits
- **Issue**: Low default `max_position_usd` ($500) prevented buy orders.
- **Fix**: Increased limit to **$5000** in `config.yaml`.

### 4. Dashboard Flickering (File Access Conflict)
- **Issue**: The Dashboard would frequently flash "No active orders" or show empty data because of file access conflicts (Windows locking) when reading `paper_status.json` while the bot was writing to it.
- **Fix**:
    - **Bot (`paper_exchange.py`)**: Changed from atomic `os.replace` to **direct file overwrite** to avoid Windows locking issues. Added retry logic.
    - **Dashboard (`dashboard.py`)**: Implemented **separate caching** for the order list. It now maintains the last valid order list if the current read is empty, preventing the UI from clearing out transiently. Added stricter data validation (timestamp checks).

## Verification
- **Bot Status**: Running and successfully placing/maintaining orders.
- **Dashboard**: Accessible at `http://localhost:8503`. Prices update in real-time, and the "No active orders" flickering issue is resolved.

## Phase 7: Risk Management (Circuit Breaker Optimization)
**Goal**: Configure the bot to survive a "Black Swan" event (simulated using Oct 2025 crash data).

### 1. Circuit Breaker Threshold ($350 -> $200)
- **Analysis**: Ran sensitivity analysis on crash data (`backtest/run_breaker_scan.py`).
- **Finding**: $350 was too loose (too much drawdown before trigger). $150 was too tight (too many false positives).
- **Result**: **$200** identified as the "Golden Zone" – balancing capital protection with operational stability.

### 2. Emergency Liquidation Logic (v4.0.2)
- **Problem**: Default behavior (Cancel Orders & Stop) left the bot holding a losing position, which continued to bleed value ("drift").
- **Evidence**: Backtest comparison showed "Hold" strategy lost an additional ~$3 post-trigger, while "Liquidation" preserved capital perfectly.
- **Solution**: Updated `strategies/market_maker.py` to trigger **Immediate Market Close** when the $200 limit is breached.

### 3. Verification
- **Backtest**: Validated via `backtest/compare_liquidation.py`.
- **Live Test**: Bot successfully triggered liquidation on startup when detecting a stale losing position (-$252).
