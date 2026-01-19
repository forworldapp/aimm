# Changelog

## [v3.8.0] - 2026-01-18
### Fixed
- **MockExchange Overflow Bug (CRITICAL)**: Fixed numeric overflow causing $7.7 quadrillion loss display
- **Circuit Breaker Infinite Loop**: Fixed strategy.stopped flag not being honored
- **Position Limits**: Added 1 BTC max position to prevent overflow
- **Order Limits**: Added 100 max open orders safety limit

### Added
- **`get_equity()` Method**: MockExchange now properly calculates USDT + BTC value
- **`get_current_price()` Method**: Convenience method for current mid price
- **Multi-Period Comparison Script** (`backtest/run_multi_period_comparison.py`):
  - Luna Collapse (May 2022) - High volatility test
  - October 2025 - Medium volatility test  
  - Recent Month - Low volatility test
- **Overflow Clamping**: All equity calculations clamped to reasonable ranges

### Backtest Results (3 Periods)
| Period | PnL | Return | Trades | MaxDD | Volatility |
|--------|-----|--------|--------|-------|------------|
| Luna Collapse | +$170 | +1.7% | 167 | 0.8% | 0.118% |
| October 2025 | +$173 | +1.7% | 309 | 1.5% | 0.068% |
| Recent Month | -$37 | -0.4% | 312 | 2.1% | 0.051% |
| **TOTAL** | **+$306** | **+3.1%** | 788 | 1.5% avg | - |

### ML Components Complete (Phase 1-5)
- Phase 1: Dynamic Order Sizing, Adverse Selection Detection
- Phase 2: GMM/HMM Regime Detection (8 features)
- Phase 3: Fill Probability, Funding Prediction, Liquidation Detection
- Phase 4: Contextual Bandit Spread, Online Learning
- Phase 5: IntegratedMarketMaker, Monitoring Dashboard, Alert System

## [v1.3.0] - 2025-12-16
### Added
- **Advanced Trend Filters (`strategies/filters.py`)**:
    - Implemented `ADXFilter` (Trend Strength), `ATRFilter` (Volatility), `ChopFilter` (Market Efficiency).
    - Added `ComboFilter` (ADX + ATR) for high-probability trend detection.
- **Selectable Strategy Mode**: Dashboard now allows choosing between `off`, `ma_trend`, `adaptive`, `adx`, `atr`, `chop`, `combo`.
- **Candle Processing**: `MarketMaker` now builds real-time 1m OHLC candles for technical indicators.
- **Testing Optimization**: Temporarily reduced `Combo` requirement to 7 candles (approx. 15m wait) for faster validation. (To be restored to 14 in v1.4).
## [v1.4.2] - 2025-12-25
### Added
- **Signal Latch Logic**: Retains 'Signal' mode (0.5% stop loss) even if indicators revert to neutral, until RSI enters safe zone (40-60) or position clears.
- **Regime-Based Risk Management**: Strict Profit-Only (0% tolerance) in Neutral vs. Stop-Loss Allowed (0.5%) in Signal mode.
- **Dashboard Enhancements**: detailed view of Open Orders (Price & Amount) in expandable section.

### Changed
- **Dynamic Spread**: Implemented USD-based clamping ($100 Min ~ $200 Max) per grid level (Dynamic Spread V2).
- **RSI Thresholds**: Adjusted Latch reset triggers to RSI 40 (Buy) and 60 (Sell).
- **Inventory Skew**: Skew factor confirmed at 0.1% for gentle inventory management.

## [v1.4.1] - 2025-12-20
### Fixed
- Retention logic comments and minor stability fixes.

### Added
- **Adaptive Market Making**: New strategy mode `'adaptive'` that automatically detects market regime.
  - **Ranging Mode**: When Moving Average divergence < 0.03%, Trend Skew is disabled to prevent whipsaws.
  - **Trending Mode**: When divergence expands, Trend Skew is enabled to follow the trend.
- **Strategy Selector**: Dashboard now supports selecting strategies (`Off`, `MA Trend`, `Adaptive`) in real-time.
- **Market Regime Indicator**: Dashboard now displays current market state (`RANGING` or `TRENDING`) and logic status (`WAITING`).
- **Entry Anchor Mode**: Logic to prevent pyramiding at unfavorable prices (buying higher than entry / selling lower than entry).

### Fixed
- **Critical Bug in Anchor Mode**: Dictionary key error (`averagePrice` -> `entryPrice`) fixed, ensuring correct entry price filtering.
- **GrvtExchange Robustness**: Implemented exponential backoff retry logic for `place_limit_order` to prevent crashes on network blips.
- **Status Display**: Fixed 'UNKNOWN' status by handling insufficient data history cases.

### Changed
- **Performance Tuning**: Reduced `refresh_interval` to 3 seconds for faster reaction.
- **Waiting Time**: Shortened MA window from 60 to 30 ticks to reduce boot time.
- **Spread**: Tightened default spread to 0.05% for better fill rates in ranging markets.

## [v1.1.0] - 2025-12-14
### Added
- Multi-layer Grid Strategy (5 layers).
- Smart Order Update (Cancel & Replace optimization).
- Dynamic Trend Skew (MA basd).

## [v1.0.0] - 2025-12-14
### Added
- Initial release with basic Market Making logic.
- Risk Manager integration.
- Dashboard UI setup.
