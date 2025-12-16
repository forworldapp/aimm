# Changelog

## [v1.2.0] - 2025-12-15
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
