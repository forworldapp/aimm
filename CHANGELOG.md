# Changelog

## [v1.0.0] - 2025-12-14
### Stability & Architecture (Critical Fixes)
- **Zombie Process Prevention**: Implemented Singleton Pattern using Socket Lock to ensure only one bot instance runs.
- **Atomic File Writes**: Fixed `paper_status.json` read/write race conditions using atomic replace.
- **Dashboard Stability**: Implemented session persistence in Streamlit to eliminate UI flickering.
- **Process Control**: Added "Shutdown Process" button to Dashboard for clean termination.
- **Documentation**: Added `docs/STABILITY_IMPROVEMENTS.md`.

## [v0.2.0] - 2025-12-06
### Added
- **Centralized Configuration**: `config.yaml` for easy parameter tuning.
- **Risk Management Module**: `RiskManager` class to enforce position limits (`max_position_usd`).
- **Inventory Skew**: Market Maker strategy now adjusts bid/ask prices based on current inventory to maintain neutrality.
- **Position Tracking**: Added `get_position` method to Exchange Interface.

## [v0.1.0] - 2025-12-06
### Added
- Initial project structure.
- Basic `MarketMaker` strategy (Fixed Spread).
- GRVT Exchange integration via `grvt-pysdk`.
- Modular architecture (`ExchangeInterface`).
