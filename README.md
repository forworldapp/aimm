# GRVT Trading Bot

## Overview
This project is a high-performance trading bot designed for the **GRVT Exchange**. It supports **Market Making** and **Grid Trading** strategies, built with a modular architecture to allow easy switching between exchanges or strategies.

## Features
- **Modular Design**: Exchange interactions are abstracted via `ExchangeInterface`.
- **AsyncIO Core**: Built on Python's `asyncio` for high-concurrency performance.
- **GRVT Integration**: Uses official `grvt-pysdk` for secure, authenticated trading.
- **Strategies**:
    - *Market Making* (Planned): Captures spread and rebates.
    - *Grid Trading* (Planned): Fallback for volatile markets.

## Setup

### 1. Prerequisites
- Python 3.9+
- Git

### 2. Installation
```bash
git clone https://github.com/forworldapp/grvtmmgrid.git
cd grvtmmgrid
pip install -r requirements.txt
```

### 3. Configuration
1. Copy `.env.example` to `.env`.
2. Fill in your GRVT API credentials:
   ```env
   GRVT_API_KEY=your_api_key
   GRVT_PRIVATE_KEY=your_private_key
   GRVT_ENV=testnet
   ```

### 4. Running the Bot
```bash
python main.py
```

### 5. Data Collection
To collect market data for backtesting:
```bash
python scripts/data_collector.py
```
Data will be saved in `data/ticker_data_<timestamp>.csv`.

## Configuration
The bot is configured via `config.yaml`. You can adjust strategy parameters and risk limits without changing code.

```yaml
# Example config.yaml
strategy:
  name: market_maker
  spread_pct: 0.001       # 0.1% spread
  order_amount: 0.001     # Order size
  refresh_interval: 5     # Loop time (seconds)

risk:
  max_position_usd: 1000.0  # Max exposure
  inventory_skew_factor: 0.5 # Inventory balancing strength
```

## Architecture
- `core/`: Core infrastructure (Exchange adapters, Config, Risk Manager).
- `strategies/`: Trading logic implementations (Market Maker with Skew).
- `utils/`: Helper functions.

## Security
- **Never commit `.env` files.**
- Private keys are handled locally and never logged.

## License
Private / Proprietary
