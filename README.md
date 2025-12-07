# 🚀 GRVT Market Maker Bot

A high-performance, asynchronous Market Making bot designed for the **GRVT Exchange**.
It features a robust **Paper Trading Mode** that utilizes live mainnet data to simulate trading strategies without financial risk, coupled with a real-time **Streamlit Dashboard** for monitoring and control.

## 🌟 Key Features

*   **Real-time Paper Trading**: Connects to GRVT Mainnet to fetch live Orderbook ticks and simulates order execution locally with realistic mechanics (spread crossing, touch probablistic fills).
*   **Market Maker Strategy**:
    *   **Inventory Skew**: Automatically adjusts quotes to balance inventory.
    *   **Trend Following**: Shifts spreads based on short-term price trends.
    *   **Volatility Adjustment**: Widens spreads during turbulent market conditions.
    *   **Post-Only Enforcement**: Ensures liquidity provision (Maker) and prevents Taker fees.
*   **Interactive Dashboard**:
    *   Real-time Equity Curve & Price History (5s resampling).
    *   Live Position & PnL Monitoring.
    *   Bot Control (Start/Stop/Close Position).
    *   Trade History Logs (CSV Download).

## 🛠️ Architecture

*   **Core**: Python, `asyncio` for non-blocking operations.
*   **Exchange Layer**:
    *   `GrvtExchange`: Computes signatures and connects to GRVT SDK.
    *   `PaperGrvtExchange`: Mocks execution engine while streaming real data.
*   **Dashboard**: `Streamlit` with `Plotly` for interactive charting.
*   **Data Interop**: JSON/CSV file-based IPC (Inter-Process Communication) between Bot and Dashboard.

## 🚀 Getting Started

### Prerequisites

*   Python 3.10+
*   GRVT API Credentials (for Real Data Access)

### Installation

1.  **Clone the Repository**
    ```bash
    git clone <repo-url>
    cd grvt_bot
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configuration**
    *   Create a `.env` file with your GRVT credentials:
        ```env
        GRVT_API_KEY=your_api_key
        GRVT_PRIVATE_KEY=your_private_key
        ```
    *   Adjust `config.yaml` for strategy parameters (Spread, Order Amount, Risk Limits).

### Usage

1.  **Start the Bot (Backend)**
    ```bash
    python main.py
    ```

2.  **Launch the Dashboard (Frontend)**
    ```bash
    streamlit run dashboard.py
    ```
    *   Open `http://localhost:8501` in your browser.

## 📊 Strategy Tuning

You can tune the bot behavior in `config.yaml`:

*   `spread_pct`: Base spread between Bid and Ask (e.g., `0.001` for 0.1%).
*   `refresh_interval`: How often to cancel and replace orders (e.g., `3` seconds).
*   `inventory_skew_factor`: How aggressively to shift quotes to neutralize position.

## 🔄 Recent Updates

*   **v1.2.0**: Added **Post-Only** logic to prevent Taker fills.
*   **v1.1.5**: Improved Dashboard with **5s Resampling** and High-Sensitivity Charts.
*   **v1.1.0**: Implemented **Paper Trading** engine with real-time GRVT data feed.
*   **v1.0.0**: Initial Release.

## ⚠️ Disclaimer

This software is for educational and testing purposes only. Use at your own risk.
Paper Trading does not guarantee future performance in live markets due to latency and slippage differences.
