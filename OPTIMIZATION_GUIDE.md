# Parameter Optimization & Backtesting Guide

This guide explains how to run the automated parameter optimization system on a separate machine (e.g., a high-performance cloud instance like AWS EC2, Google Cloud, or a dedicated workstation).

## 1. Prerequisites
*   **Operating System**: Linux (Ubuntu 22.04+ recommended) or Windows
*   **Python**: 3.9 or higher
*   **Git**: Installed and configured
*   **Data File**: You MUST have the historical data file `data/btcusdt_1m_1year.csv` (approx 70MB).

> [!IMPORTANT]
> The data file is **not** included in the git repository (it is .gitignored). You must transfer it manually.

## 2. Setup (On the New Machine)

### Clone the Repository
```bash
git clone https://github.com/forworldapp/aimm.git
cd aimm
git checkout v0.2.0-regime-optimization-fix
```

### Install Dependencies
Create a virtual environment (recommended) and install packages:
```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate

# Install
pip install -r requirements.txt
```

### Transfer Data
Upload your `btcusdt_1m_1year.csv` file to the `data/` directory.
```bash
# Example using SCP from your local machine
scp data/btcusdt_1m_1year.csv user@remote-server:~/aimm/data/
```

## 3. Running Optimization (The "Set and Forget" Method)

We use a robust "chunked" optimizer that saves progress after every regime. This prevents losing hours of work if the script is interrupted.

### Command
Run the following command to optimize all 4 market regimes sequentially:

```bash
python backtest/run_optuna_chunked.py --all --trials 50 --candles 35000
```
*   `--all`: Runs Low Volatility, Uptrend, Downtrend, and High Volatility in order.
*   `--trials 50`: Higher is better (try 100 on a fast server).
*   `--candles 35000`: Use ~24 days of data for training (recommended).

### What Happens?
1.  **Low Volatility** runs first.
2.  Results are saved to `data/optuna_partial/low_vol.json`.
3.  **Uptrend** runs next, saving to `data/optuna_partial/trend_up.json`.
4.  ...and so on.
5.  Finally, it merges all results into `data/optimized_regime_params.json`.

## 3.1 High-Performance Mode (Parallel Execution)

Since your machine has a powerful CPU, you can run all 4 regimes **at the same time** to finish 4x faster.
Open **4 separate terminals** and run one command in each:

**Terminal 1 (Low Vol):**
```bash
python backtest/run_optuna_chunked.py low_vol --trials 100
```

**Terminal 2 (High Vol):**
```bash
python backtest/run_optuna_chunked.py high_vol --trials 100
```

**Terminal 3 (Uptrend):**
```bash
python backtest/run_optuna_chunked.py trend_up --trials 100
```

**Terminal 4 (Downtrend):**
```bash
python backtest/run_optuna_chunked.py trend_down --trials 100
```

> [!NOTE]
> **GPU vs CPU**: This process is **CPU-bound** (single-core logic per trial). It does **NOT** use GPU/CUDA.
> The "speed" comes from running multiple parallel processes on your multi-core CPU. No CUDA setup is required.

After all 4 verify they are done, run the merge command:
```bash
python backtest/run_optuna_chunked.py --merge
```

## 4. Handling Interruptions (Resume)

If the script stops (e.g., SSH disconnect, timeout), **simply run the same command again**.

```bash
python backtest/run_optuna_chunked.py --all --trials 50 --candles 35000
```

*   The system uses a database (`data/optuna_studies.db`) to track progress.
*   It will **skip** regimes that are already completed.
*   It will **resume** regimes that were partially done (e.g., if it did 15/50 trials, it will run the remaining 35).

## 5. Using the Results

Once optimization finishes:
1.  The file `data/optimized_regime_params.json` will be created/updated.
2.  **Download this file** back to your local machine (or commit it if you want).
3.  Place it in your local `data/` folder.
4.  Restart your bot. The `HMMRegimeDetector` will automatically load these new parameters.

## Troubleshooting

*   **"Target regime not found"**: If you see warnings about missing regimes in validation, this is normal for `low_vol` on recent data. The optimizer handles this by skipping validation metrics for that specific regime.
*   **Database Locked**: If you get SQLite errors, ensure no other optimization script is running. Delete `data/optuna_studies.db` ONLY if you want to start completely fresh.
