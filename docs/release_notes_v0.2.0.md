# Release Notes v0.2.0 - Regime Optimization Fix

**Date:** 2026-02-16
**Tag:** `v0.2.0-regime-optimization-fix`

## Critical Fixes
*   **Regime Parameter Separation**: Fixed a bug where the optimizer was using the entire dataset for `low_vol`, `trend_up`, etc., causing all regimes to converge to identical parameters.
    *   *Fix*: `RegimeParamOptimizer` now correctly filters performance metrics (PnL, Sharpe) to count *only* the periods matching the target regime.
*   **Validation Robustness**: Fixed a crash when a specific regime (e.g., `low_vol`) was missing from the validation dataset.
    *   *Fix*: The optimizer now logs a warning and returns 'N/A' for validation metrics instead of crashing or returning failure codes.

## Improvements
*   **Chunked Optimization**: The `run_optuna_chunked.py` script now supports a robust "Resume" workflow. It checks defining database state and effectively manages partial results.
*   **Metrics**: Added `status` flag to backtest results to distinguish between "crash" and "no data".

## How to Run
To run the optimization on a new machine:
1.  Ensure `data/btcusdt_1m_1year.csv` exists.
2.  Install dependencies: `pip install -r requirements.txt` (ensure `optuna`, `hmmlearn` are included).
3.  Run the chunked optimizer:
    ```bash
    # Run all regimes sequentially
    python backtest/run_optuna_chunked.py --all --trials 50 --candles 35000
    ```
4.  If interrupted, simply run the command again. It will resume for regimes that haven't completed their trial count.

## Known Issues
*   `low_vol` regime is remarkably rare in the recent validation set, often resulting in "N/A" validation metrics. This is expected behavior now, not a bug.
