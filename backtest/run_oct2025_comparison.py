"""
October 2025 High Volatility Backtest - GMM vs HMM Comparison
Uses subprocess isolation for accurate comparison
"""

import subprocess
import sys
import json
import os

AIMM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_EXE = os.path.join(AIMM_DIR, "venv", "Scripts", "python.exe")
DATA_FILE = "data/btcusdt_1m_oct2025.csv"


def run_single_backtest(model_type: str) -> dict:
    """Run backtest in subprocess with specific model type."""
    script = f'''
import sys
import os
sys.path.append(r"{AIMM_DIR}")

import pandas as pd
import asyncio
import json

from core.config import Config
from backtest.mock_exchange import MockExchange
from strategies.market_maker import MarketMaker

async def run():
    Config.set("strategy", "regime_model_type", "{model_type}")
    
    df = pd.read_csv(r"{AIMM_DIR}/{DATA_FILE}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["best_bid"] = df["close"] * (1 - 0.00005)
    df["best_ask"] = df["close"] * (1 + 0.00005)
    
    exchange = MockExchange(df, initial_balance=10000.0)
    strategy = MarketMaker(exchange)
    
    strategy.max_loss_usd = 100000.0
    strategy.max_drawdown_pct = 100.0
    strategy.is_active = True
    
    def mock_pred(symbol="BTCUSDT"):
        if strategy.candles.empty or len(strategy.candles) < 50:
            return {{}}
        return strategy.regime_detector.predict_proba(strategy.candles)
    
    if strategy.regime_detector:
        strategy.regime_detector.predict_live_proba = mock_pred
    
    steps = len(df)  # Full month
    equity_history = []
    regime_counts = {{}}
    exchange.current_index = 50
    
    for i in range(50):
        mid = (df.iloc[i]["best_bid"] + df.iloc[i]["best_ask"]) / 2
        strategy.price_history.append(mid)
    
    for i in range(50, steps - 1):
        if not exchange.next_tick():
            break
        idx = exchange.current_index
        window = max(0, idx - 60)
        strategy.candles = df.iloc[window:idx+1][["open","high","low","close","volume"]]
        row = df.iloc[idx]
        mid = (row["best_bid"] + row["best_ask"]) / 2
        strategy.price_history.append(mid)
        await strategy.cycle()
        
        # Track regime
        regime = getattr(strategy, 'current_ml_regime', 'unknown')
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        if not strategy.is_active:
            break
        pos = exchange.position
        eq = exchange.balance["USDT"] + (pos["amount"] * mid)
        equity_history.append(eq)
    
    final = equity_history[-1] if equity_history else 10000
    pnl = final - 10000
    trades = len(exchange.trade_history)
    
    peak = 10000.0
    max_dd = 0.0
    for eq in equity_history:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd: max_dd = dd
    
    print(json.dumps({{"pnl": pnl, "trades": trades, "max_dd": max_dd * 100, "regimes": regime_counts}}))

asyncio.run(run())
'''
    
    result = subprocess.run(
        [PYTHON_EXE, "-c", script],
        capture_output=True,
        text=True,
        cwd=AIMM_DIR
    )
    
    try:
        lines = result.stdout.strip().split("\n")
        for line in reversed(lines):
            if line.startswith("{"):
                return json.loads(line)
    except:
        print(f"Error parsing {model_type} output:")
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
    
    return {"pnl": 0, "trades": 0, "max_dd": 0, "regimes": {}}


def main():
    print("=" * 60)
    print("OCTOBER 2025 HIGH VOLATILITY BACKTEST (GMM vs HMM)")
    print("=" * 60)
    print(f"Data: {DATA_FILE}")
    print(f"Period: October 2025 (44,000 1-minute candles)")
    print()
    
    print("[1/2] Running GMM backtest...")
    gmm = run_single_backtest("gmm")
    print(f"  GMM: PnL=${gmm['pnl']:.2f}, Trades={gmm['trades']}")
    
    print("\n[2/2] Running HMM backtest...")
    hmm = run_single_backtest("hmm")
    print(f"  HMM: PnL=${hmm['pnl']:.2f}, Trades={hmm['trades']}")
    
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS - OCTOBER 2025")
    print("=" * 60)
    print(f"{'Metric':<20} {'GMM':>15} {'HMM':>15}")
    print("-" * 60)
    print(f"{'PnL ($)':<20} {gmm['pnl']:>15.2f} {hmm['pnl']:>15.2f}")
    print(f"{'Total Trades':<20} {gmm['trades']:>15} {hmm['trades']:>15}")
    print(f"{'Max Drawdown (%)':<20} {gmm['max_dd']:>15.2f} {hmm['max_dd']:>15.2f}")
    print("=" * 60)
    
    # Regime distribution
    print("\nRegime Distribution:")
    print(f"  GMM: {gmm.get('regimes', {})}")
    print(f"  HMM: {hmm.get('regimes', {})}")
    
    # Winner
    if hmm['pnl'] > gmm['pnl']:
        diff = hmm['pnl'] - gmm['pnl']
        print(f"\n🏆 WINNER: HMM (+${diff:.2f})")
    elif gmm['pnl'] > hmm['pnl']:
        diff = gmm['pnl'] - hmm['pnl']
        print(f"\n🏆 WINNER: GMM (+${diff:.2f})")
    else:
        print("\n🤝 TIE")


if __name__ == "__main__":
    main()
