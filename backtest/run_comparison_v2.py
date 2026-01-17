"""
GMM vs HMM Backtest Comparison Script V2
- Uses subprocess to isolate each model run
- Prevents config caching issues
"""

import subprocess
import sys
import json
import os

AIMM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_EXE = os.path.join(AIMM_DIR, "venv", "Scripts", "python.exe")


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
    
    df = pd.read_csv(r"{AIMM_DIR}/data/btcusdt_1m_1year.csv")
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
    
    steps = min(len(df), 43200)
    equity_history = []
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
    
    print(json.dumps({{"pnl": pnl, "trades": trades, "max_dd": max_dd * 100}}))

asyncio.run(run())
'''
    
    result = subprocess.run(
        [PYTHON_EXE, "-c", script],
        capture_output=True,
        text=True,
        cwd=AIMM_DIR
    )
    
    # Parse output
    try:
        lines = result.stdout.strip().split("\n")
        for line in reversed(lines):
            if line.startswith("{"):
                return json.loads(line)
    except:
        print(f"Error parsing {model_type} output:")
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
    
    return {"pnl": 0, "trades": 0, "max_dd": 0}


def main():
    print("=" * 50)
    print("GMM vs HMM BACKTEST COMPARISON (V2 - Subprocess)")
    print("=" * 50)
    
    print("\n[1/2] Running GMM backtest...")
    gmm = run_single_backtest("gmm")
    print(f"  GMM: PnL=${gmm['pnl']:.2f}, Trades={gmm['trades']}, MaxDD={gmm['max_dd']:.2f}%")
    
    print("\n[2/2] Running HMM backtest...")
    hmm = run_single_backtest("hmm")
    print(f"  HMM: PnL=${hmm['pnl']:.2f}, Trades={hmm['trades']}, MaxDD={hmm['max_dd']:.2f}%")
    
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS (1 Month)")
    print("=" * 50)
    print(f"{'Metric':<20} {'GMM':>12} {'HMM':>12}")
    print("-" * 50)
    print(f"{'PnL ($)':<20} {gmm['pnl']:>12.2f} {hmm['pnl']:>12.2f}")
    print(f"{'Total Trades':<20} {gmm['trades']:>12} {hmm['trades']:>12}")
    print(f"{'Max Drawdown (%)':<20} {gmm['max_dd']:>12.2f} {hmm['max_dd']:>12.2f}")
    print("=" * 50)
    
    if hmm['pnl'] > gmm['pnl']:
        print("🏆 WINNER: HMM")
    elif gmm['pnl'] > hmm['pnl']:
        print("🏆 WINNER: GMM")
    else:
        print("🤝 TIE")


if __name__ == "__main__":
    main()
