"""
1-Year Long-Term Backtest Comparison
====================================
Compares v3.8.0 (baseline) vs v3.10.0 (hybrid) across 1 full year of data.
Uses proven async pattern from run_multi_period_comparison.py
"""
import sys
import os
sys.path.insert(0, r'c:\Users\camel\.gemini\antigravity\scratch\aimm')

import asyncio
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import logging

from core.config import Config
from backtest.mock_exchange import MockExchange
from strategies.market_maker import MarketMaker

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("LongTermBT")
logger.setLevel(logging.INFO)
logging.getLogger("MarketMaker").setLevel(logging.CRITICAL)


def fetch_binance_klines(symbol: str, interval: str, start_time: int, end_time: int) -> list:
    """Fetch klines from Binance API with pagination."""
    all_data = []
    current_start = start_time
    
    while current_start < end_time:
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000
        }
        
        try:
            resp = requests.get(url, params=params, timeout=30)
            data = resp.json()
            
            if not data:
                break
                
            all_data.extend(data)
            current_start = data[-1][6] + 1
            
            if len(all_data) % 50000 == 0:
                print(f"  Fetched {len(all_data)} candles...")
            
        except Exception as e:
            print(f"Error: {e}")
            break
    
    return all_data


async def run_backtest(df: pd.DataFrame, disable_vol_adapt: bool = False) -> dict:
    """Run backtest with or without volatility adaptation."""
    Config.load("config.yaml")
    
    # CRITICAL: Override config at the source level for v3.8.0 mode
    if disable_vol_adapt:
        # Disable volatility adaptation entirely at config level
        if 'strategy' in Config._config and 'volatility_adaptation' in Config._config['strategy']:
            Config._config['strategy']['volatility_adaptation']['enabled'] = False
    else:
        # Ensure hybrid mode is enabled for v3.10.0 test
        if 'strategy' in Config._config and 'volatility_adaptation' in Config._config['strategy']:
            Config._config['strategy']['volatility_adaptation']['enabled'] = True
            Config._config['strategy']['volatility_adaptation']['mode'] = 'hybrid'
    
    # Prepare data
    df = df.copy()
    df['best_bid'] = df['close'] * 0.99995
    df['best_ask'] = df['close'] * 1.00005
    df = df.reset_index(drop=True)
    
    initial_balance = 10000.0
    exchange = MockExchange(df, initial_balance)
    strategy = MarketMaker(exchange)
    strategy.max_loss_usd = 50000.0  # High limit for long-term test
    strategy.order_size_usd = 100.0
    
    # Also set initial state for safety
    if disable_vol_adapt:
        strategy._vol_regime = 'high'
        strategy._vol_skip_prob = 0.0
    
    peak = initial_balance
    max_dd = 0
    
    for i in range(len(df)):
        try:
            await strategy.cycle()
        except Exception as e:
            if "CIRCUIT BREAKER" in str(e):
                logger.warning(f"Circuit breaker at step {i}")
                break
        
        if getattr(strategy, 'stopped', False):
            break
        
        if not exchange.next_tick():
            break
        
        # Track DD
        equity = exchange.balance['USDT'] + exchange.balance['BTC'] * df.iloc[min(i, len(df)-1)]['close']
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
        
        # Progress
        if (i + 1) % 100000 == 0:
            print(f"    Progress: {i+1}/{len(df)} ({(i+1)/len(df)*100:.0f}%), Equity: ${equity:.0f}")
    
    final_price = df.iloc[-1]['close']
    final_equity = exchange.balance['USDT'] + exchange.balance['BTC'] * final_price
    trades = len(exchange.trade_history)
    pnl = final_equity - initial_balance
    
    return {
        'pnl': pnl,
        'pnl_pct': pnl / initial_balance * 100,
        'trades': trades,
        'max_dd': max_dd * 100,
        'final_equity': final_equity
    }


async def main():
    print("=" * 70)
    print("           1-YEAR LONG-TERM BACKTEST COMPARISON")
    print("=" * 70)
    
    # Date range: 1 year of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"\nPeriod: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("\n📥 Fetching 1-year of 1-minute data from Binance...")
    
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    data = fetch_binance_klines("BTCUSDT", "1m", start_ts, end_ts)
    
    print(f"\nTotal candles fetched: {len(data):,}")
    
    if len(data) == 0:
        print("ERROR: No data fetched!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df = df.astype({
        'open': float, 'high': float, 'low': float, 
        'close': float, 'volume': float
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    print(f"Price range: ${df['close'].min():.0f} - ${df['close'].max():.0f}")
    
    # Run backtests
    print("\n" + "=" * 70)
    print("                    RUNNING BACKTESTS")
    print("=" * 70)
    
    # Test 1: v3.8.0 (volatility adaptation disabled)
    print("\n📊 Test 1: v3.8.0 Mode (No Volatility Adaptation)")
    print("-" * 50)
    result_v38 = await run_backtest(df, disable_vol_adapt=True)
    print(f"  Result: PnL=${result_v38['pnl']:.2f} ({result_v38['pnl_pct']:.2f}%), Trades={result_v38['trades']}, MaxDD={result_v38['max_dd']:.1f}%")
    
    # Test 2: v3.10.0 (hybrid mode enabled)
    print("\n📊 Test 2: v3.10.0 Hybrid Mode")
    print("-" * 50)
    result_v310 = await run_backtest(df, disable_vol_adapt=False)
    print(f"  Result: PnL=${result_v310['pnl']:.2f} ({result_v310['pnl_pct']:.2f}%), Trades={result_v310['trades']}, MaxDD={result_v310['max_dd']:.1f}%")
    
    # Print final comparison
    print("\n" + "=" * 70)
    print("                 1-YEAR BACKTEST RESULTS")
    print("=" * 70)
    print(f"\n{'Strategy':<25} {'PnL':>12} {'Return':>10} {'Trades':>10} {'MaxDD':>10}")
    print("-" * 70)
    print(f"{'v3.8.0 (Baseline)':<25} ${result_v38['pnl']:>10.2f} {result_v38['pnl_pct']:>9.2f}% {result_v38['trades']:>10} {result_v38['max_dd']:>9.1f}%")
    print(f"{'v3.10.0 (Hybrid)':<25} ${result_v310['pnl']:>10.2f} {result_v310['pnl_pct']:>9.2f}% {result_v310['trades']:>10} {result_v310['max_dd']:>9.1f}%")
    print("-" * 70)
    
    diff = result_v38['pnl'] - result_v310['pnl']
    winner = "v3.8.0" if diff > 0 else "v3.10.0"
    print(f"\n🏆 Winner: {winner} (by ${abs(diff):.2f})")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
