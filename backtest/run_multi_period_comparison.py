"""
Multi-Period Backtest Comparison - Simplified Version
Uses same backtest engine as run_ml_backtest_1m.py
"""

import sys
import os
import pandas as pd
import asyncio
import logging
import requests
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from backtest.mock_exchange import MockExchange
from strategies.market_maker import MarketMaker

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("MultiPeriodBT")
logger.setLevel(logging.INFO)
logging.getLogger("MarketMaker").setLevel(logging.CRITICAL)


def fetch_binance_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch 1m candle data from Binance."""
    logger.info(f"Fetching {symbol} {start_date} to {end_date}...")
    
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    all_candles = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        params = {"symbol": symbol, "interval": "1m", "startTime": current_ts, "endTime": end_ts, "limit": 1000}
        try:
            resp = requests.get(base_url, params=params, timeout=30)
            data = resp.json()
            if not data:
                break
            all_candles.extend(data)
            current_ts = data[-1][0] + 60000
        except Exception as e:
            logger.error(f"Fetch error: {e}")
            break
    
    if not all_candles:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                             'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    logger.info(f"  -> Fetched {len(df)} candles")
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


async def run_backtest_period(df: pd.DataFrame, period_name: str, max_steps: int = 43200) -> dict:
    """Run backtest on given data."""
    if df.empty:
        return {"period": period_name, "error": "No data"}
    
    # Prepare data
    df = df.copy()
    df['best_bid'] = df['close'] * 0.99995
    df['best_ask'] = df['close'] * 1.00005
    df = df.reset_index(drop=True)
    
    # Limit steps
    if len(df) > max_steps:
        df = df.tail(max_steps).reset_index(drop=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"{period_name}")
    logger.info(f"Price Range: ${df['close'].min():.0f} - ${df['close'].max():.0f}")
    logger.info(f"Candles: {len(df)}")
    logger.info(f"{'='*60}")
    
    # Initialize
    exchange = MockExchange(df, initial_balance=10000.0)
    strategy = MarketMaker(exchange)
    strategy.max_loss_usd = 5000.0
    strategy.order_size_usd = 100.0
    
    # Track
    trades = 0
    peak = 10000
    max_dd = 0
    
    for i in range(len(df)):
        try:
            await strategy.cycle()
        except Exception as e:
            if "CIRCUIT BREAKER" in str(e):
                logger.warning(f"  Circuit breaker at step {i}")
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
        
        if (i + 1) % 15000 == 0:
            logger.info(f"  Step {i+1}/{len(df)}: ${equity:.0f}")
    
    # Final calc
    final_price = df.iloc[-1]['close']
    final_equity = exchange.balance['USDT'] + exchange.balance['BTC'] * final_price
    trades = len(exchange.trade_history)
    pnl = final_equity - 10000
    
    return {
        "period": period_name,
        "pnl": pnl,
        "pnl_pct": pnl / 100,
        "trades": trades,
        "max_dd": max_dd * 100,
        "price_range": f"${df['close'].min():.0f}-${df['close'].max():.0f}",
        "volatility": df['close'].pct_change().std() * 100
    }


async def main():
    results = []
    
    # 1. Luna Collapse (May 2022)
    luna_file = "data/btcusdt_1m_luna_may2022.csv"
    if os.path.exists(luna_file):
        luna_data = pd.read_csv(luna_file)
        luna_data['timestamp'] = pd.to_datetime(luna_data['timestamp'])
    else:
        luna_data = fetch_binance_data("BTCUSDT", "2022-05-01", "2022-05-31")
        if not luna_data.empty:
            luna_data.to_csv(luna_file, index=False)
    
    if not luna_data.empty:
        r = await run_backtest_period(luna_data, "1. Luna Collapse (May 2022)")
        results.append(r)
    
    # 2. October 2025
    oct_file = "data/btcusdt_1m_oct2025.csv"
    if os.path.exists(oct_file):
        oct_data = pd.read_csv(oct_file)
        oct_data['timestamp'] = pd.to_datetime(oct_data['timestamp'])
    else:
        oct_data = fetch_binance_data("BTCUSDT", "2025-10-01", "2025-10-31")
        if not oct_data.empty:
            oct_data.to_csv(oct_file, index=False)
    
    if not oct_data.empty:
        r = await run_backtest_period(oct_data, "2. October 2025")
        results.append(r)
    
    # 3. Recent Month
    recent_file = "data/btcusdt_1m_1year.csv"
    if os.path.exists(recent_file):
        full_data = pd.read_csv(recent_file)
        full_data['timestamp'] = pd.to_datetime(full_data['timestamp'], unit='ms')
        recent_data = full_data.tail(43200)
    else:
        end = datetime.now()
        start = end - timedelta(days=30)
        recent_data = fetch_binance_data("BTCUSDT", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    
    if not recent_data.empty:
        r = await run_backtest_period(recent_data, "3. Recent Month")
        results.append(r)
    
    # Print results
    print("\n" + "="*80)
    print("              MULTI-PERIOD BACKTEST COMPARISON RESULTS")
    print("="*80)
    print(f"{'Period':<35} {'PnL':>10} {'PnL%':>8} {'Trades':>8} {'MaxDD':>8} {'Vol':>8}")
    print("-"*80)
    
    for r in results:
        if "error" in r:
            print(f"{r['period']:<35} {'ERROR':>10}")
        else:
            print(f"{r['period']:<35} ${r['pnl']:>8.0f} {r['pnl_pct']:>7.1f}% {r['trades']:>8} {r['max_dd']:>7.1f}% {r['volatility']:>7.3f}%")
    
    print("="*80)
    
    # Summary
    total_pnl = sum(r.get('pnl', 0) for r in results if 'pnl' in r)
    avg_dd = sum(r.get('max_dd', 0) for r in results if 'max_dd' in r) / max(len(results), 1)
    print(f"{'TOTAL / AVERAGE':<35} ${total_pnl:>8.0f} {total_pnl/100:>7.1f}% {'-':>8} {avg_dd:>7.1f}%")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
