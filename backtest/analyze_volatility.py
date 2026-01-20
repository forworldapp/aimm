"""
1-Year BTC Volatility Regime Analysis
=====================================
Analyzes volatility distribution to understand regime proportions
for hybrid strategy optimization.
"""
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time

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
            resp = requests.get(url, params=params)
            data = resp.json()
            
            if not data:
                break
                
            all_data.extend(data)
            current_start = data[-1][6] + 1  # Close time + 1ms
            
            print(f"  Fetched {len(all_data)} candles so far...")
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"Error: {e}")
            break
    
    return all_data

def analyze_volatility_regimes(df: pd.DataFrame, thresholds: dict) -> dict:
    """Analyze volatility regime distribution."""
    # Calculate rolling volatility (20-period std of returns)
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df = df.dropna()
    
    low_thresh = thresholds.get('low', 0.0005)
    med_thresh = thresholds.get('medium', 0.0008)
    
    # Classify regimes
    df['regime'] = 'high'
    df.loc[df['volatility'] < med_thresh, 'regime'] = 'medium'
    df.loc[df['volatility'] < low_thresh, 'regime'] = 'low'
    
    # Calculate statistics
    total = len(df)
    regime_counts = df['regime'].value_counts()
    
    results = {
        'total_candles': total,
        'regime_distribution': {
            'low': regime_counts.get('low', 0) / total * 100,
            'medium': regime_counts.get('medium', 0) / total * 100,
            'high': regime_counts.get('high', 0) / total * 100
        },
        'avg_volatility': {
            'low': df[df['regime'] == 'low']['volatility'].mean() if 'low' in regime_counts else 0,
            'medium': df[df['regime'] == 'medium']['volatility'].mean() if 'medium' in regime_counts else 0,
            'high': df[df['regime'] == 'high']['volatility'].mean() if 'high' in regime_counts else 0
        }
    }
    
    # Calculate regime durations (consecutive periods)
    df['regime_change'] = df['regime'] != df['regime'].shift()
    df['regime_group'] = df['regime_change'].cumsum()
    
    duration_stats = df.groupby(['regime_group', 'regime']).size().reset_index(name='duration')
    
    for regime in ['low', 'medium', 'high']:
        regime_durations = duration_stats[duration_stats['regime'] == regime]['duration']
        if len(regime_durations) > 0:
            results[f'{regime}_avg_duration'] = regime_durations.mean()
            results[f'{regime}_max_duration'] = regime_durations.max()
            results[f'{regime}_occurrences'] = len(regime_durations)
        else:
            results[f'{regime}_avg_duration'] = 0
            results[f'{regime}_max_duration'] = 0
            results[f'{regime}_occurrences'] = 0
    
    return results

def main():
    print("=" * 60)
    print("      1-YEAR BTC VOLATILITY REGIME ANALYSIS")
    print("=" * 60)
    
    # Date range: 1 year of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"\nPeriod: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("\nFetching 1-year of 1-minute data from Binance...")
    
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    data = fetch_binance_klines("BTCUSDT", "1m", start_ts, end_ts)
    
    print(f"\nTotal candles fetched: {len(data)}")
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['close'] = df['close'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Analyze with thresholds from config
    thresholds = {'low': 0.0005, 'medium': 0.0008}
    results = analyze_volatility_regimes(df, thresholds)
    
    # Print results
    print("\n" + "=" * 60)
    print("                    ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\nTotal 1-minute candles analyzed: {results['total_candles']:,}")
    
    print("\n📊 REGIME DISTRIBUTION:")
    print("-" * 40)
    for regime in ['low', 'medium', 'high']:
        pct = results['regime_distribution'][regime]
        bar = '█' * int(pct / 2)
        print(f"  {regime.upper():8s}: {pct:5.1f}% {bar}")
    
    print("\n📈 AVERAGE VOLATILITY (1m returns std):")
    print("-" * 40)
    for regime in ['low', 'medium', 'high']:
        vol = results['avg_volatility'][regime] * 100
        print(f"  {regime.upper():8s}: {vol:.4f}%")
    
    print("\n⏱️ REGIME DURATION (in 1-min candles):")
    print("-" * 40)
    for regime in ['low', 'medium', 'high']:
        avg_dur = results.get(f'{regime}_avg_duration', 0)
        max_dur = results.get(f'{regime}_max_duration', 0)
        occ = results.get(f'{regime}_occurrences', 0)
        print(f"  {regime.upper():8s}: Avg {avg_dur:5.0f}m, Max {max_dur:5.0f}m, Count {occ:5.0f}")
    
    print("\n" + "=" * 60)
    print("                 STRATEGY IMPLICATIONS")
    print("=" * 60)
    
    low_pct = results['regime_distribution']['low']
    high_med_pct = results['regime_distribution']['medium'] + results['regime_distribution']['high']
    
    print(f"""
    Low Volatility Time:    {low_pct:.1f}%
    Med/High Volatility:    {high_med_pct:.1f}%
    
    Expected Monthly Performance (Hybrid):
    - Low Vol ({low_pct:.1f}%): Quote Skip → ~+${7 * low_pct / 100 * 3:.0f}
    - Med/High Vol ({high_med_pct:.1f}%): v3.8.0 → ~+${343 * high_med_pct / 100:.0f}
    - TOTAL: ~+${7 * low_pct / 100 * 3 + 343 * high_med_pct / 100:.0f}/3-periods
    """)
    
    print("=" * 60)

if __name__ == "__main__":
    main()
