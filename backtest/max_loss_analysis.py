"""
Max Unrealized Loss Analysis
- Tracks position and unrealized PnL at each minute
- Finds maximum unrealized loss (drawdown) for circuit breaker calibration
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import time
import gc
import warnings
warnings.filterwarnings('ignore')

def fetch_month_1m_data(symbol: str, year: int, month: int) -> pd.DataFrame:
    """Fetch 1 month of 1-minute data"""
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)
    
    start_time = int(start_date.timestamp() * 1000)
    end_time = int(end_date.timestamp() * 1000)
    
    all_klines = []
    current_start = start_time
    
    while current_start < end_time:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            if not data or isinstance(data, dict):
                break
            all_klines.extend(data)
            current_start = data[-1][0] + 60000
        except:
            time.sleep(1)
            continue
    
    if not all_klines:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    return df

def simulate_with_position_tracking(df: pd.DataFrame, spread_pct: float = 0.001,
                                     order_usd: float = 100, max_position_usd: float = 1000) -> dict:
    """
    Simulate market making with detailed position and unrealized PnL tracking
    """
    n = len(df)
    mid = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    prev_mid = np.roll(mid, 1)
    prev_mid[0] = mid[0]
    
    # Spread levels
    bid_level = prev_mid * (1 - spread_pct/2)
    ask_level = prev_mid * (1 + spread_pct/2)
    
    # Fill detection
    bid_fill = (low <= bid_level).astype(float)
    ask_fill = (high >= ask_level).astype(float)
    
    # Track position and PnL at each step
    position_btc = np.zeros(n)
    avg_entry_price = np.zeros(n)
    realized_pnl = np.zeros(n)
    unrealized_pnl = np.zeros(n)
    
    cumulative_realized = 0
    current_position = 0
    current_avg_entry = 0
    
    for i in range(1, n):
        # Process bid fill (buy)
        if bid_fill[i] == 1:
            fill_price = bid_level[i]
            fill_qty = order_usd / fill_price
            
            # Check position limit
            current_pos_usd = abs(current_position) * mid[i]
            if current_pos_usd < max_position_usd:
                if current_position >= 0:
                    # Adding to long or opening long
                    new_cost = current_position * current_avg_entry + fill_qty * fill_price
                    current_position += fill_qty
                    current_avg_entry = new_cost / current_position if current_position > 0 else fill_price
                else:
                    # Closing short
                    close_qty = min(fill_qty, abs(current_position))
                    pnl = (current_avg_entry - fill_price) * close_qty
                    cumulative_realized += pnl
                    current_position += close_qty
        
        # Process ask fill (sell)
        if ask_fill[i] == 1:
            fill_price = ask_level[i]
            fill_qty = order_usd / fill_price
            
            current_pos_usd = abs(current_position) * mid[i]
            if current_pos_usd < max_position_usd:
                if current_position <= 0:
                    # Adding to short or opening short
                    if current_position == 0:
                        current_avg_entry = fill_price
                    else:
                        new_cost = abs(current_position) * current_avg_entry + fill_qty * fill_price
                        current_position -= fill_qty
                        current_avg_entry = new_cost / abs(current_position)
                    current_position -= fill_qty
                else:
                    # Closing long
                    close_qty = min(fill_qty, current_position)
                    pnl = (fill_price - current_avg_entry) * close_qty
                    cumulative_realized += pnl
                    current_position -= close_qty
        
        # Record state
        position_btc[i] = current_position
        avg_entry_price[i] = current_avg_entry
        realized_pnl[i] = cumulative_realized
        
        # Calculate unrealized PnL
        if current_position > 0:
            unrealized_pnl[i] = (mid[i] - current_avg_entry) * current_position
        elif current_position < 0:
            unrealized_pnl[i] = (current_avg_entry - mid[i]) * abs(current_position)
        else:
            unrealized_pnl[i] = 0
    
    # Total PnL = realized + unrealized
    total_pnl = realized_pnl + unrealized_pnl
    
    # Calculate drawdown from peak
    cumulative_total = np.cumsum(realized_pnl) + unrealized_pnl
    running_max = np.maximum.accumulate(cumulative_total)
    drawdown = running_max - cumulative_total
    
    return {
        'position_btc': position_btc,
        'realized_pnl': realized_pnl,
        'unrealized_pnl': unrealized_pnl,
        'total_pnl': total_pnl,
        'drawdown': drawdown,
        'max_unrealized_loss': abs(unrealized_pnl.min()),
        'max_drawdown': drawdown.max(),
        'min_unrealized': unrealized_pnl.min(),
        'max_unrealized': unrealized_pnl.max(),
        'final_pnl': cumulative_realized + unrealized_pnl[-1]
    }

def run_drawdown_analysis(months_back: int = 12):
    """Analyze max unrealized loss across 12 months"""
    print("=" * 70)
    print("  📊 MAX UNREALIZED LOSS ANALYSIS")
    print("  For Circuit Breaker Calibration")
    print("=" * 70)
    
    all_max_losses = []
    all_max_drawdowns = []
    monthly_stats = []
    
    now = datetime.now()
    months = []
    for i in range(months_back, 0, -1):
        target = now - timedelta(days=30*i)
        months.append((target.year, target.month))
    
    for idx, (year, month) in enumerate(months):
        month_name = f"{year}-{month:02d}"
        print(f"\n[{idx+1}/{len(months)}] {month_name}...")
        
        df = fetch_month_1m_data("BTCUSDT", year, month)
        if df.empty:
            continue
        
        print(f"  📥 {len(df):,} candles")
        
        result = simulate_with_position_tracking(
            df,
            spread_pct=0.001,
            order_usd=100,
            max_position_usd=1000
        )
        
        monthly_stats.append({
            'month': month_name,
            'max_unrealized_loss': result['max_unrealized_loss'],
            'min_unrealized': result['min_unrealized'],
            'max_drawdown': result['max_drawdown'],
            'final_pnl': result['final_pnl']
        })
        
        all_max_losses.append(result['max_unrealized_loss'])
        all_max_drawdowns.append(result['max_drawdown'])
        
        print(f"  Max Unrealized Loss: ${result['max_unrealized_loss']:.2f}")
        print(f"  Max Drawdown: ${result['max_drawdown']:.2f}")
        
        del df
        gc.collect()
    
    # Summary
    print("\n" + "=" * 70)
    print("  📊 SUMMARY: MAX UNREALIZED LOSS BY MONTH")
    print("=" * 70)
    
    print(f"\n  {'Month':<12} {'Max Loss':>12} {'Max DD':>12} {'Final PnL':>12}")
    print("  " + "-" * 52)
    
    for stat in monthly_stats:
        print(f"  {stat['month']:<12} ${stat['max_unrealized_loss']:>10.2f} ${stat['max_drawdown']:>10.2f} ${stat['final_pnl']:>10.2f}")
    
    # Overall stats
    overall_max_loss = max(all_max_losses)
    avg_max_loss = np.mean(all_max_losses)
    p95_max_loss = np.percentile(all_max_losses, 95)
    
    print("\n" + "=" * 70)
    print("  🎯 CIRCUIT BREAKER RECOMMENDATIONS")
    print("=" * 70)
    
    print(f"\n  Overall Max Unrealized Loss: ${overall_max_loss:.2f}")
    print(f"  Average Max Loss per Month:  ${avg_max_loss:.2f}")
    print(f"  95th Percentile:             ${p95_max_loss:.2f}")
    
    print("\n  📋 Recommended Circuit Breaker Settings:")
    print(f"     Conservative: ${p95_max_loss * 1.2:.0f}  (95th pct + 20% buffer)")
    print(f"     Moderate:     ${overall_max_loss * 1.0:.0f}  (Historical max)")
    print(f"     Aggressive:   ${avg_max_loss * 1.5:.0f}  (Avg max + 50% buffer)")
    
    print("=" * 70)
    
    # Generate chart
    output_path = r"C:\Users\camel\.gemini\antigravity\brain\ed4efe75-6a4b-4be3-971f-923ed59be9c8\max_loss_analysis.png"
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    months_names = [s['month'] for s in monthly_stats]
    max_losses = [s['max_unrealized_loss'] for s in monthly_stats]
    
    bars = ax.bar(months_names, max_losses, color='#FF5722', alpha=0.7, edgecolor='black')
    
    # Add threshold lines
    ax.axhline(y=overall_max_loss, color='red', linestyle='--', linewidth=2, label=f'Max: ${overall_max_loss:.0f}')
    ax.axhline(y=p95_max_loss, color='orange', linestyle='--', linewidth=2, label=f'95th pct: ${p95_max_loss:.0f}')
    ax.axhline(y=avg_max_loss, color='green', linestyle='--', linewidth=2, label=f'Avg: ${avg_max_loss:.0f}')
    
    ax.set_ylabel('Max Unrealized Loss ($)', fontsize=11)
    ax.set_xlabel('Month', fontsize=11)
    ax.set_title('Maximum Unrealized Loss by Month (Circuit Breaker Calibration)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n📊 Chart saved to: {output_path}")
    
    return monthly_stats, overall_max_loss, p95_max_loss, avg_max_loss

if __name__ == "__main__":
    stats, max_loss, p95, avg = run_drawdown_analysis(months_back=12)
