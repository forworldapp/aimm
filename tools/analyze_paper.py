import pandas as pd
from datetime import datetime

# Load trade history
df = pd.read_csv('data/trade_history_BTC_USDT_Perp.csv')
df['dt'] = pd.to_datetime(df['timestamp'], unit='s')
for col in ['realized_pnl', 'grid_profit', 'rebate', 'cost', 'price', 'amount']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print('=' * 60)
print('  PAPER TRADING PERFORMANCE REPORT')
print('=' * 60)

# Basic stats
start = df.dt.min().strftime("%Y-%m-%d %H:%M")
end = df.dt.max().strftime("%Y-%m-%d %H:%M")
duration = (df.dt.max() - df.dt.min()).days
print(f"\nPeriod: {start} ~ {end}")
print(f"Duration: {duration} days")
print(f"Total Trades: {len(df)}")
print(f"Buy Trades: {len(df[df.side=='buy'])}")
print(f"Sell Trades: {len(df[df.side=='sell'])}")

# PnL analysis
total_realized = df['realized_pnl'].sum()
total_grid_profit = df['grid_profit'].sum()
total_rebates = df['rebate'].sum()
total_volume = df['cost'].sum()

print(f"\n--- PnL ---")
print(f"Realized PnL: ${total_realized:.2f}")
print(f"Grid Profit: ${total_grid_profit:.2f}")
print(f"Total Rebates: ${total_rebates:.2f}")
print(f"Total Volume: ${total_volume:,.2f}")

# Winning/Losing trades
profitable = df[df.realized_pnl > 0]
losing = df[df.realized_pnl < 0]
neutral = df[df.realized_pnl == 0]
print(f"\n--- Win/Loss ---")
print(f"Profitable Trades: {len(profitable)} ({len(profitable)/len(df)*100:.1f}%)")
print(f"Losing Trades: {len(losing)} ({len(losing)/len(df)*100:.1f}%)")
print(f"Neutral (open/increase): {len(neutral)}")

if len(profitable) > 0:
    print(f"Avg Win: ${profitable.realized_pnl.mean():.2f}")
if len(losing) > 0:
    print(f"Avg Loss: ${losing.realized_pnl.mean():.2f}")

# Best/Worst trades
print(f"\nBest Trade: ${df.realized_pnl.max():.2f}")
print(f"Worst Trade: ${df.realized_pnl.min():.2f}")

# Daily breakdown
df['date'] = df['dt'].dt.date
daily = df.groupby('date').agg({
    'realized_pnl': 'sum',
    'grid_profit': 'sum',
    'rebate': 'sum',
    'cost': 'sum',
    'timestamp': 'count'
}).rename(columns={'timestamp': 'trades'})

print(f"\n--- Daily Breakdown ---")
for date, row in daily.iterrows():
    pnl = row['realized_pnl']
    sym = '+' if pnl >= 0 else ''
    print(f"{date} | PnL: {sym}${pnl:.2f} | Trades: {int(row['trades']):>4} | Vol: ${row['cost']:>10,.0f}")

print(f"\n--- Summary ---")
print(f"Net PnL: ${total_realized:.2f}")
print(f"Net PnL + Rebates: ${total_realized + total_rebates:.2f}")
profitable_days = len(daily[daily.realized_pnl > 0])
total_days = len(daily)
print(f"Profitable Days: {profitable_days}/{total_days} ({profitable_days/total_days*100:.0f}%)")
print(f"Avg Daily PnL: ${total_realized / total_days:.2f}")
print(f"Avg Daily Volume: ${total_volume / total_days:,.0f}")
