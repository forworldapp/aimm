import pandas as pd
import os
import json

trade_file = "data/trade_history.csv"
history_file = "data/pnl_history.csv"
status_file = "data/paper_status.json"

print("=== Bot Performance Diagnosis ===")

# 1. Trade History Analysis
if os.path.exists(trade_file):
    try:
        df = pd.read_csv(trade_file)
        if not df.empty:
            total_trades = len(df)
            
            # Filter Closed Trades (Realized PnL != 0 or Sell side closing)
            # PaperExchange logic records realized_pnl on SELL or Reduce
            closed_trades = df[df['realized_pnl'] != 0]
            
            total_pnl = df['realized_pnl'].sum()
            win_trades = closed_trades[closed_trades['realized_pnl'] > 0]
            loss_trades = closed_trades[closed_trades['realized_pnl'] <= 0]
            
            win_rate = (len(win_trades) / len(closed_trades)) * 100 if len(closed_trades) > 0 else 0
            
            print(f"\n[Trade Statistics]")
            print(f"Total Transactions: {total_trades}")
            print(f"PnL Realizing Trades: {len(closed_trades)}")
            print(f"Total Realized PnL: ${total_pnl:.4f}")
            print(f"Win Rate: {win_rate:.2f}% ({len(win_trades)} Wins / {len(loss_trades)} Losses)")
            
            if not closed_trades.empty:
                avg_win = win_trades['realized_pnl'].mean() if not win_trades.empty else 0
                avg_loss = loss_trades['realized_pnl'].mean() if not loss_trades.empty else 0
                print(f"Avg Win: ${avg_win:.4f} | Avg Loss: ${avg_loss:.4f}")
            
            print(f"\n[Last 5 Trades]")
            print(df[['timestamp', 'side', 'price', 'quantity', 'realized_pnl']].tail(5).to_string(index=False))
            
        else:
            print("\n[Trade History] File is empty.")
    except Exception as e:
        print(f"\n[Error] Reading trade history: {e}")
else:
    print(f"\n[Warning] {trade_file} not found.")

# 2. Equity History Analysis
if os.path.exists(history_file):
    try:
        df_hist = pd.read_csv(history_file)
        if not df_hist.empty:
            start_eq = df_hist.iloc[0]['total_usdt_value']
            end_eq = df_hist.iloc[-1]['total_usdt_value']
            print(f"\n[Equity Trend]")
            print(f"Start: ${start_eq:,.2f} -> Current: ${end_eq:,.2f}")
            print(f"Change: ${end_eq - start_eq:.2f} ({(end_eq - start_eq)/start_eq*100:.2f}%)")
        else:
            print("\n[History] File is empty.")
    except Exception as e:
        print(f"\n[Error] Reading pnl history: {e}")

# 3. Current Status
print("\n[Current Status]")
if os.path.exists(status_file):
    try:
        with open(status_file, 'r') as f:
            status = json.load(f)
            print(json.dumps(status, indent=2))
    except:
        print("Error reading status file")
