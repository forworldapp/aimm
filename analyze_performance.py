
import pandas as pd
import os

def analyze():
    if not os.path.exists('data/trade_history.csv'):
        print("No trade history found.")
        return

    try:
        df = pd.read_csv('data/trade_history.csv')
        if df.empty:
            print("Trade history is empty.")
            return

        # Basic Stats
        total_trades = len(df)
        total_pnl = df['realized_pnl'].sum()
        win_trades = df[df['realized_pnl'] > 0]
        loss_trades = df[df['realized_pnl'] < 0]
        
        win_rate = len(win_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_win = win_trades['realized_pnl'].mean() if not win_trades.empty else 0
        avg_loss = loss_trades['realized_pnl'].mean() if not loss_trades.empty else 0
        
        print(f"Total Trades: {total_trades}")
        print(f"Total PnL: {total_pnl:.2f}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Avg Win: {avg_win:.4f} | Avg Loss: {avg_loss:.4f}")
        print(f"Profit Factor: {abs(win_trades['realized_pnl'].sum() / loss_trades['realized_pnl'].sum()) if not loss_trades.empty and loss_trades['realized_pnl'].sum() != 0 else 0:.2f}")

        # Recent performace (last 24h)
        # Assuming timestamp is available? Let's check columns first
        print("Columns:", df.columns)
        print("Last 5 trades:\n", df.tail(5))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze()
