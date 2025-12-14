import streamlit as st
import pandas as pd
import time
import os
import yaml
import json
import plotly.express as px
import datetime

# --- Page Config ---
st.set_page_config(
    page_title="GRVT Bot Dashboard",
    page_icon="🤖",
    layout="wide",
)

# --- Title ---
st.title("🚀 GRVT Market Maker Bot")

# --- Sidebar: Configuration ---
st.sidebar.header("⚙️ Configuration")

# Refresh Control
refresh_interval = st.sidebar.slider("Refresh Interval (sec)", 1, 60, 10)
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=True)

CONFIG_PATH = "config.yaml"

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return f.read()
    return ""

def save_config(config_str):
    with open(CONFIG_PATH, "w") as f:
        f.write(config_str)

config_content = load_config()
with st.sidebar.expander("📝 Edit Config.yaml"):
    new_config = st.text_area("YAML Config", value=config_content, height=300)
    if st.button("💾 Save Config"):
        save_config(new_config)
        st.success("Configuration Saved!")

# --- Main Area: Bot Control ---
st.subheader("🎮 Bot Control")
col_ctrl1, col_ctrl2 = st.columns(2)
command_file = os.path.join("data", "command.json")

with col_ctrl1:
    if st.button("▶️ Start Bot", type="primary", use_container_width=True):
        with open(command_file, "w") as f:
            json.dump({"command": "start"}, f)
        st.toast("Sent START command!", icon="🚀")

with col_ctrl2:
    if st.button("🛑 Stop & Close Position", type="secondary", use_container_width=True):
        with open(command_file, "w") as f:
            json.dump({"command": "stop_close"}, f)
        st.toast("Sent STOP & CLOSE command!", icon="🛑")

st.divider()

# --- Status Data Loading with Persistence ---
paper_status_file = os.path.join("data", "paper_status.json")
history_file = os.path.join("data", "pnl_history.csv")
trade_file = os.path.join("data", "trade_history.csv")

# Initialize Session State for Status if not present to prevent flickering
if 'last_valid_status' not in st.session_state:
    st.session_state.last_valid_status = {}

status = {}
try:
    if os.path.exists(paper_status_file):
        with open(paper_status_file, "r") as f:
            data = json.load(f)
            if data:  # Ensure it's not empty
                status = data
                st.session_state.last_valid_status = data # Update cache
except Exception as e:
    # On read failure (race condition), fallback to last known good state
    # This prevents the dashboard from flashing "0"
    pass

# Fallback to last valid state if current read failed
if not status and st.session_state.last_valid_status:
    status = st.session_state.last_valid_status

# --- Metrics Section ---
st.subheader("📊 Live Performance")
col1, col2, col3, col4 = st.columns(4)

balance = status.get('balance', {})
pos = status.get('position', {})
mid_price = status.get('mid_price', 0)

usdt_bal = balance.get('USDT', 0.0)
quantity = pos.get('amount', 0.0)
entry_price = pos.get('entryPrice', 0.0)
unrealized_pnl = pos.get('unrealizedPnL', 0.0)
total_equity = usdt_bal + unrealized_pnl

with col1:
    st.metric("💰 Total Equity", f"${total_equity:,.2f}", delta=f"{unrealized_pnl:+.2f} (Unrealized)")

with col2:
    st.metric("📉 Mid Price", f"${mid_price:,.1f}")

with col3:
    st.metric("📦 Position (BTC)", f"{quantity:.4f}", delta=f"Entry: ${entry_price:,.0f}")

with col4:
    open_orders = status.get('open_orders', 0)
    st.metric("Open Orders", open_orders)

# --- Charts Section ---
st.divider()
col_chart1, col_chart2 = st.columns(2)

df_hist = pd.DataFrame()
try:
    if os.path.exists(history_file):
        # Try reading with header inference
        df_temp = pd.read_csv(history_file, on_bad_lines='skip')
        
        # Check if header is missing (e.g. first column is float timestamp)
        # If 'timestamp' column is missing, likely headerless
        if 'timestamp' not in df_temp.columns and not df_temp.empty:
             # Reload with explicit column names
             df_temp = pd.read_csv(history_file, names=["timestamp", "total_usdt_value", "realized_pnl", "price"], on_bad_lines='skip')
             
        if not df_temp.empty and len(df_temp) > 0:
            # Convert to KST (Korea Standard Time)
            df_temp['datetime'] = pd.to_datetime(df_temp['timestamp'], unit='s', utc=True)
            
            # Filter Invalid Data (Prevent Memory Explosion from bad timestamps)
            df_temp = df_temp[df_temp['datetime'].dt.year >= 2024]
            
            df_temp['datetime'] = df_temp['datetime'].dt.tz_convert('Asia/Seoul')
            
            # Resampling Logic (5s)
            # Note: resample needs index. dt accessors work, but easier to set index
            df_resampled = df_temp.set_index('datetime').resample('5s').last().dropna().reset_index()
            
            if len(df_resampled) > 12:
                df_hist = df_resampled.tail(600) # Show last ~50 mins of smoothed data
            else:
                df_hist = df_temp.tail(2000) # Fallback to raw data
except Exception as e:
    st.error(f"Error loading history: {e}")

with col_chart1:
    st.subheader("📈 Equity Curve")
    if not df_hist.empty:
        fig_equity = px.line(df_hist, x='datetime', y='total_usdt_value', 
                             title="Total Equity (USDT)", 
                             labels={'total_usdt_value': 'USDT'})
        
        # High Sensitivity Y-Axis Scaling
        min_eq = df_hist['total_usdt_value'].min()
        max_eq = df_hist['total_usdt_value'].max()
        range_eq = max_eq - min_eq
        
        margin = range_eq * 0.05 # 5% margin
        if margin < 0.5: margin = 0.5 # Minimum buffer
        
        fig_equity.update_layout(
            xaxis_title=None, 
            height=350,
            yaxis=dict(range=[min_eq - margin, max_eq + margin], tickformat=".2f")
        )
        st.plotly_chart(fig_equity, use_container_width=True)
    else:
        st.info("Waiting for data...")

with col_chart2:
    st.subheader("📉 Price History")
    if not df_hist.empty:
        fig_price = px.line(df_hist, x='datetime', y='price', 
                            title="BTC Price",
                            labels={'price': 'USDT'})
        
        # High Sensitivity Y-Axis Scaling
        min_p = df_hist['price'].min()
        max_p = df_hist['price'].max()
        range_p = max_p - min_p
        
        margin_p = range_p * 0.05
        if margin_p < 5.0: margin_p = 5.0
        
        fig_price.update_layout(
            xaxis_title=None, 
            height=350,
            yaxis=dict(range=[min_p - margin_p, max_p + margin_p], tickformat=".1f")
        )
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.info("Waiting for data...")

# --- Trade History Section ---
st.divider()
st.subheader("📜 Trade History")

if os.path.exists(trade_file):
    try:
        df_trade = pd.read_csv(trade_file)
        if not df_trade.empty:
            # Convert to KST
            df_trade['datetime'] = pd.to_datetime(df_trade['timestamp'], unit='s', utc=True)
            df_trade['datetime'] = df_trade['datetime'].dt.tz_convert('Asia/Seoul')
            
            df_display = df_trade[['datetime', 'action', 'symbol', 'side', 'price', 'quantity', 'cost', 'fee', 'realized_pnl']].sort_values(by='datetime', ascending=False)
            
            st.dataframe(
                df_display, 
                use_container_width=True,
                column_config={
                    "datetime": st.column_config.DatetimeColumn("Time (KST)", format="MM-DD HH:mm:ss"),
                    "action": "Action",
                    "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "cost": st.column_config.NumberColumn("Cost", format="$%.2f"),
                    "fee": st.column_config.NumberColumn("Fee (Rebate if +)", format="$%.4f"),
                    "realized_pnl": st.column_config.NumberColumn("Realized PnL", format="$%.2f"),
                }
            )
            
            csv_data = df_trade.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"trade_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No trades executed yet.")
    except Exception as e:
        st.error(f"Error reading trade history: {e}")
else:
    st.info("Trade history file not found.")

# --- Footer & Auto Refresh ---
if status:
    ts = status.get('timestamp', 0)
    # Also convert footer timestamp to KST
    try:
        last_update = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=9)))
        st.caption(f"Last Bot Update: {last_update.strftime('%H:%M:%S')} (KST)")
    except:
        st.caption(f"Last Bot Update: {ts}")

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
