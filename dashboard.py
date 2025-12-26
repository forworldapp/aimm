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

# --- Utils ---
def load_yaml_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def save_yaml_config(data):
    with open("config.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False)

def send_command(action):
    """Send command to bot via JSON file."""
    command_file = os.path.join("data", "command.json")
    try:
        with open(command_file, "w") as f:
            json.dump({"command": action, "action": action}, f) # Support both keys for compatibility
    except Exception as e:
        st.error(f"Failed to send command: {e}")

# --- Sidebar Configuration ---
st.sidebar.title("⚙️ Configuration")
# (Rest of Sidebar)

# Refresh Control
refresh_interval = st.sidebar.slider("Refresh Interval (sec)", 1, 60, 5)
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=True)

CONFIG_PATH = "config.yaml"

# Load Config
try:
    with open(CONFIG_PATH, "r") as f:
        config_data = yaml.safe_load(f) or {}
except Exception:
    config_data = {}

st.sidebar.subheader("⚡ Quick Settings")
# Grid Layers
current_layers = int(config_data.get('strategy', {}).get('grid_layers', 3))
layers_input = st.sidebar.number_input("Grid Layers", min_value=1, max_value=10, value=current_layers)

# Entry Anchor Mode (v1.2)
current_anchor = bool(config_data.get('strategy', {}).get('entry_anchor_mode', True))
anchor_mode = st.sidebar.checkbox("🛡️ Entry Anchor Mode", value=current_anchor, help="If ON, only buys below entry (long) or sells above entry (short).")

# Trend Follow Strategy (v1.3)
current_strategy = config_data.get('strategy', {}).get('trend_strategy', 'adaptive')
# Backward compatibility
if str(current_strategy).lower() == 'true': current_strategy = 'ma_trend'
if str(current_strategy).lower() == 'false': current_strategy = 'off'

trend_options = ['off', 'ma_trend', 'adaptive', 'adx', 'atr', 'chop', 'combo', 'rsi', 'bollinger']
try:
    idx = trend_options.index(current_strategy)
except:
    idx = 2 # Default adaptive

target_strategy = st.sidebar.selectbox(
    "📊 Trend Strategy", 
    trend_options, 
    index=idx, 
    help="Select Technical Indicator:\n- off: Pure Grid\n- ma_trend: SMA Divergence\n- adx: ADX > 25\n- atr: Volatility Breakout\n- chop: Choppiness Index < 40\n- combo: ADX + ATR (Ultimate)"
)

# Risk Params
current_dd = float(config_data.get('risk', {}).get('max_drawdown_pct', 0.10))
drawdown_input = st.sidebar.slider("Max Drawdown %", 0.01, 0.20, value=current_dd, format="%.2f")

current_pos_usd = float(config_data.get('risk', {}).get('max_position_usd', 1000.0))
pos_limit = st.sidebar.number_input("Max Pos (USD)", 100.0, 10000.0, value=current_pos_usd)

current_order_usd = float(config_data.get('strategy', {}).get('order_size_usd', 100.0))
order_size_usd = st.sidebar.number_input("Order Size (USD)", 10.0, 5000.0, value=current_order_usd, step=10.0)

if st.sidebar.button("💾 Apply & Reload Bot"):
    # Update Data
    if 'strategy' not in config_data: config_data['strategy'] = {}
    if 'risk' not in config_data: config_data['risk'] = {}
    
    config_data['strategy']['grid_layers'] = layers_input
    config_data['strategy']['entry_anchor_mode'] = anchor_mode
    config_data['strategy']['trend_strategy'] = target_strategy
    config_data['risk']['max_drawdown_pct'] = drawdown_input
    config_data['risk']['max_drawdown_pct'] = drawdown_input
    config_data['risk']['max_position_usd'] = pos_limit
    config_data['strategy']['order_size_usd'] = order_size_usd
    
    # Save YAML
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)
    
    # Send Reload Command
    command_file = os.path.join("data", "command.json")
    with open(command_file, "w") as f:
        json.dump({"command": "reload_config"}, f)
        
    st.sidebar.success("Params Updated & Bot Reloaded!")

# Advanced: Raw YAML Editor
with st.sidebar.expander("📝 Advanced YAML Edit"):
    # Use config_data to show current state
    raw_yaml = yaml.dump(config_data, default_flow_style=False)
    st.text_area("Read-Only Config View", value=raw_yaml, height=200, disabled=True)

st.sidebar.markdown("---")

# --- V1.5 Multi-Symbol Support ---
st.sidebar.subheader("📈 Trading Pair")

# Read current config directly to avoid stale state
try:
    with open("config.yaml", "r") as f:
        raw_config = yaml.safe_load(f)
        current_symbol = raw_config.get('exchange', {}).get('symbol', 'BTC_USDT_Perp')
except:
    current_symbol = 'BTC_USDT_Perp'

available_symbols = ["BTC_USDT_Perp", "ETH_USDT_Perp", "SOL_USDT_Perp", "XRP_USDT_Perp"]

# If current symbol is not in list, add it
if current_symbol not in available_symbols:
    available_symbols.insert(0, current_symbol)
    
selected_symbol = st.sidebar.selectbox("Select Pair", available_symbols, index=available_symbols.index(current_symbol))

if selected_symbol != current_symbol:
    if st.sidebar.button(f"Apply {selected_symbol} & Restart"):
        # Update Config
        if 'exchange' not in raw_config: raw_config['exchange'] = {}
        raw_config['exchange']['symbol'] = selected_symbol
        with open("config.yaml", "w") as f:
            yaml.dump(raw_config, f)
        
        # Send Restart Command
        send_command("restart")
        st.sidebar.success(f"Switched to {selected_symbol}! Bot Restarting...")
        time.sleep(2)
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.title("🎮 Bot Control")
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
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

with col_ctrl3:
    if st.button("💀 Shutdown Process", type="primary", use_container_width=True):
        with open(command_file, "w") as f:
            json.dump({"command": "shutdown"}, f)
        st.toast("Sent SHUTDOWN command! Bot will exit.", icon="💀")

st.divider()

# --- Status Data Loading with Persistence ---
# --- Status Data Loading with Persistence ---
# V1.5: Dynamic File Paths based on Symbol
# Ensure current_symbol is loaded (it is loaded above in Sidebar section)
if 'current_symbol' not in locals():
    # Fallback if sidebar code didn't run yet (unlikely but safe)
    try:
        with open("config.yaml", "r") as f:
            raw_c = yaml.safe_load(f)
            current_symbol = raw_c.get('exchange', {}).get('symbol', 'BTC_USDT_Perp')
    except:
        current_symbol = 'BTC_USDT_Perp'

paper_status_file = os.path.join("data", f"paper_status_{current_symbol}.json")
history_file = os.path.join("data", f"pnl_history_{current_symbol}.csv")
trade_file = os.path.join("data", f"trade_history_{current_symbol}.csv")

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
# Adjust column ratios to give more space to Regime (col5)
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 0.7, 1.8])

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
    
    # Order Details Expander
    orders_list = status.get('open_orders_list', [])
    if orders_list:
        with st.expander("Order Details", expanded=True):
            # Sort: Buy (Desc), Sell (Asc)
            bids = sorted([o for o in orders_list if o['side'] == 'buy'], key=lambda x: x['price'], reverse=True)
            asks = sorted([o for o in orders_list if o['side'] == 'sell'], key=lambda x: x['price'])
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🟢 Bids**")
                for o in bids:
                    st.text(f"${o['price']:,.1f} ({o['amount']})")
            with c2:
                st.markdown("**🔴 Asks**")
                for o in asks:
                    st.text(f"${o['price']:,.1f} ({o['amount']})")

# Move Regime to full width box to prevent truncation
regime = status.get('market_regime', 'N/A').upper()
if "BUY" in regime:
    st.success(f"🚦 Regime: {regime}")
elif "SELL" in regime:
    st.error(f"🚦 Regime: {regime}")
else:
    st.info(f"🚦 Regime: {regime}")
# with col5:
#    st.metric("🚦 Regime", regime, help="Adaptive Mode State")

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
            
            # Match new CSV Header: timestamp, symbol, side, price, amount, cost, rebate, realized_pnl, note
            df_display = df_trade[['datetime', 'note', 'symbol', 'side', 'price', 'amount', 'cost', 'rebate', 'realized_pnl']].sort_values(by='datetime', ascending=False)
            
            st.dataframe(
                df_display, 
                use_container_width=True,
                column_config={
                    "datetime": st.column_config.DatetimeColumn("Time (KST)", format="MM-DD HH:mm:ss"),
                    "note": "Action",
                    "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "cost": st.column_config.NumberColumn("Cost", format="$%.2f"),
                    "rebate": st.column_config.NumberColumn("Fee/Rebate", format="$%.4f"),
                    "realized_pnl": st.column_config.NumberColumn("Realized PnL", format="$%.4f"),      
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
