import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import os
import yaml
import json

# Page Config
st.set_page_config(page_title="GRVT Bot Dashboard", layout="wide")

# Title
st.title("🚀 GRVT Trading Bot Dashboard")

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

CONFIG_PATH = "config.yaml"

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {}

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f)

config = load_config()

if config:
    # Strategy Settings
    st.sidebar.subheader("Strategy")
    spread = st.sidebar.number_input("Spread %", value=float(config.get('strategy', {}).get('spread_pct', 0.001)), format="%.4f")
    amount = st.sidebar.number_input("Order Amount", value=float(config.get('strategy', {}).get('order_amount', 0.001)), format="%.4f")
    
    # Risk Settings
    st.sidebar.subheader("Risk Management")
    max_pos = st.sidebar.number_input("Max Position (USD)", value=float(config.get('risk', {}).get('max_position_usd', 1000.0)))
    skew = st.sidebar.slider("Inventory Skew Factor", 0.0, 1.0, float(config.get('risk', {}).get('inventory_skew_factor', 0.0)))

    if st.sidebar.button("Save Config"):
        config['strategy']['spread_pct'] = spread
        config['strategy']['order_amount'] = amount
        config['risk']['max_position_usd'] = max_pos
        config['risk']['inventory_skew_factor'] = skew
        save_config(config)
        st.sidebar.success("Configuration Saved!")

# Main Dashboard Area
# 1. Paper Trading Status Section
st.subheader("📝 Paper Trading Status")
paper_status_file = os.path.join("data", "paper_status.json")

if os.path.exists(paper_status_file):
    try:
        with open(paper_status_file, "r") as f:
            status = json.load(f)
            
        col1, col2, col3, col4 = st.columns(4)
        
        balance = status.get('balance', {})
        pos = status.get('position', {})
        
        # Calculate Total Equity (USDT + BTC Value)
        # Note: We don't have real-time price here easily unless we read from CSV or API.
        # We'll use Entry Price as approximation or just show components.
        
        col1.metric("USDT Balance", f"${balance.get('USDT', 0):.2f}")
        col2.metric("BTC Position", f"{pos.get('amount', 0):.4f} BTC")
        col3.metric("Entry Price", f"${pos.get('entryPrice', 0):.2f}")
        col4.metric("Open Orders", status.get('open_orders', 0))
        
        st.caption(f"Last Updated: {time.ctime(status.get('timestamp', 0))}")
        
    except Exception as e:
        st.error(f"Error reading paper status: {e}")
else:
    st.info("Paper trading status not found. Run the bot in 'paper' mode.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Live Market Data")
    # Placeholder for live chart
    chart_placeholder = st.empty()

with col2:
    st.subheader("💰 Account & Performance")
    # Placeholder for metrics
    metrics_placeholder = st.empty()

# Real-time Data Simulation (Reading from latest CSV)
DATA_DIR = "data"

def get_latest_data_file():
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not files: return None
    return max(files, key=os.path.getctime)

latest_file = get_latest_data_file()

if latest_file:
    st.write(f"Reading from: `{latest_file}`")
    
    # Auto-refresh loop
    # In Streamlit, we usually use st.empty() and rerun, but for simple dashboard:
    if st.button("Refresh Data"):
        df = pd.read_csv(latest_file)
        if not df.empty:
            # Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['best_bid'], name='Bid', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['best_ask'], name='Ask', line=dict(color='red')))
            fig.update_layout(title="Price History", xaxis_title="Time", yaxis_title="Price")
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            last_row = df.iloc[-1]
            metrics_placeholder.metric("Current Price", f"${last_row['best_bid']:.2f}")
            metrics_placeholder.metric("Spread", f"{last_row['spread']:.2f}")
else:
    st.warning("No data found. Please run the Data Collector.")
