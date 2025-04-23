import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from lob_engine import LimitOrderBook
import plotly.graph_objects as go
import time

st.set_page_config(layout="wide")

# Load Data Function
@st.cache_data
def load_data():
    messages = pd.read_csv('data/AAPL_2012-06-21_34200000_57600000_message_10.csv', header=None)
    messages.columns = ['Time', 'Type', 'OrderID', 'Size', 'Price', 'Direction']
    messages['Price'] = messages['Price'] / 10000
    messages['ReadableTime'] = pd.to_datetime(messages['Time'], unit='s').dt.strftime('%H:%M:%S.%f')
    return messages

# Load messages
messages = load_data()

# Streamlit App Title
st.title("Limit Order Book Simulation")

# Sidebar Controls
st.header("Simulation Settings")
steps = st.slider("Number of Steps", min_value=10, max_value=1000, value=100, step=10)
delay = st.slider("Delay per Step (seconds)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

# Placeholders for live updates
status_placeholder = st.empty()
table_placeholder = st.empty()

# Layout for charts
col1, col2 = st.columns(2)
with col1:
    chart_placeholder = st.empty()
with col2:
    depth_chart_placeholder = st.empty()

# Run Simulation Button
if st.button("Run Live Simulation"):
    lob = LimitOrderBook()

    for idx, row in messages.iloc[:steps].iterrows():
        lob.process_message(row)

        # Capture snapshot every 5 steps
        if idx % 5 == 0 or idx == steps - 1:
            lob.capture_snapshot(idx)

            snapshots_df = pd.DataFrame(lob.snapshots)
            executions_df = pd.DataFrame(lob.executions)

            # Update Best Bid/Ask Status
            latest_bid = snapshots_df['BestBid'].iloc[-1]
            latest_ask = snapshots_df['BestAsk'].iloc[-1]
            status_placeholder.markdown(f"### Step {idx}: Best Bid = `{latest_bid}`, Best Ask = `{latest_ask}`")

            # --- Plot Best Bid/Ask Evolution ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=snapshots_df['Step'], y=snapshots_df['BestBid'], name='Best Bid'))
            fig.add_trace(go.Scatter(x=snapshots_df['Step'], y=snapshots_df['BestAsk'], name='Best Ask'))
            fig.update_layout(title="Best Bid / Ask Evolution", xaxis_title="Step", yaxis_title="Price ($)")
            chart_placeholder.plotly_chart(fig, use_container_width=True)

            # --- Plot Depth Chart ---
            if lob.bids and lob.asks:
                bid_prices = sorted(lob.bids.keys(), reverse=True)[:10]
                ask_prices = sorted(lob.asks.keys())[:10]

                bid_sizes = [sum(order[1] for order in lob.bids[p]) for p in bid_prices]
                ask_sizes = [sum(order[1] for order in lob.asks[p]) for p in ask_prices]

                bid_cum_sizes = pd.Series(bid_sizes).cumsum()
                ask_cum_sizes = pd.Series(ask_sizes).cumsum()

                depth_fig = go.Figure()
                depth_fig.add_trace(go.Scatter(x=bid_prices, y=bid_cum_sizes, mode='lines', name='Bid Depth', fill='tozeroy'))
                depth_fig.add_trace(go.Scatter(x=ask_prices, y=ask_cum_sizes, mode='lines', name='Ask Depth', fill='tozeroy'))

                depth_fig.update_layout(title="Live Order Book Depth Chart", xaxis_title="Price ($)", yaxis_title="Cumulative Size")
                depth_chart_placeholder.plotly_chart(depth_fig, use_container_width=True)

            # --- Update Execution Log ---
            table_placeholder.dataframe(executions_df.tail(5), use_container_width=True)

        time.sleep(delay)

    st.success("Simulation Complete!")

    # --- Execution Summary ---
    st.subheader("Simulation Summary")
    total_trades = len(lob.executions)
    total_volume = sum(exec['Size'] for exec in lob.executions)
    avg_price = (sum(exec['Price'] * exec['Size'] for exec in lob.executions) / total_volume) if total_volume > 0 else 0

    st.write(f"**Total Trades:** {total_trades}")
    st.write(f"**Total Volume Traded:** {total_volume}")
    st.write(f"**Average Execution Price:** ${avg_price:.2f}")