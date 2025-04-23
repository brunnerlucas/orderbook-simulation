import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from lob_engine import LimitOrderBook
import plotly.graph_objects as go
import time

# --- Order Book Class ---
class OrderBook:
    def __init__(self):
        self.bids = defaultdict(deque)
        self.asks = defaultdict(deque)
        self.order_map = {}  # order_id -> (side, price)

    def add_order(self, order_id, side, price, size):
        book = self.bids if side == 'buy' else self.asks
        book[price].append((order_id, size))
        self.order_map[order_id] = (side, price)

    def execute_order(self, side, size):
        book = self.asks if side == 'buy' else self.bids
        sorted_prices = sorted(book.keys()) if side == 'buy' else sorted(book.keys(), reverse=True)
        remaining = size

        for price in sorted_prices:
            while book[price] and remaining > 0:
                order_id, order_size = book[price][0]
                if order_size > remaining:
                    book[price][0] = (order_id, order_size - remaining)
                    remaining = 0
                else:
                    remaining -= order_size
                    book[price].popleft()
                    self.order_map.pop(order_id, None)
            if not book[price]:
                del book[price]
            if remaining <= 0:
                break

    def cancel_order(self, order_id):
        if order_id not in self.order_map:
            return
        side, price = self.order_map.pop(order_id)
        book = self.bids if side == 'buy' else self.asks
        book[price] = deque([(oid, s) for oid, s in book[price] if oid != order_id])
        if not book[price]:
            del book[price]

    def replace_order(self, order_id, new_size):
        if order_id not in self.order_map:
            return
        side, price = self.order_map[order_id]
        book = self.bids if side == 'buy' else self.asks
        book[price] = deque([(oid, new_size if oid == order_id else s) for oid, s in book[price]])

    def get_depth(self):
        bid_prices = sorted(self.bids.keys(), reverse=True)[:10]
        ask_prices = sorted(self.asks.keys())[:10]
        bid_sizes = [sum(s for _, s in self.bids[p]) for p in bid_prices]
        ask_sizes = [sum(s for _, s in self.asks[p]) for p in ask_prices]
        return bid_prices, bid_sizes, ask_prices, ask_sizes

# --- Plotting Function ---
def plot_order_book_snapshot(bid_prices, bid_sizes, ask_prices, ask_sizes):
    cum_bid = np.cumsum(bid_sizes[::-1])[::-1] if bid_sizes else []
    cum_ask = np.cumsum(ask_sizes) if ask_sizes else []

    fig, ax = plt.subplots(figsize=(10, 5))
    if bid_prices:
        ax.fill_between(bid_prices, cum_bid, step="post", color="green", alpha=0.4, label="Bids")
        st.write(f"Top Bid: {bid_prices[0]} | Size: {bid_sizes[0]}k")
    else:
        st.warning("No bids in this snapshot.")

    if ask_prices:
        ax.fill_between(ask_prices, cum_ask, step="post", color="red", alpha=0.4, label="Asks")
        st.write(f"Top Ask: {ask_prices[0]} | Size: {ask_sizes[0]}k")
    else:
        st.warning("No asks in this snapshot.")

    ax.set_title("Limit Order Book Depth")
    ax.set_xlabel("Price")
    ax.set_ylabel("Cumulative Size")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


# --- Streamlit UI ---
st.title("Limit Order Book Simulator")



df = pd.read_csv("./data/AAPL_2012-06-21_34200000_57600000_message_10.csv")
df.columns= ["Time", "Type", "Order ID", "Size", "Price", "Direction"]
base_date = pd.to_datetime("2012-06-21")
df["Time"] = df["Time"].apply(lambda x: base_date + pd.to_timedelta(x, unit="s"))

df['Side'] = df['Direction'].map({1: 'buy', -1: 'sell'})

book = OrderBook()
snapshots = []

for i, row in df.iterrows():
    order_type = row['Type']
    order_id = row['Order ID']
    side = row['Side']
    price = row['Price']
    size = row['Size']

    if order_type == 1:
        book.add_order(order_id, side, price, size)
    elif order_type == 3:
        book.execute_order(side, size)
    elif order_type == 4:
        book.cancel_order(order_id)
    elif order_type == 5:
        book.replace_order(order_id, size)

    if i % 10000 == 0:
        snapshots.append(book.get_depth())

# --- UI: Only run after snapshots exist ---
if snapshots:
    snapshot_idx = st.slider("Select orders", 0, len(snapshots) - 1, 0)
    bid_prices, bid_sizes, ask_prices, ask_sizes = snapshots[snapshot_idx]
    plot_order_book_snapshot(bid_prices, bid_sizes, ask_prices, ask_sizes)

    if st.checkbox("Add Custom Market Order"):
        side = st.selectbox("Side", ["buy", "sell"])
        size = st.number_input("Size", min_value=1, value=10)
        if st.button("Submit Market Order"):
            book.execute_order(side, size)
            bid_prices, bid_sizes, ask_prices, ask_sizes = book.get_depth()
            plot_order_book_snapshot(bid_prices, bid_sizes, ask_prices, ask_sizes)
else:
    st.warning("No snapshots available. Make sure your file has valid order data.")


st.divider()
st.title('LOB Engine')

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

# Sidebar Controls
st.sidebar.header("Simulation Settings")
steps = st.sidebar.slider("Number of Steps", min_value=10, max_value=1000, value=100, step=10)
delay = st.sidebar.slider("Delay per Step (seconds)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

# Placeholders for live updates
status_placeholder = st.empty()
chart_placeholder = st.empty()
table_placeholder = st.empty()

if st.sidebar.button("Run Live Simulation"):
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

            # Update Plotly Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=snapshots_df['Step'], y=snapshots_df['BestBid'], name='Best Bid'))
            fig.add_trace(go.Scatter(x=snapshots_df['Step'], y=snapshots_df['BestAsk'], name='Best Ask'))
            fig.update_layout(title="Best Bid / Ask Evolution", xaxis_title="Step", yaxis_title="Price ($)")
            chart_placeholder.plotly_chart(fig)

            # Update Execution Log (last 5 trades)
            table_placeholder.dataframe(executions_df.tail(5))

        time.sleep(delay)

    st.success("Simulation Complete!")