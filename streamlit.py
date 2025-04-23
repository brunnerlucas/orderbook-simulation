import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from lob_engine import LimitOrderBook
import plotly.graph_objects as go
import time
st.set_page_config(layout="wide")

tab1, tab2 = st.tabs(['Order Book Simulation', 'Order Book Step Injection'])

with tab1:
    import streamlit as st
    import pandas as pd
    import time
    from lob_engine import LimitOrderBook
    import plotly.graph_objects as go

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

    # Controls Section (Placed on main page instead of sidebar)
    with st.container():
        st.header("Simulation Settings")
        steps = st.slider("Number of Steps", min_value=10, max_value=1000, value=100, step=10)
        delay = st.slider("Delay per Step (seconds)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        run_simulation = st.button("Run Live Simulation")

    # Placeholders for live updates
    status_placeholder = st.empty()
    table_placeholder = st.empty()

    # Layout for charts
    col1, col2 = st.columns(2)
    with col1:
        chart_placeholder = st.empty()
    with col2:
        depth_chart_placeholder = st.empty()

    # Run Simulation Logic
    if run_simulation:
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
                fig.add_trace(go.Scatter(x=snapshots_df['Step'], y=snapshots_df['BestBid'], name='Best Bid', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=snapshots_df['Step'], y=snapshots_df['BestAsk'], name='Best Ask', line=dict(color='red')))
                fig.update_layout(title="Best Bid / Ask Evolution", xaxis_title="Step", yaxis_title="Price ($)")
                chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"chart_{idx}")


                # --- Plot Depth Chart ---
                if lob.bids and lob.asks:
                    bid_prices = sorted(lob.bids.keys(), reverse=True)[:10]
                    ask_prices = sorted(lob.asks.keys())[:10]

                    bid_sizes = [sum(order[1] for order in lob.bids[p]) for p in bid_prices]
                    ask_sizes = [sum(order[1] for order in lob.asks[p]) for p in ask_prices]

                    bid_cum_sizes = pd.Series(bid_sizes).cumsum()
                    ask_cum_sizes = pd.Series(ask_sizes).cumsum()

                    depth_fig = go.Figure()
                    depth_fig.add_trace(go.Scatter(x=bid_prices, y=bid_cum_sizes, mode='lines', name='Bid Depth', fill='tozeroy', line=dict(color='green')))
                    depth_fig.add_trace(go.Scatter(x=ask_prices, y=ask_cum_sizes, mode='lines', name='Ask Depth', fill='tozeroy', line=dict(color='red')))
                    depth_fig.update_layout(title="Live Order Book Depth Chart", xaxis_title="Price ($)", yaxis_title="Cumulative Size")
                    depth_chart_placeholder.plotly_chart(depth_fig, use_container_width=True, key=f"depth_{idx}")


                else:
                    depth_chart_placeholder.info("Waiting for enough liquidity on both sides to display Depth Chart.")

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





with tab2:
    # Load Data
    @st.cache_data
    def load_data():
        messages = pd.read_csv('data/AAPL_2012-06-21_34200000_57600000_message_10.csv', header=None)
        messages.columns = ['Time', 'Type', 'OrderID', 'Size', 'Price', 'Direction']
        messages['Price'] = messages['Price'] / 10000
        messages['ReadableTime'] = pd.to_datetime(messages['Time'], unit='s').dt.strftime('%H:%M:%S.%f')
        return messages

    messages = load_data()

    # Initialize session state
    if 'lob' not in st.session_state:
        st.session_state.lob = LimitOrderBook()
        st.session_state.current_step = 0
        st.session_state.custom_order_count = 1

    st.title("Manual Step Limit Order Book Simulator")

    # --- Inject Custom Order Controls ---
    st.write("### Inject Custom Order")

    # Select order type outside the form for dynamic input behavior
    order_type_input = st.selectbox("Order Type", ["Market", "Limit"], key="order_type_select")

    with st.form(key='inject_form'):
        side_input = st.selectbox("Side", ["Buy", "Sell"])
        if order_type_input == "Limit":
            price_input = st.number_input("Price ($)", min_value=0.0, step=0.01)
        else:
            price_input = None
        size_input = st.number_input("Size", min_value=1, step=1)
        submit_button = st.form_submit_button(label='Submit Order')

        if submit_button:
            direction = 1 if side_input == "Buy" else -1
            order_id = f'Custom_{st.session_state.custom_order_count}'
            book = st.session_state.lob.bids if direction == 1 else st.session_state.lob.asks
            opposite_book = st.session_state.lob.asks if direction == 1 else st.session_state.lob.bids

            executed = False
            remaining_size = size_input

            if order_type_input == "Market":
                prices = sorted(opposite_book.keys()) if direction == 1 else sorted(opposite_book.keys(), reverse=True)

                for p in prices:
                    orders = opposite_book[p]
                    for order in orders[:]:
                        trade_size = min(remaining_size, order[1])
                        remaining_size -= trade_size
                        order[1] -= trade_size
                        st.session_state.lob.executions.append({
                            'OrderID': order_id,
                            'Price': p,
                            'Size': trade_size,
                            'Direction': direction
                        })
                        if order[1] <= 0:
                            orders.remove(order)
                        if remaining_size <= 0:
                            break
                    if not orders:
                        del opposite_book[p]
                    if remaining_size <= 0:
                        break
                executed = True

            elif order_type_input == "Limit":
                cross_condition = (
                    (direction == 1 and st.session_state.lob.asks and price_input >= min(st.session_state.lob.asks.keys())) or
                    (direction == -1 and st.session_state.lob.bids and price_input <= max(st.session_state.lob.bids.keys()))
                )

                if cross_condition:
                    prices = sorted(opposite_book.keys()) if direction == 1 else sorted(opposite_book.keys(), reverse=True)
                    for p in prices:
                        if (direction == 1 and p > price_input) or (direction == -1 and p < price_input):
                            break
                        orders = opposite_book[p]
                        for order in orders[:]:
                            trade_size = min(remaining_size, order[1])
                            remaining_size -= trade_size
                            order[1] -= trade_size
                            st.session_state.lob.executions.append({
                                'OrderID': order_id,
                                'Price': p,
                                'Size': trade_size,
                                'Direction': direction
                            })
                            if order[1] <= 0:
                                orders.remove(order)
                            if remaining_size <= 0:
                                break
                        if not orders:
                            del opposite_book[p]
                        if remaining_size <= 0:
                            break
                    executed = True

                if remaining_size > 0:
                    book.setdefault(price_input, []).append([order_id, remaining_size])
                    st.session_state.lob.order_lookup[order_id] = (price_input, remaining_size, direction)

            if executed:
                st.success(f"Executed {order_type_input} {side_input} Order for {size_input - remaining_size} shares.")
            elif order_type_input == "Limit":
                st.success(f"Limit Order added to book: {remaining_size} @ ${price_input}")
            else:
                st.warning("Market order could not execute (no liquidity available).")

            st.session_state.custom_order_count += 1

    # --- Next Step Button ---
    if st.button("Next Step"):
        if st.session_state.current_step < len(messages):
            row = messages.iloc[st.session_state.current_step]
            st.session_state.lob.process_message(row)
            st.session_state.lob.capture_snapshot(st.session_state.current_step)
            st.session_state.current_step += 1
        else:
            st.warning("Simulation Complete!")

    # --- Reset Simulation Button ---
    if st.button("Reset Simulation"):
        st.session_state.lob = LimitOrderBook()
        st.session_state.current_step = 0
        st.session_state.custom_order_count = 1
        st.success("Simulation reset!")

    # --- Display Current Status ---
    snapshots_df = pd.DataFrame(st.session_state.lob.snapshots)
    executions_df = pd.DataFrame(st.session_state.lob.executions)

    if not snapshots_df.empty:
        latest_bid = snapshots_df['BestBid'].iloc[-1]
        latest_ask = snapshots_df['BestAsk'].iloc[-1]
        st.markdown(f"### Current Step: `{st.session_state.current_step}` | Best Bid = `{latest_bid}` | Best Ask = `{latest_ask}`")

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=snapshots_df['Step'], y=snapshots_df['BestBid'], name='Best Bid', line_color='green'))
            fig.add_trace(go.Scatter(x=snapshots_df['Step'], y=snapshots_df['BestAsk'], name='Best Ask', line_color='red'))
            fig.update_layout(title="Best Bid / Ask Evolution", xaxis_title="Step", yaxis_title="Price ($)")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if st.session_state.lob.bids and st.session_state.lob.asks:
                bid_prices = sorted(st.session_state.lob.bids.keys(), reverse=True)[:10]
                ask_prices = sorted(st.session_state.lob.asks.keys())[:10]

                bid_sizes = [sum(order[1] for order in st.session_state.lob.bids[p]) for p in bid_prices]
                ask_sizes = [sum(order[1] for order in st.session_state.lob.asks[p]) for p in ask_prices]

                bid_cum_sizes = pd.Series(bid_sizes).cumsum()
                ask_cum_sizes = pd.Series(ask_sizes).cumsum()

                depth_fig = go.Figure()
                depth_fig.add_trace(go.Scatter(x=bid_prices, y=bid_cum_sizes, mode='lines', name='Bid Depth', fill='tozeroy', line_color='green'))
                depth_fig.add_trace(go.Scatter(x=ask_prices, y=ask_cum_sizes, mode='lines', name='Ask Depth', fill='tozeroy', line_color='red'))
                depth_fig.update_layout(title="Live Order Book Depth Chart", xaxis_title="Price ($)", yaxis_title="Cumulative Size")
                st.plotly_chart(depth_fig, use_container_width=True)
            else:
                st.info("Waiting for enough liquidity on both sides to display Depth Chart.")

    # --- Execution Log ---
    st.subheader("Last 5 Executions")
    if executions_df.empty:
        st.info("No executions yet.")
    else:
        st.dataframe(executions_df.tail(5), use_container_width=True)