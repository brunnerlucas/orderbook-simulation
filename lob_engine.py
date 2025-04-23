import time

class LimitOrderBook:
    def __init__(self):
        self.bids = {}              # Buy side orders: price -> list of [order_id, size]
        self.asks = {}              # Sell side orders: price -> list of [order_id, size]
        self.order_lookup = {}      # order_id -> (price, size, direction)
        self.snapshots = []         # Track best bid/ask over time
        self.executions = []        # Log of executions

    def process_message(self, msg):
        msg_type = msg['Type']
        order_id = msg['OrderID']
        price = msg['Price']
        size = msg['Size']
        direction = msg['Direction']

        if msg_type == 1:  # New Limit Order
            book = self.bids if direction == 1 else self.asks
            book.setdefault(price, []).append([order_id, size])
            self.order_lookup[order_id] = (price, size, direction)

        elif msg_type in [2, 3]:  # Cancel/Delete
            self._update_order(order_id, size)

        elif msg_type in [4, 5]:  # Execution (assume market order consumes liquidity)
            self._update_order(order_id, size)
            self.executions.append({
                'OrderID': order_id,
                'Price': price,
                'Size': size,
                'Direction': direction
            })

        elif msg_type == 7:
            pass  # For now, ignore trading halts

    def _update_order(self, order_id, size):
        if order_id in self.order_lookup:
            price, _, direction = self.order_lookup[order_id]
            book = self.bids if direction == 1 else self.asks
            for order in book.get(price, []):
                if order[0] == order_id:
                    order[1] -= size
                    if order[1] <= 0:
                        book[price].remove(order)
                    break

    def capture_snapshot(self, step):
        best_bid = max(self.bids.keys()) if self.bids else None
        best_ask = min(self.asks.keys()) if self.asks else None
        self.snapshots.append({
            'Step': step,
            'BestBid': best_bid,
            'BestAsk': best_ask
        })

    def run_simulation(self, messages, steps=100, delay=0.1, snapshot_interval=5):
        for idx, row in messages.iloc[:steps].iterrows():
            self.process_message(row)
            if idx % snapshot_interval == 0:
                self.capture_snapshot(idx)
            time.sleep(delay)