import json
import os
import time
from typing import Any


class EventLog:
    """
    Append-only Event Log for state reconstruction in Research OS v2.
    Supports event-sourcing design to prevent volatile in-memory loss.
    """

    def __init__(self, log_path: str = "data/event_log.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def append(self, event_type: str, payload: dict[str, Any]):
        """
        Append a new event to the JSONL log file.
        """
        event = {"timestamp": time.time(), "event_type": event_type, "payload": payload}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        """
        Read all events sequentially from the log file.
        """
        if not os.path.exists(self.log_path):
            return []

        events = []
        with open(self.log_path) as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
        return events


def rebuild_positions(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Rebuild the active positions dictionary by replaying events sequentially.
    Reconstructed structure: { symbol: { "qty": int, "avg_price": float } }
    """
    positions: dict[str, dict[str, Any]] = {}

    for event in events:
        payload = event.get("payload", {})
        event_type = event.get("event_type")

        if event_type == "TRADE_EXECUTION":
            symbol = payload["symbol"]
            side = payload["side"].upper()
            qty = int(payload["qty"])
            price = float(payload["price"])

            if symbol not in positions:
                positions[symbol] = {"qty": 0, "avg_price": 0.0}

            pos = positions[symbol]
            current_qty = pos["qty"]

            if side == "BUY":
                new_qty = current_qty + qty
                if new_qty > 0:
                    # Update average cost basis
                    total_cost = (current_qty * pos["avg_price"]) + (qty * price)
                    pos["avg_price"] = total_cost / new_qty
                else:
                    pos["avg_price"] = 0.0
                pos["qty"] = new_qty
            elif side == "SELL":
                new_qty = current_qty - qty
                if new_qty < 0:
                    # Short position average cost basis
                    total_cost = (abs(current_qty) * pos["avg_price"]) + (qty * price)
                    pos["avg_price"] = total_cost / abs(new_qty)
                elif new_qty == 0:
                    pos["avg_price"] = 0.0
                pos["qty"] = new_qty

            # If position is fully closed, remove it
            if positions[symbol]["qty"] == 0:
                positions.pop(symbol)

    return positions
