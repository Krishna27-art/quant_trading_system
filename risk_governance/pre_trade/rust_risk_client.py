import json
import os
import socket
import time

from utils.logger import get_logger

logger = get_logger("rust_risk_client")


class RustRiskClient:
    """
    Python client for the Rust Out-of-Process Pre-Trade Risk Gateway.
    Uses Unix Domain Sockets for low-latency checks and checks heartbeat files.
    """

    def __init__(
        self,
        socket_path: str = "data/risk_gateway.sock",
        heartbeat_path: str = "data/risk_heartbeat.bin",
    ):
        self.socket_path = socket_path
        self.heartbeat_path = heartbeat_path
        self.logger = logger

    def is_gateway_alive(self) -> bool:
        """Check if the Rust Risk Gateway daemon heartbeat is active."""
        if not os.path.exists(self.heartbeat_path):
            self.logger.warning("Heartbeat file not found")
            return False

        try:
            with open(self.heartbeat_path, "rb") as f:
                data = f.read(8)
                if len(data) < 8:
                    return False
                import struct

                heartbeat_ts = struct.unpack("d", data)[0]

            current_time = time.time()
            if current_time - heartbeat_ts > 5.0:
                self.logger.error(
                    f"Heartbeat timeout: current={current_time}, last={heartbeat_ts}, diff={current_time - heartbeat_ts:.2f}s"
                )
                return False
            return True
        except Exception as e:
            self.logger.error(f"Failed to read heartbeat: {e}")
            return False

    def check_order_risk(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float | None,
        portfolio_value: float,
        current_exposure: float,
        symbol_exposure: float,
    ) -> tuple[bool, str | None]:
        """
        Verify if order passes pre-trade risk checks with Rust Gateway.

        Returns:
            (passed, rejection_reason)
        """
        # Failsafe: block trading if the gateway is dead
        if not self.is_gateway_alive():
            return False, "Rust Risk Gateway heartbeat is inactive (Gateway offline)"

        if not os.path.exists(self.socket_path):
            return False, f"Risk Gateway socket path {self.socket_path} does not exist"

        try:
            # Connect to Unix domain socket
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(1.0)  # 1 second timeout limit for risk checks
            client.connect(self.socket_path)

            # Send JSON request
            request = {
                "type": "CheckRisk",
                "symbol": symbol,
                "side": side.lower(),
                "quantity": int(quantity),
                "price": price,
                "portfolio_value": float(portfolio_value),
                "current_exposure": float(current_exposure),
                "symbol_exposure": float(symbol_exposure),
            }

            client.sendall(json.dumps(request).encode())

            # Read response
            response_data = client.recv(1024)
            client.close()

            response = json.loads(response_data.decode())
            if response.get("type") == "RiskResult":
                return response.get("passed", False), response.get("rejection_reason")
            elif response.get("type") == "Error":
                return False, f"Gateway error: {response.get('message')}"
            else:
                return False, f"Unexpected response from gateway: {response}"
        except Exception as e:
            self.logger.error(f"Risk Gateway connection failed: {e}")
            return False, f"Risk Gateway IPC failure: {str(e)}"
