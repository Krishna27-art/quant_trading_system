from abc import ABC, abstractmethod
from typing import Any

from portfolio_execution.oms import ManagedOrder


class BaseExecutionAdapter(ABC):
    IS_LIVE: bool = False  # Subclasses override

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the broker."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the broker."""
        pass

    @abstractmethod
    async def heartbeat(self) -> bool:
        """Check connection status."""
        pass

    @abstractmethod
    async def get_quote(self, symbols: list[str]) -> dict[str, float]:
        """Fetch current prices for symbols."""
        pass

    @abstractmethod
    async def get_margin(self) -> dict[str, float]:
        """Fetch margin/collateral availability."""
        pass

    @abstractmethod
    async def get_positions(self) -> list[dict[str, Any]]:
        """Fetch active positions."""
        pass

    @abstractmethod
    async def place_order(self, order: ManagedOrder) -> str:
        """Submit order to broker and return broker_order_id."""
        pass

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order."""
        pass

    @abstractmethod
    async def get_order_status(self, broker_order_id: str) -> dict[str, Any]:
        """Fetch current status of an order."""
        pass

    @abstractmethod
    async def exit_all_positions(self) -> list[str]:
        """Emergency liquidator. Returns list of exit order IDs."""
        pass

    @abstractmethod
    async def cancel_all_pending_orders(self) -> list[str]:
        """Emergency order cancellation. Returns list of cancelled order IDs."""
        pass
