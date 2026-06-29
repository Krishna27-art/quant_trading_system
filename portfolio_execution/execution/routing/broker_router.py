"""
Broker Router

Monitors broker health and routes orders based on broker status.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from utils.logger import get_logger

logger = get_logger("broker_router")


class BrokerStatus(str, Enum):
    """Broker status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DOWN = "down"


@dataclass
class BrokerHealth:
    """Broker health metrics."""

    broker_id: str
    broker_type: str  # zerodha, angel, fyers, iifl, fenix
    status: BrokerStatus
    latency_ms: float = 0.0
    rejection_rate: float = 0.0
    disconnect_count: int = 0
    last_disconnect: datetime | None = None
    margin_available: float = 0.0
    margin_utilization: float = 0.0
    last_update: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "broker_id": self.broker_id,
            "broker_type": self.broker_type,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "rejection_rate": self.rejection_rate,
            "disconnect_count": self.disconnect_count,
            "last_disconnect": self.last_disconnect.isoformat() if self.last_disconnect else None,
            "margin_available": self.margin_available,
            "margin_utilization": self.margin_utilization,
            "last_update": self.last_update.isoformat(),
        }


@dataclass
class BrokerRoutingRule:
    """Broker routing rule."""

    rule_id: str
    condition: str  # e.g., "margin_low", "broker_down", "options"
    target_broker: str
    priority: int = 1
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class BrokerRouter:
    """
    Broker router for intelligent broker selection.

    Monitors broker health and routes orders based on:
    - Latency
    - Rejections
    - Disconnects
    - Margin availability
    """

    def __init__(self):
        """Initialize broker router."""
        self.logger = logger

        # Broker health tracking
        self._broker_health: dict[str, BrokerHealth] = {}

        # Routing rules
        self._routing_rules: list[BrokerRoutingRule] = []

        # Latency history (for moving average)
        self._latency_history: dict[str, deque] = {}
        self._latency_window = 100  # Keep last 100 measurements

        # Rejection tracking
        self._rejection_counts: dict[str, int] = {}
        self._total_orders: dict[str, int] = {}

        self.logger.info("BrokerRouter initialized")

    def register_broker(self, broker_id: str, broker_type: str):
        """
        Register a broker for health monitoring.

        Args:
            broker_id: Broker identifier
            broker_type: Type of broker (zerodha, angel, fyers, iifl, fenix)
        """
        self._broker_health[broker_id] = BrokerHealth(
            broker_id=broker_id, broker_type=broker_type, status=BrokerStatus.HEALTHY
        )
        self._latency_history[broker_id] = deque(maxlen=self._latency_window)
        self._rejection_counts[broker_id] = 0
        self._total_orders[broker_id] = 0

        self.logger.info(f"Registered broker: {broker_id} ({broker_type})")

    def update_latency(self, broker_id: str, latency_ms: float):
        """
        Update broker latency.

        Args:
            broker_id: Broker identifier
            latency_ms: Latency in milliseconds
        """
        if broker_id not in self._broker_health:
            self.logger.warning(f"Broker {broker_id} not registered")
            return

        # Add to history
        self._latency_history[broker_id].append(latency_ms)

        # Calculate moving average
        avg_latency = sum(self._latency_history[broker_id]) / len(self._latency_history[broker_id])

        # Update health
        health = self._broker_health[broker_id]
        health.latency_ms = avg_latency
        health.last_update = datetime.utcnow()

        # Update status based on latency
        if avg_latency > 1000:  # > 1 second
            health.status = BrokerStatus.UNHEALTHY
        elif avg_latency > 500:  # > 500ms
            health.status = BrokerStatus.DEGRADED
        else:
            health.status = BrokerStatus.HEALTHY

    def record_rejection(self, broker_id: str):
        """
        Record a rejection for a broker.

        Args:
            broker_id: Broker identifier
        """
        if broker_id not in self._broker_health:
            return

        self._rejection_counts[broker_id] += 1
        self._total_orders[broker_id] += 1

        # Update rejection rate
        health = self._broker_health[broker_id]
        health.rejection_rate = self._rejection_counts[broker_id] / self._total_orders[broker_id]
        health.last_update = datetime.utcnow()

        # Update status based on rejection rate
        if health.rejection_rate > 0.1:  # > 10% rejection rate
            health.status = BrokerStatus.UNHEALTHY
        elif health.rejection_rate > 0.05:  # > 5% rejection rate
            health.status = BrokerStatus.DEGRADED

    def record_success(self, broker_id: str):
        """
        Record a successful order for a broker.

        Args:
            broker_id: Broker identifier
        """
        if broker_id not in self._broker_health:
            return

        self._total_orders[broker_id] += 1

        # Update rejection rate
        health = self._broker_health[broker_id]
        health.rejection_rate = self._rejection_counts[broker_id] / self._total_orders[broker_id]
        health.last_update = datetime.utcnow()

    def record_disconnect(self, broker_id: str):
        """
        Record a disconnect for a broker.

        Args:
            broker_id: Broker identifier
        """
        if broker_id not in self._broker_health:
            return

        health = self._broker_health[broker_id]
        health.disconnect_count += 1
        health.last_disconnect = datetime.utcnow()
        health.status = BrokerStatus.DOWN
        health.last_update = datetime.utcnow()

        self.logger.warning(f"Broker {broker_id} disconnected")

    def record_reconnect(self, broker_id: str):
        """
        Record a reconnect for a broker.

        Args:
            broker_id: Broker identifier
        """
        if broker_id not in self._broker_health:
            return

        health = self._broker_health[broker_id]
        health.status = BrokerStatus.HEALTHY
        health.last_update = datetime.utcnow()

        self.logger.info(f"Broker {broker_id} reconnected")

    def update_margin(self, broker_id: str, margin_available: float, margin_utilization: float):
        """
        Update broker margin information.

        Args:
            broker_id: Broker identifier
            margin_available: Available margin
            margin_utilization: Margin utilization percentage (0-100)
        """
        if broker_id not in self._broker_health:
            return

        health = self._broker_health[broker_id]
        health.margin_available = margin_available
        health.margin_utilization = margin_utilization
        health.last_update = datetime.utcnow()

        # Update status based on margin
        if margin_utilization > 95:  # > 95% margin utilization
            health.status = BrokerStatus.UNHEALTHY
        elif margin_utilization > 90:  # > 90% margin utilization
            health.status = BrokerStatus.DEGRADED

    def get_broker_health(self, broker_id: str) -> BrokerHealth | None:
        """
        Get health status for a broker.

        Args:
            broker_id: Broker identifier

        Returns:
            Broker health or None
        """
        return self._broker_health.get(broker_id)

    def get_all_health(self) -> dict[str, BrokerHealth]:
        """
        Get health status for all brokers.

        Returns:
            Dictionary of broker health
        """
        return self._broker_health.copy()

    def get_healthy_brokers(self) -> list[str]:
        """
        Get list of healthy brokers.

        Returns:
            List of healthy broker IDs
        """
        return [
            broker_id
            for broker_id, health in self._broker_health.items()
            if health.status == BrokerStatus.HEALTHY
        ]

    def route_order(
        self,
        order_type: str = "equity",
        symbol: str | None = None,
        preferred_broker: str | None = None,
    ) -> str | None:
        """
        Route order to appropriate broker.

        Args:
            order_type: Type of order (equity, options, futures)
            symbol: Trading symbol (optional)
            preferred_broker: Preferred broker (optional)

        Returns:
            Selected broker ID or None
        """
        # Check routing rules first
        for rule in sorted(self._routing_rules, key=lambda r: r.priority):
            if not rule.enabled:
                continue

            if self._evaluate_rule(rule, order_type, symbol):
                target_health = self._broker_health.get(rule.target_broker)
                if target_health and target_health.status != BrokerStatus.DOWN:
                    self.logger.info(
                        f"Routed to {rule.target_broker} based on rule: {rule.condition}"
                    )
                    return rule.target_broker

        # If preferred broker is healthy, use it
        if preferred_broker:
            health = self._broker_health.get(preferred_broker)
            if health and health.status == BrokerStatus.HEALTHY:
                return preferred_broker

        # Route to healthiest broker
        healthy_brokers = self.get_healthy_brokers()
        if healthy_brokers:
            # Select based on lowest latency
            best_broker = min(healthy_brokers, key=lambda b: self._broker_health[b].latency_ms)
            return best_broker

        # Fallback to any non-down broker
        available_brokers = [
            b for b, h in self._broker_health.items() if h.status != BrokerStatus.DOWN
        ]
        if available_brokers:
            return available_brokers[0]

        self.logger.error("No available brokers for routing")
        return None

    def _evaluate_rule(self, rule: BrokerRoutingRule, order_type: str, symbol: str | None) -> bool:
        """
        Evaluate a routing rule.

        Args:
            rule: Routing rule
            order_type: Order type
            symbol: Trading symbol

        Returns:
            True if rule matches
        """
        condition = rule.condition

        # Margin low rule
        if condition == "margin_low":
            target_health = self._broker_health.get(rule.target_broker)
            return target_health and target_health.margin_utilization < 50

        # Broker down rule
        if condition == "broker_down":
            target_health = self._broker_health.get(rule.target_broker)
            return target_health and target_health.status == BrokerStatus.DOWN

        # Options rule
        if condition == "options":
            return order_type == "options"

        # Equity rule
        if condition == "equity":
            return order_type == "equity"

        return False

    def add_routing_rule(self, rule: BrokerRoutingRule):
        """
        Add a routing rule.

        Args:
            rule: Routing rule
        """
        self._routing_rules.append(rule)
        self.logger.info(f"Added routing rule: {rule.rule_id} -> {rule.target_broker}")

    def remove_routing_rule(self, rule_id: str):
        """
        Remove a routing rule.

        Args:
            rule_id: Rule ID
        """
        self._routing_rules = [r for r in self._routing_rules if r.rule_id != rule_id]
        self.logger.info(f"Removed routing rule: {rule_id}")

    def get_status(self) -> dict[str, Any]:
        """
        Get router status.

        Returns:
            Status dictionary
        """
        healthy_count = sum(
            1 for h in self._broker_health.values() if h.status == BrokerStatus.HEALTHY
        )
        degraded_count = sum(
            1 for h in self._broker_health.values() if h.status == BrokerStatus.DEGRADED
        )
        unhealthy_count = sum(
            1 for h in self._broker_health.values() if h.status == BrokerStatus.UNHEALTHY
        )
        down_count = sum(1 for h in self._broker_health.values() if h.status == BrokerStatus.DOWN)

        return {
            "total_brokers": len(self._broker_health),
            "healthy": healthy_count,
            "degraded": degraded_count,
            "unhealthy": unhealthy_count,
            "down": down_count,
            "routing_rules": len(self._routing_rules),
            "timestamp": datetime.utcnow().isoformat(),
        }
