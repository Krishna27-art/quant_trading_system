"""
Venue Router

Selects optimal venue for order execution based on liquidity, cost, and constraints.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from utils.logger import get_logger

logger = get_logger("venue_router")


class VenueType(str, Enum):
    """Types of trading venues."""

    EXCHANGE = "exchange"
    DARK_POOL = "dark_pool"
    INTERNAL_CROSS = "internal_cross"
    OTC = "otc"


@dataclass
class Venue:
    """Trading venue."""

    venue_id: str
    venue_name: str
    venue_type: VenueType
    exchange: str  # NSE, BSE, etc.
    supports_options: bool = False
    supports_futures: bool = False
    supports_equity: bool = True
    min_order_size: int = 1
    max_order_size: int = 10000000
    avg_latency_ms: float = 10.0
    liquidity_score: float = 0.5  # 0-1, higher is better
    cost_bps: float = 0.0  # Cost in basis points
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "venue_id": self.venue_id,
            "venue_name": self.venue_name,
            "venue_type": self.venue_type.value,
            "exchange": self.exchange,
            "supports_options": self.supports_options,
            "supports_futures": self.supports_futures,
            "supports_equity": self.supports_equity,
            "min_order_size": self.min_order_size,
            "max_order_size": self.max_order_size,
            "avg_latency_ms": self.avg_latency_ms,
            "liquidity_score": self.liquidity_score,
            "cost_bps": self.cost_bps,
            "metadata": self.metadata,
        }


@dataclass
class VenueRoutingRule:
    """Venue routing rule."""

    rule_id: str
    condition: str  # e.g., "large_order", "options", "low_latency"
    target_venue: str
    priority: int = 1
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class VenueRouter:
    """
    Venue router for optimal venue selection.

    Selects venue based on:
    - Order size constraints
    - Instrument type (equity, options, futures)
    - Liquidity requirements
    - Cost considerations
    - Latency requirements
    """

    def __init__(self):
        """Initialize venue router."""
        self.logger = logger

        # Venue registry
        self._venues: dict[str, Venue] = {}

        # Routing rules
        self._routing_rules: list[VenueRoutingRule] = []

        # Register default exchanges
        self.register_venue(
            Venue(
                venue_id="NSE",
                venue_name="National Stock Exchange",
                venue_type=VenueType.EXCHANGE,
                exchange="NSE",
                supports_options=True,
                supports_futures=True,
                supports_equity=True,
            )
        )
        self.register_venue(
            Venue(
                venue_id="BSE",
                venue_name="Bombay Stock Exchange",
                venue_type=VenueType.EXCHANGE,
                exchange="BSE",
                supports_options=True,
                supports_futures=True,
                supports_equity=True,
            )
        )

        self.logger.info("VenueRouter initialized")

    def register_venue(self, venue: Venue):
        """
        Register a trading venue.

        Args:
            venue: Venue to register
        """
        self._venues[venue.venue_id] = venue
        self.logger.info(f"Registered venue: {venue.venue_id} ({venue.venue_name})")

    def get_venue(self, venue_id: str) -> Venue | None:
        """
        Get venue by ID.

        Args:
            venue_id: Venue identifier

        Returns:
            Venue or None
        """
        return self._venues.get(venue_id)

    def get_all_venues(self) -> dict[str, Venue]:
        """
        Get all registered venues.

        Returns:
            Dictionary of venues
        """
        return self._venues.copy()

    def get_venues_by_type(self, venue_type: VenueType) -> list[Venue]:
        """
        Get venues by type.

        Args:
            venue_type: Type of venue

        Returns:
            List of venues
        """
        return [v for v in self._venues.values() if v.venue_type == venue_type]

    def get_venues_by_exchange(self, exchange: str) -> list[Venue]:
        """
        Get venues by exchange.

        Args:
            exchange: Exchange name

        Returns:
            List of venues
        """
        return [v for v in self._venues.values() if v.exchange == exchange]

    def route_order(
        self,
        order_size: int,
        instrument_type: str = "equity",
        exchange: str | None = None,
        max_latency_ms: float | None = None,
        min_liquidity: float | None = None,
        preferred_venue: str | None = None,
    ) -> str | None:
        """
        Route order to optimal venue.

        Args:
            order_size: Order size
            instrument_type: Type of instrument (equity, options, futures)
            exchange: Preferred exchange (optional)
            max_latency_ms: Maximum acceptable latency (optional)
            min_liquidity: Minimum liquidity score (optional)
            preferred_venue: Preferred venue (optional)

        Returns:
            Selected venue ID or None
        """
        # Filter venues by instrument type
        if instrument_type == "options":
            eligible_venues = [v for v in self._venues.values() if v.supports_options]
        elif instrument_type == "futures":
            eligible_venues = [v for v in self._venues.values() if v.supports_futures]
        else:
            eligible_venues = [v for v in self._venues.values() if v.supports_equity]

        if not eligible_venues:
            self.logger.error(f"No venues support {instrument_type}")
            return None

        # Filter by exchange if specified
        if exchange:
            eligible_venues = [v for v in eligible_venues if v.exchange == exchange]
            if not eligible_venues:
                self.logger.warning(f"No venues on exchange {exchange}")

        # Filter by order size constraints
        eligible_venues = [
            v for v in eligible_venues if v.min_order_size <= order_size <= v.max_order_size
        ]
        if not eligible_venues:
            self.logger.error(f"No venues support order size {order_size}")
            return None

        # Filter by latency if specified
        if max_latency_ms:
            eligible_venues = [v for v in eligible_venues if v.avg_latency_ms <= max_latency_ms]
            if not eligible_venues:
                self.logger.warning(f"No venues meet latency requirement {max_latency_ms}ms")

        # Filter by liquidity if specified
        if min_liquidity:
            eligible_venues = [v for v in eligible_venues if v.liquidity_score >= min_liquidity]
            if not eligible_venues:
                self.logger.warning(f"No venues meet liquidity requirement {min_liquidity}")

        # Check routing rules
        for rule in sorted(self._routing_rules, key=lambda r: r.priority):
            if not rule.enabled:
                continue

            if self._evaluate_rule(rule, order_size, instrument_type):
                target_venue = self._venues.get(rule.target_venue)
                if target_venue and target_venue in eligible_venues:
                    self.logger.info(
                        f"Routed to {target_venue.venue_id} based on rule: {rule.condition}"
                    )
                    return target_venue.venue_id

        # Use preferred venue if eligible
        if preferred_venue:
            preferred = self._venues.get(preferred_venue)
            if preferred and preferred in eligible_venues:
                return preferred.venue_id

        # Select venue with best liquidity
        if eligible_venues:
            best_venue = max(eligible_venues, key=lambda v: v.liquidity_score)
            return best_venue.venue_id

        self.logger.error("No eligible venues for routing")
        return None

    def _evaluate_rule(self, rule: VenueRoutingRule, order_size: int, instrument_type: str) -> bool:
        """
        Evaluate a venue routing rule.

        Args:
            rule: Routing rule
            order_size: Order size
            instrument_type: Instrument type

        Returns:
            True if rule matches
        """
        condition = rule.condition

        # Large order rule
        if condition == "large_order":
            return order_size > 10000

        # Small order rule
        if condition == "small_order":
            return order_size < 100

        # Options rule
        if condition == "options":
            return instrument_type == "options"

        # Equity rule
        if condition == "equity":
            return instrument_type == "equity"

        # Low latency rule
        if condition == "low_latency":
            target_venue = self._venues.get(rule.target_venue)
            return target_venue and target_venue.avg_latency_ms < 5.0

        return False

    def add_routing_rule(self, rule: VenueRoutingRule):
        """
        Add a venue routing rule.

        Args:
            rule: Routing rule
        """
        self._routing_rules.append(rule)
        self.logger.info(f"Added venue routing rule: {rule.rule_id} -> {rule.target_venue}")

    def remove_routing_rule(self, rule_id: str):
        """
        Remove a venue routing rule.

        Args:
            rule_id: Rule ID
        """
        self._routing_rules = [r for r in self._routing_rules if r.rule_id != rule_id]
        self.logger.info(f"Removed venue routing rule: {rule_id}")

    def update_venue_liquidity(self, venue_id: str, liquidity_score: float):
        """
        Update venue liquidity score.

        Args:
            venue_id: Venue identifier
            liquidity_score: New liquidity score (0-1)
        """
        if venue_id in self._venues:
            self._venues[venue_id].liquidity_score = liquidity_score
            self.logger.info(f"Updated liquidity for {venue_id}: {liquidity_score}")

    def update_venue_latency(self, venue_id: str, latency_ms: float):
        """
        Update venue latency.

        Args:
            venue_id: Venue identifier
            latency_ms: New latency in milliseconds
        """
        if venue_id in self._venues:
            self._venues[venue_id].avg_latency_ms = latency_ms
            self.logger.info(f"Updated latency for {venue_id}: {latency_ms}ms")

    def get_status(self) -> dict[str, Any]:
        """
        Get router status.

        Returns:
            Status dictionary
        """
        return {
            "total_venues": len(self._venues),
            "exchange_venues": len(self.get_venues_by_exchange("NSE")),
            "dark_pools": len(self.get_venues_by_type(VenueType.DARK_POOL)),
            "routing_rules": len(self._routing_rules),
            "timestamp": datetime.utcnow().isoformat(),
        }
