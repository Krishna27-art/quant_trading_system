"""
Smart Order Router

Intelligent order routing combining broker health, venue selection, and cost optimization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from utils.logger import get_logger

from .broker_router import BrokerHealth, BrokerRouter, BrokerStatus
from .venue_router import Venue, VenueRouter

logger = get_logger("smart_router")


class RoutingStrategy(str, Enum):
    """Routing strategies."""

    LOWEST_COST = "lowest_cost"
    FASTEST_EXECUTION = "fastest_execution"
    BEST_LIQUIDITY = "best_liquidity"
    BALANCED = "balanced"


@dataclass
class CostModel:
    """Cost model for routing decisions."""

    brokerage_bps: float = 0.01  # Brokerage in basis points
    slippage_bps: float = 0.05  # Expected slippage in basis points
    impact_bps: float = 0.02  # Market impact in basis points
    fixed_cost: float = 0.0  # Fixed cost per order

    def calculate_total_cost(self, order_value: float) -> float:
        """
        Calculate total cost for an order.

        Args:
            order_value: Order value in currency

        Returns:
            Total cost
        """
        variable_cost = (
            order_value * (self.brokerage_bps + self.slippage_bps + self.impact_bps) / 10000
        )
        return variable_cost + self.fixed_cost

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "brokerage_bps": self.brokerage_bps,
            "slippage_bps": self.slippage_bps,
            "impact_bps": self.impact_bps,
            "fixed_cost": self.fixed_cost,
        }


@dataclass
class RoutingDecision:
    """Routing decision."""

    order_id: str
    selected_broker: str
    selected_venue: str
    strategy: RoutingStrategy
    estimated_cost: float
    estimated_latency_ms: float
    confidence: float  # 0-1, higher is better
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "selected_broker": self.selected_broker,
            "selected_venue": self.selected_venue,
            "strategy": self.strategy.value,
            "estimated_cost": self.estimated_cost,
            "estimated_latency_ms": self.estimated_latency_ms,
            "confidence": self.confidence,
            "alternatives": self.alternatives,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class RoutingRequest:
    """Routing request."""

    order_id: str
    symbol: str
    side: str  # buy/sell
    quantity: int
    order_type: str  # equity, options, futures
    price: float | None = None
    strategy: RoutingStrategy = RoutingStrategy.BALANCED
    preferred_broker: str | None = None
    preferred_venue: str | None = None
    max_latency_ms: float | None = None
    min_liquidity: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "price": self.price,
            "strategy": self.strategy.value,
            "preferred_broker": self.preferred_broker,
            "preferred_venue": self.preferred_venue,
            "max_latency_ms": self.max_latency_ms,
            "min_liquidity": self.min_liquidity,
            "metadata": self.metadata,
        }


class SmartOrderRouter:
    """
    Smart order router combining broker and venue selection.

    Routes orders based on:
    - Broker health (latency, rejections, disconnects)
    - Venue selection (liquidity, constraints)
    - Cost model (brokerage, slippage, impact)
    - Routing strategy (cost, speed, liquidity, balanced)
    """

    def __init__(self):
        """Initialize smart order router."""
        self.logger = logger

        # Sub-routers
        self.broker_router = BrokerRouter()
        self.venue_router = VenueRouter()

        # Cost models per broker
        self._cost_models: dict[str, CostModel] = {}

        # Default cost model
        self._default_cost_model = CostModel()

        # Routing statistics
        self._routing_stats: dict[str, Any] = {
            "total_routed": 0,
            "by_broker": {},
            "by_venue": {},
            "by_strategy": {},
        }

        self.logger.info("SmartOrderRouter initialized")

    def register_broker(
        self, broker_id: str, broker_type: str, cost_model: CostModel | None = None
    ):
        """
        Register a broker with optional cost model.

        Args:
            broker_id: Broker identifier
            broker_type: Type of broker
            cost_model: Cost model for this broker
        """
        self.broker_router.register_broker(broker_id, broker_type)

        if cost_model:
            self._cost_models[broker_id] = cost_model
        else:
            self._cost_models[broker_id] = self._default_cost_model

        self.logger.info(f"Registered broker with cost model: {broker_id}")

    def register_venue(self, venue: Venue):
        """
        Register a venue.

        Args:
            venue: Venue to register
        """
        self.venue_router.register_venue(venue)

    def route_order(self, request: RoutingRequest) -> RoutingDecision:
        """
        Route an order using smart routing logic.

        Args:
            request: Routing request

        Returns:
            Routing decision
        """
        try:
            # Select broker
            selected_broker = self.broker_router.route_order(
                order_type=request.order_type,
                symbol=request.symbol,
                preferred_broker=request.preferred_broker,
            )

            if not selected_broker:
                return self._create_failed_decision(request, "No available broker")

            # Select venue
            order_value = request.quantity * request.price if request.price else 0
            selected_venue = self.venue_router.route_order(
                order_size=request.quantity,
                instrument_type=request.order_type,
                max_latency_ms=request.max_latency_ms,
                min_liquidity=request.min_liquidity,
                preferred_venue=request.preferred_venue,
            )

            if not selected_venue:
                return self._create_failed_decision(request, "No available venue")

            # Calculate estimated cost
            cost_model = self._cost_models.get(selected_broker, self._default_cost_model)
            estimated_cost = cost_model.calculate_total_cost(order_value)

            # Calculate estimated latency
            broker_health = self.broker_router.get_broker_health(selected_broker)
            venue = self.venue_router.get_venue(selected_venue)
            estimated_latency = (broker_health.latency_ms if broker_health else 0) + (
                venue.avg_latency_ms if venue else 0
            )

            # Calculate confidence based on health and liquidity
            confidence = self._calculate_confidence(broker_health, venue)

            # Generate alternatives
            alternatives = self._generate_alternatives(request, selected_broker, selected_venue)

            # Create decision
            decision = RoutingDecision(
                order_id=request.order_id,
                selected_broker=selected_broker,
                selected_venue=selected_venue,
                strategy=request.strategy,
                estimated_cost=estimated_cost,
                estimated_latency_ms=estimated_latency,
                confidence=confidence,
                alternatives=alternatives,
            )

            # Update statistics
            self._update_stats(selected_broker, selected_venue, request.strategy)

            self.logger.info(
                f"Routed order {request.order_id}: broker={selected_broker}, "
                f"venue={selected_venue}, cost={estimated_cost:.2f}, latency={estimated_latency:.2f}ms"
            )

            return decision

        except Exception as e:
            self.logger.error(f"Failed to route order {request.order_id}: {e}")
            return self._create_failed_decision(request, str(e))

    def _calculate_confidence(
        self, broker_health: BrokerHealth | None, venue: Venue | None
    ) -> float:
        """
        Calculate routing confidence.

        Args:
            broker_health: Broker health
            venue: Venue

        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5

        # Broker health contribution
        if broker_health:
            if broker_health.status == BrokerStatus.HEALTHY:
                confidence += 0.3
            elif broker_health.status == BrokerStatus.DEGRADED:
                confidence += 0.1
            else:
                confidence -= 0.2

        # Venue liquidity contribution
        if venue:
            confidence += venue.liquidity_score * 0.2

        return max(0.0, min(1.0, confidence))

    def _generate_alternatives(
        self, request: RoutingRequest, selected_broker: str, selected_venue: str
    ) -> list[dict[str, Any]]:
        """
        Generate alternative routing options.

        Args:
            request: Routing request
            selected_broker: Selected broker
            selected_venue: Selected venue

        Returns:
            List of alternatives
        """
        alternatives = []

        # Alternative brokers
        healthy_brokers = self.broker_router.get_healthy_brokers()
        for broker_id in healthy_brokers[:3]:  # Top 3 alternatives
            if broker_id != selected_broker:
                cost_model = self._cost_models.get(broker_id, self._default_cost_model)
                order_value = request.quantity * request.price if request.price else 0
                cost = cost_model.calculate_total_cost(order_value)

                alternatives.append(
                    {
                        "broker": broker_id,
                        "venue": selected_venue,
                        "estimated_cost": cost,
                        "type": "broker_alternative",
                    }
                )

        # Alternative venues
        all_venues = self.venue_router.get_all_venues()
        for venue_id in list(all_venues.keys())[:3]:  # Top 3 alternatives
            if venue_id != selected_venue:
                venue = all_venues[venue_id]
                alternatives.append(
                    {
                        "broker": selected_broker,
                        "venue": venue_id,
                        "liquidity_score": venue.liquidity_score,
                        "latency_ms": venue.avg_latency_ms,
                        "type": "venue_alternative",
                    }
                )

        return alternatives

    def _create_failed_decision(self, request: RoutingRequest, reason: str) -> RoutingDecision:
        """
        Create a failed routing decision.

        Args:
            request: Routing request
            reason: Failure reason

        Returns:
            Failed routing decision
        """
        return RoutingDecision(
            order_id=request.order_id,
            selected_broker="",
            selected_venue="",
            strategy=request.strategy,
            estimated_cost=0.0,
            estimated_latency_ms=0.0,
            confidence=0.0,
            metadata={"error": reason},
        )

    def _update_stats(self, broker_id: str, venue_id: str, strategy: RoutingStrategy):
        """
        Update routing statistics.

        Args:
            broker_id: Selected broker
            venue_id: Selected venue
            strategy: Routing strategy
        """
        self._routing_stats["total_routed"] += 1

        if broker_id not in self._routing_stats["by_broker"]:
            self._routing_stats["by_broker"][broker_id] = 0
        self._routing_stats["by_broker"][broker_id] += 1

        if venue_id not in self._routing_stats["by_venue"]:
            self._routing_stats["by_venue"][venue_id] = 0
        self._routing_stats["by_venue"][venue_id] += 1

        strategy_key = strategy.value
        if strategy_key not in self._routing_stats["by_strategy"]:
            self._routing_stats["by_strategy"][strategy_key] = 0
        self._routing_stats["by_strategy"][strategy_key] += 1

    def update_broker_cost_model(self, broker_id: str, cost_model: CostModel):
        """
        Update cost model for a broker.

        Args:
            broker_id: Broker identifier
            cost_model: New cost model
        """
        self._cost_models[broker_id] = cost_model
        self.logger.info(f"Updated cost model for {broker_id}")

    def get_routing_stats(self) -> dict[str, Any]:
        """
        Get routing statistics.

        Returns:
            Statistics dictionary
        """
        return self._routing_stats.copy()

    def get_status(self) -> dict[str, Any]:
        """
        Get router status.

        Returns:
            Status dictionary
        """
        return {
            "broker_router_status": self.broker_router.get_status(),
            "venue_router_status": self.venue_router.get_status(),
            "cost_models": len(self._cost_models),
            "routing_stats": self._routing_stats,
            "timestamp": datetime.utcnow().isoformat(),
        }
