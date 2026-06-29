"""
Portfolio-Level Circuit Breaker Manager

Monitors portfolio P&L against configurable drawdown thresholds and triggers
HALT / REDUCE / CONTINUE signals for the orchestrator. Covers:
  - Daily loss limit (default -2% of NAV)
  - Weekly loss limit (default -5% of NAV)
  - Per-position loss limit
  - Per-sector concentration kill switch
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from utils.logger import get_logger

logger = get_logger("circuit_breaker_manager")


class CircuitAction(Enum):
    """Orchestrator actions returned by circuit breaker checks."""

    HALT = "HALT"
    REDUCE = "REDUCE"
    CONTINUE = "CONTINUE"


@dataclass
class CircuitBreakerConfig:
    """All thresholds for the circuit breaker manager."""

    daily_loss_limit_pct: float = -0.02  # -2% of NAV
    daily_reduce_threshold_pct: float = -0.015  # -1.5% → start reducing
    weekly_loss_limit_pct: float = -0.05  # -5% of NAV
    weekly_reduce_threshold_pct: float = -0.035  # -3.5% → start reducing
    per_position_loss_limit_pct: float = -0.10  # -10% per position
    per_position_reduce_pct: float = -0.07  # -7% → start reducing
    sector_concentration_limit_pct: float = 0.30  # 30% max in any sector
    sector_reduce_threshold_pct: float = 0.25  # 25% → start reducing
    cooldown_minutes: int = 30  # Re-check cooldown after HALT


@dataclass
class CircuitBreakerState:
    """Runtime state tracked by the manager."""

    is_halted: bool = False
    halt_reason: str = ""
    halt_timestamp: datetime | None = None
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    position_pnl: dict[str, float] = field(default_factory=dict)
    sector_weights: dict[str, float] = field(default_factory=dict)
    breach_history: list[dict] = field(default_factory=list)


@dataclass
class CircuitBreakerResult:
    """Result returned to the orchestrator."""

    action: CircuitAction
    reason: str
    breaches: list[str]
    reduce_factor: float = 1.0  # 1.0 = full size, 0.5 = half, 0.0 = halt


class CircuitBreakerManager:
    """
    Portfolio-level circuit breaker that integrates with the trading
    orchestrator to signal HALT / REDUCE / CONTINUE.
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState()
        self._nav_start_of_day: float = 0.0
        self._nav_start_of_week: float = 0.0
        self._position_entry_values: dict[str, float] = {}
        logger.info(
            f"CircuitBreakerManager initialised | daily_limit={self.config.daily_loss_limit_pct * 100:.2f}% weekly_limit={self.config.weekly_loss_limit_pct * 100:.2f}%"
        )

    # ------------------------------------------------------------------
    # NAV anchoring
    # ------------------------------------------------------------------
    def set_start_of_day_nav(self, nav: float) -> None:
        """Call at market open to anchor daily P&L measurement."""
        if nav <= 0:
            raise ValueError(f"NAV must be positive, got {nav}")
        self._nav_start_of_day = nav
        logger.info(f"SOD NAV set to ₹{nav:.2f}")

    def set_start_of_week_nav(self, nav: float) -> None:
        """Call on Monday open to anchor weekly P&L measurement."""
        if nav <= 0:
            raise ValueError(f"NAV must be positive, got {nav}")
        self._nav_start_of_week = nav
        logger.info(f"SOW NAV set to ₹{nav:.2f}")

    def register_position_entry(self, symbol: str, entry_value: float) -> None:
        """Register a position's entry value for per-position loss tracking."""
        self._position_entry_values[symbol] = entry_value
        logger.debug(f"Registered entry for {symbol} @ ₹{entry_value:.2f}")

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        current_nav: float,
        position_values: dict[str, float],
        sector_map: dict[str, str],
    ) -> CircuitBreakerResult:
        """
        Run all circuit breaker checks and return an action for the orchestrator.

        Args:
            current_nav: Current portfolio NAV.
            position_values: {symbol: current_market_value}.
            sector_map: {symbol: sector_name}.

        Returns:
            CircuitBreakerResult with action and metadata.
        """
        # Check cooldown – if halted recently, respect cooldown window
        if self.state.is_halted and self.state.halt_timestamp:
            elapsed = (datetime.now() - self.state.halt_timestamp).total_seconds()
            if elapsed < self.config.cooldown_minutes * 60:
                return CircuitBreakerResult(
                    action=CircuitAction.HALT,
                    reason=f"Still in cooldown ({int(elapsed)}s / {self.config.cooldown_minutes * 60}s)",
                    breaches=[self.state.halt_reason],
                    reduce_factor=0.0,
                )

        breaches: list[str] = []
        reduce_factors: list[float] = []

        # --- 1. Daily loss check ---
        daily_action, daily_msg, daily_rf = self._check_daily_loss(current_nav)
        if daily_action != CircuitAction.CONTINUE:
            breaches.append(daily_msg)
            reduce_factors.append(daily_rf)

        # --- 2. Weekly loss check ---
        weekly_action, weekly_msg, weekly_rf = self._check_weekly_loss(current_nav)
        if weekly_action != CircuitAction.CONTINUE:
            breaches.append(weekly_msg)
            reduce_factors.append(weekly_rf)

        # --- 3. Per-position loss check ---
        pos_action, pos_msgs, pos_rf = self._check_position_losses(position_values)
        if pos_action != CircuitAction.CONTINUE:
            breaches.extend(pos_msgs)
            reduce_factors.append(pos_rf)

        # --- 4. Sector concentration check ---
        sec_action, sec_msgs, sec_rf = self._check_sector_concentration(
            position_values, sector_map, current_nav
        )
        if sec_action != CircuitAction.CONTINUE:
            breaches.extend(sec_msgs)
            reduce_factors.append(sec_rf)

        # --- Aggregate decision ---
        if not breaches:
            self._clear_halt()
            return CircuitBreakerResult(
                action=CircuitAction.CONTINUE,
                reason="All checks passed",
                breaches=[],
                reduce_factor=1.0,
            )

        # Any HALT-level breach dominates
        any_halt = any(
            a == CircuitAction.HALT for a in [daily_action, weekly_action, pos_action, sec_action]
        )

        if any_halt:
            combined_reason = " | ".join(breaches)
            self._set_halt(combined_reason)
            self._record_breach(combined_reason)
            return CircuitBreakerResult(
                action=CircuitAction.HALT,
                reason=combined_reason,
                breaches=breaches,
                reduce_factor=0.0,
            )

        # REDUCE: take the most conservative factor
        min_rf = min(reduce_factors) if reduce_factors else 1.0
        combined_reason = " | ".join(breaches)
        logger.warning(f"REDUCE signal: factor={min_rf:.2f} | {combined_reason}")
        return CircuitBreakerResult(
            action=CircuitAction.REDUCE,
            reason=combined_reason,
            breaches=breaches,
            reduce_factor=min_rf,
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------
    def _check_daily_loss(self, current_nav: float) -> tuple[CircuitAction, str, float]:
        if self._nav_start_of_day <= 0:
            return CircuitAction.CONTINUE, "", 1.0

        daily_ret = (current_nav - self._nav_start_of_day) / self._nav_start_of_day
        self.state.daily_pnl = daily_ret

        if daily_ret <= self.config.daily_loss_limit_pct:
            msg = (
                f"DAILY LOSS HALT: {daily_ret:.2%} breaches "
                f"limit {self.config.daily_loss_limit_pct:.2%}"
            )
            logger.critical(msg)
            return CircuitAction.HALT, msg, 0.0

        if daily_ret <= self.config.daily_reduce_threshold_pct:
            # Linear interpolation for reduce factor
            span = self.config.daily_loss_limit_pct - self.config.daily_reduce_threshold_pct
            if abs(span) < 1e-9:
                rf = 0.5
            else:
                rf = max(0.1, (daily_ret - self.config.daily_loss_limit_pct) / (-span))
            msg = (
                f"DAILY LOSS REDUCE: {daily_ret:.2%} nearing limit "
                f"{self.config.daily_loss_limit_pct:.2%}"
            )
            logger.warning(msg)
            return CircuitAction.REDUCE, msg, rf

        return CircuitAction.CONTINUE, "", 1.0

    def _check_weekly_loss(self, current_nav: float) -> tuple[CircuitAction, str, float]:
        if self._nav_start_of_week <= 0:
            return CircuitAction.CONTINUE, "", 1.0

        weekly_ret = (current_nav - self._nav_start_of_week) / self._nav_start_of_week
        self.state.weekly_pnl = weekly_ret

        if weekly_ret <= self.config.weekly_loss_limit_pct:
            msg = (
                f"WEEKLY LOSS HALT: {weekly_ret:.2%} breaches "
                f"limit {self.config.weekly_loss_limit_pct:.2%}"
            )
            logger.critical(msg)
            return CircuitAction.HALT, msg, 0.0

        if weekly_ret <= self.config.weekly_reduce_threshold_pct:
            span = self.config.weekly_loss_limit_pct - self.config.weekly_reduce_threshold_pct
            if abs(span) < 1e-9:
                rf = 0.5
            else:
                rf = max(0.1, (weekly_ret - self.config.weekly_loss_limit_pct) / (-span))
            msg = (
                f"WEEKLY LOSS REDUCE: {weekly_ret:.2%} nearing limit "
                f"{self.config.weekly_loss_limit_pct:.2%}"
            )
            logger.warning(msg)
            return CircuitAction.REDUCE, msg, rf

        return CircuitAction.CONTINUE, "", 1.0

    def _check_position_losses(
        self, position_values: dict[str, float]
    ) -> tuple[CircuitAction, list[str], float]:
        msgs: list[str] = []
        worst_rf = 1.0
        worst_action = CircuitAction.CONTINUE

        for symbol, current_val in position_values.items():
            entry_val = self._position_entry_values.get(symbol)
            if entry_val is None or entry_val == 0:
                continue

            pnl_pct = (current_val - entry_val) / abs(entry_val)
            self.state.position_pnl[symbol] = pnl_pct

            if pnl_pct <= self.config.per_position_loss_limit_pct:
                msg = (
                    f"POSITION HALT {symbol}: {pnl_pct:.2%} breaches "
                    f"limit {self.config.per_position_loss_limit_pct:.2%}"
                )
                logger.critical(msg)
                msgs.append(msg)
                worst_action = CircuitAction.HALT
                worst_rf = 0.0

            elif pnl_pct <= self.config.per_position_reduce_pct:
                span = self.config.per_position_loss_limit_pct - self.config.per_position_reduce_pct
                if abs(span) < 1e-9:
                    rf = 0.5
                else:
                    rf = max(0.1, (pnl_pct - self.config.per_position_loss_limit_pct) / (-span))
                msg = (
                    f"POSITION REDUCE {symbol}: {pnl_pct:.2%} nearing "
                    f"limit {self.config.per_position_loss_limit_pct:.2%}"
                )
                logger.warning(msg)
                msgs.append(msg)
                if worst_action != CircuitAction.HALT:
                    worst_action = CircuitAction.REDUCE
                worst_rf = min(worst_rf, rf)

        return worst_action, msgs, worst_rf

    def _check_sector_concentration(
        self,
        position_values: dict[str, float],
        sector_map: dict[str, str],
        current_nav: float,
    ) -> tuple[CircuitAction, list[str], float]:
        if current_nav <= 0:
            return CircuitAction.CONTINUE, [], 1.0

        sector_exposure: dict[str, float] = {}
        for symbol, value in position_values.items():
            sector = sector_map.get(symbol, "UNKNOWN")
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + abs(value)

        self.state.sector_weights = {s: v / current_nav for s, v in sector_exposure.items()}

        msgs: list[str] = []
        worst_rf = 1.0
        worst_action = CircuitAction.CONTINUE

        for sector, exposure in sector_exposure.items():
            weight = exposure / current_nav
            if weight >= self.config.sector_concentration_limit_pct:
                msg = (
                    f"SECTOR HALT {sector}: {weight:.2%} breaches "
                    f"limit {self.config.sector_concentration_limit_pct:.2%}"
                )
                logger.critical(msg)
                msgs.append(msg)
                worst_action = CircuitAction.HALT
                worst_rf = 0.0
            elif weight >= self.config.sector_reduce_threshold_pct:
                span = (
                    self.config.sector_concentration_limit_pct
                    - self.config.sector_reduce_threshold_pct
                )
                if abs(span) < 1e-9:
                    rf = 0.5
                else:
                    rf = max(0.1, 1.0 - (weight - self.config.sector_reduce_threshold_pct) / span)
                msg = (
                    f"SECTOR REDUCE {sector}: {weight:.2%} nearing "
                    f"limit {self.config.sector_concentration_limit_pct:.2%}"
                )
                logger.warning(msg)
                msgs.append(msg)
                if worst_action != CircuitAction.HALT:
                    worst_action = CircuitAction.REDUCE
                worst_rf = min(worst_rf, rf)

        return worst_action, msgs, worst_rf

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def _set_halt(self, reason: str) -> None:
        self.state.is_halted = True
        self.state.halt_reason = reason
        self.state.halt_timestamp = datetime.now()

    def _clear_halt(self) -> None:
        if self.state.is_halted:
            logger.info("Circuit breaker CLEARED — resuming normal operations")
        self.state.is_halted = False
        self.state.halt_reason = ""
        self.state.halt_timestamp = None

    def _record_breach(self, reason: str) -> None:
        self.state.breach_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
            }
        )
        # Keep last 100 breaches
        if len(self.state.breach_history) > 100:
            self.state.breach_history = self.state.breach_history[-100:]

    def force_halt(self, reason: str) -> None:
        """Manually halt trading — e.g. from a monitoring dashboard."""
        logger.critical(f"MANUAL HALT: {reason}")
        self._set_halt(f"MANUAL: {reason}")
        self._record_breach(f"MANUAL: {reason}")

    def force_resume(self) -> None:
        """Manually resume trading after manual review."""
        logger.info("MANUAL RESUME from halt")
        self._clear_halt()

    def get_state_snapshot(self) -> dict:
        """Return a serialisable snapshot of current state for dashboards."""
        return {
            "is_halted": self.state.is_halted,
            "halt_reason": self.state.halt_reason,
            "daily_pnl_pct": round(self.state.daily_pnl * 100, 4),
            "weekly_pnl_pct": round(self.state.weekly_pnl * 100, 4),
            "position_pnl": {k: round(v * 100, 4) for k, v in self.state.position_pnl.items()},
            "sector_weights": {k: round(v * 100, 2) for k, v in self.state.sector_weights.items()},
            "recent_breaches": self.state.breach_history[-5:],
        }
