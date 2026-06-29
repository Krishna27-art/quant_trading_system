"""
Pre-Trade Risk Guard

Hard checks that run BEFORE any order reaches the broker.
"""

import datetime
import threading
from dataclasses import dataclass

from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger("pre_trade_risk_guard")


NSE_HOLIDAYS_2025 = [
    "2025-01-26",
    "2025-03-14",
    "2025-04-10",
    "2025-04-14",
    "2025-04-18",
    "2025-05-01",
    "2025-06-07",
    "2025-08-15",
    "2025-08-27",
    "2025-10-02",
    "2025-10-02",
    "2025-10-24",
    "2025-11-05",
    "2025-12-25",
]


@dataclass
class RiskCheckResult:
    """Result of risk check."""

    ok_to_trade: bool
    reasons_blocked: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {"ok_to_trade": self.ok_to_trade, "reasons_blocked": self.reasons_blocked}


class PreTradeRiskGuard:
    """
    Hard checks that run BEFORE any order reaches the broker.

    If any check fails: block ALL orders for the session and alert.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.02,
        max_order_value: float = 50000,
        max_total_orders: int = 20,
        allowed_start: str = "09:20",
        allowed_end: str = "15:20",
        blocked_days: list[str] | None = None,
    ):
        """
        Initialize pre-trade risk guard.

        Args:
            max_daily_loss_pct: Halt if down >2% today
            max_order_value: No single order > ₹50K
            max_total_orders: Max 20 orders per rebalance
            allowed_start: Don't trade in first 5 mins
            allowed_end: Don't trade in last 10 mins
            blocked_days: NSE holidays
        """
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_order_value = max_order_value
        self.max_total_orders = max_total_orders
        self.allowed_start = datetime.time(*map(int, allowed_start.split(":")))
        self.allowed_end = datetime.time(*map(int, allowed_end.split(":")))
        self.blocked_days = blocked_days or []
        self.daily_pnl = 0.0
        self.orders_placed = 0
        self.KILL_SWITCH = False  # manual override
        self.lock = threading.Lock()
        self.logger = logger

    def check_all(self, orders: list[dict], current_capital: float) -> RiskCheckResult:
        """
        Run all pre-trade risk checks.

        Returns:
            RiskCheckResult with ok_to_trade and reasons_blocked
        """
        with self.lock:
            blocks = []

            # 1. Manual kill switch
            if self.KILL_SWITCH:
                blocks.append("KILL SWITCH ACTIVE — manual halt")

            # 2. Time window
            now = now_ist().time()
            if not (self.allowed_start <= now <= self.allowed_end):
                blocks.append(
                    f"Outside trading hours ({now} not in {self.allowed_start}–{self.allowed_end})"
                )

            # 3. Holiday check
            today = datetime.date.today().isoformat()
            if today in self.blocked_days:
                blocks.append(f"NSE holiday: {today}")

            # 4. Daily loss limit
            loss_pct = -self.daily_pnl / current_capital if current_capital > 0 else 0
            if loss_pct > self.max_daily_loss_pct:
                blocks.append(
                    f"Daily loss limit hit: {loss_pct:.1%} > {self.max_daily_loss_pct:.1%}"
                )

            # 5. Per-order value check
            oversized = [o for o in orders if o.get("value", 0) > self.max_order_value]
            if oversized:
                blocks.append(
                    f"{len(oversized)} orders exceed ₹{self.max_order_value:,}: "
                    f"{[o.get('symbol') for o in oversized]}"
                )

            # 6. Order count sanity
            if len(orders) > self.max_total_orders:
                blocks.append(f"Too many orders: {len(orders)} > {self.max_total_orders}")

            # 7. Duplicate symbol check (would cause double-buying)
            syms = [o.get("symbol") for o in orders]
            dupes = [s for s in syms if syms.count(s) > 1]
            if dupes:
                blocks.append(f"Duplicate symbols in order list: {set(dupes)}")

            ok = len(blocks) == 0
            if not ok:
                self.logger.error("PRE-TRADE GUARD BLOCKED:")
                for b in blocks:
                    self.logger.error(f"  - {b}")

            return RiskCheckResult(ok_to_trade=ok, reasons_blocked=blocks)

    def activate_kill_switch(self, reason: str = "manual"):
        """
        Activate kill switch.

        Args:
            reason: Reason for activation
        """
        self.KILL_SWITCH = True
        self.logger.error(f"KILL SWITCH ACTIVATED: {reason}")
        # Send Telegram/email alert
        try:
            from observability.alerting import AlertManager

            alert_manager = AlertManager()
            alert_manager.send_critical_alert(
                "KILL SWITCH ACTIVATED", f"Reason: {reason}\nAll executions halted."
            )
        except Exception as e:
            self.logger.error(f"Failed to dispatch kill switch alert: {e}")

    def deactivate_kill_switch(self, reason: str = "manual"):
        """
        Deactivate kill switch.

        Args:
            reason: Reason for deactivation
        """
        self.KILL_SWITCH = False
        self.logger.info(f"KILL SWITCH DEACTIVATED: {reason}")

    def update_daily_pnl(self, pnl_change: float):
        """
        Update daily P&L.

        Args:
            pnl_change: Change in P&L
        """
        with self.lock:
            self.daily_pnl += pnl_change
            self.logger.info(f"Daily P&L updated: {self.daily_pnl:.2f}")

    def reset_daily_pnl(self):
        """Reset daily P&L (call at start of new trading day)."""
        with self.lock:
            self.daily_pnl = 0.0
            self.orders_placed = 0
            self.logger.info("Daily P&L reset")

    def increment_orders_placed(self, count: int = 1):
        """
        Increment orders placed counter.

        Args:
            count: Number of orders placed
        """
        with self.lock:
            self.orders_placed += count
            self.logger.info(f"Orders placed: {self.orders_placed}")

    def get_status(self) -> dict:
        """
        Get current status of risk guard.

        Returns:
            Status dictionary
        """
        return {
            "kill_switch_active": self.KILL_SWITCH,
            "daily_pnl": self.daily_pnl,
            "orders_placed": self.orders_placed,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_order_value": self.max_order_value,
            "max_total_orders": self.max_total_orders,
            "allowed_start": self.allowed_start.strftime("%H:%M"),
            "allowed_end": self.allowed_end.strftime("%H:%M"),
        }


def create_default_risk_guard() -> PreTradeRiskGuard:
    """
    Create default risk guard with NSE 2025 holidays.

    Returns:
        PreTradeRiskGuard instance
    """
    return PreTradeRiskGuard(blocked_days=NSE_HOLIDAYS_2025)
