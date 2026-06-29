"""
Inventory Reconciliation Engine

Compares the internal (local) position state against the broker's reported
positions and flags discrepancies at three severity levels:

  - INFO:     Minor rounding / fractional share differences
  - WARNING:  Material qty or value divergence requiring investigation
  - CRITICAL: Large mismatch — auto-halt trading until resolved

Designed to run:
  1. Post-trade intraday (after every fill)
  2. End-of-day batch (EOD snap vs broker contract notes)
  3. Pre-market open (verify overnight corporate actions)
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pandas as pd

from utils.logger import get_logger

logger = get_logger("reconciliation")


# ── Data models ───────────────────────────────────────────────────────
class MismatchSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class PositionRecord:
    """A position as seen by one source (local or broker)."""

    symbol: str
    quantity: int
    avg_price: float
    market_value: float  # qty × current_price
    segment: str = "EQ"  # EQ, FO, CDS


@dataclass
class ReconciliationMismatch:
    """One discrepancy detected during reconciliation."""

    symbol: str
    field: str  # 'quantity', 'avg_price', 'value', 'missing'
    local_value: float
    broker_value: float
    difference: float
    difference_pct: float
    severity: MismatchSeverity
    message: str


@dataclass
class ReconciliationReport:
    """Full reconciliation result."""

    timestamp: datetime
    total_symbols_checked: int
    matched: int
    mismatched: int
    info_count: int
    warning_count: int
    critical_count: int
    mismatches: list[ReconciliationMismatch]
    should_halt: bool
    halt_reason: str


class ReconciliationEngine:
    """
    Position reconciliation between internal state and broker.
    """

    # Thresholds
    QTY_INFO_THRESHOLD = 0  # any diff at INFO level
    QTY_WARNING_THRESHOLD = 5  # > 5 shares or lots
    QTY_CRITICAL_THRESHOLD_PCT = 0.10  # > 10% of position
    VALUE_WARNING_THRESHOLD = 50_000  # ₹50k
    VALUE_CRITICAL_THRESHOLD = 5_00_000  # ₹5L
    PRICE_WARNING_PCT = 0.02  # 2% avg price difference
    PRICE_CRITICAL_PCT = 0.05  # 5% avg price difference

    def __init__(
        self,
        qty_critical_pct: float = 0.10,
        value_critical_inr: float = 5_00_000,
        auto_halt_on_critical: bool = True,
    ):
        self.qty_critical_pct = qty_critical_pct
        self.value_critical_inr = value_critical_inr
        self.auto_halt = auto_halt_on_critical
        self._halt_active: bool = False
        self._halt_reason: str = ""
        self._history: list[ReconciliationReport] = []
        logger.info(
            "ReconciliationEngine initialised | auto_halt=%s qty_crit_pct=%.0f%% value_crit=₹%.0f",
            auto_halt_on_critical,
            qty_critical_pct * 100,
            value_critical_inr,
        )

    # ------------------------------------------------------------------
    # Core reconciliation
    # ------------------------------------------------------------------
    def reconcile(
        self,
        local_positions: dict[str, PositionRecord],
        broker_positions: dict[str, PositionRecord],
    ) -> ReconciliationReport:
        """
        Compare local vs broker positions and produce a report.

        Args:
            local_positions: {symbol: PositionRecord} from internal state.
            broker_positions: {symbol: PositionRecord} from broker API.

        Returns:
            ReconciliationReport with all mismatches classified.
        """
        all_symbols = set(local_positions.keys()) | set(broker_positions.keys())
        mismatches: list[ReconciliationMismatch] = []
        matched = 0

        for symbol in sorted(all_symbols):
            local = local_positions.get(symbol)
            broker = broker_positions.get(symbol)

            symbol_mismatches = self._compare_position(symbol, local, broker)

            if symbol_mismatches:
                mismatches.extend(symbol_mismatches)
            else:
                matched += 1

        info_ct = sum(1 for m in mismatches if m.severity == MismatchSeverity.INFO)
        warn_ct = sum(1 for m in mismatches if m.severity == MismatchSeverity.WARNING)
        crit_ct = sum(1 for m in mismatches if m.severity == MismatchSeverity.CRITICAL)

        should_halt = self.auto_halt and crit_ct > 0
        halt_reason = ""
        if should_halt:
            crit_symbols = sorted(
                {m.symbol for m in mismatches if m.severity == MismatchSeverity.CRITICAL}
            )
            halt_reason = f"CRITICAL reconciliation mismatch on: {', '.join(crit_symbols)}"
            self._halt_active = True
            self._halt_reason = halt_reason
            logger.critical("🚨 AUTO-HALT: %s", halt_reason)

        report = ReconciliationReport(
            timestamp=datetime.now(),
            total_symbols_checked=len(all_symbols),
            matched=matched,
            mismatched=len(all_symbols) - matched,
            info_count=info_ct,
            warning_count=warn_ct,
            critical_count=crit_ct,
            mismatches=mismatches,
            should_halt=should_halt,
            halt_reason=halt_reason,
        )

        self._history.append(report)
        if len(self._history) > 200:
            self._history = self._history[-200:]

        self._log_report(report)
        return report

    # ------------------------------------------------------------------
    # Position comparison
    # ------------------------------------------------------------------
    def _compare_position(
        self,
        symbol: str,
        local: PositionRecord | None,
        broker: PositionRecord | None,
    ) -> list[ReconciliationMismatch]:
        mismatches: list[ReconciliationMismatch] = []

        # --- Missing on one side ---
        if local is None and broker is not None:
            severity = self._severity_for_qty(0, broker.quantity)
            mismatches.append(
                ReconciliationMismatch(
                    symbol=symbol,
                    field="missing_local",
                    local_value=0,
                    broker_value=broker.quantity,
                    difference=broker.quantity,
                    difference_pct=1.0,
                    severity=severity,
                    message=(
                        f"{symbol}: exists at broker (qty={broker.quantity}) but NOT in local state"
                    ),
                )
            )
            return mismatches

        if broker is None and local is not None:
            severity = self._severity_for_qty(local.quantity, 0)
            mismatches.append(
                ReconciliationMismatch(
                    symbol=symbol,
                    field="missing_broker",
                    local_value=local.quantity,
                    broker_value=0,
                    difference=local.quantity,
                    difference_pct=1.0,
                    severity=severity,
                    message=(f"{symbol}: exists locally (qty={local.quantity}) but NOT at broker"),
                )
            )
            return mismatches

        assert local is not None and broker is not None

        # --- Quantity comparison ---
        qty_diff = local.quantity - broker.quantity
        if qty_diff != 0:
            max_qty = max(abs(local.quantity), abs(broker.quantity), 1)
            pct = abs(qty_diff) / max_qty
            severity = self._severity_for_qty(local.quantity, broker.quantity)
            mismatches.append(
                ReconciliationMismatch(
                    symbol=symbol,
                    field="quantity",
                    local_value=local.quantity,
                    broker_value=broker.quantity,
                    difference=qty_diff,
                    difference_pct=pct,
                    severity=severity,
                    message=(
                        f"{symbol} qty: local={local.quantity} broker="
                        f"{broker.quantity} diff={qty_diff}"
                    ),
                )
            )

        # --- Average price comparison ---
        price_diff = local.avg_price - broker.avg_price
        if broker.avg_price > 0:
            price_pct = abs(price_diff) / broker.avg_price
        else:
            price_pct = 0.0 if local.avg_price == 0 else 1.0

        if price_pct > 0.001:  # ignore sub-0.1% rounding
            severity = self._severity_for_price(price_pct)
            mismatches.append(
                ReconciliationMismatch(
                    symbol=symbol,
                    field="avg_price",
                    local_value=local.avg_price,
                    broker_value=broker.avg_price,
                    difference=price_diff,
                    difference_pct=price_pct,
                    severity=severity,
                    message=(
                        f"{symbol} avg_price: local=₹{local.avg_price:.2f} "
                        f"broker=₹{broker.avg_price:.2f} diff={price_pct:.2%}"
                    ),
                )
            )

        # --- Market value comparison ---
        val_diff = local.market_value - broker.market_value
        if abs(val_diff) > 1:  # ignore ₹1 rounding
            max_val = max(abs(local.market_value), abs(broker.market_value), 1)
            val_pct = abs(val_diff) / max_val
            severity = self._severity_for_value(abs(val_diff))
            mismatches.append(
                ReconciliationMismatch(
                    symbol=symbol,
                    field="market_value",
                    local_value=local.market_value,
                    broker_value=broker.market_value,
                    difference=val_diff,
                    difference_pct=val_pct,
                    severity=severity,
                    message=(
                        f"{symbol} value: local=₹{local.market_value:,.0f} "
                        f"broker=₹{broker.market_value:,.0f} "
                        f"diff=₹{val_diff:,.0f}"
                    ),
                )
            )

        return mismatches

    # ------------------------------------------------------------------
    # Severity classification
    # ------------------------------------------------------------------
    def _severity_for_qty(self, local_qty: int, broker_qty: int) -> MismatchSeverity:
        diff = abs(local_qty - broker_qty)
        max_qty = max(abs(local_qty), abs(broker_qty), 1)
        pct = diff / max_qty

        if pct >= self.qty_critical_pct or diff >= self.QTY_CRITICAL_THRESHOLD_PCT * max_qty:
            return MismatchSeverity.CRITICAL
        if diff > self.QTY_WARNING_THRESHOLD:
            return MismatchSeverity.WARNING
        return MismatchSeverity.INFO

    def _severity_for_price(self, pct_diff: float) -> MismatchSeverity:
        if pct_diff >= self.PRICE_CRITICAL_PCT:
            return MismatchSeverity.CRITICAL
        if pct_diff >= self.PRICE_WARNING_PCT:
            return MismatchSeverity.WARNING
        return MismatchSeverity.INFO

    def _severity_for_value(self, abs_diff: float) -> MismatchSeverity:
        if abs_diff >= self.value_critical_inr:
            return MismatchSeverity.CRITICAL
        if abs_diff >= self.VALUE_WARNING_THRESHOLD:
            return MismatchSeverity.WARNING
        return MismatchSeverity.INFO

    # ------------------------------------------------------------------
    # Halt management
    # ------------------------------------------------------------------
    @property
    def is_halted(self) -> bool:
        return self._halt_active

    def acknowledge_and_resume(self, operator: str, notes: str) -> None:
        """Operator acknowledges a critical mismatch and clears the halt."""
        logger.info(
            "HALT CLEARED by %s | reason was: %s | notes: %s",
            operator,
            self._halt_reason,
            notes,
        )
        self._halt_active = False
        self._halt_reason = ""

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def _log_report(self, report: ReconciliationReport) -> None:
        if report.critical_count > 0:
            logger.critical(
                "RECON: %d symbols | %d matched | %d mismatched "
                "(INFO=%d WARN=%d CRIT=%d) → HALT=%s",
                report.total_symbols_checked,
                report.matched,
                report.mismatched,
                report.info_count,
                report.warning_count,
                report.critical_count,
                report.should_halt,
            )
        elif report.warning_count > 0:
            logger.warning(
                "RECON: %d symbols | %d matched | %d mismatched (INFO=%d WARN=%d)",
                report.total_symbols_checked,
                report.matched,
                report.mismatched,
                report.info_count,
                report.warning_count,
            )
        else:
            logger.info(
                "RECON OK: %d symbols matched, %d INFO-level diffs",
                report.matched,
                report.info_count,
            )

    def report_to_df(self, report: ReconciliationReport | None = None) -> pd.DataFrame:
        """Convert the latest (or given) report to a DataFrame."""
        rpt = report or (self._history[-1] if self._history else None)
        if rpt is None:
            return pd.DataFrame()
        rows = []
        for m in rpt.mismatches:
            rows.append(
                {
                    "symbol": m.symbol,
                    "field": m.field,
                    "local": m.local_value,
                    "broker": m.broker_value,
                    "diff": m.difference,
                    "diff_pct": round(m.difference_pct * 100, 2),
                    "severity": m.severity.value,
                    "message": m.message,
                }
            )
        return pd.DataFrame(rows)

    def get_history_summary(self) -> pd.DataFrame:
        """Summarise all historical reconciliation runs."""
        rows = []
        for r in self._history:
            rows.append(
                {
                    "timestamp": r.timestamp,
                    "symbols": r.total_symbols_checked,
                    "matched": r.matched,
                    "mismatched": r.mismatched,
                    "info": r.info_count,
                    "warning": r.warning_count,
                    "critical": r.critical_count,
                    "halt": r.should_halt,
                }
            )
        return pd.DataFrame(rows)
