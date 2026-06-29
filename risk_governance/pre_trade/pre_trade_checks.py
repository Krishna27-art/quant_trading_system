"""
Pre-Trade Risk Validation Engine

Every order must pass through this gate before reaching the execution layer.

Checks:
  1. Fat-finger detection (qty > N% of ADV or price deviation > N× ATR)
  2. SEBI position limit enforcement (OI limits, market-wide position limits)
  3. Restricted / banned stock list check
  4. Borrow availability check before short-selling
  5. Returns PreTradeResult(approved, reason, adjusted_qty)
"""

from dataclasses import dataclass, field

from utils.logger import get_logger

logger = get_logger("pre_trade_checks")


# ── Result type ───────────────────────────────────────────────────────
@dataclass
class PreTradeResult:
    """Structured result of pre-trade validation."""

    approved: bool
    reason: str
    adjusted_qty: int  # May be reduced from original if partially approved
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ── Configuration ─────────────────────────────────────────────────────
@dataclass
class PreTradeConfig:
    """Configurable thresholds for all pre-trade checks."""

    # Fat-finger: order qty as fraction of ADV
    max_qty_pct_of_adv: float = 0.05  # 5% of 20-day ADV
    # Fat-finger: price deviation in multiples of ATR
    max_price_atr_multiple: float = 3.0
    # Fat-finger: absolute notional cap per order (₹)
    max_order_notional: float = 5_00_00_000  # ₹5 crore
    # SEBI: max OI as fraction of market-wide position limit
    max_oi_utilisation: float = 0.90  # 90% of MWPL
    # SEBI: max qty per client as fraction of MWPL
    sebi_client_oi_limit_pct: float = 0.05  # 5% of MWPL for a single client
    # Borrow: minimum borrow pool qty to allow short
    min_borrow_pool_qty: int = 100


# ── Market data required for checks ──────────────────────────────────
@dataclass
class SymbolMarketData:
    """Per-symbol market data needed by the pre-trade checker."""

    symbol: str
    last_price: float
    adv_20d: float  # Average Daily Volume (20-day)
    atr_14d: float  # Average True Range (14-day)
    mwpl: int  # Market-Wide Position Limit (SEBI)
    current_oi: int  # Current open interest
    is_fno: bool  # Is in F&O segment
    lot_size: int = 1
    upper_circuit_limit: float = 0.0
    lower_circuit_limit: float = 0.0


class PreTradeChecker:
    """
    Pre-trade risk gate.  Validates every order against configurable
    safety checks before it reaches the broker gateway.
    """

    def __init__(self, config: PreTradeConfig | None = None):
        self.config = config or PreTradeConfig()
        self._restricted_symbols: set[str] = set()
        self._banned_symbols: set[str] = set()  # SEBI F&O ban list
        self._borrow_pool: dict[str, int] = {}  # symbol → available qty
        self._market_data: dict[str, SymbolMarketData] = {}
        self._client_oi: dict[str, int] = {}  # symbol → client OI
        import threading

        self._borrow_lock = threading.Lock()
        self._exceptions_timestamps: list[float] = []
        self._exceptions_lock = threading.Lock()
        logger.info("PreTradeChecker initialised")

    def _track_checker_exception(self) -> None:
        import time

        from risk_governance.pre_trade.kill_switch import execute_kill_switch

        now = time.time()
        with self._exceptions_lock:
            # Clean old timestamps (> 300 seconds)
            self._exceptions_timestamps = [
                t for t in self._exceptions_timestamps if now - t <= 300.0
            ]
            self._exceptions_timestamps.append(now)
            if len(self._exceptions_timestamps) > 3:
                logger.critical(
                    "🚨 PreTradeChecker encountered more than 3 exceptions in 5 minutes! Triggering global kill switch."
                )
                execute_kill_switch(dry_run=False)

    # ------------------------------------------------------------------
    # Data feeds
    # ------------------------------------------------------------------
    def load_restricted_list(self, symbols: list[str]) -> None:
        """Load internally restricted symbols (compliance, ESG, etc.)."""
        self._restricted_symbols = {s.upper() for s in symbols}
        logger.info("Loaded %d restricted symbols", len(self._restricted_symbols))

    def load_fno_ban_list(self, symbols: list[str]) -> None:
        """Load SEBI F&O ban list (securities exceeding 95% of MWPL)."""
        self._banned_symbols = {s.upper() for s in symbols}
        if self._banned_symbols:
            logger.warning("F&O BAN active for: %s", ", ".join(sorted(self._banned_symbols)))

    def update_borrow_pool(self, pool: dict[str, int]) -> None:
        """Update available borrow quantities (from SLB desk / broker)."""
        with self._borrow_lock:
            self._borrow_pool = {k.upper(): v for k, v in pool.items()}

    def update_market_data(self, data: dict[str, SymbolMarketData]) -> None:
        """Feed latest market data for the symbols we trade."""
        self._market_data = {k.upper(): v for k, v in data.items()}

    def update_client_oi(self, oi: dict[str, int]) -> None:
        """Update client-level open interest per symbol."""
        self._client_oi = {k.upper(): v for k, v in oi.items()}

    # ------------------------------------------------------------------
    # Core validation
    # ------------------------------------------------------------------
    def validate_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        order_type: str = "LIMIT",
        current_position: int = 0,
    ) -> PreTradeResult:
        """
        Run all pre-trade checks. Defaults to approved = False (fail-closed).
        """
        symbol = symbol.upper()
        side = side.upper()
        quantity = abs(quantity)

        approved = False
        reason = "Validation did not complete"
        passed: list[str] = []
        failed: list[str] = []
        warnings: list[str] = []
        adjusted_qty = quantity

        try:
            # 1. Restricted / banned check
            try:
                ok, msg = self._check_restricted(symbol, side, quantity, current_position)
                if ok:
                    passed.append("restricted_list")
                else:
                    failed.append(msg)
            except Exception as e:
                logger.error(f"Error in restricted check for {symbol}: {e}", exc_info=True)
                failed.append(f"Restricted check exception: {e}")
                self._track_checker_exception()

            # 2. Fat-finger checks
            try:
                ff_ok, ff_msg, ff_adj = self._check_fat_finger(
                    symbol,
                    quantity,
                    price,
                    order_type,
                )
                if ff_ok:
                    passed.append("fat_finger")
                    if ff_adj < quantity:
                        adjusted_qty = ff_adj
                        warnings.append(f"Qty reduced from {quantity} to {ff_adj} (ADV cap)")
                else:
                    failed.append(ff_msg)
            except Exception as e:
                logger.error(f"Error in fat-finger check for {symbol}: {e}", exc_info=True)
                failed.append(f"Fat-finger check exception: {e}")
                self._track_checker_exception()

            # 3. SEBI position limits
            try:
                sebi_ok, sebi_msg, sebi_adj = self._check_sebi_limits(
                    symbol,
                    side,
                    adjusted_qty,
                )
                if sebi_ok:
                    passed.append("sebi_limits")
                    if sebi_adj < adjusted_qty:
                        adjusted_qty = sebi_adj
                        warnings.append(
                            f"Qty reduced from {quantity} to {sebi_adj} (SEBI OI limit)"
                        )
                else:
                    failed.append(sebi_msg)
            except Exception as e:
                logger.error(f"Error in SEBI limits check for {symbol}: {e}", exc_info=True)
                failed.append(f"SEBI limits check exception: {e}")
                self._track_checker_exception()
                try:
                    from observability_mlops.alerting import send_critical_alert

                    send_critical_alert(f"SEBI pre-trade check exception for {symbol}: {e}")
                except Exception:
                    pass

            # 4. Borrow availability (short-sell only)
            if side == "SELL":
                try:
                    borrow_ok, borrow_msg = self._check_borrow(symbol, adjusted_qty)
                    if borrow_ok:
                        passed.append("borrow_availability")
                    else:
                        failed.append(borrow_msg)
                except Exception as e:
                    logger.error(f"Error in borrow check for {symbol}: {e}", exc_info=True)
                    failed.append(f"Borrow check exception: {e}")
                    self._track_checker_exception()
            else:
                passed.append("borrow_not_required")

            # 5. Notional cap
            try:
                notional_ok, notional_msg = self._check_notional_cap(adjusted_qty, price)
                if notional_ok:
                    passed.append("notional_cap")
                else:
                    failed.append(notional_msg)
            except Exception as e:
                logger.error(f"Error in notional check for {symbol}: {e}", exc_info=True)
                failed.append(f"Notional check exception: {e}")
                self._track_checker_exception()

            # 6. Exchange Circuit Bands check
            try:
                circuits_ok, circuits_msg = self._check_price_bands(symbol, price)
                if circuits_ok:
                    passed.append("circuit_bands")
                else:
                    failed.append(circuits_msg)
            except Exception as e:
                logger.error(f"Error in price bands check for {symbol}: {e}", exc_info=True)
                failed.append(f"Price bands check exception: {e}")
                self._track_checker_exception()

            approved = len(failed) == 0
            reason = "All checks passed" if approved else " | ".join(failed)

        except Exception as global_exc:
            logger.critical(f"Global exception in PreTradeChecker: {global_exc}", exc_info=True)
            self._track_checker_exception()
            approved = False
            reason = f"Global checker exception: {global_exc}"

        result = PreTradeResult(
            approved=approved,
            reason=reason,
            adjusted_qty=adjusted_qty if approved else 0,
            checks_passed=passed,
            checks_failed=failed,
            warnings=warnings,
        )

        if not approved:
            logger.warning(
                "ORDER REJECTED %s %s %d @ %.2f: %s",
                side,
                symbol,
                quantity,
                price,
                reason,
            )
        else:
            logger.info(
                "ORDER APPROVED %s %s %d @ %.2f (adj=%d)",
                side,
                symbol,
                quantity,
                price,
                adjusted_qty,
            )

        return result

    def check_symbol_heartbeats(self, last_tick_times: dict[str, float]) -> list[str]:
        """
        If no ticks received for 60 seconds during market hours, flag as halted.
        Returns list of symbols to liquidate immediately.
        """
        import time

        halted = []
        now = time.time()
        for symbol, last_time in last_tick_times.items():
            if now - last_time > 60.0:
                logger.error(
                    f"HEARTBEAT FAILED: {symbol} has not sent ticks for > 60s. Flagging as halted."
                )
                halted.append(symbol)
        return halted

    def _check_price_bands(self, symbol: str, price: float) -> tuple[bool, str]:
        """Check if limit price is within 0.5% of upper or lower circuits."""
        md = self._market_data.get(symbol)
        if md is None or md.upper_circuit_limit <= 0 or md.lower_circuit_limit <= 0:
            return True, ""

        upper_threshold = md.upper_circuit_limit * 0.995
        lower_threshold = md.lower_circuit_limit * 1.005

        if price >= upper_threshold:
            return (
                False,
                f"Price {price} is within 0.5% of Upper Circuit Limit {md.upper_circuit_limit}",
            )
        if price <= lower_threshold:
            return (
                False,
                f"Price {price} is within 0.5% of Lower Circuit Limit {md.lower_circuit_limit}",
            )

        return True, ""

    # ------------------------------------------------------------------
    # Individual check implementations
    # ------------------------------------------------------------------
    def _check_restricted(
        self, symbol: str, side: str, qty: int, current_position: int
    ) -> tuple[bool, str]:
        """Check restricted and F&O ban lists."""
        if symbol in self._restricted_symbols:
            return False, f"{symbol} is on the restricted list"

        if symbol in self._banned_symbols:
            if side == "BUY":
                if current_position >= 0:
                    return False, f"{symbol} is under SEBI F&O ban; cannot open new long"
                if qty > abs(current_position):
                    return (
                        False,
                        f"{symbol} is under SEBI F&O ban; can only cover {abs(current_position)} shares",
                    )
            elif side == "SELL":
                if current_position <= 0:
                    return False, f"{symbol} is under SEBI F&O ban; cannot open new short"
                if qty > current_position:
                    return (
                        False,
                        f"{symbol} is under SEBI F&O ban; can only reduce {current_position} shares",
                    )

        return True, ""

    def _check_fat_finger(
        self,
        symbol: str,
        quantity: int,
        price: float,
        order_type: str,
    ) -> tuple[bool, str, int]:
        """
        Detect likely erroneous orders:
          - Qty > N% of 20-day ADV
          - Price > N × ATR away from last trade
        """
        md = self._market_data.get(symbol)
        if md is None:
            # No market data → conservative pass with warning
            logger.warning("No market data for %s — fat-finger check skipped", symbol)
            return True, "", quantity

        adjusted = quantity

        # --- Volume check ---
        if md.adv_20d > 0:
            max_qty = int(md.adv_20d * self.config.max_qty_pct_of_adv)
            if quantity > max_qty:
                adjusted = max_qty
                logger.warning(
                    "Fat-finger: %s qty %d > %d (%.1f%% of ADV %d). Capping.",
                    symbol,
                    quantity,
                    max_qty,
                    self.config.max_qty_pct_of_adv * 100,
                    int(md.adv_20d),
                )

        # --- Price deviation check (limit orders only) ---
        if order_type == "LIMIT" and md.atr_14d > 0 and md.last_price > 0:
            deviation = abs(price - md.last_price)
            max_dev = md.atr_14d * self.config.max_price_atr_multiple
            if deviation > max_dev:
                msg = (
                    f"Fat-finger PRICE: {symbol} order price ₹{price:.2f} deviates "
                    f"₹{deviation:.2f} from last ₹{md.last_price:.2f} "
                    f"(>{self.config.max_price_atr_multiple:.0f}× ATR ₹{md.atr_14d:.2f})"
                )
                return False, msg, 0

        return True, "", adjusted

    def _check_sebi_limits(self, symbol: str, side: str, quantity: int) -> tuple[bool, str, int]:
        """
        SEBI position limit enforcement:
          - Market-wide OI must be < 95% of MWPL to open fresh positions
          - Client-level OI limit (typically 5% of MWPL for a single client)
        """
        md = self._market_data.get(symbol)
        if md is None:
            # Missing market data for F&O checks must fail closed!
            return False, f"Missing market data for {symbol} to verify SEBI limits", 0

        if not md.is_fno:
            return True, "", quantity

        if md.mwpl <= 0:
            return False, f"Invalid MWPL {md.mwpl} for {symbol}", 0

        # Market-wide check: block if OI ≥ configured threshold
        oi_util = md.current_oi / md.mwpl
        if side == "BUY" and oi_util >= self.config.max_oi_utilisation:
            return (
                False,
                (
                    f"SEBI OI limit: {symbol} market OI {md.current_oi:,} is "
                    f"{oi_util:.1%} of MWPL {md.mwpl:,} (threshold "
                    f"{self.config.max_oi_utilisation:.0%})"
                ),
                0,
            )

        # Client-level OI check
        client_oi = self._client_oi.get(symbol, 0)
        max_client_oi = int(md.mwpl * self.config.sebi_client_oi_limit_pct)
        if side == "BUY" and (client_oi + quantity) > max_client_oi:
            allowed = max(0, max_client_oi - client_oi)
            # Round down to lot size
            if md.lot_size > 1:
                allowed = (allowed // md.lot_size) * md.lot_size
            if allowed <= 0:
                return (
                    False,
                    (
                        f"Client OI limit: {symbol} client OI {client_oi:,} + "
                        f"{quantity:,} exceeds {max_client_oi:,} "
                        f"({self.config.sebi_client_oi_limit_pct:.0%} of MWPL)"
                    ),
                    0,
                )
            return True, "", allowed

        return True, "", quantity

    def _check_borrow(self, symbol: str, quantity: int) -> tuple[bool, str]:
        """Verify shares can be borrowed before short-selling using local cache."""
        with self._borrow_lock:
            available = self._borrow_pool.get(symbol, 0)
            if quantity > available:
                return False, f"Insufficient borrow pool (req {quantity}, have {available})"

            # Deduct from pool immediately to prevent double spending
            self._borrow_pool[symbol] -= quantity
            return True, ""

    def _check_notional_cap(self, quantity: int, price: float) -> tuple[bool, str]:
        """Absolute notional value cap per single order."""
        notional = quantity * price
        if notional > self.config.max_order_notional:
            return False, (
                f"Order notional ₹{notional:,.0f} exceeds cap "
                f"₹{self.config.max_order_notional:,.0f}"
            )
        return True, ""

    # ------------------------------------------------------------------
    # Batch validation
    # ------------------------------------------------------------------
    def validate_order_batch(
        self,
        orders: list[dict],
    ) -> list[PreTradeResult]:
        """
        Validate a batch of orders. Each dict must have keys:
        symbol, side, quantity, price, and optionally order_type and current_position.
        """
        results = []
        for o in orders:
            r = self.validate_order(
                symbol=o["symbol"],
                side=o["side"],
                quantity=o["quantity"],
                price=o["price"],
                order_type=o.get("order_type", "LIMIT"),
                current_position=o.get("current_position", 0),
            )
            results.append(r)
        return results
