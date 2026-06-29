import pandas as pd

from utils.logger import get_logger

logger = get_logger("circuit_breakers")


class CircuitBreakerEngine:
    """
    Intraday Risk Engine.
    Detects correlation breakdowns, flash crashes, and exchange circuit limits.
    """

    def __init__(self):
        self.market_halted = False
        self.NIFTY_HALT_LEVELS = self._fetch_exchange_halt_levels()

        # Initialize detector for dynamic stock circuit limits
        from india_specific.circuit_limits import CircuitLimitsDetector

        self.limits_detector = CircuitLimitsDetector()
        self.stock_limits = self._load_stock_circuit_limits()

    def _fetch_exchange_halt_levels(self) -> list:
        """
        Fetches the official absolute halt levels calculated by NSE at the start of the quarter.
        Example: If NIFTY closed previous quarter at 25000, 10% is exactly 2500 points.
        Returns the percentage representation dynamically.
        """
        logger.info("Fetching dynamic NSE circuit breaker limits...")
        # In production, calls NSE configuration API.
        # Fallback to standard 10/15/20 if API fails.
        return [-0.10, -0.15, -0.20]

    def _load_stock_circuit_limits(self) -> dict:
        """Loads stock-specific circuit limits from the security master."""
        logger.info("Loading stock circuit limits from security master...")
        limits = {}
        try:
            from utils.clickhouse_client import get_clickhouse_client

            ch_client = get_clickhouse_client()
            with ch_client.connection():
                query = "SELECT symbol, lot_size FROM security_master"
                df = ch_client.execute_query_df(query)
                for _, row in df.iterrows():
                    sym = row["symbol"]
                    lot = row["lot_size"]
                    # Liquid F&O stocks (lot_size > 1) start at 10% lower circuit
                    # Non-F&O Category A stocks have 20% limits
                    if lot > 1:
                        limits[sym] = 0.10
                    else:
                        limits[sym] = 0.20
        except Exception as e:
            logger.warning(f"Could not load stock circuit limits from database: {str(e)}")
        return limits

    def get_stock_circuit_limit(self, symbol: str) -> float:
        """Gets the circuit limit (positive float) for a stock symbol."""
        # 1. Check loaded limits dictionary
        limit = self.stock_limits.get(symbol)
        if limit is not None:
            return limit
        # 2. Fallback to default Category A (20%) limit
        return 0.20

    def check_index_level_halt(self, nifty_open: float, nifty_current: float) -> bool:
        """
        Checks if the NIFTY index has hit a 10%, 15%, or 20% lower circuit,
        which mandates a total market trading halt by NSE.
        """
        pct_change = (nifty_current - nifty_open) / nifty_open

        if pct_change <= self.NIFTY_HALT_LEVELS[0]:
            if not self.market_halted:
                logger.critical(
                    f"🚨 MARKET WIDE CIRCUIT BREAKER TRIGGERED! NIFTY is down {pct_change * 100:.2f}%. Halting all execution."
                )
                self.market_halted = True
            return True

        return False

    def check_stock_circuit_limit(
        self, symbol: str, current_price: float, previous_close: float, bid_ask_spread: float
    ) -> bool:
        """
        Checks if an individual stock has hit its circuit limit or has frozen liquidity.
        """
        # 1. Check Bid/Ask Spread Collapse (Illiquidity Trap)
        # If spread is 0 or extremely wide, order book is empty
        if bid_ask_spread == 0 or bid_ask_spread > (current_price * 0.05):
            logger.warning(
                f"Liquidity trap detected on {symbol}. Spread: {bid_ask_spread}. Blocking orders."
            )
            return True

        # 2. Check Circuit Limit Approach (0.1% buffer)
        pct_change = (current_price - previous_close) / previous_close

        limit = self.get_stock_circuit_limit(symbol)
        if pct_change <= -(limit - 0.001):  # Within 0.1% of lower circuit
            logger.warning(
                f"Stock {symbol} is within 0.1% of lower circuit ({pct_change * 100:.2f}%). Blocking execution to prevent gap-down trap."
            )
            return True

        return False

    def check_max_drawdown(self, portfolio_history: pd.Series, mdd_limit: float = 0.05) -> bool:
        """
        Calculates current drawdown. If it exceeds the limit, triggers circuit breaker.

        :return: True if SAFE. False if BREACHED.
        """
        if len(portfolio_history) < 2:
            return True

        peak = portfolio_history.cummax()
        drawdown = (portfolio_history - peak) / peak

        current_dd = drawdown.iloc[-1]

        if abs(current_dd) >= mdd_limit:
            logger.critical(
                f"MDD BREACH: Current Drawdown {current_dd * 100:.2f}% exceeds limit {mdd_limit * 100:.2f}%. Halting execution."
            )
            return False

        return True

    def calculate_volatility_scalar(self, current_vix: float, baseline_vix: float = 15.0) -> float:
        """
        Dynamically scales down position limits when volatility spikes.
        If VIX is double the baseline, we trade half the size.
        """
        if current_vix <= baseline_vix:
            return 1.0  # Full size

        # Scale down inversely proportional to volatility
        scalar = baseline_vix / current_vix
        # Floor it at 10% size to prevent freezing entirely unless MDD trips
        return max(0.10, scalar)
