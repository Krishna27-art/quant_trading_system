from utils.logger import get_logger

logger = get_logger("beta_hedger")


class OvernightBetaHedger:
    """
    Prevents black swan overnight gap risk by ensuring the portfolio's
    net Beta to the NIFTY 50 is within acceptable institutional bounds
    before the market closes.
    """

    MAX_OVERNIGHT_NET_BETA = 0.15  # Max 15% directional exposure

    def __init__(self):
        from utils.clickhouse_client import get_clickhouse_client

        self.ch_client = get_clickhouse_client()
        self.stock_betas = self._fetch_live_betas()

    def _fetch_live_betas(self) -> dict[str, float]:
        """Fetches the 6-month trailing Beta from the Gold Feature Store or calculates it from prices."""
        logger.info("Fetching dynamic 6-month trailing betas from Gold Store...")

        # 1. Try to fetch from Gold Feature Store
        try:
            query = "SELECT symbol, beta_6m FROM gold_features WHERE date = (SELECT MAX(date) FROM gold_features)"
            df = self.ch_client.execute_query_df(query)
            if not df.empty and "symbol" in df.columns and "beta_6m" in df.columns:
                logger.info(f"Successfully loaded {len(df)} betas from Gold Store")
                return dict(zip(df["symbol"], df["beta_6m"], strict=False))
        except Exception as e:
            logger.warning(f"Could not fetch betas from Gold Store: {str(e)}")

        # 2. Fallback: Calculate rolling covariance dynamically from adjusted_equity_history (6 months / 180 days)
        try:
            logger.info("Calculating 6-month rolling covariance for beta from equity history...")
            import pandas as pd

            # Query returns from adjusted_equity_history or equity_history for the last 180 days
            query = """
                SELECT symbol, date, adj_close as close
                FROM adjusted_equity_history
                WHERE date >= today() - INTERVAL 180 DAY
                ORDER BY date ASC
            """
            try:
                df_prices = self.ch_client.execute_query_df(query)
            except Exception:
                df_prices = pd.DataFrame()

            if df_prices.empty:
                # Try raw equity history
                query = """
                    SELECT symbol, date, close
                    FROM equity_history
                    WHERE date >= today() - INTERVAL 180 DAY
                    ORDER BY date ASC
                """
                df_prices = self.ch_client.execute_query_df(query)

            if not df_prices.empty:
                # Pivot to get date as index, symbols as columns, close as values
                df_pivot = df_prices.pivot(index="date", columns="symbol", values="close")
                # Calculate returns
                df_returns = df_pivot.pct_change().dropna(how="all")

                # Use NIFTY 50 returns if available, otherwise fallback to mean of all stocks
                market_symbol = None
                for col in df_returns.columns:
                    if col.upper() in ["NIFTY 50", "NIFTY_50", "NIFTY"]:
                        market_symbol = col
                        break

                if market_symbol:
                    market_return = df_returns[market_symbol]
                else:
                    # Universe average proxy
                    market_return = df_returns.mean(axis=1)

                market_var = market_return.var()
                betas = {}
                if market_var > 0:
                    for col in df_returns.columns:
                        cov = df_returns[col].cov(market_return)
                        if not pd.isna(cov):
                            betas[col] = float(cov / market_var)

                if betas:
                    logger.info(f"Successfully calculated dynamic betas for {len(betas)} stocks.")
                    return betas
        except Exception as ex:
            logger.error(f"Failed to calculate rolling beta dynamically: {str(ex)}")

        logger.warning("Failed all dynamic beta fetches. Falling back to default empty dictionary.")
        return {}

    def calculate_portfolio_beta(
        self, net_positions: dict[str, float], total_capital: float
    ) -> float:
        """
        Calculates the weighted average Beta of the portfolio.
        net_positions: Dictionary of Symbol -> Position Value in INR.
        """
        portfolio_beta = 0.0

        for symbol, value in net_positions.items():
            weight = value / total_capital
            beta = self.stock_betas.get(symbol, 1.0)  # Default to 1.0 if unknown
            portfolio_beta += weight * beta

        logger.info(f"Current Portfolio Net Beta: {portfolio_beta:.3f}")
        return portfolio_beta

    def check_overnight_exposure(
        self, net_positions: dict[str, float], total_capital: float
    ) -> bool:
        """
        Validates if the portfolio is sufficiently market-neutral to be held overnight.
        If it fails, the execution engine must either buy/sell NIFTY Futures to hedge,
        or flatten the portfolio.
        """
        net_beta = self.calculate_portfolio_beta(net_positions, total_capital)

        if abs(net_beta) > self.MAX_OVERNIGHT_NET_BETA:
            logger.critical(
                f"OVERNIGHT RISK VIOLATION: Net Beta {net_beta:.3f} exceeds limit "
                f"of {self.MAX_OVERNIGHT_NET_BETA}. Flattening required before 15:15 IST."
            )
            return False

        logger.info("Overnight Beta Exposure is within safe institutional bounds.")
        return True
