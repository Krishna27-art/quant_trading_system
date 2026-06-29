import numpy as np

from utils.logger import get_logger

logger = get_logger("portfolio_risk")

from scipy.stats import t as student_t

# Static Sector Mapping for Nifty 30 MVP
SECTOR_MAP = {
    "HDFCBANK": "Financials",
    "ICICIBANK": "Financials",
    "SBIN": "Financials",
    "KOTAKBANK": "Financials",
    "BAJFINANCE": "Financials",
    "BAJAJFINSV": "Financials",
    "RELIANCE": "Energy",
    "ONGC": "Energy",
    "NTPC": "Energy",
    "POWERGRID": "Energy",
    "COALINDIA": "Energy",
    "TCS": "IT",
    "INFY": "IT",
    "HCLTECH": "IT",
    "TATAMOTORS": "Auto",
    "M&M": "Auto",
    "MARUTI": "Auto",
    "ITC": "FMCG",
    "HINDUNILVR": "FMCG",
    "ASIANPAINT": "FMCG",
    "TITAN": "Consumer",
    "LT": "Capital Goods",
    "BHARTIARTL": "Telecom",
    "SUNPHARMA": "Pharma",
    "ULTRACEMCO": "Materials",
    "TATASTEEL": "Materials",
    "ADANIENT": "Conglomerate",
    "ADANIPORTS": "Infrastructure",
}


class PortfolioRiskEngine:
    """
    Portfolio-Level Risk Controls
    Oversees the entire book to prevent systemic blowups (concentration, beta, daily limit).
    Uses fat-tail historical simulation and Student-t Monte Carlo VaR models.
    """

    def __init__(self, total_capital: float = 1000000.0):
        self.total_capital = total_capital
        self.max_open_positions = 5
        self.max_positions_per_sector = 2
        self.daily_loss_limit_pct = -0.02  # Halt if loss > 2% of capital
        self.max_var_limit_pct = 0.03  # 3% VaR limit

    def check_daily_loss_limit(self, current_day_pnl: float) -> bool:
        """
        Returns True if trading should be HALTED for the day.
        """
        pnl_pct = current_day_pnl / self.total_capital
        if pnl_pct <= self.daily_loss_limit_pct:
            logger.critical(
                f"DAILY LOSS LIMIT REACHED! PnL: {pnl_pct * 100:.2f}%. Halting new trades."
            )
            return True
        return False

    def check_position_limits(
        self, open_positions: list[dict], new_symbol: str
    ) -> tuple[bool, str]:
        """
        Checks if a new trade would violate max positions or sector concentration.
        Returns (is_allowed, reason).
        """
        if len(open_positions) >= self.max_open_positions:
            return False, f"Max open positions ({self.max_open_positions}) reached."

        new_sector = SECTOR_MAP.get(new_symbol.split(".")[0], "Unknown")

        sector_count = 0
        for pos in open_positions:
            sym_clean = pos["symbol"].split(".")[0]
            if SECTOR_MAP.get(sym_clean, "Unknown") == new_sector:
                sector_count += 1

        if sector_count >= self.max_positions_per_sector:
            return (
                False,
                f"Sector limit ({self.max_positions_per_sector}) reached for {new_sector}.",
            )

        return True, "Allowed"

    def check_beta_neutrality(
        self,
        open_positions: list[dict],
        new_symbol: str,
        new_is_buy: bool,
        stock_betas: dict[str, float],
    ) -> tuple[bool, str]:
        """
        Ensures net portfolio beta remains between -0.5 and +0.5.
        `stock_betas` should be a dictionary mapping symbol to its Nifty beta.
        """
        net_beta = 0.0

        # Calculate current net beta
        for pos in open_positions:
            beta = stock_betas.get(pos["symbol"], 1.0)
            weight = pos["position_size"]  # Assuming unit sizes
            if pos["side"] == "BUY":
                net_beta += beta * weight
            else:
                net_beta -= beta * weight

        # Add proposed trade
        new_beta = stock_betas.get(new_symbol, 1.0)
        # Assume 1.0 unit size for the check
        proposed_net_beta = net_beta + new_beta if new_is_buy else net_beta - new_beta

        # Normalize by total allowed positions (5)
        portfolio_beta = proposed_net_beta / self.max_open_positions

        if portfolio_beta < -0.5 or portfolio_beta > 0.5:
            return False, f"Trade violates Beta Neutrality. Proposed Beta: {portfolio_beta:.2f}"

        return True, f"Allowed (Beta: {portfolio_beta:.2f})"

    def calculate_historical_var_and_cvar(
        self, rolling_500d_returns: np.ndarray, confidence_level: float = 0.95
    ) -> tuple[float, float]:
        """
        Calculates 1-day Historical Simulation VaR (Value at Risk) and CVaR (Expected Shortfall).
        rolling_500d_returns: (500,) numpy array of daily returns of the current portfolio composition.
        Returns: (VaR_pct, CVaR_pct)
        """
        if len(rolling_500d_returns) < 100:
            # Fallback to parametric if lookback is insufficient
            return 0.02, 0.03

        sorted_returns = np.sort(rolling_500d_returns)
        percentile_idx = int((1.0 - confidence_level) * len(sorted_returns))

        var_pct = abs(sorted_returns[percentile_idx])
        cvar_pct = abs(np.mean(sorted_returns[:percentile_idx]))

        return var_pct, cvar_pct

    def run_student_t_monte_carlo_var(
        self,
        current_weights: dict[str, float],
        historical_returns: np.ndarray,
        num_simulations: int = 10000,
        degrees_of_freedom: int = 5,
    ) -> float:
        """
        Runs Monte Carlo simulation using Student-t distribution to capture fat-tail risks.
        historical_returns: (T, N) daily returns matrix.
        Returns: 1-day 95% Monte Carlo VaR percentage.
        """
        if historical_returns.ndim != 2:
            return 0.02

        weights = np.array([current_weights.get(sym, 0.0) for sym in current_weights])
        if weights.sum() == 0:
            return 0.0

        # Compute mean and covariance matrix of returns
        mean_returns = np.mean(historical_returns, axis=0)
        cov_matrix = np.cov(historical_returns, rowvar=False)

        # Cholesky decomposition of cov matrix
        try:
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # Fallback if covariance matrix is not positive-definite
            L = np.linalg.cholesky(cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6)

        # Draw Student-t random variables
        t_rand = student_t.rvs(df=degrees_of_freedom, size=(num_simulations, len(weights)))

        # Correlate simulations using Cholesky factor L
        simulated_returns = np.dot(t_rand, L.T) + mean_returns

        # Portfolio returns
        portfolio_sim_returns = np.dot(simulated_returns, weights)

        # Sort simulated returns worst to best
        sorted_sim_returns = np.sort(portfolio_sim_returns)
        percentile_idx = int(0.05 * num_simulations)

        mc_var_pct = abs(sorted_sim_returns[percentile_idx])
        return mc_var_pct

    def check_liquidity_filter(
        self, adv_20d_rupees: float, bid_ask_spread_pct: float, trades_per_hour: int
    ) -> bool:
        """
        Institutional Liquidity Filter.
        Halt trades on stocks that fail standard liquidity checks.
        """
        min_adv = 500000000.0  # ₹50 Cr
        max_spread = 0.0015  # 0.15%
        min_trades = 500

        if adv_20d_rupees < min_adv:
            logger.warning(f"Liquidity Filter FAILED: ADV ₹{adv_20d_rupees:,.0f} < ₹{min_adv:,.0f}")
            return False
        if bid_ask_spread_pct > max_spread:
            logger.warning(
                f"Liquidity Filter FAILED: Spread {bid_ask_spread_pct:.4f} > {max_spread}"
            )
            return False
        if trades_per_hour < min_trades:
            logger.warning(f"Liquidity Filter FAILED: Trades/Hour {trades_per_hour} < {min_trades}")
            return False

        return True
