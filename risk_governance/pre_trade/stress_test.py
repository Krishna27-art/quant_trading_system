"""
Portfolio Stress Testing Engine — Indian Markets

Simulates extreme tail events (black swans), correlation breakdowns,
and historical Indian market stress scenarios (e.g., COVID-19 limit down open,
IL&FS crisis, 2016 Demonetization) to verify portfolio survival.
"""

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


class StressTestEngine:
    """
    Simulates severe risk scenarios and correlation shock conditions.
    Fails the portfolio if estimated drawdown exceeds maximum tolerable limit.
    """

    @staticmethod
    def simulate_correlation_breakdown(
        weights: dict[str, float], volatilities: pd.Series, correlation_shock: float = 0.8
    ) -> float:
        """
        Simulates a systematic correlation breakdown where diversification vanishes.
        Cross-asset correlations are pushed towards a high positive shock value.

        Parameters
        ----------
        weights : Dict[str, float]
            Portfolio weights.
        volatilities : pd.Series
            Asset standard deviations.
        correlation_shock : float
            Correlation level to force across all assets (default 0.80).

        Returns
        -------
        float
            Estimated portfolio crash volatility.
        """
        w = pd.Series(weights)
        common_assets = w.index.intersection(volatilities.index)
        if len(common_assets) == 0:
            return 0.0

        w = w.loc[common_assets]
        vols = volatilities.loc[common_assets]
        n = len(w)

        # Reconstruct shocked correlation matrix: 1 on diagonal, correlation_shock off-diagonal
        corr_matrix = np.full((n, n), correlation_shock)
        np.fill_diagonal(corr_matrix, 1.0)

        # Covariance matrix: V = diag(vol) * R * diag(vol)
        v_diag = np.diag(vols.values)
        cov_matrix = v_diag @ corr_matrix @ v_diag

        # Portfolio variance: w^T * V * w
        port_var = w.values.T @ cov_matrix @ w.values
        shock_vol = np.sqrt(max(0.0, port_var))

        logger.info("Correlation breakdown simulation | Shock Volatility: %.2f%%", shock_vol * 100)
        return shock_vol

    @staticmethod
    def simulate_india_demonetization(weights: dict[str, float]) -> float:
        """
        Simulates the 2016 Demonetization shock.
        Characterized by a sharp drop in cash-intensive sectors (Real Estate, Auto, FMCG)
        and a surge/stabilization in financials.
        - Real Estate: -15%
        - Auto: -10%
        - Financials: -2%
        - Others: -5%
        """
        w = pd.Series(weights)
        # In a production setup, sector maps are fetched from the security master.
        # Fallback to sector category checks or general mock sector rules.
        expected_loss = 0.0
        for asset, weight in w.items():
            asset_upper = asset.upper()
            if "DLF" in asset_upper or "REAL" in asset_upper or "PROP" in asset_upper:
                expected_loss += weight * -0.15
            elif "MARUTI" in asset_upper or "TATMOTO" in asset_upper or "AUTO" in asset_upper:
                expected_loss += weight * -0.10
            elif "SBI" in asset_upper or "HDFCBANK" in asset_upper or "FIN" in asset_upper:
                expected_loss += weight * -0.02
            else:
                expected_loss += weight * -0.05

        return expected_loss

    @staticmethod
    def simulate_ilfs_crisis(weights: dict[str, float]) -> float:
        """
        Simulates the 2018 IL&FS crisis.
        Severe credit squeeze causing extreme selloffs in NBFCs and housing finance.
        - NBFCs / HFCs (e.g., L&TFH, IBULHSGFIN, DHFL): -30%
        - Private Banks: -10%
        - Broad Market: -5%
        """
        w = pd.Series(weights)
        expected_loss = 0.0
        for asset, weight in w.items():
            asset_upper = asset.upper()
            if (
                "DHFL" in asset_upper
                or "IBUL" in asset_upper
                or "L&T" in asset_upper
                or "NBFC" in asset_upper
            ):
                expected_loss += weight * -0.30
            elif "HDFC" in asset_upper or "ICICI" in asset_upper or "AXIS" in asset_upper:
                expected_loss += weight * -0.10
            else:
                expected_loss += weight * -0.05

        return expected_loss

    @staticmethod
    def simulate_covid_circuit_breaker(weights: dict[str, float]) -> float:
        """
        Simulates the March 2020 limit-down circuit breaker days.
        Index halts at 10% lower circuit within minutes of open. Huge volatility.
        - Broad Equities: -12% shock
        - High-beta Stocks: -18% shock
        - Safe-haven/Gold/USD: +2% buffer
        """
        w = pd.Series(weights)
        expected_loss = 0.0
        for asset, weight in w.items():
            asset_upper = asset.upper()
            if "GOLD" in asset_upper or "USD" in asset_upper:
                expected_loss += weight * 0.02
            elif weight > 0:  # Long equity positions
                # Simple high beta proxy (e.g. mid/small caps or financial leverage)
                if "FIN" in asset_upper or "INFY" in asset_upper or "TCS" in asset_upper:
                    expected_loss += weight * -0.12
                else:
                    expected_loss += weight * -0.18
            else:  # Short equity positions
                expected_loss += (
                    weight * -0.10
                )  # Shorts profit from drop, but slippage reduces returns

        return expected_loss

    @staticmethod
    def run_full_stress_test(
        weights: dict[str, float],
        volatilities: pd.Series,
        max_tolerable_loss: float = -0.15,
    ) -> tuple[bool, dict[str, float]]:
        """
        Runs all historical and hypothetical stress test scenarios.

        Returns
        -------
        Tuple[bool, Dict[str, float]]
            - True if portfolio passes all scenarios (loss >= max_tolerable_loss), False otherwise.
            - Dictionary of expected return/loss per scenario.
        """
        # Correlation breakdown: 3-sigma event using shocked vol
        shock_vol = StressTestEngine.simulate_correlation_breakdown(weights, volatilities)
        loss_breakdown = -3.0 * shock_vol

        # Historical India scenarios
        loss_demo = StressTestEngine.simulate_india_demonetization(weights)
        loss_ilfs = StressTestEngine.simulate_ilfs_crisis(weights)
        loss_covid = StressTestEngine.simulate_covid_circuit_breaker(weights)

        # Basic overnight gap and short squeeze (re-implemented from prior simple engine)
        net_exposure = sum(weights.values())
        loss_gap = net_exposure * -0.20

        shorts = [w for w in weights.values() if w < 0]
        loss_squeeze = min(shorts) * 5.0 if shorts else 0.0

        results = {
            "correlation_breakdown": loss_breakdown,
            "demonetization": loss_demo,
            "ilfs_crisis": loss_ilfs,
            "covid_circuit": loss_covid,
            "systemic_overnight_gap": loss_gap,
            "short_squeeze": loss_squeeze,
        }

        worst_case = min(results.values())
        passed = worst_case >= max_tolerable_loss

        if not passed:
            logger.warning(
                "Stress test FAILED | Worst Case: %.2f%% scenario: %s exceeds tolerance of %.2f%%",
                worst_case * 100,
                [k for k, v in results.items() if v == worst_case][0],
                max_tolerable_loss * 100,
            )
        else:
            logger.info("Stress test PASSED | Worst Case Scenario Loss: %.2f%%", worst_case * 100)

        return passed, results
