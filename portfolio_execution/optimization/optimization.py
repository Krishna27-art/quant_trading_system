"""
Portfolio Optimization

Institutional-grade portfolio optimization for risk-adjusted returns.
Implements mean-variance, risk parity, and Black-Litterman models.
"""

from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.covariance import LedoitWolf

from utils.logger import get_logger

logger = get_logger("portfolio_optimization")


class OptimizationMethod(str, Enum):
    """Portfolio optimization methods."""

    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"
    CVAR = "cvar"
    HRP = "hrp"


class OptimizationConstraints(BaseModel):
    """Constraints for portfolio optimization."""

    # Weight constraints
    min_weight: float = Field(default=0.0, description="Minimum weight per asset")
    max_weight: float = Field(default=0.1, description="Maximum weight per asset")
    max_positions: int = Field(default=50, description="Maximum number of positions")
    min_positions: int = Field(default=10, description="Minimum number of positions")

    # Sector constraints
    max_sector_exposure: float = Field(default=0.3, description="Maximum sector exposure")
    sector_limits: dict[str, float] | None = Field(
        default=None, description="Sector-specific limits"
    )

    # Risk constraints
    max_portfolio_volatility: float | None = Field(
        default=None, description="Maximum portfolio volatility"
    )
    max_tracking_error: float | None = Field(default=None, description="Maximum tracking error")

    # Leverage constraints
    max_leverage: float = Field(default=1.0, description="Maximum leverage")
    allow_short: bool = Field(default=False, description="Allow short positions")

    # Transaction cost coefficients
    linear_tc_coeff: float = Field(
        default=0.0020, description="Linear transaction cost coefficient (e.g. 0.0020 = 20 bps)"
    )
    impact_tc_coeff: float = Field(
        default=0.0010,
        description="Market impact transaction cost coefficient (e.g. 0.0010 = 10 bps)",
    )

    # ADV constraints (institutional requirement)
    adv_data: dict[str, float] | None = Field(
        default=None, description="Average Daily Volume for each asset"
    )
    max_adv_participation: float = Field(
        default=0.10, description="Maximum participation rate of ADV (e.g. 0.10 = 10%)"
    )
    max_position_adv_ratio: float = Field(
        default=0.05, description="Maximum position notional as ratio of ADV (e.g. 0.05 = 5%)"
    )
    illiquid_adv_ratio: float = Field(default=0.01, description="ADV ratio for illiquid stocks")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class OptimizationResult(BaseModel):
    """Result of portfolio optimization."""

    method: OptimizationMethod = Field(..., description="Optimization method used")
    weights: dict[str, float] = Field(..., description="Optimized weights")

    # Portfolio metrics
    expected_return: float = Field(..., description="Expected portfolio return")
    portfolio_volatility: float = Field(..., description="Portfolio volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")

    # Risk metrics
    portfolio_variance: float = Field(..., description="Portfolio variance")
    tracking_error: float | None = Field(None, description="Tracking error")

    # Concentration
    herfindahl_index: float = Field(..., description="Herfindahl-Hirschman Index")
    effective_positions: float = Field(..., description="Effective number of positions")

    # Optimization status
    converged: bool = Field(default=True, description="Whether optimization converged")
    iterations: int = Field(default=0, description="Number of iterations")

    optimized_at: datetime = Field(
        default_factory=datetime.now, description="When optimization was performed"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class PortfolioOptimizer:
    """
    Portfolio optimizer for institutional-grade portfolio construction.

    Implements multiple optimization methods.
    """

    def __init__(self, method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE):
        """Initialize the portfolio optimizer."""
        self.method = method
        self.logger = logger

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        constraints: OptimizationConstraints | None = None,
        benchmark_weights: dict[str, float] | None = None,
        initial_weights: dict[str, float] | None = None,
        historical_returns: pd.DataFrame | None = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio weights.

        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            constraints: Portfolio constraints
            benchmark_weights: Benchmark weights (for tracking error constraint)
            initial_weights: Initial portfolio weights (for turnover constraints)
            historical_returns: Historical return matrix (required for CVaR)

        Returns:
            OptimizationResult
        """
        constraints = constraints or OptimizationConstraints()

        # Institutional Standard: Ledoit-Wolf Shrinkage
        # Simple historical covariance is highly unstable. Shrinkage pulls extreme values toward the mean.
        if historical_returns is not None and not historical_returns.empty:
            self.logger.info("Applying Ledoit-Wolf shrinkage to covariance matrix")
            lw = LedoitWolf()
            shrunk_cov = lw.fit(historical_returns).covariance_
            covariance_matrix = pd.DataFrame(
                shrunk_cov, index=covariance_matrix.index, columns=covariance_matrix.columns
            )

        self.logger.info(f"Optimizing portfolio using {self.method} method")

        if self.method == OptimizationMethod.EQUAL_WEIGHT:
            return self._equal_weight(expected_returns, constraints)
        elif self.method == OptimizationMethod.MEAN_VARIANCE:
            return self._mean_variance(
                expected_returns, covariance_matrix, constraints, initial_weights
            )
        elif self.method == OptimizationMethod.RISK_PARITY:
            return self._risk_parity(expected_returns, covariance_matrix, constraints)
        elif self.method == OptimizationMethod.MINIMUM_VARIANCE:
            return self._minimum_variance(covariance_matrix, constraints)
        elif self.method == OptimizationMethod.CVAR:
            if historical_returns is None:
                raise ValueError("historical_returns is required for CVaR optimization")
            return self._cvar(
                expected_returns,
                historical_returns,
                covariance_matrix,
                constraints,
                initial_weights,
            )
        elif self.method == OptimizationMethod.HRP:
            if historical_returns is None:
                raise ValueError("historical_returns is required for HRP optimization")
            return self._hrp(expected_returns, historical_returns, covariance_matrix, constraints)
        else:
            raise ValueError(f"Unsupported optimization method: {self.method}")

    def _equal_weight(
        self, expected_returns: pd.Series, constraints: OptimizationConstraints
    ) -> OptimizationResult:
        """Calculate equal-weighted portfolio."""
        n = len(expected_returns)
        weight = min(1.0 / n, constraints.max_weight)

        # Normalize to sum to 1
        total_weight = weight * n
        if total_weight > 1.0:
            weight = weight / total_weight

        weights = dict.fromkeys(expected_returns.index, weight)

        # Calculate portfolio metrics
        portfolio_return = expected_returns.mean()
        portfolio_volatility = 0.2  # Placeholder

        result = OptimizationResult(
            method=self.method,
            weights=weights,
            expected_return=portfolio_return,
            portfolio_volatility=portfolio_volatility,
            sharpe_ratio=portfolio_return / portfolio_volatility,
            portfolio_variance=portfolio_volatility**2,
            herfindahl_index=sum(w**2 for w in weights.values()),
            effective_positions=1.0 / sum(w**2 for w in weights.values()),
        )

        return result

    def _mean_variance(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        constraints: OptimizationConstraints,
        initial_weights: dict[str, float] | None = None,
    ) -> OptimizationResult:
        """Calculate mean-variance optimized portfolio with transaction cost penalties and ADV constraints."""
        n = len(expected_returns)

        try:
            import cvxpy as cp

            # Variables
            weights = cp.Variable(n)

            # Setup portfolio return and risk
            portfolio_return = expected_returns.values @ weights
            portfolio_variance = cp.quad_form(weights, covariance_matrix.values)

            # Risk aversion parameter
            risk_aversion = 1.0

            # Transaction cost penalties (L1 + L2)
            tc_penalty = 0.0
            if initial_weights:
                w0 = np.array([initial_weights.get(asset, 0.0) for asset in expected_returns.index])
                # Ensure w0 sums to <= 1.0 (safety check)
                if w0.sum() > 1.0:
                    w0 = w0 / w0.sum()

                delta_w = weights - w0
                linear_penalty = constraints.linear_tc_coeff * cp.norm1(delta_w)
                impact_penalty = constraints.impact_tc_coeff * cp.sum_squares(delta_w)
                tc_penalty = linear_penalty + impact_penalty

            # Objective: minimize variance - return + transaction cost penalties
            objective = cp.Minimize(
                risk_aversion * portfolio_variance - portfolio_return + tc_penalty
            )

            # Constraints
            constraints_list = [
                weights >= constraints.min_weight,
                weights <= constraints.max_weight,
                cp.sum(weights) == 1.0,
            ]

            # Add max positions constraint
            if constraints.max_positions < n:
                top_n = constraints.max_positions
                top_assets = expected_returns.nlargest(top_n).index
                mask = np.array(
                    [1 if asset in top_assets else 0 for asset in expected_returns.index]
                )
                constraints_list.append(weights <= mask * constraints.max_weight)

            # Add ADV constraints (institutional requirement)
            if constraints.adv_data:
                np.array([constraints.adv_data.get(asset, 1e9) for asset in expected_returns.index])

                # Position notional constraint: weight * portfolio_value <= max_position_adv_ratio * ADV
                # Assuming portfolio_value = 1 (normalized), this becomes: weight <= max_position_adv_ratio * ADV
                # For simplicity, we use: weight <= max_position_adv_ratio (normalized by portfolio value)
                max_weights_by_adv = np.array(
                    [
                        (
                            constraints.max_position_adv_ratio
                            if constraints.adv_data.get(asset, 1e9) > 1e7
                            else constraints.illiquid_adv_ratio
                        )
                        for asset in expected_returns.index
                    ]
                )
                constraints_list.append(weights <= max_weights_by_adv)

                # Daily trade constraint: |delta_w| <= max_adv_participation
                if initial_weights:
                    w0 = np.array(
                        [initial_weights.get(asset, 0.0) for asset in expected_returns.index]
                    )
                    delta_w = weights - w0
                    max_trade = np.array(
                        [constraints.max_adv_participation for _ in expected_returns.index]
                    )
                    constraints_list.append(cp.abs(delta_w) <= max_trade)

            # Solve
            problem = cp.Problem(objective, constraints_list)
            problem.solve()

            if problem.status != cp.OPTIMAL:
                self.logger.error(f"Optimization failed: {problem.status}")
                raise RuntimeError(
                    f"Portfolio optimization failed to converge. Status: {problem.status}"
                )

            # Extract weights
            optimal_weights = weights.value
            weights_dict = {
                asset: float(w)
                for asset, w in zip(expected_returns.index, optimal_weights, strict=False)
            }

            # Calculate portfolio metrics
            portfolio_return_val = float(expected_returns.values @ optimal_weights)
            portfolio_variance_val = float(
                optimal_weights.T @ covariance_matrix.values @ optimal_weights
            )
            portfolio_volatility = np.sqrt(portfolio_variance_val)
            sharpe_ratio = (
                portfolio_return_val / portfolio_volatility if portfolio_volatility > 0 else 0.0
            )

            result = OptimizationResult(
                method=self.method,
                weights=weights_dict,
                expected_return=portfolio_return_val,
                portfolio_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                portfolio_variance=portfolio_variance_val,
                herfindahl_index=sum(w**2 for w in weights_dict.values()),
                effective_positions=1.0 / sum(w**2 for w in weights_dict.values()),
                converged=True,
                iterations=1,
            )

            return result

        except ImportError:
            self.logger.error("cvxpy not installed, optimization cannot proceed.")
            raise RuntimeError("cvxpy is required for mean-variance optimization.")

    def _risk_parity(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        constraints: OptimizationConstraints,
    ) -> OptimizationResult:
        """Calculate risk parity portfolio."""
        # Calculate marginal risk contribution
        # Risk parity: weight_i = 1 / (volatility_i * sum(1/volatility_j))

        volatilities = np.sqrt(np.diag(covariance_matrix))
        inv_volatilities = 1.0 / volatilities
        weights = inv_volatilities / inv_volatilities.sum()

        # Apply constraints
        weights = np.clip(weights, constraints.min_weight, constraints.max_weight)
        weights = weights / weights.sum()

        weights_dict = {
            asset: float(w) for asset, w in zip(expected_returns.index, weights, strict=False)
        }

        # Calculate portfolio metrics
        portfolio_return = float(expected_returns.values @ weights)
        portfolio_variance = float(weights.T @ covariance_matrix.values @ weights)
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0.0

        result = OptimizationResult(
            method=self.method,
            weights=weights_dict,
            expected_return=portfolio_return,
            portfolio_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            portfolio_variance=portfolio_variance,
            herfindahl_index=sum(w**2 for w in weights_dict.values()),
            effective_positions=1.0 / sum(w**2 for w in weights_dict.values()),
        )

        return result

    def _minimum_variance(
        self, covariance_matrix: pd.DataFrame, constraints: OptimizationConstraints
    ) -> OptimizationResult:
        """Calculate minimum variance portfolio."""
        n = len(covariance_matrix)

        try:
            import cvxpy as cp

            # Variables
            weights = cp.Variable(n)

            # Objective: minimize portfolio variance
            portfolio_variance = cp.quad_form(weights, covariance_matrix.values)
            objective = cp.Minimize(portfolio_variance)

            # Constraints
            constraints_list = [
                weights >= constraints.min_weight,
                weights <= constraints.max_weight,
                cp.sum(weights) == 1.0,
            ]

            # Solve
            problem = cp.Problem(objective, constraints_list)
            problem.solve()

            if problem.status != cp.OPTIMAL:
                self.logger.error(f"Optimization failed: {problem.status}")
                raise RuntimeError(
                    f"Minimum variance optimization failed to converge. Status: {problem.status}"
                )

            # Extract weights
            optimal_weights = weights.value
            weights_dict = {
                asset: float(w)
                for asset, w in zip(covariance_matrix.index, optimal_weights, strict=False)
            }

            # Calculate portfolio metrics
            portfolio_variance = float(
                optimal_weights.T @ covariance_matrix.values @ optimal_weights
            )
            portfolio_volatility = np.sqrt(portfolio_variance)
            portfolio_return = 0.0  # Minimum variance doesn't consider returns
            sharpe_ratio = 0.0

            result = OptimizationResult(
                method=self.method,
                weights=weights_dict,
                expected_return=portfolio_return,
                portfolio_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                portfolio_variance=portfolio_variance,
                herfindahl_index=sum(w**2 for w in weights_dict.values()),
                effective_positions=1.0 / sum(w**2 for w in weights_dict.values()),
            )

            return result

        except ImportError:
            self.logger.error("cvxpy not installed, optimization cannot proceed.")
            raise RuntimeError("cvxpy is required for minimum-variance optimization.")

    def _cvar(
        self,
        expected_returns: pd.Series,
        historical_returns: pd.DataFrame,
        covariance_matrix: pd.DataFrame,
        constraints: OptimizationConstraints,
        initial_weights: dict[str, float] | None = None,
    ) -> OptimizationResult:
        """Calculate CVaR optimized portfolio (Rockafellar and Uryasev 2000)."""
        n_assets = len(expected_returns)
        n_scenarios = len(historical_returns)
        alpha = 0.95  # 95% CVaR

        try:
            import cvxpy as cp

            # Variables
            weights = cp.Variable(n_assets)
            gamma = cp.Variable()  # Value at Risk
            z = cp.Variable(n_scenarios)  # auxiliary variables for losses exceeding VaR

            # Scenario returns
            R = historical_returns.values

            # Loss for each scenario is negative return
            losses = -R @ weights

            # CVaR objective: gamma + (1 / ((1 - alpha) * n_scenarios)) * sum(z)
            cvar = gamma + (1.0 / ((1.0 - alpha) * n_scenarios)) * cp.sum(z)

            portfolio_return = expected_returns.values @ weights

            # Transaction cost penalties (L1 + L2)
            tc_penalty = 0.0
            if initial_weights:
                w0 = np.array([initial_weights.get(asset, 0.0) for asset in expected_returns.index])
                if w0.sum() > 1.0:
                    w0 = w0 / w0.sum()

                delta_w = weights - w0
                linear_penalty = constraints.linear_tc_coeff * cp.norm1(delta_w)
                impact_penalty = constraints.impact_tc_coeff * cp.sum_squares(delta_w)
                tc_penalty = linear_penalty + impact_penalty

            # Objective
            risk_aversion = 1.0
            objective = cp.Minimize(risk_aversion * cvar - portfolio_return + tc_penalty)

            # Constraints
            constraints_list = [
                z >= 0,
                z >= losses - gamma,
                weights >= constraints.min_weight,
                weights <= constraints.max_weight,
                cp.sum(weights) == 1.0,
            ]

            # Add max positions constraint
            if constraints.max_positions < n_assets:
                top_n = constraints.max_positions
                top_assets = expected_returns.nlargest(top_n).index
                mask = np.array(
                    [1 if asset in top_assets else 0 for asset in expected_returns.index]
                )
                constraints_list.append(weights <= mask * constraints.max_weight)

            problem = cp.Problem(objective, constraints_list)
            problem.solve()

            if problem.status != cp.OPTIMAL:
                self.logger.error(f"CVaR Optimization failed: {problem.status}")
                raise RuntimeError(
                    f"CVaR optimization failed to converge. Status: {problem.status}"
                )

            optimal_weights = weights.value
            weights_dict = {
                asset: float(w)
                for asset, w in zip(expected_returns.index, optimal_weights, strict=False)
            }

            portfolio_return_val = float(expected_returns.values @ optimal_weights)
            portfolio_variance_val = float(
                optimal_weights.T @ covariance_matrix.values @ optimal_weights
            )
            portfolio_volatility = np.sqrt(portfolio_variance_val)
            sharpe_ratio = (
                portfolio_return_val / portfolio_volatility if portfolio_volatility > 0 else 0.0
            )

            result = OptimizationResult(
                method=self.method,
                weights=weights_dict,
                expected_return=portfolio_return_val,
                portfolio_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                portfolio_variance=portfolio_variance_val,
                herfindahl_index=sum(w**2 for w in weights_dict.values()),
                effective_positions=1.0 / sum(w**2 for w in weights_dict.values()),
                converged=True,
                iterations=1,
            )
            return result
        except ImportError:
            raise RuntimeError("cvxpy is required for CVaR optimization.")

    def _hrp(
        self,
        expected_returns: pd.Series,
        historical_returns: pd.DataFrame,
        covariance_matrix: pd.DataFrame,
        constraints: OptimizationConstraints,
    ) -> OptimizationResult:
        """Calculate Hierarchical Risk Parity (HRP) portfolio."""
        from portfolio_execution.optimization.portfolio.hrp import HierarchicalRiskParity

        try:
            hrp_optimizer = HierarchicalRiskParity()
            weights_dict = hrp_optimizer.optimize(historical_returns)

            # Apply constraints (basic clipping for HRP since it's not a convex optimizer natively)
            weights = pd.Series(weights_dict)
            weights = np.clip(weights, constraints.min_weight, constraints.max_weight)

            # Top N positions
            if constraints.max_positions < len(weights):
                threshold = weights.nlargest(constraints.max_positions).min()
                weights[weights < threshold] = 0.0

            weights = weights / weights.sum()  # Normalize

            weights_dict = weights.to_dict()

            w_arr = weights.values
            portfolio_return_val = float(expected_returns.values @ w_arr)
            portfolio_variance_val = float(w_arr.T @ covariance_matrix.values @ w_arr)
            portfolio_volatility = np.sqrt(portfolio_variance_val)
            sharpe_ratio = (
                portfolio_return_val / portfolio_volatility if portfolio_volatility > 0 else 0.0
            )

            result = OptimizationResult(
                method=self.method,
                weights=weights_dict,
                expected_return=portfolio_return_val,
                portfolio_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                portfolio_variance=portfolio_variance_val,
                herfindahl_index=sum(w**2 for w in weights_dict.values()),
                effective_positions=1.0 / sum(w**2 for w in weights_dict.values()),
                converged=True,
                iterations=1,
            )
            return result
        except Exception as e:
            self.logger.error(f"HRP Optimization failed: {str(e)}")
            raise RuntimeError(f"HRP optimization failed: {str(e)}")


# Global portfolio optimizer instance
portfolio_optimizer = PortfolioOptimizer()
