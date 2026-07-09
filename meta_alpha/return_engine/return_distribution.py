"""
Return Distribution

Predicts expected return distribution instead of single point estimate.
Provides best case, median, worst case, and percentiles.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("meta_alpha.return_engine")


@dataclass
class ReturnDistribution:
    """Return distribution for a trading signal."""
    expected_return: float
    median_return: float
    best_case: float
    worst_case: float
    percentiles: Dict[str, float]
    expected_value: float
    risk_metrics: Dict[str, float]
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate return distribution.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for unreasonably extreme values
        if abs(self.expected_return) > 1.0:
            errors.append(f"Expected return too extreme: {self.expected_return}")
        if abs(self.median_return) > 1.0:
            errors.append(f"Median return too extreme: {self.median_return}")
        if abs(self.best_case) > 1.0:
            errors.append(f"Best case too extreme: {self.best_case}")
        if abs(self.worst_case) > 1.0:
            errors.append(f"Worst case too extreme: {self.worst_case}")
        
        # Check for NaN or Inf
        for name, value in [
            ("expected_return", self.expected_return),
            ("median_return", self.median_return),
            ("best_case", self.best_case),
            ("worst_case", self.worst_case),
        ]:
            if np.isnan(value) or np.isinf(value):
                errors.append(f"{name} cannot be NaN or Inf")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "expected_return": round(self.expected_return, 4),
            "median_return": round(self.median_return, 4),
            "best_case": round(self.best_case, 4),
            "worst_case": round(self.worst_case, 4),
            "percentiles": {k: round(v, 4) for k, v in self.percentiles.items()},
            "expected_value": round(self.expected_value, 4),
            "risk_metrics": {k: round(v, 4) for k, v in self.risk_metrics.items()},
        }


class ReturnDistributionEngine:
    """
    Predicts expected return distribution instead of single point estimate.
    
    Provides:
    - Expected return
    - Median return
    - Best case (90th percentile)
    - Worst case (10th percentile)
    - Full percentile distribution
    - Risk metrics
    """
    
    def __init__(
        self,
        holding_period_days: int = 5,
        risk_free_rate: float = 0.05,
    ):
        """
        Initialize return distribution engine.
        
        Args:
            holding_period_days: Expected holding period in days
            risk_free_rate: Annual risk-free rate
        """
        self.holding_period_days = holding_period_days
        self.risk_free_rate = risk_free_rate
        self._logger = get_logger("meta_alpha.return_engine")
    
    def calculate_distribution(
        self,
        probability: float,
        historical_returns: Optional[pd.Series] = None,
        volatility: Optional[float] = None,
        confidence_level: Optional[str] = None,
    ) -> ReturnDistribution:
        """
        Calculate return distribution.
        
        Args:
            probability: Probability of success
            historical_returns: Optional historical returns for calibration
            volatility: Optional volatility estimate
            confidence_level: Optional confidence level
            
        Returns:
            ReturnDistribution
        """
        # Calculate base expected return from probability
        base_return = self._probability_to_return(probability)
        
        # Adjust by volatility if available
        if volatility is not None:
            base_return *= (1.0 - volatility * 0.5)
        
        # Adjust by confidence level
        if confidence_level == "HIGH":
            base_return *= 1.2
        elif confidence_level == "LOW":
            base_return *= 0.8
        
        # Calculate distribution parameters
        if historical_returns is not None and len(historical_returns) > 30:
            # Use historical distribution
            distribution = self._historical_distribution(
                historical_returns,
                base_return,
            )
        else:
            # Use parametric distribution
            distribution = self._parametric_distribution(
                base_return,
                volatility,
            )
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(distribution)
        
        return ReturnDistribution(
            expected_return=distribution["expected"],
            median_return=distribution["median"],
            best_case=distribution["best_case"],
            worst_case=distribution["worst_case"],
            percentiles=distribution["percentiles"],
            expected_value=distribution["expected_value"],
            risk_metrics=risk_metrics,
        )
    
    def _probability_to_return(self, probability: float) -> float:
        """
        Convert probability to expected return.
        
        Args:
            probability: Probability of success (0-1)
            
        Returns:
            Expected return
        """
        # Probability > 0.5 implies positive return
        # Probability < 0.5 implies negative return
        # Probability = 0.5 implies neutral return
        
        # Convert to range [-0.1, 0.1] for typical holding period
        return (probability - 0.5) * 0.2
    
    def _historical_distribution(
        self,
        historical_returns: pd.Series,
        base_return: float,
    ) -> Dict[str, Any]:
        """
        Calculate distribution from historical returns.
        
        Args:
            historical_returns: Historical returns
            base_return: Base expected return
            
        Returns:
            Distribution parameters
        """
        # Scale historical returns to match base return
        scaling_factor = base_return / historical_returns.mean() if historical_returns.mean() != 0 else 1.0
        scaled_returns = historical_returns * scaling_factor
        
        # Calculate percentiles
        percentiles = {
            "p10": scaled_returns.quantile(0.1),
            "p25": scaled_returns.quantile(0.25),
            "p50": scaled_returns.quantile(0.5),
            "p75": scaled_returns.quantile(0.75),
            "p90": scaled_returns.quantile(0.9),
        }
        
        return {
            "expected": scaled_returns.mean(),
            "median": percentiles["p50"],
            "best_case": percentiles["p90"],
            "worst_case": percentiles["p10"],
            "percentiles": percentiles,
            "expected_value": scaled_returns.mean(),
        }
    
    def _parametric_distribution(
        self,
        base_return: float,
        volatility: Optional[float],
    ) -> Dict[str, Any]:
        """
        Calculate parametric distribution.
        
        Args:
            base_return: Base expected return
            volatility: Volatility estimate
            
        Returns:
            Distribution parameters
        """
        # Default volatility if not provided
        if volatility is None:
            volatility = 0.02  # 2% daily volatility
        
        # Annualize volatility
        annual_volatility = volatility * np.sqrt(252)
        
        # Scale to holding period
        period_volatility = annual_volatility * np.sqrt(self.holding_period_days / 252)
        
        # Generate normal distribution
        np.random.seed(42)
        samples = np.random.normal(base_return, period_volatility, 10000)
        
        # Calculate percentiles
        percentiles = {
            "p10": np.percentile(samples, 10),
            "p25": np.percentile(samples, 25),
            "p50": np.percentile(samples, 50),
            "p75": np.percentile(samples, 75),
            "p90": np.percentile(samples, 90),
        }
        
        return {
            "expected": np.mean(samples),
            "median": percentiles["p50"],
            "best_case": percentiles["p90"],
            "worst_case": percentiles["p10"],
            "percentiles": percentiles,
            "expected_value": np.mean(samples),
        }
    
    def _calculate_risk_metrics(self, distribution: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate risk metrics.
        
        Args:
            distribution: Distribution parameters
            
        Returns:
            Risk metrics
        """
        expected = distribution["expected"]
        worst_case = distribution["worst_case"]
        best_case = distribution["best_case"]
        
        # Risk/reward ratio
        if worst_case < 0:
            risk_reward = abs(expected / worst_case) if worst_case != 0 else 0.0
        else:
            risk_reward = 0.0
        
        # Downside risk
        downside_risk = abs(worst_case) if worst_case < 0 else 0.0
        
        # Upside potential
        upside_potential = best_case if best_case > 0 else 0.0
        
        # Sharpe ratio (simplified)
        excess_return = expected - (self.risk_free_rate * self.holding_period_days / 252)
        sharpe = excess_return / abs(worst_case) if worst_case != 0 else 0.0
        
        return {
            "risk_reward_ratio": risk_reward,
            "downside_risk": downside_risk,
            "upside_potential": upside_potential,
            "sharpe_ratio": sharpe,
        }
    
    def calculate_holding_period(
        self,
        probability: float,
        volatility: Optional[float] = None,
    ) -> int:
        """
        Calculate optimal holding period.
        
        Args:
            probability: Probability of success
            volatility: Volatility estimate
            
        Returns:
            Optimal holding period in days
        """
        # Higher probability = longer holding period
        # Higher volatility = shorter holding period
        
        base_period = 5  # Default 5 days
        
        if probability > 0.7:
            base_period = 7
        elif probability > 0.6:
            base_period = 5
        elif probability > 0.5:
            base_period = 3
        else:
            base_period = 1
        
        if volatility and volatility > 0.03:
            base_period = max(1, base_period - 2)
        
        return base_period


def calculate_return_distribution(
    probability: float,
    historical_returns: Optional[pd.Series] = None,
) -> ReturnDistribution:
    """
    Convenience function to calculate return distribution.
    
    Args:
        probability: Probability of success
        historical_returns: Optional historical returns
        
    Returns:
        ReturnDistribution
    """
    engine = ReturnDistributionEngine()
    return engine.calculate_distribution(probability, historical_returns)
