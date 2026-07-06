"""
Statistical Validator
Provides robust statistical significance tests for backtest results.
"""

import numpy as np
from pydantic import BaseModel
from scipy.stats import t
from utils.logger import get_logger

logger = get_logger("statistical_validator")

class ValidationResult(BaseModel):
    is_valid: bool
    p_value: float
    confidence_interval: tuple[float, float]
    kelly_fraction: float | None = None
    warning_message: str | None = None

class StatisticalValidator:
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        
    def validate_returns(self, returns: list[float]) -> ValidationResult:
        """
        Validates returns using t-tests and Kelly Criterion.
        """
        if not returns or len(returns) < 30:
            return ValidationResult(
                is_valid=False, 
                p_value=1.0, 
                confidence_interval=(0.0, 0.0),
                warning_message="Insufficient data (N < 30)"
            )
            
        returns_arr = np.array(returns)
        mean = np.mean(returns_arr)
        std = np.std(returns_arr, ddof=1)
        
        if std == 0:
            return ValidationResult(
                is_valid=False, p_value=1.0, confidence_interval=(mean, mean),
                warning_message="Zero volatility"
            )
            
        n = len(returns_arr)
        t_stat = mean / (std / np.sqrt(n))
        
        p_val = 1 - t.cdf(t_stat, df=n-1)
        
        ci_lower = mean - t.ppf(1 - self.significance_level/2, df=n-1) * (std / np.sqrt(n))
        ci_upper = mean + t.ppf(1 - self.significance_level/2, df=n-1) * (std / np.sqrt(n))
        
        kelly = mean / (std**2) if std > 0 else 0
        
        is_valid = bool(p_val < self.significance_level and mean > 0)
        
        return ValidationResult(
            is_valid=is_valid,
            p_value=float(p_val),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            kelly_fraction=float(kelly),
            warning_message=None if is_valid else f"Strategy not statistically significant (p={p_val:.4f})"
        )
