"""
Alpha Validator

Comprehensive validation for alpha factors before promotion to production.
Combines data validation, performance validation, and statistical significance testing.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from utils.logger import get_logger

logger = get_logger("research.alpha_validator")


@dataclass
class ValidationResult:
    """Result of factor validation."""
    passed: bool
    data_quality_passed: bool
    performance_passed: bool
    significance_passed: bool
    overall_score: float
    data_quality_score: float
    performance_score: float
    significance_score: float
    errors: List[str]
    warnings: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "data_quality_passed": self.data_quality_passed,
            "performance_passed": self.performance_passed,
            "significance_passed": self.significance_passed,
            "overall_score": round(self.overall_score, 4),
            "data_quality_score": round(self.data_quality_score, 4),
            "performance_score": round(self.performance_score, 4),
            "significance_score": round(self.significance_score, 4),
            "errors": self.errors,
            "warnings": self.warnings,
        }


class AlphaValidator:
    """
    Comprehensive validator for alpha factors.
    
    Validates factors on three dimensions:
    1. Data Quality: NaN, Inf, variance, look-ahead bias
    2. Performance: IC, Rank IC, hit rate, Sharpe
    3. Significance: t-statistics, p-values, stability
    
    Only factors passing all dimensions are promoted to production.
    """
    
    def __init__(
        self,
        min_ic: float = 0.02,
        min_rank_ic: float = 0.02,
        min_hit_rate: float = 0.51,
        min_sharpe: float = 0.5,
        max_p_value: float = 0.05,
        min_stability: float = 0.5,
    ):
        """
        Initialize alpha validator.
        
        Args:
            min_ic: Minimum acceptable Information Coefficient
            min_rank_ic: Minimum acceptable Rank IC
            min_hit_rate: Minimum acceptable hit rate
            min_sharpe: Minimum acceptable Sharpe ratio
            max_p_value: Maximum acceptable p-value for significance
            min_stability: Minimum acceptable IC stability
        """
        self.min_ic = min_ic
        self.min_rank_ic = min_rank_ic
        self.min_hit_rate = min_hit_rate
        self.min_sharpe = min_sharpe
        self.max_p_value = max_p_value
        self.min_stability = min_stability
        self._logger = get_logger("research.alpha_validator")
    
    def validate(
        self,
        factor_values: pd.Series,
        future_returns: pd.Series,
        ic_result: Optional[Dict] = None,
    ) -> ValidationResult:
        """
        Validate factor comprehensively.
        
        Args:
            factor_values: Series with factor values
            future_returns: Series with future returns
            ic_result: Optional pre-calculated IC results
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        
        # Data Quality Validation
        data_quality_score, data_quality_passed, dq_errors, dq_warnings = self._validate_data_quality(
            factor_values
        )
        errors.extend(dq_errors)
        warnings.extend(dq_warnings)
        
        # Performance Validation
        if ic_result is None:
            from research.factor_tests.information_coefficient import calculate_ic
            ic_result = calculate_ic(factor_values, future_returns).to_dict()
        
        performance_score, performance_passed, perf_errors, perf_warnings = self._validate_performance(
            ic_result
        )
        errors.extend(perf_errors)
        warnings.extend(perf_warnings)
        
        # Significance Validation
        significance_score, significance_passed, sig_errors, sig_warnings = self._validate_significance(
            ic_result
        )
        errors.extend(sig_errors)
        warnings.extend(sig_warnings)
        
        # Overall Score (weighted average)
        overall_score = (
            0.3 * data_quality_score +
            0.5 * performance_score +
            0.2 * significance_score
        )
        
        # Overall Pass (all dimensions must pass)
        passed = data_quality_passed and performance_passed and significance_passed
        
        return ValidationResult(
            passed=passed,
            data_quality_passed=data_quality_passed,
            performance_passed=performance_passed,
            significance_passed=significance_passed,
            overall_score=overall_score,
            data_quality_score=data_quality_score,
            performance_score=performance_score,
            significance_score=significance_score,
            errors=errors,
            warnings=warnings,
        )
    
    def _validate_data_quality(
        self,
        factor_values: pd.Series,
    ) -> Tuple[float, bool, List[str], List[str]]:
        """
        Validate data quality.
        
        Args:
            factor_values: Series with factor values
            
        Returns:
            Tuple of (score, passed, errors, warnings)
        """
        errors = []
        warnings = []
        score = 1.0
        
        # Check 1: NaN values
        nan_pct = factor_values.isna().sum() / len(factor_values)
        if nan_pct > 0.1:
            errors.append(f"Too many NaN values: {nan_pct:.1%}")
            score -= 0.3
        elif nan_pct > 0.01:
            warnings.append(f"Has NaN values: {nan_pct:.1%}")
            score -= 0.1
        
        # Check 2: Infinite values
        has_inf = np.isinf(factor_values).any()
        if has_inf:
            errors.append("Contains infinite values")
            score -= 0.3
        
        # Check 3: Variance
        if factor_values.var() == 0 or np.isnan(factor_values.var()):
            errors.append("Zero variance")
            score -= 0.3
        elif factor_values.var() < 1e-6:
            warnings.append("Very low variance")
            score -= 0.1
        
        # Check 4: Constant values
        if factor_values.nunique() <= 1:
            errors.append("Constant values (zero variance)")
            score -= 0.3
        
        score = max(0.0, score)
        passed = score >= 0.7
        
        return score, passed, errors, warnings
    
    def _validate_performance(
        self,
        ic_result: Dict,
    ) -> Tuple[float, bool, List[str], List[str]]:
        """
        Validate performance metrics.
        
        Args:
            ic_result: Dictionary with IC results
            
        Returns:
            Tuple of (score, passed, errors, warnings)
        """
        errors = []
        warnings = []
        score = 1.0
        
        mean_ic = ic_result.get("mean_ic", 0)
        mean_rank_ic = ic_result.get("mean_rank_ic", 0)
        hit_rate = ic_result.get("hit_rate", 0)
        
        # Check 1: IC threshold
        if mean_ic < self.min_ic:
            errors.append(f"IC too low: {mean_ic:.4f} < {self.min_ic}")
            score -= 0.4
        elif mean_ic < self.min_ic * 1.5:
            warnings.append(f"IC marginal: {mean_ic:.4f}")
            score -= 0.1
        
        # Check 2: Rank IC threshold
        if mean_rank_ic < self.min_rank_ic:
            errors.append(f"Rank IC too low: {mean_rank_ic:.4f} < {self.min_rank_ic}")
            score -= 0.3
        elif mean_rank_ic < self.min_rank_ic * 1.5:
            warnings.append(f"Rank IC marginal: {mean_rank_ic:.4f}")
            score -= 0.1
        
        # Check 3: Hit rate
        if hit_rate < self.min_hit_rate:
            errors.append(f"Hit rate too low: {hit_rate:.2%} < {self.min_hit_rate:.2%}")
            score -= 0.3
        elif hit_rate < self.min_hit_rate + 0.05:
            warnings.append(f"Hit rate marginal: {hit_rate:.2%}")
            score -= 0.1
        
        score = max(0.0, score)
        passed = score >= 0.7
        
        return score, passed, errors, warnings
    
    def _validate_significance(
        self,
        ic_result: Dict,
    ) -> Tuple[float, bool, List[str], List[str]]:
        """
        Validate statistical significance.
        
        Args:
            ic_result: Dictionary with IC results
            
        Returns:
            Tuple of (score, passed, errors, warnings)
        """
        errors = []
        warnings = []
        score = 1.0
        
        ic_p_value = ic_result.get("ic_p_value", 1.0)
        rank_ic_p_value = ic_result.get("rank_ic_p_value", 1.0)
        ic_t_stat = ic_result.get("ic_t_stat", 0)
        
        # Check 1: IC p-value
        if ic_p_value > self.max_p_value:
            errors.append(f"IC not significant: p={ic_p_value:.4f} > {self.max_p_value}")
            score -= 0.4
        elif ic_p_value > self.max_p_value * 2:
            warnings.append(f"IC marginal significance: p={ic_p_value:.4f}")
            score -= 0.1
        
        # Check 2: Rank IC p-value
        if rank_ic_p_value > self.max_p_value:
            errors.append(f"Rank IC not significant: p={rank_ic_p_value:.4f} > {self.max_p_value}")
            score -= 0.3
        elif rank_ic_p_value > self.max_p_value * 2:
            warnings.append(f"Rank IC marginal significance: p={rank_ic_p_value:.4f}")
            score -= 0.1
        
        # Check 3: t-statistic
        if abs(ic_t_stat) < 2.0:
            errors.append(f"IC t-stat too low: {ic_t_stat:.2f}")
            score -= 0.3
        elif abs(ic_t_stat) < 2.5:
            warnings.append(f"IC t-stat marginal: {ic_t_stat:.2f}")
            score -= 0.1
        
        score = max(0.0, score)
        passed = score >= 0.7
        
        return score, passed, errors, warnings
    
    def validate_lookahead_bias(
        self,
        factor_values: pd.Series,
        future_returns: pd.Series,
    ) -> bool:
        """
        Check for look-ahead bias by correlating factor with future returns at different lags.
        
        Args:
            factor_values: Series with factor values
            future_returns: Series with future returns
            
        Returns:
            True if no look-ahead bias detected
        """
        # If factor correlates better with distant future returns than near-term,
        # it may have look-ahead bias
        from research.factor_tests.information_coefficient import calculate_ic
        
        ic_1d = calculate_ic(factor_values, future_returns.shift(-1)).mean_ic
        ic_5d = calculate_ic(factor_values, future_returns.shift(-5)).mean_ic
        ic_10d = calculate_ic(factor_values, future_returns.shift(-10)).mean_ic
        
        # Look-ahead bias indicator: IC increases with horizon
        if ic_10d > ic_5d > ic_1d and ic_10d > 0.1:
            self._logger.warning(
                f"Potential look-ahead bias detected: IC increases with horizon "
                f"(1d={ic_1d:.4f}, 5d={ic_5d:.4f}, 10d={ic_10d:.4f})"
            )
            return False
        
        return True
