"""
Expected Return Engine

Calculates expected returns for trading signals.
Estimates target, stop, and expected value based on historical performance.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("signal_engine.expected_return")


@dataclass
class ExpectedReturn:
    """Expected return calculation for a signal."""
    target_return: float
    target_probability: float
    stop_loss: float
    stop_probability: float
    expected_value: float
    best_case: float
    worst_case: float
    median_return: float
    risk_reward_ratio: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "target_return": round(self.target_return, 4),
            "target_probability": round(self.target_probability, 4),
            "stop_loss": round(self.stop_loss, 4),
            "stop_probability": round(self.stop_probability, 4),
            "expected_value": round(self.expected_value, 4),
            "best_case": round(self.best_case, 4),
            "worst_case": round(self.worst_case, 4),
            "median_return": round(self.median_return, 4),
            "risk_reward_ratio": round(self.risk_reward_ratio, 4),
        }


class ExpectedReturnEngine:
    """
    Calculates expected returns for trading signals.
    
    Estimates:
    - Target return and probability
    - Stop loss and probability
    - Expected value
    - Best/worst case scenarios
    - Risk/reward ratio
    """
    
    def __init__(
        self,
        default_target: float = 0.06,
        default_stop: float = -0.02,
        min_risk_reward: float = 1.5,
    ):
        """
        Initialize expected return engine.
        
        Args:
            default_target: Default target return (6%)
            default_stop: Default stop loss (-2%)
            min_risk_reward: Minimum risk/reward ratio
        """
        self.default_target = default_target
        self.default_stop = default_stop
        self.min_risk_reward = min_risk_reward
        self._logger = get_logger("signal_engine.expected_return")
    
    def calculate_expected_return(
        self,
        probability: float,
        historical_returns: Optional[List[float]] = None,
        custom_target: Optional[float] = None,
        custom_stop: Optional[float] = None,
    ) -> ExpectedReturn:
        """
        Calculate expected return for a signal.
        
        Args:
            probability: Signal probability (0 to 1)
            historical_returns: Optional historical returns for calibration
            custom_target: Optional custom target return
            custom_stop: Optional custom stop loss
            
        Returns:
            ExpectedReturn
        """
        # Use custom or default targets
        target = custom_target if custom_target is not None else self.default_target
        stop = custom_stop if custom_stop is not None else self.default_stop
        
        # Calibrate based on historical data if available
        if historical_returns:
            target, stop = self._calibrate_from_history(historical_returns, target, stop)
        
        # Calculate probabilities based on signal probability
        target_prob = probability
        stop_prob = 1.0 - probability
        
        # Calculate expected value
        expected_value = (target * target_prob) + (stop * stop_prob)
        
        # Calculate best/worst cases
        best_case = target * 1.5  # 50% better than target
        worst_case = stop * 1.5  # 50% worse than stop
        
        # Calculate median return
        median_return = (target + stop) / 2
        
        # Calculate risk/reward ratio
        risk_reward_ratio = abs(target / stop) if stop != 0 else 0.0
        
        return ExpectedReturn(
            target_return=target,
            target_probability=target_prob,
            stop_loss=stop,
            stop_probability=stop_prob,
            expected_value=expected_value,
            best_case=best_case,
            worst_case=worst_case,
            median_return=median_return,
            risk_reward_ratio=risk_reward_ratio,
        )
    
    def _calibrate_from_history(
        self,
        historical_returns: List[float],
        default_target: float,
        default_stop: float,
    ) -> tuple:
        """
        Calibrate targets based on historical returns.
        
        Args:
            historical_returns: List of historical returns
            default_target: Default target return
            default_stop: Default stop loss
            
        Returns:
            Tuple of (target, stop)
        """
        if not historical_returns:
            return default_target, default_stop
        
        returns = np.array(historical_returns)
        
        # Calculate percentiles
        p75 = np.percentile(returns, 75)
        p25 = np.percentile(returns, 25)
        
        # Use historical percentiles if reasonable
        target = max(p75, default_target * 0.5) if p75 > 0 else default_target
        stop = min(p25, default_stop * 0.5) if p25 < 0 else default_stop
        
        return target, stop
    
    def is_trade_worthwhile(self, expected_return: ExpectedReturn) -> bool:
        """
        Check if trade meets minimum criteria.
        
        Args:
            expected_return: ExpectedReturn to validate
            
        Returns:
            True if trade is worthwhile
        """
        # Positive expected value
        if expected_return.expected_value <= 0:
            return False
        
        # Minimum risk/reward ratio
        if expected_return.risk_reward_ratio < self.min_risk_reward:
            return False
        
        return True
    
    def calculate_position_size(
        self,
        expected_return: ExpectedReturn,
        account_value: float,
        max_risk_per_trade: float = 0.02,
    ) -> float:
        """
        Calculate position size based on expected return.
        
        Args:
            expected_return: ExpectedReturn
            account_value: Total account value
            max_risk_per_trade: Maximum risk per trade (2%)
            
        Returns:
            Position size in currency units
        """
        # Risk amount
        risk_amount = account_value * max_risk_per_trade
        
        # Position size based on stop loss
        if abs(expected_return.stop_loss) > 0:
            position_size = risk_amount / abs(expected_return.stop_loss)
        else:
            position_size = account_value * 0.1  # Default 10%
        
        return position_size
    
    def get_holding_period(
        self,
        expected_return: ExpectedReturn,
        volatility: float,
    ) -> tuple:
        """
        Estimate holding period based on expected return and volatility.
        
        Args:
            expected_return: ExpectedReturn
            volatility: Annualized volatility
            
        Returns:
            Tuple of (min_days, max_days, expected_days)
        """
        # Rough estimate: time to reach target at current volatility
        if volatility > 0:
            expected_days = int(abs(expected_return.target_return) / (volatility / np.sqrt(252)))
            expected_days = max(1, min(expected_days, 30))  # 1-30 days
        else:
            expected_days = 5
        
        min_days = max(1, expected_days // 2)
        max_days = expected_days * 2
        
        return min_days, max_days, expected_days


def calculate_expected_return(
    probability: float,
    historical_returns: Optional[List[float]] = None,
    custom_target: Optional[float] = None,
    custom_stop: Optional[float] = None,
) -> ExpectedReturn:
    """
    Convenience function to calculate expected return.
    
    Args:
        probability: Signal probability
        historical_returns: Optional historical returns
        custom_target: Optional custom target
        custom_stop: Optional custom stop
        
    Returns:
        ExpectedReturn
    """
    engine = ExpectedReturnEngine()
    return engine.calculate_expected_return(
        probability,
        historical_returns,
        custom_target,
        custom_stop,
    )
