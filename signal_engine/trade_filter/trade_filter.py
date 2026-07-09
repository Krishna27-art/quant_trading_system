"""
Trade Filter Engine

Filters trading signals based on quality thresholds.
Rejects weak signals and returns only high-quality opportunities.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("signal_engine.trade_filter")


@dataclass
class FilterCriteria:
    """Filter criteria for trading signals."""
    min_probability: float
    min_confidence: float
    min_expected_value: float
    min_quality: float
    min_risk_reward: float
    max_risk_per_trade: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "min_probability": round(self.min_probability, 4),
            "min_confidence": round(self.min_confidence, 4),
            "min_expected_value": round(self.min_expected_value, 4),
            "min_quality": round(self.min_quality, 2),
            "min_risk_reward": round(self.min_risk_reward, 4),
            "max_risk_per_trade": round(self.max_risk_per_trade, 4),
        }


@dataclass
class FilterResult:
    """Result of trade filtering."""
    symbol: str
    passed: bool
    rejection_reason: Optional[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "passed": self.passed,
            "rejection_reason": self.rejection_reason,
        }


class TradeFilter:
    """
    Filters trading signals based on quality thresholds.
    
    Rejects signals that:
    - Have low probability
    - Have low confidence
    - Have negative or low expected value
    - Have poor risk/reward ratio
    - Have low quality score
    - Are in wrong market regime
    - Have low liquidity
    """
    
    def __init__(
        self,
        criteria: Optional[FilterCriteria] = None,
    ):
        """
        Initialize trade filter.
        
        Args:
            criteria: Optional filter criteria
        """
        self.criteria = criteria or FilterCriteria(
            min_probability=0.6,
            min_confidence=0.5,
            min_expected_value=0.01,
            min_quality=70.0,
            min_risk_reward=1.5,
            max_risk_per_trade=0.02,
        )
        self._logger = get_logger("signal_engine.trade_filter")
    
    def filter_signal(
        self,
        signal_data: Dict,
        market_regime: Optional[str] = None,
        sector_strength: Optional[float] = None,
        liquidity: Optional[float] = None,
    ) -> FilterResult:
        """
        Filter a single trading signal.
        
        Args:
            signal_data: Signal data dictionary
            market_regime: Optional current market regime
            sector_strength: Optional sector strength score
            liquidity: Optional liquidity score
            
        Returns:
            FilterResult
        """
        symbol = signal_data.get("symbol", "unknown")
        
        # Check probability
        probability = signal_data.get("probability", 0.5)
        if probability < self.criteria.min_probability:
            return FilterResult(
                symbol=symbol,
                passed=False,
                rejection_reason=f"Probability {probability:.2%} below threshold {self.criteria.min_probability:.2%}",
            )
        
        # Check confidence
        confidence = signal_data.get("confidence", 0.5)
        if confidence < self.criteria.min_confidence:
            return FilterResult(
                symbol=symbol,
                passed=False,
                rejection_reason=f"Confidence {confidence:.2%} below threshold {self.criteria.min_confidence:.2%}",
            )
        
        # Check expected value
        expected_value = signal_data.get("expected_value", 0.0)
        if expected_value < self.criteria.min_expected_value:
            return FilterResult(
                symbol=symbol,
                passed=False,
                rejection_reason=f"Expected value {expected_value:.2%} below threshold {self.criteria.min_expected_value:.2%}",
            )
        
        # Check quality
        quality = signal_data.get("quality_score", 50.0)
        if quality < self.criteria.min_quality:
            return FilterResult(
                symbol=symbol,
                passed=False,
                rejection_reason=f"Quality {quality:.1f} below threshold {self.criteria.min_quality:.1f}",
            )
        
        # Check risk/reward
        risk_reward = signal_data.get("risk_reward_ratio", 1.0)
        if risk_reward < self.criteria.min_risk_reward:
            return FilterResult(
                symbol=symbol,
                passed=False,
                rejection_reason=f"Risk/reward {risk_reward:.2f} below threshold {self.criteria.min_risk_reward:.2f}",
            )
        
        # Check market regime if provided
        if market_regime and market_regime in ["bear", "strong_bear"]:
            return FilterResult(
                symbol=symbol,
                passed=False,
                rejection_reason=f"Market regime {market_regime} unfavorable",
            )
        
        # Check sector strength if provided
        if sector_strength is not None and sector_strength < 0.3:
            return FilterResult(
                symbol=symbol,
                passed=False,
                rejection_reason=f"Sector strength {sector_strength:.2f} too weak",
            )
        
        # Check liquidity if provided
        if liquidity is not None and liquidity < 0.5:
            return FilterResult(
                symbol=symbol,
                passed=False,
                rejection_reason=f"Liquidity {liquidity:.2f} too low",
            )
        
        return FilterResult(
            symbol=symbol,
            passed=True,
            rejection_reason=None,
        )
    
    def filter_signals(
        self,
        signals: Dict[str, Dict],
        market_regime: Optional[str] = None,
        sector_data: Optional[Dict[str, float]] = None,
        liquidity_data: Optional[Dict[str, float]] = None,
    ) -> Dict[str, FilterResult]:
        """
        Filter multiple trading signals.
        
        Args:
            signals: Dictionary mapping symbols to signal data
            market_regime: Optional current market regime
            sector_data: Optional dictionary mapping symbols to sector strength
            liquidity_data: Optional dictionary mapping symbols to liquidity
            
        Returns:
            Dictionary mapping symbols to FilterResult
        """
        results = {}
        
        for symbol, signal_data in signals.items():
            signal_data["symbol"] = symbol
            
            sector_strength = sector_data.get(symbol) if sector_data else None
            liquidity = liquidity_data.get(symbol) if liquidity_data else None
            
            result = self.filter_signal(
                signal_data,
                market_regime,
                sector_strength,
                liquidity,
            )
            
            results[symbol] = result
        
        return results
    
    def get_passed_signals(
        self,
        filter_results: Dict[str, FilterResult],
    ) -> List[str]:
        """
        Get list of symbols that passed filtering.
        
        Args:
            filter_results: Dictionary mapping symbols to FilterResult
            
        Returns:
            List of symbols that passed
        """
        return [symbol for symbol, result in filter_results.items() if result.passed]
    
    def get_rejection_summary(
        self,
        filter_results: Dict[str, FilterResult],
    ) -> Dict[str, int]:
        """
        Get summary of rejection reasons.
        
        Args:
            filter_results: Dictionary mapping symbols to FilterResult
            
        Returns:
            Dictionary mapping rejection reasons to counts
        """
        summary = {}
        
        for result in filter_results.values():
            if not result.passed and result.rejection_reason:
                reason = result.rejection_reason
                summary[reason] = summary.get(reason, 0) + 1
        
        return summary


def filter_trades(
    signals: Dict[str, Dict],
    min_probability: float = 0.6,
    min_confidence: float = 0.5,
    min_expected_value: float = 0.01,
    min_quality: float = 70.0,
    min_risk_reward: float = 1.5,
) -> Dict[str, FilterResult]:
    """
    Convenience function to filter trades.
    
    Args:
        signals: Dictionary mapping symbols to signal data
        min_probability: Minimum probability threshold
        min_confidence: Minimum confidence threshold
        min_expected_value: Minimum expected value
        min_quality: Minimum quality score
        min_risk_reward: Minimum risk/reward ratio
        
    Returns:
        Dictionary mapping symbols to FilterResult
    """
    criteria = FilterCriteria(
        min_probability=min_probability,
        min_confidence=min_confidence,
        min_expected_value=min_expected_value,
        min_quality=min_quality,
        min_risk_reward=min_risk_reward,
        max_risk_per_trade=0.02,
    )
    
    filter_engine = TradeFilter(criteria=criteria)
    return filter_engine.filter_signals(signals)
