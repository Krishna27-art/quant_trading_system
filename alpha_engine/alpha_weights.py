"""
Alpha Weights

Manages dynamic weighting of alpha categories based on:
- Trading timeframe (intraday, swing, longterm)
- Market regime adjustments
- Historical performance adjustments

STEP 4: Weight each category
STEP 6: Historical performance adjustment
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from alpha_engine.alpha_builder import TimeFrame
from utils.logger import get_logger

logger = get_logger("alpha_engine.weights")


@dataclass
class WeightConfiguration:
    """
    Weight configuration for a specific timeframe.
    
    Weights should sum to 1.0.
    """
    timeframe: TimeFrame
    weights: Dict[str, float]  # category_name -> weight
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timeframe": self.timeframe.value,
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "description": self.description,
        }
    
    def validate(self) -> bool:
        """Validate that weights sum to approximately 1.0."""
        total = sum(self.weights.values())
        return abs(total - 1.0) < 0.01


@dataclass
class CategoryPerformance:
    """
    Historical performance metrics for a category.
    
    Used for dynamic weight adjustment based on performance.
    """
    category: str
    win_rate: float  # 0-1
    average_return: float  # decimal
    sharpe_ratio: float
    sample_size: int
    last_updated: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "win_rate": round(self.win_rate, 4),
            "average_return": round(self.average_return, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sample_size": self.sample_size,
            "last_updated": self.last_updated.isoformat(),
        }


class AlphaWeights:
    """
    Manages dynamic weighting of alpha categories.
    
    Provides base weight configurations for different timeframes
    and adjusts them based on market regime and historical performance.
    """
    
    def __init__(self):
        """Initialize Alpha Weights."""
        self._logger = logger
        
        # Base weight configurations for different timeframes
        self.base_weights = self._initialize_base_weights()
        
        # Historical performance data for categories
        self.category_performance: Dict[str, CategoryPerformance] = {}
        
        # Performance adjustment factor (0-1)
        # Higher = more aggressive adjustment based on performance
        self.performance_adjustment_factor = 0.3
    
    def get_weights(
        self,
        timeframe: TimeFrame,
        regime_adjustments: Optional[Dict[str, float]] = None,
        use_performance_adjustment: bool = True,
    ) -> Dict[str, float]:
        """
        Get weights for a specific timeframe with optional adjustments.
        
        Args:
            timeframe: Trading timeframe
            regime_adjustments: Optional regime-based weight multipliers
            use_performance_adjustment: Whether to adjust based on historical performance
            
        Returns:
            Dictionary of category_name -> weight
        """
        # Get base weights
        base = self.base_weights[timeframe].weights.copy()
        
        # Apply regime adjustments if provided
        if regime_adjustments:
            base = self._apply_regime_adjustments(base, regime_adjustments)
        
        # Apply performance adjustments if enabled
        if use_performance_adjustment and self.category_performance:
            base = self._apply_performance_adjustments(base)
        
        # Normalize to ensure sum = 1.0
        normalized = self._normalize_weights(base)
        
        self._logger.info(
            f"Weights calculated for {timeframe.value}",
            extra={"weights": {k: round(v, 4) for k, v in normalized.items()}},
        )
        
        return normalized
    
    def update_category_performance(
        self,
        category: str,
        win_rate: float,
        average_return: float,
        sharpe_ratio: float,
        sample_size: int,
    ) -> None:
        """
        Update historical performance for a category.
        
        Args:
            category: Category name
            win_rate: Win rate (0-1)
            average_return: Average return (decimal)
            sharpe_ratio: Sharpe ratio
            sample_size: Number of samples
        """
        self.category_performance[category] = CategoryPerformance(
            category=category,
            win_rate=win_rate,
            average_return=average_return,
            sharpe_ratio=sharpe_ratio,
            sample_size=sample_size,
            last_updated=datetime.now(),
        )
        
        self._logger.info(
            f"Updated performance for {category}",
            extra={
                "win_rate": round(win_rate, 4),
                "sharpe_ratio": round(sharpe_ratio, 4),
                "sample_size": sample_size,
            },
        )
    
    def get_weight_configuration(self, timeframe: TimeFrame) -> WeightConfiguration:
        """
        Get the base weight configuration for a timeframe.
        
        Args:
            timeframe: Trading timeframe
            
        Returns:
            WeightConfiguration
        """
        return self.base_weights[timeframe]
    
    def _initialize_base_weights(self) -> Dict[TimeFrame, WeightConfiguration]:
        """
        Initialize base weight configurations for different timeframes.
        
        These are research-driven starting weights that should be
        validated and optimized using walk-forward analysis.
        
        Returns:
            Dictionary of TimeFrame -> WeightConfiguration
        """
        return {
            TimeFrame.INTRADAY: WeightConfiguration(
                timeframe=TimeFrame.INTRADAY,
                weights={
                    "technical": 0.25,
                    "volume": 0.30,
                    "options": 0.25,
                    "sentiment": 0.10,
                    "macro": 0.05,
                    "fundamental": 0.03,
                    "sector": 0.02,
                },
                description="Intraday: Focus on volume, technical, and options signals",
            ),
            TimeFrame.SWING: WeightConfiguration(
                timeframe=TimeFrame.SWING,
                weights={
                    "technical": 0.30,
                    "volume": 0.20,
                    "options": 0.15,
                    "fundamental": 0.15,
                    "sector": 0.10,
                    "sentiment": 0.05,
                    "macro": 0.05,
                },
                description="Swing: Balanced approach with emphasis on trend and fundamentals",
            ),
            TimeFrame.LONGTERM: WeightConfiguration(
                timeframe=TimeFrame.LONGTERM,
                weights={
                    "fundamental": 0.35,
                    "technical": 0.20,
                    "sector": 0.15,
                    "macro": 0.15,
                    "sentiment": 0.10,
                    "volume": 0.03,
                    "options": 0.02,
                },
                description="Longterm: Fundamental and macro factors dominate",
            ),
        }
    
    def _apply_regime_adjustments(
        self,
        base_weights: Dict[str, float],
        regime_adjustments: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply regime-based weight adjustments.
        
        Args:
            base_weights: Base weights
            regime_adjustments: Category -> multiplier (e.g., {"momentum": 0.5})
            
        Returns:
            Adjusted weights
        """
        adjusted = base_weights.copy()
        
        for category, multiplier in regime_adjustments.items():
            if category in adjusted:
                adjusted[category] *= multiplier
        
        return adjusted
    
    def _apply_performance_adjustments(
        self,
        base_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Adjust weights based on historical category performance.
        
        Categories with higher win rates and Sharpe ratios get increased weights.
        
        Args:
            base_weights: Base weights
            
        Returns:
            Performance-adjusted weights
        """
        adjusted = base_weights.copy()
        
        # Calculate performance scores for each category
        performance_scores = {}
        for category, perf in self.category_performance.items():
            if category not in adjusted:
                continue
            
            # Combine win rate and Sharpe into a single score
            # Win rate (0-1) + Sharpe (typically 0-3) normalized
            score = (perf.win_rate * 0.6) + (min(perf.sharpe_ratio, 3) / 3 * 0.4)
            performance_scores[category] = score
        
        if not performance_scores:
            return adjusted
        
        # Calculate average performance
        avg_performance = np.mean(list(performance_scores.values()))
        
        # Adjust weights based on performance relative to average
        for category, score in performance_scores.items():
            if category in adjusted:
                # Calculate adjustment factor
                # If score > average, increase weight
                # If score < average, decrease weight
                relative_performance = score / avg_performance if avg_performance > 0 else 1.0
                
                # Apply adjustment with damping factor
                adjustment = 1.0 + (relative_performance - 1.0) * self.performance_adjustment_factor
                adjusted[category] *= adjustment
        
        return adjusted
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.0.
        
        Args:
            weights: Dictionary of category -> weight
            
        Returns:
            Normalized weights
        """
        total = sum(weights.values())
        
        if total == 0:
            self._logger.warning("Total weight is 0, returning equal weights")
            categories = list(weights.keys())
            equal_weight = 1.0 / len(categories) if categories else 0.0
            return {cat: equal_weight for cat in categories}
        
        return {cat: w / total for cat, w in weights.items()}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of category performance data.
        
        Returns:
            Dictionary with performance summary
        """
        if not self.category_performance:
            return {"message": "No performance data available"}
        
        summary = {
            "categories": {},
            "best_category": None,
            "worst_category": None,
        }
        
        best_sharpe = -np.inf
        worst_sharpe = np.inf
        
        for category, perf in self.category_performance.items():
            summary["categories"][category] = perf.to_dict()
            
            if perf.sharpe_ratio > best_sharpe:
                best_sharpe = perf.sharpe_ratio
                summary["best_category"] = category
            
            if perf.sharpe_ratio < worst_sharpe:
                worst_sharpe = perf.sharpe_ratio
                summary["worst_category"] = category
        
        return summary
    
    def reset_performance_data(self) -> None:
        """Reset all historical performance data."""
        self.category_performance.clear()
        self._logger.info("Performance data reset")
