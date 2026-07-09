"""
Dynamic Weighting

Dynamically weights evidence based on current market conditions.
Adjusts weights based on historical quality, regime, and recent performance.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

from meta_alpha.evidence_engine.evidence import Evidence
from meta_alpha.evidence_weighting.quality_score import QualityScore
from research.interactions.market_context.market_context import MarketContext
from utils.logger import get_logger

logger = get_logger("meta_alpha.evidence_weighting")


@dataclass
class WeightingResult:
    """Result of dynamic weighting."""
    weights: Dict[str, float]
    normalization_factor: float
    regime_adjustments: Dict[str, float]
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate weighting result.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check sum of weights is approximately 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Sum of weights must be 1.0, got {total_weight}")
        
        # Check all weights are between 0 and 1
        for evidence_id, weight in self.weights.items():
            if not (0.0 <= weight <= 1.0):
                errors.append(f"Weight for {evidence_id} must be between 0 and 1, got {weight}")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "normalization_factor": round(self.normalization_factor, 4),
            "regime_adjustments": {k: round(v, 4) for k, v in self.regime_adjustments.items()},
        }


class DynamicWeighting:
    """
    Dynamically weights evidence based on current market conditions.
    
    Considers:
    - Historical quality scores
    - Current market regime
    - Recent performance
    - Missing data penalties
    """
    
    def __init__(
        self,
        quality_weight: float = 0.4,
        regime_weight: float = 0.3,
        recent_performance_weight: float = 0.2,
        missing_data_weight: float = 0.1,
    ):
        """
        Initialize dynamic weighting.
        
        Args:
            quality_weight: Weight for historical quality
            regime_weight: Weight for regime matching
            recent_performance_weight: Weight for recent performance
            missing_data_weight: Weight for missing data penalty
        """
        self.quality_weight = quality_weight
        self.regime_weight = regime_weight
        self.recent_performance_weight = recent_performance_weight
        self.missing_data_weight = missing_data_weight
        self._logger = get_logger("meta_alpha.evidence_weighting")
    
    def calculate_weights(
        self,
        evidence_list: List[Evidence],
        quality_scores: Dict[str, QualityScore],
        current_context: Optional[MarketContext] = None,
        recent_performance: Optional[Dict[str, float]] = None,
    ) -> WeightingResult:
        """
        Calculate dynamic weights for evidence.
        
        Args:
            evidence_list: List of Evidence objects
            quality_scores: Dictionary mapping evidence IDs to QualityScore
            current_context: Current market context
            recent_performance: Optional dictionary of recent performance scores
            
        Returns:
            WeightingResult
        """
        if not evidence_list:
            return WeightingResult(
                weights={},
                normalization_factor=1.0,
                regime_adjustments={},
            )
        
        # Calculate base scores
        base_scores = {}
        regime_adjustments = {}
        
        for evidence in evidence_list:
            evidence_id = f"{evidence.source}_{evidence.factor_name}"
            
            # Get quality score
            quality_score = quality_scores.get(evidence_id, QualityScore(
                data_quality=50.0,
                historical_ic=50.0,
                historical_sharpe=50.0,
                regime_stability=50.0,
                missing_values=100.0,
                overall_score=50.0,
            ))
            
            # Calculate quality component
            quality_component = quality_score.overall_score / 100.0
            
            # Calculate regime component
            regime_component = self._calculate_regime_component(evidence, current_context)
            regime_adjustments[evidence_id] = regime_component
            
            # Calculate recent performance component
            recent_component = self._calculate_recent_component(
                evidence_id,
                recent_performance,
            )
            
            # Calculate missing data component
            missing_component = quality_score.missing_values / 100.0
            
            # Combine components
            base_score = (
                quality_component * self.quality_weight +
                regime_component * self.regime_weight +
                recent_component * self.recent_performance_weight +
                missing_component * self.missing_data_weight
            )
            
            base_scores[evidence_id] = base_score
        
        # Normalize weights to sum to 1.0
        total_score = sum(base_scores.values())
        
        if total_score > 0:
            normalization_factor = 1.0 / total_score
            weights = {
                evidence_id: score * normalization_factor
                for evidence_id, score in base_scores.items()
            }
        else:
            # Equal weights if all scores are zero
            normalization_factor = 1.0 / len(base_scores)
            weights = {
                evidence_id: normalization_factor
                for evidence_id in base_scores.keys()
            }
        
        return WeightingResult(
            weights=weights,
            normalization_factor=normalization_factor,
            regime_adjustments=regime_adjustments,
        )
    
    def _calculate_regime_component(
        self,
        evidence: Evidence,
        current_context: Optional[MarketContext],
    ) -> float:
        """
        Calculate regime matching component.
        
        Args:
            evidence: Evidence object
            current_context: Current market context
            
        Returns:
            Regime component score (0-1)
        """
        if current_context is None:
            return 0.5  # Neutral if no context
        
        # Check if evidence category matches current regime
        regime_score = 0.5
        
        # Trend evidence in trending market
        if evidence.category == "trend":
            if current_context.trend in ["bull", "bear"]:
                regime_score = 0.8
            else:
                regime_score = 0.3
        
        # Momentum evidence in high volatility
        if evidence.category == "momentum":
            if current_context.volatility == "high":
                regime_score = 0.7
            elif current_context.volatility == "low":
                regime_score = 0.4
            else:
                regime_score = 0.5
        
        # Options sentiment evidence
        if evidence.category == "options":
            if current_context.options_sentiment == evidence.signal_direction:
                regime_score = 0.8
            else:
                regime_score = 0.3
        
        # Liquidity evidence
        if evidence.category == "liquidity":
            if current_context.liquidity == "high":
                regime_score = 0.7
            else:
                regime_score = 0.4
        
        return regime_score
    
    def _calculate_recent_component(
        self,
        evidence_id: str,
        recent_performance: Optional[Dict[str, float]],
    ) -> float:
        """
        Calculate recent performance component.
        
        Args:
            evidence_id: Evidence identifier
            recent_performance: Dictionary of recent performance scores
            
        Returns:
            Recent performance component (0-1)
        """
        if recent_performance is None or evidence_id not in recent_performance:
            return 0.5  # Neutral if no recent data
        
        performance = recent_performance[evidence_id]
        
        # Normalize performance to 0-1 range
        # Assuming performance is in range [-0.2, 0.2]
        normalized = (performance + 0.2) / 0.4
        return max(0.0, min(1.0, normalized))
    
    def adjust_for_correlation(
        self,
        weights: Dict[str, float],
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        """
        Adjust weights to account for correlation between evidence.
        
        Args:
            weights: Original weights
            correlation_matrix: Correlation matrix between evidence
            
        Returns:
            Adjusted weights
        """
        if correlation_matrix is None:
            return weights
        
        # Simple adjustment: reduce weight of highly correlated evidence
        adjusted_weights = weights.copy()
        
        for evidence_id_1 in weights.keys():
            for evidence_id_2 in weights.keys():
                if evidence_id_1 == evidence_id_2:
                    continue
                
                correlation = correlation_matrix.get(evidence_id_1, {}).get(evidence_id_2, 0.0)
                
                # If highly correlated, reduce weight
                if abs(correlation) > 0.7:
                    adjustment_factor = 1.0 - (abs(correlation) - 0.7) * 0.5
                    adjusted_weights[evidence_id_1] *= adjustment_factor
        
        # Renormalize
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {
                k: v / total
                for k, v in adjusted_weights.items()
            }
        
        return adjusted_weights


def calculate_dynamic_weights(
    evidence_list: List[Evidence],
    quality_scores: Dict[str, QualityScore],
    current_context: Optional[MarketContext] = None,
) -> WeightingResult:
    """
    Convenience function to calculate dynamic weights.
    
    Args:
        evidence_list: List of Evidence objects
        quality_scores: Dictionary mapping evidence IDs to QualityScore
        current_context: Current market context
        
    Returns:
        WeightingResult
    """
    weighting = DynamicWeighting()
    return weighting.calculate_weights(evidence_list, quality_scores, current_context)
