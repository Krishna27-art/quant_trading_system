"""
Alpha Fusion Engine

Combines multiple alpha factors into unified evidence.
Aggregates independent signals from different factor categories.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("signal_engine.alpha_fusion")


@dataclass
class FusionResult:
    """Result of alpha fusion."""
    bullish_evidence: float
    bearish_evidence: float
    neutral_evidence: float
    missing_evidence: int
    total_evidence: int
    fusion_score: float
    category_breakdown: Dict[str, float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "bullish_evidence": round(self.bullish_evidence, 4),
            "bearish_evidence": round(self.bearish_evidence, 4),
            "neutral_evidence": round(self.neutral_evidence, 4),
            "missing_evidence": self.missing_evidence,
            "total_evidence": self.total_evidence,
            "fusion_score": round(self.fusion_score, 4),
            "category_breakdown": {k: round(v, 4) for k, v in self.category_breakdown.items()},
        }


class AlphaFusionEngine:
    """
    Combines multiple alpha factors into unified evidence.
    
    Aggregates independent signals from different factor categories:
    - Trend
    - Momentum
    - Volume
    - Volatility
    - Liquidity
    - Options
    - Fundamental
    - Macro
    - Sentiment
    - Sector
    """
    
    def __init__(
        self,
        category_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize alpha fusion engine.
        
        Args:
            category_weights: Optional weights for each category (default: equal weights)
        """
        self.category_weights = category_weights or {
            "trend": 1.0,
            "momentum": 1.0,
            "volume": 0.8,
            "volatility": 0.6,
            "liquidity": 0.7,
            "options": 0.9,
            "fundamental": 0.8,
            "macro": 0.5,
            "sentiment": 0.6,
            "sector": 0.7,
        }
        self._logger = get_logger("signal_engine.alpha_fusion")
    
    def fuse_evidence(
        self,
        factor_evidence: Dict[str, float],
        factor_categories: Dict[str, str],
    ) -> FusionResult:
        """
        Fuse evidence from multiple alpha factors.
        
        Args:
            factor_evidence: Dictionary mapping factor names to evidence scores (-1 to 1)
            factor_categories: Dictionary mapping factor names to categories
            
        Returns:
            FusionResult
        """
        # Group evidence by category
        category_evidence = self._group_by_category(factor_evidence, factor_categories)
        
        # Calculate category scores
        category_scores = {}
        for category, scores in category_evidence.items():
            if scores:
                category_scores[category] = np.mean(scores)
            else:
                category_scores[category] = 0.0
        
        # Calculate weighted fusion score
        fusion_score = self._calculate_fusion_score(category_scores)
        
        # Count evidence types
        bullish = sum(1 for v in factor_evidence.values() if v > 0.2)
        bearish = sum(1 for v in factor_evidence.values() if v < -0.2)
        neutral = sum(1 for v in factor_evidence.values() if -0.2 <= v <= 0.2)
        
        return FusionResult(
            bullish_evidence=bullish,
            bearish_evidence=bearish,
            neutral_evidence=neutral,
            missing_evidence=0,
            total_evidence=len(factor_evidence),
            fusion_score=fusion_score,
            category_breakdown=category_scores,
        )
    
    def fuse_with_missing(
        self,
        factor_evidence: Dict[str, float],
        factor_categories: Dict[str, str],
        expected_categories: List[str],
    ) -> FusionResult:
        """
        Fuse evidence accounting for missing categories.
        
        Args:
            factor_evidence: Dictionary mapping factor names to evidence scores
            factor_categories: Dictionary mapping factor names to categories
            expected_categories: List of expected categories
            
        Returns:
            FusionResult
        """
        result = self.fuse_evidence(factor_evidence, factor_categories)
        
        # Calculate missing categories
        present_categories = set(factor_categories.values())
        missing_categories = set(expected_categories) - present_categories
        result.missing_evidence = len(missing_categories)
        
        # Adjust fusion score based on missing evidence
        if result.missing_evidence > 0:
            coverage = len(present_categories) / len(expected_categories)
            result.fusion_score *= coverage
        
        return result
    
    def _group_by_category(
        self,
        factor_evidence: Dict[str, float],
        factor_categories: Dict[str, str],
    ) -> Dict[str, List[float]]:
        """Group evidence scores by category."""
        category_evidence = {}
        
        for factor_name, score in factor_evidence.items():
            category = factor_categories.get(factor_name, "unknown")
            if category not in category_evidence:
                category_evidence[category] = []
            category_evidence[category].append(score)
        
        return category_evidence
    
    def _calculate_fusion_score(
        self,
        category_scores: Dict[str, float],
    ) -> float:
        """
        Calculate weighted fusion score from category scores.
        
        Args:
            category_scores: Dictionary mapping categories to scores
            
        Returns:
            Fusion score (-1 to 1)
        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        for category, score in category_scores.items():
            weight = self.category_weights.get(category, 1.0)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def get_dominant_signal(
        self,
        fusion_result: FusionResult,
    ) -> str:
        """
        Get dominant signal from fusion result.
        
        Args:
            fusion_result: FusionResult
            
        Returns:
            Dominant signal: "BULLISH", "BEARISH", "NEUTRAL"
        """
        if fusion_result.fusion_score > 0.3:
            return "BULLISH"
        elif fusion_result.fusion_score < -0.3:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def get_evidence_agreement(
        self,
        factor_evidence: Dict[str, float],
    ) -> float:
        """
        Calculate agreement level among evidence.
        
        Args:
            factor_evidence: Dictionary of evidence scores
            
        Returns:
            Agreement score (0 to 1)
        """
        if not factor_evidence:
            return 0.0
        
        values = list(factor_evidence.values())
        
        # Calculate standard deviation
        std_dev = np.std(values)
        
        # Agreement is inverse of dispersion
        agreement = max(0.0, 1.0 - std_dev)
        
        return agreement


def fuse_alpha_evidence(
    factor_evidence: Dict[str, float],
    factor_categories: Dict[str, str],
    category_weights: Optional[Dict[str, float]] = None,
) -> FusionResult:
    """
    Convenience function to fuse alpha evidence.
    
    Args:
        factor_evidence: Dictionary mapping factor names to evidence scores
        factor_categories: Dictionary mapping factor names to categories
        category_weights: Optional weights for each category
        
    Returns:
        FusionResult
    """
    engine = AlphaFusionEngine(category_weights=category_weights)
    return engine.fuse_evidence(factor_evidence, factor_categories)
