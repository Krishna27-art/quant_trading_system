"""
Alpha Score Calculator

Calculates raw alpha scores and assigns grades.

STEP 7: Calculate Raw Alpha Score
STEP 10: Alpha Grade

This module:
1. Combines category scores with weights to calculate raw alpha
2. Applies filter penalties
3. Assigns alpha grades (Institutional, Excellent, Good, Average, Reject)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from alpha_engine.alpha_builder import AlphaGrade, AlphaResult, AlphaCategory
from alpha_engine.alpha_filters import FilterResult
from utils.logger import get_logger

logger = get_logger("alpha_engine.score")


@dataclass
class ScoreBreakdown:
    """
    Detailed breakdown of alpha score calculation.
    """
    category_scores: Dict[str, float]
    weights: Dict[str, float]
    weighted_contributions: Dict[str, float]
    raw_alpha: float
    filter_penalty: float
    final_alpha: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "category_scores": {k: round(v, 2) for k, v in self.category_scores.items()},
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "weighted_contributions": {k: round(v, 2) for k, v in self.weighted_contributions.items()},
            "raw_alpha": round(self.raw_alpha, 2),
            "filter_penalty": round(self.filter_penalty, 2),
            "final_alpha": round(self.final_alpha, 2),
        }


class AlphaScoreCalculator:
    """
    Calculates alpha scores and assigns grades.
    
    This is the core scoring engine that combines all the previous steps
    into a final alpha score and grade.
    """
    
    def __init__(self):
        """Initialize Alpha Score Calculator."""
        self._logger = logger
        
        # Grade thresholds
        self.grade_thresholds = {
            AlphaGrade.INSTITUTIONAL: 95.0,
            AlphaGrade.EXCELLENT: 85.0,
            AlphaGrade.GOOD: 75.0,
            AlphaGrade.AVERAGE: 60.0,
        }
    
    def calculate_alpha_score(
        self,
        symbol: str,
        category_scores: Dict[str, float],
        weights: Dict[str, float],
        filter_results: List[FilterResult],
        timestamp: Optional[datetime] = None,
    ) -> AlphaResult:
        """
        Calculate alpha score for a single stock.
        
        Args:
            symbol: Stock symbol
            category_scores: Dictionary of category -> score (0-100)
            weights: Dictionary of category -> weight (sum to 1.0)
            filter_results: List of filter results
            timestamp: Timestamp of calculation
            
        Returns:
            AlphaResult with score, grade, and breakdown
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self._logger.info(f"Calculating alpha score for {symbol}")
        
        # Step 1: Calculate weighted contributions
        weighted_contributions = self._calculate_weighted_contributions(
            category_scores,
            weights,
        )
        
        # Step 2: Calculate raw alpha score
        raw_alpha = sum(weighted_contributions.values())
        
        # Step 3: Apply filter penalty
        filter_penalty = self._calculate_filter_penalty(filter_results)
        final_alpha = raw_alpha - filter_penalty
        
        # Step 4: Clamp to 0-100
        final_alpha = np.clip(final_alpha, 0, 100)
        
        # Step 5: Determine grade
        grade = self._assign_grade(final_alpha)
        
        # Step 6: Check if filters passed
        passed_filters = all(r.passed for r in filter_results)
        filter_reasons = [r.reason for r in filter_results if not r.passed]
        
        # Step 7: Create AlphaCategory objects
        categories = self._create_alpha_categories(
            category_scores,
            weights,
            weighted_contributions,
        )
        
        # Create score breakdown
        breakdown = ScoreBreakdown(
            category_scores=category_scores,
            weights=weights,
            weighted_contributions=weighted_contributions,
            raw_alpha=raw_alpha,
            filter_penalty=filter_penalty,
            final_alpha=final_alpha,
        )
        
        result = AlphaResult(
            symbol=symbol,
            timestamp=timestamp,
            categories=categories,
            raw_alpha_score=raw_alpha,
            final_alpha_score=final_alpha,
            grade=grade,
            passed_filters=passed_filters,
            filter_reasons=filter_reasons,
            explanation={"breakdown": breakdown.to_dict()},
        )
        
        self._logger.info(
            f"Alpha score calculated for {symbol}: {final_alpha:.2f} ({grade.value})",
            extra={
                "raw_alpha": round(raw_alpha, 2),
                "filter_penalty": round(filter_penalty, 2),
                "passed_filters": passed_filters,
            },
        )
        
        return result
    
    def calculate_batch_alpha_scores(
        self,
        alpha_inputs: Dict[str, Dict[str, Any]],
        weights: Dict[str, float],
        filter_results_map: Dict[str, List[FilterResult]],
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, AlphaResult]:
        """
        Calculate alpha scores for multiple stocks.
        
        Args:
            alpha_inputs: Dictionary of symbol -> alpha input
            weights: Dictionary of category -> weight
            filter_results_map: Dictionary of symbol -> filter results
            timestamp: Timestamp of calculation
            
        Returns:
            Dictionary of symbol -> AlphaResult
        """
        self._logger.info(f"Calculating batch alpha scores for {len(alpha_inputs)} stocks")
        
        results = {}
        for symbol, alpha_input in alpha_inputs.items():
            category_scores = alpha_input["category_scores"]
            filters = filter_results_map.get(symbol, [])
            
            result = self.calculate_alpha_score(
                symbol=symbol,
                category_scores=category_scores,
                weights=weights,
                filter_results=filters,
                timestamp=timestamp,
            )
            results[symbol] = result
        
        return results
    
    def _calculate_weighted_contributions(
        self,
        category_scores: Dict[str, float],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate weighted contribution of each category.
        
        Args:
            category_scores: Dictionary of category -> score
            weights: Dictionary of category -> weight
            
        Returns:
            Dictionary of category -> weighted contribution
        """
        contributions = {}
        
        for category, score in category_scores.items():
            weight = weights.get(category, 0.0)
            contributions[category] = score * weight
        
        return contributions
    
    def _calculate_filter_penalty(self, filter_results: List[FilterResult]) -> float:
        """
        Calculate penalty based on failed filters.
        
        Args:
            filter_results: List of filter results
            
        Returns:
            Penalty to subtract from raw alpha (0-100)
        """
        penalty = 0.0
        
        for result in filter_results:
            if not result.passed:
                # Different filters have different penalty levels
                # Critical filters (liquidity, circuit) have higher penalties
                if result.filter_type.value in ["liquidity", "circuit"]:
                    penalty += 20.0
                elif result.filter_type.value in ["risk_reward"]:
                    penalty += 15.0
                elif result.filter_type.value in ["news", "volatility"]:
                    penalty += 10.0
                else:
                    penalty += 5.0
        
        # Cap penalty at 50 points
        return min(penalty, 50.0)
    
    def _assign_grade(self, alpha_score: float) -> AlphaGrade:
        """
        Assign grade based on alpha score.
        
        Args:
            alpha_score: Final alpha score (0-100)
            
        Returns:
            AlphaGrade
        """
        if alpha_score >= self.grade_thresholds[AlphaGrade.INSTITUTIONAL]:
            return AlphaGrade.INSTITUTIONAL
        elif alpha_score >= self.grade_thresholds[AlphaGrade.EXCELLENT]:
            return AlphaGrade.EXCELLENT
        elif alpha_score >= self.grade_thresholds[AlphaGrade.GOOD]:
            return AlphaGrade.GOOD
        elif alpha_score >= self.grade_thresholds[AlphaGrade.AVERAGE]:
            return AlphaGrade.AVERAGE
        else:
            return AlphaGrade.REJECT
    
    def _create_alpha_categories(
        self,
        category_scores: Dict[str, float],
        weights: Dict[str, float],
        weighted_contributions: Dict[str, float],
    ) -> Dict[str, AlphaCategory]:
        """
        Create AlphaCategory objects.
        
        Args:
            category_scores: Dictionary of category -> score
            weights: Dictionary of category -> weight
            weighted_contributions: Dictionary of category -> weighted contribution
            
        Returns:
            Dictionary of category_name -> AlphaCategory
        """
        categories = {}
        
        for category_name, score in category_scores.items():
            weight = weights.get(category_name, 0.0)
            categories[category_name] = AlphaCategory(
                name=category_name,
                score=score,
                weight=weight,
                signals=[],  # Signals are tracked in the builder
            )
        
        return categories
    
    def get_grade_summary(self, alpha_results: Dict[str, AlphaResult]) -> Dict[str, Any]:
        """
        Get summary of grades across all stocks.
        
        Args:
            alpha_results: Dictionary of symbol -> AlphaResult
            
        Returns:
            Dictionary with grade summary
        """
        grade_counts = {grade.value: 0 for grade in AlphaGrade}
        
        for result in alpha_results.values():
            grade_counts[result.grade.value] += 1
        
        total = len(alpha_results)
        
        return {
            "total": total,
            "grade_distribution": {
                grade: {
                    "count": count,
                    "percentage": round(count / total * 100, 2) if total > 0 else 0,
                }
                for grade, count in grade_counts.items()
            },
            "average_alpha": round(
                np.mean([r.final_alpha_score for r in alpha_results.values()]), 2
            ) if alpha_results else 0,
            "passed_filters": sum(1 for r in alpha_results.values() if r.passed_filters),
        }
    
    def update_grade_threshold(
        self,
        grade: AlphaGrade,
        threshold: float,
    ) -> None:
        """
        Update threshold for a specific grade.
        
        Args:
            grade: Grade to update
            threshold: New threshold value
        """
        if grade not in self.grade_thresholds:
            self._logger.warning(f"Unknown grade: {grade}")
            return
        
        self.grade_thresholds[grade] = threshold
        self._logger.info(f"Updated {grade.value} threshold to {threshold}")
