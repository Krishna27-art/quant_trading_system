"""
Historical Similarity Module

Finds similar historical setups to current prediction.
Shows how similar setups performed in the past.

Example output:
- Found 20 similar days
- Average return: 6.4%
- Win rate: 78%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("historical_similarity")


@dataclass
class SimilarSetup:
    """Represents a similar historical setup."""
    
    date: date
    symbol: str
    similarity_score: float  # 0-1
    actual_return: float | None = None
    was_correct: bool | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "symbol": self.symbol,
            "similarity_score": round(self.similarity_score, 3),
            "actual_return": self.actual_return,
            "was_correct": self.was_correct,
        }


class HistoricalSimilarityFinder:
    """
    Finds similar historical setups to current prediction.
    
    Provides historical evidence for prediction confidence.
    """
    
    def __init__(self, lookback_days: int = 365):
        """
        Initialize the historical similarity finder.
        
        Args:
            lookback_days: How far back to search for similar setups
        """
        self.lookback_days = lookback_days
        self.logger = logger
    
    def find_similar_setups(
        self,
        current_features: dict[str, float],
        historical_data: pd.DataFrame,
        top_n: int = 20,
    ) -> list[SimilarSetup]:
        """
        Find similar historical setups.
        
        Args:
            current_features: Current prediction features
            historical_data: DataFrame with historical features and outcomes
            top_n: Number of similar setups to return
            
        Returns:
            List of SimilarSetup objects
        """
        self.logger.info(f"Finding {top_n} similar historical setups...")
        
        if historical_data.empty:
            self.logger.warning("No historical data available")
            return []
        
        # Calculate similarity scores
        similarities = []
        
        for _, row in historical_data.iterrows():
            similarity = self._calculate_similarity(current_features, row)
            
            if similarity > 0.5:  # Only include reasonably similar setups
                similar_setup = SimilarSetup(
                    date=row.get('date', date.today()),
                    symbol=row.get('symbol', 'Unknown'),
                    similarity_score=similarity,
                    actual_return=row.get('actual_return'),
                    was_correct=row.get('was_correct'),
                )
                similarities.append(similar_setup)
        
        # Sort by similarity score and take top N
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        top_similarities = similarities[:top_n]
        
        self.logger.info(f"Found {len(top_similarities)} similar setups")
        return top_similarities
    
    def _calculate_similarity(
        self,
        current_features: dict[str, float],
        historical_row: pd.Series,
    ) -> float:
        """
        Calculate similarity between current and historical features.
        
        Uses cosine similarity on normalized feature vectors.
        """
        # Get common features
        common_features = set(current_features.keys()) & set(historical_row.index)
        
        if not common_features:
            return 0.0
        
        # Extract feature values
        current_values = np.array([current_features[f] for f in common_features])
        historical_values = np.array([historical_row[f] for f in common_features])
        
        # Handle missing values
        mask = ~np.isnan(current_values) & ~np.isnan(historical_values)
        if np.sum(mask) < 2:
            return 0.0
        
        current_values = current_values[mask]
        historical_values = historical_values[mask]
        
        # Normalize
        current_norm = current_values / (np.linalg.norm(current_values) + 1e-8)
        historical_norm = historical_values / (np.linalg.norm(historical_values) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(current_norm, historical_norm)
        
        return max(0, similarity)  # Ensure non-negative
    
    def analyze_similar_setups(
        self,
        similar_setups: list[SimilarSetup],
    ) -> dict[str, Any]:
        """
        Analyze performance of similar historical setups.
        
        Args:
            similar_setups: List of SimilarSetup objects
            
        Returns:
            Dict with performance analysis
        """
        if not similar_setups:
            return {
                "found": 0,
                "avg_return": None,
                "win_rate": None,
            }
        
        # Filter setups with outcomes
        with_outcomes = [s for s in similar_setups if s.actual_return is not None]
        
        if not with_outcomes:
            return {
                "found": len(similar_setups),
                "avg_return": None,
                "win_rate": None,
                "note": "No outcome data available",
            }
        
        returns = [s.actual_return for s in with_outcomes]
        correct = [s.was_correct for s in with_outcomes if s.was_correct is not None]
        
        avg_return = np.mean(returns)
        win_rate = sum(correct) / len(correct) * 100 if correct else None
        
        return {
            "found": len(similar_setups),
            "with_outcomes": len(with_outcomes),
            "avg_return": round(avg_return, 2),
            "win_rate": round(win_rate, 2) if win_rate else None,
            "best_return": round(max(returns), 2) if returns else None,
            "worst_return": round(min(returns), 2) if returns else None,
            "avg_similarity": round(np.mean([s.similarity_score for s in similar_setups]), 3),
        }
    
    def format_explanation(
        self,
        similar_setups: list[SimilarSetup],
        analysis: dict[str, Any],
    ) -> str:
        """
        Format historical similarity explanation.
        
        Args:
            similar_setups: List of SimilarSetup objects
            analysis: Analysis dict from analyze_similar_setups
            
        Returns:
            Formatted string
        """
        lines = [f"Historical Similarity Analysis:"]
        lines.append(f"Found {analysis['found']} similar setups")
        
        if analysis.get('avg_return') is not None:
            lines.append(f"Average return: {analysis['avg_return']:.2f}%")
        
        if analysis.get('win_rate') is not None:
            lines.append(f"Win rate: {analysis['win_rate']:.1f}%")
        
        if analysis.get('best_return') is not None:
            lines.append(f"Best return: {analysis['best_return']:.2f}%")
        
        if analysis.get('worst_return') is not None:
            lines.append(f"Worst return: {analysis['worst_return']:.2f}%")
        
        lines.append(f"Average similarity: {analysis['avg_similarity']:.3f}")
        
        return "\n".join(lines)


def find_historical_similarity(
    current_features: dict[str, float],
    historical_data: pd.DataFrame,
    top_n: int = 20,
    lookback_days: int = 365,
) -> dict[str, Any]:
    """
    Convenience function to find and analyze similar historical setups.
    
    Args:
        current_features: Current prediction features
        historical_data: DataFrame with historical features
        top_n: Number of similar setups to return
        lookback_days: How far back to search
        
    Returns:
        Historical similarity analysis
    """
    finder = HistoricalSimilarityFinder(lookback_days=lookback_days)
    similar_setups = finder.find_similar_setups(current_features, historical_data, top_n)
    analysis = finder.analyze_similar_setups(similar_setups)
    
    return {
        "similar_setups": [s.to_dict() for s in similar_setups],
        "analysis": analysis,
        "formatted": finder.format_explanation(similar_setups, analysis),
    }
