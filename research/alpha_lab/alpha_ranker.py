"""
Alpha Ranker

Ranks factors based on multiple dimensions of performance.
Combines IC, stability, regime performance, sector performance, and decay into an overall score.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("research.alpha_ranker")


@dataclass
class FactorRank:
    """Rank for a single factor."""
    factor_name: str
    alpha_score: float
    information_score: float
    stability_score: float
    regime_score: float
    sector_score: float
    decay_score: float
    overall_score: float
    rank: int
    tier: str  # S, A, B, C, D
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "factor_name": self.factor_name,
            "alpha_score": round(self.alpha_score, 4),
            "information_score": round(self.information_score, 4),
            "stability_score": round(self.stability_score, 4),
            "regime_score": round(self.regime_score, 4),
            "sector_score": round(self.sector_score, 4),
            "decay_score": round(self.decay_score, 4),
            "overall_score": round(self.overall_score, 4),
            "rank": self.rank,
            "tier": self.tier,
        }


class AlphaRanker:
    """
    Ranks factors based on multiple performance dimensions.
    
    Scoring dimensions:
    1. Alpha Score: IC, Rank IC, hit rate, Sharpe
    2. Information Score: t-statistics, p-values, mutual information
    3. Stability Score: IC stability across time splits
    4. Regime Score: Performance across market regimes
    5. Sector Score: Performance across sectors
    6. Decay Score: Signal decay over time
    
    Overall score is weighted combination of all dimensions.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize alpha ranker.
        
        Args:
            weights: Optional custom weights for scoring dimensions
        """
        self.weights = weights or {
            "alpha": 0.35,
            "information": 0.20,
            "stability": 0.20,
            "regime": 0.10,
            "sector": 0.10,
            "decay": 0.05,
        }
        self._logger = get_logger("research.alpha_ranker")
    
    def rank_factor(
        self,
        factor_name: str,
        ic_result: Dict,
        stability_result: Optional[Dict] = None,
        regime_performance: Optional[Dict] = None,
        sector_performance: Optional[Dict] = None,
        decay_result: Optional[Dict] = None,
    ) -> FactorRank:
        """
        Rank a single factor.
        
        Args:
            factor_name: Name of factor
            ic_result: IC results dictionary
            stability_result: Optional stability results
            regime_performance: Optional regime performance dictionary
            sector_performance: Optional sector performance dictionary
            decay_result: Optional decay results
            
        Returns:
            FactorRank
        """
        # Calculate individual scores
        alpha_score = self._calculate_alpha_score(ic_result)
        information_score = self._calculate_information_score(ic_result)
        stability_score = self._calculate_stability_score(stability_result)
        regime_score = self._calculate_regime_score(regime_performance)
        sector_score = self._calculate_sector_score(sector_performance)
        decay_score = self._calculate_decay_score(decay_result)
        
        # Calculate overall score
        overall_score = (
            self.weights["alpha"] * alpha_score +
            self.weights["information"] * information_score +
            self.weights["stability"] * stability_score +
            self.weights["regime"] * regime_score +
            self.weights["sector"] * sector_score +
            self.weights["decay"] * decay_score
        )
        
        # Determine tier
        tier = self._determine_tier(overall_score)
        
        return FactorRank(
            factor_name=factor_name,
            alpha_score=alpha_score,
            information_score=information_score,
            stability_score=stability_score,
            regime_score=regime_score,
            sector_score=sector_score,
            decay_score=decay_score,
            overall_score=overall_score,
            rank=0,  # Will be set when ranking multiple factors
            tier=tier,
        )
    
    def rank_factors(
        self,
        factor_results: List[Dict],
    ) -> List[FactorRank]:
        """
        Rank multiple factors.
        
        Args:
            factor_results: List of dictionaries with factor results
            
        Returns:
            List of FactorRank sorted by overall score
        """
        ranks = []
        
        for result in factor_results:
            rank = self.rank_factor(
                factor_name=result["factor_name"],
                ic_result=result.get("ic_result", {}),
                stability_result=result.get("stability_result"),
                regime_performance=result.get("regime_performance"),
                sector_performance=result.get("sector_performance"),
                decay_result=result.get("decay_result"),
            )
            ranks.append(rank)
        
        # Sort by overall score
        ranks.sort(key=lambda r: r.overall_score, reverse=True)
        
        # Assign ranks
        for i, rank in enumerate(ranks):
            rank.rank = i + 1
        
        return ranks
    
    def _calculate_alpha_score(self, ic_result: Dict) -> float:
        """Calculate alpha score from IC results."""
        mean_ic = abs(ic_result.get("mean_ic", 0))
        mean_rank_ic = abs(ic_result.get("mean_rank_ic", 0))
        hit_rate = ic_result.get("hit_rate", 0.5)
        
        # Normalize IC to 0-1 scale (assuming max reasonable IC is 0.1)
        ic_score = min(mean_ic / 0.1, 1.0)
        rank_ic_score = min(mean_rank_ic / 0.1, 1.0)
        
        # Normalize hit rate to 0-1 scale (0.5 is neutral, 1.0 is perfect)
        hit_rate_score = (hit_rate - 0.5) / 0.5
        
        # Combine
        alpha_score = 0.4 * ic_score + 0.4 * rank_ic_score + 0.2 * hit_rate_score
        return max(0.0, min(1.0, alpha_score))
    
    def _calculate_information_score(self, ic_result: Dict) -> float:
        """Calculate information score from t-statistics and p-values."""
        ic_t_stat = abs(ic_result.get("ic_t_stat", 0))
        ic_p_value = ic_result.get("ic_p_value", 1.0)
        
        # t-stat score (assuming max reasonable t-stat is 5)
        t_stat_score = min(ic_t_stat / 5.0, 1.0)
        
        # p-value score (lower is better)
        p_value_score = max(0, 1 - ic_p_value / 0.05)
        
        # Combine
        information_score = 0.6 * t_stat_score + 0.4 * p_value_score
        return max(0.0, min(1.0, information_score))
    
    def _calculate_stability_score(self, stability_result: Optional[Dict]) -> float:
        """Calculate stability score from IC stability results."""
        if stability_result is None:
            return 0.5  # Default neutral score
        
        ic_stability = stability_result.get("ic_stability", 0)
        rank_ic_stability = stability_result.get("rank_ic_stability", 0)
        
        # Average stability
        stability_score = (ic_stability + rank_ic_stability) / 2
        return max(0.0, min(1.0, stability_score))
    
    def _calculate_regime_score(self, regime_performance: Optional[Dict]) -> float:
        """Calculate regime score from performance across regimes."""
        if regime_performance is None:
            return 0.5  # Default neutral score
        
        # Score based on how many regimes the factor works in
        regimes = regime_performance.get("regimes", {})
        working_regimes = sum(1 for perf in regimes.values() if perf > 0.02)
        total_regimes = len(regimes)
        
        if total_regimes == 0:
            return 0.5
        
        regime_score = working_regimes / total_regimes
        return max(0.0, min(1.0, regime_score))
    
    def _calculate_sector_score(self, sector_performance: Optional[Dict]) -> float:
        """Calculate sector score from performance across sectors."""
        if sector_performance is None:
            return 0.5  # Default neutral score
        
        # Score based on average performance across sectors
        sector_scores = list(sector_performance.values())
        if not sector_scores:
            return 0.5
        
        # Average absolute IC across sectors
        avg_sector_ic = np.mean([abs(s) for s in sector_scores])
        sector_score = min(avg_sector_ic / 0.1, 1.0)
        return max(0.0, min(1.0, sector_score))
    
    def _calculate_decay_score(self, decay_result: Optional[Dict]) -> float:
        """Calculate decay score from signal decay results."""
        if decay_result is None:
            return 0.5  # Default neutral score
        
        # Score based on how long the signal persists
        decay_horizons = decay_result.get("decay_horizons", [])
        decay_ics = decay_result.get("decay_ics", [])
        
        if not decay_ics:
            return 0.5
        
        # Calculate area under decay curve (persistence)
        persistence = sum(decay_ics) / len(decay_ics)
        decay_score = min(persistence / 0.05, 1.0)  # Assuming 0.05 is good persistence
        return max(0.0, min(1.0, decay_score))
    
    def _determine_tier(self, overall_score: float) -> str:
        """Determine tier based on overall score."""
        if overall_score >= 0.9:
            return "S"
        elif overall_score >= 0.8:
            return "A"
        elif overall_score >= 0.7:
            return "B"
        elif overall_score >= 0.6:
            return "C"
        else:
            return "D"
    
    def get_top_factors(
        self,
        ranks: List[FactorRank],
        n: int = 20,
        min_tier: Optional[str] = None,
    ) -> List[FactorRank]:
        """
        Get top N factors with optional tier filter.
        
        Args:
            ranks: List of FactorRank
            n: Number of top factors to return
            min_tier: Optional minimum tier (e.g., "A" for A and above)
            
        Returns:
            List of top FactorRank
        """
        filtered = ranks
        
        if min_tier:
            tier_order = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
            min_tier_value = tier_order.get(min_tier, 0)
            filtered = [r for r in ranks if tier_order.get(r.tier, 0) >= min_tier_value]
        
        return filtered[:n]
