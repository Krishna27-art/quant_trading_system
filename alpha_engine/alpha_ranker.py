"""
Alpha Ranker

Ranks stocks by alpha score and selects top N for prediction.

STEP 11: Alpha Ranking

This module:
1. Ranks all stocks by final alpha score
2. Filters by minimum grade
3. Selects top N stocks
4. Provides ranked list for prediction engine
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from alpha_engine.alpha_builder import AlphaGrade, AlphaResult
from utils.logger import get_logger

logger = get_logger("alpha_engine.ranker")


@dataclass
class RankingResult:
    """
    Result of ranking operation.
    """
    ranked_stocks: List[Tuple[str, float, AlphaGrade]]  # (symbol, alpha_score, grade)
    total_evaluated: int
    total_passed: int
    top_n_selected: int
    min_score: float
    max_score: float
    avg_score: float
    grade_distribution: Dict[str, int]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "ranked_stocks": [
                {"symbol": sym, "alpha_score": round(score, 2), "grade": grade.value}
                for sym, score, grade in self.ranked_stocks
            ],
            "total_evaluated": self.total_evaluated,
            "total_passed": self.total_passed,
            "top_n_selected": self.top_n_selected,
            "min_score": round(self.min_score, 2),
            "max_score": round(self.max_score, 2),
            "avg_score": round(self.avg_score, 2),
            "grade_distribution": self.grade_distribution,
        }


@dataclass
class RankingConfig:
    """
    Configuration for ranking operation.
    """
    top_n: int = 20  # Number of top stocks to select
    min_grade: AlphaGrade = AlphaGrade.EXCELLENT  # Minimum grade to consider
    min_score: float = 0.0  # Minimum alpha score
    diversify_sectors: bool = True  # Whether to diversify across sectors
    max_per_sector: int = 5  # Maximum stocks per sector if diversifying
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "top_n": self.top_n,
            "min_grade": self.min_grade.value,
            "min_score": self.min_score,
            "diversify_sectors": self.diversify_sectors,
            "max_per_sector": self.max_per_sector,
        }


class AlphaRanker:
    """
    Ranks stocks by alpha score and selects top N.
    
    This is the final step before passing stocks to the prediction engine.
    """
    
    def __init__(self):
        """Initialize Alpha Ranker."""
        self._logger = logger
        
        # Default ranking configuration
        self.config = RankingConfig()
    
    def rank_stocks(
        self,
        alpha_results: Dict[str, AlphaResult],
        sector_map: Optional[Dict[str, str]] = None,
        config: Optional[RankingConfig] = None,
    ) -> RankingResult:
        """
        Rank stocks by alpha score and select top N.
        
        Args:
            alpha_results: Dictionary of symbol -> AlphaResult
            sector_map: Optional dictionary of symbol -> sector
            config: Optional ranking configuration
            
        Returns:
            RankingResult with ranked stocks
        """
        if config is None:
            config = self.config
        
        self._logger.info(
            f"Ranking {len(alpha_results)} stocks (top_n={config.top_n}, min_grade={config.min_grade.value})"
        )
        
        # Step 1: Filter by minimum grade and score
        filtered = self._filter_by_thresholds(alpha_results, config)
        
        # Step 2: Sort by alpha score
        sorted_stocks = self._sort_by_alpha_score(filtered)
        
        # Step 3: Apply sector diversification if enabled
        if config.diversify_sectors and sector_map:
            sorted_stocks = self._diversify_by_sector(
                sorted_stocks,
                sector_map,
                config.max_per_sector,
            )
        
        # Step 4: Select top N
        top_stocks = sorted_stocks[:config.top_n]
        
        # Step 5: Calculate statistics
        stats = self._calculate_statistics(alpha_results, filtered, top_stocks)
        
        result = RankingResult(
            ranked_stocks=top_stocks,
            total_evaluated=len(alpha_results),
            total_passed=len(filtered),
            top_n_selected=len(top_stocks),
            **stats,
        )
        
        self._logger.info(
            f"Ranking complete: {len(top_stocks)} stocks selected from {len(alpha_results)} evaluated",
            extra={
                "passed": len(filtered),
                "top_symbols": [s[0] for s in top_stocks[:5]],
            },
        )
        
        return result
    
    def get_top_symbols(
        self,
        ranking_result: RankingResult,
    ) -> List[str]:
        """
        Extract top symbols from ranking result.
        
        Args:
            ranking_result: RankingResult from rank_stocks
            
        Returns:
            List of top stock symbols
        """
        return [symbol for symbol, _, _ in ranking_result.ranked_stocks]
    
    def get_top_alpha_results(
        self,
        ranking_result: RankingResult,
        alpha_results: Dict[str, AlphaResult],
    ) -> Dict[str, AlphaResult]:
        """
        Get AlphaResult objects for top ranked stocks.
        
        Args:
            ranking_result: RankingResult from rank_stocks
            alpha_results: Original alpha results dictionary
            
        Returns:
            Dictionary of symbol -> AlphaResult for top stocks
        """
        top_symbols = self.get_top_symbols(ranking_result)
        return {sym: alpha_results[sym] for sym in top_symbols if sym in alpha_results}
    
    def _filter_by_thresholds(
        self,
        alpha_results: Dict[str, AlphaResult],
        config: RankingConfig,
    ) -> List[Tuple[str, float, AlphaGrade]]:
        """
        Filter stocks by minimum grade and score.
        
        Args:
            alpha_results: Dictionary of symbol -> AlphaResult
            config: Ranking configuration
            
        Returns:
            List of (symbol, alpha_score, grade) tuples
        """
        filtered = []
        
        # Get grade threshold
        grade_thresholds = {
            AlphaGrade.INSTITUTIONAL: 95,
            AlphaGrade.EXCELLENT: 85,
            AlphaGrade.GOOD: 75,
            AlphaGrade.AVERAGE: 60,
            AlphaGrade.REJECT: 0,
        }
        
        min_threshold = grade_thresholds.get(config.min_grade, 0)
        
        for symbol, result in alpha_results.items():
            # Check if filters passed
            if not result.passed_filters:
                continue
            
            # Check minimum grade
            grade_threshold = grade_thresholds.get(result.grade, 0)
            if grade_threshold < min_threshold:
                continue
            
            # Check minimum score
            if result.final_alpha_score < config.min_score:
                continue
            
            filtered.append((symbol, result.final_alpha_score, result.grade))
        
        return filtered
    
    def _sort_by_alpha_score(
        self,
        stocks: List[Tuple[str, float, AlphaGrade]],
    ) -> List[Tuple[str, float, AlphaGrade]]:
        """
        Sort stocks by alpha score (descending).
        
        Args:
            stocks: List of (symbol, alpha_score, grade) tuples
            
        Returns:
            Sorted list
        """
        return sorted(stocks, key=lambda x: x[1], reverse=True)
    
    def _diversify_by_sector(
        self,
        stocks: List[Tuple[str, float, AlphaGrade]],
        sector_map: Dict[str, str],
        max_per_sector: int,
    ) -> List[Tuple[str, float, AlphaGrade]]:
        """
        Diversify stocks by sector to avoid concentration.
        
        Args:
            stocks: Sorted list of (symbol, alpha_score, grade) tuples
            sector_map: Dictionary of symbol -> sector
            max_per_sector: Maximum stocks per sector
            
        Returns:
            Diversified list
        """
        diversified = []
        sector_counts = {}
        
        for symbol, score, grade in stocks:
            sector = sector_map.get(symbol, "unknown")
            
            # Check sector limit
            if sector_counts.get(sector, 0) >= max_per_sector:
                continue
            
            diversified.append((symbol, score, grade))
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        # If we filtered too many stocks, add back the remaining
        if len(diversified) < len(stocks) * 0.5:
            self._logger.warning(
                f"Sector diversification removed too many stocks ({len(diversified)} from {len(stocks)})"
            )
            # Fall back to original list
            return stocks
        
        return diversified
    
    def _calculate_statistics(
        self,
        all_results: Dict[str, AlphaResult],
        filtered_stocks: List[Tuple[str, float, AlphaGrade]],
        top_stocks: List[Tuple[str, float, AlphaGrade]],
    ) -> Dict[str, Any]:
        """
        Calculate ranking statistics.
        
        Args:
            all_results: All alpha results
            filtered_stocks: Stocks that passed filters
            top_stocks: Top N selected stocks
            
        Returns:
            Dictionary with statistics
        """
        if not filtered_stocks:
            return {
                "min_score": 0.0,
                "max_score": 0.0,
                "avg_score": 0.0,
                "grade_distribution": {},
            }
        
        scores = [score for _, score, _ in filtered_stocks]
        
        # Calculate grade distribution
        grade_dist = {}
        for _, _, grade in filtered_stocks:
            grade_dist[grade.value] = grade_dist.get(grade.value, 0) + 1
        
        return {
            "min_score": min(scores),
            "max_score": max(scores),
            "avg_score": np.mean(scores),
            "grade_distribution": grade_dist,
        }
    
    def update_config(
        self,
        top_n: Optional[int] = None,
        min_grade: Optional[AlphaGrade] = None,
        min_score: Optional[float] = None,
        diversify_sectors: Optional[bool] = None,
        max_per_sector: Optional[int] = None,
    ) -> None:
        """
        Update ranking configuration.
        
        Args:
            top_n: Number of top stocks to select
            min_grade: Minimum grade to consider
            min_score: Minimum alpha score
            diversify_sectors: Whether to diversify across sectors
            max_per_sector: Maximum stocks per sector
        """
        if top_n is not None:
            self.config.top_n = top_n
        if min_grade is not None:
            self.config.min_grade = min_grade
        if min_score is not None:
            self.config.min_score = min_score
        if diversify_sectors is not None:
            self.config.diversify_sectors = diversify_sectors
        if max_per_sector is not None:
            self.config.max_per_sector = max_per_sector
        
        self._logger.info(f"Ranking config updated: {self.config.to_dict()}")
    
    def get_ranking_dataframe(
        self,
        ranking_result: RankingResult,
        alpha_results: Dict[str, AlphaResult],
    ) -> pd.DataFrame:
        """
        Convert ranking result to pandas DataFrame for analysis.
        
        Args:
            ranking_result: RankingResult from rank_stocks
            alpha_results: Original alpha results dictionary
            
        Returns:
            DataFrame with ranking information
        """
        rows = []
        
        for rank, (symbol, score, grade) in enumerate(ranking_result.ranked_stocks, 1):
            result = alpha_results.get(symbol)
            if result is None:
                continue
            
            row = {
                "rank": rank,
                "symbol": symbol,
                "alpha_score": score,
                "grade": grade.value,
                "raw_alpha": result.raw_alpha_score,
                "passed_filters": result.passed_filters,
            }
            
            # Add category scores
            for cat_name, cat_obj in result.categories.items():
                row[f"cat_{cat_name}"] = cat_obj.score
            
            rows.append(row)
        
        return pd.DataFrame(rows)
