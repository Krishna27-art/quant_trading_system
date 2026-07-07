"""
Institutional Swing Trading Watchlist - Professional Selection System

Integrates all swing trading components into a unified watchlist builder:
- Market regime analysis
- Sector rotation tracking
- Relative strength scoring
- Trend quality assessment
- Base formation detection
- Catalyst identification
- Risk/reward calculation

Output: 10-30 high-conviction swing trading candidates with 0-100 scores.
Only stocks scoring 85+ qualify for the final watchlist.

Institutional workflow:
1. Analyze Nifty (market regime)
2. Analyze sectors (rotation)
3. Rank sectors (strength)
4. Score all stocks (0-100)
5. Filter by minimum score (85+)
6. Apply liquidity filters
7. Validate with technical analysis
8. Create final watchlist (10-30 stocks)
9. Set entry/exit parameters
10. Monitor for valid setups
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from data_platform.pipelines.base_formation_detector import BaseFormationDetector
from data_platform.pipelines.swing_trading_scorer import (
    SwingTradingScorer,
    SwingTradingScore,
    MarketRegime,
    EntryType
)
from utils.logger import get_logger

logger = get_logger("swing_watchlist")


@dataclass
class SwingWatchlistResult:
    """Complete swing trading watchlist result."""
    market_regime: MarketRegime
    market_regime_score: float
    
    # All scored stocks
    all_scores: list[SwingTradingScore]
    
    # Qualified stocks (score >= 85)
    qualified_stocks: list[SwingTradingScore]
    
    # Final watchlist (10-30 stocks)
    final_watchlist: list[SwingTradingScore]
    
    # Statistics
    total_universe_size: int
    qualified_count: int
    final_watchlist_size: int
    
    # Sector breakdown
    sector_distribution: dict[str, int]
    
    # Entry type distribution
    entry_type_distribution: dict[str, int]
    
    # Average metrics
    avg_score: float
    avg_risk_reward_ratio: float
    
    timestamp: datetime


class SwingWatchlistBuilder:
    """
    Builds institutional swing trading watchlists.
    
    Process:
    1. Score entire universe (0-100)
    2. Filter by minimum score (85+)
    3. Apply liquidity filters
    4. Integrate base formation analysis
    5. Validate technical setups
    6. Select top 10-30 candidates
    7. Calculate entry/exit parameters
    """

    def __init__(
        self,
        min_score_threshold: float = 85.0,
        min_avg_daily_value_cr: float = 50.0,
        target_watchlist_size: int = 20,
        use_mock_data: bool = True
    ):
        """
        Initialize swing watchlist builder.
        
        Parameters
        ----------
        min_score_threshold
            Minimum score to qualify (default: 85)
        min_avg_daily_value_cr
            Minimum average daily turnover in crores
        target_watchlist_size
            Target number of stocks in final watchlist
        use_mock_data
            Use mock data for development
        """
        self.min_score_threshold = min_score_threshold
        self.min_avg_daily_value_cr = min_avg_daily_value_cr
        self.target_watchlist_size = target_watchlist_size
        self.use_mock_data = use_mock_data
        
        # Initialize components
        self.scorer = SwingTradingScorer(
            min_score_threshold=min_score_threshold,
            min_avg_daily_value_cr=min_avg_daily_value_cr,
            use_mock_data=use_mock_data
        )
        self.base_detector = BaseFormationDetector(use_mock_data=use_mock_data)

    def build_watchlist(self) -> SwingWatchlistResult:
        """
        Build complete swing trading watchlist.
        
        Returns
        -------
        SwingWatchlistResult
            Complete watchlist with all analysis
        """
        logger.info("Building institutional swing trading watchlist")
        
        # Step 1: Score entire universe
        logger.info("Step 1: Scoring universe")
        all_scores = self.scorer.score_universe()
        
        # Get market regime from first score
        market_regime = MarketRegime.WEAK_BULL  # Default
        if all_scores:
            market_regime_score = all_scores[0].market_regime_score
            # Determine regime from score
            if market_regime_score >= 90:
                market_regime = MarketRegime.STRONG_BULL
            elif market_regime_score >= 70:
                market_regime = MarketRegime.WEAK_BULL
            elif market_regime_score >= 50:
                market_regime = MarketRegime.SIDEWAYS
            elif market_regime_score >= 30:
                market_regime = MarketRegime.BEAR
            else:
                market_regime = MarketRegime.HIGH_VOLATILITY
        
        logger.info(f"Market regime: {market_regime.value}")
        
        # Step 2: Filter by minimum score
        logger.info("Step 2: Filtering by minimum score")
        qualified = [s for s in all_scores if s.qualifies]
        logger.info(f"Qualified stocks: {len(qualified)}/{len(all_scores)}")
        
        # Step 3: Apply additional liquidity filters
        logger.info("Step 3: Applying liquidity filters")
        liquidity_filtered = self._apply_liquidity_filters(qualified)
        logger.info(f"After liquidity filter: {len(liquidity_filtered)}")
        
        # Step 4: Integrate base formation analysis
        logger.info("Step 4: Analyzing base formations")
        with_bases = self._integrate_base_formation(liquidity_filtered)
        logger.info(f"Stocks with valid bases: {len(with_bases)}")
        
        # Step 5: Validate technical setups
        logger.info("Step 5: Validating technical setups")
        validated = self._validate_technical_setups(with_bases)
        logger.info(f"Technically validated: {len(validated)}")
        
        # Step 6: Select top candidates
        logger.info("Step 6: Selecting final watchlist")
        final_watchlist = self._select_final_candidates(validated)
        logger.info(f"Final watchlist size: {len(final_watchlist)}")
        
        # Step 7: Calculate statistics
        sector_dist = self._calculate_sector_distribution(final_watchlist)
        entry_type_dist = self._calculate_entry_type_distribution(final_watchlist)
        avg_score = sum(s.final_score for s in final_watchlist) / len(final_watchlist) if final_watchlist else 0
        avg_rr = sum(s.risk_reward_ratio for s in final_watchlist) / len(final_watchlist) if final_watchlist else 0
        
        result = SwingWatchlistResult(
            market_regime=market_regime,
            market_regime_score=all_scores[0].market_regime_score if all_scores else 0,
            all_scores=all_scores,
            qualified_stocks=qualified,
            final_watchlist=final_watchlist,
            total_universe_size=len(all_scores),
            qualified_count=len(qualified),
            final_watchlist_size=len(final_watchlist),
            sector_distribution=sector_dist,
            entry_type_distribution=entry_type_dist,
            avg_score=avg_score,
            avg_risk_reward_ratio=avg_rr,
            timestamp=datetime.now()
        )
        
        logger.info(
            f"Watchlist build complete: {len(final_watchlist)} stocks, "
            f"avg_score={avg_score:.1f}, avg_rr={avg_rr:.2f}"
        )
        
        return result

    def _apply_liquidity_filters(self, scores: list[SwingTradingScore]) -> list[SwingTradingScore]:
        """Apply additional liquidity filters."""
        filtered = []
        for score in scores:
            # Check liquidity score
            if score.liquidity_score >= 60:
                filtered.append(score)
        return filtered

    def _integrate_base_formation(
        self,
        scores: list[SwingTradingScore]
    ) -> list[SwingTradingScore]:
        """Integrate base formation analysis into scores."""
        for score in scores:
            base = self.base_detector.detect_base(score.symbol)
            score.base_formation = base
            
            # Boost score if has valid base
            if base and base.has_base and base.pattern_quality_score >= 70:
                score.final_score = min(100, score.final_score + 5)
        
        # Re-sort by updated scores
        return sorted(scores, key=lambda x: x.final_score, reverse=True)

    def _validate_technical_setups(
        self,
        scores: list[SwingTradingScore]
    ) -> list[SwingTradingScore]:
        """Validate technical setups."""
        validated = []
        for score in scores:
            # Must have valid trend quality
            if not score.trend_quality:
                continue
            
            # Must have valid entry type
            if score.entry_type == EntryType.NONE:
                continue
            
            # Risk-reward must be reasonable
            if score.risk_reward_ratio < 1.5:
                continue
            
            # Trend health must be decent
            if score.trend_quality.trend_health_score < 50:
                continue
            
            validated.append(score)
        
        return validated

    def _select_final_candidates(
        self,
        scores: list[SwingTradingScore]
    ) -> list[SwingTradingScore]:
        """Select final candidates for watchlist."""
        # Sort by final score
        sorted_scores = sorted(scores, key=lambda x: x.final_score, reverse=True)
        
        # Select top N
        return sorted_scores[:self.target_watchlist_size]

    def _calculate_sector_distribution(
        self,
        scores: list[SwingTradingScore]
    ) -> dict[str, int]:
        """Calculate sector distribution."""
        distribution = {}
        for score in scores:
            sector = score.sector
            distribution[sector] = distribution.get(sector, 0) + 1
        return distribution

    def _calculate_entry_type_distribution(
        self,
        scores: list[SwingTradingScore]
    ) -> dict[str, int]:
        """Calculate entry type distribution."""
        distribution = {}
        for score in scores:
            entry_type = score.entry_type.value
            distribution[entry_type] = distribution.get(entry_type, 0) + 1
        return distribution

    def get_watchlist_summary(self, result: SwingWatchlistResult) -> dict[str, Any]:
        """
        Get summary of watchlist for dashboard display.
        
        Parameters
        ----------
        result
            Watchlist result
            
        Returns
        -------
        dict[str, Any]
            Summary data
        """
        return {
            "market_regime": result.market_regime.value,
            "market_regime_score": result.market_regime_score,
            "total_universe_size": result.total_universe_size,
            "qualified_count": result.qualified_count,
            "final_watchlist_size": result.final_watchlist_size,
            "avg_score": round(result.avg_score, 1),
            "avg_risk_reward_ratio": round(result.avg_risk_reward_ratio, 2),
            "sector_distribution": result.sector_distribution,
            "entry_type_distribution": result.entry_type_distribution,
            "top_stocks": [
                {
                    "symbol": s.symbol,
                    "name": s.name,
                    "sector": s.sector,
                    "final_score": round(s.final_score, 1),
                    "entry_type": s.entry_type.value,
                    "suggested_stop_loss_pct": s.suggested_stop_loss_pct,
                    "suggested_target_pct": s.suggested_target_pct,
                    "risk_reward_ratio": round(s.risk_reward_ratio, 2),
                    "has_base": s.base_formation.has_base if s.base_formation else False,
                    "base_type": s.base_formation.base_type.value if s.base_formation else None,
                    "catalyst": s.catalyst.catalyst_type.value if s.catalyst else None
                }
                for s in result.final_watchlist[:10]
            ],
            "timestamp": result.timestamp.isoformat()
        }
