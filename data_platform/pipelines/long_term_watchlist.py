"""
Long-Term Investing Watchlist - Institutional Portfolio Builder

Integrates all long-term investing components into a unified portfolio builder:
- Industry/theme analysis for structural trends
- Business quality scoring
- Management quality assessment
- Financial quality analysis
- Competitive moat detection
- Valuation analysis (GARP/GARV)
- Scalability assessment
- Conviction-based position sizing

Output: 15-25 high-conviction long-term candidates with position sizing
based on conviction level. Focus on businesses that can compound
earnings for 10-30 years.

Institutional portfolio construction:
1. Score universe (0-100)
2. Filter by minimum quality (80+)
3. Assess diversification needs
4. Allocate by conviction
5. Monitor quarterly
6. Rebalance annually
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from data_platform.pipelines.long_term_investing_scorer import (
    LongTermInvestingScorer,
    LongTermInvestingScore
)
from utils.logger import get_logger

logger = get_logger("long_term_watchlist")


@dataclass
class LongTermPortfolioResult:
    """Complete long-term investing portfolio result."""
    
    # All scored stocks
    all_scores: list[LongTermInvestingScore]
    
    # Qualified stocks (score >= 80)
    qualified_stocks: list[LongTermInvestingScore]
    
    # Final portfolio (15-25 stocks)
    final_portfolio: list[LongTermInvestingScore]
    
    # Portfolio metrics
    total_allocation_pct: float
    avg_score: float
    avg_expected_10y_return_pct: float
    
    # Diversification metrics
    sector_distribution: dict[str, float]  # sector -> allocation %
    theme_distribution: dict[str, float]  # theme -> allocation %
    conviction_distribution: dict[str, int]  # conviction -> count
    
    # Multibagger potential
    multibagger_distribution: dict[str, int]  # potential -> count
    
    # Risk metrics
    avg_debt_to_equity: float
    avg_pe_ratio: float
    portfolio_risk_score: float  # 0-100
    
    # Statistics
    total_universe_size: int
    qualified_count: int
    final_portfolio_size: int
    
    timestamp: datetime


class LongTermPortfolioBuilder:
    """
    Builds institutional long-term investing portfolios.
    
    Process:
    1. Score universe (0-100)
    2. Filter by minimum quality (80+)
    3. Ensure sector diversification
    4. Allocate by conviction level
    5. Validate portfolio risk
    6. Select 15-25 candidates
    7. Generate monitoring schedule
    """

    def __init__(
        self,
        min_score_threshold: float = 80.0,
        target_portfolio_size: int = 20,
        max_sector_allocation_pct: float = 25.0,
        max_single_position_pct: float = 10.0,
        use_mock_data: bool = True
    ):
        """
        Initialize long-term portfolio builder.
        
        Parameters
        ----------
        min_score_threshold
            Minimum score to qualify (default: 80)
        target_portfolio_size
            Target number of stocks in portfolio
        max_sector_allocation_pct
            Maximum allocation to any single sector
        max_single_position_pct
            Maximum allocation to any single position
        use_mock_data
            Use mock data for development
        """
        self.min_score_threshold = min_score_threshold
        self.target_portfolio_size = target_portfolio_size
        self.max_sector_allocation_pct = max_sector_allocation_pct
        self.max_single_position_pct = max_single_position_pct
        self.use_mock_data = use_mock_data
        
        # Initialize scorer
        self.scorer = LongTermInvestingScorer(
            min_score_threshold=min_score_threshold,
            use_mock_data=use_mock_data
        )

    def build_portfolio(self) -> LongTermPortfolioResult:
        """
        Build complete long-term investing portfolio.
        
        Returns
        -------
        LongTermPortfolioResult
            Complete portfolio with all analysis
        """
        logger.info("Building institutional long-term investing portfolio")
        
        # Step 1: Score universe
        logger.info("Step 1: Scoring universe")
        all_scores = self.scorer.score_universe()
        
        # Step 2: Filter by minimum quality
        logger.info("Step 2: Filtering by minimum quality")
        qualified = [s for s in all_scores if s.qualifies]
        logger.info(f"Qualified stocks: {len(qualified)}/{len(all_scores)}")
        
        # Step 3: Apply diversification constraints
        logger.info("Step 3: Applying diversification constraints")
        diversified = self._apply_diversification_constraints(qualified)
        logger.info(f"After diversification filter: {len(diversified)}")
        
        # Step 4: Select top candidates by score
        logger.info("Step 4: Selecting top candidates")
        top_candidates = sorted(diversified, key=lambda x: x.final_score, reverse=True)
        top_candidates = top_candidates[:self.target_portfolio_size * 2]  # Get 2x for selection
        
        # Step 5: Final portfolio selection with conviction sizing
        logger.info("Step 5: Final portfolio selection")
        final_portfolio = self._select_final_portfolio(top_candidates)
        logger.info(f"Final portfolio size: {len(final_portfolio)}")
        
        # Step 6: Calculate portfolio metrics
        logger.info("Step 6: Calculating portfolio metrics")
        sector_dist = self._calculate_sector_distribution(final_portfolio)
        theme_dist = self._calculate_theme_distribution(final_portfolio)
        conviction_dist = self._calculate_conviction_distribution(final_portfolio)
        multibagger_dist = self._calculate_multibagger_distribution(final_portfolio)
        
        total_allocation = sum(s.suggested_allocation_pct for s in final_portfolio)
        avg_score = sum(s.final_score for s in final_portfolio) / len(final_portfolio) if final_portfolio else 0
        avg_return = sum(s.expected_10y_return_pct for s in final_portfolio) / len(final_portfolio) if final_portfolio else 0
        
        # Risk metrics
        avg_debt = self._calculate_avg_debt(final_portfolio)
        avg_pe = self._calculate_avg_pe(final_portfolio)
        portfolio_risk = self._calculate_portfolio_risk(final_portfolio)
        
        result = LongTermPortfolioResult(
            all_scores=all_scores,
            qualified_stocks=qualified,
            final_portfolio=final_portfolio,
            total_allocation_pct=total_allocation,
            avg_score=avg_score,
            avg_expected_10y_return_pct=avg_return,
            sector_distribution=sector_dist,
            theme_distribution=theme_dist,
            conviction_distribution=conviction_dist,
            multibagger_distribution=multibagger_dist,
            avg_debt_to_equity=avg_debt,
            avg_pe_ratio=avg_pe,
            portfolio_risk_score=portfolio_risk,
            total_universe_size=len(all_scores),
            qualified_count=len(qualified),
            final_portfolio_size=len(final_portfolio),
            timestamp=datetime.now()
        )
        
        logger.info(
            f"Portfolio build complete: {len(final_portfolio)} stocks, "
            f"avg_score={avg_score:.1f}, avg_return={avg_return:.1f}%, "
            f"total_allocation={total_allocation:.1f}%"
        )
        
        return result

    def _apply_diversification_constraints(
        self,
        scores: list[LongTermInvestingScore]
    ) -> list[LongTermInvestingScore]:
        """Apply diversification constraints to avoid concentration."""
        # Count stocks per sector
        sector_counts = {}
        for score in scores:
            sector_counts[score.sector] = sector_counts.get(score.sector, 0) + 1
        
        # Limit each sector to max 30% of qualified list
        max_per_sector = max(3, int(len(scores) * 0.3))
        
        diversified = []
        sector_used = {}
        
        for score in scores:
            sector = score.sector
            used = sector_used.get(sector, 0)
            
            if used < max_per_sector:
                diversified.append(score)
                sector_used[sector] = used + 1
        
        return diversified

    def _select_final_portfolio(
        self,
        candidates: list[LongTermInvestingScore]
    ) -> list[LongTermInvestingScore]:
        """Select final portfolio with conviction-based sizing."""
        # Sort by score
        sorted_candidates = sorted(candidates, key=lambda x: x.final_score, reverse=True)
        
        # Select top N
        final = sorted_candidates[:self.target_portfolio_size]
        
        # Adjust allocations to respect max single position
        for stock in final:
            if stock.suggested_allocation_pct > self.max_single_position_pct:
                stock.suggested_allocation_pct = self.max_single_position_pct
        
        # Normalize to 100%
        total = sum(s.suggested_allocation_pct for s in final)
        if total > 0:
            for stock in final:
                stock.suggested_allocation_pct = (stock.suggested_allocation_pct / total) * 100
        
        return final

    def _calculate_sector_distribution(
        self,
        portfolio: list[LongTermInvestingScore]
    ) -> dict[str, float]:
        """Calculate sector distribution by allocation."""
        distribution = {}
        for stock in portfolio:
            sector = stock.sector
            allocation = stock.suggested_allocation_pct
            distribution[sector] = distribution.get(sector, 0) + allocation
        return distribution

    def _calculate_theme_distribution(
        self,
        portfolio: list[LongTermInvestingScore]
    ) -> dict[str, float]:
        """Calculate theme distribution by allocation."""
        distribution = {}
        for stock in portfolio:
            if stock.industry_theme and stock.industry_theme.theme:
                theme = stock.industry_theme.theme.value
                allocation = stock.suggested_allocation_pct
                distribution[theme] = distribution.get(theme, 0) + allocation
        return distribution

    def _calculate_conviction_distribution(
        self,
        portfolio: list[LongTermInvestingScore]
    ) -> dict[str, int]:
        """Calculate conviction distribution."""
        distribution = {}
        for stock in portfolio:
            conviction = stock.conviction_level
            distribution[conviction] = distribution.get(conviction, 0) + 1
        return distribution

    def _calculate_multibagger_distribution(
        self,
        portfolio: list[LongTermInvestingScore]
    ) -> dict[str, int]:
        """Calculate multibagger potential distribution."""
        distribution = {}
        for stock in portfolio:
            multibagger = stock.potential_multibagger
            distribution[multibagger] = distribution.get(multibagger, 0) + 1
        return distribution

    def _calculate_avg_debt(self, portfolio: list[LongTermInvestingScore]) -> float:
        """Calculate average debt-to-equity."""
        if not portfolio:
            return 0.0
        
        debt_values = []
        for stock in portfolio:
            if stock.financial_quality:
                debt_values.append(stock.financial_quality.debt_to_equity)
        
        return sum(debt_values) / len(debt_values) if debt_values else 0.0

    def _calculate_avg_pe(self, portfolio: list[LongTermInvestingScore]) -> float:
        """Calculate average PE ratio."""
        if not portfolio:
            return 0.0
        
        pe_values = []
        for stock in portfolio:
            if stock.valuation:
                pe_values.append(stock.valuation.pe_ratio)
        
        return sum(pe_values) / len(pe_values) if pe_values else 0.0

    def _calculate_portfolio_risk(self, portfolio: list[LongTermInvestingScore]) -> float:
        """Calculate portfolio risk score (0-100, higher is riskier)."""
        if not portfolio:
            return 50.0
        
        risk_factors = []
        
        # Debt risk
        avg_debt = self._calculate_avg_debt(portfolio)
        debt_risk = min(100, avg_debt * 40)
        risk_factors.append(debt_risk)
        
        # Valuation risk
        avg_pe = self._calculate_avg_pe(portfolio)
        valuation_risk = min(100, (avg_pe - 20) * 2) if avg_pe > 20 else 0
        risk_factors.append(valuation_risk)
        
        # Concentration risk
        max_allocation = max(s.suggested_allocation_pct for s in portfolio) if portfolio else 0
        concentration_risk = max_allocation * 5
        risk_factors.append(concentration_risk)
        
        # Quality risk (inverse of avg score)
        avg_score = sum(s.final_score for s in portfolio) / len(portfolio)
        quality_risk = max(0, 100 - avg_score)
        risk_factors.append(quality_risk)
        
        return sum(risk_factors) / len(risk_factors)

    def get_portfolio_summary(self, result: LongTermPortfolioResult) -> dict[str, Any]:
        """
        Get summary of portfolio for dashboard display.
        
        Parameters
        ----------
        result
            Portfolio result
            
        Returns
        -------
        dict[str, Any]
            Summary data
        """
        return {
            "total_universe_size": result.total_universe_size,
            "qualified_count": result.qualified_count,
            "final_portfolio_size": result.final_portfolio_size,
            "avg_score": round(result.avg_score, 1),
            "avg_expected_10y_return_pct": round(result.avg_expected_10y_return_pct, 1),
            "total_allocation_pct": round(result.total_allocation_pct, 1),
            "avg_debt_to_equity": round(result.avg_debt_to_equity, 2),
            "avg_pe_ratio": round(result.avg_pe_ratio, 1),
            "portfolio_risk_score": round(result.portfolio_risk_score, 1),
            "sector_distribution": result.sector_distribution,
            "theme_distribution": result.theme_distribution,
            "conviction_distribution": result.conviction_distribution,
            "multibagger_distribution": result.multibagger_distribution,
            "top_holdings": [
                {
                    "symbol": s.symbol,
                    "name": s.name,
                    "sector": s.sector,
                    "final_score": round(s.final_score, 1),
                    "conviction_level": s.conviction_level,
                    "suggested_allocation_pct": round(s.suggested_allocation_pct, 1),
                    "expected_10y_return_pct": round(s.expected_10y_return_pct, 1),
                    "potential_multibagger": s.potential_multibagger,
                    "investment_thesis": s.investment_thesis,
                    "key_risks": s.key_risks,
                    "theme": s.industry_theme.theme.value if s.industry_theme and s.industry_theme.theme else None,
                    "has_moat": s.moat.has_moat if s.moat else False,
                    "moat_type": s.moat.moat_type.value if s.moat and s.moat.moat_type else None
                }
                for s in result.final_portfolio[:15]
            ],
            "timestamp": result.timestamp.isoformat()
        }
