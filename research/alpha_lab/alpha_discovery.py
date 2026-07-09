"""
Alpha Discovery Pipeline

Automatic factor evaluation and promotion pipeline.
Integrates all research components to discover, validate, and promote alpha factors.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from utils.logger import get_logger

logger = get_logger("research.alpha_discovery")


@dataclass
class DiscoveryResult:
    """Result of alpha discovery for a single factor."""
    factor_name: str
    works: bool
    when: str
    where: str
    why: str
    how_long: str
    independent: bool
    should_promote: bool
    confidence: float
    details: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "factor_name": self.factor_name,
            "works": self.works,
            "when": self.when,
            "where": self.where,
            "why": self.why,
            "how_long": self.how_long,
            "independent": self.independent,
            "should_promote": self.should_promote,
            "confidence": round(self.confidence, 4),
            "details": self.details,
        }


class AlphaDiscoveryPipeline:
    """
    Automatic factor evaluation and promotion pipeline.
    
    Integrates all research components to:
    - Discover if a factor works
    - Understand when it works (regimes, timeframes)
    - Understand where it works (sectors)
    - Understand why it works (IC, Sharpe, etc.)
    - Understand how long it works (signal decay)
    - Check if it's independent from existing factors
    - Decide whether to promote to production
    """
    
    def __init__(
        self,
        min_ic: float = 0.02,
        min_sharpe: float = 0.5,
        min_hit_rate: float = 0.51,
        correlation_threshold: float = 0.7,
    ):
        """
        Initialize alpha discovery pipeline.
        
        Args:
            min_ic: Minimum acceptable IC
            min_sharpe: Minimum acceptable Sharpe
            min_hit_rate: Minimum acceptable hit rate
            correlation_threshold: Threshold for factor independence
        """
        self.min_ic = min_ic
        self.min_sharpe = min_sharpe
        self.min_hit_rate = min_hit_rate
        self.correlation_threshold = correlation_threshold
        self._logger = get_logger("research.alpha_discovery")
    
    def discover(
        self,
        factor_name: str,
        factor_values: pd.Series,
        future_returns: pd.Series,
        market_prices: Optional[pd.Series] = None,
        sector_mapping: Optional[pd.Series] = None,
        timeframe_mapping: Optional[pd.Series] = None,
        existing_factors: Optional[pd.DataFrame] = None,
    ) -> DiscoveryResult:
        """
        Run complete alpha discovery for a factor.
        
        Args:
            factor_name: Name of factor
            factor_values: Series with factor values
            future_returns: Series with future returns
            market_prices: Optional market index prices for regime analysis
            sector_mapping: Optional sector mapping for sector analysis
            timeframe_mapping: Optional timeframe mapping for timeframe analysis
            existing_factors: Optional DataFrame with existing factor values for correlation check
            
        Returns:
            DiscoveryResult with complete analysis
        """
        from research.factor_tests.information_coefficient import calculate_ic
        from research.factor_tests.performance_metrics import calculate_performance_metrics
        from research.factor_tests.signal_decay import analyze_signal_decay
        from research.factor_tests.correlation_engine import analyze_factor_correlations
        from research.regime_engine.market_regime import MarketRegimeClassifier
        from research.regime_engine.sector_analysis import SectorAnalyzer
        from research.regime_engine.timeframe_analysis import TimeframeAnalyzer
        
        details = {}
        
        # 1. Does it work? (IC, Sharpe, Hit Rate)
        ic_result = calculate_ic(factor_values, future_returns)
        perf_metrics = calculate_performance_metrics(factor_values, future_returns)
        
        works = (
            abs(ic_result.mean_ic) >= self.min_ic and
            perf_metrics.sharpe_ratio >= self.min_sharpe and
            ic_result.hit_rate >= self.min_hit_rate
        )
        
        details["ic"] = ic_result.to_dict()
        details["performance"] = perf_metrics.to_dict()
        
        # 2. When does it work? (Regime analysis)
        when = "Unknown"
        if market_prices is not None:
            regime_classifier = MarketRegimeClassifier()
            regime_performance = regime_classifier.analyze_factor_by_regime(
                factor_values, future_returns, market_prices
            )
            details["regime_performance"] = regime_performance
            
            # Identify best regime
            best_regime = max(regime_performance.items(), key=lambda x: x[1]["mean_ic"])
            when = f"Best in {best_regime[0]} regime"
        
        # 3. Where does it work? (Sector analysis)
        where = "Unknown"
        if sector_mapping is not None:
            sector_analyzer = SectorAnalyzer()
            sector_performance = sector_analyzer.analyze_factor_by_sector(
                factor_values, future_returns, sector_mapping
            )
            details["sector_performance"] = {
                k: v.to_dict() for k, v in sector_performance.items()
            }
            
            # Identify best sector
            if sector_performance:
                best_sector = max(sector_performance.items(), key=lambda x: x[1].mean_ic)
                where = f"Best in {best_sector[0]} sector"
        
        # 4. Why does it work? (IC, Sharpe, etc.)
        why = f"IC={ic_result.mean_ic:.4f}, Sharpe={perf_metrics.sharpe_ratio:.2f}, Hit Rate={ic_result.hit_rate:.2%}"
        
        # 5. How long does it work? (Signal decay)
        decay_result = analyze_signal_decay(factor_values, future_returns)
        details["decay"] = decay_result.to_dict()
        how_long = f"Optimal holding period: {decay_result.optimal_horizon} days"
        
        # 6. Is it independent? (Correlation with existing factors)
        independent = True
        if existing_factors is not None:
            existing_factors[factor_name] = factor_values
            correlation_result = analyze_factor_correlations(existing_factors)
            independent = factor_name not in correlation_result.redundant_factors
            details["correlation"] = correlation_result.to_dict()
        
        # 7. Should it be promoted?
        should_promote = works and independent
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            ic_result.mean_ic,
            perf_metrics.sharpe_ratio,
            ic_result.hit_rate,
            decay_result.persistence_score,
        )
        
        return DiscoveryResult(
            factor_name=factor_name,
            works=works,
            when=when,
            where=where,
            why=why,
            how_long=how_long,
            independent=independent,
            should_promote=should_promote,
            confidence=confidence,
            details=details,
        )
    
    def _calculate_confidence(
        self,
        ic: float,
        sharpe: float,
        hit_rate: float,
        persistence: float,
    ) -> float:
        """Calculate overall confidence in factor."""
        # Normalize each metric to 0-1
        ic_score = min(abs(ic) / 0.1, 1.0)
        sharpe_score = min(sharpe / 2.0, 1.0)
        hit_rate_score = (hit_rate - 0.5) / 0.5
        persistence_score = min(persistence / 0.05, 1.0)
        
        # Weighted average
        confidence = 0.4 * ic_score + 0.3 * sharpe_score + 0.2 * hit_rate_score + 0.1 * persistence_score
        
        return max(0.0, min(1.0, confidence))
    
    def discover_batch(
        self,
        factor_data: Dict[str, pd.Series],
        future_returns: pd.Series,
        market_prices: Optional[pd.Series] = None,
        sector_mapping: Optional[pd.Series] = None,
        timeframe_mapping: Optional[pd.Series] = None,
    ) -> List[DiscoveryResult]:
        """
        Run alpha discovery for multiple factors.
        
        Args:
            factor_data: Dictionary mapping factor names to values
            future_returns: Series with future returns
            market_prices: Optional market index prices
            sector_mapping: Optional sector mapping
            timeframe_mapping: Optional timeframe mapping
            
        Returns:
            List of DiscoveryResult
        """
        results = []
        
        # Build existing factors DataFrame for correlation check
        existing_factors = pd.DataFrame(factor_data)
        
        for factor_name, factor_values in factor_data.items():
            try:
                result = self.discover(
                    factor_name=factor_name,
                    factor_values=factor_values,
                    future_returns=future_returns,
                    market_prices=market_prices,
                    sector_mapping=sector_mapping,
                    timeframe_mapping=timeframe_mapping,
                    existing_factors=existing_factors.drop(columns=[factor_name], errors='ignore'),
                )
                results.append(result)
            except Exception as e:
                self._logger.error(f"Failed to discover factor {factor_name}: {e}")
        
        return results
    
    def get_promotion_candidates(
        self,
        results: List[DiscoveryResult],
        min_confidence: float = 0.6,
    ) -> List[DiscoveryResult]:
        """
        Get factors that should be promoted to production.
        
        Args:
            results: List of DiscoveryResult
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of DiscoveryResult that should be promoted
        """
        candidates = [
            r for r in results
            if r.should_promote and r.confidence >= min_confidence
        ]
        
        # Sort by confidence
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        return candidates


def run_alpha_discovery(
    factor_name: str,
    factor_values: pd.Series,
    future_returns: pd.Series,
    market_prices: Optional[pd.Series] = None,
    sector_mapping: Optional[pd.Series] = None,
    timeframe_mapping: Optional[pd.Series] = None,
    existing_factors: Optional[pd.DataFrame] = None,
) -> DiscoveryResult:
    """
    Convenience function to run alpha discovery.
    
    Args:
        factor_name: Name of factor
        factor_values: Series with factor values
        future_returns: Series with future returns
        market_prices: Optional market index prices
        sector_mapping: Optional sector mapping
        timeframe_mapping: Optional timeframe mapping
        existing_factors: Optional DataFrame with existing factors
        
    Returns:
        DiscoveryResult
    """
    pipeline = AlphaDiscoveryPipeline()
    return pipeline.discover(
        factor_name=factor_name,
        factor_values=factor_values,
        future_returns=future_returns,
        market_prices=market_prices,
        sector_mapping=sector_mapping,
        timeframe_mapping=timeframe_mapping,
        existing_factors=existing_factors,
    )
