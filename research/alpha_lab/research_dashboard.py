"""
Research Dashboard Generator

Generates comprehensive research dashboards for factor evaluation.
Creates IC charts, decay charts, regime performance, correlation matrix visualizations.
"""

from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd

from utils.logger import get_logger

logger = get_logger("research.research_dashboard")


class ResearchDashboard:
    """
    Generates comprehensive research dashboards for factor evaluation.
    
    Creates visualizations for:
    - IC charts over time
    - Decay charts
    - Regime performance
    - Sector performance
    - Correlation matrix
    - Factor rankings
    """
    
    def __init__(self, output_path: str = "research/factor_reports"):
        """
        Initialize research dashboard generator.
        
        Args:
            output_path: Path to save dashboard outputs
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self._logger = get_logger("research.research_dashboard")
    
    def generate_factor_dashboard(
        self,
        factor_name: str,
        factor_values: pd.Series,
        future_returns: pd.Series,
        market_prices: Optional[pd.Series] = None,
        sector_mapping: Optional[pd.Series] = None,
        timeframe_mapping: Optional[pd.Series] = None,
        existing_factors: Optional[pd.DataFrame] = None,
    ) -> Dict[str, str]:
        """
        Generate complete dashboard for a single factor.
        
        Args:
            factor_name: Name of factor
            factor_values: Series with factor values
            future_returns: Series with future returns
            market_prices: Optional market index prices
            sector_mapping: Optional sector mapping
            timeframe_mapping: Optional timeframe mapping
            existing_factors: Optional DataFrame with existing factors
            
        Returns:
            Dictionary mapping chart names to file paths
        """
        from research.factor_tests.information_coefficient import calculate_ic
        from research.factor_tests.signal_decay import analyze_signal_decay
        from research.factor_tests.correlation_engine import analyze_factor_correlations
        from research.regime_engine.market_regime import MarketRegimeClassifier
        from research.regime_engine.sector_analysis import SectorAnalyzer
        from research.regime_engine.timeframe_analysis import TimeframeAnalyzer
        
        charts = {}
        
        # 1. IC Chart
        try:
            ic_result = calculate_ic(factor_values, future_returns)
            ic_chart_path = self._plot_ic_chart(factor_name, ic_result)
            charts["ic_chart"] = ic_chart_path
        except Exception as e:
            self._logger.error(f"Failed to generate IC chart: {e}")
        
        # 2. Decay Chart
        try:
            decay_result = analyze_signal_decay(factor_values, future_returns)
            decay_chart_path = self._plot_decay_chart(factor_name, decay_result)
            charts["decay_chart"] = decay_chart_path
        except Exception as e:
            self._logger.error(f"Failed to generate decay chart: {e}")
        
        # 3. Regime Performance Chart
        if market_prices is not None:
            try:
                regime_classifier = MarketRegimeClassifier()
                regime_performance = regime_classifier.analyze_factor_by_regime(
                    factor_values, future_returns, market_prices
                )
                regime_chart_path = self._plot_regime_performance(factor_name, regime_performance)
                charts["regime_chart"] = regime_chart_path
            except Exception as e:
                self._logger.error(f"Failed to generate regime chart: {e}")
        
        # 4. Sector Performance Chart
        if sector_mapping is not None:
            try:
                sector_analyzer = SectorAnalyzer()
                sector_performance = sector_analyzer.analyze_factor_by_sector(
                    factor_values, future_returns, sector_mapping
                )
                sector_chart_path = self._plot_sector_performance(factor_name, sector_performance)
                charts["sector_chart"] = sector_chart_path
            except Exception as e:
                self._logger.error(f"Failed to generate sector chart: {e}")
        
        # 5. Correlation Matrix
        if existing_factors is not None:
            try:
                existing_factors[factor_name] = factor_values
                correlation_result = analyze_factor_correlations(existing_factors)
                correlation_chart_path = self._plot_correlation_matrix(factor_name, correlation_result)
                charts["correlation_chart"] = correlation_chart_path
            except Exception as e:
                self._logger.error(f"Failed to generate correlation chart: {e}")
        
        self._logger.info(f"Generated dashboard for {factor_name} with {len(charts)} charts")
        return charts
    
    def _plot_ic_chart(
        self,
        factor_name: str,
        ic_result,
    ) -> str:
        """Plot IC chart."""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Rolling IC
            ax1.plot(ic_result.rolling_ic.index, ic_result.rolling_ic.values, label='IC', color='blue')
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax1.set_ylabel('Information Coefficient')
            ax1.set_title(f'{factor_name} - Rolling IC')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Rolling Rank IC
            ax2.plot(ic_result.rolling_rank_ic.index, ic_result.rolling_rank_ic.values, label='Rank IC', color='red')
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Rank IC')
            ax2.set_title(f'{factor_name} - Rolling Rank IC')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filepath = self.output_path / f"{factor_name}_ic_chart.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        except ImportError:
            self._logger.warning("Matplotlib not available, skipping IC chart")
            return ""
    
    def _plot_decay_chart(
        self,
        factor_name: str,
        decay_result,
    ) -> str:
        """Plot decay chart."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(decay_result.decay_horizons, decay_result.decay_ics, 'b-o', label='IC')
            plt.plot(decay_result.decay_horizons, decay_result.decay_rank_ics, 'r-s', label='Rank IC')
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            plt.axvline(x=decay_result.optimal_horizon, color='green', linestyle='--', alpha=0.5, label=f'Optimal: {decay_result.optimal_horizon}')
            plt.xlabel('Horizon (days)')
            plt.ylabel('Information Coefficient')
            plt.title(f'{factor_name} - Signal Decay')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            filepath = self.output_path / f"{factor_name}_decay_chart.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        except ImportError:
            self._logger.warning("Matplotlib not available, skipping decay chart")
            return ""
    
    def _plot_regime_performance(
        self,
        factor_name: str,
        regime_performance: Dict,
    ) -> str:
        """Plot regime performance chart."""
        try:
            import matplotlib.pyplot as plt
            
            regimes = list(regime_performance.keys())
            ics = [regime_performance[r]["mean_ic"] for r in regimes]
            
            plt.figure(figsize=(10, 6))
            colors = ['green' if ic > 0 else 'red' for ic in ics]
            plt.bar(regimes, ics, color=colors)
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            plt.xlabel('Regime')
            plt.ylabel('Mean IC')
            plt.title(f'{factor_name} - Performance by Regime')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            filepath = self.output_path / f"{factor_name}_regime_chart.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        except ImportError:
            self._logger.warning("Matplotlib not available, skipping regime chart")
            return ""
    
    def _plot_sector_performance(
        self,
        factor_name: str,
        sector_performance: Dict,
    ) -> str:
        """Plot sector performance chart."""
        try:
            import matplotlib.pyplot as plt
            
            sectors = list(sector_performance.keys())
            ics = [sector_performance[s].mean_ic for s in sectors]
            
            plt.figure(figsize=(12, 6))
            colors = ['green' if ic > 0 else 'red' for ic in ics]
            plt.barh(sectors, ics, color=colors)
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            plt.xlabel('Mean IC')
            plt.ylabel('Sector')
            plt.title(f'{factor_name} - Performance by Sector')
            plt.tight_layout()
            
            filepath = self.output_path / f"{factor_name}_sector_chart.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        except ImportError:
            self._logger.warning("Matplotlib not available, skipping sector chart")
            return ""
    
    def _plot_correlation_matrix(
        self,
        factor_name: str,
        correlation_result,
    ) -> str:
        """Plot correlation matrix."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                correlation_result.correlation_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f',
                cbar_kws={'label': 'Correlation'},
            )
            plt.title(f'{factor_name} - Factor Correlation Matrix')
            plt.tight_layout()
            
            filepath = self.output_path / f"{factor_name}_correlation_matrix.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        except ImportError:
            self._logger.warning("Matplotlib/Seaborn not available, skipping correlation matrix")
            return ""
    
    def generate_summary_dashboard(
        self,
        discovery_results: List,
    ) -> str:
        """
        Generate summary dashboard for multiple factors.
        
        Args:
            discovery_results: List of DiscoveryResult
            
        Returns:
            Path to summary dashboard
        """
        try:
            import matplotlib.pyplot as plt
            
            # Extract data
            factor_names = [r.factor_name for r in discovery_results]
            ics = [r.details.get("ic", {}).get("mean_ic", 0) for r in discovery_results]
            sharpes = [r.details.get("performance", {}).get("sharpe_ratio", 0) for r in discovery_results]
            confidences = [r.confidence for r in discovery_results]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # IC comparison
            colors_ic = ['green' if ic > 0 else 'red' for ic in ics]
            axes[0, 0].barh(factor_names, ics, color=colors_ic)
            axes[0, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            axes[0, 0].set_xlabel('Mean IC')
            axes[0, 0].set_title('Factor IC Comparison')
            
            # Sharpe comparison
            axes[0, 1].barh(factor_names, sharpes)
            axes[0, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            axes[0, 1].set_xlabel('Sharpe Ratio')
            axes[0, 1].set_title('Factor Sharpe Comparison')
            
            # Confidence comparison
            axes[1, 0].barh(factor_names, confidences)
            axes[1, 0].set_xlabel('Confidence')
            axes[1, 0].set_title('Factor Confidence Comparison')
            
            # Promotion status
            promote_status = ['Yes' if r.should_promote else 'No' for r in discovery_results]
            promote_colors = ['green' if s == 'Yes' else 'red' for s in promote_status]
            axes[1, 1].barh(factor_names, [1] * len(factor_names), color=promote_colors)
            axes[1, 1].set_xlim(0, 1.2)
            axes[1, 1].set_yticks([])
            axes[1, 1].set_title('Promotion Status')
            
            plt.tight_layout()
            
            filepath = self.output_path / "summary_dashboard.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        except ImportError:
            self._logger.warning("Matplotlib not available, skipping summary dashboard")
            return ""


def generate_research_dashboard(
    factor_name: str,
    factor_values: pd.Series,
    future_returns: pd.Series,
    market_prices: Optional[pd.Series] = None,
    sector_mapping: Optional[pd.Series] = None,
    timeframe_mapping: Optional[pd.Series] = None,
    existing_factors: Optional[pd.DataFrame] = None,
    output_path: str = "research/factor_reports",
) -> Dict[str, str]:
    """
    Convenience function to generate research dashboard.
    
    Args:
        factor_name: Name of factor
        factor_values: Series with factor values
        future_returns: Series with future returns
        market_prices: Optional market index prices
        sector_mapping: Optional sector mapping
        timeframe_mapping: Optional timeframe mapping
        existing_factors: Optional DataFrame with existing factors
        output_path: Path to save outputs
        
    Returns:
        Dictionary mapping chart names to file paths
    """
    dashboard = ResearchDashboard(output_path=output_path)
    return dashboard.generate_factor_dashboard(
        factor_name=factor_name,
        factor_values=factor_values,
        future_returns=future_returns,
        market_prices=market_prices,
        sector_mapping=sector_mapping,
        timeframe_mapping=timeframe_mapping,
        existing_factors=existing_factors,
    )
