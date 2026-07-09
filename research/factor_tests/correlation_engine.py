"""
Correlation Engine

Analyzes correlations between factors to identify duplicates and redundancy.
Clusters factors to keep independent alpha signals.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform

from utils.logger import get_logger

logger = get_logger("research.correlation_engine")


@dataclass
class CorrelationResult:
    """Result of correlation analysis."""
    factor_pairs: List[Tuple[str, str, float]]
    correlation_matrix: pd.DataFrame
    highly_correlated: List[Tuple[str, str, float]]
    clusters: Dict[int, List[str]]
    independent_factors: List[str]
    redundant_factors: List[str]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "factor_pairs": [(f1, f2, round(c, 4)) for f1, f2, c in self.factor_pairs],
            "correlation_matrix": self.correlation_matrix.to_dict(),
            "highly_correlated": [(f1, f2, round(c, 4)) for f1, f2, c in self.highly_correlated],
            "clusters": {str(k): v for k, v in self.clusters.items()},
            "independent_factors": self.independent_factors,
            "redundant_factors": self.redundant_factors,
        }


class CorrelationEngine:
    """
    Analyzes correlations between factors.
    
    Identifies duplicate/redundant factors and clusters them to keep
    independent alpha signals. This is critical for multi-factor models
    where combining independent signals is more valuable than having
    many versions of the same indicator.
    """
    
    def __init__(self, correlation_threshold: float = 0.7):
        """
        Initialize correlation engine.
        
        Args:
            correlation_threshold: Threshold for considering factors highly correlated
        """
        self.correlation_threshold = correlation_threshold
        self._logger = get_logger("research.correlation_engine")
    
    def analyze_correlations(
        self,
        factor_data: pd.DataFrame,
    ) -> CorrelationResult:
        """
        Analyze correlations between factors.
        
        Args:
            factor_data: DataFrame with factors as columns
            
        Returns:
            CorrelationResult with correlation analysis
        """
        # Calculate correlation matrix
        corr_matrix = factor_data.corr()
        
        # Get all factor pairs with correlations
        factor_pairs = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates and self-correlation
                    correlation = corr_matrix.loc[col1, col2]
                    factor_pairs.append((col1, col2, correlation))
        
        # Identify highly correlated pairs
        highly_correlated = [
            (f1, f2, corr) for f1, f2, corr in factor_pairs
            if abs(corr) >= self.correlation_threshold
        ]
        
        # Cluster factors based on correlation
        clusters = self._cluster_factors(corr_matrix)
        
        # Identify independent and redundant factors
        independent_factors, redundant_factors = self._identify_redundancy(
            clusters, highly_correlated
        )
        
        return CorrelationResult(
            factor_pairs=factor_pairs,
            correlation_matrix=corr_matrix,
            highly_correlated=highly_correlated,
            clusters=clusters,
            independent_factors=independent_factors,
            redundant_factors=redundant_factors,
        )
    
    def _cluster_factors(
        self,
        corr_matrix: pd.DataFrame,
    ) -> Dict[int, List[str]]:
        """
        Cluster factors using hierarchical clustering.
        
        Args:
            corr_matrix: Correlation matrix
            
        Returns:
            Dictionary mapping cluster IDs to factor names
        """
        # Convert correlation to distance (1 - abs(correlation))
        distance_matrix = 1 - corr_matrix.abs()
        
        # Handle NaN values
        distance_matrix = distance_matrix.fillna(1.0)
        
        # Perform hierarchical clustering
        condensed_dist = squareform(distance_matrix.values)
        linkage_matrix = linkage(condensed_dist, method='average')
        
        # Form clusters (threshold based on correlation threshold)
        cluster_threshold = 1 - self.correlation_threshold
        cluster_labels = fcluster(linkage_matrix, t=cluster_threshold, criterion='distance')
        
        # Group factors by cluster
        clusters = {}
        for factor, label in zip(corr_matrix.columns, cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(factor)
        
        return clusters
    
    def _identify_redundancy(
        self,
        clusters: Dict[int, List[str]],
        highly_correlated: List[Tuple[str, str, float]],
    ) -> Tuple[List[str], List[str]]:
        """
        Identify independent and redundant factors.
        
        Args:
            clusters: Factor clusters
            highly_correlated: List of highly correlated pairs
            
        Returns:
            Tuple of (independent_factors, redundant_factors)
        """
        independent_factors = []
        redundant_factors = []
        
        # For each cluster, keep one representative
        for cluster_id, factors in clusters.items():
            if len(factors) == 1:
                independent_factors.extend(factors)
            else:
                # Keep the first factor as representative, mark others as redundant
                independent_factors.append(factors[0])
                redundant_factors.extend(factors[1:])
        
        return independent_factors, redundant_factors
    
    def plot_correlation_heatmap(
        self,
        correlation_result: CorrelationResult,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot correlation heatmap for visualization.
        
        Args:
            correlation_result: CorrelationResult to plot
            save_path: Optional path to save plot
        """
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
            plt.title('Factor Correlation Matrix')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self._logger.info(f"Saved correlation heatmap to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            self._logger.warning("Matplotlib/Seaborn not available, skipping plot")
        except Exception as e:
            self._logger.error(f"Failed to plot correlation heatmap: {e}")
    
    def plot_dendrogram(
        self,
        correlation_result: CorrelationResult,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot dendrogram for factor clustering.
        
        Args:
            correlation_result: CorrelationResult to plot
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            
            # Convert correlation to distance
            distance_matrix = 1 - correlation_result.correlation_matrix.abs()
            distance_matrix = distance_matrix.fillna(1.0)
            
            # Perform hierarchical clustering
            condensed_dist = squareform(distance_matrix.values)
            linkage_matrix = linkage(condensed_dist, method='average')
            
            # Plot dendrogram
            plt.figure(figsize=(12, 8))
            dendrogram(
                linkage_matrix,
                labels=correlation_result.correlation_matrix.columns,
                leaf_rotation=90,
                leaf_font_size=10,
            )
            plt.title('Factor Clustering Dendrogram')
            plt.xlabel('Factors')
            plt.ylabel('Distance (1 - |Correlation|)')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self._logger.info(f"Saved dendrogram to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            self._logger.warning("Matplotlib not available, skipping plot")
        except Exception as e:
            self._logger.error(f"Failed to plot dendrogram: {e}")
    
    def get_recommended_subset(
        self,
        correlation_result: CorrelationResult,
        max_factors: Optional[int] = None,
    ) -> List[str]:
        """
        Get recommended subset of independent factors.
        
        Args:
            correlation_result: CorrelationResult
            max_factors: Optional maximum number of factors to return
            
        Returns:
            List of recommended factor names
        """
        recommended = correlation_result.independent_factors.copy()
        
        if max_factors and len(recommended) > max_factors:
            # Sort by some criterion (could be IC, Sharpe, etc.)
            # For now, just take first N
            recommended = recommended[:max_factors]
        
        return recommended


def analyze_factor_correlations(
    factor_data: pd.DataFrame,
    correlation_threshold: float = 0.7,
) -> CorrelationResult:
    """
    Convenience function to analyze factor correlations.
    
    Args:
        factor_data: DataFrame with factors as columns
        correlation_threshold: Threshold for high correlation
        
    Returns:
        CorrelationResult
    """
    engine = CorrelationEngine(correlation_threshold=correlation_threshold)
    return engine.analyze_correlations(factor_data)
