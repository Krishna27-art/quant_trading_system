"""
Regime and Sector Performance Tracker

Tracks performance by market regime and sector.
Shows where the model works best.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from research_platform.experiments.base import (
    RegimePerformance,
    SectorPerformance,
)
from utils.logger import get_logger

logger = get_logger("experiments.regime_sector_performance")


class RegimeSectorPerformanceTracker:
    """
    Regime and Sector Performance Tracker.
    
    Tracks:
    - Performance by market regime (bull, bear, sideways, high volatility)
    - Performance by sector (IT, Banks, Pharma, etc.)
    """
    
    def __init__(self):
        """Initialize tracker."""
        self.regime_performance: Dict[str, RegimePerformance] = {}
        self.sector_performance: Dict[str, SectorPerformance] = {}
        self._logger = get_logger("experiments.regime_sector_performance")
    
    def log_regime_performance(
        self,
        experiment_id: str,
        regime: str,
        returns: List[float],
    ) -> RegimePerformance:
        """
        Log performance for a specific regime.
        
        Args:
            experiment_id: Experiment ID
            regime: Regime name (bull, bear, sideways, high_volatility, etc.)
            returns: List of returns for this regime
            
        Returns:
            RegimePerformance object
        """
        performance_id = f"RP-{uuid.uuid4().hex[:8].upper()}"
        
        if not returns:
            raise ValueError("Returns list cannot be empty")
        
        returns_array = np.array(returns)
        
        # Calculate metrics
        total_trades = len(returns)
        winning_trades = sum(1 for r in returns if r > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Sharpe ratio
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        performance = RegimePerformance(
            performance_id=performance_id,
            experiment_id=experiment_id,
            regime=regime,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            total_trades=total_trades,
        )
        
        self.regime_performance[performance_id] = performance
        
        self._logger.info(
            f"Logged regime performance {performance_id}: "
            f"{regime} win_rate={win_rate:.2%}, sharpe={sharpe_ratio:.2f}"
        )
        
        return performance
    
    def log_sector_performance(
        self,
        experiment_id: str,
        sector: str,
        returns: List[float],
    ) -> SectorPerformance:
        """
        Log performance for a specific sector.
        
        Args:
            experiment_id: Experiment ID
            sector: Sector name (IT, Banks, Pharma, etc.)
            returns: List of returns for this sector
            
        Returns:
            SectorPerformance object
        """
        performance_id = f"SP-{uuid.uuid4().hex[:8].upper()}"
        
        if not returns:
            raise ValueError("Returns list cannot be empty")
        
        returns_array = np.array(returns)
        
        # Calculate metrics
        total_trades = len(returns)
        winning_trades = sum(1 for r in returns if r > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Sharpe ratio
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        performance = SectorPerformance(
            performance_id=performance_id,
            experiment_id=experiment_id,
            sector=sector,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            total_trades=total_trades,
        )
        
        self.sector_performance[performance_id] = performance
        
        self._logger.info(
            f"Logged sector performance {performance_id}: "
            f"{sector} win_rate={win_rate:.2%}, sharpe={sharpe_ratio:.2f}"
        )
        
        return performance
    
    def get_regime_performance(self, performance_id: str) -> Optional[RegimePerformance]:
        """Get regime performance by ID."""
        return self.regime_performance.get(performance_id)
    
    def get_sector_performance(self, performance_id: str) -> Optional[SectorPerformance]:
        """Get sector performance by ID."""
        return self.sector_performance.get(performance_id)
    
    def get_regime_performance_by_experiment(self, experiment_id: str) -> List[RegimePerformance]:
        """Get all regime performance for an experiment."""
        return [
            perf for perf in self.regime_performance.values()
            if perf.experiment_id == experiment_id
        ]
    
    def get_sector_performance_by_experiment(self, experiment_id: str) -> List[SectorPerformance]:
        """Get all sector performance for an experiment."""
        return [
            perf for perf in self.sector_performance.values()
            if perf.experiment_id == experiment_id
        ]
    
    def get_best_regime(
        self,
        experiment_id: str,
        metric: str = "win_rate",
    ) -> Optional[RegimePerformance]:
        """
        Get best performing regime for an experiment.
        
        Args:
            experiment_id: Experiment ID
            metric: Metric to compare (win_rate, sharpe_ratio)
            
        Returns:
            Best performing RegimePerformance
        """
        regime_perfs = self.get_regime_performance_by_experiment(experiment_id)
        
        if not regime_perfs:
            return None
        
        best = max(regime_perfs, key=lambda x: getattr(x, metric, 0))
        return best
    
    def get_best_sector(
        self,
        experiment_id: str,
        metric: str = "win_rate",
    ) -> Optional[SectorPerformance]:
        """
        Get best performing sector for an experiment.
        
        Args:
            experiment_id: Experiment ID
            metric: Metric to compare (win_rate, sharpe_ratio)
            
        Returns:
            Best performing SectorPerformance
        """
        sector_perfs = self.get_sector_performance_by_experiment(experiment_id)
        
        if not sector_perfs:
            return None
        
        best = max(sector_perfs, key=lambda x: getattr(x, metric, 0))
        return best
    
    def get_regime_summary(self, experiment_id: str) -> Dict:
        """
        Get summary of regime performance for an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Dictionary with regime summary
        """
        regime_perfs = self.get_regime_performance_by_experiment(experiment_id)
        
        summary = {
            'experiment_id': experiment_id,
            'regimes': {},
            'best_regime': None,
            'worst_regime': None,
        }
        
        if regime_perfs:
            best = max(regime_perfs, key=lambda x: x.win_rate)
            worst = min(regime_perfs, key=lambda x: x.win_rate)
            
            summary['best_regime'] = {
                'regime': best.regime,
                'win_rate': best.win_rate,
                'sharpe_ratio': best.sharpe_ratio,
            }
            
            summary['worst_regime'] = {
                'regime': worst.regime,
                'win_rate': worst.win_rate,
                'sharpe_ratio': worst.sharpe_ratio,
            }
            
            for perf in regime_perfs:
                summary['regimes'][perf.regime] = {
                    'win_rate': perf.win_rate,
                    'sharpe_ratio': perf.sharpe_ratio,
                    'total_trades': perf.total_trades,
                }
        
        return summary
    
    def get_sector_summary(self, experiment_id: str) -> Dict:
        """
        Get summary of sector performance for an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Dictionary with sector summary
        """
        sector_perfs = self.get_sector_performance_by_experiment(experiment_id)
        
        summary = {
            'experiment_id': experiment_id,
            'sectors': {},
            'best_sector': None,
            'worst_sector': None,
        }
        
        if sector_perfs:
            best = max(sector_perfs, key=lambda x: x.win_rate)
            worst = min(sector_perfs, key=lambda x: x.win_rate)
            
            summary['best_sector'] = {
                'sector': best.sector,
                'win_rate': best.win_rate,
                'sharpe_ratio': best.sharpe_ratio,
            }
            
            summary['worst_sector'] = {
                'sector': worst.sector,
                'win_rate': worst.win_rate,
                'sharpe_ratio': worst.sharpe_ratio,
            }
            
            for perf in sector_perfs:
                summary['sectors'][perf.sector] = {
                    'win_rate': perf.win_rate,
                    'sharpe_ratio': perf.sharpe_ratio,
                    'total_trades': perf.total_trades,
                }
        
        return summary
    
    def compare_regime_performance(
        self,
        experiment_id_1: str,
        experiment_id_2: str,
    ) -> Optional[Dict]:
        """
        Compare regime performance between two experiments.
        
        Args:
            experiment_id_1: First experiment ID
            experiment_id_2: Second experiment ID
            
        Returns:
            Dictionary with comparison results
        """
        regime_perfs_1 = self.get_regime_performance_by_experiment(experiment_id_1)
        regime_perfs_2 = self.get_regime_performance_by_experiment(experiment_id_2)
        
        if not regime_perfs_1 or not regime_perfs_2:
            return None
        
        comparison = {
            'experiment_1': {perf.regime: perf.to_dict() for perf in regime_perfs_1},
            'experiment_2': {perf.regime: perf.to_dict() for perf in regime_perfs_2},
            'differences': {},
        }
        
        # Compare common regimes
        regimes_1 = {perf.regime: perf for perf in regime_perfs_1}
        regimes_2 = {perf.regime: perf for perf in regime_perfs_2}
        
        common_regimes = set(regimes_1.keys()) & set(regimes_2.keys())
        
        for regime in common_regimes:
            perf_1 = regimes_1[regime]
            perf_2 = regimes_2[regime]
            
            comparison['differences'][regime] = {
                'win_rate_change': perf_2.win_rate - perf_1.win_rate,
                'sharpe_change': perf_2.sharpe_ratio - perf_1.sharpe_ratio,
            }
        
        return comparison
