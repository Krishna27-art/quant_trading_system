"""
Experiment Comparison Engine

Compares experiments side by side.
Enables systematic comparison of research results.
"""

from typing import Dict, List, Optional

from research_platform.experiments.base import (
    Experiment,
    TrainingMetrics,
    TradingMetrics,
)
from utils.logger import get_logger

logger = get_logger("experiments.comparison")


class ExperimentComparisonEngine:
    """
    Experiment Comparison Engine.
    
    Compares:
    - Multiple experiments side by side
    - Trading metrics
    - Training metrics
    - Feature importance
    - Configurations
    """
    
    def __init__(self):
        """Initialize comparison engine."""
        self._logger = get_logger("experiments.comparison")
    
    def compare_experiments(
        self,
        experiments: List[Experiment],
        trading_metrics_map: Dict[str, TradingMetrics],
        training_metrics_map: Optional[Dict[str, TrainingMetrics]] = None,
    ) -> Dict:
        """
        Compare multiple experiments.
        
        Args:
            experiments: List of Experiment objects
            trading_metrics_map: Dictionary mapping experiment IDs to TradingMetrics
            training_metrics_map: Optional dictionary mapping experiment IDs to TrainingMetrics
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {
            'experiments': [],
            'trading_metrics_comparison': [],
            'training_metrics_comparison': [],
            'rankings': {},
        }
        
        # Build experiment data
        for exp in experiments:
            exp_data = {
                'experiment_id': exp.experiment_id,
                'name': exp.name,
                'type': exp.experiment_type.value,
                'status': exp.status.value,
                'decision': exp.decision.value if exp.decision else None,
                'created_at': exp.created_at.isoformat(),
            }
            comparison['experiments'].append(exp_data)
        
        # Compare trading metrics
        trading_comparison = []
        for exp in experiments:
            metrics = trading_metrics_map.get(exp.experiment_id)
            if metrics:
                trading_comparison.append({
                    'experiment_id': exp.experiment_id,
                    'name': exp.name,
                    'win_rate': metrics.win_rate,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'sortino_ratio': metrics.sortino_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'average_return': metrics.average_return,
                    'profit_factor': metrics.profit_factor,
                    'total_trades': metrics.total_trades,
                })
        
        comparison['trading_metrics_comparison'] = trading_comparison
        
        # Compare training metrics if available
        if training_metrics_map:
            training_comparison = []
            for exp in experiments:
                metrics = training_metrics_map.get(exp.experiment_id)
                if metrics:
                    training_comparison.append({
                        'experiment_id': exp.experiment_id,
                        'name': exp.name,
                        'accuracy': metrics.accuracy,
                        'precision': metrics.precision,
                        'recall': metrics.recall,
                        'f1_score': metrics.f1_score,
                        'roc_auc': metrics.roc_auc,
                        'log_loss': metrics.log_loss,
                    })
            
            comparison['training_metrics_comparison'] = trading_comparison
        
        # Generate rankings
        comparison['rankings'] = self._generate_rankings(trading_comparison)
        
        self._logger.info(
            f"Compared {len(experiments)} experiments"
        )
        
        return comparison
    
    def _generate_rankings(self, trading_comparison: List[Dict]) -> Dict:
        """Generate rankings by different metrics."""
        if not trading_comparison:
            return {}
        
        rankings = {}
        
        # Rank by Sharpe ratio
        sharpe_ranked = sorted(
            trading_comparison,
            key=lambda x: x['sharpe_ratio'],
            reverse=True,
        )
        rankings['by_sharpe'] = [
            {'rank': i+1, 'experiment_id': x['experiment_id'], 'name': x['name'], 'value': x['sharpe_ratio']}
            for i, x in enumerate(sharpe_ranked)
        ]
        
        # Rank by win rate
        win_rate_ranked = sorted(
            trading_comparison,
            key=lambda x: x['win_rate'],
            reverse=True,
        )
        rankings['by_win_rate'] = [
            {'rank': i+1, 'experiment_id': x['experiment_id'], 'name': x['name'], 'value': x['win_rate']}
            for i, x in enumerate(win_rate_ranked)
        ]
        
        # Rank by profit factor
        profit_factor_ranked = sorted(
            trading_comparison,
            key=lambda x: x['profit_factor'] if x['profit_factor'] != float('inf') else 0,
            reverse=True,
        )
        rankings['by_profit_factor'] = [
            {'rank': i+1, 'experiment_id': x['experiment_id'], 'name': x['name'], 'value': x['profit_factor']}
            for i, x in enumerate(profit_factor_ranked)
        ]
        
        # Rank by drawdown (lower is better)
        drawdown_ranked = sorted(
            trading_comparison,
            key=lambda x: x['max_drawdown'],
        )
        rankings['by_drawdown'] = [
            {'rank': i+1, 'experiment_id': x['experiment_id'], 'name': x['name'], 'value': x['max_drawdown']}
            for i, x in enumerate(drawdown_ranked)
        ]
        
        return rankings
    
    def compare_two_experiments(
        self,
        experiment_id_1: str,
        experiment_id_2: str,
        trading_metrics_map: Dict[str, TradingMetrics],
        training_metrics_map: Optional[Dict[str, TrainingMetrics]] = None,
    ) -> Dict:
        """
        Compare two experiments in detail.
        
        Args:
            experiment_id_1: First experiment ID
            experiment_id_2: Second experiment ID
            trading_metrics_map: Dictionary mapping experiment IDs to TradingMetrics
            training_metrics_map: Optional dictionary mapping experiment IDs to TrainingMetrics
            
        Returns:
            Dictionary with detailed comparison
        """
        metrics_1 = trading_metrics_map.get(experiment_id_1)
        metrics_2 = trading_metrics_map.get(experiment_id_2)
        
        if not metrics_1 or not metrics_2:
            return {
                'error': 'One or both experiments missing trading metrics'
            }
        
        comparison = {
            'experiment_1': experiment_id_1,
            'experiment_2': experiment_id_2,
            'trading_metrics_differences': {
                'win_rate_change': metrics_2.win_rate - metrics_1.win_rate,
                'sharpe_change': metrics_2.sharpe_ratio - metrics_1.sharpe_ratio,
                'sortino_change': metrics_2.sortino_ratio - metrics_1.sortino_ratio,
                'drawdown_change': metrics_2.max_drawdown - metrics_1.max_drawdown,
                'return_change': metrics_2.average_return - metrics_1.average_return,
                'profit_factor_change': metrics_2.profit_factor - metrics_1.profit_factor,
                'trades_change': metrics_2.total_trades - metrics_1.total_trades,
            },
            'improvement': self._calculate_improvement(metrics_1, metrics_2),
        }
        
        # Add training metrics comparison if available
        if training_metrics_map:
            train_metrics_1 = training_metrics_map.get(experiment_id_1)
            train_metrics_2 = training_metrics_map.get(experiment_id_2)
            
            if train_metrics_1 and train_metrics_2:
                comparison['training_metrics_differences'] = {
                    'accuracy_change': train_metrics_2.accuracy - train_metrics_1.accuracy,
                    'precision_change': train_metrics_2.precision - train_metrics_1.precision,
                    'recall_change': train_metrics_2.recall - train_metrics_1.recall,
                    'f1_change': train_metrics_2.f1_score - train_metrics_1.f1_score,
                    'roc_auc_change': train_metrics_2.roc_auc - train_metrics_1.roc_auc,
                    'log_loss_change': train_metrics_2.log_loss - train_metrics_1.log_loss,
                }
        
        return comparison
    
    def _calculate_improvement(
        self,
        metrics_1: TradingMetrics,
        metrics_2: TradingMetrics,
    ) -> Dict:
        """Calculate improvement metrics."""
        improvement = {}
        
        # Sharpe improvement
        if metrics_1.sharpe_ratio > 0:
            sharpe_improvement = (metrics_2.sharpe_ratio - metrics_1.sharpe_ratio) / metrics_1.sharpe_ratio
            improvement['sharpe_improvement_pct'] = sharpe_improvement * 100
        else:
            improvement['sharpe_improvement_pct'] = 0
        
        # Win rate improvement
        if metrics_1.win_rate > 0:
            win_rate_improvement = (metrics_2.win_rate - metrics_1.win_rate) / metrics_1.win_rate
            improvement['win_rate_improvement_pct'] = win_rate_improvement * 100
        else:
            improvement['win_rate_improvement_pct'] = 0
        
        # Drawdown improvement (lower is better)
        if metrics_1.max_drawdown > 0:
            drawdown_improvement = (metrics_1.max_drawdown - metrics_2.max_drawdown) / metrics_1.max_drawdown
            improvement['drawdown_improvement_pct'] = drawdown_improvement * 100
        else:
            improvement['drawdown_improvement_pct'] = 0
        
        # Overall improvement score (simple average)
        overall = (
            improvement['sharpe_improvement_pct'] +
            improvement['win_rate_improvement_pct'] +
            improvement['drawdown_improvement_pct']
        ) / 3
        improvement['overall_improvement_pct'] = overall
        
        return improvement
    
    def get_best_experiment(
        self,
        experiments: List[Experiment],
        trading_metrics_map: Dict[str, TradingMetrics],
        metric: str = "sharpe_ratio",
    ) -> Optional[Experiment]:
        """
        Get best experiment by a specific metric.
        
        Args:
            experiments: List of Experiment objects
            trading_metrics_map: Dictionary mapping experiment IDs to TradingMetrics
            metric: Metric to compare (sharpe_ratio, win_rate, profit_factor, etc.)
            
        Returns:
            Best performing Experiment
        """
        best_exp = None
        best_value = -float('inf')
        
        for exp in experiments:
            metrics = trading_metrics_map.get(exp.experiment_id)
            if metrics:
                value = getattr(metrics, metric, 0)
                if value > best_value:
                    best_value = value
                    best_exp = exp
        
        return best_exp
    
    def generate_comparison_table(
        self,
        experiments: List[Experiment],
        trading_metrics_map: Dict[str, TradingMetrics],
    ) -> List[Dict]:
        """
        Generate a comparison table for display.
        
        Args:
            experiments: List of Experiment objects
            trading_metrics_map: Dictionary mapping experiment IDs to TradingMetrics
            
        Returns:
            List of dictionaries with comparison data
        """
        table = []
        
        for exp in experiments:
            metrics = trading_metrics_map.get(exp.experiment_id)
            if metrics:
                row = {
                    'Experiment': exp.name,
                    'ID': exp.experiment_id,
                    'Win Rate': f"{metrics.win_rate:.1%}",
                    'Sharpe': f"{metrics.sharpe_ratio:.2f}",
                    'Sortino': f"{metrics.sortino_ratio:.2f}",
                    'Drawdown': f"{metrics.max_drawdown:.1f}%",
                    'Avg Return': f"{metrics.average_return:.2f}%",
                    'Profit Factor': f"{metrics.profit_factor:.2f}",
                    'Trades': metrics.total_trades,
                }
                table.append(row)
        
        return table
