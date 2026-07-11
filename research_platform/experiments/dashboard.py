"""
Research Dashboard

Provides overview of research projects and experiments.
Shows key metrics and statistics.
"""

from typing import Dict, List, Optional

from research_platform.experiments.base import (
    Experiment,
    ExperimentStatus,
    ExperimentDecision,
    TradingMetrics,
)
from utils.logger import get_logger

logger = get_logger("experiments.dashboard")


class ResearchDashboard:
    """
    Research Dashboard.
    
    Provides:
    - Project overview
    - Experiment statistics
    - Best performing models/features/signals
    - Decision summary
    - Recent activity
    """
    
    def __init__(self):
        """Initialize research dashboard."""
        self._logger = get_logger("experiments.dashboard")
    
    def get_dashboard_summary(
        self,
        projects: List,
        experiments: List[Experiment],
        trading_metrics_map: Dict[str, TradingMetrics],
        decision_history: Dict,
    ) -> Dict:
        """
        Get overall dashboard summary.
        
        Args:
            projects: List of ResearchProject objects
            experiments: List of Experiment objects
            trading_metrics_map: Dictionary mapping experiment IDs to TradingMetrics
            decision_history: Dictionary of decision history
            
        Returns:
            Dictionary with dashboard summary
        """
        summary = {
            'projects': self._get_project_summary(projects),
            'experiments': self._get_experiment_summary(experiments),
            'decisions': self._get_decision_summary(decision_history),
            'best_performers': self._get_best_performers(experiments, trading_metrics_map),
            'recent_activity': self._get_recent_activity(experiments),
        }
        
        return summary
    
    def _get_project_summary(self, projects: List) -> Dict:
        """Get project summary."""
        if not projects:
            return {
                'total': 0,
                'by_type': {},
                'by_status': {},
            }
        
        by_type = {}
        by_status = {}
        
        for project in projects:
            ptype = project.project_type.value
            status = project.status.value
            
            by_type[ptype] = by_type.get(ptype, 0) + 1
            by_status[status] = by_status.get(status, 0) + 1
        
        return {
            'total': len(projects),
            'by_type': by_type,
            'by_status': by_status,
        }
    
    def _get_experiment_summary(self, experiments: List[Experiment]) -> Dict:
        """Get experiment summary."""
        if not experiments:
            return {
                'total': 0,
                'by_type': {},
                'by_status': {},
                'running': 0,
                'completed': 0,
                'failed': 0,
            }
        
        by_type = {}
        by_status = {}
        
        for exp in experiments:
            etype = exp.experiment_type.value
            status = exp.status.value
            
            by_type[etype] = by_type.get(etype, 0) + 1
            by_status[status] = by_status.get(status, 0) + 1
        
        return {
            'total': len(experiments),
            'by_type': by_type,
            'by_status': by_status,
            'running': by_status.get('RUNNING', 0),
            'completed': by_status.get('COMPLETED', 0),
            'failed': by_status.get('FAILED', 0),
        }
    
    def _get_decision_summary(self, decision_history: Dict) -> Dict:
        """Get decision summary."""
        if not decision_history:
            return {
                'total': 0,
                'accepted': 0,
                'rejected': 0,
                'needs_testing': 0,
                'merged': 0,
                'archived': 0,
            }
        
        by_type = {
            'ACCEPTED': 0,
            'REJECTED': 0,
            'NEEDS_MORE_TESTING': 0,
            'MERGED': 0,
            'ARCHIVED': 0,
        }
        
        for decision_data in decision_history.values():
            decision = decision_data.get('decision')
            if decision:
                by_type[decision.value] = by_type.get(decision.value, 0) + 1
        
        return {
            'total': len(decision_history),
            'accepted': by_type['ACCEPTED'],
            'rejected': by_type['REJECTED'],
            'needs_testing': by_type['NEEDS_MORE_TESTING'],
            'merged': by_type['MERGED'],
            'archived': by_type['ARCHIVED'],
        }
    
    def _get_best_performers(
        self,
        experiments: List[Experiment],
        trading_metrics_map: Dict[str, TradingMetrics],
    ) -> Dict:
        """Get best performing experiments."""
        if not experiments or not trading_metrics_map:
            return {
                'best_sharpe': None,
                'best_win_rate': None,
                'best_profit_factor': None,
                'lowest_drawdown': None,
            }
        
        best_sharpe = None
        best_win_rate = None
        best_profit_factor = None
        lowest_drawdown = None
        
        for exp in experiments:
            metrics = trading_metrics_map.get(exp.experiment_id)
            if not metrics:
                continue
            
            if best_sharpe is None or metrics.sharpe_ratio > best_sharpe['value']:
                best_sharpe = {
                    'experiment_id': exp.experiment_id,
                    'name': exp.name,
                    'value': metrics.sharpe_ratio,
                }
            
            if best_win_rate is None or metrics.win_rate > best_win_rate['value']:
                best_win_rate = {
                    'experiment_id': exp.experiment_id,
                    'name': exp.name,
                    'value': metrics.win_rate,
                }
            
            if best_profit_factor is None or metrics.profit_factor > best_profit_factor['value']:
                if metrics.profit_factor != float('inf'):
                    best_profit_factor = {
                        'experiment_id': exp.experiment_id,
                        'name': exp.name,
                        'value': metrics.profit_factor,
                    }
            
            if lowest_drawdown is None or metrics.max_drawdown < lowest_drawdown['value']:
                lowest_drawdown = {
                    'experiment_id': exp.experiment_id,
                    'name': exp.name,
                    'value': metrics.max_drawdown,
                }
        
        return {
            'best_sharpe': best_sharpe,
            'best_win_rate': best_win_rate,
            'best_profit_factor': best_profit_factor,
            'lowest_drawdown': lowest_drawdown,
        }
    
    def _get_recent_activity(self, experiments: List[Experiment]) -> List[Dict]:
        """Get recent experiment activity."""
        if not experiments:
            return []
        
        # Sort by creation time descending
        sorted_experiments = sorted(
            experiments,
            key=lambda x: x.created_at,
            reverse=True,
        )
        
        recent = []
        for exp in sorted_experiments[:10]:
            recent.append({
                'experiment_id': exp.experiment_id,
                'name': exp.name,
                'type': exp.experiment_type.value,
                'status': exp.status.value,
                'created_at': exp.created_at.isoformat(),
            })
        
        return recent
    
    def get_project_dashboard(
        self,
        project_id: str,
        experiments: List[Experiment],
        trading_metrics_map: Dict[str, TradingMetrics],
    ) -> Dict:
        """
        Get dashboard for a specific project.
        
        Args:
            project_id: Project ID
            experiments: List of Experiment objects
            trading_metrics_map: Dictionary mapping experiment IDs to TradingMetrics
            
        Returns:
            Dictionary with project dashboard
        """
        # Filter experiments for this project
        project_experiments = [
            exp for exp in experiments
            if exp.project_id == project_id
        ]
        
        if not project_experiments:
            return {
                'project_id': project_id,
                'total_experiments': 0,
                'experiments': [],
            }
        
        dashboard = {
            'project_id': project_id,
            'total_experiments': len(project_experiments),
            'experiments': [],
            'summary': self._get_experiment_summary(project_experiments),
        }
        
        for exp in project_experiments:
            metrics = trading_metrics_map.get(exp.experiment_id)
            exp_data = {
                'experiment_id': exp.experiment_id,
                'name': exp.name,
                'type': exp.experiment_type.value,
                'status': exp.status.value,
                'created_at': exp.created_at.isoformat(),
            }
            
            if metrics:
                exp_data['metrics'] = {
                    'win_rate': metrics.win_rate,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'total_trades': metrics.total_trades,
                }
            
            dashboard['experiments'].append(exp_data)
        
        return dashboard
