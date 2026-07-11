"""
Decision Engine

Makes decisions on experiment outcomes.
Every experiment must end with a decision.
"""

from datetime import datetime
from typing import Dict, List, Optional

from research_platform.experiments.base import (
    Experiment,
    ExperimentDecision,
    ExperimentStatus,
    TradingMetrics,
)
from utils.logger import get_logger

logger = get_logger("experiments.decision_engine")


class DecisionEngine:
    """
    Decision Engine.
    
    Makes decisions on experiments:
    - Accepted: Deploy to production
    - Rejected: Not suitable for production
    - Needs More Testing: Promising but requires more validation
    - Merged: Merged with existing system
    - Archived: Saved for reference but not active
    """
    
    def __init__(self):
        """Initialize decision engine."""
        self.decision_history: Dict[str, Dict] = {}
        self._logger = get_logger("experiments.decision_engine")
    
    def make_decision(
        self,
        experiment: Experiment,
        trading_metrics: TradingMetrics,
        min_sharpe: float = 1.5,
        min_win_rate: float = 0.55,
        max_drawdown: float = 15.0,
        min_trades: int = 50,
    ) -> ExperimentDecision:
        """
        Make an automated decision based on metrics.
        
        Args:
            experiment: Experiment object
            trading_metrics: Trading metrics
            min_sharpe: Minimum acceptable Sharpe ratio
            min_win_rate: Minimum acceptable win rate
            max_drawdown: Maximum acceptable drawdown
            min_trades: Minimum number of trades
            
        Returns:
            ExperimentDecision
        """
        # Check if experiment is completed
        if experiment.status != ExperimentStatus.COMPLETED:
            self._logger.warning(
                f"Experiment {experiment.experiment_id} not completed, cannot make decision"
            )
            return ExperimentDecision.NEEDS_MORE_TESTING
        
        # Evaluate metrics against thresholds
        sharpe_pass = trading_metrics.sharpe_ratio >= min_sharpe
        win_rate_pass = trading_metrics.win_rate >= min_win_rate
        drawdown_pass = trading_metrics.max_drawdown <= max_drawdown
        trades_pass = trading_metrics.total_trades >= min_trades
        
        # Make decision
        if sharpe_pass and win_rate_pass and drawdown_pass and trades_pass:
            decision = ExperimentDecision.ACCEPTED
            reason = "All metrics meet production thresholds"
        elif sharpe_pass and win_rate_pass and trades_pass:
            decision = ExperimentDecision.NEEDS_MORE_TESTING
            reason = "Good performance but drawdown exceeds threshold"
        elif sharpe_pass >= 1.0 and win_rate_pass:
            decision = ExperimentDecision.NEEDS_MORE_TESTING
            reason = "Promising but needs more validation"
        else:
            decision = ExperimentDecision.REJECTED
            reason = "Metrics do not meet minimum requirements"
        
        # Record decision
        self.decision_history[experiment.experiment_id] = {
            'decision': decision,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'metrics': trading_metrics.to_dict(),
            'thresholds': {
                'min_sharpe': min_sharpe,
                'min_win_rate': min_win_rate,
                'max_drawdown': max_drawdown,
                'min_trades': min_trades,
            },
        }
        
        self._logger.info(
            f"Made decision for experiment {experiment.experiment_id}: "
            f"{decision.value} - {reason}"
        )
        
        return decision
    
    def manual_decision(
        self,
        experiment_id: str,
        decision: ExperimentDecision,
        reason: str,
        decided_by: str,
    ) -> bool:
        """
        Make a manual decision override.
        
        Args:
            experiment_id: Experiment ID
            decision: Manual decision
            reason: Reason for decision
            decided_by: Person making the decision
            
        Returns:
            True if decision recorded successfully
        """
        self.decision_history[experiment_id] = {
            'decision': decision,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'decided_by': decided_by,
            'manual': True,
        }
        
        self._logger.info(
            f"Manual decision for experiment {experiment_id}: "
            f"{decision.value} by {decided_by} - {reason}"
        )
        
        return True
    
    def get_decision(self, experiment_id: str) -> Optional[Dict]:
        """Get decision for an experiment."""
        return self.decision_history.get(experiment_id)
    
    def get_decisions_by_type(self, decision: ExperimentDecision) -> List[Dict]:
        """Get all experiments with a specific decision."""
        return [
            decision_data for decision_data in self.decision_history.values()
            if decision_data['decision'] == decision
        ]
    
    def get_decision_summary(self) -> Dict:
        """Get summary of all decisions."""
        summary = {
            'total_decisions': len(self.decision_history),
            'by_type': {},
        }
        
        for decision in ExperimentDecision:
            count = len(self.get_decisions_by_type(decision))
            summary['by_type'][decision.value] = count
        
        return summary
    
    def get_accepted_experiments(self) -> List[str]:
        """Get list of accepted experiment IDs."""
        accepted = self.get_decisions_by_type(ExperimentDecision.ACCEPTED)
        return [exp_id for exp_id in self.decision_history.keys() 
                if self.decision_history[exp_id]['decision'] == ExperimentDecision.ACCEPTED]
    
    def get_rejected_experiments(self) -> List[str]:
        """Get list of rejected experiment IDs."""
        rejected = self.get_decisions_by_type(ExperimentDecision.REJECTED)
        return [exp_id for exp_id in self.decision_history.keys() 
                if self.decision_history[exp_id]['decision'] == ExperimentDecision.REJECTED]
    
    def get_experiments_needing_testing(self) -> List[str]:
        """Get list of experiments needing more testing."""
        needing_testing = self.get_decisions_by_type(ExperimentDecision.NEEDS_MORE_TESTING)
        return [exp_id for exp_id in self.decision_history.keys() 
                if self.decision_history[exp_id]['decision'] == ExperimentDecision.NEEDS_MORE_TESTING]
