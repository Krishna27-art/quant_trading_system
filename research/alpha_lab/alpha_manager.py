"""
Alpha Manager

Manages research experiments and factor lifecycle.
Tracks every research idea, parameters, results, and decisions.
"""

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.logger import get_logger

logger = get_logger("research.alpha_manager")


@dataclass
class Experiment:
    """A research experiment."""
    experiment_id: str
    idea: str
    factor_name: str
    parameters: Dict[str, Any]
    dataset: str
    results: Dict[str, Any]
    charts: List[str]
    notes: str
    decision: str  # PROMOTE, REJECT, RESEARCH
    created_at: datetime
    updated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "idea": self.idea,
            "factor_name": self.factor_name,
            "parameters": self.parameters,
            "dataset": self.dataset,
            "results": self.results,
            "charts": self.charts,
            "notes": self.notes,
            "decision": self.decision,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class AlphaManager:
    """
    Manages research experiments and factor lifecycle.
    
    Every research idea is tracked as an experiment with:
    - Unique ID
    - Idea description
    - Factor name
    - Parameters
    - Dataset used
    - Results (IC, Sharpe, etc.)
    - Charts (saved paths)
    - Notes
    - Decision (PROMOTE, REJECT, RESEARCH)
    
    Never delete experiments - they form the research history.
    """
    
    def __init__(self, experiments_path: str = "research/experiments"):
        """
        Initialize alpha manager.
        
        Args:
            experiments_path: Path to experiments directory
        """
        self.experiments_path = Path(experiments_path)
        self.experiments_path.mkdir(parents=True, exist_ok=True)
        self._experiments: Dict[str, Experiment] = {}
        self._logger = get_logger("research.alpha_manager")
        
        # Load existing experiments
        self._load_experiments()
    
    def _load_experiments(self) -> None:
        """Load existing experiments from disk."""
        for exp_file in self.experiments_path.glob("*.json"):
            try:
                with open(exp_file, "r") as f:
                    data = json.load(f)
                    experiment = Experiment(
                        experiment_id=data["experiment_id"],
                        idea=data["idea"],
                        factor_name=data["factor_name"],
                        parameters=data["parameters"],
                        dataset=data["dataset"],
                        results=data["results"],
                        charts=data["charts"],
                        notes=data["notes"],
                        decision=data["decision"],
                        created_at=datetime.fromisoformat(data["created_at"]),
                        updated_at=datetime.fromisoformat(data["updated_at"]),
                    )
                    self._experiments[experiment.experiment_id] = experiment
            except Exception as e:
                self._logger.warning(f"Failed to load experiment {exp_file}: {e}")
        
        self._logger.info(f"Loaded {len(self._experiments)} experiments")
    
    def create_experiment(
        self,
        idea: str,
        factor_name: str,
        parameters: Dict[str, Any],
        dataset: str,
        notes: str = "",
    ) -> Experiment:
        """
        Create a new experiment.
        
        Args:
            idea: Research idea description
            factor_name: Name of factor being tested
            parameters: Factor parameters
            dataset: Dataset identifier
            notes: Additional notes
            
        Returns:
            Created Experiment
        """
        experiment_id = str(uuid.uuid4())
        now = datetime.now()
        
        experiment = Experiment(
            experiment_id=experiment_id,
            idea=idea,
            factor_name=factor_name,
            parameters=parameters,
            dataset=dataset,
            results={},
            charts=[],
            notes=notes,
            decision="RESEARCH",  # Default decision
            created_at=now,
            updated_at=now,
        )
        
        self._experiments[experiment_id] = experiment
        self._save_experiment(experiment)
        
        self._logger.info(f"Created experiment {experiment_id}: {idea}")
        return experiment
    
    def update_results(
        self,
        experiment_id: str,
        results: Dict[str, Any],
        charts: Optional[List[str]] = None,
    ) -> Optional[Experiment]:
        """
        Update experiment results.
        
        Args:
            experiment_id: Experiment ID
            results: Results dictionary (IC, Sharpe, etc.)
            charts: List of chart file paths
            
        Returns:
            Updated Experiment or None if not found
        """
        if experiment_id not in self._experiments:
            self._logger.error(f"Experiment {experiment_id} not found")
            return None
        
        experiment = self._experiments[experiment_id]
        experiment.results = results
        experiment.updated_at = datetime.now()
        
        if charts is not None:
            experiment.charts = charts
        
        self._save_experiment(experiment)
        self._logger.info(f"Updated results for experiment {experiment_id}")
        return experiment
    
    def update_decision(
        self,
        experiment_id: str,
        decision: str,
        notes: Optional[str] = None,
    ) -> Optional[Experiment]:
        """
        Update experiment decision.
        
        Args:
            experiment_id: Experiment ID
            decision: Decision (PROMOTE, REJECT, RESEARCH)
            notes: Optional additional notes
            
        Returns:
            Updated Experiment or None if not found
        """
        if experiment_id not in self._experiments:
            self._logger.error(f"Experiment {experiment_id} not found")
            return None
        
        if decision not in ["PROMOTE", "REJECT", "RESEARCH"]:
            self._logger.error(f"Invalid decision: {decision}")
            return None
        
        experiment = self._experiments[experiment_id]
        experiment.decision = decision
        experiment.updated_at = datetime.now()
        
        if notes is not None:
            experiment.notes = notes
        
        self._save_experiment(experiment)
        self._logger.info(f"Updated decision for experiment {experiment_id}: {decision}")
        return experiment
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """
        Get experiment by ID.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment or None if not found
        """
        return self._experiments.get(experiment_id)
    
    def list_experiments(
        self,
        factor_name: Optional[str] = None,
        decision: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Experiment]:
        """
        List experiments with optional filters.
        
        Args:
            factor_name: Optional filter by factor name
            decision: Optional filter by decision
            limit: Optional limit on number of results
            
        Returns:
            List of Experiments
        """
        experiments = list(self._experiments.values())
        
        if factor_name:
            experiments = [e for e in experiments if e.factor_name == factor_name]
        
        if decision:
            experiments = [e for e in experiments if e.decision == decision]
        
        # Sort by updated_at descending
        experiments.sort(key=lambda e: e.updated_at, reverse=True)
        
        if limit:
            experiments = experiments[:limit]
        
        return experiments
    
    def get_promoted_factors(self) -> List[str]:
        """
        Get list of promoted factors.
        
        Returns:
            List of factor names that have been promoted
        """
        promoted = [
            e.factor_name
            for e in self._experiments.values()
            if e.decision == "PROMOTE"
        ]
        return list(set(promoted))  # Remove duplicates
    
    def get_factor_history(self, factor_name: str) -> List[Experiment]:
        """
        Get all experiments for a specific factor.
        
        Args:
            factor_name: Factor name
            
        Returns:
            List of Experiments for the factor
        """
        return self.list_experiments(factor_name=factor_name)
    
    def _save_experiment(self, experiment: Experiment) -> None:
        """Save experiment to disk."""
        exp_file = self.experiments_path / f"{experiment.experiment_id}.json"
        with open(exp_file, "w") as f:
            json.dump(experiment.to_dict(), f, indent=2)
    
    def export_summary(self) -> pd.DataFrame:
        """
        Export experiment summary as DataFrame.
        
        Returns:
            DataFrame with experiment summary
        """
        data = []
        for experiment in self._experiments.values():
            data.append({
                "experiment_id": experiment.experiment_id,
                "idea": experiment.idea,
                "factor_name": experiment.factor_name,
                "decision": experiment.decision,
                "created_at": experiment.created_at,
                "updated_at": experiment.updated_at,
                "mean_ic": experiment.results.get("mean_ic", None),
                "mean_rank_ic": experiment.results.get("mean_rank_ic", None),
                "sharpe": experiment.results.get("sharpe", None),
            })
        
        return pd.DataFrame(data).sort_values("updated_at", ascending=False)
