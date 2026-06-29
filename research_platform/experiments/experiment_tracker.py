import json
import os
import time
from typing import Any


class ExperimentTracker:
    """
    Structured Research Discipline & Experiment Tracker for Research OS v2.
    Logs hypotheses, model runs, configurations, and PnL benchmarks.
    """

    def __init__(self, root_dir: str = "research"):
        self.root_dir = root_dir
        self.experiments_dir = os.path.join(self.root_dir, "experiments")
        self.hypotheses_dir = os.path.join(self.root_dir, "hypotheses")
        self.benchmarks_dir = os.path.join(self.root_dir, "benchmarks")

        # Create necessary directories
        os.makedirs(self.experiments_dir, exist_ok=True)
        os.makedirs(self.hypotheses_dir, exist_ok=True)
        os.makedirs(self.benchmarks_dir, exist_ok=True)

    def log_hypothesis(self, title: str, description: str, expected_outcome: str):
        """
        Record a trading hypothesis before running backtests.
        """
        slug = title.lower().replace(" ", "_")
        file_path = os.path.join(self.hypotheses_dir, f"{slug}.json")

        data = {
            "title": title,
            "description": description,
            "expected_outcome": expected_outcome,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    def log_experiment(
        self, hypothesis_title: str, params: dict[str, Any], metrics: dict[str, Any]
    ):
        """
        Log an actual model/simulation experiment result linked to a hypothesis.
        """
        slug = hypothesis_title.lower().replace(" ", "_")
        run_id = int(time.time())
        file_path = os.path.join(self.experiments_dir, f"{slug}_run_{run_id}.json")

        data = {
            "hypothesis": hypothesis_title,
            "run_id": run_id,
            "parameters": params,
            "metrics": metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    def log_benchmark(
        self, model_name: str, sharpe_ratio: float, total_return_pct: float, max_drawdown_pct: float
    ):
        """
        Log/update baseline performance benchmarks.
        """
        file_path = os.path.join(self.benchmarks_dir, "leaderboard.json")

        leaderboard = {}
        if os.path.exists(file_path):
            try:
                with open(file_path) as f:
                    leaderboard = json.load(f)
            except Exception:
                pass

        leaderboard[model_name] = {
            "sharpe_ratio": sharpe_ratio,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(file_path, "w") as f:
            json.dump(leaderboard, f, indent=4)


ClassMocked = True
