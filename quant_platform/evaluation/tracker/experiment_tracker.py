"""
Institutional Experiment Tracker

Logs and queries research experiments to prevent data-mining bias and track what works.
Records dataset versions, feature IDs, git commit hashes, hyperparameters, and OOS metrics.
"""

import uuid
import json
import logging
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import sys
from pathlib import Path

# Add project root to sys.path if needed
sys.path.append(str(Path(__file__).resolve().parents[2]))
from database.schema import get_db_connection, init_db

logger = logging.getLogger("ExperimentTracker")

class ExperimentTracker:
    def __init__(self):
        init_db()
        self._git_hash = self._fetch_git_commit()

    def _fetch_git_commit(self) -> str:
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL
            )
            return out.decode("utf-8").strip()
        except Exception:
            return "UNKNOWN"

    def log_experiment(
        self,
        dataset_version: str,
        feature_set_id: str,
        model_type: str,
        hyperparameters: Dict[str, Any],
        oos_sharpe: float,
        information_coefficient_ic: float,
        ece_calibration_error: float,
        win_rate: float,
        top_shap_features: List[str],
        notes: str = ""
    ) -> str:
        """
        Logs a completed research experiment and returns the generated experiment_id.
        """
        exp_id = f"exp_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    experiment_id, timestamp, git_commit_hash, dataset_version,
                    feature_set_id, model_type, hyperparameters, oos_sharpe,
                    information_coefficient_ic, ece_calibration_error, win_rate,
                    top_shap_features, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exp_id,
                    timestamp,
                    self._git_hash,
                    dataset_version,
                    feature_set_id,
                    model_type,
                    json.dumps(hyperparameters),
                    float(oos_sharpe),
                    float(information_coefficient_ic),
                    float(ece_calibration_error),
                    float(win_rate),
                    json.dumps(top_shap_features),
                    notes
                )
            )
        logger.info(f"✅ Logged Experiment {exp_id} | Sharpe: {oos_sharpe:.2f} | IC: {information_coefficient_ic:.4f}")
        return exp_id

    def get_best_experiments(self, metric: str = "oos_sharpe", limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves top experiments ranked by the specified metric (descending).
        For ECE calibration error, ascending is preferred.
        """
        order_dir = "ASC" if metric == "ece_calibration_error" else "DESC"
        valid_metrics = {"oos_sharpe", "information_coefficient_ic", "ece_calibration_error", "win_rate"}
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric: {metric}. Must be one of {valid_metrics}")

        query = f"SELECT * FROM experiments WHERE {metric} IS NOT NULL ORDER BY {metric} {order_dir} LIMIT ?"
        
        with get_db_connection() as conn:
            rows = conn.execute(query, (limit,)).fetchall()
            
        results = []
        for r in rows:
            d = dict(r)
            d["hyperparameters"] = json.loads(d["hyperparameters"]) if d["hyperparameters"] else {}
            d["top_shap_features"] = json.loads(d["top_shap_features"]) if d["top_shap_features"] else []
            results.append(d)
        return results

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["hyperparameters"] = json.loads(d["hyperparameters"]) if d["hyperparameters"] else {}
            d["top_shap_features"] = json.loads(d["top_shap_features"]) if d["top_shap_features"] else []
            return d

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tracker = ExperimentTracker()
    exp_id = tracker.log_experiment(
        dataset_version="v2026.07.01_OHLCV",
        feature_set_id="fset_mom_vol_v1",
        model_type="LightGBM_Ensemble",
        hyperparameters={"max_depth": 6, "learning_rate": 0.03, "num_leaves": 31},
        oos_sharpe=2.45,
        information_coefficient_ic=0.048,
        ece_calibration_error=0.018,
        win_rate=0.68,
        top_shap_features=["15m_momentum_zscore", "fii_net_flow_3d", "vix_spread"],
        notes="Baseline LightGBM run with FII institutional flows."
    )
    print("Best experiments by OOS Sharpe:")
    best = tracker.get_best_experiments(limit=5)
    for b in best:
        print(f"[{b['experiment_id']}] Sharpe: {b['oos_sharpe']} | IC: {b['information_coefficient_ic']} | Model: {b['model_type']}")
