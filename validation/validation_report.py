"""
validation/validation_report.py
================================
Generates a structured audit report from all resolved predictions.

Call build_report() to get a dict you can log, print, or serve via API.

Sections
--------
  overview        - totals, win/loss/timeout, overall win rate, avg return
  by_timeframe    - per-INTRADAY/SWING/LONGTERM breakdown
  calibration     - mean predicted probability vs actual win rate (bucketed)
  mfe_mae         - p25/p50/p75/p95 distributions by outcome
  duration        - mean/median hold_bars by outcome
  barriers        - which hit first: SL vs target counts
  feature_audit   - % of predictions with feature_snapshot populated

Usage
-----
    from validation.validation_report import build_report
    from database.db_sync import SessionLocal

    db = SessionLocal()
    report = build_report(db)
    db.close()

    import json
    print(json.dumps(report, indent=2, default=str))
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("validation_report")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pct(num: int, denom: int) -> float:
    return round(num / denom * 100, 2) if denom else 0.0


def _quantiles(values: List[float], label: str) -> Dict[str, float]:
    if not values:
        return {f"{label}_p25": None, f"{label}_p50": None,
                f"{label}_p75": None, f"{label}_p95": None,
                f"{label}_mean": None}
    arr = np.array(values, dtype=float)
    return {
        f"{label}_p25":  round(float(np.percentile(arr, 25)), 6),
        f"{label}_p50":  round(float(np.percentile(arr, 50)), 6),
        f"{label}_p75":  round(float(np.percentile(arr, 75)), 6),
        f"{label}_p95":  round(float(np.percentile(arr, 95)), 6),
        f"{label}_mean": round(float(arr.mean()), 6),
    }


def _calibration_buckets(
    rows: List[Any],
    n_buckets: int = 10,
) -> List[Dict[str, Any]]:
    """
    Reliability diagram data.
    Bins predictions by their predicted probability and checks actual win rate.
    A perfectly calibrated model has actual_win_rate ≈ mean_predicted_prob in each bucket.
    """
    if not rows:
        return []

    bucket_size = 1.0 / n_buckets
    buckets = []

    for b in range(n_buckets):
        lo = b * bucket_size
        hi = lo + bucket_size
        in_bucket = [
            r for r in rows
            if r.confidence is not None and lo <= float(r.confidence) < hi
        ]
        if not in_bucket:
            continue
        wins = sum(1 for r in in_bucket if r.actual_outcome == "WIN")
        mean_pred = float(np.mean([float(r.confidence) for r in in_bucket]))
        actual_rate = wins / len(in_bucket)
        buckets.append({
            "bucket": f"{lo:.1f}-{hi:.1f}",
            "count": len(in_bucket),
            "mean_predicted_prob": round(mean_pred, 4),
            "actual_win_rate": round(actual_rate, 4),
            "calibration_error": round(abs(mean_pred - actual_rate), 4),
        })

    return buckets


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------

def build_report(db: Any, timeframe: Optional[str] = None) -> Dict[str, Any]:
    """
    Build the full validation report.

    Args:
        db:        SQLAlchemy Session (open, caller closes)
        timeframe: Optional filter ("INTRADAY" / "SWING" / "LONGTERM")

    Returns:
        Nested dict with all report sections.
    """
    from database.models import Prediction

    # ── fetch all resolved ────────────────────────────────────────────────────
    q = db.query(Prediction).filter(
        Prediction.actual_outcome.in_(["WIN", "LOSS", "TIMEOUT"])
    )
    if timeframe:
        q = q.filter(Prediction.horizon == timeframe.upper())
    resolved: List[Prediction] = q.all()

    # ── fetch all (including OPEN) for feature audit ──────────────────────────
    all_preds: List[Prediction] = db.query(Prediction).all()

    if not resolved:
        logger.warning("build_report: no resolved predictions found.")
        return {"error": "no_resolved_predictions", "total_all": len(all_preds)}

    wins     = [r for r in resolved if r.actual_outcome == "WIN"]
    losses   = [r for r in resolved if r.actual_outcome == "LOSS"]
    timeouts = [r for r in resolved if r.actual_outcome == "TIMEOUT"]

    # ── 1. Overview ───────────────────────────────────────────────────────────
    all_returns = [float(r.actual_return) for r in resolved if r.actual_return is not None]
    win_returns  = [float(r.actual_return) for r in wins if r.actual_return is not None]
    loss_returns = [float(r.actual_return) for r in losses if r.actual_return is not None]

    overview = {
        "total_resolved": len(resolved),
        "wins":           len(wins),
        "losses":         len(losses),
        "timeouts":       len(timeouts),
        "win_rate_pct":   _pct(len(wins), len(resolved)),
        "avg_return_pct": round(float(np.mean(all_returns)) * 100, 3) if all_returns else None,
        "avg_win_pct":    round(float(np.mean(win_returns))  * 100, 3) if win_returns else None,
        "avg_loss_pct":   round(float(np.mean(loss_returns)) * 100, 3) if loss_returns else None,
        "profit_factor":  (
            round(sum(win_returns) / abs(sum(loss_returns)), 3)
            if loss_returns and sum(loss_returns) != 0 else None
        ),
    }

    # Sharpe estimate (daily returns, annualised for reference)
    if len(all_returns) >= 5:
        ret_arr = np.array(all_returns)
        sharpe_raw = ret_arr.mean() / ret_arr.std() if ret_arr.std() > 0 else 0.0
        overview["sharpe_estimate"] = round(float(sharpe_raw * np.sqrt(252)), 3)

    # ── 2. By timeframe ───────────────────────────────────────────────────────
    by_tf: Dict[str, Any] = {}
    for tf in ("INTRADAY", "SWING", "LONGTERM"):
        subset = [r for r in resolved if (r.horizon or "").upper() == tf]
        if not subset:
            continue
        tf_wins = sum(1 for r in subset if r.actual_outcome == "WIN")
        tf_rets  = [float(r.actual_return) for r in subset if r.actual_return is not None]
        by_tf[tf] = {
            "total":          len(subset),
            "wins":           tf_wins,
            "win_rate_pct":   _pct(tf_wins, len(subset)),
            "avg_return_pct": round(float(np.mean(tf_rets)) * 100, 3) if tf_rets else None,
        }

    # ── 3. Calibration ────────────────────────────────────────────────────────
    calibration = {
        "buckets": _calibration_buckets(resolved),
        "mean_predicted_prob": round(
            float(np.mean([float(r.confidence) for r in resolved if r.confidence is not None])), 4
        ) if resolved else None,
        "actual_win_rate": round(len(wins) / len(resolved), 4),
        "expected_calibration_error": None,
    }
    if calibration["buckets"]:
        bucket_errors = [b["calibration_error"] * b["count"] for b in calibration["buckets"]]
        total_in_buckets = sum(b["count"] for b in calibration["buckets"])
        calibration["expected_calibration_error"] = round(
            sum(bucket_errors) / total_in_buckets, 4
        ) if total_in_buckets else None

    # ── 4. MFE / MAE distributions ───────────────────────────────────────────
    mfe_mae: Dict[str, Any] = {}

    win_mfe  = [float(r.mfe) for r in wins if r.mfe is not None]
    win_mae  = [float(r.mae) for r in wins if r.mae is not None]
    loss_mfe = [float(r.mfe) for r in losses if r.mfe is not None]
    loss_mae = [float(r.mae) for r in losses if r.mae is not None]
    all_mfe  = [float(r.mfe) for r in resolved if r.mfe is not None]
    all_mae  = [float(r.mae) for r in resolved if r.mae is not None]

    mfe_mae["winners"]  = {**_quantiles(win_mfe, "mfe"),  **_quantiles(win_mae, "mae")}
    mfe_mae["losers"]   = {**_quantiles(loss_mfe, "mfe"), **_quantiles(loss_mae, "mae")}
    mfe_mae["all"]      = {**_quantiles(all_mfe, "mfe"),  **_quantiles(all_mae, "mae")}

    # Edge quality: do winners ride their MFE, or are they stopped early?
    # A good system has: winner MFE p50 >> winner actual_return p50
    mfe_mae["mfe_capture_ratio"] = None
    winner_rets = [float(r.actual_return) for r in wins if r.actual_return is not None]
    if win_mfe and winner_rets:
        mfe_capture = [
            r / m for r, m in zip(sorted(winner_rets), sorted(win_mfe)) if m > 0
        ]
        mfe_mae["mfe_capture_ratio"] = round(float(np.mean(mfe_capture)), 4)

    # ── 5. Duration analysis ─────────────────────────────────────────────────
    duration: Dict[str, Any] = {}
    for label, subset in [("wins", wins), ("losses", losses), ("all", resolved)]:
        bars = [r.hold_bars for r in subset if r.hold_bars is not None]
        if bars:
            arr = np.array(bars, dtype=float)
            duration[label] = {
                "mean_bars":   round(float(arr.mean()), 1),
                "median_bars": round(float(np.median(arr)), 1),
                "min_bars":    int(arr.min()),
                "max_bars":    int(arr.max()),
            }
        else:
            duration[label] = {"mean_bars": None, "note": "hold_bars not yet populated"}

    # ── 6. Barriers — which hit first ─────────────────────────────────────────
    target_hits = sum(1 for r in resolved if r.target_hit)
    sl_hits     = sum(1 for r in resolved if r.stop_hit)
    timeout_n   = len(timeouts)

    barriers = {
        "target_hit":        target_hits,
        "target_hit_pct":    _pct(target_hits, len(resolved)),
        "sl_hit":            sl_hits,
        "sl_hit_pct":        _pct(sl_hits, len(resolved)),
        "timeout":           timeout_n,
        "timeout_pct":       _pct(timeout_n, len(resolved)),
        "win_at_target_pct": _pct(target_hits, target_hits + sl_hits) if (target_hits + sl_hits) else None,
    }

    # ── 7. Feature snapshot audit ─────────────────────────────────────────────
    total_all = len(all_preds)
    with_snapshot  = sum(1 for p in all_preds if p.feature_snapshot and len(p.feature_snapshot) > 2)
    open_preds     = sum(1 for p in all_preds if p.actual_outcome == "OPEN")

    feature_audit = {
        "total_predictions":          total_all,
        "open":                       open_preds,
        "resolved":                   len(resolved),
        "with_feature_snapshot":      with_snapshot,
        "snapshot_coverage_pct":      _pct(with_snapshot, total_all),
        "without_snapshot":           total_all - with_snapshot,
        "note": (
            "Predictions without a feature_snapshot were written before the "
            "validation framework was deployed. All new predictions should have snapshots."
            if (total_all - with_snapshot) > 0 else
            "All predictions have feature snapshots."
        ),
    }

    # ── 8. Direction breakdown ────────────────────────────────────────────────
    buy_preds  = [r for r in resolved if (r.prediction or "").upper() == "BUY"]
    sell_preds = [r for r in resolved if (r.prediction or "").upper() == "SELL"]
    direction_stats = {}
    for label, subset in [("BUY", buy_preds), ("SELL", sell_preds)]:
        if subset:
            w = sum(1 for r in subset if r.actual_outcome == "WIN")
            direction_stats[label] = {
                "total":        len(subset),
                "wins":         w,
                "win_rate_pct": _pct(w, len(subset)),
            }

    # ── assemble ──────────────────────────────────────────────────────────────
    return {
        "generated_at": _now_ist_str(),
        "filter_timeframe": timeframe,
        "overview": overview,
        "by_timeframe": by_tf,
        "by_direction": direction_stats,
        "calibration": calibration,
        "mfe_mae": mfe_mae,
        "duration": duration,
        "barriers": barriers,
        "feature_audit": feature_audit,
    }


def _now_ist_str() -> str:
    try:
        from utils.time_utils import now_ist
        return now_ist().isoformat()
    except Exception:
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"


def print_report(db: Any, timeframe: Optional[str] = None) -> None:
    """Convenience: build and pretty-print the report to stdout."""
    report = build_report(db, timeframe=timeframe)
    print(json.dumps(report, indent=2, default=str))
