"""
prediction_intelligence/lightgbm_ranker.py

LightGBM LambdaMART cross-sectional ranker.

Two classes:
  LightGBMRankerModel   — unchanged core model (train / predict / save / load)
  LightGBMRankerAlpha   — AlphaModel subclass that plugs the ranker into the
                          live orchestrator's _alpha_models list.

CRITICAL FIX: Previously LightGBMRankerModel was only used in
scripts/run_daytrading_backtest.py — never in the live path.
scripts/run_live_loop.py → generate_live_predictions.run() never loaded or
called it, meaning every live signal was generated without the ranker.

To wire it in, add to scripts/run_live_loop.py:

    from prediction_intelligence.lightgbm_ranker import LightGBMRankerAlpha
    ranker_alpha = LightGBMRankerAlpha.from_saved()   # loads saved model
    orchestrator.register_alpha(ranker_alpha)           # plugs into live loop
"""

from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from utils.logger import get_logger

logger = get_logger("lightgbm_ranker")

# Default model storage — overridden by MODEL_PATH env
_DEFAULT_MODEL_DIR = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "production", "models"),
)
_DEFAULT_MODEL_FILE = "lgbm_ranker_v1.pkl"

# Minimum calibrated win probability to emit a TradeSignal
_MIN_WIN_PROB = 0.52

# Features the ranker was trained on (must match training script)
RANKER_FEATURES = [
    "returns_1d",
    "momentum_5d",
    "momentum_10d",
    "volatility_10d",
]


# ---------------------------------------------------------------------------
# Core model — unchanged from original except bug fixes noted inline
# ---------------------------------------------------------------------------

class LightGBMRankerModel:
    """
    Wrapper around lightgbm.LGBMRanker for cross-sectional alpha ranking.
    Designed to ingest multi-horizon labels and sector-neutralized features.
    """

    def __init__(self, model_dir: str = _DEFAULT_MODEL_DIR):
        self.model_dir     = model_dir
        self.model         = None
        self.feature_names: list[str] = []
        self.calibrator    = None  # IsotonicRegression fitted on OOS preds

        os.makedirs(self.model_dir, exist_ok=True)

        if lgb is None:
            logger.warning("lightgbm not installed. Training/inference will fail.")

    # ── internals ───────────────────────────────────────────────────────────

    def _prepare_query_groups(
        self, data: pd.DataFrame, date_column: str = "date"
    ) -> np.ndarray:
        return data.groupby(date_column).size().values

    def _groupby_col(self, index: pd.Index | pd.MultiIndex) -> str:
        """Infer which index level holds the date dimension."""
        if isinstance(index, pd.MultiIndex):
            names = list(index.names)
            # prefer "date" then first non-symbol level
            return "date" if "date" in names else names[0]
        return index.name or "date"

    # ── training ────────────────────────────────────────────────────────────

    def train(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        feature_columns: list[str],
        n_splits: int = 5,
        params: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """
        Walk-forward LambdaRank training with OOS isotonic calibration.

        Args:
            features_df : MultiIndex(date, symbol) feature DataFrame
            labels_df   : Series or single-col DataFrame of raw forward returns
            feature_columns : columns to use from features_df
            n_splits    : number of walk-forward folds
            params      : override default LGB params

        Returns:
            dict with "mean_cv_ic" key
        """
        if lgb is None:
            raise ImportError("lightgbm is required to train the ranker.")

        self.feature_names = feature_columns
        labels = labels_df.iloc[:, 0] if isinstance(labels_df, pd.DataFrame) else labels_df

        groupby_col = self._groupby_col(features_df.index)

        # Discretize to decile ranks within each cross-section date
        rank_labels = (
            labels.groupby(level=groupby_col)
            .transform(lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"))
            .fillna(0)
            .astype(int)
        )

        default_params = {
            "objective":       "lambdarank",
            "metric":          "ndcg",
            "ndcg_eval_at":    [5, 10],
            "n_estimators":    300,
            "learning_rate":   0.03,
            "num_leaves":      31,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.8,
            "bagging_freq":    5,
            "min_child_samples": 30,
            "verbose":         -1,
            "n_jobs":          -1,
            "random_state":    42,
        }
        if params:
            default_params.update(params)

        self.model = lgb.LGBMRanker(**default_params)

        all_dates = (
            features_df.index.get_level_values(groupby_col).unique().sort_values()
        )

        if len(all_dates) < n_splits:
            logger.warning(
                f"Only {len(all_dates)} dates — training on all data without CV."
            )
            X_all = features_df[feature_columns]
            y_all = rank_labels
            g_all = features_df.groupby(level=groupby_col).size().values
            self.model.fit(X_all, y_all, group=g_all)
            self._fit_calibrator_insample(features_df, feature_columns, labels)
            return {"mean_cv_ic": 0.0}

        # Walk-forward CV
        fold_size   = len(all_dates) // n_splits
        ic_scores   = []
        oos_preds   = []
        oos_wins    = []

        from scipy.stats import spearmanr

        for fold in range(n_splits - 1):
            train_end  = all_dates[fold * fold_size + fold_size]
            val_end_idx = min((fold + 1) * fold_size + fold_size, len(all_dates) - 1)
            val_end    = all_dates[val_end_idx]

            idx_col = features_df.index.get_level_values(groupby_col)
            train_mask = idx_col < train_end
            val_mask   = (idx_col >= train_end) & (idx_col < val_end)

            X_tr = features_df.loc[train_mask, feature_columns]
            y_tr = rank_labels[train_mask]
            X_va = features_df.loc[val_mask, feature_columns]
            y_va = labels[val_mask]

            g_tr = features_df[train_mask].groupby(level=groupby_col).size().values

            if len(X_tr) == 0 or len(X_va) == 0:
                continue

            self.model.fit(X_tr, y_tr, group=g_tr)

            preds = self.model.predict(X_va)
            oos_preds.extend(preds.tolist())
            oos_wins.extend((y_va > 0).astype(int).values.tolist())

            # Per-day IC
            pred_series = pd.Series(preds, index=X_va.index)
            daily_ics   = []
            for date, grp in pred_series.groupby(level=groupby_col):
                try:
                    actual = y_va.xs(date, level=groupby_col)
                    aligned = pd.concat(
                        [grp.droplevel(groupby_col), actual], axis=1
                    ).dropna()
                    if len(aligned) >= 5:
                        ic, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                        if not np.isnan(ic):
                            daily_ics.append(ic)
                except Exception:
                    pass

            fold_ic = float(np.mean(daily_ics)) if daily_ics else 0.0
            ic_scores.append(fold_ic)
            logger.info(f"  Fold {fold + 1}/{n_splits - 1}: IC={fold_ic:.4f}")

        mean_ic = float(np.mean(ic_scores)) if ic_scores else 0.0
        logger.info(f"Mean walk-forward IC: {mean_ic:.4f}")

        # Final fit on all data
        logger.info("Final fit on full dataset...")
        X_all = features_df[feature_columns]
        y_all = rank_labels
        g_all = features_df.groupby(level=groupby_col).size().values
        self.model.fit(X_all, y_all, group=g_all)

        # Calibrate on OOS predictions if we have enough
        if len(oos_preds) >= 20:
            self._fit_calibrator(np.array(oos_preds), np.array(oos_wins))
        else:
            logger.warning("Insufficient OOS data — calibrating on in-sample.")
            self._fit_calibrator_insample(features_df, feature_columns, labels)

        logger.info("Training complete.")
        return {"mean_cv_ic": mean_ic}

    def _fit_calibrator(self, raw_scores: np.ndarray, binary_wins: np.ndarray) -> None:
        from sklearn.isotonic import IsotonicRegression
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(raw_scores, binary_wins)
        logger.info(f"Calibrator fitted on {len(raw_scores)} OOS data points.")

    def _fit_calibrator_insample(
        self,
        features_df: pd.DataFrame,
        feature_columns: list[str],
        labels: pd.Series,
    ) -> None:
        from sklearn.isotonic import IsotonicRegression
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        raw = self.model.predict(features_df[feature_columns])
        wins = (labels > 0).astype(int).values
        self.calibrator.fit(raw, wins)
        logger.info("Calibrator fitted on in-sample data (OOS fallback).")

    # ── inference ───────────────────────────────────────────────────────────

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns DataFrame with columns [alpha_score, win_probability].

        alpha_score    : raw LGBMRanker score (higher = better rank)
        win_probability: isotonic-calibrated P(forward_return > 0) in [0, 1]
        """
        if self.model is None:
            raise ValueError(
                "LightGBMRankerModel is not trained or loaded. "
                "Call .train() or .load() first."
            )

        missing = [f for f in self.feature_names if f not in features_df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        X      = features_df[self.feature_names]
        scores = self.model.predict(X)

        win_probs = (
            self.calibrator.predict(scores)
            if self.calibrator is not None
            else np.full(len(scores), 0.5)
        )

        return pd.DataFrame(
            {"alpha_score": scores, "win_probability": win_probs},
            index=features_df.index,
        )

    def is_ready(self) -> bool:
        return self.model is not None

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self, model_name: str = _DEFAULT_MODEL_FILE) -> str:
        if self.model is None:
            raise ValueError("No trained model to save.")
        path = os.path.join(self.model_dir, model_name)
        with open(path, "wb") as f:
            pickle.dump({
                "model":         self.model,
                "feature_names": self.feature_names,
                "calibrator":    self.calibrator,
            }, f)
        logger.info(f"LightGBMRankerModel saved → {path}")
        return path

    def load(self, model_name: str = _DEFAULT_MODEL_FILE) -> None:
        path = os.path.join(self.model_dir, model_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ranker model not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model         = data["model"]
        self.feature_names = data["feature_names"]
        self.calibrator    = data.get("calibrator")
        logger.info(f"LightGBMRankerModel loaded from {path}")


# ---------------------------------------------------------------------------
# AlphaModel adapter — plugs ranker into orchestrator live path
# ---------------------------------------------------------------------------

try:
    from portfolio_execution.signals.base import (
        AlphaModel,
        SignalDirection,
        SignalNorm,
    )
    _ALPHA_BASE_AVAILABLE = True
except ImportError:
    _ALPHA_BASE_AVAILABLE = False
    AlphaModel = object           # fallback so class definition doesn't crash


class LightGBMRankerAlpha(AlphaModel if _ALPHA_BASE_AVAILABLE else object):
    """
    AlphaModel subclass that wraps LightGBMRankerModel.

    Plug into the live orchestrator:

        ranker = LightGBMRankerAlpha.from_saved()
        orchestrator.register_alpha(ranker)

    Each call to generate_signals():
      1. Pulls 1-min candles from state_managers
      2. Computes RANKER_FEATURES (same schema as training)
      3. Calls LightGBMRankerModel.predict() → alpha_score + win_probability
      4. Emits TradeSignal for top-N symbols where win_prob > _MIN_WIN_PROB
    """

    # Number of top-ranked symbols to emit as signals each bar
    TOP_N = 3

    def __init__(
        self,
        ranker: LightGBMRankerModel,
        top_n: int = TOP_N,
        min_win_prob: float = _MIN_WIN_PROB,
        lookback: int = 20,
    ):
        if _ALPHA_BASE_AVAILABLE:
            super().__init__(
                name="lgbm_ranker",
                lookback=lookback,
                norm=SignalNorm.RANK,
                direction=SignalDirection.LONG_SHORT,
            )
        self._ranker      = ranker
        self._top_n       = top_n
        self._min_win_prob = min_win_prob
        self._lookback    = lookback
        self._logger      = get_logger("lgbm_ranker_alpha")

    # ── factory ─────────────────────────────────────────────────────────────

    @classmethod
    def from_saved(
        cls,
        model_dir: str = _DEFAULT_MODEL_DIR,
        model_name: str = _DEFAULT_MODEL_FILE,
        top_n: int = TOP_N,
        min_win_prob: float = _MIN_WIN_PROB,
    ) -> "LightGBMRankerAlpha":
        """Load saved ranker from disk and return ready-to-use AlphaModel."""
        ranker = LightGBMRankerModel(model_dir=model_dir)
        ranker.load(model_name)
        return cls(ranker=ranker, top_n=top_n, min_win_prob=min_win_prob)

    # ── AlphaModel abstract method ───────────────────────────────────────────

    def _compute_raw_signal(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        data: wide DataFrame (index=timestamps, columns=symbols) of close prices.
        Returns: Series(symbol → alpha_score), latest cross-section only.
        """
        if not self._ranker.is_ready():
            return pd.Series(dtype=float)

        features = self._build_features_wide(data)
        if features.empty:
            return pd.Series(dtype=float)

        try:
            result = self._ranker.predict(features)
            # Return the latest cross-section scores indexed by symbol
            return result["alpha_score"]
        except Exception as exc:
            self._logger.error(f"Ranker predict failed: {exc}")
            return pd.Series(dtype=float)

    # ── orchestrator entry point ─────────────────────────────────────────────

    def generate_signals(self, state_managers: dict[str, Any]) -> list[Any]:
        """
        Called by orchestrator on every tick/bar.
        Returns list[TradeSignal] for top-ranked symbols.
        """
        if not _ALPHA_BASE_AVAILABLE:
            self._logger.error(
                "portfolio_execution.signals.base not importable — signals disabled"
            )
            return []

        if not self._ranker.is_ready():
            self._logger.warning(
                "Ranker not loaded — call LightGBMRankerAlpha.from_saved() first."
            )
            return []

        # ── build feature panel from live candles ────────────────────────────
        closes: dict[str, pd.Series] = {}
        volumes: dict[str, pd.Series] = {}

        for sym, state_mgr in state_managers.items():
            candles = getattr(state_mgr, "candles_1m", [])
            if len(candles) < self._lookback + 10:
                continue
            idx  = [pd.Timestamp(c.timestamp) for c in candles]
            closes[sym]  = pd.Series([c.close  for c in candles], index=idx)
            volumes[sym] = pd.Series([c.volume for c in candles], index=idx)

        if len(closes) < 2:
            return []

        df_close  = pd.DataFrame(closes).sort_index().ffill()
        df_volume = pd.DataFrame(volumes).sort_index().ffill()

        features = self._build_features_wide(df_close, df_volume)
        if features.empty:
            return []

        # ── run ranker ───────────────────────────────────────────────────────
        try:
            result = self._ranker.predict(features)
        except Exception as exc:
            self._logger.error(f"Ranker inference failed: {exc}")
            return []

        # ── filter and emit TradeSignals ─────────────────────────────────────
        result = result.sort_values("alpha_score", ascending=False)
        result = result[result["win_probability"] >= self._min_win_prob]
        top    = result.head(self._top_n)

        if top.empty:
            return []

        try:
            from portfolio_execution.orchestrator import build_signal
        except ImportError:
            self._logger.error("Cannot import build_signal from orchestrator")
            return []

        signals = []
        for sym, row in top.iterrows():
            state_mgr = state_managers.get(sym)
            if state_mgr is None:
                continue

            ltp = float(df_close[sym].iloc[-1]) if sym in df_close.columns else None
            if ltp is None or np.isnan(ltp):
                continue

            sig = build_signal(
                symbol=sym,
                side="LONG",
                model_score=float(row["alpha_score"]),
                obi_features={"spread_bps": 2.0},
                market_data={
                    "ltp": ltp,
                    "adv_20d": getattr(state_mgr, "get_adv", lambda: 1_000_000)() or 1_000_000,
                },
            )
            # Attach ranker metadata for downstream tracking
            sig.metadata["win_probability"] = float(row["win_probability"])
            sig.metadata["alpha_score"]     = float(row["alpha_score"])
            sig.metadata["model_version"]   = "LGBM_RANKER_v1"
            sig.alpha_source                = "lgbm_ranker"
            signals.append(sig)

        self._logger.info(
            f"Ranker emitted {len(signals)} signals "
            f"(top alpha_scores: {list(top['alpha_score'].round(4))})"
        )
        return signals

    # ── feature engineering — MUST match training (run_daytrading_backtest) ──

    @staticmethod
    def _build_features_wide(
        df_close: pd.DataFrame,
        df_volume: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Build RANKER_FEATURES for the latest bar across all symbols.
        Returns DataFrame with index=symbols, columns=RANKER_FEATURES.

        This matches feature_engineering() in run_daytrading_backtest.py:
            returns_1d   = pct_change(1)
            momentum_5d  = pct_change(5)
            momentum_10d = pct_change(10)
            volatility_10d = rolling(10).std() of returns_1d
        """
        if df_close.empty or len(df_close) < 12:
            return pd.DataFrame()

        ret1  = df_close.pct_change(1)
        mom5  = df_close.pct_change(5)
        mom10 = df_close.pct_change(10)
        vol10 = ret1.rolling(10).std()

        # Take latest row — current cross-section
        latest = pd.DataFrame({
            "returns_1d":    ret1.iloc[-1],
            "momentum_5d":   mom5.iloc[-1],
            "momentum_10d":  mom10.iloc[-1],
            "volatility_10d": vol10.iloc[-1],
        })

        latest = latest.dropna()
        return latest[RANKER_FEATURES] if not latest.empty else pd.DataFrame()