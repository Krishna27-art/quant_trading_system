"""
Signal Adapter Module

Provides a unified adapter for converting model predictions to trading signals
across all consumers (backtest, live orchestrator, paper trading).

This replaces the inline SignalPrediction class that was duplicated in
scripts/run_daytrading_backtest.py and ensures consistent signal translation
throughout the system.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SignalPrediction(BaseModel):
    """
    Unified signal prediction model for all consumers.

    This is the canonical contract between ML models and execution systems.
    All backtests, live trading, and paper trading should use this same
    structure to ensure consistent signal handling.
    """

    date: datetime = Field(..., description="Signal timestamp")
    symbol: str = Field(..., description="Instrument symbol")
    prediction: int = Field(..., description="Prediction class (0=hold, 1=short, 2=long)")
    confidence: float = Field(..., description="Model confidence score")
    win_probability: float = Field(..., description="Estimated win probability")

    # Optional extended fields for institutional-grade signals
    target_price: float | None = Field(None, description="Suggested take-profit price")
    stop_loss: float | None = Field(None, description="Suggested stop-loss price")
    expected_return: float | None = Field(None, description="Expected return %")
    risk_reward_ratio: float | None = Field(None, description="Risk/reward ratio")
    model_version: str | None = Field(None, description="Model version identifier")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional signal context")

    @property
    def is_long(self) -> bool:
        """Check if signal is a long recommendation."""
        return self.prediction == 2

    @property
    def is_short(self) -> bool:
        """Check if signal is a short recommendation."""
        return self.prediction == 1

    @property
    def is_hold(self) -> bool:
        """Check if signal is a hold recommendation."""
        return self.prediction == 0

    def to_trade_signal(self, entry_price: float) -> dict[str, Any]:
        """
        Convert to a trade signal dict compatible with orchestrator.

        Args:
            entry_price: Current market price for the instrument

        Returns:
            Dictionary with trade signal fields
        """
        direction = "long" if self.is_long else "short" if self.is_short else "hold"

        # Calculate default stop/target if not provided
        if self.is_long:
            default_stop = entry_price * 0.9925  # 0.75% stop loss
            default_target = entry_price * 1.015  # 1.5% target
        elif self.is_short:
            default_stop = entry_price * 1.0075
            default_target = entry_price * 0.985
        else:
            default_stop = None
            default_target = None

        return {
            "symbol": self.symbol,
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": self.stop_loss or default_stop,
            "target_price": self.target_price or default_target,
            "confidence": self.confidence,
            "win_probability": self.win_probability,
            "model_version": self.model_version,
            "metadata": self.metadata,
        }


def from_lightgbm_ranker_output(
    df_row: dict[str, Any], date: datetime, model_version: str = "lightgbm_ranker_v1"
) -> SignalPrediction:
    """
    Create SignalPrediction from LightGBMRankerModel output row.

    This is the adapter for the specific output format of
    prediction_intelligence.lightgbm_ranker.LightGBMRankerModel.predict()

    Args:
        df_row: Dictionary row from model prediction output
        date: Signal timestamp
        model_version: Model version identifier

    Returns:
        SignalPrediction instance
    """
    alpha_score = df_row.get("alpha_score", 0.0)
    win_prob = df_row.get("win_probability", 0.5)

    # Map alpha_score to prediction class
    # Top quintile (alpha_score > 0.8) -> long (2)
    # Bottom quintile (alpha_score < -0.8) -> short (1)
    # Otherwise -> hold (0)
    if alpha_score > 0.8:
        prediction = 2
    elif alpha_score < -0.8:
        prediction = 1
    else:
        prediction = 0

    return SignalPrediction(
        date=date,
        symbol=df_row.get("symbol", ""),
        prediction=prediction,
        confidence=abs(alpha_score),
        win_probability=win_prob,
        model_version=model_version,
        metadata={"alpha_score": alpha_score, "raw_features": df_row},
    )


def from_base_logistic_output(
    symbol: str,
    prediction: int,
    win_probability: float,
    confidence: float,
    date: datetime,
    model_version: str = "base_logistic_v1",
) -> SignalPrediction:
    """
    Create SignalPrediction from BaseLogistic model output.

    This is the adapter for prediction_intelligence.base_logistic.BaseLogistic

    Args:
        symbol: Instrument symbol
        prediction: Model prediction (0/1/2)
        win_probability: Estimated win probability
        confidence: Model confidence score
        date: Signal timestamp
        model_version: Model version identifier

    Returns:
        SignalPrediction instance
    """
    return SignalPrediction(
        date=date,
        symbol=symbol,
        prediction=prediction,
        confidence=confidence,
        win_probability=win_probability,
        model_version=model_version,
    )
