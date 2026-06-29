"""
2-Week Paper Trading System

Live predictions, no money. Run model every day at 3:25 PM IST.
"""

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("paper_trade_log")


@dataclass
class PaperTradeEntry:
    """Paper trade log entry."""

    date: str
    top_picks: dict[str, float]  # symbol -> score
    bottom_picks: dict[str, float]  # symbol -> score
    nifty_level: float | None = None
    nifty_return: float | None = None
    top_picks_return: float | None = None
    alpha: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date,
            "top_picks": self.top_picks,
            "bottom_picks": self.bottom_picks,
            "nifty_level": self.nifty_level,
            "nifty_return": self.nifty_return,
            "top_picks_return": self.top_picks_return,
            "alpha": self.alpha,
        }


@dataclass
class PaperTradeResults:
    """Results from paper trading period."""

    total_days: int
    winning_days: int  # Days where top picks beat NIFTY
    win_rate: float
    average_alpha: float
    cumulative_alpha: float
    nifty_cumulative_return: float
    top_picks_cumulative_return: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_days": self.total_days,
            "winning_days": self.winning_days,
            "win_rate": self.win_rate,
            "average_alpha": self.average_alpha,
            "cumulative_alpha": self.cumulative_alpha,
            "nifty_cumulative_return": self.nifty_cumulative_return,
            "top_picks_cumulative_return": self.top_picks_cumulative_return,
        }


def daily_prediction_log(
    model: Any, feature_store: Any, log_file: str = "paper_trade_log.jsonl", top_n: int = 10
) -> PaperTradeEntry:
    """
    Run at 3:25 PM IST every trading day.
    Produces next-day predicted outperformers.

    Args:
        model: Prediction model
        feature_store: Feature store with latest features
        log_file: Path to log file
        top_n: Number of top picks to log

    Returns:
        Paper trade entry
    """
    today = date.today().isoformat()

    # Get today's features (already computed in feature store)
    todays_features = feature_store.get_latest()

    # Predict scores
    scores = model.predict(todays_features)
    scores_series = pd.Series(scores, index=todays_features.index)

    # Top 10 predicted outperformers
    top_picks = scores_series.nlargest(top_n)
    bottom_picks = scores_series.nsmallest(top_n)

    # Get NIFTY level
    nifty_level = get_nifty_close()

    log_entry = PaperTradeEntry(
        date=today,
        top_picks=top_picks.to_dict(),
        bottom_picks=bottom_picks.to_dict(),
        nifty_level=nifty_level,
    )

    # Append to log file
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry.to_dict()) + "\n")

    logger.info(f"[{today}] Top picks: {list(top_picks.index)}")
    print(f"[{today}] Top picks: {list(top_picks.index)}")

    return log_entry


def get_nifty_close() -> float | None:
    """
    Get NIFTY closing price.

    In production, this would fetch from data source.
    For now, returns None.

    Returns:
        NIFTY closing price or None
    """
    # In production, fetch from NSE/Yahoo Finance
    # For now, return None as placeholder
    return None


def score_yesterday(
    log_file: str = "paper_trade_log.jsonl", price_df: pd.DataFrame | None = None
) -> float | None:
    """
    Next day at 3:25 PM — score yesterday's picks.

    Args:
        log_file: Path to log file
        price_df: DataFrame with prices (date x symbol)

    Returns:
        Alpha (top picks return - NIFTY return) or None
    """
    log_path = Path(log_file)

    if not log_path.exists():
        logger.warning(f"Log file {log_file} does not exist")
        return None

    with open(log_path) as f:
        lines = f.readlines()

    if len(lines) < 2:
        logger.warning("Need at least 2 days of data")
        return None

    yesterday_data = json.loads(lines[-2])
    today_data = json.loads(lines[-1])

    yesterday_date = yesterday_data["date"]
    today_date = today_data["date"]

    # Calculate top picks return
    if price_df is not None:
        top_picks = yesterday_data["top_picks"]
        top_picks_symbols = list(top_picks.keys())

        returns = []
        for symbol in top_picks_symbols:
            if symbol in price_df.columns:
                yesterday_price = price_df.loc[yesterday_date, symbol]
                today_price = price_df.loc[today_date, symbol]

                if pd.notna(yesterday_price) and pd.notna(today_price) and yesterday_price > 0:
                    stock_return = today_price / yesterday_price - 1
                    returns.append(stock_return)

        top_return = np.mean(returns) if returns else 0.0
    else:
        top_return = 0.0

    # Calculate NIFTY return
    yesterday_nifty = yesterday_data.get("nifty_level")
    today_nifty = today_data.get("nifty_level")

    if yesterday_nifty and today_nifty and yesterday_nifty > 0:
        nifty_return = today_nifty / yesterday_nifty - 1
    else:
        nifty_return = 0.0

    # Calculate alpha
    alpha = top_return - nifty_return

    # Update log file with performance data
    updated_today_data = today_data.copy()
    updated_today_data["nifty_return"] = nifty_return
    updated_today_data["top_picks_return"] = top_return
    updated_today_data["alpha"] = alpha

    # Rewrite the last line with updated data
    with open(log_path) as f:
        lines = f.readlines()

    lines[-1] = json.dumps(updated_today_data) + "\n"

    with open(log_path, "w") as f:
        f.writelines(lines)

    logger.info(
        f"Yesterday's picks: {top_return:.2%} vs NIFTY: {nifty_return:.2%} | Alpha: {alpha:.2%}"
    )
    print(f"Yesterday's picks: {top_return:.2%} vs NIFTY: {nifty_return:.2%} | Alpha: {alpha:.2%}")

    return alpha


def evaluate_paper_trading(
    log_file: str = "paper_trade_log.jsonl", min_days: int = 10, required_win_rate: float = 0.6
) -> PaperTradeResults:
    """
    Evaluate paper trading results.

    After 10 trading days, top picks should beat NIFTY on at least 6 out of 10 days.

    Args:
        log_file: Path to log file
        min_days: Minimum number of days required
        required_win_rate: Required win rate (default 0.6 = 60%)

    Returns:
        Paper trade results
    """
    log_path = Path(log_file)

    if not log_path.exists():
        logger.warning(f"Log file {log_file} does not exist")
        return PaperTradeResults(
            total_days=0,
            winning_days=0,
            win_rate=0.0,
            average_alpha=0.0,
            cumulative_alpha=0.0,
            nifty_cumulative_return=0.0,
            top_picks_cumulative_return=0.0,
        )

    with open(log_path) as f:
        lines = f.readlines()

    if len(lines) < min_days:
        logger.warning(f"Need at least {min_days} days, have {len(lines)}")
        return PaperTradeResults(
            total_days=len(lines),
            winning_days=0,
            win_rate=0.0,
            average_alpha=0.0,
            cumulative_alpha=0.0,
            nifty_cumulative_return=0.0,
            top_picks_cumulative_return=0.0,
        )

    # Count winning days (alpha > 0)
    winning_days = 0
    alphas = []
    nifty_returns = []
    top_picks_returns = []

    for line in lines:
        data = json.loads(line)
        alpha = data.get("alpha")
        nifty_return = data.get("nifty_return")
        top_picks_return = data.get("top_picks_return")

        if alpha is not None:
            alphas.append(alpha)
            if alpha > 0:
                winning_days += 1

        if nifty_return is not None:
            nifty_returns.append(nifty_return)

        if top_picks_return is not None:
            top_picks_returns.append(top_picks_return)

    total_days = len(alphas)
    win_rate = winning_days / total_days if total_days > 0 else 0.0
    average_alpha = np.mean(alphas) if alphas else 0.0
    cumulative_alpha = sum(alphas) if alphas else 0.0

    # Calculate cumulative returns
    nifty_cumulative = (1 + pd.Series(nifty_returns)).prod() - 1 if nifty_returns else 0.0
    top_picks_cumulative = (
        (1 + pd.Series(top_picks_returns)).prod() - 1 if top_picks_returns else 0.0
    )

    results = PaperTradeResults(
        total_days=total_days,
        winning_days=winning_days,
        win_rate=win_rate,
        average_alpha=average_alpha,
        cumulative_alpha=cumulative_alpha,
        nifty_cumulative_return=nifty_cumulative,
        top_picks_cumulative_return=top_picks_cumulative,
    )

    # Print results
    print("\n=== PAPER TRADING RESULTS ===")
    print(f"Total Days:          {total_days}")
    print(f"Winning Days:        {winning_days}/{total_days}")
    print(f"Win Rate:            {win_rate:.1%} (need > {required_win_rate:.0%})")
    print(f"Average Alpha:       {average_alpha:.2%}")
    print(f"Cumulative Alpha:    {cumulative_alpha:.2%}")
    print(f"NIFTY Cumulative:    {nifty_cumulative:.2%}")
    print(f"Top Picks Cumulative: {top_picks_cumulative:.2%}")

    # Decision
    if total_days >= min_days and win_rate >= required_win_rate:
        print("\n✓ PASS: Proceed to real capital")
    else:
        print(
            f"\n✗ FAIL: Go back to feature selection (need {min_days} days, {required_win_rate:.0%} win rate)"
        )

    return results


def validate_paper_trading(
    results: PaperTradeResults, min_days: int = 10, required_win_rate: float = 0.6
) -> bool:
    """
    Validate if paper trading results are sufficient for real capital.

    Args:
        results: Paper trade results
        min_days: Minimum number of days required
        required_win_rate: Required win rate

    Returns:
        True if ready for real capital, False otherwise
    """
    if results.total_days < min_days:
        logger.warning(f"Insufficient days: {results.total_days} < {min_days}")
        return False

    if results.win_rate < required_win_rate:
        logger.warning(f"Win rate too low: {results.win_rate:.1%} < {required_win_rate:.1%}")
        return False

    logger.info(
        f"Paper trading validation passed: {results.winning_days}/{results.total_days} winning days"
    )
    return True


def get_paper_trade_history(log_file: str = "paper_trade_log.jsonl") -> list[PaperTradeEntry]:
    """
    Get paper trading history from log file.

    Args:
        log_file: Path to log file

    Returns:
        List of paper trade entries
    """
    log_path = Path(log_file)

    if not log_path.exists():
        return []

    entries = []

    with open(log_path) as f:
        for line in f:
            data = json.loads(line)
            entry = PaperTradeEntry(
                date=data["date"],
                top_picks=data["top_picks"],
                bottom_picks=data["bottom_picks"],
                nifty_level=data.get("nifty_level"),
                nifty_return=data.get("nifty_return"),
                top_picks_return=data.get("top_picks_return"),
                alpha=data.get("alpha"),
            )
            entries.append(entry)

    return entries
