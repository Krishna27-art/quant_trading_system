"""
Backtesting Cross-Validation Framework

Implements institutional-grade cross-validation for backtesting:
- In-sample / Out-of-sample testing
- Walk-forward validation
- Time-series cross-validation
- Performance comparison across folds
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from research_platform.backtesting.engine import BacktestConfig, BacktestingEngine, BacktestResult
from utils.logger import get_logger

logger = get_logger("backtesting_cross_validation")


class CrossValidationMethod(Enum):
    """Cross-validation methods for time-series data."""

    WALK_FORWARD = "walk_forward"
    EXPANDING_WINDOW = "expanding_window"
    ROLLING_WINDOW = "rolling_window"
    K_FOLD_TIME = "k_fold_time"


@dataclass
class CrossValidationFold:
    """Represents a single fold in cross-validation."""

    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # Results
    train_result: BacktestResult | None = None
    test_result: BacktestResult | None = None

    # Performance comparison
    performance_drift: float | None = None
    sharpe_drift: float | None = None
    max_drawdown_drift: float | None = None


@dataclass
class CrossValidationSummary:
    """Summary of cross-validation results."""

    method: CrossValidationMethod
    total_folds: int
    fold_results: list[CrossValidationFold] = field(default_factory=list)

    # Aggregate metrics
    avg_train_return: float = 0.0
    avg_test_return: float = 0.0
    avg_train_sharpe: float = 0.0
    avg_test_sharpe: float = 0.0
    avg_max_drawdown: float = 0.0

    # Stability metrics
    return_std: float = 0.0
    sharpe_std: float = 0.0
    consistency_score: float = 0.0

    # Overfitting detection
    overfitting_detected: bool = False
    overfitting_score: float = 0.0


class BacktestCrossValidator:
    """
    Cross-validation framework for backtesting.

    Prevents overfitting by testing strategy performance on out-of-sample data.
    """

    def __init__(
        self,
        base_config: BacktestConfig,
        method: CrossValidationMethod = CrossValidationMethod.WALK_FORWARD,
    ):
        """
        Initialize cross-validator.

        Args:
            base_config: Base backtest configuration
            method: Cross-validation method
        """
        self.base_config = base_config
        self.method = method
        self.logger = logger

    def generate_folds(
        self,
        start_date: datetime,
        end_date: datetime,
        n_folds: int = 5,
        train_size_months: int = 12,
        test_size_months: int = 3,
    ) -> list[CrossValidationFold]:
        """
        Generate cross-validation folds.

        Args:
            start_date: Overall start date
            end_date: Overall end date
            n_folds: Number of folds
            train_size_months: Size of training window in months
            test_size_months: Size of test window in months

        Returns:
            List of CrossValidationFold
        """
        self.logger.info(f"Generating {n_folds} folds using {self.method.value} method")

        folds = []

        if self.method == CrossValidationMethod.WALK_FORWARD:
            folds = self._generate_walk_forward_folds(
                start_date, end_date, n_folds, train_size_months, test_size_months
            )
        elif self.method == CrossValidationMethod.EXPANDING_WINDOW:
            folds = self._generate_expanding_window_folds(
                start_date, end_date, n_folds, train_size_months, test_size_months
            )
        elif self.method == CrossValidationMethod.ROLLING_WINDOW:
            folds = self._generate_rolling_window_folds(
                start_date, end_date, n_folds, train_size_months, test_size_months
            )
        elif self.method == CrossValidationMethod.K_FOLD_TIME:
            folds = self._generate_k_fold_time_folds(start_date, end_date, n_folds)

        self.logger.info(f"Generated {len(folds)} folds")
        return folds

    def _generate_walk_forward_folds(
        self,
        start_date: datetime,
        end_date: datetime,
        n_folds: int,
        train_size_months: int,
        test_size_months: int,
    ) -> list[CrossValidationFold]:
        """Generate walk-forward folds."""
        folds = []
        current_date = start_date

        for i in range(n_folds):
            train_start = current_date
            train_end = train_start + timedelta(days=train_size_months * 30)
            test_start = train_end
            test_end = test_start + timedelta(days=test_size_months * 30)

            if test_end > end_date:
                break

            fold = CrossValidationFold(
                fold_id=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
            folds.append(fold)

            # Move to next fold
            current_date = test_start

        return folds

    def _generate_expanding_window_folds(
        self,
        start_date: datetime,
        end_date: datetime,
        n_folds: int,
        train_size_months: int,
        test_size_months: int,
    ) -> list[CrossValidationFold]:
        """Generate expanding window folds (train window grows)."""
        folds = []
        current_date = start_date
        train_start = start_date

        for i in range(n_folds):
            train_end = current_date
            test_start = train_end
            test_end = test_start + timedelta(days=test_size_months * 30)

            if test_end > end_date:
                break

            fold = CrossValidationFold(
                fold_id=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
            folds.append(fold)

            # Expand train window
            current_date = test_start

        return folds

    def _generate_rolling_window_folds(
        self,
        start_date: datetime,
        end_date: datetime,
        n_folds: int,
        train_size_months: int,
        test_size_months: int,
    ) -> list[CrossValidationFold]:
        """Generate rolling window folds (fixed train window size)."""
        folds = []
        current_date = start_date

        for i in range(n_folds):
            train_start = current_date
            train_end = train_start + timedelta(days=train_size_months * 30)
            test_start = train_end
            test_end = test_start + timedelta(days=test_size_months * 30)

            if test_end > end_date:
                break

            fold = CrossValidationFold(
                fold_id=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
            folds.append(fold)

            # Roll forward
            current_date = current_date + timedelta(days=test_size_months * 30)

        return folds

    def _generate_k_fold_time_folds(
        self, start_date: datetime, end_date: datetime, n_folds: int
    ) -> list[CrossValidationFold]:
        """Generate K-fold time series folds."""
        total_days = (end_date - start_date).days
        fold_size = total_days // n_folds

        folds = []
        for i in range(n_folds):
            # Train on all folds except i
            train_start = start_date
            train_end = start_date + timedelta(days=i * fold_size)

            # Test on fold i
            test_start = train_end
            test_end = test_start + timedelta(days=fold_size)

            fold = CrossValidationFold(
                fold_id=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
            folds.append(fold)

        return folds

    def run_cross_validation(
        self, folds: list[CrossValidationFold], predictions: list[Any], price_data: pd.DataFrame
    ) -> CrossValidationSummary:
        """
        Run cross-validation on given folds.

        Args:
            folds: List of cross-validation folds
            predictions: List of predictions
            price_data: Price data

        Returns:
            CrossValidationSummary
        """
        self.logger.info(f"Running cross-validation on {len(folds)} folds")

        summary = CrossValidationSummary(method=self.method, total_folds=len(folds))

        train_returns = []
        test_returns = []
        train_sharpes = []
        test_sharpes = []
        max_drawdowns = []

        for fold in folds:
            self.logger.info(f"Processing fold {fold.fold_id + 1}/{len(folds)}")

            # Filter data for train period
            train_predictions = self._filter_predictions_by_date(
                predictions, fold.train_start, fold.train_end
            )
            train_price_data = self._filter_price_data_by_date(
                price_data, fold.train_start, fold.train_end
            )

            # Run train backtest
            train_config = BacktestConfig(
                start_date=fold.train_start,
                end_date=fold.train_end,
                initial_capital=self.base_config.initial_capital,
                max_positions=self.base_config.max_positions,
                position_size=self.base_config.position_size,
                rebalance_frequency=self.base_config.rebalance_frequency,
                commission_rate=self.base_config.commission_rate,
                slippage_rate=self.base_config.slippage_rate,
            )

            train_engine = BacktestingEngine(train_config)
            fold.train_result = train_engine.run_backtest(train_predictions, train_price_data)

            # Filter data for test period
            test_predictions = self._filter_predictions_by_date(
                predictions, fold.test_start, fold.test_end
            )
            test_price_data = self._filter_price_data_by_date(
                price_data, fold.test_start, fold.test_end
            )

            # Run test backtest
            test_config = BacktestConfig(
                start_date=fold.test_start,
                end_date=fold.test_end,
                initial_capital=self.base_config.initial_capital,
                max_positions=self.base_config.max_positions,
                position_size=self.base_config.position_size,
                rebalance_frequency=self.base_config.rebalance_frequency,
                commission_rate=self.base_config.commission_rate,
                slippage_rate=self.base_config.slippage_rate,
            )

            test_engine = BacktestingEngine(test_config)
            fold.test_result = test_engine.run_backtest(test_predictions, test_price_data)

            # Calculate performance drift
            fold.performance_drift = fold.test_result.total_return - fold.train_result.total_return
            fold.sharpe_drift = fold.test_result.sharpe_ratio - fold.train_result.sharpe_ratio
            fold.max_drawdown_drift = fold.test_result.max_drawdown - fold.train_result.max_drawdown

            # Collect metrics
            train_returns.append(fold.train_result.total_return)
            test_returns.append(fold.test_result.total_return)
            train_sharpes.append(fold.train_result.sharpe_ratio)
            test_sharpes.append(fold.test_result.sharpe_ratio)
            max_drawdowns.append(fold.test_result.max_drawdown)

            summary.fold_results.append(fold)

        # Calculate aggregate metrics
        summary.avg_train_return = np.mean(train_returns)
        summary.avg_test_return = np.mean(test_returns)
        summary.avg_train_sharpe = np.mean(train_sharpes)
        summary.avg_test_sharpe = np.mean(test_sharpes)
        summary.avg_max_drawdown = np.mean(max_drawdowns)

        # Calculate stability metrics
        summary.return_std = np.std(test_returns)
        summary.sharpe_std = np.std(test_sharpes)

        # Calculate consistency score (inverse of coefficient of variation)
        if summary.avg_test_return != 0:
            summary.consistency_score = 1 - (summary.return_std / abs(summary.avg_test_return))
        else:
            summary.consistency_score = 0.0

        # Detect overfitting
        summary.overfitting_score = summary.avg_train_return - summary.avg_test_return
        summary.overfitting_detected = summary.overfitting_score > 0.1  # 10% difference threshold

        if summary.overfitting_detected:
            self.logger.warning(
                f"Overfitting detected: Train return {summary.avg_train_return:.2%} "
                f"> Test return {summary.avg_test_return:.2%}"
            )

        self.logger.info(
            f"Cross-validation complete. Avg test return: {summary.avg_test_return:.2%}"
        )

        return summary

    def _filter_predictions_by_date(
        self, predictions: list[Any], start_date: datetime, end_date: datetime
    ) -> list[Any]:
        """Filter predictions by date range."""
        return [p for p in predictions if start_date <= p.date <= end_date]

    def _filter_price_data_by_date(
        self, price_data: pd.DataFrame, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Filter price data by date range."""
        mask = (price_data["date"] >= start_date) & (price_data["date"] <= end_date)
        return price_data[mask].copy()

    def compare_folds(self, summary: CrossValidationSummary) -> dict[str, Any]:
        """
        Compare performance across folds.

        Args:
            summary: Cross-validation summary

        Returns:
            Comparison results
        """
        comparison = {
            "method": summary.method.value,
            "total_folds": summary.total_folds,
            "fold_comparison": [],
        }

        for fold in summary.fold_results:
            comparison["fold_comparison"].append(
                {
                    "fold_id": fold.fold_id,
                    "train_return": fold.train_result.total_return if fold.train_result else None,
                    "test_return": fold.test_result.total_return if fold.test_result else None,
                    "performance_drift": fold.performance_drift,
                    "train_sharpe": fold.train_result.sharpe_ratio if fold.train_result else None,
                    "test_sharpe": fold.test_result.sharpe_ratio if fold.test_result else None,
                    "sharpe_drift": fold.sharpe_drift,
                }
            )

        return comparison


def run_walk_forward_validation(
    base_config: BacktestConfig,
    predictions: list[Any],
    price_data: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    n_folds: int = 5,
    train_size_months: int = 12,
    test_size_months: int = 3,
) -> CrossValidationSummary:
    """
    Convenience function to run walk-forward validation.

    Args:
        base_config: Base backtest configuration
        predictions: List of predictions
        price_data: Price data
        start_date: Overall start date
        end_date: Overall end date
        n_folds: Number of folds
        train_size_months: Size of training window in months
        test_size_months: Size of test window in months

    Returns:
        CrossValidationSummary
    """
    validator = BacktestCrossValidator(
        base_config=base_config, method=CrossValidationMethod.WALK_FORWARD
    )

    folds = validator.generate_folds(
        start_date=start_date,
        end_date=end_date,
        n_folds=n_folds,
        train_size_months=train_size_months,
        test_size_months=test_size_months,
    )

    summary = validator.run_cross_validation(folds, predictions, price_data)

    return summary
