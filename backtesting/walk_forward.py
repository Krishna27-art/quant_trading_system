"""
Walk-Forward Validation Module

Implements walk-forward validation with rolling train/test windows.
This is much more reliable than single train/test split for time series data.

Example:
    2015-2018 Train → 2019 Test
    2016-2019 Train → 2020 Test
    2017-2020 Train → 2021 Test
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from utils.logger import get_logger

logger = get_logger("walk_forward")


@dataclass
class WalkForwardWindow:
    """Represents a single train/test window."""
    
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    window_id: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "train_days": (self.train_end - self.train_start).days + 1,
            "test_days": (self.test_end - self.test_start).days + 1,
        }


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    
    start_date: date
    end_date: date
    train_period_months: int = 36  # 3 years training
    test_period_months: int = 6   # 6 months testing
    step_months: int = 6          # Roll forward by 6 months
    
    def __post_init__(self):
        """Validate configuration."""
        if self.train_period_months < 12:
            raise ValueError("Train period should be at least 12 months")
        if self.test_period_months < 1:
            raise ValueError("Test period should be at least 1 month")
        if self.step_months < 1:
            raise ValueError("Step period should be at least 1 month")


class WalkForwardValidator:
    """
    Walk-forward validation engine.
    
    Generates rolling train/test windows and executes backtests on each.
    """
    
    def __init__(self, config: WalkForwardConfig):
        """
        Initialize the walk-forward validator.
        
        Args:
            config: Walk-forward configuration
        """
        self.config = config
        self.logger = logger
        self.windows = self._generate_windows()
    
    def _generate_windows(self) -> list[WalkForwardWindow]:
        """Generate all train/test windows."""
        windows = []
        
        current_train_start = self.config.start_date
        window_id = 0
        
        while True:
            train_end = current_train_start + timedelta(days=self.config.train_period_months * 30)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.config.test_period_months * 30)
            
            # Check if we've gone beyond the end date
            if test_end > self.config.end_date:
                break
            
            window = WalkForwardWindow(
                train_start=current_train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                window_id=window_id,
            )
            windows.append(window)
            
            # Roll forward
            current_train_start += timedelta(days=self.config.step_months * 30)
            window_id += 1
        
        self.logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows
    
    def run_validation(
        self,
        train_func: Callable,
        test_func: Callable,
        data_loader: Callable,
    ) -> dict[str, Any]:
        """
        Run walk-forward validation.
        
        Args:
            train_func: Function to train model (takes train_data, returns model)
            test_func: Function to test model (takes model, test_data, returns results)
            data_loader: Function to load data for a date range
            
        Returns:
            Combined results from all windows
        """
        self.logger.info("Starting walk-forward validation...")
        
        all_results = []
        
        for window in self.windows:
            self.logger.info(f"Processing window {window.window_id}: {window.train_start} to {window.test_end}")
            
            try:
                # Load training data
                train_data = data_loader(window.train_start, window.train_end)
                
                # Train model
                model = train_func(train_data)
                
                # Load test data
                test_data = data_loader(window.test_start, window.test_end)
                
                # Test model
                results = test_func(model, test_data)
                results["window"] = window.to_dict()
                
                all_results.append(results)
                
                self.logger.info(f"Window {window.window_id} complete")
                
            except Exception as e:
                self.logger.error(f"Window {window.window_id} failed: {e}")
                all_results.append({
                    "window": window.to_dict(),
                    "error": str(e),
                    "success": False,
                })
        
        # Aggregate results
        aggregated = self._aggregate_results(all_results)
        
        self.logger.info("Walk-forward validation complete")
        return {
            "windows": all_results,
            "aggregated": aggregated,
            "config": {
                "start_date": self.config.start_date.isoformat(),
                "end_date": self.config.end_date.isoformat(),
                "train_period_months": self.config.train_period_months,
                "test_period_months": self.config.test_period_months,
                "step_months": self.config.step_months,
                "total_windows": len(self.windows),
            },
        }
    
    def _aggregate_results(self, all_results: list[dict]) -> dict[str, Any]:
        """Aggregate results across all windows."""
        successful_results = [r for r in all_results if r.get("success", True)]
        
        if not successful_results:
            return {"error": "No successful windows"}
        
        # Extract numeric metrics
        metrics = {}
        for result in successful_results:
            for key, value in result.items():
                if key != "window" and isinstance(value, (int, float)):
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
        
        # Compute statistics
        aggregated = {}
        for metric, values in metrics.items():
            aggregated[metric] = {
                "mean": sum(values) / len(values),
                "std": pd.Series(values).std(),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }
        
        return aggregated
    
    def get_windows_summary(self) -> pd.DataFrame:
        """Get summary of all windows as DataFrame."""
        data = [window.to_dict() for window in self.windows]
        return pd.DataFrame(data)


def create_walk_forward_config(
    start_date: str | date,
    end_date: str | date,
    train_period_months: int = 36,
    test_period_months: int = 6,
    step_months: int = 6,
) -> WalkForwardConfig:
    """
    Convenience function to create walk-forward config.
    
    Args:
        start_date: Start date (ISO string or date object)
        end_date: End date (ISO string or date object)
        train_period_months: Training period in months
        test_period_months: Testing period in months
        step_months: Step size in months
        
    Returns:
        WalkForwardConfig object
    """
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = date.fromisoformat(end_date)
    
    return WalkForwardConfig(
        start_date=start_date,
        end_date=end_date,
        train_period_months=train_period_months,
        test_period_months=test_period_months,
        step_months=step_months,
    )
