"""
Backtest Results Analysis

Institutional-grade analysis of backtesting results.
Provides detailed insights into strategy performance.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from research_platform.backtesting.performance_metrics import PerformanceMetrics
from utils.logger import get_logger

logger = get_logger("results_analysis")


class MonthlyPerformance(BaseModel):
    """Monthly performance summary."""

    year: int = Field(..., description="Year")
    month: int = Field(..., description="Month")
    return_pct: float = Field(..., description="Monthly return percentage")
    volatility: float = Field(..., description="Monthly volatility")
    sharpe_ratio: float = Field(..., description="Monthly Sharpe ratio")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class SectorPerformance(BaseModel):
    """Sector-wise performance summary."""

    sector: str = Field(..., description="Sector name")
    return_pct: float = Field(..., description="Sector return percentage")
    contribution_pct: float = Field(..., description="Contribution to portfolio return")
    exposure_pct: float = Field(..., description="Average sector exposure")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class DrawdownAnalysis(BaseModel):
    """Drawdown analysis."""

    start_date: datetime = Field(..., description="Drawdown start date")
    end_date: datetime = Field(..., description="Drawdown end date")
    depth_pct: float = Field(..., description="Drawdown depth percentage")
    duration_days: int = Field(..., description="Drawdown duration in days")
    recovery_days: int | None = Field(None, description="Recovery duration in days")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class BacktestAnalysis(BaseModel):
    """Comprehensive backtest analysis."""

    # Overall metrics
    metrics: PerformanceMetrics = Field(..., description="Performance metrics")

    # Monthly breakdown
    monthly_performance: list[MonthlyPerformance] = Field(
        default_factory=list, description="Monthly performance"
    )

    # Sector breakdown
    sector_performance: list[SectorPerformance] = Field(
        default_factory=list, description="Sector performance"
    )

    # Drawdown analysis
    drawdowns: list[DrawdownAnalysis] = Field(default_factory=list, description="Drawdown analysis")

    # Trade analysis
    best_trade: dict[str, Any] | None = Field(None, description="Best trade")
    worst_trade: dict[str, Any] | None = Field(None, description="Worst trade")
    avg_holding_period: float = Field(default=0.0, description="Average holding period in days")

    # Position analysis
    top_performers: list[dict[str, Any]] = Field(
        default_factory=list, description="Top performing positions"
    )
    worst_performers: list[dict[str, Any]] = Field(
        default_factory=list, description="Worst performing positions"
    )

    # Risk analysis
    var_95: float | None = Field(None, description="95% VaR")
    var_99: float | None = Field(None, description="99% VaR")
    cvar_95: float | None = Field(None, description="95% CVaR")

    # Recommendations
    strengths: list[str] = Field(default_factory=list, description="Strategy strengths")
    weaknesses: list[str] = Field(default_factory=list, description="Strategy weaknesses")
    recommendations: list[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )

    analyzed_at: datetime = Field(
        default_factory=datetime.now, description="When analysis was performed"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ResultsAnalyzer:
    """
    Analyzer for backtesting results.

    Provides comprehensive analysis and insights.
    """

    def __init__(self):
        """Initialize the results analyzer."""
        self.logger = logger

    def analyze_results(
        self,
        metrics: PerformanceMetrics,
        equity_curve: list[tuple[datetime, float]],
        returns: list[float],
        positions: list[Any],
        sector_data: pd.DataFrame | None = None,
    ) -> BacktestAnalysis:
        """
        Analyze backtesting results.

        Args:
            metrics: Performance metrics
            equity_curve: Equity curve data
            returns: Daily returns
            positions: All positions
            sector_data: Sector mapping data

        Returns:
            BacktestAnalysis
        """
        self.logger.info("Analyzing backtest results")

        # Monthly performance
        monthly_performance = self._analyze_monthly_performance(equity_curve, returns)

        # Sector performance
        sector_performance = self._analyze_sector_performance(positions, sector_data)

        # Drawdown analysis
        drawdowns = self._analyze_drawdowns(equity_curve)

        # Trade analysis
        best_trade, worst_trade, avg_holding_period = self._analyze_trades(positions)

        # Position analysis
        top_performers, worst_performers = self._analyze_positions(positions)

        # Risk analysis
        var_95, var_99, cvar_95 = self._calculate_risk_metrics(returns)

        # Generate insights
        strengths, weaknesses, recommendations = self._generate_insights(metrics, drawdowns)

        analysis = BacktestAnalysis(
            metrics=metrics,
            monthly_performance=monthly_performance,
            sector_performance=sector_performance,
            drawdowns=drawdowns,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_holding_period=avg_holding_period,
            top_performers=top_performers,
            worst_performers=worst_performers,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
        )

        self.logger.info("Backtest analysis completed")

        return analysis

    def _analyze_monthly_performance(
        self, equity_curve: list[tuple[datetime, float]], returns: list[float]
    ) -> list[MonthlyPerformance]:
        """Analyze monthly performance."""
        if not equity_curve or not returns:
            return []

        # Create DataFrame
        df = pd.DataFrame(equity_curve, columns=["date", "value"])
        df["returns"] = [0] + returns

        # Extract year and month
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        # Group by year-month
        monthly = (
            df.groupby(["year", "month"])
            .agg({"returns": lambda x: (1 + x).prod() - 1, "value": "std"})
            .reset_index()
        )

        monthly_performance = []
        for _, row in monthly.iterrows():
            monthly_performance.append(
                MonthlyPerformance(
                    year=int(row["year"]),
                    month=int(row["month"]),
                    return_pct=float(row["returns"]),
                    volatility=float(row["value"]),
                    sharpe_ratio=float(row["returns"] / row["value"] if row["value"] > 0 else 0),
                )
            )

        return monthly_performance

    def _analyze_sector_performance(
        self, positions: list[Any], sector_data: pd.DataFrame | None
    ) -> list[SectorPerformance]:
        """Analyze sector-wise performance."""
        if not positions or sector_data is None:
            return []

        # Map symbols to sectors
        symbol_to_sector = dict(zip(sector_data["symbol"], sector_data["sector"], strict=False))

        # Group by sector
        sector_pnl = {}
        sector_exposure = {}

        for position in positions:
            sector = symbol_to_sector.get(position.symbol, "Unknown")

            if sector not in sector_pnl:
                sector_pnl[sector] = 0.0
                sector_exposure[sector] = 0.0

            sector_pnl[sector] += position.pnl
            sector_exposure[sector] += position.shares * position.entry_price

        # Calculate sector performance
        total_pnl = sum(sector_pnl.values())
        total_exposure = sum(sector_exposure.values())

        sector_performance = []
        for sector, pnl in sector_pnl.items():
            exposure = sector_exposure[sector]

            sector_performance.append(
                SectorPerformance(
                    sector=sector,
                    return_pct=float((pnl / exposure) * 100 if exposure > 0 else 0),
                    contribution_pct=float((pnl / total_pnl) * 100 if total_pnl > 0 else 0),
                    exposure_pct=float(
                        (exposure / total_exposure) * 100 if total_exposure > 0 else 0
                    ),
                )
            )

        # Sort by return
        sector_performance.sort(key=lambda x: x.return_pct, reverse=True)

        return sector_performance

    def _analyze_drawdowns(
        self, equity_curve: list[tuple[datetime, float]]
    ) -> list[DrawdownAnalysis]:
        """Analyze drawdowns."""
        if len(equity_curve) < 2:
            return []

        values = [v for _, v in equity_curve]
        dates = [d for d, _ in equity_curve]

        cumulative = np.array(values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        # Find drawdown periods
        drawdowns = []
        in_drawdown = False
        drawdown_start = None
        drawdown_start_value = None

        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = dates[i]
                drawdown_start_value = running_max[i]
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                drawdown_end = dates[i]
                drawdown_depth = drawdown[i - 1]
                duration = (drawdown_end - drawdown_start).days

                # Calculate recovery time
                recovery_days = None
                for j in range(i + 1, len(drawdown)):
                    if cumulative[j] >= drawdown_start_value:
                        recovery_days = (dates[j] - drawdown_end).days
                        break

                drawdowns.append(
                    DrawdownAnalysis(
                        start_date=drawdown_start,
                        end_date=drawdown_end,
                        depth_pct=float(drawdown_depth * 100),
                        duration_days=duration,
                        recovery_days=recovery_days,
                    )
                )

        # Sort by depth
        drawdowns.sort(key=lambda x: x.depth_pct, reverse=True)

        return drawdowns[:10]  # Top 10 drawdowns

    def _analyze_trades(self, positions: list[Any]) -> tuple[dict | None, dict | None, float]:
        """Analyze trades."""
        if not positions:
            return None, None, 0.0

        # Best trade
        best_position = max(positions, key=lambda p: p.pnl)
        best_trade = {
            "symbol": best_position.symbol,
            "pnl": best_position.pnl,
            "pnl_pct": best_position.pnl_pct,
            "entry_date": best_position.entry_date,
            "exit_date": best_position.exit_date,
            "holding_days": (
                (best_position.exit_date - best_position.entry_date).days
                if best_position.exit_date
                else 0
            ),
        }

        # Worst trade
        worst_position = min(positions, key=lambda p: p.pnl)
        worst_trade = {
            "symbol": worst_position.symbol,
            "pnl": worst_position.pnl,
            "pnl_pct": worst_position.pnl_pct,
            "entry_date": worst_position.entry_date,
            "exit_date": worst_position.exit_date,
            "holding_days": (
                (worst_position.exit_date - worst_position.entry_date).days
                if worst_position.exit_date
                else 0
            ),
        }

        # Average holding period
        holding_periods = []
        for position in positions:
            if position.exit_date:
                holding_periods.append((position.exit_date - position.entry_date).days)

        avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0

        return best_trade, worst_trade, avg_holding_period

    def _analyze_positions(self, positions: list[Any]) -> tuple[list[dict], list[dict]]:
        """Analyze positions."""
        if not positions:
            return [], []

        # Sort by P&L
        sorted_positions = sorted(positions, key=lambda p: p.pnl, reverse=True)

        # Top performers
        top_performers = []
        for position in sorted_positions[:10]:
            top_performers.append(
                {
                    "symbol": position.symbol,
                    "pnl": position.pnl,
                    "pnl_pct": position.pnl_pct,
                    "entry_date": position.entry_date,
                    "exit_date": position.exit_date,
                }
            )

        # Worst performers
        worst_performers = []
        for position in sorted_positions[-10:]:
            worst_performers.append(
                {
                    "symbol": position.symbol,
                    "pnl": position.pnl,
                    "pnl_pct": position.pnl_pct,
                    "entry_date": position.entry_date,
                    "exit_date": position.exit_date,
                }
            )

        return top_performers, worst_performers

    def _calculate_risk_metrics(
        self, returns: list[float]
    ) -> tuple[float | None, float | None, float | None]:
        """Calculate risk metrics (VaR, CVaR)."""
        if not returns:
            return None, None, None

        returns_array = np.array(returns)

        # VaR at 95% and 99%
        var_95 = np.percentile(returns_array, 5)
        var_99 = np.percentile(returns_array, 1)

        # CVaR at 95% (average of worst 5% returns)
        worst_5_pct = np.percentile(returns_array, 5)
        worst_returns = returns_array[returns_array <= worst_5_pct]
        cvar_95 = np.mean(worst_returns) if len(worst_returns) > 0 else None

        return var_95, var_99, cvar_95

    def _generate_insights(
        self, metrics: PerformanceMetrics, drawdowns: list[DrawdownAnalysis]
    ) -> tuple[list[str], list[str], list[str]]:
        """Generate insights from analysis."""
        strengths = []
        weaknesses = []
        recommendations = []

        # Analyze Sharpe ratio
        if metrics.sharpe_ratio > 1.5:
            strengths.append("Excellent risk-adjusted returns (Sharpe > 1.5)")
        elif metrics.sharpe_ratio < 0.5:
            weaknesses.append("Poor risk-adjusted returns (Sharpe < 0.5)")
            recommendations.append("Consider reducing volatility or improving return generation")

        # Analyze drawdowns
        if metrics.max_drawdown > -0.3:
            weaknesses.append("Large maximum drawdown (> 30%)")
            recommendations.append("Implement tighter risk controls and position sizing")

        if len(drawdowns) > 5:
            weaknesses.append("Frequent drawdowns indicate high volatility")
            recommendations.append("Consider adding trend-following or volatility filters")

        # Analyze win rate
        if metrics.win_rate > 0.6:
            strengths.append("High win rate (> 60%)")
        elif metrics.win_rate < 0.4:
            weaknesses.append("Low win rate (< 40%)")
            recommendations.append("Review signal generation and entry/exit criteria")

        # Analyze profit factor
        if metrics.profit_factor > 2.0:
            strengths.append("Excellent profit factor (> 2.0)")
        elif metrics.profit_factor < 1.0:
            weaknesses.append("Profit factor < 1.0 (losing strategy)")
            recommendations.append("Strategy needs fundamental re-evaluation")

        # Analyze turnover
        if metrics.turnover > 0.5:
            weaknesses.append("High portfolio turnover (> 50%)")
            recommendations.append(
                "Consider reducing rebalancing frequency to lower transaction costs"
            )

        return strengths, weaknesses, recommendations


# Global results analyzer instance
results_analyzer = ResultsAnalyzer()
