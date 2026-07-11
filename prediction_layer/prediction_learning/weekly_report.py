"""
Weekly Report Generator

Generates comprehensive weekly performance reports for research.
Includes top/worst signals, features, sectors, and regime performance.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np

from prediction_layer.prediction_learning.prediction_history import PredictionMetadata
from prediction_layer.prediction_learning.prediction_result import PredictionResult, PredictionQuality

from utils.logger import get_logger

logger = get_logger("prediction_layer.prediction_learning.weekly_report")


@dataclass
class WeeklyPerformanceMetrics:
    """Performance metrics for a week."""
    week_start: datetime
    week_end: datetime
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    win_rate: float
    average_return: float
    average_holding_time: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "week_start": self.week_start.isoformat(),
            "week_end": self.week_end.isoformat(),
            "total_predictions": self.total_predictions,
            "successful_predictions": self.successful_predictions,
            "failed_predictions": self.failed_predictions,
            "win_rate": round(self.win_rate, 4),
            "average_return": round(self.average_return, 4),
            "average_holding_time": round(self.average_holding_time, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "profit_factor": round(self.profit_factor, 4),
        }


@dataclass
class WeeklyReport:
    """Comprehensive weekly report."""
    metrics: WeeklyPerformanceMetrics
    top_performing_signals: List[Dict[str, any]]
    worst_performing_signals: List[Dict[str, any]]
    top_performing_features: List[Dict[str, any]]
    worst_performing_features: List[Dict[str, any]]
    best_sectors: List[Dict[str, any]]
    worst_sectors: List[Dict[str, any]]
    bull_performance: Dict[str, float]
    bear_performance: Dict[str, float]
    sideways_performance: Dict[str, float]
    prediction_accuracy_by_confidence: Dict[str, float]
    key_insights: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "metrics": self.metrics.to_dict(),
            "top_performing_signals": self.top_performing_signals,
            "worst_performing_signals": self.worst_performing_signals,
            "top_performing_features": self.top_performing_features,
            "worst_performing_features": self.worst_performing_features,
            "best_sectors": self.best_sectors,
            "worst_sectors": self.worst_sectors,
            "bull_performance": self.bull_performance,
            "bear_performance": self.bear_performance,
            "sideways_performance": self.sideways_performance,
            "prediction_accuracy_by_confidence": self.prediction_accuracy_by_confidence,
            "key_insights": self.key_insights,
            "recommendations": self.recommendations,
        }


class WeeklyReportGenerator:
    """
    Generates weekly performance reports.
    
    Reports include:
    - Overall performance metrics
    - Top/worst performing signals
    - Top/worst performing features
    - Best/worst sectors
    - Performance by regime
    - Accuracy by confidence level
    - Key insights and recommendations
    """
    
    def __init__(self):
        """Initialize weekly report generator."""
        self._logger = get_logger("prediction_layer.prediction_learning.weekly_report")
    
    def generate_report(
        self,
        predictions: List[PredictionMetadata],
        results: List[PredictionResult],
        week_start: Optional[datetime] = None,
    ) -> WeeklyReport:
        """
        Generate weekly performance report.
        
        Args:
            predictions: List of predictions for the week
            results: List of results for the week
            week_start: Optional week start date (defaults to 7 days ago)
            
        Returns:
            WeeklyReport
        """
        # Determine week range
        if week_start is None:
            week_start = datetime.now() - timedelta(days=7)
        week_end = week_start + timedelta(days=7)
        
        # Filter predictions by week
        weekly_predictions = [
            p for p in predictions
            if week_start <= p.prediction_timestamp < week_end
        ]
        
        # Create prediction ID to result mapping
        result_map = {r.prediction_id: r for r in results}
        
        # Calculate overall metrics
        metrics = self._calculate_metrics(weekly_predictions, result_map)
        
        # Analyze signal performance
        top_signals, worst_signals = self._analyze_signal_performance(
            weekly_predictions,
            result_map,
        )
        
        # Analyze feature performance
        top_features, worst_features = self._analyze_feature_performance(
            weekly_predictions,
            result_map,
        )
        
        # Analyze sector performance
        best_sectors, worst_sectors = self._analyze_sector_performance(
            weekly_predictions,
            result_map,
        )
        
        # Analyze regime performance
        bull_perf, bear_perf, sideways_perf = self._analyze_regime_performance(
            weekly_predictions,
            result_map,
        )
        
        # Analyze accuracy by confidence
        accuracy_by_confidence = self._analyze_accuracy_by_confidence(
            weekly_predictions,
            result_map,
        )
        
        # Generate insights and recommendations
        key_insights, recommendations = self._generate_insights_and_recommendations(
            metrics,
            top_signals,
            worst_signals,
            top_features,
            worst_features,
        )
        
        self._logger.info(f"Generated weekly report for {week_start.date()} to {week_end.date()}")
        
        return WeeklyReport(
            metrics=metrics,
            top_performing_signals=top_signals,
            worst_performing_signals=worst_signals,
            top_performing_features=top_features,
            worst_performing_features=worst_features,
            best_sectors=best_sectors,
            worst_sectors=worst_sectors,
            bull_performance=bull_perf,
            bear_performance=bear_perf,
            sideways_performance=sideways_perf,
            prediction_accuracy_by_confidence=accuracy_by_confidence,
            key_insights=key_insights,
            recommendations=recommendations,
        )
    
    def _calculate_metrics(
        self,
        predictions: List[PredictionMetadata],
        result_map: Dict[str, PredictionResult],
    ) -> WeeklyPerformanceMetrics:
        """
        Calculate overall performance metrics.
        
        Args:
            predictions: List of predictions
            result_map: Mapping of prediction ID to result
            
        Returns:
            WeeklyPerformanceMetrics
        """
        if not predictions:
            return WeeklyPerformanceMetrics(
                week_start=datetime.now(),
                week_end=datetime.now(),
                total_predictions=0,
                successful_predictions=0,
                failed_predictions=0,
                win_rate=0.0,
                average_return=0.0,
                average_holding_time=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                profit_factor=0.0,
            )
        
        total = len(predictions)
        successful = 0
        failed = 0
        returns = []
        holding_times = []
        
        for pred in predictions:
            result = result_map.get(pred.prediction_id)
            if result:
                if result.actual_return_percentage > 0:
                    successful += 1
                else:
                    failed += 1
                returns.append(result.actual_return_percentage)
                holding_times.append(result.holding_time_hours)
        
        win_rate = successful / total if total > 0 else 0.0
        average_return = np.mean(returns) if returns else 0.0
        average_holding_time = np.mean(holding_times) if holding_times else 0.0
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = (
            average_return / np.std(returns) if returns and np.std(returns) > 0 else 0.0
        )
        
        # Calculate max drawdown
        max_drawdown = min(returns) if returns else 0.0
        
        # Calculate profit factor
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        return WeeklyPerformanceMetrics(
            week_start=predictions[0].prediction_timestamp if predictions else datetime.now(),
            week_end=predictions[-1].prediction_timestamp if predictions else datetime.now(),
            total_predictions=total,
            successful_predictions=successful,
            failed_predictions=failed,
            win_rate=win_rate,
            average_return=average_return,
            average_holding_time=average_holding_time,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
        )
    
    def _analyze_signal_performance(
        self,
        predictions: List[PredictionMetadata],
        result_map: Dict[str, PredictionResult],
    ) -> tuple:
        """
        Analyze signal performance.
        
        Args:
            predictions: List of predictions
            result_map: Mapping of prediction ID to result
            
        Returns:
            Tuple of (top_signals, worst_signals)
        """
        signal_performance = {}
        
        for pred in predictions:
            result = result_map.get(pred.prediction_id)
            if not result:
                continue
            
            for signal in pred.signals:
                signal_name = signal.get("name", "unknown")
                if signal_name not in signal_performance:
                    signal_performance[signal_name] = {
                        "count": 0,
                        "total_return": 0.0,
                        "wins": 0,
                    }
                
                signal_performance[signal_name]["count"] += 1
                signal_performance[signal_name]["total_return"] += result.actual_return_percentage
                if result.actual_return_percentage > 0:
                    signal_performance[signal_name]["wins"] += 1
        
        # Calculate average returns and win rates
        for signal_name, data in signal_performance.items():
            data["average_return"] = data["total_return"] / data["count"]
            data["win_rate"] = data["wins"] / data["count"]
        
        # Sort by average return
        sorted_signals = sorted(
            signal_performance.items(),
            key=lambda x: x[1]["average_return"],
            reverse=True,
        )
        
        # Get top and worst
        top_signals = [
            {"name": name, **data}
            for name, data in sorted_signals[:5]
        ]
        worst_signals = [
            {"name": name, **data}
            for name, data in sorted_signals[-5:]
        ]
        
        return top_signals, worst_signals
    
    def _analyze_feature_performance(
        self,
        predictions: List[PredictionMetadata],
        result_map: Dict[str, PredictionResult],
    ) -> tuple:
        """
        Analyze feature performance.
        
        Args:
            predictions: List of predictions
            result_map: Mapping of prediction ID to result
            
        Returns:
            Tuple of (top_features, worst_features)
        """
        feature_performance = {}
        
        for pred in predictions:
            result = result_map.get(pred.prediction_id)
            if not result:
                continue
            
            for feature_name, feature_value in pred.features.items():
                if feature_name not in feature_performance:
                    feature_performance[feature_name] = {
                        "count": 0,
                        "total_return": 0.0,
                        "wins": 0,
                    }
                
                feature_performance[feature_name]["count"] += 1
                feature_performance[feature_name]["total_return"] += result.actual_return_percentage
                if result.actual_return_percentage > 0:
                    feature_performance[feature_name]["wins"] += 1
        
        # Calculate average returns and win rates
        for feature_name, data in feature_performance.items():
            data["average_return"] = data["total_return"] / data["count"]
            data["win_rate"] = data["wins"] / data["count"]
        
        # Sort by average return
        sorted_features = sorted(
            feature_performance.items(),
            key=lambda x: x[1]["average_return"],
            reverse=True,
        )
        
        # Get top and worst
        top_features = [
            {"name": name, **data}
            for name, data in sorted_features[:5]
        ]
        worst_features = [
            {"name": name, **data}
            for name, data in sorted_features[-5:]
        ]
        
        return top_features, worst_features
    
    def _analyze_sector_performance(
        self,
        predictions: List[PredictionMetadata],
        result_map: Dict[str, PredictionResult],
    ) -> tuple:
        """
        Analyze sector performance.
        
        Args:
            predictions: List of predictions
            result_map: Mapping of prediction ID to result
            
        Returns:
            Tuple of (best_sectors, worst_sectors)
        """
        sector_performance = {}
        
        for pred in predictions:
            result = result_map.get(pred.prediction_id)
            if not result:
                continue
            
            # Extract sector from symbol (simplified)
            sector = pred.symbol.split("_")[0] if "_" in pred.symbol else "unknown"
            
            if sector not in sector_performance:
                sector_performance[sector] = {
                    "count": 0,
                    "total_return": 0.0,
                    "wins": 0,
                }
            
            sector_performance[sector]["count"] += 1
            sector_performance[sector]["total_return"] += result.actual_return_percentage
            if result.actual_return_percentage > 0:
                sector_performance[sector]["wins"] += 1
        
        # Calculate average returns and win rates
        for sector, data in sector_performance.items():
            data["average_return"] = data["total_return"] / data["count"]
            data["win_rate"] = data["wins"] / data["count"]
        
        # Sort by average return
        sorted_sectors = sorted(
            sector_performance.items(),
            key=lambda x: x[1]["average_return"],
            reverse=True,
        )
        
        # Get best and worst
        best_sectors = [
            {"name": name, **data}
            for name, data in sorted_sectors[:5]
        ]
        worst_sectors = [
            {"name": name, **data}
            for name, data in sorted_sectors[-5:]
        ]
        
        return best_sectors, worst_sectors
    
    def _analyze_regime_performance(
        self,
        predictions: List[PredictionMetadata],
        result_map: Dict[str, PredictionResult],
    ) -> tuple:
        """
        Analyze performance by market regime.
        
        Args:
            predictions: List of predictions
            result_map: Mapping of prediction ID to result
            
        Returns:
            Tuple of (bull_performance, bear_performance, sideways_performance)
        """
        regime_performance = {
            "bull": {"wins": 0, "total": 0, "total_return": 0.0},
            "bear": {"wins": 0, "total": 0, "total_return": 0.0},
            "sideways": {"wins": 0, "total": 0, "total_return": 0.0},
        }
        
        for pred in predictions:
            result = result_map.get(pred.prediction_id)
            if not result:
                continue
            
            regime = pred.market_regime.lower()
            if regime not in regime_performance:
                regime = "sideways"
            
            regime_performance[regime]["total"] += 1
            regime_performance[regime]["total_return"] += result.actual_return_percentage
            if result.actual_return_percentage > 0:
                regime_performance[regime]["wins"] += 1
        
        # Calculate metrics for each regime
        def calculate_regime_metrics(data):
            total = data["total"]
            if total == 0:
                return {"win_rate": 0.0, "average_return": 0.0, "total_predictions": 0}
            return {
                "win_rate": data["wins"] / total,
                "average_return": data["total_return"] / total,
                "total_predictions": total,
            }
        
        bull_performance = calculate_regime_metrics(regime_performance["bull"])
        bear_performance = calculate_regime_metrics(regime_performance["bear"])
        sideways_performance = calculate_regime_metrics(regime_performance["sideways"])
        
        return bull_performance, bear_performance, sideways_performance
    
    def _analyze_accuracy_by_confidence(
        self,
        predictions: List[PredictionMetadata],
        result_map: Dict[str, PredictionResult],
    ) -> Dict[str, float]:
        """
        Analyze accuracy by confidence level.
        
        Args:
            predictions: List of predictions
            result_map: Mapping of prediction ID to result
            
        Returns:
            Dictionary mapping confidence level to accuracy
        """
        confidence_performance = {
            "HIGH": {"wins": 0, "total": 0},
            "MEDIUM": {"wins": 0, "total": 0},
            "LOW": {"wins": 0, "total": 0},
        }
        
        for pred in predictions:
            result = result_map.get(pred.prediction_id)
            if not result:
                continue
            
            confidence = pred.confidence
            if confidence not in confidence_performance:
                confidence = "MEDIUM"
            
            confidence_performance[confidence]["total"] += 1
            if result.actual_return_percentage > 0:
                confidence_performance[confidence]["wins"] += 1
        
        # Calculate accuracy for each confidence level
        accuracy_by_confidence = {}
        for confidence, data in confidence_performance.items():
            total = data["total"]
            accuracy = data["wins"] / total if total > 0 else 0.0
            accuracy_by_confidence[confidence] = accuracy
        
        return accuracy_by_confidence
    
    def _generate_insights_and_recommendations(
        self,
        metrics: WeeklyPerformanceMetrics,
        top_signals: List[Dict],
        worst_signals: List[Dict],
        top_features: List[Dict],
        worst_features: List[Dict],
    ) -> tuple:
        """
        Generate key insights and recommendations.
        
        Args:
            metrics: Performance metrics
            top_signals: Top performing signals
            worst_signals: Worst performing signals
            top_features: Top performing features
            worst_features: Worst performing features
            
        Returns:
            Tuple of (insights, recommendations)
        """
        insights = []
        recommendations = []
        
        # Win rate insight
        if metrics.win_rate > 0.6:
            insights.append(f"Strong win rate of {metrics.win_rate:.1%}")
        elif metrics.win_rate < 0.4:
            insights.append(f"Weak win rate of {metrics.win_rate:.1%} - needs attention")
        
        # Average return insight
        if metrics.average_return > 2.0:
            insights.append(f"Healthy average return of {metrics.average_return:.2f}%")
        elif metrics.average_return < 0:
            insights.append(f"Negative average return of {metrics.average_return:.2f}%")
        
        # Top signal insight
        if top_signals:
            best_signal = top_signals[0]
            insights.append(
                f"Best signal: {best_signal['name']} "
                f"({best_signal['win_rate']:.1%} win rate, {best_signal['average_return']:.2f}% avg return)"
            )
        
        # Worst signal insight
        if worst_signals:
            worst_signal = worst_signals[0]
            insights.append(
                f"Worst signal: {worst_signal['name']} "
                f"({worst_signal['win_rate']:.1%} win rate, {worst_signal['average_return']:.2f}% avg return)"
            )
        
        # Recommendations
        if metrics.win_rate < 0.5:
            recommendations.append("Review signal selection criteria")
            recommendations.append("Consider increasing confidence threshold")
        
        if worst_signals and worst_signals[0]["win_rate"] < 0.3:
            recommendations.append(f"Disable or reduce weight of {worst_signals[0]['name']} signal")
        
        if worst_features and worst_features[0]["win_rate"] < 0.3:
            recommendations.append(f"Review {worst_features[0]['name']} feature calculation")
        
        if metrics.profit_factor < 1.0:
            recommendations.append("Improve risk management - losses exceeding gains")
        
        return insights, recommendations


def generate_weekly_report(
    predictions: List[PredictionMetadata],
    results: List[PredictionResult],
    week_start: Optional[datetime] = None,
) -> WeeklyReport:
    """
    Convenience function to generate weekly report.
    
    Args:
        predictions: List of predictions for the week
        results: List of results for the week
        week_start: Optional week start date
        
    Returns:
        WeeklyReport
    """
    generator = WeeklyReportGenerator()
    return generator.generate_report(predictions, results, week_start)
