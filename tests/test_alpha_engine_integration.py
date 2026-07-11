"""
Integration Tests for Alpha Engine

Tests the complete alpha engine workflow:
1. Signal collection and normalization
2. Weight calculation and regime adjustment
3. Filter application
4. Alpha score calculation
5. Ranking and selection
6. Explanation generation
7. Performance tracking
8. Report generation
"""

from datetime import date, datetime
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from alpha_engine.alpha_builder import AlphaBuilder, AlphaGrade, TimeFrame
from alpha_engine.alpha_weights import AlphaWeights
from alpha_engine.alpha_regime import AlphaRegimeAdjuster
from alpha_engine.alpha_filters import AlphaFilters, FilterType
from alpha_engine.alpha_score import AlphaScoreCalculator
from alpha_engine.alpha_ranker import AlphaRanker
from alpha_engine.alpha_explainer import AlphaExplainer
from alpha_engine.alpha_tracker import AlphaTracker, PredictionOutcome
from alpha_engine.alpha_reports import AlphaReports
from signal_engine.base import Signal, SignalCategory, SignalDirection, SignalSet


@pytest.fixture
def sample_signal_set() -> SignalSet:
    """Create a sample signal set for testing."""
    signal_set = SignalSet(
        symbol="INFY",
        timestamp=datetime.now(),
    )
    
    # Add sample signals
    signal_set.add_signal(Signal(
        name="Technical",
        category=SignalCategory.TECHNICAL,
        score=92.0,
        direction=SignalDirection.BULLISH,
        confidence=85.0,
        reason="Strong trend alignment",
        raw_values={"ema_short": 1450, "ema_long": 1400, "rsi": 65},
    ))
    
    signal_set.add_signal(Signal(
        name="Volume",
        category=SignalCategory.VOLUME,
        score=88.0,
        direction=SignalDirection.BULLISH,
        confidence=80.0,
        reason="Volume spike detected",
        raw_values={"avg_volume": 50000000, "current_volume": 80000000},
    ))
    
    signal_set.add_signal(Signal(
        name="Options",
        category=SignalCategory.OPTIONS,
        score=84.0,
        direction=SignalDirection.BULLISH,
        confidence=75.0,
        reason="Bullish options flow",
        raw_values={"call_put_ratio": 1.5},
    ))
    
    signal_set.add_signal(Signal(
        name="Fundamental",
        category=SignalCategory.FUNDAMENTAL,
        score=90.0,
        direction=SignalDirection.BULLISH,
        confidence=85.0,
        reason="Strong fundamentals",
        raw_values={"pe_ratio": 22, "roe": 18},
    ))
    
    signal_set.add_signal(Signal(
        name="Sentiment",
        category=SignalCategory.SENTIMENT,
        score=76.0,
        direction=SignalDirection.BULLISH,
        confidence=70.0,
        reason="Positive sentiment",
        raw_values={"sentiment_score": 0.6},
    ))
    
    return signal_set


@pytest.fixture
def sample_market_data() -> Dict:
    """Create sample market data for testing."""
    return {
        "adv": 50000000,  # 50M INR
        "delivery_pct": 35.0,
        "spread_bps": 3.0,
        "volatility": 0.08,
        "atr_pct": 0.02,
        "expected_return": 0.08,
        "news_sentiment": 0.5,
        "has_major_negative_news": False,
        "has_corporate_action": False,
        "near_circuit": False,
        "circuit_hit": False,
    }


@pytest.fixture
def sector_map() -> Dict[str, str]:
    """Create sample sector map."""
    return {
        "INFY": "IT",
        "TCS": "IT",
        "RELIANCE": "ENERGY",
        "HDFCBANK": "FINANCE",
        "BEL": "DEFENSE",
    }


class TestAlphaEngineIntegration:
    """Integration tests for the complete alpha engine workflow."""
    
    def test_complete_workflow(
        self,
        sample_signal_set,
        sample_market_data,
        sector_map,
    ):
        """Test the complete alpha engine workflow."""
        # Step 1: Build alpha input
        builder = AlphaBuilder()
        alpha_input = builder.build_alpha_input(sample_signal_set, sample_market_data)
        
        assert alpha_input["symbol"] == "INFY"
        assert "category_scores" in alpha_input
        assert len(alpha_input["category_scores"]) > 0
        
        # Step 2: Get weights
        weights = AlphaWeights()
        base_weights = weights.get_weights(TimeFrame.SWING)
        
        assert sum(base_weights.values()) == pytest.approx(1.0, rel=0.01)
        
        # Step 3: Apply regime adjustments
        regime_adjuster = AlphaRegimeAdjuster()
        adjusted_weights = regime_adjuster.get_adjusted_weights(base_weights)
        
        assert sum(adjusted_weights.values()) == pytest.approx(1.0, rel=0.01)
        
        # Step 4: Apply filters
        filters = AlphaFilters()
        passed, filter_results = filters.apply_filters(
            symbol="INFY",
            alpha_score=85.0,
            market_data=sample_market_data,
        )
        
        assert passed is True  # Should pass with good market data
        
        # Step 5: Calculate alpha score
        score_calculator = AlphaScoreCalculator()
        alpha_result = score_calculator.calculate_alpha_score(
            symbol="INFY",
            category_scores=alpha_input["category_scores"],
            weights=adjusted_weights,
            filter_results=filter_results,
        )
        
        assert alpha_result.symbol == "INFY"
        assert 0 <= alpha_result.final_alpha_score <= 100
        assert alpha_result.passed_filters is True
        
        # Step 6: Generate explanation
        explainer = AlphaExplainer()
        explanation = explainer.explain_alpha(alpha_result, sample_market_data)
        
        assert explanation.symbol == "INFY"
        assert explanation.summary is not None
        assert len(explanation.category_explanations) > 0
        
    def test_batch_workflow(
        self,
        sample_signal_set,
        sample_market_data,
        sector_map,
    ):
        """Test batch processing of multiple stocks."""
        # Create multiple signal sets
        signal_sets = {}
        for symbol in ["INFY", "TCS", "RELIANCE", "HDFCBANK", "BEL"]:
            # Clone the sample signal set with different scores
            signal_set = SignalSet(
                symbol=symbol,
                timestamp=datetime.now(),
            )
            
            # Vary scores slightly for each stock
            score_modifier = {"INFY": 0, "TCS": -5, "RELIANCE": -10, "HDFCBANK": -15, "BEL": -20}
            
            for category, signal in sample_signal_set.signals.items():
                new_signal = Signal(
                    name=signal.name,
                    category=signal.category,
                    score=max(0, min(100, signal.score + score_modifier[symbol])),
                    direction=signal.direction,
                    confidence=signal.confidence,
                    reason=signal.reason,
                    raw_values=signal.raw_values,
                )
                signal_set.add_signal(new_signal)
            
            signal_sets[symbol] = signal_set
        
        # Build batch alpha inputs
        builder = AlphaBuilder()
        alpha_inputs = builder.build_batch_alpha_input(signal_sets, sample_market_data)
        
        assert len(alpha_inputs) == 5
        
        # Get weights
        weights = AlphaWeights()
        base_weights = weights.get_weights(TimeFrame.SWING)
        
        # Apply regime adjustments
        regime_adjuster = AlphaRegimeAdjuster()
        adjusted_weights = regime_adjuster.get_adjusted_weights(base_weights)
        
        # Calculate batch alpha scores
        score_calculator = AlphaScoreCalculator()
        
        # Apply filters for each stock
        filters = AlphaFilters()
        filter_results_map = {}
        for symbol in alpha_inputs.keys():
            passed, filter_results = filters.apply_filters(
                symbol=symbol,
                alpha_score=75.0,
                market_data=sample_market_data,
            )
            filter_results_map[symbol] = filter_results
        
        alpha_results = score_calculator.calculate_batch_alpha_scores(
            alpha_inputs=alpha_inputs,
            weights=adjusted_weights,
            filter_results_map=filter_results_map,
        )
        
        assert len(alpha_results) == 5
        
        # Rank stocks
        ranker = AlphaRanker()
        ranker.update_config(min_grade=AlphaGrade.AVERAGE)
        ranking_result = ranker.rank_stocks(
            alpha_results=alpha_results,
            sector_map=sector_map,
        )
        
        assert ranking_result.total_evaluated == 5
        assert ranking_result.top_n_selected > 0
        assert len(ranking_result.ranked_stocks) > 0
        
        # Get top symbols
        top_symbols = ranker.get_top_symbols(ranking_result)
        assert len(top_symbols) > 0
        
    def test_performance_tracking(
        self,
        sample_signal_set,
        sample_market_data,
    ):
        """Test performance tracking functionality."""
        tracker = AlphaTracker()
        
        # Record a prediction
        record = tracker.record_prediction(
            date=date.today(),
            symbol="INFY",
            alpha_score=88.5,
            grade=AlphaGrade.EXCELLENT,
            predicted_direction="long",
            entry_price=1450.0,
            target_price=1500.0,
            stop_loss=1400.0,
            regime="BULL",
            category_scores={"technical": 92, "volume": 88, "options": 84},
            signals_used=["Technical", "Volume", "Options"],
            features_used=["ema", "rsi", "volume"],
        )
        
        assert record.symbol == "INFY"
        assert record.outcome == PredictionOutcome.PENDING
        
        # Update outcome
        updated = tracker.update_outcome(
            symbol="INFY",
            date=date.today(),
            actual_return=0.05,
            outcome=PredictionOutcome.CORRECT,
        )
        
        assert updated is not None
        assert updated.actual_return == 0.05
        assert updated.outcome == PredictionOutcome.CORRECT
        
        # Get performance summary
        summary = tracker.get_performance_summary()
        
        assert summary["total_predictions"] == 1
        assert summary["completed_predictions"] == 1
        assert summary["overall_win_rate"] == 1.0
        
    def test_weekly_report_generation(
        self,
        sample_signal_set,
        sample_market_data,
    ):
        """Test weekly report generation."""
        tracker = AlphaTracker()
        
        # Add some historical records
        for i in range(10):
            tracker.record_prediction(
                date=date.today() - pd.Timedelta(days=i),
                symbol=f"STOCK{i}",
                alpha_score=75.0 + i,
                grade=AlphaGrade.GOOD if i < 5 else AlphaGrade.EXCELLENT,
                predicted_direction="long",
                entry_price=1000.0,
                target_price=1050.0,
                stop_loss=950.0,
                regime="BULL",
                category_scores={"technical": 75 + i, "volume": 75 + i},
                signals_used=["Technical", "Volume"],
                features_used=["ema", "volume"],
            )
            
            # Update outcomes for some
            if i < 5:
                tracker.update_outcome(
                    symbol=f"STOCK{i}",
                    date=date.today() - pd.Timedelta(days=i),
                    actual_return=0.03 if i % 2 == 0 else -0.02,
                    outcome=PredictionOutcome.CORRECT if i % 2 == 0 else PredictionOutcome.INCORRECT,
                )
        
        # Generate report
        reports = AlphaReports(tracker)
        report = reports.generate_weekly_report()
        
        assert report.total_predictions > 0
        assert len(report.insights) > 0
        assert len(report.recommendations) > 0
        
        # Format as markdown
        markdown = reports.format_report_markdown(report)
        assert markdown is not None
        assert len(markdown) > 0
        
    def test_filter_configurations(
        self,
        sample_market_data,
    ):
        """Test filter configuration updates."""
        filters = AlphaFilters()
        
        # Get default config
        summary = filters.get_filter_summary()
        assert FilterType.LIQUIDITY.value in summary
        
        # Update filter config
        filters.update_filter_config(
            filter_type=FilterType.LIQUIDITY,
            enabled=False,
        )
        
        # Verify update
        updated_summary = filters.get_filter_summary()
        assert updated_summary[FilterType.LIQUIDITY.value]["enabled"] is False
        
    def test_weight_performance_adjustment(self):
        """Test performance-based weight adjustment."""
        weights = AlphaWeights()
        
        # Update performance for categories
        weights.update_category_performance(
            category="technical",
            win_rate=0.75,
            average_return=0.08,
            sharpe_ratio=1.5,
            sample_size=100,
        )
        
        weights.update_category_performance(
            category="volume",
            win_rate=0.60,
            average_return=0.04,
            sharpe_ratio=0.8,
            sample_size=100,
        )
        
        # Get weights with performance adjustment
        adjusted = weights.get_weights(
            timeframe=TimeFrame.SWING,
            use_performance_adjustment=True,
        )
        
        # Technical should have higher weight due to better performance
        assert adjusted["technical"] > adjusted["volume"]
        
        # Get performance summary
        summary = weights.get_performance_summary()
        assert summary["best_category"] == "technical"
        
    def test_explanation_formatting(
        self,
        sample_signal_set,
        sample_market_data,
    ):
        """Test explanation formatting for display."""
        builder = AlphaBuilder()
        alpha_input = builder.build_alpha_input(sample_signal_set, sample_market_data)
        
        weights = AlphaWeights()
        base_weights = weights.get_weights(TimeFrame.SWING)
        
        filters = AlphaFilters()
        passed, filter_results = filters.apply_filters(
            symbol="INFY",
            alpha_score=85.0,
            market_data=sample_market_data,
        )
        
        score_calculator = AlphaScoreCalculator()
        alpha_result = score_calculator.calculate_alpha_score(
            symbol="INFY",
            category_scores=alpha_input["category_scores"],
            weights=base_weights,
            filter_results=filter_results,
        )
        
        explainer = AlphaExplainer()
        explanation = explainer.explain_alpha(alpha_result, sample_market_data)
        
        # Format for display
        formatted = explainer.format_explanation_for_display(explanation)
        
        assert formatted is not None
        assert "INFY" in formatted
        assert "ALPHA SCORE" in formatted
        assert "CATEGORY BREAKDOWN" in formatted
        assert "RECOMMENDATION" in formatted
        
    def test_grade_thresholds(self):
        """Test grade threshold updates."""
        calculator = AlphaScoreCalculator()
        
        # Test default thresholds
        result = calculator._assign_grade(95.0)
        assert result == AlphaGrade.INSTITUTIONAL
        
        result = calculator._assign_grade(85.0)
        assert result == AlphaGrade.EXCELLENT
        
        result = calculator._assign_grade(75.0)
        assert result == AlphaGrade.GOOD
        
        result = calculator._assign_grade(60.0)
        assert result == AlphaGrade.AVERAGE
        
        result = calculator._assign_grade(50.0)
        assert result == AlphaGrade.REJECT
        
        # Update threshold
        calculator.update_grade_threshold(AlphaGrade.EXCELLENT, 90.0)
        
        result = calculator._assign_grade(88.0)
        assert result == AlphaGrade.GOOD  # Should be GOOD now with higher threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
