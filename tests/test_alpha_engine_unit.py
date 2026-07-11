"""
Unit Tests for Alpha Engine

Tests individual alpha engine modules without external dependencies.
"""

from datetime import date, datetime
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from alpha_engine.alpha_builder import (
    AlphaBuilder,
    AlphaCategory,
    AlphaGrade,
    AlphaResult,
    TimeFrame,
)
from alpha_engine.alpha_weights import AlphaWeights, WeightConfiguration
from alpha_engine.alpha_filters import (
    AlphaFilters,
    FilterConfig,
    FilterResult,
    FilterType,
)
from alpha_engine.alpha_score import AlphaScoreCalculator
from alpha_engine.alpha_ranker import AlphaRanker, RankingConfig
from alpha_engine.alpha_explainer import AlphaExplainer
from alpha_engine.alpha_tracker import (
    AlphaRecord,
    AlphaTracker,
    GradePerformance,
    PredictionOutcome,
)
from alpha_engine.alpha_reports import AlphaReports


@pytest.fixture
def sample_category_scores() -> Dict[str, float]:
    """Create sample category scores."""
    return {
        "technical": 92.0,
        "volume": 88.0,
        "options": 84.0,
        "fundamental": 90.0,
        "sentiment": 76.0,
        "macro": 72.0,
        "sector": 85.0,
    }


@pytest.fixture
def sample_market_data() -> Dict:
    """Create sample market data."""
    return {
        "adv": 50000000,
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


class TestAlphaBuilder:
    """Tests for AlphaBuilder."""
    
    def test_normalize_signals(self):
        """Test signal normalization."""
        builder = AlphaBuilder()
        
        # Create mock normalized signals
        normalized = {
            "technical": {"Technical": 92.0},
            "volume": {"Volume": 88.0},
            "options": {"Options": 84.0},
        }
        
        category_scores = builder._create_category_scores(normalized)
        
        assert "technical" in category_scores
        assert category_scores["technical"] == 92.0
        assert category_scores["volume"] == 88.0
    
    def test_category_score_averaging(self):
        """Test category score averaging."""
        builder = AlphaBuilder()
        
        normalized = {
            "technical": {"EMA": 90.0, "RSI": 85.0, "MACD": 88.0},
        }
        
        category_scores = builder._create_category_scores(normalized)
        
        assert category_scores["technical"] == pytest.approx(87.67, rel=0.01)


class TestAlphaWeights:
    """Tests for AlphaWeights."""
    
    def test_base_weights_validation(self):
        """Test that base weights sum to 1.0."""
        weights = AlphaWeights()
        
        for timeframe in [TimeFrame.INTRADAY, TimeFrame.SWING, TimeFrame.LONGTERM]:
            config = weights.get_weight_configuration(timeframe)
            assert config.validate() is True
            assert sum(config.weights.values()) == pytest.approx(1.0, rel=0.01)
    
    def test_weight_normalization(self):
        """Test weight normalization."""
        weights = AlphaWeights()
        
        unnormalized = {
            "technical": 0.4,
            "volume": 0.3,
            "options": 0.2,
        }
        
        normalized = weights._normalize_weights(unnormalized)
        
        assert sum(normalized.values()) == pytest.approx(1.0, rel=0.01)
    
    def test_performance_adjustment(self):
        """Test performance-based weight adjustment."""
        weights = AlphaWeights()
        
        # Update performance
        weights.update_category_performance(
            category="technical",
            win_rate=0.80,
            average_return=0.10,
            sharpe_ratio=2.0,
            sample_size=100,
        )
        
        weights.update_category_performance(
            category="volume",
            win_rate=0.55,
            average_return=0.03,
            sharpe_ratio=0.5,
            sample_size=100,
        )
        
        # Get adjusted weights
        base_weights = weights.get_weights(TimeFrame.SWING, use_performance_adjustment=False)
        adjusted_weights = weights.get_weights(TimeFrame.SWING, use_performance_adjustment=True)
        
        # Technical should get higher weight due to better performance
        assert adjusted_weights["technical"] >= base_weights["technical"]


class TestAlphaFilters:
    """Tests for AlphaFilters."""
    
    def test_liquidity_filter_pass(self, sample_market_data):
        """Test liquidity filter with good data."""
        filters = AlphaFilters()
        
        result = filters._filter_liquidity(
            symbol="INFY",
            market_data=sample_market_data,
            params=filters.filter_configs[FilterType.LIQUIDITY].params,
        )
        
        assert result.passed is True
    
    def test_liquidity_filter_fail_low_adv(self):
        """Test liquidity filter with low ADV."""
        filters = AlphaFilters()
        
        market_data = {
            "adv": 500000,  # Below minimum
            "delivery_pct": 35.0,
            "spread_bps": 3.0,
        }
        
        result = filters._filter_liquidity(
            symbol="STOCK",
            market_data=market_data,
            params=filters.filter_configs[FilterType.LIQUIDITY].params,
        )
        
        assert result.passed is False
        assert "ADV too low" in result.reason
    
    def test_risk_reward_filter_pass(self, sample_market_data):
        """Test risk/reward filter with good ratio."""
        filters = AlphaFilters()
        
        result = filters._filter_risk_reward(
            symbol="INFY",
            alpha_score=85.0,
            market_data=sample_market_data,
            params=filters.filter_configs[FilterType.RISK_REWARD].params,
        )
        
        assert result.passed is True
    
    def test_risk_reward_filter_fail_poor_ratio(self):
        """Test risk/reward filter with poor ratio."""
        filters = AlphaFilters()
        
        market_data = {
            "expected_return": 0.02,  # Low return
            "atr_pct": 0.05,  # High risk
        }
        
        result = filters._filter_risk_reward(
            symbol="STOCK",
            alpha_score=70.0,
            market_data=market_data,
            params=filters.filter_configs[FilterType.RISK_REWARD].params,
        )
        
        assert result.passed is False
        assert "RR ratio too low" in result.reason
    
    def test_filter_config_update(self):
        """Test filter configuration update."""
        filters = AlphaFilters()
        
        filters.update_filter_config(
            filter_type=FilterType.LIQUIDITY,
            enabled=False,
        )
        
        assert filters.filter_configs[FilterType.LIQUIDITY].enabled is False


class TestAlphaScoreCalculator:
    """Tests for AlphaScoreCalculator."""
    
    def test_calculate_alpha_score(self, sample_category_scores):
        """Test alpha score calculation."""
        calculator = AlphaScoreCalculator()
        
        weights = {
            "technical": 0.30,
            "volume": 0.20,
            "options": 0.15,
            "fundamental": 0.15,
            "sentiment": 0.05,
            "macro": 0.05,
            "sector": 0.10,
        }
        
        # Calculate expected raw score
        expected_raw = sum(
            sample_category_scores[cat] * weight
            for cat, weight in weights.items()
        )
        
        result = calculator.calculate_alpha_score(
            symbol="INFY",
            category_scores=sample_category_scores,
            weights=weights,
            filter_results=[],
        )
        
        assert result.symbol == "INFY"
        assert result.raw_alpha_score == pytest.approx(expected_raw, rel=0.01)
        assert 0 <= result.final_alpha_score <= 100
    
    def test_grade_assignment(self):
        """Test grade assignment based on score."""
        calculator = AlphaScoreCalculator()
        
        assert calculator._assign_grade(96.0) == AlphaGrade.INSTITUTIONAL
        assert calculator._assign_grade(88.0) == AlphaGrade.EXCELLENT
        assert calculator._assign_grade(78.0) == AlphaGrade.GOOD
        assert calculator._assign_grade(65.0) == AlphaGrade.AVERAGE
        assert calculator._assign_grade(50.0) == AlphaGrade.REJECT
    
    def test_filter_penalty(self):
        """Test filter penalty calculation."""
        calculator = AlphaScoreCalculator()
        
        from alpha_engine.alpha_filters import FilterType
        
        # Create failed filters
        failed_filters = [
            FilterResult(
                filter_type=FilterType.LIQUIDITY,
                passed=False,
                reason="Low ADV",
            ),
            FilterResult(
                filter_type=FilterType.RISK_REWARD,
                passed=False,
                reason="Poor RR ratio",
            ),
        ]
        
        penalty = calculator._calculate_filter_penalty(failed_filters)
        
        # Liquidity (20) + Risk/Reward (15) = 35
        assert penalty == 35.0
    
    def test_grade_threshold_update(self):
        """Test grade threshold update."""
        calculator = AlphaScoreCalculator()
        
        calculator.update_grade_threshold(AlphaGrade.EXCELLENT, 90.0)
        
        assert calculator.grade_thresholds[AlphaGrade.EXCELLENT] == 90.0
        
        result = calculator._assign_grade(88.0)
        assert result == AlphaGrade.GOOD  # Should be GOOD now


class TestAlphaRanker:
    """Tests for AlphaRanker."""
    
    def test_rank_stocks(self, sample_category_scores):
        """Test stock ranking."""
        ranker = AlphaRanker()
        ranker.update_config(min_grade=AlphaGrade.GOOD)
        
        # Create mock alpha results
        alpha_results = {}
        for i, symbol in enumerate(["INFY", "TCS", "RELIANCE", "HDFCBANK", "BEL"]):
            # Vary scores
            scores = sample_category_scores.copy()
            scores = {k: v - i * 5 for k, v in scores.items()}
            
            result = AlphaResult(
                symbol=symbol,
                timestamp=datetime.now(),
                categories={},
                raw_alpha_score=sum(scores.values()) / len(scores),
                final_alpha_score=sum(scores.values()) / len(scores),
                grade=AlphaGrade.EXCELLENT if i < 2 else AlphaGrade.GOOD,
                passed_filters=True,
            )
            alpha_results[symbol] = result
        
        ranking = ranker.rank_stocks(
            alpha_results=alpha_results,
            sector_map={"INFY": "IT", "TCS": "IT", "RELIANCE": "ENERGY"},
        )
        
        assert ranking.total_evaluated == 5
        assert ranking.total_passed == 5
        assert ranking.top_n_selected > 0
        assert len(ranking.ranked_stocks) > 0
        
        # Check that stocks are sorted by score
        scores = [score for _, score, _ in ranking.ranked_stocks]
        assert scores == sorted(scores, reverse=True)
    
    def test_ranking_config_update(self):
        """Test ranking configuration update."""
        ranker = AlphaRanker()
        
        ranker.update_config(top_n=10, min_grade=AlphaGrade.GOOD)
        
        assert ranker.config.top_n == 10
        assert ranker.config.min_grade == AlphaGrade.GOOD


class TestAlphaExplainer:
    """Tests for AlphaExplainer."""
    
    def test_explain_alpha(self, sample_category_scores):
        """Test alpha explanation generation."""
        explainer = AlphaExplainer()
        
        # Create mock alpha result
        categories = {
            name: AlphaCategory(name=name, score=score, weight=0.15)
            for name, score in sample_category_scores.items()
        }
        
        result = AlphaResult(
            symbol="INFY",
            timestamp=datetime.now(),
            categories=categories,
            raw_alpha_score=85.0,
            final_alpha_score=85.0,
            grade=AlphaGrade.EXCELLENT,
            passed_filters=True,
        )
        
        explanation = explainer.explain_alpha(result)
        
        assert explanation.symbol == "INFY"
        assert explanation.alpha_score == 85.0
        assert explanation.grade == AlphaGrade.EXCELLENT
        assert explanation.summary is not None
        assert len(explanation.category_explanations) > 0
        assert len(explanation.strengths) > 0
        assert explanation.recommendation is not None
    
    def test_rating_generation(self):
        """Test star rating generation."""
        explainer = AlphaExplainer()
        
        assert explainer._generate_rating(95.0) == "★★★★★"
        assert explainer._generate_rating(85.0) == "★★★★☆"
        assert explainer._generate_rating(75.0) == "★★★☆☆"
        assert explainer._generate_rating(65.0) == "★★☆☆☆"
        assert explainer._generate_rating(50.0) == "★☆☆☆☆"
    
    def test_explanation_formatting(self, sample_category_scores):
        """Test explanation formatting for display."""
        explainer = AlphaExplainer()
        
        categories = {
            name: AlphaCategory(name=name, score=score, weight=0.15)
            for name, score in sample_category_scores.items()
        }
        
        result = AlphaResult(
            symbol="INFY",
            timestamp=datetime.now(),
            categories=categories,
            raw_alpha_score=85.0,
            final_alpha_score=85.0,
            grade=AlphaGrade.EXCELLENT,
            passed_filters=True,
        )
        
        explanation = explainer.explain_alpha(result)
        formatted = explainer.format_explanation_for_display(explanation)
        
        assert "INFY" in formatted
        assert "ALPHA SCORE" in formatted
        assert "CATEGORY BREAKDOWN" in formatted
        assert "RECOMMENDATION" in formatted


class TestAlphaTracker:
    """Tests for AlphaTracker."""
    
    def test_record_prediction(self):
        """Test recording a prediction."""
        tracker = AlphaTracker()
        
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
            category_scores={"technical": 92, "volume": 88},
            signals_used=["Technical", "Volume"],
            features_used=["ema", "volume"],
        )
        
        assert record.symbol == "INFY"
        assert record.alpha_score == 88.5
        assert record.outcome == PredictionOutcome.PENDING
        assert len(tracker.records) == 1
    
    def test_update_outcome(self):
        """Test updating prediction outcome."""
        tracker = AlphaTracker()
        
        tracker.record_prediction(
            date=date.today(),
            symbol="INFY",
            alpha_score=88.5,
            grade=AlphaGrade.EXCELLENT,
            predicted_direction="long",
            entry_price=1450.0,
            target_price=1500.0,
            stop_loss=1400.0,
            regime="BULL",
            category_scores={"technical": 92},
            signals_used=["Technical"],
            features_used=["ema"],
        )
        
        updated = tracker.update_outcome(
            symbol="INFY",
            date=date.today(),
            actual_return=0.05,
            outcome=PredictionOutcome.CORRECT,
        )
        
        assert updated is not None
        assert updated.actual_return == 0.05
        assert updated.outcome == PredictionOutcome.CORRECT
    
    def test_grade_performance_calculation(self):
        """Test grade performance calculation."""
        tracker = AlphaTracker()
        
        # Add multiple predictions
        for i in range(10):
            grade = AlphaGrade.EXCELLENT if i < 5 else AlphaGrade.GOOD
            tracker.record_prediction(
                date=date.today(),
                symbol=f"STOCK{i}",
                alpha_score=75.0 + i,
                grade=grade,
                predicted_direction="long",
                entry_price=1000.0,
                target_price=1050.0,
                stop_loss=950.0,
                regime="BULL",
                category_scores={"technical": 75 + i},
                signals_used=["Technical"],
                features_used=["ema"],
            )
            
            # Update outcomes
            outcome = PredictionOutcome.CORRECT if i % 2 == 0 else PredictionOutcome.INCORRECT
            tracker.update_outcome(
                symbol=f"STOCK{i}",
                date=date.today(),
                actual_return=0.03 if i % 2 == 0 else -0.02,
                outcome=outcome,
            )
        
        performance = tracker.get_grade_performance()
        
        assert AlphaGrade.EXCELLENT in performance
        assert AlphaGrade.GOOD in performance
        assert performance[AlphaGrade.EXCELLENT].total_predictions == 5
    
    def test_performance_summary(self):
        """Test performance summary."""
        tracker = AlphaTracker()
        
        tracker.record_prediction(
            date=date.today(),
            symbol="INFY",
            alpha_score=88.5,
            grade=AlphaGrade.EXCELLENT,
            predicted_direction="long",
            entry_price=1450.0,
            target_price=1500.0,
            stop_loss=1400.0,
            regime="BULL",
            category_scores={"technical": 92},
            signals_used=["Technical"],
            features_used=["ema"],
        )
        
        tracker.update_outcome(
            symbol="INFY",
            date=date.today(),
            actual_return=0.05,
            outcome=PredictionOutcome.CORRECT,
        )
        
        summary = tracker.get_performance_summary()
        
        assert summary["total_predictions"] == 1
        assert summary["completed_predictions"] == 1
        assert summary["overall_win_rate"] == 1.0
    
    def test_dataframe_conversion(self):
        """Test conversion to DataFrame."""
        tracker = AlphaTracker()
        
        tracker.record_prediction(
            date=date.today(),
            symbol="INFY",
            alpha_score=88.5,
            grade=AlphaGrade.EXCELLENT,
            predicted_direction="long",
            entry_price=1450.0,
            target_price=1500.0,
            stop_loss=1400.0,
            regime="BULL",
            category_scores={"technical": 92},
            signals_used=["Technical"],
            features_used=["ema"],
        )
        
        df = tracker.get_dataframe()
        
        assert len(df) == 1
        assert "symbol" in df.columns
        assert "alpha_score" in df.columns
        assert df.iloc[0]["symbol"] == "INFY"


class TestAlphaReports:
    """Tests for AlphaReports."""
    
    def test_weekly_report_generation(self):
        """Test weekly report generation."""
        tracker = AlphaTracker()
        
        # Add historical data
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
                category_scores={"technical": 75 + i},
                signals_used=["Technical"],
                features_used=["ema"],
            )
            
            if i < 5:
                outcome = PredictionOutcome.CORRECT if i % 2 == 0 else PredictionOutcome.INCORRECT
                tracker.update_outcome(
                    symbol=f"STOCK{i}",
                    date=date.today() - pd.Timedelta(days=i),
                    actual_return=0.03 if i % 2 == 0 else -0.02,
                    outcome=outcome,
                )
        
        reports = AlphaReports(tracker)
        report = reports.generate_weekly_report()
        
        assert report.total_predictions > 0
        assert len(report.insights) > 0
        assert len(report.recommendations) > 0
    
    def test_report_markdown_formatting(self):
        """Test report markdown formatting."""
        tracker = AlphaTracker()
        
        tracker.record_prediction(
            date=date.today(),
            symbol="INFY",
            alpha_score=88.5,
            grade=AlphaGrade.EXCELLENT,
            predicted_direction="long",
            entry_price=1450.0,
            target_price=1500.0,
            stop_loss=1400.0,
            regime="BULL",
            category_scores={"technical": 92},
            signals_used=["Technical"],
            features_used=["ema"],
        )
        
        reports = AlphaReports(tracker)
        report = reports.generate_weekly_report()
        markdown = reports.format_report_markdown(report)
        
        assert markdown is not None
        assert len(markdown) > 0
        assert "Weekly Alpha Research Report" in markdown
        assert "Summary" in markdown


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
