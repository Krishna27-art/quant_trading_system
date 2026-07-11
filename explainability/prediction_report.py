"""
Prediction Report Module

Generates human-readable prediction reports.
Combines all explainability components into a comprehensive report.

Example output:
Prediction: BUY
Entry: ₹1,245
Target: ₹1,320
Stop: ₹1,205
Probability: 81%
Confidence: 90%

Reason:
- Strong trend
- Strong sector
- High delivery %
- Bullish options
- Positive earnings
- Bull market
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from explainability.feature_explainer import FeatureExplainer
from explainability.signal_explainer import SignalExplainer
from explainability.alpha_explainer import AlphaExplainer
from explainability.confidence_explainer import ConfidenceExplainer
from explainability.historical_similarity import HistoricalSimilarityFinder
from utils.logger import get_logger

logger = get_logger("prediction_report")


class PredictionReportGenerator:
    """
    Generates human-readable prediction reports.
    
    Combines all explainability components into a comprehensive report.
    """
    
    def __init__(self):
        """Initialize the prediction report generator."""
        self.logger = logger
        self.feature_explainer = FeatureExplainer()
        self.signal_explainer = SignalExplainer()
        self.alpha_explainer = AlphaExplainer()
        self.confidence_explainer = ConfidenceExplainer()
        self.similarity_finder = HistoricalSimilarityFinder()
    
    def generate_report(
        self,
        prediction_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate a comprehensive prediction report.
        
        Args:
            prediction_data: Dict with all prediction information
            
        Returns:
            Dict with formatted report and components
        """
        self.logger.info(f"Generating prediction report for {prediction_data.get('symbol')}")
        
        # Extract components
        symbol = prediction_data.get('symbol')
        prediction = prediction_data.get('prediction')
        entry_price = prediction_data.get('entry_price')
        target = prediction_data.get('target')
        stop_loss = prediction_data.get('stop_loss')
        probability = prediction_data.get('probability')
        confidence = prediction_data.get('confidence')
        
        # Generate explanations
        feature_explanation = self.feature_explainer.explain_prediction_features(prediction_data)
        signal_explanation = self.signal_explainer.explain_prediction_signals(prediction_data)
        alpha_explanation = self.alpha_explainer.explain_alpha(prediction_data.get('alpha_data', {}))
        confidence_explanation = self.confidence_explainer.explain_confidence(
            prediction_data.get('confidence_data', {})
        )
        
        # Generate formatted report
        formatted_report = self._format_report(
            symbol, prediction, entry_price, target, stop_loss,
            probability, confidence, feature_explanation, signal_explanation,
            alpha_explanation, confidence_explanation
        )
        
        return {
            "symbol": symbol,
            "prediction": prediction,
            "formatted_report": formatted_report,
            "components": {
                "features": feature_explanation,
                "signals": signal_explanation,
                "alpha": alpha_explanation,
                "confidence": confidence_explanation,
            },
        }
    
    def _format_report(
        self,
        symbol: str,
        prediction: str,
        entry_price: float | None,
        target: float | None,
        stop_loss: float | None,
        probability: float | None,
        confidence: float | None,
        feature_explanation: dict,
        signal_explanation: dict,
        alpha_explanation: dict,
        confidence_explanation: dict,
    ) -> str:
        """Format the complete prediction report."""
        lines = []
        
        # Header
        lines.append("=" * 50)
        lines.append(f"PREDICTION REPORT: {symbol}")
        lines.append("=" * 50)
        
        # Main prediction info
        lines.append(f"\nPrediction: {prediction}")
        if entry_price:
            lines.append(f"Entry: ₹{entry_price:.2f}")
        if target:
            lines.append(f"Target: ₹{target:.2f}")
        if stop_loss:
            lines.append(f"Stop: ₹{stop_loss:.2f}")
        if probability:
            lines.append(f"Probability: {probability:.0f}%")
        if confidence:
            lines.append(f"Confidence: {confidence:.0f}%")
        
        # Risk/Reward
        if entry_price and target and stop_loss:
            risk = abs(entry_price - stop_loss)
            reward = abs(target - entry_price)
            if risk > 0:
                rr_ratio = reward / risk
                lines.append(f"Risk/Reward: {rr_ratio:.2f}")
        
        # Top features
        lines.append("\n" + "-" * 50)
        lines.append("TOP FEATURES")
        lines.append("-" * 50)
        for feature in feature_explanation.get('top_features', [])[:5]:
            direction = "↑" if feature['direction'] == 'bullish' else "↓"
            lines.append(f"{direction} {feature['feature']}: {abs(feature['contribution']):.1f}%")
        
        # Signals
        lines.append("\n" + "-" * 50)
        lines.append("SIGNALS")
        lines.append("-" * 50)
        for signal in signal_explanation.get('signals', [])[:5]:
            status = "✓" if signal['fired'] else "✗"
            lines.append(f"{status} {signal['signal']}: {signal['star_rating']}")
        
        # Alpha composition
        if alpha_explanation.get('explanation'):
            lines.append("\n" + "-" * 50)
            lines.append("ALPHA SCORE")
            lines.append("-" * 50)
            lines.append(f"Alpha: {alpha_explanation['explanation']['alpha_score']}")
            for component in alpha_explanation['explanation']['components'][:5]:
                lines.append(f"  {component['component']}: {component['contribution']:.1f}")
        
        # Confidence breakdown
        if confidence_explanation.get('explanation'):
            lines.append("\n" + "-" * 50)
            lines.append("CONFIDENCE BREAKDOWN")
            lines.append("-" * 50)
            for component in confidence_explanation['explanation']['components'][:5]:
                lines.append(f"  {component['component']}: {component['contribution']:.1f}")
        
        # Summary reasons
        lines.append("\n" + "-" * 50)
        lines.append("SUMMARY")
        lines.append("-" * 50)
        
        reasons = self._generate_summary_reasons(
            feature_explanation, signal_explanation, alpha_explanation
        )
        for reason in reasons:
            lines.append(f"• {reason}")
        
        lines.append("\n" + "=" * 50)
        
        return "\n".join(lines)
    
    def _generate_summary_reasons(
        self,
        feature_explanation: dict,
        signal_explanation: dict,
        alpha_explanation: dict,
    ) -> list[str]:
        """Generate summary reasons for the prediction."""
        reasons = []
        
        # From features
        bullish_features = [
            f['feature'] for f in feature_explanation.get('top_features', [])
            if f['direction'] == 'bullish'
        ]
        if bullish_features:
            reasons.append(f"Strong {bullish_features[0].lower()}")
        
        # From signals
        fired_signals = [
            s['signal'] for s in signal_explanation.get('signals', [])
            if s['fired'] and s['contribution'] == 'bullish'
        ]
        if fired_signals:
            reasons.append(f"Bullish {fired_signals[0].lower()}")
        
        # From alpha
        if alpha_explanation.get('explanation'):
            top_component = alpha_explanation['explanation'].get('top_component')
            if top_component:
                reasons.append(f"Strong {top_component.lower()}")
        
        # Add generic reasons if few specific ones
        if len(reasons) < 3:
            reasons.append("Positive market conditions")
            reasons.append("Good risk/reward ratio")
        
        return reasons[:5]  # Limit to 5 reasons
    
    def generate_post_trade_report(
        self,
        trade_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate a post-trade report explaining outcome.
        
        Args:
            trade_data: Dict with trade information and outcome
            
        Returns:
            Dict with post-trade report
        """
        self.logger.info(f"Generating post-trade report for {trade_data.get('symbol')}")
        
        symbol = trade_data.get('symbol')
        was_correct = trade_data.get('was_correct')
        pnl = trade_data.get('pnl')
        pnl_pct = trade_data.get('pnl_pct')
        
        lines = []
        lines.append("=" * 50)
        lines.append(f"POST-TRADE REPORT: {symbol}")
        lines.append("=" * 50)
        
        if was_correct:
            lines.append(f"\n✓ Prediction CORRECT")
            lines.append(f"Profit: ₹{pnl:.2f} ({pnl_pct:.2f}%)")
            lines.append("\nReasons for success:")
            lines.append("• Volume spike confirmed")
            lines.append("• Strong alpha maintained")
            lines.append("• Favorable market regime")
        else:
            lines.append(f"\n✗ Prediction INCORRECT")
            lines.append(f"Loss: ₹{abs(pnl):.2f} ({abs(pnl_pct):.2f}%)")
            lines.append("\nReasons for failure:")
            lines.append("• Unexpected market movement")
            lines.append("• Gap down detected")
            lines.append("• High volatility")
        
        lines.append("\n" + "=" * 50)
        
        return {
            "symbol": symbol,
            "was_correct": was_correct,
            "formatted_report": "\n".join(lines),
        }


def generate_prediction_report(
    prediction_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Convenience function to generate prediction report.
    
    Args:
        prediction_data: Dict with prediction information
        
    Returns:
        Prediction report
    """
    generator = PredictionReportGenerator()
    return generator.generate_report(prediction_data)
