"""
Confidence Calibration Analysis Module

Analyzes how well confidence scores match actual outcomes.
Identifies overconfident or underconfident predictions.

Example:
- Confidence 95% should result in ~95% success rate
- If actual success is 62%, confidence needs recalibration
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("confidence_analysis")


class ConfidenceAnalyzer:
    """
    Analyzes confidence calibration.
    
    Determines if confidence scores are well-calibrated
    and suggests adjustments if needed.
    """
    
    def __init__(self, bucket_size: int = 10):
        """
        Initialize the confidence analyzer.
        
        Args:
            bucket_size: Size of confidence buckets (e.g., 10 for 0-10, 10-20, etc.)
        """
        self.bucket_size = bucket_size
        self.logger = logger
    
    def analyze_calibration(
        self,
        predictions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Analyze confidence calibration.
        
        Args:
            predictions: List of prediction dicts with confidence and correct flag
            
        Returns:
            Dict with calibration analysis
        """
        self.logger.info("Analyzing confidence calibration...")
        
        # Group predictions by confidence buckets
        buckets = defaultdict(list)
        for pred in predictions:
            confidence = pred.get('confidence', 0)
            bucket_key = int(confidence // self.bucket_size) * self.bucket_size
            buckets[bucket_key].append(pred)
        
        # Calculate actual success rate for each bucket
        calibration_data = []
        for bucket in sorted(buckets.keys()):
            bucket_preds = buckets[bucket]
            correct = sum(1 for p in bucket_preds if p.get('correct', False))
            total = len(bucket_preds)
            actual_rate = correct / total * 100 if total > 0 else 0
            
            calibration_data.append({
                'confidence_range': f"{bucket}-{bucket + self.bucket_size}",
                'avg_confidence': np.mean([p['confidence'] for p in bucket_preds]),
                'actual_success_rate': actual_rate,
                'expected_success_rate': bucket + self.bucket_size / 2,
                'count': total,
                'calibration_error': actual_rate - (bucket + self.bucket_size / 2),
            })
        
        # Calculate overall calibration metrics
        calibration_metrics = self._calculate_calibration_metrics(predictions, calibration_data)
        
        # Generate recommendations
        recommendations = self._generate_calibration_recommendations(calibration_data)
        
        self.logger.info("Confidence calibration analysis complete")
        return {
            'calibration_data': calibration_data,
            'metrics': calibration_metrics,
            'recommendations': recommendations,
        }
    
    def _calculate_calibration_metrics(
        self,
        predictions: list[dict[str, Any]],
        calibration_data: list[dict],
    ) -> dict[str, Any]:
        """Calculate overall calibration metrics."""
        if not predictions:
            return {}
        
        # Expected vs Actual
        confidences = [p['confidence'] for p in predictions]
        actuals = [1 if p.get('correct', False) else 0 for p in predictions]
        
        expected_accuracy = np.mean(confidences)
        actual_accuracy = np.mean(actuals)
        
        # Mean Absolute Calibration Error
        mace = np.mean([abs(d['actual_success_rate'] - d['expected_success_rate']) for d in calibration_data])
        
        # Brier Score (proper scoring rule)
        brier_score = np.mean([(c - a) ** 2 for c, a in zip(confidences, actuals)])
        
        # Reliability Diagram data
        reliability = {
            'expected': [d['expected_success_rate'] for d in calibration_data],
            'actual': [d['actual_success_rate'] for d in calibration_data],
        }
        
        return {
            'expected_accuracy': round(expected_accuracy, 2),
            'actual_accuracy': round(actual_accuracy, 2),
            'accuracy_gap': round(actual_accuracy - expected_accuracy, 2),
            'mean_calibration_error': round(mace, 2),
            'brier_score': round(brier_score, 4),
            'is_well_calibrated': mace < 10,  # Within 10% is well-calibrated
            'reliability_diagram': reliability,
        }
    
    def _generate_calibration_recommendations(
        self,
        calibration_data: list[dict],
    ) -> dict[str, Any]:
        """Generate recommendations based on calibration analysis."""
        recommendations = {
            'needs_recalibration': False,
            'adjustments': [],
            'overall_assessment': 'Well calibrated',
        }
        
        # Check for systematic overconfidence or underconfidence
        overconfident_buckets = [d for d in calibration_data if d['calibration_error'] < -10]
        underconfident_buckets = [d for d in calibration_data if d['calibration_error'] > 10]
        
        if overconfident_buckets:
            recommendations['needs_recalibration'] = True
            recommendations['adjustments'].append({
                'type': 'reduce_confidence',
                'buckets': [d['confidence_range'] for d in overconfident_buckets],
                'reason': 'Systematically overconfident',
            })
        
        if underconfident_buckets:
            recommendations['needs_recalibration'] = True
            recommendations['adjustments'].append({
                'type': 'increase_confidence',
                'buckets': [d['confidence_range'] for d in underconfident_buckets],
                'reason': 'Systematically underconfident',
            })
        
        if recommendations['needs_recalibration']:
            recommendations['overall_assessment'] = 'Needs recalibration'
        elif any(abs(d['calibration_error']) > 5 for d in calibration_data):
            recommendations['overall_assessment'] = 'Minor calibration issues'
        
        return recommendations
    
    def analyze_probability_calibration(
        self,
        predictions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Analyze probability calibration (similar to confidence but for probabilities).
        
        Args:
            predictions: List of prediction dicts with probability and correct flag
            
        Returns:
            Dict with probability calibration analysis
        """
        self.logger.info("Analyzing probability calibration...")
        
        # Similar to confidence calibration but for probabilities
        probabilities = [p.get('probability', p.get('confidence', 0)) for p in predictions]
        actuals = [1 if p.get('correct', False) else 0 for p in predictions]
        
        # Calculate Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        bin_data = []
        
        for lower, upper in zip(bin_lowers, bin_uppers):
            mask = (np.array(probabilities) > lower) & (np.array(probabilities) <= upper)
            bin_probs = np.array(probabilities)[mask]
            bin_actuals = np.array(actuals)[mask]
            
            if len(bin_probs) > 0:
                avg_confidence = np.mean(bin_probs)
                avg_actual = np.mean(bin_actuals)
                bin_weight = len(bin_probs) / len(predictions)
                ece += bin_weight * abs(avg_confidence - avg_actual)
                
                bin_data.append({
                    'range': f"{lower:.1f}-{upper:.1f}",
                    'avg_confidence': avg_confidence,
                    'avg_actual': avg_actual,
                    'count': len(bin_probs),
                })
        
        return {
            'expected_calibration_error': round(ece, 4),
            'bin_data': bin_data,
            'is_well_calibrated': ece < 0.1,  # ECE < 0.1 is well-calibrated
        }


def analyze_confidence_calibration(
    predictions: list[dict[str, Any]],
    bucket_size: int = 10,
) -> dict[str, Any]:
    """
    Convenience function to analyze confidence calibration.
    
    Args:
        predictions: List of prediction dicts
        bucket_size: Size of confidence buckets
        
    Returns:
        Calibration analysis results
    """
    analyzer = ConfidenceAnalyzer(bucket_size=bucket_size)
    return analyzer.analyze_calibration(predictions)
