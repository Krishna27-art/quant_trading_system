"""
Signal Filter Engine

Filters weak signals and applies multi-signal confirmation rules.
Only high-quality signals pass through to prediction models.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from signal_engine.base import Signal, SignalCategory, SignalDirection, SignalSet, SignalFilterResult
from utils.logger import get_logger

logger = get_logger("signal_engine.filtering")


@dataclass
class FilterRule:
    """Single filter rule."""
    category: SignalCategory
    min_score: float
    required_direction: Optional[SignalDirection] = None
    min_confidence: float = 50.0


@dataclass
class MultiSignalConfirmation:
    """Multi-signal confirmation configuration."""
    required_categories: List[SignalCategory]
    min_categories: int
    require_same_direction: bool = True
    min_overall_score: float = 70.0


class SignalFilter:
    """
    Signal Filter Engine.
    
    Filters signals by:
    1. Minimum score thresholds per category
    2. Minimum confidence thresholds
    3. Direction requirements
    4. Multi-signal confirmation rules
    5. Overall score thresholds
    
    Most stocks should be rejected. Only high-quality setups pass.
    """
    
    def __init__(
        self,
        filter_rules: Optional[List[FilterRule]] = None,
        confirmation_config: Optional[MultiSignalConfirmation] = None,
    ):
        """
        Initialize signal filter.
        
        Args:
            filter_rules: List of filter rules for each category
            confirmation_config: Multi-signal confirmation configuration
        """
        self.filter_rules = filter_rules or self._default_filter_rules()
        self.confirmation_config = confirmation_config or self._default_confirmation_config()
        self._logger = get_logger("signal_engine.filtering")
    
    def _default_filter_rules(self) -> List[FilterRule]:
        """Create default filter rules."""
        return [
            FilterRule(
                category=SignalCategory.TECHNICAL,
                min_score=60.0,
                min_confidence=60.0,
            ),
            FilterRule(
                category=SignalCategory.VOLUME,
                min_score=55.0,
                min_confidence=50.0,
            ),
            FilterRule(
                category=SignalCategory.OPTIONS,
                min_score=50.0,
                min_confidence=50.0,
            ),
            FilterRule(
                category=SignalCategory.FUNDAMENTAL,
                min_score=50.0,
                min_confidence=50.0,
            ),
            FilterRule(
                category=SignalCategory.SENTIMENT,
                min_score=50.0,
                min_confidence=40.0,
            ),
        ]
    
    def _default_confirmation_config(self) -> MultiSignalConfirmation:
        """Create default multi-signal confirmation configuration."""
        return MultiSignalConfirmation(
            required_categories=[
                SignalCategory.TECHNICAL,
                SignalCategory.VOLUME,
            ],
            min_categories=2,
            require_same_direction=True,
            min_overall_score=70.0,
        )
    
    def filter_signal_set(self, signal_set: SignalSet) -> SignalFilterResult:
        """
        Filter a signal set.
        
        Args:
            signal_set: SignalSet to filter
            
        Returns:
            SignalFilterResult with pass/fail status
        """
        if not signal_set.signals:
            return SignalFilterResult(
                passed=False,
                reason="No signals available",
                filtered_signals={},
                overall_score=0.0,
            )
        
        # Apply individual filter rules
        filtered_signals = {}
        failed_reasons = []
        
        for category, signal in signal_set.signals.items():
            rule = self._get_rule_for_category(category)
            
            if rule:
                # Check minimum score
                if signal.score < rule.min_score:
                    failed_reasons.append(
                        f"{category.value} score {signal.score:.1f} below threshold {rule.min_score}"
                    )
                    continue
                
                # Check minimum confidence
                if signal.confidence < rule.min_confidence:
                    failed_reasons.append(
                        f"{category.value} confidence {signal.confidence:.1f}% below threshold {rule.min_confidence}%"
                    )
                    continue
                
                # Check direction requirement
                if rule.required_direction and signal.direction != rule.required_direction:
                    failed_reasons.append(
                        f"{category.value} direction {signal.direction.value} not {rule.required_direction.value}"
                    )
                    continue
            
            # Signal passed all rules
            filtered_signals[category] = signal
        
        # Check multi-signal confirmation
        if not self._check_confirmation(filtered_signals):
            failed_reasons.append(
                f"Multi-signal confirmation failed: need at least {self.confirmation_config.min_categories} categories"
            )
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(filtered_signals)
        
        # Check minimum overall score
        if overall_score < self.confirmation_config.min_overall_score:
            failed_reasons.append(
                f"Overall score {overall_score:.1f} below threshold {self.confirmation_config.min_overall_score}"
            )
        
        # Determine pass/fail
        passed = len(failed_reasons) == 0
        reason = "; ".join(failed_reasons) if failed_reasons else "All filters passed"
        
        return SignalFilterResult(
            passed=passed,
            reason=reason,
            filtered_signals=filtered_signals,
            overall_score=overall_score,
        )
    
    def _get_rule_for_category(self, category: SignalCategory) -> Optional[FilterRule]:
        """Get filter rule for a category."""
        for rule in self.filter_rules:
            if rule.category == category:
                return rule
        return None
    
    def _check_confirmation(self, filtered_signals: Dict[SignalCategory, Signal]) -> bool:
        """
        Check multi-signal confirmation.
        
        Args:
            filtered_signals: Dictionary of filtered signals
            
        Returns:
            True if confirmation requirements are met
        """
        # Check minimum number of categories
        if len(filtered_signals) < self.confirmation_config.min_categories:
            return False
        
        # Check if required categories are present
        for required_category in self.confirmation_config.required_categories:
            if required_category not in filtered_signals:
                return False
        
        # Check direction agreement if required
        if self.confirmation_config.require_same_direction:
            directions = [signal.direction for signal in filtered_signals.values()]
            
            # All signals should have the same direction
            if len(set(directions)) > 1:
                return False
        
        return True
    
    def _calculate_overall_score(self, filtered_signals: Dict[SignalCategory, Signal]) -> float:
        """Calculate overall score from filtered signals."""
        if not filtered_signals:
            return 0.0
        
        total_score = sum(signal.score for signal in filtered_signals.values())
        return total_score / len(filtered_signals)
    
    def filter_multiple_signal_sets(
        self,
        signal_sets: Dict[str, SignalSet],
    ) -> Dict[str, SignalFilterResult]:
        """
        Filter multiple signal sets.
        
        Args:
            signal_sets: Dictionary mapping symbols to SignalSets
            
        Returns:
            Dictionary mapping symbols to SignalFilterResults
        """
        results = {}
        
        for symbol, signal_set in signal_sets.items():
            results[symbol] = self.filter_signal_set(signal_set)
        
        return results
    
    def get_passed_symbols(
        self,
        filter_results: Dict[str, SignalFilterResult],
    ) -> List[str]:
        """
        Get list of symbols that passed filtering.
        
        Args:
            filter_results: Dictionary of filter results
            
        Returns:
            List of symbols that passed
        """
        return [symbol for symbol, result in filter_results.items() if result.passed]
    
    def get_rejection_stats(
        self,
        filter_results: Dict[str, SignalFilterResult],
    ) -> Dict:
        """
        Get statistics about rejections.
        
        Args:
            filter_results: Dictionary of filter results
            
        Returns:
            Dictionary with rejection statistics
        """
        total = len(filter_results)
        passed = sum(1 for result in filter_results.values() if result.passed)
        rejected = total - passed
        
        # Analyze rejection reasons
        rejection_reasons = {}
        for result in filter_results.values():
            if not result.passed:
                for reason in result.reason.split("; "):
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        
        return {
            'total': total,
            'passed': passed,
            'rejected': rejected,
            'pass_rate': passed / total if total > 0 else 0.0,
            'rejection_reasons': rejection_reasons,
        }
