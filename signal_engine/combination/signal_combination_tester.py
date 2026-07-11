"""
Signal Combination Tester

Tests different combinations of signals to find profitable alpha rules.
This is where real research begins - finding which signal combinations work best.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from signal_engine.base import SignalCategory, SignalSet
from utils.logger import get_logger

logger = get_logger("signal_engine.combination")


@dataclass
class CombinationRule:
    """A signal combination rule."""
    name: str
    category_thresholds: Dict[SignalCategory, float]
    require_direction_agreement: bool = True
    min_categories: int = 2
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'category_thresholds': {cat.value: thresh for cat, thresh in self.category_thresholds.items()},
            'require_direction_agreement': self.require_direction_agreement,
            'min_categories': self.min_categories,
        }


@dataclass
class CombinationTestResult:
    """Result of testing a signal combination."""
    rule: CombinationRule
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_return: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'rule': self.rule.to_dict(),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate, 4),
            'average_return': round(self.average_return, 4),
            'profit_factor': round(self.profit_factor, 4),
            'sharpe_ratio': round(self.sharpe_ratio, 4),
            'max_drawdown': round(self.max_drawdown, 4),
        }


class SignalCombinationTester:
    """
    Signal Combination Tester.
    
    Tests combinations like:
    - Trend > 80 + Volume > 75 + Options > 70
    - Trend > 90 + Sentiment > 80 + Sector > 85
    
    Saves the best combinations as alpha rules.
    """
    
    def __init__(self):
        """Initialize combination tester."""
        self.tested_rules: Dict[str, CombinationTestResult] = {}
        self.best_rules: List[CombinationTestResult] = []
        self._logger = get_logger("signal_engine.combination")
    
    def test_combination(
        self,
        rule: CombinationRule,
        historical_signal_sets: List[tuple[SignalSet, float]],
    ) -> CombinationTestResult:
        """
        Test a signal combination rule on historical data.
        
        Args:
            rule: Combination rule to test
            historical_signal_sets: List of (SignalSet, actual_return) tuples
            
        Returns:
            CombinationTestResult with performance metrics
        """
        # Filter signal sets that match the rule
        matching_trades = []
        
        for signal_set, actual_return in historical_signal_sets:
            if self._matches_rule(signal_set, rule):
                matching_trades.append(actual_return)
        
        # Calculate performance metrics
        total_trades = len(matching_trades)
        
        if total_trades == 0:
            return CombinationTestResult(
                rule=rule,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                average_return=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
            )
        
        winning_trades = sum(1 for ret in matching_trades if ret > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades
        
        average_return = sum(matching_trades) / total_trades
        
        winning_returns = [ret for ret in matching_trades if ret > 0]
        losing_returns = [ret for ret in matching_trades if ret < 0]
        
        total_wins = sum(winning_returns)
        total_losses = abs(sum(losing_returns))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Sharpe ratio
        if len(matching_trades) > 1:
            import numpy as np
            sharpe_ratio = np.mean(matching_trades) / np.std(matching_trades) if np.std(matching_trades) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        cumulative_returns = []
        cumulative = 0.0
        for ret in matching_trades:
            cumulative += ret / 100
            cumulative_returns.append(cumulative)
        
        if cumulative_returns:
            peak = max(cumulative_returns)
            drawdowns = [peak - x for x in cumulative_returns]
            max_drawdown = max(drawdowns) * 100
        else:
            max_drawdown = 0.0
        
        result = CombinationTestResult(
            rule=rule,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            average_return=average_return,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
        )
        
        # Store result
        self.tested_rules[rule.name] = result
        
        self._logger.info(
            f"Tested rule {rule.name}: win_rate={win_rate:.2%}, avg_return={average_return:.2f}%, trades={total_trades}"
        )
        
        return result
    
    def _matches_rule(self, signal_set: SignalSet, rule: CombinationRule) -> bool:
        """
        Check if a signal set matches the rule.
        
        Args:
            signal_set: SignalSet to check
            rule: Combination rule
            
        Returns:
            True if signal set matches the rule
        """
        # Check minimum categories
        if len(signal_set.signals) < rule.min_categories:
            return False
        
        # Check category thresholds
        passing_categories = 0
        for category, threshold in rule.category_thresholds.items():
            signal = signal_set.get_signal(category)
            if signal and signal.score >= threshold:
                passing_categories += 1
        
        if passing_categories < rule.min_categories:
            return False
        
        # Check direction agreement if required
        if rule.require_direction_agreement:
            directions = [signal.direction for signal in signal_set.signals.values()]
            if len(set(directions)) > 1:
                return False
        
        return True
    
    def generate_test_rules(
        self,
        categories: List[SignalCategory],
        score_ranges: Dict[tuple[float, float], List[float]],
    ) -> List[CombinationRule]:
        """
        Generate test rules for combination testing.
        
        Args:
            categories: List of categories to include in rules
            score_ranges: Dictionary mapping (min_score, max_score) to list of threshold values
            
        Returns:
            List of CombinationRule objects
        """
        rules = []
        rule_id = 0
        
        # Generate combinations of 2-3 categories
        from itertools import combinations
        
        for num_categories in [2, 3]:
            for category_combo in combinations(categories, num_categories):
                # Generate threshold combinations
                threshold_combinations = self._generate_threshold_combinations(
                    category_combo,
                    score_ranges,
                )
                
                for thresholds in threshold_combinations:
                    rule_id += 1
                    rule = CombinationRule(
                        name=f"rule_{rule_id}",
                        category_thresholds=dict(zip(category_combo, thresholds)),
                        require_direction_agreement=True,
                        min_categories=num_categories,
                    )
                    rules.append(rule)
        
        return rules
    
    def _generate_threshold_combinations(
        self,
        categories: List[SignalCategory],
        score_ranges: Dict[tuple[float, float], List[float]],
    ) -> List[List[float]]:
        """Generate threshold combinations for categories."""
        from itertools import product
        
        # Use default score ranges if not provided
        if not score_ranges:
            score_ranges = {
                (60, 90): [60, 70, 80, 90],
            }
        
        # Get threshold values for each category
        threshold_lists = []
        for category in categories:
            # Use same thresholds for all categories (can be customized)
            for range_key, thresholds in score_ranges.items():
                threshold_lists.append(thresholds)
                break
        
        # Generate all combinations
        return list(product(*threshold_lists))
    
    def find_best_rules(
        self,
        metric: str = 'win_rate',
        top_n: int = 10,
        min_trades: int = 20,
    ) -> List[CombinationTestResult]:
        """
        Find the best performing rules by metric.
        
        Args:
            metric: Metric to compare (win_rate, average_return, profit_factor, sharpe_ratio)
            top_n: Number of top rules to return
            min_trades: Minimum number of trades required
            
        Returns:
            List of best performing rules
        """
        # Filter rules with minimum trades
        qualified_rules = [
            result for result in self.tested_rules.values()
            if result.total_trades >= min_trades
        ]
        
        # Sort by metric
        qualified_rules.sort(key=lambda x: getattr(x, metric), reverse=True)
        
        self.best_rules = qualified_rules[:top_n]
        
        return self.best_rules
    
    def get_best_rule(self, metric: str = 'win_rate', min_trades: int = 20) -> Optional[CombinationTestResult]:
        """Get the single best performing rule."""
        best_rules = self.find_best_rules(metric=metric, top_n=1, min_trades=min_trades)
        return best_rules[0] if best_rules else None
    
    def get_test_summary(self) -> Dict:
        """Get summary of tested rules."""
        if not self.tested_rules:
            return {
                'total_rules_tested': 0,
                'qualified_rules': 0,
                'best_win_rate': 0.0,
                'best_average_return': 0.0,
            }
        
        qualified_rules = [r for r in self.tested_rules.values() if r.total_trades >= 20]
        
        return {
            'total_rules_tested': len(self.tested_rules),
            'qualified_rules': len(qualified_rules),
            'best_win_rate': max(r.win_rate for r in qualified_rules) if qualified_rules else 0.0,
            'best_average_return': max(r.average_return for r in qualified_rules) if qualified_rules else 0.0,
            'best_profit_factor': max(r.profit_factor for r in qualified_rules if r.profit_factor != float('inf')) if qualified_rules else 0.0,
        }
