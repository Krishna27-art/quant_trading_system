"""
Alpha Engine

Deterministic alpha scoring engine that combines multiple signal categories
into a single alpha score for ranking stocks before prediction.

Workflow:
1. Collect signals from Signal Engine
2. Normalize all signals to 0-100 scale
3. Create category scores
4. Apply dynamic weighting based on regime and timeframe
5. Adjust based on historical performance
6. Calculate raw alpha score
7. Apply filters (liquidity, risk/reward, news)
8. Grade alpha (Institutional, Excellent, Good, Average, Reject)
9. Rank stocks and select top N
10. Generate explanations
11. Track performance over time
12. Generate weekly research reports
"""

from alpha_engine.alpha_builder import AlphaBuilder
from alpha_engine.alpha_weights import AlphaWeights
from alpha_engine.alpha_filters import AlphaFilters
from alpha_engine.alpha_score import AlphaScoreCalculator
from alpha_engine.alpha_ranker import AlphaRanker
from alpha_engine.alpha_explainer import AlphaExplainer
from alpha_engine.alpha_tracker import AlphaTracker
from alpha_engine.alpha_reports import AlphaReports

# Optional import for regime integration
try:
    from alpha_engine.alpha_regime import AlphaRegimeAdjuster
    _regime_available = True
except ImportError:
    _regime_available = False
    AlphaRegimeAdjuster = None

__all__ = [
    "AlphaBuilder",
    "AlphaWeights",
    "AlphaFilters",
    "AlphaScoreCalculator",
    "AlphaRanker",
    "AlphaExplainer",
    "AlphaTracker",
    "AlphaReports",
]

if _regime_available:
    __all__.append("AlphaRegimeAdjuster")
