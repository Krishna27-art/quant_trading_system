"""
Base Formation Detector - Consolidation Pattern Analysis

Detects base/consolidation patterns that often precede strong swing trades.
Institutions accumulate gradually during these quieter periods before larger moves.

Base types:
- Flat Base: Price consolidates horizontally
- Ascending Triangle: Higher lows, flat highs (bullish)
- Descending Triangle: Lower highs, flat lows (bearish)
- Rectangle: Price trapped between support and resistance

Professional workflow:
1. Identify consolidation period
2. Determine base type
3. Calculate breakout level
4. Assess breakout probability
5. Estimate potential upside
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("base_formation_detector")


class BaseType(str, Enum):
    """Types of base formations."""
    FLAT_BASE = "flat_base"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    RECTANGLE = "rectangle"
    CUP_AND_HANDLE = "cup_and_handle"
    NONE = "none"


@dataclass
class BaseMetrics:
    """Metrics for base formation."""
    symbol: str
    has_base: bool
    base_type: BaseType
    base_start_date: datetime
    base_end_date: datetime
    base_duration_days: int
    
    # Price levels
    base_high: float
    base_low: float
    base_mid: float
    base_range_pct: float
    
    # Breakout levels
    breakout_level: float
    breakdown_level: float
    
    # Volume analysis
    avg_volume_during_base: int
    volume_dry_up: bool  # Volume decreases during base (accumulation sign)
    
    # Pattern quality
    breakout_probability: float  # 0-1
    pattern_quality_score: float  # 0-100
    
    # Potential targets
    measured_move_target: float
    conservative_target: float
    aggressive_target: float
    
    timestamp: datetime


class BaseFormationDetector:
    """
    Detects base/consolidation patterns for swing trading.
    
    Key principles:
    - Bases represent accumulation/distribution periods
    - Volume dry-up during base is bullish (accumulation)
    - Longer bases often lead to larger moves
    - Breakout with volume confirmation is key signal
    """

    def __init__(
        self,
        min_base_duration_days: int = 10,
        max_base_duration_days: int = 60,
        max_base_range_pct: float = 15.0,
        min_volume_dry_up_ratio: float = 0.7,
        use_mock_data: bool = True
    ):
        """
        Initialize base formation detector.
        
        Parameters
        ----------
        min_base_duration_days
            Minimum base duration in days
        max_base_duration_days
            Maximum base duration in days
        max_base_range_pct
            Maximum base range as percentage
        min_volume_dry_up_ratio
            Minimum volume reduction ratio for dry-up detection
        use_mock_data
            Use mock data for development
        """
        self.min_base_duration_days = min_base_duration_days
        self.max_base_duration_days = max_base_duration_days
        self.max_base_range_pct = max_base_range_pct
        self.min_volume_dry_up_ratio = min_volume_dry_up_ratio
        self.use_mock_data = use_mock_data

    def detect_base(self, symbol: str) -> Optional[BaseMetrics]:
        """
        Detect base formation for a symbol.
        
        Parameters
        ----------
        symbol
            Stock symbol
            
        Returns
        -------
        Optional[BaseMetrics]
            Base metrics if base detected, None otherwise
        """
        if self.use_mock_data:
            return self._mock_base_detection(symbol)
        
        # In production, fetch historical data and analyze
        return None

    def _mock_base_detection(self, symbol: str) -> Optional[BaseMetrics]:
        """Generate mock base detection for development."""
        import random
        
        # 40% chance of having a base
        if random.random() < 0.4:
            base_types = [
                BaseType.FLAT_BASE,
                BaseType.ASCENDING_TRIANGLE,
                BaseType.RECTANGLE,
                BaseType.CUP_AND_HANDLE
            ]
            
            base_type = random.choice(base_types)
            duration = random.randint(self.min_base_duration_days, self.max_base_duration_days)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=duration)
            
            current_price = random.uniform(100.0, 2000.0)
            base_range = current_price * random.uniform(0.03, self.max_base_range_pct / 100.0)
            
            base_high = current_price + base_range / 2
            base_low = current_price - base_range / 2
            base_mid = (base_high + base_low) / 2
            
            base_range_pct = (base_high - base_low) / base_low * 100
            
            # Breakout level
            if base_type == BaseType.ASCENDING_TRIANGLE:
                breakout_level = base_high
                breakdown_level = base_low
            elif base_type == BaseType.DESCENDING_TRIANGLE:
                breakout_level = base_mid
                breakdown_level = base_low
            else:
                breakout_level = base_high
                breakdown_level = base_low
            
            # Volume analysis
            avg_volume = random.randint(500_000, 5_000_000)
            volume_dry_up = random.random() < 0.6  # 60% chance of dry-up
            
            # Calculate quality metrics
            pattern_quality = self._calculate_pattern_quality(
                base_type, duration, base_range_pct, volume_dry_up
            )
            
            breakout_prob = pattern_quality / 100.0
            
            # Calculate targets
            measured_move = breakout_level * (1 + base_range_pct / 100.0)
            conservative_target = breakout_level * (1 + base_range_pct / 200.0)
            aggressive_target = breakout_level * (1 + base_range_pct / 50.0)
            
            return BaseMetrics(
                symbol=symbol,
                has_base=True,
                base_type=base_type,
                base_start_date=start_date,
                base_end_date=end_date,
                base_duration_days=duration,
                base_high=base_high,
                base_low=base_low,
                base_mid=base_mid,
                base_range_pct=base_range_pct,
                breakout_level=breakout_level,
                breakdown_level=breakdown_level,
                avg_volume_during_base=avg_volume,
                volume_dry_up=volume_dry_up,
                breakout_probability=breakout_prob,
                pattern_quality_score=pattern_quality,
                measured_move_target=measured_move,
                conservative_target=conservative_target,
                aggressive_target=aggressive_target,
                timestamp=datetime.now()
            )
        
        return None

    def _calculate_pattern_quality(
        self,
        base_type: BaseType,
        duration: int,
        range_pct: float,
        volume_dry_up: bool
    ) -> float:
        """
        Calculate pattern quality score (0-100).
        
        Higher quality bases have:
        - Appropriate duration (not too short, not too long)
        - Reasonable range (not too wide)
        - Volume dry-up (accumulation)
        - Clear structure
        """
        score = 50.0  # Base score
        
        # Duration score
        if 15 <= duration <= 30:
            score += 20
        elif 10 <= duration < 15 or 30 < duration <= 45:
            score += 10
        elif duration > 45:
            score -= 10
        
        # Range score (tighter is better)
        if range_pct <= 5:
            score += 15
        elif range_pct <= 10:
            score += 10
        elif range_pct <= 15:
            score += 5
        else:
            score -= 10
        
        # Volume dry-up score
        if volume_dry_up:
            score += 15
        
        # Base type score
        type_scores = {
            BaseType.FLAT_BASE: 10,
            BaseType.ASCENDING_TRIANGLE: 15,
            BaseType.CUP_AND_HANDLE: 20,
            BaseType.RECTANGLE: 5,
            BaseType.DESCENDING_TRIANGLE: 0
        }
        score += type_scores.get(base_type, 0)
        
        return min(100, max(0, score))

    def analyze_breakout_potential(
        self,
        base: BaseMetrics,
        current_price: float,
        current_volume: int
    ) -> dict[str, Any]:
        """
        Analyze breakout potential given current price and volume.
        
        Parameters
        ----------
        base
            Base metrics

        current_price
            Current price
        current_volume
            Current volume
            
        Returns
        -------
        dict[str, Any]
            Breakout analysis with recommendations
        """
        distance_to_breakout = (base.breakout_level - current_price) / current_price * 100
        distance_to_breakdown = (current_price - base.breakdown_level) / current_price * 100
        
        volume_ratio = current_volume / base.avg_volume_during_base if base.avg_volume_during_base > 0 else 1.0
        
        # Breakout imminent if close to level with volume
        breakout_imminent = (
            abs(distance_to_breakout) < 2.0 and
            volume_ratio > 1.5 and
            base.breakout_probability > 0.7
        )
        
        # Breakdown risk
        breakdown_risk = (
            distance_to_breakdown < 3.0 and
            volume_ratio > 1.3 and
            base.pattern_quality_score < 60
        )
        
        return {
            "distance_to_breakout_pct": distance_to_breakout,
            "distance_to_breakdown_pct": distance_to_breakdown,
            "volume_ratio": volume_ratio,
            "breakout_imminent": breakout_imminent,
            "breakdown_risk": breakdown_risk,
            "action": self._recommend_action(base, current_price, volume_ratio, breakout_imminent, breakdown_risk)
        }

    def _recommend_action(
        self,
        base: BaseMetrics,
        current_price: float,
        volume_ratio: float,
        breakout_imminent: bool,
        breakdown_risk: bool
    ) -> str:
        """Recommend action based on base analysis."""
        if breakdown_risk:
            return "AVOID - Breakdown risk high"
        elif breakout_imminent:
            return "WATCH - Breakout imminent, prepare entry"
        elif volume_ratio > 1.2 and current_price > base.base_mid:
            return "ACCUMULATE - Building position before breakout"
        elif current_price < base.base_low:
            return "WAIT - Price below base, wait for re-entry"
        else:
            return "MONITOR - Base intact, watch for breakout"
