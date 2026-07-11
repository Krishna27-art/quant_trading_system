"""
Alpha Builder

Collects and normalizes signals from the Signal Engine into a unified structure
for alpha scoring.

STEP 1: Collect all signals from Signal Engine
STEP 2: Normalize everything to 0-100 scale
STEP 3: Create alpha category scores
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from signal_engine.base import Signal, SignalCategory, SignalDirection, SignalSet
from utils.logger import get_logger

logger = get_logger("alpha_engine.builder")


class AlphaGrade(Enum):
    """Alpha grade classification."""
    INSTITUTIONAL = "institutional"  # 95-100
    EXCELLENT = "excellent"  # 85-94
    GOOD = "good"  # 75-84
    AVERAGE = "average"  # 60-74
    REJECT = "reject"  # Below 60


class TimeFrame(Enum):
    """Trading timeframes for weight configuration."""
    INTRADAY = "intraday"
    SWING = "swing"
    LONGTERM = "longterm"


@dataclass
class AlphaCategory:
    """
    Represents a single alpha category score.
    
    Each category combines multiple signals from the same domain.
    """
    name: str
    score: float  # 0-100
    weight: float  # 0-1
    signals: List[Signal] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "score": round(self.score, 2),
            "weight": round(self.weight, 4),
            "signal_count": len(self.signals),
        }


@dataclass
class AlphaResult:
    """
    Complete alpha scoring result for a single stock.
    
    Contains:
    - Category scores
    - Raw alpha score
    - Final alpha score (after filters)
    - Alpha grade
    - Filter results
    - Explanation
    """
    symbol: str
    timestamp: datetime
    categories: Dict[str, AlphaCategory]
    raw_alpha_score: float
    final_alpha_score: float
    grade: AlphaGrade
    passed_filters: bool
    filter_reasons: List[str] = field(default_factory=list)
    explanation: Dict[str, Any] = field(default_factory=dict)
    risk_reward: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "raw_alpha_score": round(self.raw_alpha_score, 2),
            "final_alpha_score": round(self.final_alpha_score, 2),
            "grade": self.grade.value,
            "passed_filters": self.passed_filters,
            "filter_reasons": self.filter_reasons,
            "categories": {k: v.to_dict() for k, v in self.categories.items()},
            "explanation": self.explanation,
            "risk_reward": self.risk_reward,
        }


class AlphaBuilder:
    """
    Collects and normalizes signals from Signal Engine.
    
    This is the entry point for the alpha scoring pipeline.
    """
    
    def __init__(self):
        """Initialize Alpha Builder."""
        self._logger = logger
        
        # Define category mappings
        self.category_mappings = {
            SignalCategory.TECHNICAL: "technical",
            SignalCategory.VOLUME: "volume",
            SignalCategory.OPTIONS: "options",
            SignalCategory.FUNDAMENTAL: "fundamental",
            SignalCategory.SENTIMENT: "sentiment",
            SignalCategory.MACRO: "macro",
            SignalCategory.SECTOR: "sector",
            SignalCategory.MARKET: "market",
        }
    
    def build_alpha_input(
        self,
        signal_set: SignalSet,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build alpha input from signal set.
        
        Args:
            signal_set: SignalSet from Signal Engine
            additional_data: Additional data (price, volume, etc.)
            
        Returns:
            Dictionary with normalized signals and category scores
        """
        self._logger.info(f"Building alpha input for {signal_set.symbol}")
        
        # Step 1: Normalize all signals to 0-100
        normalized_signals = self._normalize_signals(signal_set.signals)
        
        # Step 2: Create category scores
        category_scores = self._create_category_scores(normalized_signals)
        
        # Step 3: Add additional data if provided
        result = {
            "symbol": signal_set.symbol,
            "timestamp": signal_set.timestamp,
            "normalized_signals": normalized_signals,
            "category_scores": category_scores,
            "additional_data": additional_data or {},
        }
        
        self._logger.info(
            f"Alpha input built: {len(category_scores)} categories",
            extra={"categories": list(category_scores.keys())},
        )
        
        return result
    
    def build_batch_alpha_input(
        self,
        signal_sets: Dict[str, SignalSet],
        additional_data: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build alpha input for multiple stocks.
        
        Args:
            signal_sets: Dictionary of symbol -> SignalSet
            additional_data: Dictionary of symbol -> additional data
            
        Returns:
            Dictionary of symbol -> alpha input
        """
        self._logger.info(f"Building batch alpha input for {len(signal_sets)} stocks")
        
        results = {}
        for symbol, signal_set in signal_sets.items():
            add_data = additional_data.get(symbol) if additional_data else None
            results[symbol] = self.build_alpha_input(signal_set, add_data)
        
        return results
    
    def _normalize_signals(
        self,
        signals: Dict[SignalCategory, Signal],
    ) -> Dict[str, Dict[str, float]]:
        """
        Normalize all signals to 0-100 scale.
        
        Args:
            signals: Dictionary of SignalCategory -> Signal
            
        Returns:
            Dictionary of category_name -> {signal_name: normalized_score}
        """
        normalized = {}
        
        for category, signal in signals.items():
            category_name = self.category_mappings.get(category.value, category.value)
            
            # Signal score should already be 0-100 from Signal Engine
            # But we validate and clamp it
            normalized_score = np.clip(signal.score, 0, 100)
            
            if category_name not in normalized:
                normalized[category_name] = {}
            
            normalized[category_name][signal.name] = float(normalized_score)
        
        return normalized
    
    def _create_category_scores(
        self,
        normalized_signals: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Create category scores from normalized signals.
        
        For each category, average all signal scores in that category.
        
        Args:
            normalized_signals: Dictionary of category -> {signal: score}
            
        Returns:
            Dictionary of category_name -> category_score
        """
        category_scores = {}
        
        for category_name, signals in normalized_signals.items():
            if not signals:
                category_scores[category_name] = 50.0  # Neutral
                continue
            
            # Average all signals in the category
            scores = list(signals.values())
            category_score = np.mean(scores)
            category_scores[category_name] = float(category_score)
        
        return category_scores
    
    def validate_alpha_input(self, alpha_input: Dict[str, Any]) -> bool:
        """
        Validate alpha input structure.
        
        Args:
            alpha_input: Alpha input dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["symbol", "timestamp", "normalized_signals", "category_scores"]
        
        for field in required_fields:
            if field not in alpha_input:
                self._logger.error(f"Missing required field: {field}")
                return False
        
        # Validate category scores are in range
        for category, score in alpha_input["category_scores"].items():
            if not (0 <= score <= 100):
                self._logger.error(f"Category score out of range: {category}={score}")
                return False
        
        return True
    
    def get_missing_categories(
        self,
        alpha_input: Dict[str, Any],
        required_categories: List[str],
    ) -> List[str]:
        """
        Get list of missing categories.
        
        Args:
            alpha_input: Alpha input dictionary
            required_categories: List of required category names
            
        Returns:
            List of missing category names
        """
        available = set(alpha_input["category_scores"].keys())
        required = set(required_categories)
        return list(required - available)
