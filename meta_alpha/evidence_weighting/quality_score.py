"""
Quality Score

Represents the quality score for evidence.
Combines multiple quality dimensions into a single score.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

from utils.logger import get_logger

logger = get_logger("meta_alpha.evidence_weighting")


@dataclass
class QualityScore:
    """
    Represents the quality score for evidence.
    
    Dimensions:
        data_quality: Quality of underlying data (0-100)
        historical_ic: Historical Information Coefficient (0-100)
        historical_sharpe: Historical Sharpe ratio (0-100)
        regime_stability: Stability across market regimes (0-100)
        missing_values: Penalty for missing data (0-100)
        overall_score: Overall quality score (0-100)
    """
    data_quality: float
    historical_ic: float
    historical_sharpe: float
    regime_stability: float
    missing_values: float
    overall_score: float
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate quality score.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check all scores are between 0 and 100
        scores = [
            ("data_quality", self.data_quality),
            ("historical_ic", self.historical_ic),
            ("historical_sharpe", self.historical_sharpe),
            ("regime_stability", self.regime_stability),
            ("missing_values", self.missing_values),
            ("overall_score", self.overall_score),
        ]
        
        for name, value in scores:
            if not (0.0 <= value <= 100.0):
                errors.append(f"{name} must be between 0 and 100, got {value}")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            "data_quality": round(self.data_quality, 2),
            "historical_ic": round(self.historical_ic, 2),
            "historical_sharpe": round(self.historical_sharpe, 2),
            "regime_stability": round(self.regime_stability, 2),
            "missing_values": round(self.missing_values, 2),
            "overall_score": round(self.overall_score, 2),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityScore":
        """
        Create quality score from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            QualityScore object
        """
        return cls(
            data_quality=data["data_quality"],
            historical_ic=data["historical_ic"],
            historical_sharpe=data["historical_sharpe"],
            regime_stability=data["regime_stability"],
            missing_values=data["missing_values"],
            overall_score=data["overall_score"],
        )
    
    def get_quality_level(self) -> str:
        """
        Get quality level based on overall score.
        
        Returns:
            Quality level: "HIGH", "MEDIUM", "LOW"
        """
        if self.overall_score >= 70:
            return "HIGH"
        elif self.overall_score >= 40:
            return "MEDIUM"
        else:
            return "LOW"
    
    def is_high_quality(self) -> bool:
        """Check if quality is high."""
        return self.overall_score >= 70
    
    def is_low_quality(self) -> bool:
        """Check if quality is low."""
        return self.overall_score < 40


class QualityScoreBuilder:
    """
    Builder for creating quality scores.
    """
    
    def __init__(self):
        """Initialize quality score builder."""
        self._data_quality: Optional[float] = None
        self._historical_ic: Optional[float] = None
        self._historical_sharpe: Optional[float] = None
        self._regime_stability: Optional[float] = None
        self._missing_values: Optional[float] = None
        self._logger = get_logger("meta_alpha.evidence_weighting")
    
    def data_quality(self, value: float) -> "QualityScoreBuilder":
        """Set data quality score."""
        self._data_quality = value
        return self
    
    def historical_ic(self, value: float) -> "QualityScoreBuilder":
        """Set historical IC score."""
        self._historical_ic = value
        return self
    
    def historical_sharpe(self, value: float) -> "QualityScoreBuilder":
        """Set historical Sharpe score."""
        self._historical_sharpe = value
        return self
    
    def regime_stability(self, value: float) -> "QualityScoreBuilder":
        """Set regime stability score."""
        self._regime_stability = value
        return self
    
    def missing_values(self, value: float) -> "QualityScoreBuilder":
        """Set missing values score."""
        self._missing_values = value
        return self
    
    def build(self) -> QualityScore:
        """
        Build the quality score.
        
        Returns:
            QualityScore object
        """
        # Set defaults for missing values
        if self._data_quality is None:
            self._data_quality = 50.0
        if self._historical_ic is None:
            self._historical_ic = 50.0
        if self._historical_sharpe is None:
            self._historical_sharpe = 50.0
        if self._regime_stability is None:
            self._regime_stability = 50.0
        if self._missing_values is None:
            self._missing_values = 100.0  # No penalty by default
        
        # Calculate overall score (weighted average)
        weights = {
            "data_quality": 0.2,
            "historical_ic": 0.25,
            "historical_sharpe": 0.25,
            "regime_stability": 0.2,
            "missing_values": 0.1,
        }
        
        overall_score = (
            self._data_quality * weights["data_quality"] +
            self._historical_ic * weights["historical_ic"] +
            self._historical_sharpe * weights["historical_sharpe"] +
            self._regime_stability * weights["regime_stability"] +
            self._missing_values * weights["missing_values"]
        )
        
        quality_score = QualityScore(
            data_quality=self._data_quality,
            historical_ic=self._historical_ic,
            historical_sharpe=self._historical_sharpe,
            regime_stability=self._regime_stability,
            missing_values=self._missing_values,
            overall_score=overall_score,
        )
        
        # Validate
        is_valid, errors = quality_score.validate()
        if not is_valid:
            self._logger.warning(f"Built invalid quality score: {errors}")
        
        return quality_score
    
    def reset(self) -> "QualityScoreBuilder":
        """Reset builder to initial state."""
        self._data_quality = None
        self._historical_ic = None
        self._historical_sharpe = None
        self._regime_stability = None
        self._missing_values = None
        return self


def create_quality_score(
    data_quality: float = 50.0,
    historical_ic: float = 50.0,
    historical_sharpe: float = 50.0,
    regime_stability: float = 50.0,
    missing_values: float = 100.0,
) -> QualityScore:
    """
    Convenience function to create quality score.
    
    Args:
        data_quality: Data quality score
        historical_ic: Historical IC score
        historical_sharpe: Historical Sharpe score
        regime_stability: Regime stability score
        missing_values: Missing values score
        
    Returns:
        QualityScore object
    """
    builder = QualityScoreBuilder()
    return (
        builder.data_quality(data_quality)
        .historical_ic(historical_ic)
        .historical_sharpe(historical_sharpe)
        .regime_stability(regime_stability)
        .missing_values(missing_values)
        .build()
    )
