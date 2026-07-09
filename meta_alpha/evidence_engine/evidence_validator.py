"""
Evidence Validator

Validates evidence objects and checks for consistency.
Ensures evidence is well-formed and meaningful.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

from meta_alpha.evidence_engine.evidence import Evidence, SignalDirection, EvidenceCategory
from utils.logger import get_logger

logger = get_logger("meta_alpha.evidence_engine")


@dataclass
class EvidenceValidationResult:
    """Result of evidence validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class EvidenceValidator:
    """
    Validates evidence objects and checks for consistency.
    
    Ensures:
    - All required fields are present
    - Field values are valid
    - No contradictory information
    - Evidence is meaningful
    """
    
    def __init__(self):
        """Initialize evidence validator."""
        self._logger = get_logger("meta_alpha.evidence_engine")
    
    def validate(self, evidence: Evidence) -> EvidenceValidationResult:
        """
        Validate an evidence object.
        
        Args:
            evidence: Evidence to validate
            
        Returns:
            EvidenceValidationResult
        """
        errors = []
        warnings = []
        
        # Use evidence's built-in validation
        is_valid, field_errors = evidence.validate()
        errors.extend(field_errors)
        
        # Check for contradictions
        contradiction_errors = self._check_contradictions(evidence)
        errors.extend(contradiction_errors)
        
        # Check for unlikely combinations
        unlikely_warnings = self._check_unlikely_combinations(evidence)
        warnings.extend(unlikely_warnings)
        
        # Check timestamp
        if evidence.timestamp is None:
            errors.append("Timestamp is required")
        
        # Check metadata
        if evidence.metadata is None:
            warnings.append("No metadata provided")
        
        return EvidenceValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
    
    def _check_contradictions(self, evidence: Evidence) -> List[str]:
        """
        Check for contradictory information in evidence.
        
        Args:
            evidence: Evidence to check
            
        Returns:
            List of contradiction errors
        """
        errors = []
        
        # Check if strength is high but confidence is low
        if evidence.strength > 0.8 and evidence.confidence < 0.3:
            errors.append("High strength with low confidence is contradictory")
        
        # Check if direction is neutral but strength is high
        if evidence.signal_direction == SignalDirection.NEUTRAL.value and evidence.strength > 0.5:
            errors.append("Neutral signal should not have high strength")
        
        return errors
    
    def _check_unlikely_combinations(self, evidence: Evidence) -> List[str]:
        """
        Check for unlikely (but not impossible) combinations.
        
        Args:
            evidence: Evidence to check
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Very high confidence is rare
        if evidence.confidence > 0.95:
            warnings.append(f"Very high confidence: {evidence.confidence:.2f}")
        
        # Very high strength is rare
        if evidence.strength > 0.95:
            warnings.append(f"Very high strength: {evidence.strength:.2f}")
        
        # Check category-direction mismatch
        if evidence.category == EvidenceCategory.TREND.value:
            if evidence.signal_direction == SignalDirection.NEUTRAL.value:
                warnings.append("Trend evidence is neutral - may not be useful")
        
        return warnings
    
    def validate_batch(self, evidence_list: List[Evidence]) -> Dict:
        """
        Validate multiple evidence objects.
        
        Args:
            evidence_list: List of Evidence to validate
            
        Returns:
            Summary of validation results
        """
        results = {
            "total": len(evidence_list),
            "valid": 0,
            "invalid": 0,
            "errors": [],
        }
        
        for i, evidence in enumerate(evidence_list):
            validation = self.validate(evidence)
            
            if validation.is_valid:
                results["valid"] += 1
            else:
                results["invalid"] += 1
                results["errors"].append({
                    "index": i,
                    "source": evidence.source,
                    "factor_name": evidence.factor_name,
                    "errors": validation.errors,
                })
        
        return results
    
    def check_duplicates(self, evidence_list: List[Evidence]) -> List[tuple]:
        """
        Check for duplicate evidence.
        
        Args:
            evidence_list: List of Evidence
            
        Returns:
            List of tuples (index1, index2) for duplicates
        """
        duplicates = []
        
        for i in range(len(evidence_list)):
            for j in range(i + 1, len(evidence_list)):
                if evidence_list[i].serialize() == evidence_list[j].serialize():
                    duplicates.append((i, j))
        
        return duplicates
    
    def check_coverage(self, evidence_list: List[Evidence]) -> Dict:
        """
        Check how well evidence covers different categories.
        
        Args:
            evidence_list: List of Evidence
            
        Returns:
            Dictionary with coverage statistics
        """
        # Count unique categories
        categories = set()
        sources = set()
        
        for evidence in evidence_list:
            categories.add(evidence.category)
            sources.add(evidence.source)
        
        # Count by direction
        bullish = sum(1 for e in evidence_list if e.is_bullish())
        bearish = sum(1 for e in evidence_list if e.is_bearish())
        neutral = sum(1 for e in evidence_list if e.is_neutral())
        
        return {
            "total_evidence": len(evidence_list),
            "unique_categories": len(categories),
            "unique_sources": len(sources),
            "categories": list(categories),
            "bullish": bullish,
            "bearish": bearish,
            "neutral": neutral,
        }
    
    def check_agreement(self, evidence_list: List[Evidence]) -> Dict:
        """
        Check agreement among evidence.
        
        Args:
            evidence_list: List of Evidence
            
        Returns:
            Dictionary with agreement statistics
        """
        if not evidence_list:
            return {"agreement_score": 0.0, "dominant_direction": None}
        
        # Count directions
        bullish_count = sum(1 for e in evidence_list if e.is_bullish())
        bearish_count = sum(1 for e in evidence_list if e.is_bearish())
        neutral_count = sum(1 for e in evidence_list if e.is_neutral())
        
        total = len(evidence_list)
        
        # Calculate agreement score
        max_count = max(bullish_count, bearish_count, neutral_count)
        agreement_score = max_count / total if total > 0 else 0.0
        
        # Determine dominant direction
        if bullish_count == max_count:
            dominant_direction = SignalDirection.BULLISH.value
        elif bearish_count == max_count:
            dominant_direction = SignalDirection.BEARISH.value
        else:
            dominant_direction = SignalDirection.NEUTRAL.value
        
        return {
            "agreement_score": agreement_score,
            "dominant_direction": dominant_direction,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
        }


def validate_evidence(evidence: Evidence) -> EvidenceValidationResult:
    """
    Convenience function to validate evidence.
    
    Args:
        evidence: Evidence to validate
        
    Returns:
        EvidenceValidationResult
    """
    validator = EvidenceValidator()
    return validator.validate(evidence)
