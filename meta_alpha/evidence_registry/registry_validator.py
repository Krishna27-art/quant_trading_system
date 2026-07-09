"""
Registry Validator

Validates the evidence registry state and integrity.
Ensures registry consistency and data quality.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

from meta_alpha.evidence_registry.registry import EvidenceRegistry
from meta_alpha.evidence_engine.evidence import Evidence
from utils.logger import get_logger

logger = get_logger("meta_alpha.evidence_registry")


@dataclass
class RegistryValidationResult:
    """Result of registry validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "statistics": self.statistics,
        }


class RegistryValidator:
    """
    Validates the evidence registry state and integrity.
    
    Ensures:
    - Index consistency
    - No orphaned evidence
    - Valid evidence objects
    - Proper categorization
    """
    
    def __init__(self):
        """Initialize registry validator."""
        self._logger = get_logger("meta_alpha.evidence_registry")
    
    def validate(self, registry: EvidenceRegistry) -> RegistryValidationResult:
        """
        Validate the entire registry.
        
        Args:
            registry: EvidenceRegistry to validate
            
        Returns:
            RegistryValidationResult
        """
        errors = []
        warnings = []
        
        # Check index consistency
        index_errors = self._check_index_consistency(registry)
        errors.extend(index_errors)
        
        # Check for orphaned evidence
        orphan_errors = self._check_orphaned_evidence(registry)
        errors.extend(orphan_errors)
        
        # Check evidence validity
        evidence_errors = self._check_evidence_validity(registry)
        errors.extend(evidence_errors)
        
        # Check categorization
        category_warnings = self._check_categorization(registry)
        warnings.extend(category_warnings)
        
        # Check timestamps
        timestamp_warnings = self._check_timestamps(registry)
        warnings.extend(timestamp_warnings)
        
        # Collect statistics
        statistics = self._collect_statistics(registry)
        
        return RegistryValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            statistics=statistics,
        )
    
    def _check_index_consistency(self, registry: EvidenceRegistry) -> List[str]:
        """
        Check consistency between main storage and indexes.
        
        Args:
            registry: EvidenceRegistry to check
            
        Returns:
            List of error messages
        """
        errors = []
        
        # Get all evidence IDs from main storage
        main_ids = set(registry._evidence.keys())
        
        # Get all evidence IDs from category index
        category_ids = set()
        for ids in registry._by_category.values():
            category_ids.update(ids)
        
        # Get all evidence IDs from source index
        source_ids = set()
        for ids in registry._by_source.values():
            source_ids.update(ids)
        
        # Check for missing in category index
        missing_in_category = main_ids - category_ids
        if missing_in_category:
            errors.append(f"Evidence missing from category index: {missing_in_category}")
        
        # Check for missing in source index
        missing_in_source = main_ids - source_ids
        if missing_in_source:
            errors.append(f"Evidence missing from source index: {missing_in_source}")
        
        # Check for extra in category index
        extra_in_category = category_ids - main_ids
        if extra_in_category:
            errors.append(f"Orphaned evidence in category index: {extra_in_category}")
        
        # Check for extra in source index
        extra_in_source = source_ids - main_ids
        if extra_in_source:
            errors.append(f"Orphaned evidence in source index: {extra_in_source}")
        
        return errors
    
    def _check_orphaned_evidence(self, registry: EvidenceRegistry) -> List[str]:
        """
        Check for evidence with invalid references.
        
        Args:
            registry: EvidenceRegistry to check
            
        Returns:
            List of error messages
        """
        errors = []
        
        for evidence_id, evidence in registry._evidence.items():
            # Check if evidence is in correct category index
            if evidence_id not in registry._by_category.get(evidence.category, []):
                errors.append(f"Evidence {evidence_id} not in category index for {evidence.category}")
            
            # Check if evidence is in correct source index
            if evidence_id not in registry._by_source.get(evidence.source, []):
                errors.append(f"Evidence {evidence_id} not in source index for {evidence.source}")
        
        return errors
    
    def _check_evidence_validity(self, registry: EvidenceRegistry) -> List[str]:
        """
        Check validity of all evidence in registry.
        
        Args:
            registry: EvidenceRegistry to check
            
        Returns:
            List of error messages
        """
        errors = []
        
        for evidence_id, evidence in registry._evidence.items():
            is_valid, validation_errors = evidence.validate()
            if not is_valid:
                errors.append(f"Evidence {evidence_id} invalid: {validation_errors}")
        
        return errors
    
    def _check_categorization(self, registry: EvidenceRegistry) -> List[str]:
        """
        Check categorization of evidence.
        
        Args:
            registry: EvidenceRegistry to check
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Check for empty categories
        for category, ids in registry._by_category.items():
            if not ids:
                warnings.append(f"Empty category: {category}")
        
        # Check for categories with very few evidence
        for category, ids in registry._by_category.items():
            if len(ids) < 3:
                warnings.append(f"Category {category} has only {len(ids)} evidence")
        
        return warnings
    
    def _check_timestamps(self, registry: EvidenceRegistry) -> List[str]:
        """
        Check evidence timestamps.
        
        Args:
            registry: EvidenceRegistry to check
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        from datetime import datetime, timedelta
        
        now = datetime.now()
        
        for evidence_id, evidence in registry._evidence.items():
            if evidence.timestamp is None:
                warnings.append(f"Evidence {evidence_id} has no timestamp")
            else:
                # Check for very old evidence (older than 1 year)
                if (now - evidence.timestamp) > timedelta(days=365):
                    warnings.append(f"Evidence {evidence_id} is very old: {evidence.timestamp}")
                
                # Check for future timestamps
                if evidence.timestamp > now:
                    warnings.append(f"Evidence {evidence_id} has future timestamp: {evidence.timestamp}")
        
        return warnings
    
    def _collect_statistics(self, registry: EvidenceRegistry) -> Dict:
        """
        Collect registry statistics.
        
        Args:
            registry: EvidenceRegistry to analyze
            
        Returns:
            Dictionary of statistics
        """
        return {
            "total_evidence": registry.count(),
            "total_categories": len(registry.get_categories()),
            "total_sources": len(registry.get_sources()),
            "count_by_category": registry.count_by_category(),
            "count_by_source": registry.count_by_source(),
        }
    
    def validate_subset(
        self,
        registry: EvidenceRegistry,
        category: Optional[str] = None,
        source: Optional[str] = None,
    ) -> RegistryValidationResult:
        """
        Validate a subset of the registry.
        
        Args:
            registry: EvidenceRegistry to validate
            category: Optional category to filter by
            source: Optional source to filter by
            
        Returns:
            RegistryValidationResult
        """
        errors = []
        warnings = []
        
        # Get evidence subset
        if category:
            evidence_list = registry.get_by_category(category)
        elif source:
            evidence_list = registry.get_by_source(source)
        else:
            evidence_list = registry.list_all().values()
        
        # Validate each evidence
        for evidence in evidence_list:
            is_valid, validation_errors = evidence.validate()
            if not is_valid:
                errors.append(f"Evidence invalid: {validation_errors}")
        
        # Collect statistics
        statistics = {
            "subset_size": len(evidence_list),
            "filter": {"category": category, "source": source},
        }
        
        return RegistryValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            statistics=statistics,
        )


def validate_registry(registry: EvidenceRegistry) -> RegistryValidationResult:
    """
    Convenience function to validate registry.
    
    Args:
        registry: EvidenceRegistry to validate
        
    Returns:
        RegistryValidationResult
    """
    validator = RegistryValidator()
    return validator.validate(registry)
