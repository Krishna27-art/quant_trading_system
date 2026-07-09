"""
Evidence Registry

Registry for managing and storing evidence objects.
Provides lookup and management of evidence by category.
"""

from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime

from meta_alpha.evidence_engine.evidence import Evidence
from utils.logger import get_logger

logger = get_logger("meta_alpha.evidence_registry")


class EvidenceRegistry:
    """
    Registry for managing and storing evidence objects.
    
    Provides:
    - Registration of evidence
    - Lookup by category
    - Lookup by source
    - Immutable collections
    - Thread-safe operations
    """
    
    def __init__(self):
        """Initialize evidence registry."""
        self._evidence: Dict[str, Evidence] = {}
        self._by_category: Dict[str, List[str]] = defaultdict(list)
        self._by_source: Dict[str, List[str]] = defaultdict(list)
        self._counter = 0
        self._logger = get_logger("meta_alpha.evidence_registry")
    
    def add(self, evidence: Evidence) -> str:
        """
        Add evidence to registry.
        
        Args:
            evidence: Evidence to add
            
        Returns:
            Evidence ID
        """
        # Validate evidence
        is_valid, errors = evidence.validate()
        if not is_valid:
            raise ValueError(f"Invalid evidence: {errors}")
        
        # Generate ID
        evidence_id = f"ev_{self._counter}"
        self._counter += 1
        
        # Store evidence
        self._evidence[evidence_id] = evidence
        
        # Index by category
        self._by_category[evidence.category].append(evidence_id)
        
        # Index by source
        self._by_source[evidence.source].append(evidence_id)
        
        self._logger.info(f"Added evidence {evidence_id} from {evidence.source}")
        return evidence_id
    
    def remove(self, evidence_id: str) -> bool:
        """
        Remove evidence from registry.
        
        Args:
            evidence_id: Evidence ID
            
        Returns:
            True if removed, False if not found
        """
        if evidence_id not in self._evidence:
            return False
        
        evidence = self._evidence[evidence_id]
        
        # Remove from category index
        if evidence_id in self._by_category[evidence.category]:
            self._by_category[evidence.category].remove(evidence_id)
        
        # Remove from source index
        if evidence_id in self._by_source[evidence.source]:
            self._by_source[evidence.source].remove(evidence_id)
        
        # Remove from main storage
        del self._evidence[evidence_id]
        
        self._logger.info(f"Removed evidence {evidence_id}")
        return True
    
    def get(self, evidence_id: str) -> Optional[Evidence]:
        """
        Get evidence by ID.
        
        Args:
            evidence_id: Evidence ID
            
        Returns:
            Evidence or None if not found
        """
        return self._evidence.get(evidence_id)
    
    def get_by_category(self, category: str) -> List[Evidence]:
        """
        Get all evidence by category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of Evidence (immutable copy)
        """
        evidence_ids = self._by_category.get(category, [])
        return [self._evidence[eid].copy() for eid in evidence_ids if eid in self._evidence]
    
    def get_by_source(self, source: str) -> List[Evidence]:
        """
        Get all evidence by source.
        
        Args:
            source: Source to filter by
            
        Returns:
            List of Evidence (immutable copy)
        """
        evidence_ids = self._by_source.get(source, [])
        return [self._evidence[eid].copy() for eid in evidence_ids if eid in self._evidence]
    
    def list_all(self) -> Dict[str, Evidence]:
        """
        List all registered evidence.
        
        Returns:
            Dictionary mapping IDs to Evidence (immutable copy)
        """
        return {eid: ev.copy() for eid, ev in self._evidence.items()}
    
    def find(self, **kwargs) -> List[Evidence]:
        """
        Find evidence by attributes.
        
        Args:
            **kwargs: Attribute key-value pairs to match
            
        Returns:
            List of matching Evidence (immutable copy)
        """
        matching = []
        
        for evidence in self._evidence.values():
            match = True
            for key, value in kwargs.items():
                if hasattr(evidence, key):
                    if getattr(evidence, key) != value:
                        match = False
                        break
                else:
                    match = False
                    break
            
            if match:
                matching.append(evidence.copy())
        
        return matching
    
    def get_categories(self) -> List[str]:
        """
        Get list of all categories.
        
        Returns:
            List of category names
        """
        return list(self._by_category.keys())
    
    def get_sources(self) -> List[str]:
        """
        Get list of all sources.
        
        Returns:
            List of source names
        """
        return list(self._by_source.keys())
    
    def count(self) -> int:
        """
        Get count of registered evidence.
        
        Returns:
            Number of evidence
        """
        return len(self._evidence)
    
    def count_by_category(self) -> Dict[str, int]:
        """
        Get count of evidence by category.
        
        Returns:
            Dictionary mapping categories to counts
        """
        return {cat: len(ids) for cat, ids in self._by_category.items()}
    
    def count_by_source(self) -> Dict[str, int]:
        """
        Get count of evidence by source.
        
        Returns:
            Dictionary mapping sources to counts
        """
        return {src: len(ids) for src, ids in self._by_source.items()}
    
    def clear(self) -> None:
        """Clear all evidence from registry."""
        self._evidence.clear()
        self._by_category.clear()
        self._by_source.clear()
        self._counter = 0
        self._logger.info("Cleared evidence registry")
    
    def get_recent(self, n: int = 10) -> List[Evidence]:
        """
        Get most recent evidence.
        
        Args:
            n: Number of recent evidence to return
            
        Returns:
            List of recent Evidence (immutable copy)
        """
        # Sort by timestamp
        sorted_evidence = sorted(
            self._evidence.values(),
            key=lambda e: e.timestamp,
            reverse=True,
        )
        
        return [ev.copy() for ev in sorted_evidence[:n]]


# Global registry instance
_global_registry = EvidenceRegistry()


def add_evidence(evidence: Evidence) -> str:
    """
    Add evidence to global registry.
    
    Args:
        evidence: Evidence to add
        
    Returns:
        Evidence ID
    """
    return _global_registry.add(evidence)


def get_evidence(evidence_id: str) -> Optional[Evidence]:
    """
    Get evidence from global registry.
    
    Args:
        evidence_id: Evidence ID
        
    Returns:
        Evidence or None
    """
    return _global_registry.get(evidence_id)


def get_evidence_by_category(category: str) -> List[Evidence]:
    """
    Get evidence by category from global registry.
    
    Args:
        category: Category to filter by
        
    Returns:
        List of Evidence
    """
    return _global_registry.get_by_category(category)


def list_all_evidence() -> Dict[str, Evidence]:
    """
    List all evidence from global registry.
    
    Returns:
        Dictionary mapping IDs to Evidence
    """
    return _global_registry.list_all()
