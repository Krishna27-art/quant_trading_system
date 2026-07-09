"""
Registry Loader

Loads evidence from various sources into the registry.
Supports batch loading from different data formats.
"""

from typing import List, Dict, Any, Optional
import json
from pathlib import Path

from meta_alpha.evidence_engine.evidence import Evidence
from meta_alpha.evidence_registry.registry import EvidenceRegistry
from utils.logger import get_logger

logger = get_logger("meta_alpha.evidence_registry")


class RegistryLoader:
    """
    Loads evidence from various sources into the registry.
    
    Supports:
    - JSON files
    - Dictionary lists
    - Batch operations
    """
    
    def __init__(self, registry: Optional[EvidenceRegistry] = None):
        """
        Initialize registry loader.
        
        Args:
            registry: Optional EvidenceRegistry instance
        """
        self.registry = registry or EvidenceRegistry()
        self._logger = get_logger("meta_alpha.evidence_registry")
    
    def load_from_dict_list(self, data: List[Dict[str, Any]]) -> List[str]:
        """
        Load evidence from a list of dictionaries.
        
        Args:
            data: List of evidence dictionaries
            
        Returns:
            List of evidence IDs
        """
        evidence_ids = []
        
        for item in data:
            try:
                evidence = Evidence.deserialize(item)
                evidence_id = self.registry.add(evidence)
                evidence_ids.append(evidence_id)
            except Exception as e:
                self._logger.error(f"Failed to load evidence: {e}")
        
        self._logger.info(f"Loaded {len(evidence_ids)} evidence from dict list")
        return evidence_ids
    
    def load_from_json_file(self, file_path: str) -> List[str]:
        """
        Load evidence from a JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of evidence IDs
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of evidence objects")
        
        return self.load_from_dict_list(data)
    
    def load_from_json_string(self, json_string: str) -> List[str]:
        """
        Load evidence from a JSON string.
        
        Args:
            json_string: JSON string containing evidence list
            
        Returns:
            List of evidence IDs
        """
        data = json.loads(json_string)
        
        if not isinstance(data, list):
            raise ValueError("JSON string must contain a list of evidence objects")
        
        return self.load_from_dict_list(data)
    
    def load_evidence_batch(self, evidence_list: List[Evidence]) -> List[str]:
        """
        Load a batch of evidence objects.
        
        Args:
            evidence_list: List of Evidence objects
            
        Returns:
            List of evidence IDs
        """
        evidence_ids = []
        
        for evidence in evidence_list:
            try:
                evidence_id = self.registry.add(evidence)
                evidence_ids.append(evidence_id)
            except Exception as e:
                self._logger.error(f"Failed to add evidence: {e}")
        
        self._logger.info(f"Loaded {len(evidence_ids)} evidence from batch")
        return evidence_ids
    
    def load_from_factor_data(
        self,
        factor_data: Dict[str, Dict[str, Any]],
        default_confidence: float = 0.7,
    ) -> List[str]:
        """
        Load evidence from factor data dictionary.
        
        Args:
            factor_data: Dictionary mapping factor names to their data
            default_confidence: Default confidence for factors
            
        Returns:
            List of evidence IDs
        """
        evidence_ids = []
        
        for factor_name, data in factor_data.items():
            try:
                # Extract required fields
                source = data.get("source", factor_name)
                category = data.get("category", "momentum")
                signal_direction = data.get("signal_direction", "neutral")
                strength = data.get("strength", 0.5)
                confidence = data.get("confidence", default_confidence)
                metadata = data.get("metadata", {})
                
                evidence = Evidence(
                    source=source,
                    factor_name=factor_name,
                    category=category,
                    signal_direction=signal_direction,
                    strength=strength,
                    confidence=confidence,
                    timestamp=data.get("timestamp"),
                    metadata=metadata,
                )
                
                evidence_id = self.registry.add(evidence)
                evidence_ids.append(evidence_id)
                
            except Exception as e:
                self._logger.error(f"Failed to load factor {factor_name}: {e}")
        
        self._logger.info(f"Loaded {len(evidence_ids)} evidence from factor data")
        return evidence_ids
    
    def save_to_json_file(self, file_path: str) -> None:
        """
        Save all evidence in registry to a JSON file.
        
        Args:
            file_path: Path to save JSON file
        """
        path = Path(file_path)
        
        # Create parent directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get all evidence
        evidence_list = self.registry.list_all()
        
        # Serialize to dictionaries
        data = [ev.serialize() for ev in evidence_list.values()]
        
        # Write to file
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self._logger.info(f"Saved {len(data)} evidence to {file_path}")
    
    def export_by_category(self, category: str, file_path: str) -> None:
        """
        Export evidence by category to a JSON file.
        
        Args:
            category: Category to export
            file_path: Path to save JSON file
        """
        evidence_list = self.registry.get_by_category(category)
        
        # Serialize to dictionaries
        data = [ev.serialize() for ev in evidence_list]
        
        # Write to file
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self._logger.info(f"Exported {len(data)} evidence from category {category} to {file_path}")
    
    def export_by_source(self, source: str, file_path: str) -> None:
        """
        Export evidence by source to a JSON file.
        
        Args:
            source: Source to export
            file_path: Path to save JSON file
        """
        evidence_list = self.registry.get_by_source(source)
        
        # Serialize to dictionaries
        data = [ev.serialize() for ev in evidence_list]
        
        # Write to file
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self._logger.info(f"Exported {len(data)} evidence from source {source} to {file_path}")


def load_evidence_from_json(file_path: str) -> List[str]:
    """
    Convenience function to load evidence from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of evidence IDs
    """
    loader = RegistryLoader()
    return loader.load_from_json_file(file_path)
