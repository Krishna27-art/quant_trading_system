"""
Condition Registry

Registry for managing and storing condition objects.
Provides lookup and management of condition definitions.
"""

from typing import Dict, List, Optional

from research.interactions.condition_engine.condition import Condition
from utils.logger import get_logger

logger = get_logger("research.interactions.condition_engine")


class ConditionRegistry:
    """
    Registry for managing and storing condition objects.
    
    Provides:
    - Registration of conditions
    - Lookup by ID
    - Search by attributes
    - Listing all conditions
    """
    
    def __init__(self):
        """Initialize condition registry."""
        self._conditions: Dict[str, Condition] = {}
        self._counter = 0
        self._logger = get_logger("research.interactions.condition_engine")
    
    def register(self, condition: Condition) -> str:
        """
        Register a condition and return its ID.
        
        Args:
            condition: Condition to register
            
        Returns:
            Condition ID
        """
        # Validate condition
        is_valid, errors = condition.validate()
        if not is_valid:
            raise ValueError(f"Invalid condition: {errors}")
        
        # Generate ID
        condition_id = f"cond_{self._counter}"
        self._counter += 1
        
        # Store condition
        self._conditions[condition_id] = condition
        
        self._logger.info(f"Registered condition {condition_id}")
        return condition_id
    
    def get(self, condition_id: str) -> Optional[Condition]:
        """
        Get condition by ID.
        
        Args:
            condition_id: Condition ID
            
        Returns:
            Condition or None if not found
        """
        return self._conditions.get(condition_id)
    
    def list_all(self) -> Dict[str, Condition]:
        """
        List all registered conditions.
        
        Returns:
            Dictionary mapping IDs to Conditions
        """
        return self._conditions.copy()
    
    def find_by_trend(self, trend: str) -> List[str]:
        """
        Find conditions by trend.
        
        Args:
            trend: Trend value
            
        Returns:
            List of condition IDs
        """
        matching = []
        for cond_id, condition in self._conditions.items():
            if condition.trend == trend:
                matching.append(cond_id)
        return matching
    
    def find_by_volatility(self, volatility: str) -> List[str]:
        """
        Find conditions by volatility.
        
        Args:
            volatility: Volatility value
            
        Returns:
            List of condition IDs
        """
        matching = []
        for cond_id, condition in self._conditions.items():
            if condition.volatility == volatility:
                matching.append(cond_id)
        return matching
    
    def find_by_sector(self, sector: str) -> List[str]:
        """
        Find conditions by sector.
        
        Args:
            sector: Sector value
            
        Returns:
            List of condition IDs
        """
        matching = []
        for cond_id, condition in self._conditions.items():
            if condition.sector == sector:
                matching.append(cond_id)
        return matching
    
    def find_by_attributes(self, **kwargs) -> List[str]:
        """
        Find conditions by multiple attributes.
        
        Args:
            **kwargs: Attribute key-value pairs to match
            
        Returns:
            List of condition IDs
        """
        matching = []
        
        for cond_id, condition in self._conditions.items():
            match = True
            for key, value in kwargs.items():
                if hasattr(condition, key):
                    if getattr(condition, key) != value:
                        match = False
                        break
                else:
                    match = False
                    break
            
            if match:
                matching.append(cond_id)
        
        return matching
    
    def remove(self, condition_id: str) -> bool:
        """
        Remove condition from registry.
        
        Args:
            condition_id: Condition ID
            
        Returns:
            True if removed, False if not found
        """
        if condition_id in self._conditions:
            del self._conditions[condition_id]
            self._logger.info(f"Removed condition {condition_id}")
            return True
        return False
    
    def count(self) -> int:
        """
        Get count of registered conditions.
        
        Returns:
            Number of conditions
        """
        return len(self._conditions)
    
    def clear(self) -> None:
        """Clear all conditions from registry."""
        self._conditions.clear()
        self._counter = 0
        self._logger.info("Cleared condition registry")
    
    def get_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all conditions.
        
        Returns:
            Dictionary mapping IDs to descriptions
        """
        descriptions = {}
        for cond_id, condition in self._conditions.items():
            descriptions[cond_id] = condition.get_description()
        return descriptions


# Global registry instance
_global_registry = ConditionRegistry()


def register_condition(condition: Condition) -> str:
    """
    Register condition in global registry.
    
    Args:
        condition: Condition to register
        
    Returns:
        Condition ID
    """
    return _global_registry.register(condition)


def get_condition(condition_id: str) -> Optional[Condition]:
    """
    Get condition from global registry.
    
    Args:
        condition_id: Condition ID
        
    Returns:
        Condition or None
    """
    return _global_registry.get(condition_id)


def list_conditions() -> Dict[str, Condition]:
    """
    List all conditions from global registry.
    
    Returns:
        Dictionary mapping IDs to Conditions
    """
    return _global_registry.list_all()
