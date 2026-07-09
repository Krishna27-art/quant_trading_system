"""
Factor Runner

Discovers and executes factors from the factor library.
Handles failures gracefully and returns structured results.
"""

import importlib
import inspect
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.logger import get_logger

logger = get_logger("research.factor_runner")


@dataclass
class FactorExecutionResult:
    """Result of factor execution."""
    factor_name: str
    success: bool
    execution_time: float
    output: Optional[pd.Series]
    error: Optional[str]
    validation_passed: bool
    validation_errors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "factor_name": self.factor_name,
            "success": self.success,
            "execution_time": round(self.execution_time, 4),
            "output_length": len(self.output) if self.output is not None else 0,
            "error": self.error,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
        }


class FactorRunner:
    """
    Discovers and executes factors from the factor library.
    
    Automatically discovers all factor classes in research/factor_library/
    and executes them independently with failure handling.
    """
    
    def __init__(self, factor_library_path: str = "research/factor_library"):
        """
        Initialize factor runner.
        
        Args:
            factor_library_path: Path to factor library directory
        """
        self.factor_library_path = Path(factor_library_path)
        self._factors: Dict[str, Any] = {}
        self._logger = get_logger("research.factor_runner")
    
    def discover_factors(self) -> Dict[str, Any]:
        """
        Discover all factor classes in the factor library.
        
        Returns:
            Dictionary mapping factor names to factor classes
        """
        self._factors = {}
        
        # Walk through factor library directory
        for category_dir in self.factor_library_path.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith("_"):
                continue
            
            # Import all Python files in category directory
            for py_file in category_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                
                # Import module
                module_name = f"research.factor_library.{category_dir.name}.{py_file.stem}"
                try:
                    module = importlib.import_module(module_name)
                    
                    # Find all classes that inherit from BaseFactor
                    from research.factor_library.base_factor import BaseFactor
                    
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BaseFactor) and 
                            obj is not BaseFactor):
                            factor_instance = obj()
                            self._factors[factor_instance.name] = factor_instance
                            self._logger.debug(f"Discovered factor: {factor_instance.name}")
                
                except Exception as e:
                    self._logger.warning(f"Failed to import {module_name}: {e}")
        
        self._logger.info(f"Discovered {len(self._factors)} factors")
        return self._factors
    
    def execute_factor(
        self,
        factor_name: str,
        df: pd.DataFrame,
        validate: bool = True,
    ) -> FactorExecutionResult:
        """
        Execute a single factor.
        
        Args:
            factor_name: Name of factor to execute
            df: Input DataFrame
            validate: Whether to validate output
            
        Returns:
            FactorExecutionResult
        """
        if factor_name not in self._factors:
            return FactorExecutionResult(
                factor_name=factor_name,
                success=False,
                execution_time=0.0,
                output=None,
                error=f"Factor {factor_name} not found",
                validation_passed=False,
                validation_errors=[],
            )
        
        factor = self._factors[factor_name]
        start_time = time.time()
        
        try:
            # Check required columns
            missing_cols = set(factor.required_columns) - set(df.columns)
            if missing_cols:
                return FactorExecutionResult(
                    factor_name=factor_name,
                    success=False,
                    execution_time=0.0,
                    output=None,
                    error=f"Missing required columns: {missing_cols}",
                    validation_passed=False,
                    validation_errors=[],
                )
            
            # Compute factor
            if validate:
                factor_values, validation_result = factor.compute_and_validate(df)
                validation_passed = validation_result.passed
                validation_errors = validation_result.errors
            else:
                factor_values = factor.compute(df)
                validation_passed = True
                validation_errors = []
            
            execution_time = time.time() - start_time
            
            return FactorExecutionResult(
                factor_name=factor_name,
                success=True,
                execution_time=execution_time,
                output=factor_values,
                error=None,
                validation_passed=validation_passed,
                validation_errors=validation_errors,
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            self._logger.error(f"Factor {factor_name} execution failed: {e}")
            return FactorExecutionResult(
                factor_name=factor_name,
                success=False,
                execution_time=execution_time,
                output=None,
                error=str(e),
                validation_passed=False,
                validation_errors=[],
            )
    
    def execute_all(
        self,
        df: pd.DataFrame,
        validate: bool = True,
        category_filter: Optional[str] = None,
        timeframe_filter: Optional[str] = None,
    ) -> List[FactorExecutionResult]:
        """
        Execute all discovered factors.
        
        Args:
            df: Input DataFrame
            validate: Whether to validate outputs
            category_filter: Optional filter by factor category
            timeframe_filter: Optional filter by timeframe
            
        Returns:
            List of FactorExecutionResult
        """
        if not self._factors:
            self.discover_factors()
        
        results = []
        
        for factor_name, factor in self._factors.items():
            # Apply filters
            if category_filter and factor.category != category_filter:
                continue
            if timeframe_filter and factor.timeframe != timeframe_filter:
                continue
            
            result = self.execute_factor(factor_name, df, validate=validate)
            results.append(result)
        
        # Log summary
        successful = sum(1 for r in results if r.success)
        validated = sum(1 for r in results if r.validation_passed)
        self._logger.info(
            f"Executed {len(results)} factors: {successful} successful, {validated} validated"
        )
        
        return results
    
    def get_factor_metadata(self, factor_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific factor.
        
        Args:
            factor_name: Name of factor
            
        Returns:
            Metadata dictionary or None
        """
        if factor_name not in self._factors:
            return None
        
        factor = self._factors[factor_name]
        return factor.metadata().to_dict()
    
    def list_factors(
        self,
        category_filter: Optional[str] = None,
        timeframe_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all discovered factors with metadata.
        
        Args:
            category_filter: Optional filter by factor category
            timeframe_filter: Optional filter by timeframe
            
        Returns:
            List of metadata dictionaries
        """
        if not self._factors:
            self.discover_factors()
        
        factors_list = []
        
        for factor_name, factor in self._factors.items():
            # Apply filters
            if category_filter and factor.category != category_filter:
                continue
            if timeframe_filter and factor.timeframe != timeframe_filter:
                continue
            
            factors_list.append(factor.metadata().to_dict())
        
        return factors_list


def run_factor_pipeline(
    df: pd.DataFrame,
    factor_library_path: str = "research/factor_library",
    validate: bool = True,
    category_filter: Optional[str] = None,
    timeframe_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run the complete factor pipeline.
    
    Args:
        df: Input DataFrame
        factor_library_path: Path to factor library
        validate: Whether to validate outputs
        category_filter: Optional filter by factor category
        timeframe_filter: Optional filter by timeframe
        
    Returns:
        Dictionary with execution summary and results
    """
    runner = FactorRunner(factor_library_path=factor_library_path)
    runner.discover_factors()
    
    results = runner.execute_all(
        df=df,
        validate=validate,
        category_filter=category_filter,
        timeframe_filter=timeframe_filter,
    )
    
    successful = [r for r in results if r.success]
    validated = [r for r in results if r.validation_passed]
    
    return {
        "total_factors": len(results),
        "successful": len(successful),
        "validated": len(validated),
        "failed": len(results) - len(successful),
        "results": [r.to_dict() for r in results],
    }
