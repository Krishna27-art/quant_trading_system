"""
Promotion Pipeline

Complete promotion pipeline from idea to production.
Idea → Implementation → Validation → Backtest → Approval → Production
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd

from utils.logger import get_logger

logger = get_logger("research.promotion_pipeline")


class PromotionStage(Enum):
    """Stages in the promotion pipeline."""
    IDEA = "idea"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    BACKTEST = "backtest"
    APPROVAL = "approval"
    PRODUCTION = "production"
    REJECTED = "rejected"


@dataclass
class PromotionRecord:
    """Record of a factor through the promotion pipeline."""
    factor_name: str
    current_stage: PromotionStage
    idea: str
    implementation_date: Optional[datetime]
    validation_date: Optional[datetime]
    backtest_date: Optional[datetime]
    approval_date: Optional[datetime]
    production_date: Optional[datetime]
    rejection_date: Optional[datetime]
    rejection_reason: Optional[str]
    validation_results: Optional[Dict]
    backtest_results: Optional[Dict]
    metadata: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "factor_name": self.factor_name,
            "current_stage": self.current_stage.value,
            "idea": self.idea,
            "implementation_date": self.implementation_date.isoformat() if self.implementation_date else None,
            "validation_date": self.validation_date.isoformat() if self.validation_date else None,
            "backtest_date": self.backtest_date.isoformat() if self.backtest_date else None,
            "approval_date": self.approval_date.isoformat() if self.approval_date else None,
            "production_date": self.production_date.isoformat() if self.production_date else None,
            "rejection_date": self.rejection_date.isoformat() if self.rejection_date else None,
            "rejection_reason": self.rejection_reason,
            "validation_results": self.validation_results,
            "backtest_results": self.backtest_results,
            "metadata": self.metadata,
        }


class PromotionPipeline:
    """
    Complete promotion pipeline from idea to production.
    
    Stages:
    1. IDEA: Initial research idea
    2. IMPLEMENTATION: Factor implementation
    3. VALIDATION: Data quality and statistical validation
    4. BACKTEST: Historical performance testing
    5. APPROVAL: Review and approval decision
    6. PRODUCTION: Deployment to production
    7. REJECTED: Factor rejected at any stage
    """
    
    def __init__(self, experiments_path: str = "research/experiments"):
        """
        Initialize promotion pipeline.
        
        Args:
            experiments_path: Path to experiments directory
        """
        from research.alpha_lab.alpha_manager import AlphaManager
        self.alpha_manager = AlphaManager(experiments_path=experiments_path)
        self._logger = get_logger("research.promotion_pipeline")
        self._records: Dict[str, PromotionRecord] = {}
    
    def submit_idea(
        self,
        factor_name: str,
        idea: str,
        metadata: Optional[Dict] = None,
    ) -> PromotionRecord:
        """
        Submit a new factor idea.
        
        Args:
            factor_name: Name of factor
            idea: Description of the idea
            metadata: Additional metadata
            
        Returns:
            PromotionRecord
        """
        record = PromotionRecord(
            factor_name=factor_name,
            current_stage=PromotionStage.IDEA,
            idea=idea,
            implementation_date=None,
            validation_date=None,
            backtest_date=None,
            approval_date=None,
            production_date=None,
            rejection_date=None,
            rejection_reason=None,
            validation_results=None,
            backtest_results=None,
            metadata=metadata or {},
        )
        
        self._records[factor_name] = record
        
        # Create experiment in AlphaManager
        self.alpha_manager.create_experiment(
            idea=idea,
            factor_name=factor_name,
            parameters=metadata.get("parameters", {}),
            dataset=metadata.get("dataset", "unknown"),
            notes="Submitted to promotion pipeline",
        )
        
        self._logger.info(f"Submitted idea for factor {factor_name}")
        return record
    
    def implement(
        self,
        factor_name: str,
        implementation_details: Dict,
    ) -> PromotionRecord:
        """
        Mark factor as implemented.
        
        Args:
            factor_name: Name of factor
            implementation_details: Implementation details
            
        Returns:
            PromotionRecord
        """
        if factor_name not in self._records:
            raise ValueError(f"Factor {factor_name} not found in pipeline")
        
        record = self._records[factor_name]
        record.current_stage = PromotionStage.IMPLEMENTATION
        record.implementation_date = datetime.now()
        record.metadata["implementation"] = implementation_details
        
        self._logger.info(f"Implemented factor {factor_name}")
        return record
    
    def validate(
        self,
        factor_name: str,
        validation_results: Dict,
    ) -> PromotionRecord:
        """
        Submit validation results.
        
        Args:
            factor_name: Name of factor
            validation_results: Validation results
            
        Returns:
            PromotionRecord
        """
        if factor_name not in self._records:
            raise ValueError(f"Factor {factor_name} not found in pipeline")
        
        record = self._records[factor_name]
        record.current_stage = PromotionStage.VALIDATION
        record.validation_date = datetime.now()
        record.validation_results = validation_results
        
        # Check if validation passed
        if not validation_results.get("passed", False):
            self.reject(factor_name, "Validation failed")
            return record
        
        self._logger.info(f"Validated factor {factor_name}")
        return record
    
    def backtest(
        self,
        factor_name: str,
        backtest_results: Dict,
    ) -> PromotionRecord:
        """
        Submit backtest results.
        
        Args:
            factor_name: Name of factor
            backtest_results: Backtest results
            
        Returns:
            PromotionRecord
        """
        if factor_name not in self._records:
            raise ValueError(f"Factor {factor_name} not found in pipeline")
        
        record = self._records[factor_name]
        record.current_stage = PromotionStage.BACKTEST
        record.backtest_date = datetime.now()
        record.backtest_results = backtest_results
        
        # Check if backtest passed
        if not backtest_results.get("passed", False):
            self.reject(factor_name, "Backtest failed")
            return record
        
        self._logger.info(f"Backtested factor {factor_name}")
        return record
    
    def approve(
        self,
        factor_name: str,
        approver: str,
        notes: str = "",
    ) -> PromotionRecord:
        """
        Approve factor for production.
        
        Args:
            factor_name: Name of factor
            approver: Person approving
            notes: Approval notes
            
        Returns:
            PromotionRecord
        """
        if factor_name not in self._records:
            raise ValueError(f"Factor {factor_name} not found in pipeline")
        
        record = self._records[factor_name]
        record.current_stage = PromotionStage.APPROVAL
        record.approval_date = datetime.now()
        record.metadata["approver"] = approver
        record.metadata["approval_notes"] = notes
        
        # Update experiment decision
        self.alpha_manager.update_decision(
            experiment_id=self._get_experiment_id(factor_name),
            decision="PROMOTE",
            notes=f"Approved by {approver}. {notes}",
        )
        
        self._logger.info(f"Approved factor {factor_name} by {approver}")
        return record
    
    def deploy_to_production(
        self,
        factor_name: str,
    ) -> PromotionRecord:
        """
        Deploy factor to production.
        
        Args:
            factor_name: Name of factor
            
        Returns:
            PromotionRecord
        """
        if factor_name not in self._records:
            raise ValueError(f"Factor {factor_name} not found in pipeline")
        
        record = self._records[factor_name]
        record.current_stage = PromotionStage.PRODUCTION
        record.production_date = datetime.now()
        
        self._logger.info(f"Deployed factor {factor_name} to production")
        return record
    
    def reject(
        self,
        factor_name: str,
        reason: str,
    ) -> PromotionRecord:
        """
        Reject factor.
        
        Args:
            factor_name: Name of factor
            reason: Rejection reason
            
        Returns:
            PromotionRecord
        """
        if factor_name not in self._records:
            raise ValueError(f"Factor {factor_name} not found in pipeline")
        
        record = self._records[factor_name]
        record.current_stage = PromotionStage.REJECTED
        record.rejection_date = datetime.now()
        record.rejection_reason = reason
        
        # Update experiment decision
        self.alpha_manager.update_decision(
            experiment_id=self._get_experiment_id(factor_name),
            decision="REJECT",
            notes=reason,
        )
        
        self._logger.warning(f"Rejected factor {factor_name}: {reason}")
        return record
    
    def _get_experiment_id(self, factor_name: str) -> str:
        """Get experiment ID for a factor."""
        experiments = self.alpha_manager.list_experiments(factor_name=factor_name)
        if experiments:
            return experiments[0].experiment_id
        return ""
    
    def get_record(self, factor_name: str) -> Optional[PromotionRecord]:
        """Get promotion record for a factor."""
        return self._records.get(factor_name)
    
    def get_pipeline_summary(self) -> pd.DataFrame:
        """
        Get summary of all factors in pipeline.
        
        Returns:
            DataFrame with pipeline summary
        """
        data = []
        
        for factor_name, record in self._records.items():
            data.append({
                "factor_name": factor_name,
                "current_stage": record.current_stage.value,
                "idea": record.idea,
                "implementation_date": record.implementation_date,
                "validation_date": record.validation_date,
                "backtest_date": record.backtest_date,
                "approval_date": record.approval_date,
                "production_date": record.production_date,
                "rejection_date": record.rejection_date,
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values("implementation_date", ascending=False)
        
        return df
    
    def auto_promote(
        self,
        discovery_result,
        min_confidence: float = 0.7,
    ) -> PromotionRecord:
        """
        Auto-promote factor based on discovery results.
        
        Args:
            discovery_result: DiscoveryResult from alpha discovery
            min_confidence: Minimum confidence for auto-promotion
            
        Returns:
            PromotionRecord
        """
        factor_name = discovery_result.factor_name
        
        # Submit idea if not in pipeline
        if factor_name not in self._records:
            self.submit_idea(
                factor_name=factor_name,
                idea=discovery_result.why,
                metadata={"discovery": discovery_result.to_dict()},
            )
        
        # Implement
        self.implement(factor_name, {"auto_generated": True})
        
        # Validate
        self.validate(factor_name, discovery_result.details)
        
        # Backtest (using discovery results)
        backtest_passed = (
            discovery_result.works and
            discovery_result.independent and
            discovery_result.confidence >= min_confidence
        )
        self.backtest(factor_name, {
            "passed": backtest_passed,
            "discovery": discovery_result.to_dict(),
        })
        
        # Auto-approve if confidence is high enough
        if backtest_passed:
            self.approve(factor_name, "auto_promote", f"Auto-promoted with confidence {discovery_result.confidence:.2f}")
            self.deploy_to_production(factor_name)
        else:
            self.reject(factor_name, f"Confidence {discovery_result.confidence:.2f} below threshold {min_confidence}")
        
        return self._records[factor_name]
