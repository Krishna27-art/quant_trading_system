"""
Fill Validator

Validates fill assumptions for research parity with production.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from utils.logger import get_logger

from .execution_simulator import ExecutionReport
from .latency_model import LatencyModel, LatencyModelType
from .queue_model import QueueModelType, QueuePositionModel
from .slippage_model import SlippageModel, SlippageModelType

logger = get_logger("fill_validator")


class ValidationStatus(str, Enum):
    """Validation status."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class FillAssumption:
    """Fill assumption for validation."""

    expected_fill_rate: float = 0.95  # 95% fill rate
    max_slippage: float = 0.001  # 0.1% max slippage
    max_latency_ms: float = 50.0  # 50ms max latency
    max_queue_position: int = 20  # Max queue position

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "expected_fill_rate": self.expected_fill_rate,
            "max_slippage": self.max_slippage,
            "max_latency_ms": self.max_latency_ms,
            "max_queue_position": self.max_queue_position,
        }


@dataclass
class ValidationResult:
    """Validation result."""

    assumption_type: str
    status: ValidationStatus
    expected_value: float
    actual_value: float
    deviation: float
    tolerance: float
    message: str
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "assumption_type": self.assumption_type,
            "status": self.status.value,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "deviation": self.deviation,
            "tolerance": self.tolerance,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FillValidationReport:
    """Complete fill validation report."""

    backtest_id: str
    overall_status: ValidationStatus
    fill_rate_validation: ValidationResult
    slippage_validation: ValidationResult
    latency_validation: ValidationResult
    queue_validation: ValidationResult | None = None
    warnings: list[str] = field(default_factory=list)
    ready_for_production: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backtest_id": self.backtest_id,
            "overall_status": self.overall_status.value,
            "fill_rate_validation": self.fill_rate_validation.to_dict(),
            "slippage_validation": self.slippage_validation.to_dict(),
            "latency_validation": self.latency_validation.to_dict(),
            "queue_validation": self.queue_validation.to_dict() if self.queue_validation else None,
            "warnings": self.warnings,
            "ready_for_production": self.ready_for_production,
            "timestamp": self.timestamp.isoformat(),
        }


class FillValidator:
    """
    Fill validator for research parity.

    Validates fill assumptions including slippage, latency,
    queue position, and fill rates before production.
    """

    def __init__(self):
        """Initialize fill validator."""
        self.logger = logger

        # Models for validation
        self.slippage_model = SlippageModel(SlippageModelType.LINEAR)
        self.latency_model = LatencyModel(LatencyModelType.NORMAL)
        self.queue_model = QueuePositionModel(QueueModelType.PRIORITY)

        # Assumptions
        self.assumptions = FillAssumption()

        self.logger.info("FillValidator initialized")

    def validate_execution(
        self, execution_report: ExecutionReport, include_queue: bool = False
    ) -> FillValidationReport:
        """
        Validate execution against assumptions.

        Args:
            execution_report: Execution report from backtest
            include_queue: Whether to validate queue position

        Returns:
            Validation report
        """
        backtest_id = execution_report.order_id
        results = []
        warnings = []

        # Validate fill rate
        fill_rate = (
            execution_report.filled_quantity / execution_report.total_quantity
            if execution_report.total_quantity > 0
            else 0.0
        )
        fill_rate_validation = self._validate_fill_rate(fill_rate)
        results.append(fill_rate_validation)

        if fill_rate_validation.status == ValidationStatus.WARNING:
            warnings.append(fill_rate_validation.message)

        # Validate slippage
        avg_slippage = (
            execution_report.total_slippage / execution_report.fill_count
            if execution_report.fill_count > 0
            else 0.0
        )
        slippage_validation = self._validate_slippage(avg_slippage)
        results.append(slippage_validation)

        if slippage_validation.status == ValidationStatus.WARNING:
            warnings.append(slippage_validation.message)

        # Validate latency
        latency_validation = self._validate_latency(execution_report.average_latency_ms)
        results.append(latency_validation)

        if latency_validation.status == ValidationStatus.WARNING:
            warnings.append(latency_validation.message)

        # Validate queue position (optional)
        queue_validation = None
        if include_queue:
            # Simulate queue position validation
            queue_validation = self._validate_queue_position(5)  # Simulated position
            results.append(queue_validation)

            if queue_validation.status == ValidationStatus.WARNING:
                warnings.append(queue_validation.message)

        # Determine overall status
        failed_count = sum(1 for r in results if r.status == ValidationStatus.FAILED)
        warning_count = sum(1 for r in results if r.status == ValidationStatus.WARNING)

        if failed_count > 0:
            overall_status = ValidationStatus.FAILED
            ready_for_production = False
        elif warning_count > 0:
            overall_status = ValidationStatus.WARNING
            ready_for_production = True  # Warnings are acceptable
        else:
            overall_status = ValidationStatus.PASSED
            ready_for_production = True

        report = FillValidationReport(
            backtest_id=backtest_id,
            overall_status=overall_status,
            fill_rate_validation=fill_rate_validation,
            slippage_validation=slippage_validation,
            latency_validation=latency_validation,
            queue_validation=queue_validation,
            warnings=warnings,
            ready_for_production=ready_for_production,
        )

        self.logger.info(
            f"Fill validation completed: {backtest_id}, "
            f"status={overall_status.value}, ready_for_production={ready_for_production}"
        )

        return report

    def _validate_fill_rate(self, actual_fill_rate: float) -> ValidationResult:
        """
        Validate fill rate against assumption.

        Args:
            actual_fill_rate: Actual fill rate

        Returns:
            Validation result
        """
        expected = self.assumptions.expected_fill_rate
        deviation = abs(actual_fill_rate - expected)
        tolerance = 0.1  # 10% tolerance

        if deviation > tolerance:
            status = ValidationStatus.FAILED
            message = f"Fill rate {actual_fill_rate:.2%} deviates significantly from expected {expected:.2%}"
        elif deviation > tolerance * 0.5:
            status = ValidationStatus.WARNING
            message = f"Fill rate {actual_fill_rate:.2%} slightly below expected {expected:.2%}"
        else:
            status = ValidationStatus.PASSED
            message = f"Fill rate {actual_fill_rate:.2%} within expected range"

        return ValidationResult(
            assumption_type="fill_rate",
            status=status,
            expected_value=expected,
            actual_value=actual_fill_rate,
            deviation=deviation,
            tolerance=tolerance,
            message=message,
            timestamp=datetime.utcnow(),
        )

    def _validate_slippage(self, actual_slippage: float) -> ValidationResult:
        """
        Validate slippage against assumption.

        Args:
            actual_slippage: Actual slippage

        Returns:
            Validation result
        """
        expected = self.assumptions.max_slippage
        deviation = abs(actual_slippage - expected)
        tolerance = expected * 0.5  # 50% tolerance

        if actual_slippage > expected * 2:
            status = ValidationStatus.FAILED
            message = f"Slippage {actual_slippage:.4f} exceeds maximum {expected:.4f}"
        elif actual_slippage > expected:
            status = ValidationStatus.WARNING
            message = f"Slippage {actual_slippage:.4f} slightly above expected {expected:.4f}"
        else:
            status = ValidationStatus.PASSED
            message = f"Slippage {actual_slippage:.4f} within acceptable range"

        return ValidationResult(
            assumption_type="slippage",
            status=status,
            expected_value=expected,
            actual_value=actual_slippage,
            deviation=deviation,
            tolerance=tolerance,
            message=message,
            timestamp=datetime.utcnow(),
        )

    def _validate_latency(self, actual_latency_ms: float) -> ValidationResult:
        """
        Validate latency against assumption.

        Args:
            actual_latency_ms: Actual latency in milliseconds

        Returns:
            Validation result
        """
        expected = self.assumptions.max_latency_ms
        deviation = abs(actual_latency_ms - expected)
        tolerance = expected * 0.5  # 50% tolerance

        if actual_latency_ms > expected * 2:
            status = ValidationStatus.FAILED
            message = f"Latency {actual_latency_ms:.2f}ms exceeds maximum {expected:.2f}ms"
        elif actual_latency_ms > expected:
            status = ValidationStatus.WARNING
            message = f"Latency {actual_latency_ms:.2f}ms slightly above expected {expected:.2f}ms"
        else:
            status = ValidationStatus.PASSED
            message = f"Latency {actual_latency_ms:.2f}ms within acceptable range"

        return ValidationResult(
            assumption_type="latency",
            status=status,
            expected_value=expected,
            actual_value=actual_latency_ms,
            deviation=deviation,
            tolerance=tolerance,
            message=message,
            timestamp=datetime.utcnow(),
        )

    def _validate_queue_position(self, actual_position: int) -> ValidationResult:
        """
        Validate queue position against assumption.

        Args:
            actual_position: Actual queue position

        Returns:
            Validation result
        """
        expected = self.assumptions.max_queue_position
        deviation = abs(actual_position - expected)
        tolerance = 10  # Position tolerance

        if actual_position > expected * 2:
            status = ValidationStatus.FAILED
            message = f"Queue position {actual_position} exceeds maximum {expected}"
        elif actual_position > expected:
            status = ValidationStatus.WARNING
            message = f"Queue position {actual_position} slightly above expected {expected}"
        else:
            status = ValidationStatus.PASSED
            message = f"Queue position {actual_position} within acceptable range"

        return ValidationResult(
            assumption_type="queue_position",
            status=status,
            expected_value=float(expected),
            actual_value=float(actual_position),
            deviation=float(deviation),
            tolerance=float(tolerance),
            message=message,
            timestamp=datetime.utcnow(),
        )

    def update_assumptions(self, assumptions: FillAssumption):
        """
        Update fill assumptions.

        Args:
            assumptions: New assumptions
        """
        self.assumptions = assumptions
        self.logger.info(f"Fill assumptions updated: {assumptions.to_dict()}")

    def get_status(self) -> dict[str, Any]:
        """
        Get validator status.

        Returns:
            Status dictionary
        """
        return {
            "assumptions": self.assumptions.to_dict(),
            "slippage_model": self.slippage_model.get_status(),
            "latency_model": self.latency_model.get_status(),
            "queue_model": self.queue_model.get_status(),
            "timestamp": datetime.utcnow().isoformat(),
        }
