"""
Data Infrastructure Validation Module

Institutional-grade validation framework for data ingestion.
"""

from .ingestion_validator import (
    IngestionValidator,
    InvalidDataAction,
    SchemaRegistry,
    SchemaVersion,
    create_ingestion_validator,
)

__all__ = [
    "IngestionValidator",
    "SchemaRegistry",
    "SchemaVersion",
    "InvalidDataAction",
    "create_ingestion_validator",
]
