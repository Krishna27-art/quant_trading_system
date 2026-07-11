"""
Feature Laboratory

A comprehensive feature store and management system for quantitative trading.

This module provides:
- Standardized feature definitions with metadata
- Feature generation and storage
- Feature quality analysis
- Feature importance tracking
- Feature correlation analysis
- Feature versioning
- Feature testing and alpha discovery
- Dashboard API endpoints

Architecture:
    Raw Market Data
            ↓
    Feature Generator
            ↓
    Feature Database (PostgreSQL)
            ↓
    Feature Analyzer
            ↓
    Feature Ranking
            ↓
    ML Models
"""

from feature_layer.base_feature import (
    BaseFeature,
    FeatureMetadata,
    FeatureResult,
    FeatureCategory,
    Timeframe,
)

from feature_layer.feature_generator import FeatureGenerator
from feature_layer.feature_analyzer import FeatureAnalyzer
from feature_layer.feature_importance import FeatureImportanceTracker
from feature_layer.feature_correlation import FeatureCorrelationAnalyzer
from feature_layer.feature_quality import FeatureQualityScorer
from feature_layer.feature_versioning import FeatureVersionManager
from feature_layer.feature_testing import FeatureTester
from feature_layer.feature_dashboard import router as feature_dashboard_router

__all__ = [
    # Base classes
    'BaseFeature',
    'FeatureMetadata',
    'FeatureResult',
    'FeatureCategory',
    'Timeframe',
    # Core engines
    'FeatureGenerator',
    'FeatureAnalyzer',
    'FeatureImportanceTracker',
    'FeatureCorrelationAnalyzer',
    'FeatureQualityScorer',
    'FeatureVersionManager',
    'FeatureTester',
    # API
    'feature_dashboard_router',
]

__version__ = '1.0.0'
