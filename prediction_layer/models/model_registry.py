"""
Model Registry (prediction_layer) — DEPRECATED

This file is kept only to prevent ImportError on any legacy/stale import paths
that may exist in test files or research notebooks.

The LIVE ModelRegistry is in:
    prediction_intelligence/base_logistic.py → ModelRegistry (singleton pattern)

DO NOT add new code here. Do not import this in production paths.
"""

from prediction_intelligence.base_logistic import ModelRegistry  # re-export live class

__all__ = ["ModelRegistry"]
