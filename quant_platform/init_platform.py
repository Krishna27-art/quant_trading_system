#!/usr/bin/env python3
"""
Institutional Alpha Discovery Platform Initializer

Creates the clean, research-first 9-domain folder hierarchy under quant_platform/.
Ensures all modules are properly initialized as Python packages.
"""

import os
import sys
from pathlib import Path

# Base platform directory
BASE_DIR = Path(__file__).resolve().parent

# Institutional 9-Domain Hierarchy
HIERARCHY = [
    "data/market_data",
    "data/options",
    "data/macro",
    "data/fundamentals",
    "data/news",
    "features/technical",
    "features/options",
    "features/sentiment",
    "features/macro",
    "features/feature_store",
    "models/lightgbm",
    "models/catboost",
    "models/xgboost",
    "models/transformer",
    "models/ensemble",
    "research/experiments",
    "research/notebooks",
    "research/feature_selection",
    "research/optimization",
    "research/validation",
    "prediction/predictor",
    "prediction/ranking",
    "prediction/calibration",
    "prediction/explanation",
    "evaluation/tracker",
    "evaluation/metrics",
    "evaluation/calibration",
    "evaluation/drift",
    "database",
    "api",
    "dashboard",
    "tests",
]

def init_platform():
    print(f"Initializing Institutional Alpha Discovery Platform at: {BASE_DIR}")
    created_dirs = 0
    created_inits = 0

    for rel_path in HIERARCHY:
        dir_path = BASE_DIR / rel_path
        dir_path.mkdir(parents=True, exist_ok=True)
        created_dirs += 1

        # Create __init__.py along the chain
        parts = rel_path.split("/")
        for i in range(1, len(parts) + 1):
            sub_pkg = BASE_DIR.joinpath(*parts[:i])
            init_file = sub_pkg / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                created_inits += 1

    # Also create top-level __init__.py
    top_init = BASE_DIR / "__init__.py"
    if not top_init.exists():
        top_init.touch()
        created_inits += 1

    print(f"✅ Created/verified {created_dirs} module directories.")
    print(f"✅ Created {created_inits} new __init__.py files.")
    print("🚀 Institutional Research Hierarchy Ready.")

if __name__ == "__main__":
    init_platform()
