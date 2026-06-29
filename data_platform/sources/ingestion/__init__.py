"""
Ingestion Layer

Institutional-grade data ingestion with unified interface, fallback logic, and raw bronze storage.
"""

from .ingestion_engine import IngestionEngine
from .interface import IngestionResult, NSEDataSource
from .lineage import IngestionLineage
from .nselib_source import NSELibSource
from .rate_limiter import NSERateLimiter, RateLimiter, get_nse_rate_limiter
from .raw_bronze import RawBronzeLayer
from .scraper_source import ScraperSource

__all__ = [
    "NSEDataSource",
    "IngestionResult",
    "NSELibSource",
    "ScraperSource",
    "IngestionEngine",
    "RawBronzeLayer",
    "IngestionLineage",
    "RateLimiter",
    "NSERateLimiter",
    "get_nse_rate_limiter",
]
