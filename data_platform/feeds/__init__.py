"""
Data Feeds Module

Redundant data feed management with automatic failover and
real-time data quality validation for Indian equity markets.
"""

from data_platform.feeds.data_quality_gate import DataQualityGate
from data_platform.feeds.feed_manager import FeedManager

__all__ = [
    "FeedManager",
    "DataQualityGate",
]
