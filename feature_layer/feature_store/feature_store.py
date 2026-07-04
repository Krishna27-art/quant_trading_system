"""
Feature Store

Centralized feature storage and retrieval.
Computes and caches features to avoid redundant calculations.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path


class FeatureStore:
    """
    Centralized feature store for all prediction features.
    
    Features are organized by category:
    - Price: OHLCV, returns, gaps, ATR, VWAP
    - Technical: EMA, SMA, MACD, RSI, ADX, CCI, Supertrend, Bollinger, etc.
    - Options: PCR, OI change, Max Pain, IV, Delta, Gamma
    - Market Breadth: Advance/Decline, Sector Strength, Relative Strength
    - Macro: USDINR, Crude, US Futures, Bond Yield, Gold, Dollar Index
    - News: Sentiment scores, impact analysis
    - Fundamentals: ROE, PE, EPS, Sales Growth, Debt, Holdings
    """
    
    def __init__(self, db_path: str = "data/features.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize feature store database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main features table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS features (
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                feature_category TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                feature_value REAL NOT NULL,
                PRIMARY KEY (symbol, timestamp, feature_category, feature_name)
            )
        """)
        
        # Feature metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_metadata (
                feature_name TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                description TEXT,
                computation_method TEXT,
                last_updated TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def store_features(
        self,
        symbol: str,
        timestamp: datetime,
        features: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Store features for a symbol at a timestamp.
        
        Args:
            symbol: Stock symbol
            timestamp: Feature timestamp
            features: Nested dict {category: {feature_name: value}}
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for category, feature_dict in features.items():
            for feature_name, value in feature_dict.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO features
                    (symbol, timestamp, feature_category, feature_name, feature_value)
                    VALUES (?, ?, ?, ?, ?)
                """, (symbol, timestamp.isoformat(), category, feature_name, value))
        
        conn.commit()
        conn.close()
    
    def get_features(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Retrieve features for a symbol within a date range.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            categories: Optional list of categories to filter
            
        Returns:
            DataFrame with features
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT timestamp, feature_category, feature_name, feature_value
            FROM features
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
        """
        params = [symbol, start_date.isoformat(), end_date.isoformat()]
        
        if categories:
            placeholders = ','.join(['?'] * len(categories))
            query += f" AND feature_category IN ({placeholders})"
            params.extend(categories)
        
        query += " ORDER BY timestamp"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            return df
        
        # Pivot to wide format
        df['feature_key'] = df['feature_category'] + '_' + df['feature_name']
        df = df.pivot(index='timestamp', columns='feature_key', values='feature_value')
        df.index = pd.to_datetime(df.index)
        
        return df
    
    def get_latest_features(
        self,
        symbol: str,
        categories: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get the latest features for a symbol.
        
        Args:
            symbol: Stock symbol
            categories: Optional list of categories to filter
            
        Returns:
            Dictionary of feature_name -> value
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT feature_category, feature_name, feature_value
            FROM features
            WHERE symbol = ? AND timestamp = (
                SELECT MAX(timestamp) FROM features WHERE symbol = ?
            )
        """
        params = [symbol, symbol]
        
        if categories:
            placeholders = ','.join(['?'] * len(categories))
            query += f" AND feature_category IN ({placeholders})"
            params.extend(categories)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return {f"{cat}_{name}": val for cat, name, val in rows}
    
    def register_feature(
        self,
        feature_name: str,
        category: str,
        description: str,
        computation_method: str
    ) -> None:
        """Register a feature in the metadata table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO feature_metadata
            (feature_name, category, description, computation_method, last_updated)
            VALUES (?, ?, ?, ?, ?)
        """, (feature_name, category, description, computation_method, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_feature_vector(
        self,
        symbol: str,
        timestamp: datetime,
        feature_names: List[str]
    ) -> np.ndarray:
        """
        Get a specific feature vector for prediction.
        
        Args:
            symbol: Stock symbol
            timestamp: Feature timestamp
            feature_names: List of feature names to retrieve
            
        Returns:
            Feature vector as numpy array
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        placeholders = ','.join(['?'] * len(feature_names))
        query = f"""
            SELECT feature_value
            FROM features
            WHERE symbol = ? AND timestamp = ? AND feature_name IN ({placeholders})
            ORDER BY 
                CASE feature_name
        """
        
        # Add ordering to match feature_names
        for i, name in enumerate(feature_names):
            query += f" WHEN '{name}' THEN {i}"
        query += " END"
        
        params = [symbol, timestamp.isoformat()] + feature_names
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return np.array([row[0] for row in rows])
