"""
Market Regime History Module

Handles storage and retrieval of historical regime data.
Provides:
- Save regime classifications with features
- Query regime by date
- Get regime history for date ranges
- Compute performance statistics by regime
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
import json
from typing import Any

import duckdb

from config.settings import DB_PATH, DB_LOCK
from regime.regime_features import RegimeFeatures
from regime.regime_rules import RegimeClassification
from utils.logger import get_logger

logger = get_logger("regime_history")


class RegimeHistoryManager:
    """
    Manager for regime history storage and retrieval.
    
    Uses DuckDB for local storage with thread-safe writes.
    """
    
    def __init__(self):
        """Initialize the regime history manager."""
        self.logger = logger
        self._ensure_schema()
    
    def _ensure_schema(self) -> None:
        """Ensure the regime history table exists."""
        with DB_LOCK:
            try:
                conn = duckdb.connect(str(DB_PATH))
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS regime_history (
                        date DATE PRIMARY KEY,
                        regime VARCHAR,
                        confidence DOUBLE,
                        trend_score DOUBLE,
                        volatility_score DOUBLE,
                        breadth_score DOUBLE,
                        institutional_score DOUBLE,
                        liquidity_score DOUBLE,
                        trend_strength VARCHAR,
                        volatility_level VARCHAR,
                        liquidity_status VARCHAR,
                        matched_rules VARCHAR,
                        features JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for common queries
                conn.execute("CREATE INDEX IF NOT EXISTS idx_regime_history_date ON regime_history(date)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_regime_history_regime ON regime_history(regime)")
                
                conn.close()
                self.logger.debug("Regime history schema ensured")
                
            except Exception as e:
                self.logger.error(f"Failed to ensure regime history schema: {e}")
                raise
    
    def save_regime(
        self,
        date: date,
        classification: RegimeClassification,
        features: RegimeFeatures,
    ) -> None:
        """
        Save regime classification and features to database.
        
        Args:
            date: Date of the regime
            classification: RegimeClassification object
            features: RegimeFeatures object
        """
        with DB_LOCK:
            try:
                conn = duckdb.connect(str(DB_PATH))
                
                # Convert matched_rules list to JSON string
                matched_rules_str = ",".join(classification.matched_rules)
                
                conn.execute("""
                    INSERT OR REPLACE INTO regime_history (
                        date, regime, confidence,
                        trend_score, volatility_score, breadth_score,
                        institutional_score, liquidity_score,
                        trend_strength, volatility_level, liquidity_status,
                        matched_rules, features
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    date,
                    classification.regime.value,
                    classification.confidence,
                    classification.trend_score,
                    classification.volatility_score,
                    classification.breadth_score,
                    classification.institutional_score,
                    classification.liquidity_score,
                    classification.trend_strength,
                    classification.volatility_level,
                    classification.liquidity_status,
                    matched_rules_str,
                    json.dumps(features.to_dict()),
                ])
                
                conn.close()
                self.logger.info(f"Saved regime for {date}: {classification.regime.value}")
                
            except Exception as e:
                self.logger.error(f"Failed to save regime for {date}: {e}")
                raise
    
    def get_regime_for_date(self, asof_date: date) -> RegimeClassification | None:
        """
        Get regime classification for a specific date.
        
        Args:
            asof_date: Date to query
            
        Returns:
            RegimeClassification if found, None otherwise
        """
        try:
            conn = duckdb.connect(str(DB_PATH))
            
            result = conn.execute("""
                SELECT 
                    regime, confidence,
                    trend_score, volatility_score, breadth_score,
                    institutional_score, liquidity_score,
                    trend_strength, volatility_level, liquidity_status,
                    matched_rules
                FROM regime_history
                WHERE date = ?
            """, [asof_date]).fetchone()
            
            conn.close()
            
            if not result:
                self.logger.warning(f"No regime found for {asof_date}")
                return None
            
            from regime.regime_rules import RegimeType
            
            # Parse matched_rules
            matched_rules = result[10].split(",") if result[10] else []
            
            return RegimeClassification(
                regime=RegimeType(result[0]),
                confidence=result[1],
                timestamp=asof_date,
                trend_score=result[2],
                volatility_score=result[3],
                breadth_score=result[4],
                institutional_score=result[5],
                liquidity_score=result[6],
                matched_rules=matched_rules,
                trend_strength=result[7],
                volatility_level=result[8],
                liquidity_status=result[9],
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get regime for {asof_date}: {e}")
            return None
    
    def get_latest_regime(self) -> RegimeClassification:
        """
        Get the most recent regime classification.
        
        Returns:
            RegimeClassification for the latest available date
        """
        try:
            conn = duckdb.connect(str(DB_PATH))
            
            result = conn.execute("""
                SELECT date, regime, confidence,
                       trend_score, volatility_score, breadth_score,
                       institutional_score, liquidity_score,
                       trend_strength, volatility_level, liquidity_status,
                       matched_rules
                FROM regime_history
                ORDER BY date DESC
                LIMIT 1
            """).fetchone()
            
            conn.close()
            
            if not result:
                self.logger.warning("No regime history found, returning default")
                # Return a default classification
                from regime.regime_rules import RegimeType
                return RegimeClassification(
                    regime=RegimeType.SIDEWAYS,
                    confidence=50.0,
                    timestamp=date.today(),
                    trend_score=0.0,
                    volatility_score=0.0,
                    breadth_score=0.0,
                    institutional_score=0.0,
                    liquidity_score=0.0,
                    matched_rules=[],
                    trend_strength="Neutral",
                    volatility_level="Normal",
                    liquidity_status="Normal",
                )
            
            from regime.regime_rules import RegimeType
            
            matched_rules = result[11].split(",") if result[11] else []
            
            return RegimeClassification(
                regime=RegimeType(result[1]),
                confidence=result[2],
                timestamp=result[0],
                trend_score=result[3],
                volatility_score=result[4],
                breadth_score=result[5],
                institutional_score=result[6],
                liquidity_score=result[7],
                matched_rules=matched_rules,
                trend_strength=result[8],
                volatility_level=result[9],
                liquidity_status=result[10],
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get latest regime: {e}")
            # Return default classification
            from regime.regime_rules import RegimeType
            return RegimeClassification(
                regime=RegimeType.SIDEWAYS,
                confidence=50.0,
                timestamp=date.today(),
                trend_score=0.0,
                volatility_score=0.0,
                breadth_score=0.0,
                institutional_score=0.0,
                liquidity_score=0.0,
                matched_rules=[],
                trend_strength="Neutral",
                volatility_level="Normal",
                liquidity_status="Normal",
            )
    
    def get_regime_history(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[RegimeClassification]:
        """
        Get regime history for a date range.
        
        Args:
            start_date: Start date (defaults to 30 days ago)
            end_date: End date (defaults to today)
            
        Returns:
            List of RegimeClassification objects
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        try:
            conn = duckdb.connect(str(DB_PATH))
            
            results = conn.execute("""
                SELECT date, regime, confidence,
                       trend_score, volatility_score, breadth_score,
                       institutional_score, liquidity_score,
                       trend_strength, volatility_level, liquidity_status,
                       matched_rules
                FROM regime_history
                WHERE date >= ? AND date <= ?
                ORDER BY date ASC
            """, [start_date, end_date]).fetchall()
            
            conn.close()
            
            from regime.regime_rules import RegimeType
            
            classifications = []
            for row in results:
                matched_rules = row[11].split(",") if row[11] else []
                classifications.append(RegimeClassification(
                    regime=RegimeType(row[1]),
                    confidence=row[2],
                    timestamp=row[0],
                    trend_score=row[3],
                    volatility_score=row[4],
                    breadth_score=row[5],
                    institutional_score=row[6],
                    liquidity_score=row[7],
                    matched_rules=matched_rules,
                    trend_strength=row[8],
                    volatility_level=row[9],
                    liquidity_status=row[10],
                ))
            
            self.logger.info(f"Retrieved {len(classifications)} regime records from {start_date} to {end_date}")
            return classifications
            
        except Exception as e:
            self.logger.error(f"Failed to get regime history: {e}")
            return []
    
    def get_regime_distribution(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict[str, Any]:
        """
        Get distribution of regimes over a period.
        
        Args:
            start_date: Start date (defaults to 30 days ago)
            end_date: End date (defaults to today)
            
        Returns:
            Dict with regime counts and percentages
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        try:
            conn = duckdb.connect(str(DB_PATH))
            
            results = conn.execute("""
                SELECT regime, COUNT(*) as count
                FROM regime_history
                WHERE date >= ? AND date <= ?
                GROUP BY regime
                ORDER BY count DESC
            """, [start_date, end_date]).fetchall()
            
            conn.close()
            
            total = sum(row[1] for row in results)
            
            distribution = {}
            for regime, count in results:
                distribution[regime] = {
                    "count": count,
                    "percentage": round(count / total * 100, 1) if total > 0 else 0,
                }
            
            self.logger.info(f"Regime distribution: {distribution}")
            return distribution
            
        except Exception as e:
            self.logger.error(f"Failed to get regime distribution: {e}")
            return {}
    
    def get_performance_stats(self) -> dict[str, Any]:
        """
        Get performance statistics by regime type.
        
        This is a placeholder - actual performance tracking requires
        integration with prediction results.
        
        Returns:
            Dict with placeholder performance stats
        """
        # TODO: Integrate with prediction results to compute actual stats
        return {
            "Strong Bull": {
                "prediction_accuracy": None,
                "total_predictions": 0,
                "correct_predictions": 0,
            },
            "Bull": {
                "prediction_accuracy": None,
                "total_predictions": 0,
                "correct_predictions": 0,
            },
            "Sideways": {
                "prediction_accuracy": None,
                "total_predictions": 0,
                "correct_predictions": 0,
            },
            "Bear": {
                "prediction_accuracy": None,
                "total_predictions": 0,
                "correct_predictions": 0,
            },
            "High Volatility": {
                "prediction_accuracy": None,
                "total_predictions": 0,
                "correct_predictions": 0,
            },
            "Event Day": {
                "prediction_accuracy": None,
                "total_predictions": 0,
                "correct_predictions": 0,
            },
        }
    
    def record_prediction_outcome(
        self,
        prediction_date: date,
        regime: str,
        was_correct: bool,
    ) -> None:
        """
        Record prediction outcome for performance tracking.
        
        Args:
            prediction_date: Date of prediction
            regime: Regime at prediction time
            was_correct: Whether prediction was correct
        """
        # TODO: Implement prediction outcome tracking
        # This would require a separate table for prediction results
        self.logger.info(f"Recording prediction outcome: {prediction_date}, {regime}, {was_correct}")
        pass
