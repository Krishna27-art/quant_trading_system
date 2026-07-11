"""
Volume Signal Generator

Analyzes volume patterns, delivery percentage, and OBV.
Looks at Volume Spike, Delivery %, OBV, Accumulation.
"""

from typing import Dict

import numpy as np
import pandas as pd

from signal_engine.base import BaseSignalGenerator, Signal, SignalCategory, SignalDirection
from utils.logger import get_logger

logger = get_logger("signal_engine.volume")


class VolumeSignalGenerator(BaseSignalGenerator):
    """
    Volume Signal Generator.
    
    Analyzes:
    - Volume Spike (current vs average)
    - Delivery Percentage (institutional interest)
    - OBV (On-Balance Volume)
    - Accumulation/Distribution
    
    Output: Strong Buying, Normal, or Weak with score (0-100)
    """
    
    def __init__(self):
        super().__init__(name="volume", category=SignalCategory.VOLUME)
    
    def generate(self, data: Dict[str, pd.DataFrame]) -> Signal:
        """
        Generate volume signal from OHLCV data.
        
        Args:
            data: Dictionary with 'ohlcva' key containing DataFrame with columns:
                  open, high, low, close, volume, adj_close
                  
        Returns:
            Signal object
        """
        df = data.get('ohlcva')
        if df is None or df.empty:
            self._logger.error("Empty or missing OHLCV data")
            return self._create_neutral_signal("No data available")
        
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self._logger.error(f"Missing required columns: {missing_cols}")
            return self._create_neutral_signal(f"Missing columns: {missing_cols}")
        
        # Calculate volume indicators
        indicators = self._calculate_indicators(df)
        
        # Analyze each indicator
        volume_spike_analysis = self._analyze_volume_spike(indicators)
        obv_analysis = self._analyze_obv(indicators)
        accumulation_analysis = self._analyze_accumulation(indicators)
        
        # Count bullish/bearish indicators
        bullish_count = (
            volume_spike_analysis['bullish'] +
            obv_analysis['bullish'] +
            accumulation_analysis['bullish']
        )
        
        bearish_count = (
            volume_spike_analysis['bearish'] +
            obv_analysis['bearish'] +
            accumulation_analysis['bearish']
        )
        
        neutral_count = (
            volume_spike_analysis['neutral'] +
            obv_analysis['neutral'] +
            accumulation_analysis['neutral']
        )
        
        # Calculate score and direction
        score, direction = self._calculate_score(bullish_count, bearish_count, neutral_count)
        
        # Adjust confidence based on volume spike magnitude
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 2.0:
            confidence = 90
        elif volume_ratio > 1.5:
            confidence = 80
        elif volume_ratio > 1.2:
            confidence = 70
        else:
            confidence = 60
        
        # Build reason
        reason_parts = []
        if volume_spike_analysis['bullish'] > 0:
            reason_parts.append(f"Volume spike detected ({volume_ratio:.2f}x average)")
        elif volume_spike_analysis['bearish'] > 0:
            reason_parts.append("Below average volume")
        
        if obv_analysis['bullish'] > 0:
            reason_parts.append("OBV trending up")
        elif obv_analysis['bearish'] > 0:
            reason_parts.append("OBV trending down")
        
        if accumulation_analysis['bullish'] > 0:
            reason_parts.append("Accumulation pattern")
        elif accumulation_analysis['bearish'] > 0:
            reason_parts.append("Distribution pattern")
        
        reason = "; ".join(reason_parts) if reason_parts else "Normal volume activity"
        
        # Store raw values
        raw_values = {
            'volume_ratio': indicators.get('volume_ratio', 1.0),
            'obv': indicators.get('obv', 0),
            'obv_slope': indicators.get('obv_slope', 0),
            'accumulation': indicators.get('accumulation', 0),
            'current_volume': indicators.get('current_volume', 0),
            'avg_volume': indicators.get('avg_volume', 0),
        }
        
        return Signal(
            name="Volume",
            category=self.category,
            score=score,
            direction=direction,
            confidence=confidence,
            reason=reason,
            raw_values=raw_values,
        )
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume indicators."""
        close = df['close']
        volume = df['volume']
        
        # Volume ratio (current vs 20-day average)
        avg_volume = volume.rolling(20).mean()
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1.0
        
        # OBV (On-Balance Volume)
        price_change = close.diff()
        obv = (price_change > 0).astype(int) * volume - (price_change < 0).astype(int) * volume
        obv = obv.cumsum()
        
        # OBV slope (trend)
        obv_slope = obv.diff(5).iloc[-1] if len(obv) > 5 else 0
        
        # Accumulation/Distribution (simplified)
        # Using price-volume relationship
        high = df['high']
        low = df['low']
        
        # Money Flow Volume
        if (high - low).iloc[-1] > 0:
            mfv = ((close - low) / (high - low)) * volume
        else:
            mfv = pd.Series(0, index=df.index)
        
        accumulation = mfv.rolling(20).sum().iloc[-1] if len(mfv) >= 20 else 0
        
        return {
            'volume_ratio': volume_ratio,
            'obv': obv.iloc[-1] if len(obv) > 0 else 0,
            'obv_slope': obv_slope,
            'accumulation': accumulation,
            'current_volume': current_volume,
            'avg_volume': avg_volume.iloc[-1] if len(avg_volume) > 0 else 0,
        }
    
    def _analyze_volume_spike(self, indicators: Dict[str, float]) -> Dict[str, int]:
        """Analyze volume spike."""
        volume_ratio = indicators.get('volume_ratio', 1.0)
        
        if volume_ratio >= 1.5:
            return {'bullish': 1, 'bearish': 0, 'neutral': 0}
        elif volume_ratio <= 0.7:
            return {'bullish': 0, 'bearish': 1, 'neutral': 0}
        else:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
    
    def _analyze_obv(self, indicators: Dict[str, float]) -> Dict[str, int]:
        """Analyze OBV trend."""
        obv_slope = indicators.get('obv_slope', 0)
        
        if obv_slope > 0:
            return {'bullish': 1, 'bearish': 0, 'neutral': 0}
        elif obv_slope < 0:
            return {'bullish': 0, 'bearish': 1, 'neutral': 0}
        else:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
    
    def _analyze_accumulation(self, indicators: Dict[str, float]) -> Dict[str, int]:
        """Analyze accumulation/distribution."""
        accumulation = indicators.get('accumulation', 0)
        
        if accumulation > 0:
            return {'bullish': 1, 'bearish': 0, 'neutral': 0}
        elif accumulation < 0:
            return {'bullish': 0, 'bearish': 1, 'neutral': 0}
        else:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
    
    def _create_neutral_signal(self, reason: str) -> Signal:
        """Create a neutral signal when data is insufficient."""
        return Signal(
            name="Volume",
            category=self.category,
            score=50.0,
            direction=SignalDirection.NEUTRAL,
            confidence=0.0,
            reason=reason,
            raw_values={},
        )
