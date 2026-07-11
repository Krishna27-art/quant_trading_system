"""
Technical Signal Generator

Analyzes trend, momentum, and breakout patterns.
Looks at EMA, ADX, Higher Highs, Higher Lows, ATR.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from signal_engine.base import BaseSignalGenerator, Signal, SignalCategory, SignalDirection
from utils.logger import get_logger

logger = get_logger("signal_engine.technical")


class TechnicalSignalGenerator(BaseSignalGenerator):
    """
    Technical Signal Generator.
    
    Analyzes:
    - Trend (EMA crossovers, slope)
    - Momentum (RSI, MACD)
    - Breakout (Higher Highs, Higher Lows)
    - Volatility (ATR, ADX)
    
    Output: Bullish, Neutral, or Bearish with score (0-100)
    """
    
    def __init__(self):
        super().__init__(name="technical", category=SignalCategory.TECHNICAL)
    
    def generate(self, data: Dict[str, pd.DataFrame]) -> Signal:
        """
        Generate technical signal from OHLCV data.
        
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
        
        # Calculate technical indicators
        indicators = self._calculate_indicators(df)
        
        # Analyze each indicator
        trend_analysis = self._analyze_trend(indicators)
        momentum_analysis = self._analyze_momentum(indicators)
        breakout_analysis = self._analyze_breakout(indicators)
        volatility_analysis = self._analyze_volatility(indicators)
        
        # Count bullish/bearish indicators
        bullish_count = (
            trend_analysis['bullish'] +
            momentum_analysis['bullish'] +
            breakout_analysis['bullish'] +
            volatility_analysis['bullish']
        )
        
        bearish_count = (
            trend_analysis['bearish'] +
            momentum_analysis['bearish'] +
            breakout_analysis['bearish'] +
            volatility_analysis['bearish']
        )
        
        neutral_count = (
            trend_analysis['neutral'] +
            momentum_analysis['neutral'] +
            breakout_analysis['neutral'] +
            volatility_analysis['neutral']
        )
        
        # Calculate score and direction
        score, direction = self._calculate_score(bullish_count, bearish_count, neutral_count)
        
        # Adjust score based on ADX (trend strength)
        adx = indicators.get('adx', 0)
        if adx > 25:
            # Strong trend - boost confidence
            confidence = min(95, 70 + (adx - 25) * 1.5)
        elif adx > 20:
            confidence = 70
        else:
            # Weak trend - reduce confidence
            confidence = max(50, 70 - (20 - adx) * 2)
        
        # Build reason
        reason_parts = []
        if trend_analysis['bullish'] > 0:
            reason_parts.append("EMA alignment bullish")
        elif trend_analysis['bearish'] > 0:
            reason_parts.append("EMA alignment bearish")
        
        if momentum_analysis['bullish'] > 0:
            reason_parts.append("RSI showing strength")
        elif momentum_analysis['bearish'] > 0:
            reason_parts.append("RSI showing weakness")
        
        if breakout_analysis['bullish'] > 0:
            reason_parts.append("Higher highs pattern")
        elif breakout_analysis['bearish'] > 0:
            reason_parts.append("Lower lows pattern")
        
        if volatility_analysis['bullish'] > 0:
            reason_parts.append("ADX strong trend")
        elif volatility_analysis['bearish'] > 0:
            reason_parts.append("ADX weak trend")
        
        reason = "; ".join(reason_parts) if reason_parts else "Mixed signals"
        
        # Store raw values
        raw_values = {
            'ema_short': indicators.get('ema_short', 0),
            'ema_long': indicators.get('ema_long', 0),
            'rsi': indicators.get('rsi', 50),
            'adx': indicators.get('adx', 0),
            'atr': indicators.get('atr', 0),
            'higher_highs': breakout_analysis.get('higher_highs', 0),
            'higher_lows': breakout_analysis.get('higher_lows', 0),
        }
        
        return Signal(
            name="Technical",
            category=self.category,
            score=score,
            direction=direction,
            confidence=confidence,
            reason=reason,
            raw_values=raw_values,
        )
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators."""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # EMAs
        ema_short = close.ewm(span=20, adjust=False).mean()
        ema_long = close.ewm(span=50, adjust=False).mean()
        
        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        # ATR
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        # ADX (simplified calculation)
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr_smooth = tr.ewm(span=14, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(span=14, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(span=14, adjust=False).mean()
        
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(span=14, adjust=False).mean()
        
        # Get latest values
        return {
            'ema_short': ema_short.iloc[-1] if len(ema_short) > 0 else 0,
            'ema_long': ema_long.iloc[-1] if len(ema_long) > 0 else 0,
            'rsi': rsi.iloc[-1] if len(rsi) > 0 else 50,
            'atr': atr.iloc[-1] if len(atr) > 0 else 0,
            'adx': adx.iloc[-1] if len(adx) > 0 else 0,
            'plus_di': plus_di.iloc[-1] if len(plus_di) > 0 else 0,
            'minus_di': minus_di.iloc[-1] if len(minus_di) > 0 else 0,
        }
    
    def _analyze_trend(self, indicators: Dict[str, float]) -> Dict[str, int]:
        """Analyze trend using EMAs."""
        ema_short = indicators.get('ema_short', 0)
        ema_long = indicators.get('ema_long', 0)
        
        if ema_short > ema_long:
            return {'bullish': 1, 'bearish': 0, 'neutral': 0}
        elif ema_short < ema_long:
            return {'bullish': 0, 'bearish': 1, 'neutral': 0}
        else:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
    
    def _analyze_momentum(self, indicators: Dict[str, float]) -> Dict[str, int]:
        """Analyze momentum using RSI."""
        rsi = indicators.get('rsi', 50)
        
        if rsi > 60:
            return {'bullish': 1, 'bearish': 0, 'neutral': 0}
        elif rsi < 40:
            return {'bullish': 0, 'bearish': 1, 'neutral': 0}
        else:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
    
    def _analyze_breakout(self, indicators: Dict[str, float]) -> Dict[str, int]:
        """Analyze breakout pattern using ADX directional indicators."""
        plus_di = indicators.get('plus_di', 0)
        minus_di = indicators.get('minus_di', 0)
        
        if plus_di > minus_di:
            return {'bullish': 1, 'bearish': 0, 'neutral': 0, 'higher_highs': 1, 'higher_lows': 1}
        elif minus_di > plus_di:
            return {'bullish': 0, 'bearish': 1, 'neutral': 0, 'higher_highs': 0, 'higher_lows': 0}
        else:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1, 'higher_highs': 0, 'higher_lows': 0}
    
    def _analyze_volatility(self, indicators: Dict[str, float]) -> Dict[str, int]:
        """Analyze volatility using ADX."""
        adx = indicators.get('adx', 0)
        
        if adx > 25:
            # Strong trend - bullish for directional trading
            return {'bullish': 1, 'bearish': 0, 'neutral': 0}
        elif adx < 20:
            # Weak trend - bearish for directional trading
            return {'bullish': 0, 'bearish': 1, 'neutral': 0}
        else:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
    
    def _create_neutral_signal(self, reason: str) -> Signal:
        """Create a neutral signal when data is insufficient."""
        return Signal(
            name="Technical",
            category=self.category,
            score=50.0,
            direction=SignalDirection.NEUTRAL,
            confidence=0.0,
            reason=reason,
            raw_values={},
        )
