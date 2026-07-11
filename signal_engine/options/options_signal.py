"""
Options Signal Generator

Analyzes options chain data for sentiment.
Looks at PCR, OI Change, IV Rank, Max Pain.
"""

from typing import Dict, Optional

from signal_engine.base import BaseSignalGenerator, Signal, SignalCategory, SignalDirection
from utils.logger import get_logger

logger = get_logger("signal_engine.options")


class OptionsSignalGenerator(BaseSignalGenerator):
    """
    Options Signal Generator.
    
    Analyzes:
    - PCR (Put-Call Ratio)
    - OI Change (Open Interest changes)
    - IV Rank (Implied Volatility Rank)
    - Max Pain (Maximum Pain price)
    
    Output: Bullish, Neutral, or Bearish with score (0-100)
    """
    
    def __init__(self):
        super().__init__(name="options", category=SignalCategory.OPTIONS)
    
    def generate(self, data: Dict[str, Dict]) -> Signal:
        """
        Generate options signal from options chain data.
        
        Args:
            data: Dictionary with options data:
                  - pcr: Put-Call Ratio
                  - oi_change: Open Interest change percentage
                  - iv_rank: Implied Volatility Rank (0-100)
                  - max_pain: Max Pain price
                  - current_price: Current underlying price
                  
        Returns:
            Signal object
        """
        # Extract options data
        pcr = data.get('pcr')
        oi_change = data.get('oi_change')
        iv_rank = data.get('iv_rank')
        max_pain = data.get('max_pain')
        current_price = data.get('current_price')
        
        # Validate required data
        if pcr is None:
            self._logger.error("Missing PCR data")
            return self._create_neutral_signal("Missing PCR data")
        
        # Analyze each indicator
        pcr_analysis = self._analyze_pcr(pcr)
        oi_analysis = self._analyze_oi_change(oi_change)
        iv_analysis = self._analyze_iv_rank(iv_rank)
        max_pain_analysis = self._analyze_max_pain(max_pain, current_price)
        
        # Count bullish/bearish indicators
        bullish_count = (
            pcr_analysis['bullish'] +
            oi_analysis['bullish'] +
            iv_analysis['bullish'] +
            max_pain_analysis['bullish']
        )
        
        bearish_count = (
            pcr_analysis['bearish'] +
            oi_analysis['bearish'] +
            iv_analysis['bearish'] +
            max_pain_analysis['bearish']
        )
        
        neutral_count = (
            pcr_analysis['neutral'] +
            oi_analysis['neutral'] +
            iv_analysis['neutral'] +
            max_pain_analysis['neutral']
        )
        
        # Calculate score and direction
        score, direction = self._calculate_score(bullish_count, bearish_count, neutral_count)
        
        # Adjust confidence based on data quality
        confidence = self._calculate_confidence(pcr, oi_change, iv_rank, max_pain)
        
        # Build reason
        reason_parts = []
        if pcr_analysis['bullish'] > 0:
            reason_parts.append(f"PCR bullish ({pcr:.2f})")
        elif pcr_analysis['bearish'] > 0:
            reason_parts.append(f"PCR bearish ({pcr:.2f})")
        
        if oi_analysis['bullish'] > 0:
            reason_parts.append(f"OI building up ({oi_change:.1f}%)")
        elif oi_analysis['bearish'] > 0:
            reason_parts.append(f"OI unwinding ({oi_change:.1f}%)")
        
        if iv_analysis['bullish'] > 0:
            reason_parts.append(f"IV Rank elevated ({iv_rank:.1f})")
        elif iv_analysis['bearish'] > 0:
            reason_parts.append(f"IV Rank low ({iv_rank:.1f})")
        
        if max_pain_analysis['bullish'] > 0:
            reason_parts.append("Price above max pain")
        elif max_pain_analysis['bearish'] > 0:
            reason_parts.append("Price below max pain")
        
        reason = "; ".join(reason_parts) if reason_parts else "Mixed options data"
        
        # Store raw values
        raw_values = {
            'pcr': pcr,
            'oi_change': oi_change,
            'iv_rank': iv_rank,
            'max_pain': max_pain,
            'current_price': current_price,
        }
        
        return Signal(
            name="Options",
            category=self.category,
            score=score,
            direction=direction,
            confidence=confidence,
            reason=reason,
            raw_values=raw_values,
        )
    
    def _analyze_pcr(self, pcr: Optional[float]) -> Dict[str, int]:
        """
        Analyze Put-Call Ratio.
        
        PCR < 1: More calls than puts (bullish)
        PCR > 1: More puts than calls (bearish)
        PCR = 1: Balanced (neutral)
        """
        if pcr is None:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
        
        if pcr < 0.8:
            return {'bullish': 1, 'bearish': 0, 'neutral': 0}
        elif pcr > 1.2:
            return {'bullish': 0, 'bearish': 1, 'neutral': 0}
        else:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
    
    def _analyze_oi_change(self, oi_change: Optional[float]) -> Dict[str, int]:
        """
        Analyze Open Interest change.
        
        Positive OI change: Building positions (bullish if price rising)
        Negative OI change: Unwinding positions (bearish)
        """
        if oi_change is None:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
        
        if oi_change > 10:
            return {'bullish': 1, 'bearish': 0, 'neutral': 0}
        elif oi_change < -10:
            return {'bullish': 0, 'bearish': 1, 'neutral': 0}
        else:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
    
    def _analyze_iv_rank(self, iv_rank: Optional[float]) -> Dict[str, int]:
        """
        Analyze Implied Volatility Rank.
        
        High IV Rank (> 70): Elevated volatility (can be bullish for options buyers)
        Low IV Rank (< 30): Low volatility (can be bearish for options sellers)
        """
        if iv_rank is None:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
        
        if iv_rank > 70:
            return {'bullish': 1, 'bearish': 0, 'neutral': 0}
        elif iv_rank < 30:
            return {'bullish': 0, 'bearish': 1, 'neutral': 0}
        else:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
    
    def _analyze_max_pain(
        self,
        max_pain: Optional[float],
        current_price: Optional[float],
    ) -> Dict[str, int]:
        """
        Analyze Max Pain price.
        
        Price above max pain: Bullish (pain for put writers)
        Price below max pain: Bearish (pain for call writers)
        """
        if max_pain is None or current_price is None:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
        
        if current_price > max_pain:
            return {'bullish': 1, 'bearish': 0, 'neutral': 0}
        elif current_price < max_pain:
            return {'bullish': 0, 'bearish': 1, 'neutral': 0}
        else:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
    
    def _calculate_confidence(
        self,
        pcr: Optional[float],
        oi_change: Optional[float],
        iv_rank: Optional[float],
        max_pain: Optional[float],
    ) -> float:
        """Calculate confidence based on data availability."""
        available_count = sum([
            pcr is not None,
            oi_change is not None,
            iv_rank is not None,
            max_pain is not None,
        ])
        
        total_indicators = 4
        confidence = (available_count / total_indicators) * 100
        
        return confidence
    
    def _create_neutral_signal(self, reason: str) -> Signal:
        """Create a neutral signal when data is insufficient."""
        return Signal(
            name="Options",
            category=self.category,
            score=50.0,
            direction=SignalDirection.NEUTRAL,
            confidence=0.0,
            reason=reason,
            raw_values={},
        )
