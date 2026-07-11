"""
Sentiment Signal Generator

Analyzes market sentiment from news and earnings.
Looks at news sentiment, earnings surprises, analyst ratings.
"""

from typing import Dict, List, Optional

from signal_engine.base import BaseSignalGenerator, Signal, SignalCategory, SignalDirection
from utils.logger import get_logger

logger = get_logger("signal_engine.sentiment")


class SentimentSignalGenerator(BaseSignalGenerator):
    """
    Sentiment Signal Generator.
    
    Analyzes:
    - News sentiment (positive/negative news flow)
    - Earnings surprises (actual vs expected)
    - Analyst ratings (upgrade/downgrade trends)
    - Social media sentiment (if available)
    
    Output: Positive, Neutral, or Negative with score (0-100)
    """
    
    def __init__(self):
        super().__init__(name="sentiment", category=SignalCategory.SENTIMENT)
    
    def generate(self, data: Dict[str, Dict]) -> Signal:
        """
        Generate sentiment signal from sentiment data.
        
        Args:
            data: Dictionary with sentiment data:
                  - news_sentiment: News sentiment score (-1 to 1)
                  - news_count: Number of recent news articles
                  - earnings_surprise: Earnings surprise percentage
                  - analyst_rating: Current analyst rating (1-5 scale)
                  - rating_change: Recent rating change (+1 upgrade, -1 downgrade)
                  - social_sentiment: Social media sentiment score (-1 to 1)
                  
        Returns:
            Signal object
        """
        # Extract sentiment data
        news_sentiment = data.get('news_sentiment')
        news_count = data.get('news_count', 0)
        earnings_surprise = data.get('earnings_surprise')
        analyst_rating = data.get('analyst_rating')
        rating_change = data.get('rating_change')
        social_sentiment = data.get('social_sentiment')
        
        # Analyze each indicator
        news_analysis = self._analyze_news(news_sentiment, news_count)
        earnings_analysis = self._analyze_earnings(earnings_surprise)
        analyst_analysis = self._analyze_analyst(analyst_rating, rating_change)
        social_analysis = self._analyze_social(social_sentiment)
        
        # Count bullish/bearish indicators
        bullish_count = (
            news_analysis['bullish'] +
            earnings_analysis['bullish'] +
            analyst_analysis['bullish'] +
            social_analysis['bullish']
        )
        
        bearish_count = (
            news_analysis['bearish'] +
            earnings_analysis['bearish'] +
            analyst_analysis['bearish'] +
            social_analysis['bearish']
        )
        
        neutral_count = (
            news_analysis['neutral'] +
            earnings_analysis['neutral'] +
            analyst_analysis['neutral'] +
            social_analysis['neutral']
        )
        
        # Calculate score and direction
        score, direction = self._calculate_score(bullish_count, bearish_count, neutral_count)
        
        # Adjust confidence based on data availability and volume
        confidence = self._calculate_confidence(news_count, earnings_surprise, analyst_rating, social_sentiment)
        
        # Build reason
        reason_parts = []
        if news_analysis['bullish'] > 0:
            reason_parts.append(f"Positive news flow (sentiment: {news_sentiment:.2f})")
        elif news_analysis['bearish'] > 0:
            reason_parts.append(f"Negative news flow (sentiment: {news_sentiment:.2f})")
        
        if earnings_analysis['bullish'] > 0:
            reason_parts.append(f"Positive earnings surprise ({earnings_surprise:.1f}%)")
        elif earnings_analysis['bearish'] > 0:
            reason_parts.append(f"Negative earnings surprise ({earnings_surprise:.1f}%)")
        
        if analyst_analysis['bullish'] > 0:
            reason_parts.append("Analyst upgrade")
        elif analyst_analysis['bearish'] > 0:
            reason_parts.append("Analyst downgrade")
        
        if social_analysis['bullish'] > 0:
            reason_parts.append("Positive social sentiment")
        elif social_analysis['bearish'] > 0:
            reason_parts.append("Negative social sentiment")
        
        reason = "; ".join(reason_parts) if reason_parts else "Mixed sentiment data"
        
        # Store raw values
        raw_values = {
            'news_sentiment': news_sentiment,
            'news_count': news_count,
            'earnings_surprise': earnings_surprise,
            'analyst_rating': analyst_rating,
            'rating_change': rating_change,
            'social_sentiment': social_sentiment,
        }
        
        return Signal(
            name="Sentiment",
            category=self.category,
            score=score,
            direction=direction,
            confidence=confidence,
            reason=reason,
            raw_values=raw_values,
        )
    
    def _analyze_news(self, news_sentiment: Optional[float], news_count: int) -> Dict[str, int]:
        """
        Analyze news sentiment.
        
        Sentiment > 0.3: Positive (bullish)
        Sentiment < -0.3: Negative (bearish)
        Also requires minimum news count for reliability
        """
        if news_sentiment is None or news_count < 3:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
        
        if news_sentiment > 0.3:
            return {'bullish': 1, 'bearish': 0, 'neutral': 0}
        elif news_sentiment < -0.3:
            return {'bullish': 0, 'bearish': 1, 'neutral': 0}
        else:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
    
    def _analyze_earnings(self, earnings_surprise: Optional[float]) -> Dict[str, int]:
        """
        Analyze earnings surprise.
        
        Surprise > 5%: Positive surprise (bullish)
        Surprise < -5%: Negative surprise (bearish)
        """
        if earnings_surprise is None:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
        
        if earnings_surprise > 5:
            return {'bullish': 1, 'bearish': 0, 'neutral': 0}
        elif earnings_surprise < -5:
            return {'bullish': 0, 'bearish': 1, 'neutral': 0}
        else:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
    
    def _analyze_analyst(
        self,
        analyst_rating: Optional[float],
        rating_change: Optional[int],
    ) -> Dict[str, int]:
        """
        Analyze analyst ratings.
        
        Rating >= 4: Strong buy/buy (bullish)
        Rating <= 2: Sell/strong sell (bearish)
        Rating change > 0: Upgrade (bullish)
        Rating change < 0: Downgrade (bearish)
        """
        if analyst_rating is None and rating_change is None:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
        
        bullish_count = 0
        bearish_count = 0
        
        if analyst_rating is not None:
            if analyst_rating >= 4:
                bullish_count += 1
            elif analyst_rating <= 2:
                bearish_count += 1
        
        if rating_change is not None:
            if rating_change > 0:
                bullish_count += 1
            elif rating_change < 0:
                bearish_count += 1
        
        if bullish_count > bearish_count:
            return {'bullish': 1, 'bearish': 0, 'neutral': 0}
        elif bearish_count > bullish_count:
            return {'bullish': 0, 'bearish': 1, 'neutral': 0}
        else:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
    
    def _analyze_social(self, social_sentiment: Optional[float]) -> Dict[str, int]:
        """
        Analyze social media sentiment.
        
        Sentiment > 0.3: Positive (bullish)
        Sentiment < -0.3: Negative (bearish)
        """
        if social_sentiment is None:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
        
        if social_sentiment > 0.3:
            return {'bullish': 1, 'bearish': 0, 'neutral': 0}
        elif social_sentiment < -0.3:
            return {'bullish': 0, 'bearish': 1, 'neutral': 0}
        else:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
    
    def _calculate_confidence(
        self,
        news_count: int,
        earnings_surprise: Optional[float],
        analyst_rating: Optional[float],
        social_sentiment: Optional[float],
    ) -> float:
        """Calculate confidence based on data availability and quality."""
        available_count = sum([
            news_count >= 3,
            earnings_surprise is not None,
            analyst_rating is not None,
            social_sentiment is not None,
        ])
        
        total_indicators = 4
        base_confidence = (available_count / total_indicators) * 100
        
        # Boost confidence if we have high-quality news data
        if news_count >= 10:
            base_confidence = min(100, base_confidence + 10)
        
        return base_confidence
    
    def _create_neutral_signal(self, reason: str) -> Signal:
        """Create a neutral signal when data is insufficient."""
        return Signal(
            name="Sentiment",
            category=self.category,
            score=50.0,
            direction=SignalDirection.NEUTRAL,
            confidence=0.0,
            reason=reason,
            raw_values={},
        )
