"""
Pre-Market Environment Analyzer - Institutional Trading Module

Analyzes market environment before market open as institutional traders do:
- Global Markets (US, Europe, Asia)
- Gift Nifty (early indication of Indian market open)
- SGX/Global Futures (Risk ON/OFF)
- Dollar Index (DXY)
- USD/INR (currency impact on FII flows)
- Crude Oil (inflation and margin impact)
- Government News (RBI policy, GST, Budget, Regulations)
- Corporate News (Earnings, Orders, Acquisitions)
- FII/DII Activity (capital flow tracking)

Determines market regime: Trending, Range-bound, Volatile, News-driven
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import requests

from utils.logger import get_logger

logger = get_logger("pre_market_analyzer")


class MarketRegime(str, Enum):
    """Market regime classification for trading strategy selection."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    VOLATILE = "volatile"
    NEWS_DRIVEN = "news_driven"
    UNCERTAIN = "uncertain"


class RiskSentiment(str, Enum):
    """Global risk sentiment classification."""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    NEUTRAL = "neutral"


@dataclass
class GlobalMarketSnapshot:
    """Snapshot of global market indicators."""
    us_snp500: float
    us_snp500_change_pct: float
    us_dow_jones: float
    us_dow_change_pct: float
    us_nasdaq: float
    us_nasdaq_change_pct: float
    europe_ftse: float
    europe_ftse_change_pct: float
    europe_dax: float
    europe_dax_change_pct: float
    asia_nikkei: float
    asia_nikkei_change_pct: float
    asia_hang_seng: float
    asia_hang_seng_change_pct: float
    timestamp: datetime


@dataclass
class IndianMarketIndicators:
    """Indian market-specific indicators."""
    gift_nifty: float
    gift_nifty_change_pct: float
    nifty_futures: float
    nifty_futures_change_pct: float
    banknifty_futures: float
    banknifty_futures_change_pct: float
    usd_inr: float
    usd_inr_change_pct: float
    dollar_index: float
    dollar_index_change_pct: float
    crude_oil_wti: float
    crude_oil_change_pct: float
    crude_oil_brent: float
    crude_oil_brent_change_pct: float
    gold_price: float
    gold_change_pct: float
    timestamp: datetime


@dataclass
class NewsImpact:
    """Impact assessment of news events."""
    headline: str
    category: str  # government, corporate, sector, macro
    sentiment: str  # positive, negative, neutral
    impact_score: float  # -1.0 to 1.0
    affected_sectors: list[str]
    timestamp: datetime


@dataclass
class FIIActivity:
    """FII/DII activity tracking."""
    fii_net_buy_sell_cr: float  # Net FII investment in crores
    dii_net_buy_sell_cr: float  # Net DII investment in crores
    fii_index_derivatives_cr: float
    dii_index_derivatives_cr: float
    fii_stock_derivatives_cr: float
    dii_stock_derivatives_cr: float
    date: datetime


@dataclass
class MarketEnvironment:
    """Complete pre-market environment assessment."""
    global_markets: GlobalMarketSnapshot
    indian_indicators: IndianMarketIndicators
    risk_sentiment: RiskSentiment
    market_regime: MarketRegime
    news_impacts: list[NewsImpact]
    fii_activity: Optional[FIIActivity]
    overall_sentiment: str  # bullish, bearish, neutral
    confidence_score: float  # 0.0 to 1.0
    recommended_strategies: list[str]
    timestamp: datetime


class PreMarketAnalyzer:
    """
    Institutional-grade pre-market environment analyzer.
    
    Mimics how institutional desks assess market conditions before trading:
    1. Global market sentiment
    2. Indian market indicators (Gift Nifty, futures)
    3. Currency and commodity impact
    4. News flow analysis
    5. FII/DII capital flows
    6. Market regime determination
    """

    def __init__(self, use_mock_data: bool = True):
        """
        Initialize pre-market analyzer.
        
        Parameters
        ----------
        use_mock_data
            If True, uses mock data for development. Set False for production.
        """
        self.use_mock_data = use_mock_data
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def analyze_environment(self) -> MarketEnvironment:
        """
        Run complete pre-market analysis.
        
        Returns
        -------
        MarketEnvironment
            Complete assessment of market conditions.
        """
        logger.info("Starting pre-market environment analysis")
        
        # Gather data in parallel
        global_markets = self._fetch_global_markets()
        indian_indicators = self._fetch_indian_indicators()
        news_impacts = self._fetch_news_impacts()
        fii_activity = self._fetch_fii_activity()
        
        # Analyze risk sentiment
        risk_sentiment = self._determine_risk_sentiment(global_markets, indian_indicators)
        
        # Determine market regime
        market_regime = self._determine_market_regime(
            global_markets, indian_indicators, news_impacts
        )
        
        # Calculate overall sentiment
        overall_sentiment, confidence = self._calculate_overall_sentiment(
            global_markets, indian_indicators, risk_sentiment, news_impacts, fii_activity
        )
        
        # Recommend strategies
        strategies = self._recommend_strategies(market_regime, overall_sentiment)
        
        environment = MarketEnvironment(
            global_markets=global_markets,
            indian_indicators=indian_indicators,
            risk_sentiment=risk_sentiment,
            market_regime=market_regime,
            news_impacts=news_impacts,
            fii_activity=fii_activity,
            overall_sentiment=overall_sentiment,
            confidence_score=confidence,
            recommended_strategies=strategies,
            timestamp=datetime.now()
        )
        
        logger.info(
            f"Pre-market analysis complete: Regime={market_regime.value}, "
            f"Sentiment={overall_sentiment}, Confidence={confidence:.2f}"
        )
        
        return environment

    def _fetch_global_markets(self) -> GlobalMarketSnapshot:
        """Fetch global market data."""
        if self.use_mock_data:
            return self._mock_global_markets()
        
        try:
            # In production, integrate with real data sources (Yahoo Finance, etc.)
            # For now, return mock data
            return self._mock_global_markets()
        except Exception as e:
            logger.error(f"Failed to fetch global markets: {e}")
            return self._mock_global_markets()

    def _fetch_indian_indicators(self) -> IndianMarketIndicators:
        """Fetch Indian market-specific indicators."""
        if self.use_mock_data:
            return self._mock_indian_indicators()
        
        try:
            # In production, integrate with NSE, ICE (Gift Nifty), etc.
            return self._mock_indian_indicators()
        except Exception as e:
            logger.error(f"Failed to fetch Indian indicators: {e}")
            return self._mock_indian_indicators()

    def _fetch_news_impacts(self) -> list[NewsImpact]:
        """Fetch and analyze news impacts."""
        if self.use_mock_data:
            return self._mock_news_impacts()
        
        try:
            # In production, integrate with news APIs (Bloomberg, Reuters, etc.)
            return self._mock_news_impacts()
        except Exception as e:
            logger.error(f"Failed to fetch news impacts: {e}")
            return []

    def _fetch_fii_activity(self) -> Optional[FIIActivity]:
        """Fetch FII/DII activity data."""
        if self.use_mock_data:
            return self._mock_fii_activity()
        
        try:
            # In production, integrate with NSE FII/DII data
            return self._mock_fii_activity()
        except Exception as e:
            logger.error(f"Failed to fetch FII activity: {e}")
            return None

    def _determine_risk_sentiment(
        self, 
        global_markets: GlobalMarketSnapshot,
        indian_indicators: IndianMarketIndicators
    ) -> RiskSentiment:
        """
        Determine global risk sentiment (Risk ON/OFF).
        
        Logic:
        - Strong US markets + Weak Dollar = Risk ON
        - Weak US markets + Strong Dollar = Risk OFF
        - Crude oil up = Inflation concern (Risk OFF for India)
        - USD/INR up = FII outflow pressure (Risk OFF for India)
        """
        score = 0.0
        
        # US market impact
        if global_markets.us_snp500_change_pct > 0.5:
            score += 1.0
        elif global_markets.us_snp500_change_pct < -0.5:
            score -= 1.0
        
        # Dollar index impact (strong dollar = risk off)
        if indian_indicators.dollar_index_change_pct > 0.3:
            score -= 0.5
        elif indian_indicators.dollar_index_change_pct < -0.3:
            score += 0.5
        
        # USD/INR impact (INR weakness = risk off for India)
        if indian_indicators.usd_inr_change_pct > 0.3:
            score -= 0.5
        elif indian_indicators.usd_inr_change_pct < -0.3:
            score += 0.5
        
        # Crude oil impact (higher crude = inflation concern for India)
        if indian_indicators.crude_oil_change_pct > 1.0:
            score -= 0.5
        elif indian_indicators.crude_oil_change_pct < -1.0:
            score += 0.5
        
        # Gift Nifty as leading indicator
        if indian_indicators.gift_nifty_change_pct > 0.5:
            score += 0.5
        elif indian_indicators.gift_nifty_change_pct < -0.5:
            score -= 0.5
        
        if score > 1.0:
            return RiskSentiment.RISK_ON
        elif score < -1.0:
            return RiskSentiment.RISK_OFF
        else:
            return RiskSentiment.NEUTRAL

    def _determine_market_regime(
        self,
        global_markets: GlobalMarketSnapshot,
        indian_indicators: IndianMarketIndicators,
        news_impacts: list[NewsImpact]
    ) -> MarketRegime:
        """
        Determine market regime for strategy selection.
        
        Regimes:
        - Trending: Strong directional movement in indicators
        - Range-bound: Low volatility, no clear direction
        - Volatile: High volatility, choppy conditions
        - News-driven: Major news events dominating price action
        """
        # Check for major news events
        high_impact_news = [n for n in news_impacts if abs(n.impact_score) > 0.7]
        if high_impact_news:
            return MarketRegime.NEWS_DRIVEN
        
        # Calculate volatility from global markets
        global_volatility = (
            abs(global_markets.us_snp500_change_pct) +
            abs(global_markets.us_dow_change_pct) +
            abs(global_markets.europe_ftse_change_pct)
        ) / 3.0
        
        indian_volatility = abs(indian_indicators.gift_nifty_change_pct)
        
        # High volatility regime
        if global_volatility > 1.5 or indian_volatility > 1.5:
            return MarketRegime.VOLATILE
        
        # Trending regimes
        if indian_indicators.gift_nifty_change_pct > 0.8:
            return MarketRegime.TRENDING_UP
        elif indian_indicators.gift_nifty_change_pct < -0.8:
            return MarketRegime.TRENDING_DOWN
        
        # Range-bound
        if indian_volatility < 0.3:
            return MarketRegime.RANGE_BOUND
        
        return MarketRegime.UNCERTAIN

    def _calculate_overall_sentiment(
        self,
        global_markets: GlobalMarketSnapshot,
        indian_indicators: IndianMarketIndicators,
        risk_sentiment: RiskSentiment,
        news_impacts: list[NewsImpact],
        fii_activity: Optional[FIIActivity]
    ) -> tuple[str, float]:
        """
        Calculate overall market sentiment and confidence.
        
        Returns
        -------
        tuple[str, float]
            (sentiment_label, confidence_score)
        """
        sentiment_score = 0.0
        weight_sum = 0.0
        
        # Global markets weight: 25%
        global_score = (
            global_markets.us_snp500_change_pct +
            global_markets.us_dow_change_pct +
            global_markets.us_nasdaq_change_pct
        ) / 3.0
        sentiment_score += global_score * 0.25
        weight_sum += 0.25
        
        # Indian indicators weight: 30%
        indian_score = indian_indicators.gift_nifty_change_pct
        sentiment_score += indian_score * 0.30
        weight_sum += 0.30
        
        # Risk sentiment weight: 20%
        if risk_sentiment == RiskSentiment.RISK_ON:
            sentiment_score += 1.0 * 0.20
        elif risk_sentiment == RiskSentiment.RISK_OFF:
            sentiment_score -= 1.0 * 0.20
        weight_sum += 0.20
        
        # News impact weight: 15%
        if news_impacts:
            news_score = sum(n.impact_score for n in news_impacts) / len(news_impacts)
            sentiment_score += news_score * 0.15
        weight_sum += 0.15
        
        # FII activity weight: 10%
        if fii_activity:
            fii_score = fii_activity.fii_net_buy_sell_cr / 1000.0  # Normalize
            sentiment_score += fii_score * 0.10
        weight_sum += 0.10
        
        # Normalize
        if weight_sum > 0:
            sentiment_score /= weight_sum
        
        # Determine sentiment label
        if sentiment_score > 0.5:
            sentiment = "bullish"
        elif sentiment_score < -0.5:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        # Confidence based on data quality and agreement
        confidence = min(1.0, abs(sentiment_score) + 0.3)
        
        return sentiment, confidence

    def _recommend_strategies(
        self, 
        regime: MarketRegime, 
        sentiment: str
    ) -> list[str]:
        """
        Recommend trading strategies based on regime and sentiment.
        
        Institutional traders match strategies to market conditions:
        - Trending: Momentum, breakout, trend-following
        - Range-bound: Mean reversion, support/resistance trading
        - Volatile: Reduced size, wider stops, options strategies
        - News-driven: Event-driven, wait for clarity
        """
        strategies = []
        
        if regime == MarketRegime.TRENDING_UP:
            if sentiment == "bullish":
                strategies.extend([
                    "momentum_breakout_long",
                    "pullback_entry_long",
                    "sector_rotation_long"
                ])
            else:
                strategies.extend([
                    "selective_momentum_long",
                    "wait_for_confirmation"
                ])
        
        elif regime == MarketRegime.TRENDING_DOWN:
            if sentiment == "bearish":
                strategies.extend([
                    "momentum_breakdown_short",
                    "rally_entry_short",
                    "sector_rotation_short"
                ])
            else:
                strategies.extend([
                    "selective_momentum_short",
                    "wait_for_confirmation"
                ])
        
        elif regime == MarketRegime.RANGE_BOUND:
            strategies.extend([
                "mean_reversion_long",
                "mean_reversion_short",
                "support_resistance_trading",
                "option_selling"
            ])
        
        elif regime == MarketRegime.VOLATILE:
            strategies.extend([
                "reduced_position_sizing",
                "wider_stop_losses",
                "options_strategies",
                "wait_for_volatility_contraction"
            ])
        
        elif regime == MarketRegime.NEWS_DRIVEN:
            strategies.extend([
                "wait_for_clarity",
                "event_driven_trading",
                "fade_initial_reaction_if_extreme"
            ])
        
        else:  # UNCERTAIN
            strategies.extend([
                "reduce_exposure",
                "focus_on_liquidity",
                "wait_for_regime_clarity"
            ])
        
        return strategies

    # Mock data methods for development
    def _mock_global_markets(self) -> GlobalMarketSnapshot:
        import os
        env = os.getenv("ENV", "LOCAL")
        if env.upper() in ("LIVE", "PAPER"):
            raise RuntimeError("Fatal: Global markets mock fallback triggered in live/paper environment!")
        return GlobalMarketSnapshot(
            us_snp500=5234.18,
            us_snp500_change_pct=0.65,
            us_dow_jones=39127.80,
            us_dow_change_pct=0.52,
            us_nasdaq=16439.22,
            us_nasdaq_change_pct=0.92,
            europe_ftse=8164.12,
            europe_ftse_change_pct=0.34,
            europe_dax=18492.49,
            europe_dax_change_pct=0.28,
            asia_nikkei=38487.24,
            asia_nikkei_change_pct=-0.15,
            asia_hang_seng=17651.15,
            asia_hang_seng_change_pct=0.42,
            timestamp=datetime.now()
        )

    def _mock_indian_indicators(self) -> IndianMarketIndicators:
        import os
        env = os.getenv("ENV", "LOCAL")
        if env.upper() in ("LIVE", "PAPER"):
            raise RuntimeError("Fatal: Indian market indicators mock fallback triggered in live/paper environment!")
        return IndianMarketIndicators(
            gift_nifty=24280.50,
            gift_nifty_change_pct=0.72,
            nifty_futures=24265.00,
            nifty_futures_change_pct=0.68,
            banknifty_futures=50890.00,
            banknifty_futures_change_pct=0.45,
            usd_inr=83.42,
            usd_inr_change_pct=0.12,
            dollar_index=104.25,
            dollar_index_change_pct=-0.08,
            crude_oil_wti=78.45,
            crude_oil_change_pct=0.85,
            crude_oil_brent=82.30,
            crude_oil_brent_change_pct=0.72,
            gold_price=2340.50,
            gold_change_pct=0.15,
            timestamp=datetime.now()
        )

    def _mock_news_impacts(self) -> list[NewsImpact]:
        import os
        env = os.getenv("ENV", "LOCAL")
        if env.upper() in ("LIVE", "PAPER"):
            raise RuntimeError("Fatal: News impacts mock fallback triggered in live/paper environment!")
        return [
            NewsImpact(
                headline="RBI holds repo rate at 6.5%, focuses on inflation",
                category="government",
                sentiment="neutral",
                impact_score=0.3,
                affected_sectors=["Financial Services", "Banking"],
                timestamp=datetime.now()
            ),
            NewsImpact(
                headline="Crude oil prices surge on Middle East tensions",
                category="macro",
                sentiment="negative",
                impact_score=-0.5,
                affected_sectors=["Energy", "Automobile", "Aviation"],
                timestamp=datetime.now()
            ),
            NewsImpact(
                headline="IT sector sees strong deal wins in Q1",
                category="sector",
                sentiment="positive",
                impact_score=0.4,
                affected_sectors=["Technology"],
                timestamp=datetime.now()
            )
        ]

    def _mock_fii_activity(self) -> FIIActivity:
        import os
        env = os.getenv("ENV", "LOCAL")
        if env.upper() in ("LIVE", "PAPER"):
            raise RuntimeError("Fatal: FII activity mock fallback triggered in live/paper environment!")
        return FIIActivity(
            fii_net_buy_sell_cr=1250.50,
            dii_net_buy_sell_cr=890.25,
            fii_index_derivatives_cr=450.00,
            dii_index_derivatives_cr=320.50,
            fii_stock_derivatives_cr=-150.25,
            dii_stock_derivatives_cr=180.75,
            date=datetime.now()
        )
