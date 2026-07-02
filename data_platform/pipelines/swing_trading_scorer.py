"""
Institutional Swing Trading Scorer - Professional Ranking Engine

Implements institutional-grade swing trading scoring system based on:
- Market Regime (10%)
- Sector Strength (20%)
- Relative Strength vs Nifty (20%)
- Liquidity (10%)
- Trend Quality (15%)
- Volume Confirmation (10%)
- Fundamental Catalyst (10%)
- Risk/Volatility (5%)

Every stock receives a score from 0-100. Only stocks scoring 85+ qualify
for further analysis. This systematic approach reduces search space
and eliminates emotional decision-making.

Institutional workflow:
1. Analyze Nifty (market regime)
2. Analyze sectors (rotation)
3. Rank sectors (strength)
4. Scan ~2000 stocks
5. Keep only liquid stocks
6. Rank by Relative Strength
7. Check fundamentals and catalysts
8. Study charts
9. Create watchlist (10-30 stocks)
10. Wait patiently for valid entries
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

from config.universe import NSE_UNIVERSE
from utils.logger import get_logger

logger = get_logger("swing_trading_scorer")


class MarketRegime(str, Enum):
    """Market regime classification for swing trading."""
    STRONG_BULL = "strong_bull"
    WEAK_BULL = "weak_bull"
    SIDEWAYS = "sideways"
    BEAR = "bear"
    HIGH_VOLATILITY = "high_volatility"


class TrendQuality(str, Enum):
    """Trend quality classification."""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    SIDEWAYS = "sideways"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    CHOPPY = "choppy"


class EntryType(str, Enum):
    """Professional entry types."""
    BREAKOUT = "breakout"
    PULLBACK = "pullback"
    TREND_CONTINUATION = "trend_continuation"
    NONE = "none"


class CatalystType(str, Enum):
    """Catalyst types for institutional interest."""
    EARNINGS = "earnings"
    ORDER_WIN = "order_win"
    POLICY_CHANGE = "policy_change"
    SECTOR_TAILWIND = "sector_tailwind"
    PRODUCT_LAUNCH = "product_launch"
    CAPACITY_EXPANSION = "capacity_expansion"
    INDUSTRY_OUTLOOK = "industry_outlook"
    NONE = "none"


@dataclass
class RelativeStrengthMetrics:
    """Relative strength vs Nifty (not RSI)."""
    symbol: str
    nifty_return_1m: float
    stock_return_1m: float
    relative_strength_1m: float  # stock_return - nifty_return
    nifty_return_3m: float
    stock_return_3m: float
    relative_strength_3m: float
    nifty_return_6m: float
    stock_return_6m: float
    relative_strength_6m: float
    overall_rs_score: float  # 0-100
    rs_rank: int  # Rank among all stocks
    timestamp: datetime


@dataclass
class TrendQualityMetrics:
    """Trend quality analysis."""
    symbol: str
    trend_quality: TrendQuality
    higher_highs_count: int
    higher_lows_count: int
    lower_highs_count: int
    lower_lows_count: int
    trend_health_score: float  # 0-100
    avg_daily_range_pct: float
    volatility_score: float  # 0-100 (lower is better for swing)
    timestamp: datetime


@dataclass
class BaseFormation:
    """Base/consolidation pattern detection."""
    symbol: str
    has_base: bool
    base_type: str  # flat_base, ascending_triangle, descending_triangle, rectangle
    base_duration_days: int
    base_high: float
    base_low: float
    breakout_level: float
    breakout_probability: float  # 0-1
    timestamp: datetime


@dataclass
class CatalystInfo:
    """Catalyst information."""
    symbol: str
    catalyst_type: CatalystType
    catalyst_description: str
    impact_score: float  # 0-1
    time_horizon: str  # short_term, medium_term, long_term
    conviction: str  # high, medium, low
    timestamp: datetime


@dataclass
class SwingTradingScore:
    """Complete swing trading score for a stock."""
    symbol: str
    name: str
    sector: str
    
    # Component scores (0-100)
    market_regime_score: float
    sector_strength_score: float
    relative_strength_score: float
    liquidity_score: float
    trend_quality_score: float
    volume_confirmation_score: float
    catalyst_score: float
    risk_volatility_score: float
    
    # Weighted final score
    final_score: float  # 0-100
    
    # Supporting metrics
    relative_strength: Optional[RelativeStrengthMetrics]
    trend_quality: Optional[TrendQualityMetrics]
    base_formation: Optional[BaseFormation]
    catalyst: Optional[CatalystInfo]
    entry_type: EntryType
    
    # Risk metrics
    suggested_stop_loss_pct: float
    suggested_target_pct: float
    risk_reward_ratio: float
    
    # Ranking
    overall_rank: int
    qualifies: bool  # True if score >= 85
    
    timestamp: datetime


class SwingTradingScorer:
    """
    Institutional swing trading scoring engine.
    
    Scoring weights:
    - Market Regime: 10%
    - Sector Strength: 20%
    - Relative Strength vs Nifty: 20%
    - Liquidity: 10%
    - Trend Quality: 15%
    - Volume Confirmation: 10%
    - Fundamental Catalyst: 10%
    - Risk/Volatility: 5%
    """

    def __init__(
        self,
        min_score_threshold: float = 85.0,
        min_avg_daily_value_cr: float = 50.0,
        min_relative_volume: float = 1.2,
        use_mock_data: bool = True
    ):
        """
        Initialize swing trading scorer.
        
        Parameters
        ----------
        min_score_threshold
            Minimum score to qualify (default: 85)
        min_avg_daily_value_cr
            Minimum average daily turnover in crores
        min_relative_volume
            Minimum relative volume for confirmation
        use_mock_data
            Use mock data for development
        """
        self.min_score_threshold = min_score_threshold
        self.min_avg_daily_value_cr = min_avg_daily_value_cr
        self.min_relative_volume = min_relative_volume
        self.use_mock_data = use_mock_data
        
        # Scoring weights
        self.weights = {
            "market_regime": 0.10,
            "sector_strength": 0.20,
            "relative_strength": 0.20,
            "liquidity": 0.10,
            "trend_quality": 0.15,
            "volume_confirmation": 0.10,
            "catalyst": 0.10,
            "risk_volatility": 0.05
        }

    def score_universe(self) -> list[SwingTradingScore]:
        """
        Score entire NSE universe for swing trading.
        
        Returns
        -------
        list[SwingTradingScore]
            Ranked list of stocks with scores
        """
        logger.info("Starting swing trading universe scoring")
        
        # Step 1: Determine market regime
        market_regime = self._determine_market_regime()
        market_regime_score = self._score_market_regime(market_regime)
        logger.info(f"Market regime: {market_regime.value}, score: {market_regime_score}")
        
        # Step 2: Load universe
        universe = NSE_UNIVERSE
        logger.info(f"Scoring {len(universe)} stocks")
        
        # Step 3: Score each stock
        all_scores = []
        for stock in universe:
            try:
                score = self._score_stock(stock, market_regime, market_regime_score)
                all_scores.append(score)
            except Exception as e:
                logger.error(f"Failed to score {stock['symbol']}: {e}")
                continue
        
        # Step 4: Rank by final score
        all_scores.sort(key=lambda x: x.final_score, reverse=True)
        
        # Step 5: Assign ranks
        for idx, score in enumerate(all_scores, 1):
            score.overall_rank = idx
            score.qualifies = score.final_score >= self.min_score_threshold
        
        qualified_count = sum(1 for s in all_scores if s.qualifies)
        logger.info(f"Scoring complete: {qualified_count}/{len(all_scores)} stocks qualified")
        
        return all_scores

    def _score_stock(
        self,
        stock: dict,
        market_regime: MarketRegime,
        market_regime_score: float
    ) -> SwingTradingScore:
        """Score a single stock."""
        symbol = stock["symbol"]
        
        # Calculate component scores
        sector_strength_score = self._score_sector_strength(stock["sector"])
        relative_strength = self._calculate_relative_strength(symbol)
        relative_strength_score = relative_strength.overall_rs_score if relative_strength else 0
        liquidity_score = self._score_liquidity(symbol)
        trend_quality = self._analyze_trend_quality(symbol)
        trend_quality_score = trend_quality.trend_health_score if trend_quality else 0
        volume_confirmation_score = self._score_volume_confirmation(symbol)
        catalyst = self._identify_catalyst(symbol)
        catalyst_score = catalyst.impact_score * 100 if catalyst else 0
        risk_volatility_score = self._score_risk_volatility(symbol)
        
        # Calculate weighted final score
        final_score = (
            market_regime_score * self.weights["market_regime"] +
            sector_strength_score * self.weights["sector_strength"] +
            relative_strength_score * self.weights["relative_strength"] +
            liquidity_score * self.weights["liquidity"] +
            trend_quality_score * self.weights["trend_quality"] +
            volume_confirmation_score * self.weights["volume_confirmation"] +
            catalyst_score * self.weights["catalyst"] +
            risk_volatility_score * self.weights["risk_volatility"]
        )
        
        # Determine entry type
        entry_type = self._classify_entry_type(trend_quality, catalyst)
        
        # Calculate risk metrics
        stop_loss, target, rr_ratio = self._calculate_risk_metrics(
            trend_quality, volatility_score=risk_volatility_score
        )
        
        return SwingTradingScore(
            symbol=symbol,
            name=stock["name"],
            sector=stock["sector"],
            market_regime_score=market_regime_score,
            sector_strength_score=sector_strength_score,
            relative_strength_score=relative_strength_score,
            liquidity_score=liquidity_score,
            trend_quality_score=trend_quality_score,
            volume_confirmation_score=volume_confirmation_score,
            catalyst_score=catalyst_score,
            risk_volatility_score=risk_volatility_score,
            final_score=final_score,
            relative_strength=relative_strength,
            trend_quality=trend_quality,
            base_formation=None,  # To be implemented
            catalyst=catalyst,
            entry_type=entry_type,
            suggested_stop_loss_pct=stop_loss,
            suggested_target_pct=target,
            risk_reward_ratio=rr_ratio,
            overall_rank=0,  # Will be assigned after sorting
            qualifies=False,  # Will be assigned after sorting
            timestamp=datetime.now()
        )

    def _determine_market_regime(self) -> MarketRegime:
        """Determine current market regime."""
        if self.use_mock_data:
            return MarketRegime.WEAK_BULL
        
        # In production, analyze Nifty trends, volatility, breadth
        return MarketRegime.WEAK_BULL

    def _score_market_regime(self, regime: MarketRegime) -> float:
        """Score market regime (0-100)."""
        scores = {
            MarketRegime.STRONG_BULL: 100,
            MarketRegime.WEAK_BULL: 75,
            MarketRegime.SIDEWAYS: 50,
            MarketRegime.BEAR: 25,
            MarketRegime.HIGH_VOLATILITY: 30
        }
        return scores.get(regime, 50)

    def _score_sector_strength(self, sector: str) -> float:
        """Score sector strength (0-100)."""
        if self.use_mock_data:
            # Mock sector strengths
            sector_scores = {
                "Technology": 85,
                "Financial Services": 70,
                "Healthcare": 75,
                "Consumer Goods": 65,
                "Energy": 60,
                "Materials": 55,
                "Automobile": 50,
                "Capital Goods": 45,
                "Telecommunication": 40,
                "Chemicals": 35
            }
            return sector_scores.get(sector, 50)
        
        return 50

    def _calculate_relative_strength(self, symbol: str) -> Optional[RelativeStrengthMetrics]:
        """Calculate relative strength vs Nifty (not RSI)."""
        if self.use_mock_data:
            import random
            
            nifty_1m = random.uniform(2.0, 6.0)
            stock_1m = random.uniform(-5.0, 20.0)
            rs_1m = stock_1m - nifty_1m
            
            nifty_3m = random.uniform(5.0, 15.0)
            stock_3m = random.uniform(-10.0, 35.0)
            rs_3m = stock_3m - nifty_3m
            
            nifty_6m = random.uniform(10.0, 25.0)
            stock_6m = random.uniform(-15.0, 50.0)
            rs_6m = stock_6m - nifty_6m
            
            # Overall RS score (weighted average)
            overall_rs = (rs_1m * 0.5 + rs_3m * 0.3 + rs_6m * 0.2)
            # Normalize to 0-100
            overall_rs_score = min(100, max(0, (overall_rs + 10) * 5))
            
            return RelativeStrengthMetrics(
                symbol=symbol,
                nifty_return_1m=nifty_1m,
                stock_return_1m=stock_1m,
                relative_strength_1m=rs_1m,
                nifty_return_3m=nifty_3m,
                stock_return_3m=stock_3m,
                relative_strength_3m=rs_3m,
                nifty_return_6m=nifty_6m,
                stock_return_6m=stock_6m,
                relative_strength_6m=rs_6m,
                overall_rs_score=overall_rs_score,
                rs_rank=0,  # Will be assigned
                timestamp=datetime.now()
            )
        
        return None

    def _score_liquidity(self, symbol: str) -> float:
        """Score liquidity (0-100)."""
        if self.use_mock_data:
            import random
            avg_value = random.uniform(10.0, 500.0)
            
            # Score based on average daily value
            if avg_value >= 200:
                return 100
            elif avg_value >= 100:
                return 80
            elif avg_value >= 50:
                return 60
            elif avg_value >= 20:
                return 40
            else:
                return 20
        
        return 50

    def _analyze_trend_quality(self, symbol: str) -> Optional[TrendQualityMetrics]:
        """Analyze trend quality (higher highs/higher lows)."""
        if self.use_mock_data:
            import random
            
            # Generate mock trend data
            higher_highs = random.randint(0, 10)
            higher_lows = random.randint(0, 10)
            lower_highs = random.randint(0, 5)
            lower_lows = random.randint(0, 5)
            
            # Determine trend quality
            if higher_highs >= 5 and higher_lows >= 4:
                trend = TrendQuality.STRONG_UPTREND
                health = 90 + random.uniform(-5, 5)
            elif higher_highs >= 3 and higher_lows >= 2:
                trend = TrendQuality.WEAK_UPTREND
                health = 70 + random.uniform(-10, 10)
            elif lower_highs >= 5 and lower_lows >= 4:
                trend = TrendQuality.STRONG_DOWNTREND
                health = 90 + random.uniform(-5, 5)
            elif lower_highs >= 3 and lower_lows >= 2:
                trend = TrendQuality.WEAK_DOWNTREND
                health = 70 + random.uniform(-10, 10)
            else:
                trend = TrendQuality.CHOPPY
                health = 40 + random.uniform(-10, 10)
            
            avg_range = random.uniform(1.0, 5.0)
            volatility_score = min(100, max(0, 100 - (avg_range * 20)))
            
            return TrendQualityMetrics(
                symbol=symbol,
                trend_quality=trend,
                higher_highs_count=higher_highs,
                higher_lows_count=higher_lows,
                lower_highs_count=lower_highs,
                lower_lows_count=lower_lows,
                trend_health_score=health,
                avg_daily_range_pct=avg_range,
                volatility_score=volatility_score,
                timestamp=datetime.now()
            )
        
        return None

    def _score_volume_confirmation(self, symbol: str) -> float:
        """Score volume confirmation (0-100)."""
        if self.use_mock_data:
            import random
            relative_volume = random.uniform(0.5, 3.0)
            
            if relative_volume >= 2.0:
                return 100
            elif relative_volume >= 1.5:
                return 80
            elif relative_volume >= 1.2:
                return 60
            elif relative_volume >= 1.0:
                return 40
            else:
                return 20
        
        return 50

    def _identify_catalyst(self, symbol: str) -> Optional[CatalystInfo]:
        """Identify catalyst for stock movement."""
        if self.use_mock_data:
            import random
            
            # 30% chance of having a catalyst
            if random.random() < 0.3:
                catalyst_types = [
                    CatalystType.EARNINGS,
                    CatalystType.ORDER_WIN,
                    CatalystType.POLICY_CHANGE,
                    CatalystType.SECTOR_TAILWIND,
                    CatalystType.PRODUCT_LAUNCH
                ]
                
                catalyst_type = random.choice(catalyst_types)
                impact_score = random.uniform(0.5, 0.9)
                
                descriptions = {
                    CatalystType.EARNINGS: "Strong quarterly earnings beat expectations",
                    CatalystType.ORDER_WIN: "Large order win from major client",
                    CatalystType.POLICY_CHANGE: "Government policy change favors sector",
                    CatalystType.SECTOR_TAILWIND: "Strong sector tailwinds driving demand",
                    CatalystType.PRODUCT_LAUNCH: "New product launch with strong market potential"
                }
                
                return CatalystInfo(
                    symbol=symbol,
                    catalyst_type=catalyst_type,
                    catalyst_description=descriptions[catalyst_type],
                    impact_score=impact_score,
                    time_horizon=random.choice(["short_term", "medium_term", "long_term"]),
                    conviction=random.choice(["high", "medium", "low"]),
                    timestamp=datetime.now()
                )
        
        return None

    def _score_risk_volatility(self, symbol: str) -> float:
        """Score risk/volatility (0-100, higher is better for swing)."""
        if self.use_mock_data:
            import random
            volatility = random.uniform(15.0, 45.0)
            
            # Moderate volatility is better for swing trading
            if 20 <= volatility <= 30:
                return 90
            elif 15 <= volatility < 20 or 30 < volatility <= 35:
                return 75
            elif 35 < volatility <= 40:
                return 50
            else:
                return 30
        
        return 50

    def _classify_entry_type(
        self,
        trend_quality: Optional[TrendQualityMetrics],
        catalyst: Optional[CatalystInfo]
    ) -> EntryType:
        """Classify optimal entry type."""
        if not trend_quality:
            return EntryType.NONE
        
        if trend_quality.trend_quality in [TrendQuality.STRONG_UPTREND, TrendQuality.WEAK_UPTREND]:
            if catalyst and catalyst.impact_score > 0.7:
                return EntryType.BREAKOUT
            else:
                return EntryType.TREND_CONTINUATION
        elif trend_quality.trend_quality == TrendQuality.SIDEWAYS:
            return EntryType.PULLBACK
        else:
            return EntryType.NONE

    def _calculate_risk_metrics(
        self,
        trend_quality: Optional[TrendQualityMetrics],
        volatility_score: float
    ) -> tuple[float, float, float]:
        """Calculate suggested stop loss, target, and risk-reward ratio."""
        if not trend_quality:
            return 5.0, 10.0, 2.0
        
        # Stop loss based on volatility
        if volatility_score >= 80:
            stop_loss = 3.0
        elif volatility_score >= 60:
            stop_loss = 4.0
        elif volatility_score >= 40:
            stop_loss = 5.0
        else:
            stop_loss = 6.0
        
        # Target based on trend quality
        if trend_quality.trend_quality in [TrendQuality.STRONG_UPTREND, TrendQuality.STRONG_DOWNTREND]:
            target = 15.0
        elif trend_quality.trend_quality in [TrendQuality.WEAK_UPTREND, TrendQuality.WEAK_DOWNTREND]:
            target = 10.0
        else:
            target = 8.0
        
        rr_ratio = target / stop_loss
        
        return stop_loss, target, rr_ratio
