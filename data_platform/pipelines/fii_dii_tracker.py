"""
FII/DII Activity Tracker - Institutional Capital Flow Analysis

Tracks Foreign Institutional Investor (FII) and Domestic Institutional Investor (DII)
activity to understand capital flows and market sentiment.

Institutional traders monitor:
- Net FII buying/selling (cash market)
- FII/DII activity in index derivatives
- FII/DII activity in stock derivatives
- Capital flow trends
- Sector-wise FII exposure

This data is critical because:
- FIIs drive large market moves
- DII often acts as counterbalance
- Derivatives activity indicates hedging/speculation
- Capital flows show institutional conviction
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import pandas as pd
import requests

from utils.logger import get_logger

logger = get_logger("fii_dii_tracker")


class FlowDirection(str, Enum):
    """Direction of capital flow."""
    NET_BUY = "net_buy"
    NET_SELL = "net_sell"
    NEUTRAL = "neutral"


class ConvictionLevel(str, Enum):
    """Level of institutional conviction."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


@dataclass
class FIIActivitySnapshot:
    """Snapshot of FII activity for a specific date."""
    date: datetime
    
    # Cash market
    fii_cash_net_cr: float
    dii_cash_net_cr: float
    
    # Index derivatives
    fii_index_long_cr: float
    fii_index_short_cr: float
    fii_index_net_cr: float
    dii_index_long_cr: float
    dii_index_short_cr: float
    dii_index_net_cr: float
    
    # Stock derivatives
    fii_stock_long_cr: float
    fii_stock_short_cr: float
    fii_stock_net_cr: float
    dii_stock_long_cr: float
    dii_stock_short_cr: float
    dii_stock_net_cr: float
    
    # Totals
    fii_total_net_cr: float
    dii_total_net_cr: float
    net_flow_cr: float  # FII + DII combined


@dataclass
class CapitalFlowTrend:
    """Analysis of capital flow trends over time."""
    period_start: datetime
    period_end: datetime
    
    # Trend metrics
    avg_daily_fii_flow_cr: float
    avg_daily_dii_flow_cr: float
    fii_flow_trend: str  # increasing, decreasing, stable
    dii_flow_trend: str
    
    # Conviction metrics
    fii_conviction: ConvictionLevel
    dii_conviction: ConvictionLevel
    
    # Correlation with market
    correlation_with_nifty: float
    
    # Key insights
    insights: list[str]


@dataclass
class SectorFIIData:
    """FII exposure by sector."""
    sector: str
    fii_exposure_cr: float
    fii_weightage_pct: float
    change_vs_previous_day_cr: float
    change_vs_previous_week_cr: float
    conviction: ConvictionLevel
    timestamp: datetime


class FIIDIIAnalyzer:
    """
    Analyzes FII/DII activity for institutional trading insights.
    
    Key analyses:
    1. Daily FII/DII cash market flows
    2. Derivatives positioning (long/short)
    3. Flow trends over time
    4. Conviction levels
    5. Sector-wise exposure
    6. Correlation with market movements
    """

    def __init__(self, use_mock_data: bool = True):
        """
        Initialize FII/DII analyzer.
        
        Parameters
        ----------
        use_mock_data
            Use mock data for development
        """
        self.use_mock_data = use_mock_data
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def get_daily_activity(self, date: Optional[datetime] = None) -> FIIActivitySnapshot:
        """
        Get FII/DII activity for a specific date.
        
        Parameters
        ----------
        date
            Date to fetch (default: today)
            
        Returns
        -------
        FIIActivitySnapshot
            Complete FII/DII activity snapshot
        """
        if date is None:
            date = datetime.now()
        
        if self.use_mock_data:
            return self._mock_daily_activity(date)
        
        try:
            # In production, fetch from NSE
            return self._mock_daily_activity(date)
        except Exception as e:
            logger.error(f"Failed to fetch daily activity: {e}")
            return self._mock_daily_activity(date)

    def analyze_flow_trend(
        self, 
        days: int = 20
    ) -> CapitalFlowTrend:
        """
        Analyze FII/DII flow trends over a period.
        
        Parameters
        ----------
        days
            Number of days to analyze
            
        Returns
        -------
        CapitalFlowTrend
            Trend analysis with conviction levels
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        if self.use_mock_data:
            return self._mock_flow_trend(start_date, end_date)
        
        try:
            return self._mock_flow_trend(start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to analyze flow trend: {e}")
            return self._mock_flow_trend(start_date, end_date)

    def get_sector_fii_exposure(self) -> list[SectorFIIData]:
        """
        Get FII exposure by sector.
        
        Returns
        -------
        list[SectorFIIData]
            FII exposure data for each sector
        """
        if self.use_mock_data:
            return self._mock_sector_exposure()
        
        try:
            return self._mock_sector_exposure()
        except Exception as e:
            logger.error(f"Failed to fetch sector exposure: {e}")
            return []

    def assess_market_impact(
        self, 
        activity: FIIActivitySnapshot
    ) -> dict[str, Any]:
        """
        Assess market impact of FII/DII activity.
        
        Parameters
        ----------
        activity
            FII/DII activity snapshot
            
        Returns
        -------
        dict[str, Any]
            Impact assessment with recommendations
        """
        impact = {
            "overall_sentiment": self._determine_flow_sentiment(activity),
            "fii_stance": self._determine_fii_stance(activity),
            "dii_stance": self._determine_dii_stance(activity),
            "hedging_activity": self._assess_hedging(activity),
            "market_impact": self._calculate_market_impact(activity),
            "recommendations": self._generate_recommendations(activity)
        }
        
        return impact

    def _determine_flow_sentiment(self, activity: FIIActivitySnapshot) -> str:
        """Determine overall flow sentiment."""
        total_flow = activity.fii_total_net_cr + activity.dii_total_net_cr
        
        if total_flow > 1000:
            return "strongly_bullish"
        elif total_flow > 500:
            return "bullish"
        elif total_flow > -500:
            return "neutral"
        elif total_flow > -1000:
            return "bearish"
        else:
            return "strongly_bearish"

    def _determine_fii_stance(self, activity: FIIActivitySnapshot) -> str:
        """Determine FII stance based on cash and derivatives."""
        # FII cash buying + long derivatives = bullish
        if activity.fii_cash_net_cr > 500 and activity.fii_index_net_cr > 0:
            return "aggressive_bull"
        elif activity.fii_cash_net_cr > 200:
            return "moderate_bull"
        elif activity.fii_cash_net_cr > -200:
            return "neutral"
        elif activity.fii_cash_net_cr > -500:
            return "moderate_bear"
        else:
            return "aggressive_bear"

    def _determine_dii_stance(self, activity: FIIActivitySnapshot) -> str:
        """Determine DII stance."""
        if activity.dii_cash_net_cr > 300:
            return "bullish"
        elif activity.dii_cash_net_cr > -300:
            return "neutral"
        else:
            return "bearish"

    def _assess_hedging(self, activity: FIIActivitySnapshot) -> str:
        """Assess hedging activity in derivatives."""
        # High short positions in derivatives = hedging
        fii_short_ratio = abs(activity.fii_index_short_cr) / max(
            abs(activity.fii_index_long_cr) + abs(activity.fii_index_short_cr), 1
        )
        
        if fii_short_ratio > 0.6:
            return "heavy_hedging"
        elif fii_short_ratio > 0.4:
            return "moderate_hedging"
        else:
            return "minimal_hedging"

    def _calculate_market_impact(self, activity: FIIActivitySnapshot) -> float:
        """
        Calculate expected market impact score (-1 to 1).
        
        Higher positive score = bullish impact
        Higher negative score = bearish impact
        """
        score = 0.0
        
        # Cash market impact (weight: 0.5)
        cash_impact = (activity.fii_cash_net_cr + activity.dii_cash_net_cr) / 2000.0
        score += cash_impact * 0.5
        
        # Derivatives impact (weight: 0.3)
        deriv_impact = (activity.fii_index_net_cr + activity.dii_index_net_cr) / 1000.0
        score += deriv_impact * 0.3
        
        # Conviction based on magnitude (weight: 0.2)
        total_magnitude = abs(activity.fii_total_net_cr) + abs(activity.dii_total_net_cr)
        conviction = min(1.0, total_magnitude / 3000.0)
        score += conviction * 0.2 if score > 0 else -conviction * 0.2
        
        return max(-1.0, min(1.0, score))

    def _generate_recommendations(self, activity: FIIActivitySnapshot) -> list[str]:
        """Generate trading recommendations based on FII/DII activity."""
        recommendations = []
        sentiment = self._determine_flow_sentiment(activity)
        fii_stance = self._determine_fii_stance(activity)
        hedging = self._assess_hedging(activity)
        
        if sentiment in ["bullish", "strongly_bullish"]:
            recommendations.append("Align with institutional bullish bias")
            recommendations.append("Focus on sectors with FII buying")
            
            if fii_stance == "aggressive_bull":
                recommendations.append("Consider momentum strategies")
        
        elif sentiment in ["bearish", "strongly_bearish"]:
            recommendations.append("Reduce exposure, align with institutional bearish bias")
            recommendations.append("Focus on defensive sectors")
            
            if hedging == "heavy_hedging":
                recommendations.append("Expect volatility, use wider stops")
        
        else:
            recommendations.append("Wait for clarity from institutional flows")
            recommendations.append("Focus on stock-specific stories")
        
        # DII as counterbalance
        if activity.dii_cash_net_cr > 500 and activity.fii_cash_net_cr < -500:
            recommendations.append("DII supporting market, may limit downside")
        
        return recommendations

    def _mock_daily_activity(self, date: datetime) -> FIIActivitySnapshot:
        """Generate mock daily FII/DII activity."""
        import os
        env = os.getenv("ENV", "LOCAL")
        if env.upper() in ("LIVE", "PAPER"):
            raise RuntimeError("Fatal: FII/DII activity mock fallback triggered in live/paper environment!")
        import random
        
        return FIIActivitySnapshot(
            date=date,
            fii_cash_net_cr=random.uniform(-1500, 1500),
            dii_cash_net_cr=random.uniform(-800, 800),
            fii_index_long_cr=random.uniform(500, 2000),
            fii_index_short_cr=random.uniform(200, 1500),
            fii_index_net_cr=random.uniform(-500, 500),
            dii_index_long_cr=random.uniform(200, 800),
            dii_index_short_cr=random.uniform(100, 500),
            dii_index_net_cr=random.uniform(-200, 200),
            fii_stock_long_cr=random.uniform(200, 800),
            fii_stock_short_cr=random.uniform(100, 500),
            fii_stock_net_cr=random.uniform(-200, 200),
            dii_stock_long_cr=random.uniform(100, 400),
            dii_stock_short_cr=random.uniform(50, 200),
            dii_stock_net_cr=random.uniform(-100, 100),
            fii_total_net_cr=random.uniform(-1000, 1000),
            dii_total_net_cr=random.uniform(-500, 500),
            net_flow_cr=random.uniform(-1200, 1200)
        )

    def _mock_flow_trend(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> CapitalFlowTrend:
        """Generate mock flow trend analysis."""
        import os
        env = os.getenv("ENV", "LOCAL")
        if env.upper() in ("LIVE", "PAPER"):
            raise RuntimeError("Fatal: Capital flow trend mock fallback triggered in live/paper environment!")
        import random
        
        return CapitalFlowTrend(
            period_start=start_date,
            period_end=end_date,
            avg_daily_fii_flow_cr=random.uniform(-200, 200),
            avg_daily_dii_flow_cr=random.uniform(-100, 100),
            fii_flow_trend=random.choice(["increasing", "decreasing", "stable"]),
            dii_flow_trend=random.choice(["increasing", "decreasing", "stable"]),
            fii_conviction=random.choice([ConvictionLevel.STRONG, ConvictionLevel.MODERATE, ConvictionLevel.WEAK]),
            dii_conviction=random.choice([ConvictionLevel.STRONG, ConvictionLevel.MODERATE, ConvictionLevel.WEAK]),
            correlation_with_nifty=random.uniform(-0.8, 0.8),
            insights=[
                "FII flows showing increasing conviction",
                "DII acting as stabilizer in recent sessions",
                "Derivatives positioning suggests hedging activity"
            ]
        )

    def _mock_sector_exposure(self) -> list[SectorFIIData]:
        """Generate mock sector FII exposure."""
        import os
        env = os.getenv("ENV", "LOCAL")
        if env.upper() in ("LIVE", "PAPER"):
            raise RuntimeError("Fatal: Sector exposure mock fallback triggered in live/paper environment!")
        sectors = [
            "Financial Services", "Technology", "Consumer Goods", 
            "Energy", "Healthcare", "Materials", "Automobile",
            "Capital Goods", "Telecommunication", "Chemicals"
        ]
        
        exposure = []
        for sector in sectors:
            exposure.append(SectorFIIData(
                sector=sector,
                fii_exposure_cr=random.uniform(5000, 50000),
                fii_weightage_pct=random.uniform(5.0, 25.0),
                change_vs_previous_day_cr=random.uniform(-500, 500),
                change_vs_previous_week_cr=random.uniform(-2000, 2000),
                conviction=random.choice([ConvictionLevel.STRONG, ConvictionLevel.MODERATE, ConvictionLevel.WEAK]),
                timestamp=datetime.now()
            ))
        
        return exposure
