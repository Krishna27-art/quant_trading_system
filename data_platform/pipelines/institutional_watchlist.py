"""
Institutional Watchlist Builder - Professional Stock Selection

Implements institutional-grade stock selection process:
1. Start with NSE universe (5000+ stocks)
2. Remove illiquid stocks (low average daily volume)
3. Remove low-volume stocks (insufficient trading activity)
4. Keep only today's active stocks (price movement, volume surge)
5. Check sectors and rank by strength
6. Find leaders inside strongest sectors
7. Check relative volume (vs 20-day average)
8. Check news impact
9. Check futures/open interest (for derivatives)
10. Final watchlist: 10-20 high-conviction names

Institutional traders prioritize:
- Liquidity (can enter/exit without slippage)
- Volatility (movement potential)
- Relative volume (unusual activity)
- Momentum (trend alignment)
- Sector strength (follow the money)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from config.universe import NSE_UNIVERSE
from utils.logger import get_logger

logger = get_logger("institutional_watchlist")


@dataclass
class StockMetrics:
    """Key metrics for stock selection."""
    symbol: str
    name: str
    sector: str
    cap_tier: str  # Large, Mid, Small
    
    # Liquidity metrics
    avg_daily_volume: int
    avg_daily_turnover_cr: float
    bid_ask_spread_pct: float
    
    # Today's activity
    current_price: float
    price_change_pct: float
    volume_today: int
    relative_volume: float  # vs 20-day average
    
    # Volatility metrics
    atr_14: float
    atr_pct: float
    volatility_20d: float
    
    # Momentum metrics
    rsi_14: float
    macd_signal: str  # bullish, bearish, neutral
    price_vs_sma20: float  # % above/below SMA20
    price_vs_sma50: float  # % above/below SMA50
    
    # Sector metrics
    sector_strength_rank: int
    sector_strength_score: float
    
    # News and derivatives
    news_sentiment: str  # positive, negative, neutral
    news_impact_score: float
    futures_oi_change_pct: Optional[float] = None
    options_oi_change_pct: Optional[float] = None
    
    timestamp: datetime


@dataclass
class SectorMetrics:
    """Sector-level metrics for ranking."""
    sector: str
    num_stocks: int
    avg_change_pct: float
    strength_score: float
    advancers: int
    decliners: int
    unchanged: int
    volume_ratio: float  # vs 20-day average
    timestamp: datetime


@dataclass
class WatchlistResult:
    """Final institutional watchlist."""
    selected_stocks: list[StockMetrics]
    sector_rankings: list[SectorMetrics]
    selection_criteria: dict[str, Any]
    total_universe_size: int
    filtered_count: int
    final_count: int
    timestamp: datetime


class InstitutionalWatchlistBuilder:
    """
    Builds institutional-grade watchlists using systematic filtering.
    
    Process:
    1. Load NSE universe (5000+ stocks)
    2. Filter by liquidity (minimum ADVT)
    3. Filter by volume (minimum average daily volume)
    4. Filter by today's activity (price movement, volume surge)
    5. Analyze sector strength
    6. Rank sectors
    7. Select leaders from top sectors
    8. Apply final quality filters
    9. Return 10-20 high-conviction names
    """

    def __init__(
        self,
        min_avg_daily_volume: int = 1_000_000,  # 1M shares minimum
        min_avg_turnover_cr: float = 10.0,  # 10 crore minimum
        min_relative_volume: float = 1.5,  # 50% above average
        max_bid_ask_spread_pct: float = 0.5,  # 0.5% max spread
        min_sector_strenght: float = 0.0,
        target_watchlist_size: int = 15,
        use_mock_data: bool = True
    ):
        """
        Initialize watchlist builder.
        
        Parameters
        ----------
        min_avg_daily_volume
            Minimum average daily volume for liquidity
        min_avg_turnover_cr
            Minimum average daily turnover in crores
        min_relative_volume
            Minimum relative volume (vs 20-day average)
        max_bid_ask_spread_pct
            Maximum bid-ask spread percentage
        min_sector_strenght
            Minimum sector strength score
        target_watchlist_size
            Target number of stocks in final watchlist
        use_mock_data
            Use mock data for development
        """
        self.min_avg_daily_volume = min_avg_daily_volume
        self.min_avg_turnover_cr = min_avg_turnover_cr
        self.min_relative_volume = min_relative_volume
        self.max_bid_ask_spread_pct = max_bid_ask_spread_pct
        self.min_sector_strength = min_sector_strenght
        self.target_watchlist_size = target_watchlist_size
        self.use_mock_data = use_mock_data

    def build_watchlist(self) -> WatchlistResult:
        """
        Build institutional watchlist for today.
        
        Returns
        -------
        WatchlistResult
            Complete watchlist with metrics and rankings.
        """
        logger.info("Starting institutional watchlist build")
        
        # Step 1: Load universe
        universe = self._load_universe()
        logger.info(f"Loaded universe: {len(universe)} stocks")
        
        # Step 2: Fetch metrics for all stocks
        all_metrics = self._fetch_stock_metrics(universe)
        logger.info(f"Fetched metrics for {len(all_metrics)} stocks")
        
        # Step 3: Filter by liquidity
        liquidity_filtered = self._filter_by_liquidity(all_metrics)
        logger.info(f"After liquidity filter: {len(liquidity_filtered)} stocks")
        
        # Step 4: Filter by today's activity
        activity_filtered = self._filter_by_activity(liquidity_filtered)
        logger.info(f"After activity filter: {len(activity_filtered)} stocks")
        
        # Step 5: Analyze sector strength
        sector_metrics = self._analyze_sector_strength(activity_filtered)
        logger.info(f"Analyzed {len(sector_metrics)} sectors")
        
        # Step 6: Rank sectors
        ranked_sectors = self._rank_sectors(sector_metrics)
        
        # Step 7: Select leaders from top sectors
        top_sectors = ranked_sectors[:5]  # Top 5 sectors
        sector_leaders = self._select_sector_leaders(
            activity_filtered, top_sectors
        )
        logger.info(f"Selected {len(sector_leaders)} sector leaders")
        
        # Step 8: Apply final quality filters
        final_watchlist = self._apply_final_filters(sector_leaders)
        logger.info(f"Final watchlist: {len(final_watchlist)} stocks")
        
        result = WatchlistResult(
            selected_stocks=final_watchlist,
            sector_rankings=ranked_sectors,
            selection_criteria={
                "min_avg_daily_volume": self.min_avg_daily_volume,
                "min_avg_turnover_cr": self.min_avg_turnover_cr,
                "min_relative_volume": self.min_relative_volume,
                "max_bid_ask_spread_pct": self.max_bid_ask_spread_pct,
                "min_sector_strength": self.min_sector_strength,
                "target_watchlist_size": self.target_watchlist_size
            },
            total_universe_size=len(universe),
            filtered_count=len(activity_filtered),
            final_count=len(final_watchlist),
            timestamp=datetime.now()
        )
        
        return result

    def _load_universe(self) -> list[dict]:
        """Load NSE universe from config."""
        return NSE_UNIVERSE

    def _fetch_stock_metrics(self, universe: list[dict]) -> list[StockMetrics]:
        """Fetch metrics for all stocks in universe."""
        if self.use_mock_data:
            return self._mock_stock_metrics(universe)
        
        # In production, fetch from database/API
        return self._mock_stock_metrics(universe)

    def _filter_by_liquidity(self, metrics: list[StockMetrics]) -> list[StockMetrics]:
        """
        Filter stocks by liquidity criteria.
        
        Institutional traders require:
        - Sufficient average daily volume
        - Sufficient turnover
        - Reasonable bid-ask spread
        """
        filtered = []
        for m in metrics:
            if (m.avg_daily_volume >= self.min_avg_daily_volume and
                m.avg_daily_turnover_cr >= self.min_avg_turnover_cr and
                m.bid_ask_spread_pct <= self.max_bid_ask_spread_pct):
                filtered.append(m)
        
        return filtered

    def _filter_by_activity(self, metrics: list[StockMetrics]) -> list[StockMetrics]:
        """
        Filter stocks by today's activity.
        
        Focus on:
        - Price movement (not flat)
        - Volume surge (relative volume)
        - Volatility (movement potential)
        """
        filtered = []
        for m in metrics:
            # Price movement > 0.5% or volume surge
            if (abs(m.price_change_pct) > 0.5 or 
                m.relative_volume >= self.min_relative_volume):
                filtered.append(m)
        
        return filtered

    def _analyze_sector_strength(
        self, 
        metrics: list[StockMetrics]
    ) -> list[SectorMetrics]:
        """
        Analyze sector strength based on constituent stocks.
        
        Metrics:
        - Average change % across sector
        - Advance/decline ratio
        - Volume ratio
        """
        sector_data = {}
        
        for m in metrics:
            if m.sector not in sector_data:
                sector_data[m.sector] = {
                    "stocks": [],
                    "changes": [],
                    "volumes": []
                }
            sector_data[m.sector]["stocks"].append(m)
            sector_data[m.sector]["changes"].append(m.price_change_pct)
            sector_data[m.sector]["volumes"].append(m.relative_volume)
        
        sector_metrics = []
        for sector, data in sector_data.items():
            changes = data["changes"]
            volumes = data["volumes"]
            
            advancers = sum(1 for c in changes if c > 0)
            decliners = sum(1 for c in changes if c < 0)
            unchanged = len(changes) - advancers - decliners
            
            avg_change = np.mean(changes) if changes else 0.0
            avg_volume = np.mean(volumes) if volumes else 1.0
            
            # Strength score combines price performance and volume
            strength_score = avg_change * 0.6 + (avg_volume - 1.0) * 0.4
            
            sector_metrics.append(SectorMetrics(
                sector=sector,
                num_stocks=len(data["stocks"]),
                avg_change_pct=avg_change,
                strength_score=strength_score,
                advancers=advancers,
                decliners=decliners,
                unchanged=unchanged,
                volume_ratio=avg_volume,
                timestamp=datetime.now()
            ))
        
        return sector_metrics

    def _rank_sectors(self, sector_metrics: list[SectorMetrics]) -> list[SectorMetrics]:
        """Rank sectors by strength score."""
        return sorted(sector_metrics, key=lambda x: x.strength_score, reverse=True)

    def _select_sector_leaders(
        self,
        metrics: list[StockMetrics],
        top_sectors: list[SectorMetrics]
    ) -> list[StockMetrics]:
        """
        Select leading stocks from top sectors.
        
        Leaders in each sector:
        - Highest price change
        - High relative volume
        - Strong momentum
        """
        top_sector_names = {s.sector for s in top_sectors}
        sector_leaders = []
        
        for sector in top_sector_names:
            sector_stocks = [m for m in metrics if m.sector == sector]
            
            # Rank by combined score (change + volume + momentum)
            for stock in sector_stocks:
                stock.combined_score = (
                    stock.price_change_pct * 0.4 +
                    (stock.relative_volume - 1.0) * 20 * 0.3 +
                    stock.rsi_14 / 100.0 * 0.3
                )
            
            # Select top 2-3 from each sector
            sector_stocks_sorted = sorted(
                sector_stocks, 
                key=lambda x: getattr(x, 'combined_score', 0), 
                reverse=True
            )
            sector_leaders.extend(sector_stocks_sorted[:3])
        
        return sector_leaders

    def _apply_final_filters(self, metrics: list[StockMetrics]) -> list[StockMetrics]:
        """
        Apply final quality filters and select top N.
        
        Final criteria:
        - Sector strength minimum
        - News sentiment not extremely negative
        - Sort by combined score
        - Return top N
        """
        # Filter by sector strength
        filtered = [
            m for m in metrics 
            if m.sector_strength_score >= self.min_sector_strength
        ]
        
        # Filter out extremely negative news
        filtered = [
            m for m in filtered
            if not (m.news_sentiment == "negative" and m.news_impact_score < -0.7)
        ]
        
        # Calculate final score
        for stock in filtered:
            stock.final_score = (
                getattr(stock, 'combined_score', 0) * 0.5 +
                stock.sector_strength_score * 0.3 +
                (1.0 if stock.news_sentiment == "positive" else 0.0) * 0.2
            )
        
        # Sort and select top N
        filtered_sorted = sorted(
            filtered,
            key=lambda x: getattr(x, 'final_score', 0),
            reverse=True
        )
        
        return filtered_sorted[:self.target_watchlist_size]

    def _mock_stock_metrics(self, universe: list[dict]) -> list[StockMetrics]:
        """Generate mock stock metrics for development."""
        import os
        env = os.getenv("ENV", "LOCAL")
        if env.upper() in ("LIVE", "PAPER"):
            raise RuntimeError("Fatal: Stock metrics mock fallback triggered in live/paper environment!")
        import random
        
        metrics = []
        for stock in universe:
            # Generate realistic metrics
            base_volume = random.randint(500_000, 50_000_000)
            base_turnover = random.uniform(5.0, 500.0)
            price_change = random.uniform(-3.0, 3.0)
            relative_volume = random.uniform(0.5, 3.0)
            
            # Cap tier affects volume
            cap_multiplier = {
                "Large": 3.0,
                "Mid": 1.5,
                "Small": 0.5
            }.get(stock["cap"], 1.0)
            
            metrics.append(StockMetrics(
                symbol=stock["symbol"],
                name=stock["name"],
                sector=stock["sector"],
                cap_tier=stock["cap"],
                avg_daily_volume=int(base_volume * cap_multiplier),
                avg_daily_turnover_cr=base_turnover * cap_multiplier,
                bid_ask_spread_pct=random.uniform(0.05, 0.8),
                current_price=random.uniform(100.0, 3000.0),
                price_change_pct=price_change,
                volume_today=int(base_volume * relative_volume),
                relative_volume=relative_volume,
                atr_14=random.uniform(1.0, 50.0),
                atr_pct=random.uniform(0.5, 3.0),
                volatility_20d=random.uniform(10.0, 40.0),
                rsi_14=random.uniform(20.0, 80.0),
                macd_signal=random.choice(["bullish", "bearish", "neutral"]),
                price_vs_sma20=random.uniform(-5.0, 5.0),
                price_vs_sma50=random.uniform(-10.0, 10.0),
                sector_strength_rank=random.randint(1, 15),
                sector_strength_score=random.uniform(-2.0, 2.0),
                news_sentiment=random.choice(["positive", "negative", "neutral"]),
                news_impact_score=random.uniform(-0.8, 0.8),
                futures_oi_change_pct=random.uniform(-10.0, 10.0) if random.random() > 0.3 else None,
                options_oi_change_pct=random.uniform(-15.0, 15.0) if random.random() > 0.3 else None,
                timestamp=datetime.now()
            ))
        
        return metrics
