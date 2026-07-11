"""
Market Regime Features Module

Computes market-level features for regime detection.
Uses ~15-20 carefully chosen features across:
- Market Trend (NIFTY, Bank Nifty)
- Trend Indicators (EMA, ADX)
- Volatility (India VIX, ATR, Daily Range)
- Breadth (Advance/Decline, % above EMAs)
- Institutional (FII/DII flows)
- Liquidity (Volume, Delivery %)
- Options (PCR, OI Change, Max Pain)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta

from data_platform.upstox_client import (
    get_candles,
    get_fii_activity,
    get_dii_activity,
    get_index_overview,
    get_pcr,
    get_max_pain,
    get_change_oi,
    get_option_expiries,
    INDEX_KEYS,
)
from utils.logger import get_logger

logger = get_logger("regime_features")


@dataclass
class RegimeFeatures:
    """Container for all regime detection features."""
    
    # Market Trend
    nifty_close: float | None = None
    nifty_pct_change: float | None = None
    banknifty_close: float | None = None
    banknifty_pct_change: float | None = None
    
    # Trend Indicators
    ema20: float | None = None
    ema50: float | None = None
    ema200: float | None = None
    adx: float | None = None
    ema20_above_ema50: bool = False
    ema20_above_ema200: bool = False
    price_above_ema50: bool = False
    price_above_ema200: bool = False
    
    # Volatility
    india_vix: float | None = None
    atr: float | None = None
    daily_range_pct: float | None = None
    atr_pct: float | None = None
    
    # Breadth
    advance_decline_ratio: float | None = None
    pct_above_ema50: float | None = None
    pct_above_ema200: float | None = None
    
    # Institutional
    fii_net_cash: float | None = None
    fii_net_futures: float | None = None
    dii_net_cash: float | None = None
    fii_buying: bool = False
    dii_buying: bool = False
    
    # Liquidity
    nifty_volume: float | None = None
    volume_avg_20: float | None = None
    volume_ratio: float | None = None
    
    # Options
    pcr: float | None = None
    oi_change: float | None = None
    max_pain_distance: float | None = None
    
    # Computed flags
    trend_strength: str = "Neutral"  # Strong, Moderate, Weak, Neutral
    volatility_level: str = "Normal"  # Low, Normal, High, Extreme
    liquidity_status: str = "Normal"  # Low, Normal, High
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "nifty_close": self.nifty_close,
            "nifty_pct_change": self.nifty_pct_change,
            "banknifty_close": self.banknifty_close,
            "banknifty_pct_change": self.banknifty_pct_change,
            "ema20": self.ema20,
            "ema50": self.ema50,
            "ema200": self.ema200,
            "adx": self.adx,
            "ema20_above_ema50": self.ema20_above_ema50,
            "ema20_above_ema200": self.ema20_above_ema200,
            "price_above_ema50": self.price_above_ema50,
            "price_above_ema200": self.price_above_ema200,
            "india_vix": self.india_vix,
            "atr": self.atr,
            "daily_range_pct": self.daily_range_pct,
            "atr_pct": self.atr_pct,
            "advance_decline_ratio": self.advance_decline_ratio,
            "pct_above_ema50": self.pct_above_ema50,
            "pct_above_ema200": self.pct_above_ema200,
            "fii_net_cash": self.fii_net_cash,
            "fii_net_futures": self.fii_net_futures,
            "dii_net_cash": self.dii_net_cash,
            "fii_buying": self.fii_buying,
            "dii_buying": self.dii_buying,
            "nifty_volume": self.nifty_volume,
            "volume_avg_20": self.volume_avg_20,
            "volume_ratio": self.volume_ratio,
            "pcr": self.pcr,
            "oi_change": self.oi_change,
            "max_pain_distance": self.max_pain_distance,
            "trend_strength": self.trend_strength,
            "volatility_level": self.volatility_level,
            "liquidity_status": self.liquidity_status,
        }


class RegimeFeatureEngine:
    """
    Engine to compute all market regime features.
    
    Fetches data from multiple sources and computes technical indicators.
    Handles missing data gracefully.
    """
    
    def __init__(self, lookback_days: int = 200):
        """
        Initialize the feature engine.
        
        Args:
            lookback_days: Number of days of historical data for indicator computation
        """
        self.lookback_days = lookback_days
        self.logger = logger
        
    def compute_all_features(self, asof_date: date | None = None) -> RegimeFeatures:
        """
        Compute all regime features for a given date.
        
        Args:
            asof_date: Date to compute features for (defaults to today)
            
        Returns:
            RegimeFeatures object with all computed features
        """
        if asof_date is None:
            asof_date = date.today()
            
        self.logger.info(f"Computing regime features for {asof_date}")
        
        features = RegimeFeatures()
        
        try:
            # 1. Market Trend Features
            self._compute_market_trend(features, asof_date)
            
            # 2. Trend Indicators
            self._compute_trend_indicators(features, asof_date)
            
            # 3. Volatility Features
            self._compute_volatility_features(features, asof_date)
            
            # 4. Breadth Features
            self._compute_breadth_features(features, asof_date)
            
            # 5. Institutional Features
            self._compute_institutional_features(features, asof_date)
            
            # 6. Liquidity Features
            self._compute_liquidity_features(features, asof_date)
            
            # 7. Options Features
            self._compute_options_features(features, asof_date)
            
            # 8. Computed flags
            self._compute_computed_flags(features)
            
            self.logger.info(f"Successfully computed {len(features.to_dict())} regime features")
            
        except Exception as e:
            self.logger.error(f"Error computing regime features: {e}", exc_info=True)
            
        return features
    
    def _compute_market_trend(self, features: RegimeFeatures, asof_date: date) -> None:
        """Compute market trend features from index data."""
        try:
            indices = get_index_overview()
            
            nifty = indices.get("NIFTY50", {})
            features.nifty_close = nifty.get("last_price")
            features.nifty_pct_change = nifty.get("pct_change")
            
            banknifty = indices.get("BANKNIFTY", {})
            features.banknifty_close = banknifty.get("last_price")
            features.banknifty_pct_change = banknifty.get("pct_change")
            
            self.logger.debug(f"Market trend: NIFTY={features.nifty_close}, BANKNIFTY={features.banknifty_close}")
            
        except Exception as e:
            self.logger.warning(f"Failed to compute market trend: {e}")
    
    def _compute_trend_indicators(self, features: RegimeFeatures, asof_date: date) -> None:
        """Compute EMA and ADX indicators from NIFTY data."""
        try:
            nifty_key = INDEX_KEYS["NIFTY50"]
            candles = get_candles("NIFTY50", interval="1day", days=self.lookback_days)
            
            if not candles or len(candles) < 200:
                self.logger.warning("Insufficient NIFTY data for trend indicators")
                return
            
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            close = df['close'].values
            
            # Compute EMAs
            features.ema20 = ta.ema(close, length=20).iloc[-1] if len(close) >= 20 else None
            features.ema50 = ta.ema(close, length=50).iloc[-1] if len(close) >= 50 else None
            features.ema200 = ta.ema(close, length=200).iloc[-1] if len(close) >= 200 else None
            
            # Compute ADX
            adx_result = ta.adx(df['high'], df['low'], close, length=14)
            features.adx = adx_result['ADX_14'].iloc[-1] if adx_result is not None else None
            
            # Compute relationship flags
            current_price = close[-1]
            if features.ema20 and features.ema50:
                features.ema20_above_ema50 = features.ema20 > features.ema50
            if features.ema20 and features.ema200:
                features.ema20_above_ema200 = features.ema20 > features.ema200
            if features.ema50:
                features.price_above_ema50 = current_price > features.ema50
            if features.ema200:
                features.price_above_ema200 = current_price > features.ema200
            
            self.logger.debug(f"Trend indicators: EMA20={features.ema20}, EMA50={features.ema50}, ADX={features.adx}")
            
        except Exception as e:
            self.logger.warning(f"Failed to compute trend indicators: {e}")
    
    def _compute_volatility_features(self, features: RegimeFeatures, asof_date: date) -> None:
        """Compute volatility features from India VIX and ATR."""
        try:
            # India VIX
            indices = get_index_overview()
            vix_data = indices.get("INDIAVIX", {})
            features.india_vix = vix_data.get("last_price")
            
            # ATR and Daily Range from NIFTY
            candles = get_candles("NIFTY50", interval="1day", days=self.lookback_days)
            if candles and len(candles) >= 14:
                df = pd.DataFrame(candles)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # ATR
                atr_result = ta.atr(df['high'], df['low'], df['close'], length=14)
                features.atr = atr_result.iloc[-1] if atr_result is not None else None
                
                # Daily range percentage
                latest = df.iloc[-1]
                if latest['close'] > 0:
                    features.daily_range_pct = ((latest['high'] - latest['low']) / latest['close']) * 100
                
                # ATR as percentage of price
                if features.atr and latest['close'] > 0:
                    features.atr_pct = (features.atr / latest['close']) * 100
            
            self.logger.debug(f"Volatility: VIX={features.india_vix}, ATR={features.atr}")
            
        except Exception as e:
            self.logger.warning(f"Failed to compute volatility features: {e}")
    
    def _compute_breadth_features(self, features: RegimeFeatures, asof_date: date) -> None:
        """
        Compute market breadth features.
        
        Note: Advance/Decline ratio requires breadth data from NSE.
        For now, we'll use a simplified approach based on sector performance.
        """
        try:
            # Simplified breadth using sector overview
            from data_platform.upstox_client import get_sector_overview
            
            sectors = get_sector_overview()
            
            if sectors:
                # Count advancing vs declining sectors
                advancing = sum(1 for s in sectors.values() if s.get('pct_change', 0) > 0)
                declining = sum(1 for s in sectors.values() if s.get('pct_change', 0) < 0)
                
                if declining > 0:
                    features.advance_decline_ratio = advancing / declining
                else:
                    features.advance_decline_ratio = advancing if advancing > 0 else 1.0
            
            # For % above EMAs, we would need stock universe data
            # Placeholder - would require computing from stock universe
            features.pct_above_ema50 = 0.5  # Placeholder
            features.pct_above_ema200 = 0.5  # Placeholder
            
            self.logger.debug(f"Breadth: A/D ratio={features.advance_decline_ratio}")
            
        except Exception as e:
            self.logger.warning(f"Failed to compute breadth features: {e}")
    
    def _compute_institutional_features(self, features: RegimeFeatures, asof_date: date) -> None:
        """Compute FII/DII flow features."""
        try:
            # Use previous day's data (today's not available until EOD)
            prev_date = asof_date - timedelta(days=1)
            
            fii_data = get_fii_activity(prev_date.isoformat())
            dii_data = get_dii_activity(prev_date.isoformat())
            
            features.fii_net_cash = fii_data.get("total_net_cash")
            features.fii_net_futures = fii_data.get("total_net_futures")
            features.dii_net_cash = dii_data.get("net_amount")
            
            features.fii_buying = (features.fii_net_cash or 0) > 0 if features.fii_net_cash is not None else False
            features.dii_buying = (features.dii_net_cash or 0) > 0 if features.dii_net_cash is not None else False
            
            self.logger.debug(f"Institutional: FII net={features.fii_net_cash}, DII net={features.dii_net_cash}")
            
        except Exception as e:
            self.logger.warning(f"Failed to compute institutional features: {e}")
    
    def _compute_liquidity_features(self, features: RegimeFeatures, asof_date: date) -> None:
        """Compute liquidity features from volume data."""
        try:
            candles = get_candles("NIFTY50", interval="1day", days=self.lookback_days)
            
            if candles and len(candles) >= 20:
                df = pd.DataFrame(candles)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                latest = df.iloc[-1]
                features.nifty_volume = latest['volume']
                
                # 20-day average volume
                features.volume_avg_20 = df['volume'].tail(20).mean()
                
                # Volume ratio
                if features.volume_avg_20 and features.volume_avg_20 > 0:
                    features.volume_ratio = features.nifty_volume / features.volume_avg_20
            
            self.logger.debug(f"Liquidity: Volume={features.nifty_volume}, Ratio={features.volume_ratio}")
            
        except Exception as e:
            self.logger.warning(f"Failed to compute liquidity features: {e}")
    
    def _compute_options_features(self, features: RegimeFeatures, asof_date: date) -> None:
        """Compute options-related features (PCR, OI change, Max Pain)."""
        try:
            nifty_key = INDEX_KEYS["NIFTY50"]
            
            # Get nearest expiry
            expiries = get_option_expiries(nifty_key)
            if not expiries:
                self.logger.warning("No option expiries available")
                return
            
            nearest_expiry = expiries[0]
            
            # PCR
            pcr_data = get_pcr(nifty_key, nearest_expiry)
            if pcr_data:
                features.pcr = pcr_data.get("pcr")
            
            # Max Pain
            max_pain_data = get_max_pain(nifty_key, nearest_expiry)
            if max_pain_data and features.nifty_close:
                max_pain = max_pain_data.get("max_pain")
                if max_pain:
                    features.nifty_close = features.nifty_close or get_index_overview().get("NIFTY50", {}).get("last_price")
                    if features.nifty_close:
                        features.max_pain_distance = abs(features.nifty_close - max_pain) / features.nifty_close * 100
            
            # OI Change
            oi_change_data = get_change_oi(nifty_key, nearest_expiry)
            if oi_change_data:
                # Sum up total OI change across all strikes
                total_oi_change = 0
                for strike_data in oi_change_data.values():
                    if isinstance(strike_data, dict):
                        total_oi_change += strike_data.get("oi_change", 0)
                features.oi_change = total_oi_change
            
            self.logger.debug(f"Options: PCR={features.pcr}, OI change={features.oi_change}")
            
        except Exception as e:
            self.logger.warning(f"Failed to compute options features: {e}")
    
    def _compute_computed_flags(self, features: RegimeFeatures) -> None:
        """Compute derived flags from raw features."""
        # Trend Strength based on ADX and EMA alignment
        if features.adx:
            if features.adx > 40:
                features.trend_strength = "Strong"
            elif features.adx > 25:
                features.trend_strength = "Moderate"
            elif features.adx > 20:
                features.trend_strength = "Weak"
            else:
                features.trend_strength = "Neutral"
        
        # Volatility Level based on India VIX
        if features.india_vix:
            if features.india_vix > 25:
                features.volatility_level = "Extreme"
            elif features.india_vix > 20:
                features.volatility_level = "High"
            elif features.india_vix < 13:
                features.volatility_level = "Low"
            else:
                features.volatility_level = "Normal"
        
        # Liquidity Status based on volume ratio
        if features.volume_ratio:
            if features.volume_ratio > 1.5:
                features.liquidity_status = "High"
            elif features.volume_ratio < 0.7:
                features.liquidity_status = "Low"
            else:
                features.liquidity_status = "Normal"
