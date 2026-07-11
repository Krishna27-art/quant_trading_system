"""
Alpha Filters

Applies filters to alpha scores before final grading.

STEP 8: Alpha Filters
STEP 9: Risk/Reward Filter

Filters include:
- Liquidity filter (average volume, delivery percentage)
- Risk/reward filter (expected return vs risk)
- News filter (major negative events)
- Market regime filter
- Volatility filter
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("alpha_engine.filters")


class FilterType(Enum):
    """Types of alpha filters."""
    LIQUIDITY = "liquidity"
    RISK_REWARD = "risk_reward"
    NEWS = "news"
    VOLATILITY = "volatility"
    REGIME = "regime"
    CIRCUIT = "circuit"


@dataclass
class FilterResult:
    """
    Result of applying a single filter.
    """
    filter_type: FilterType
    passed: bool
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "filter_type": self.filter_type.value,
            "passed": self.passed,
            "reason": self.reason,
            "metadata": self.metadata,
        }


@dataclass
class FilterConfig:
    """
    Configuration for a specific filter.
    """
    filter_type: FilterType
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "filter_type": self.filter_type.value,
            "enabled": self.enabled,
            "params": self.params,
        }


class AlphaFilters:
    """
    Applies filters to alpha scores.
    
    Filters are applied sequentially. If any critical filter fails,
    the stock is rejected regardless of alpha score.
    """
    
    def __init__(self):
        """Initialize Alpha Filters."""
        self._logger = logger
        
        # Default filter configurations
        self.filter_configs = self._initialize_default_configs()
    
    def apply_filters(
        self,
        symbol: str,
        alpha_score: float,
        market_data: Dict[str, Any],
        regime_data: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, List[FilterResult]]:
        """
        Apply all enabled filters to a stock.
        
        Args:
            symbol: Stock symbol
            alpha_score: Raw alpha score
            market_data: Market data (volume, price, volatility, etc.)
            regime_data: Optional regime data
            
        Returns:
            Tuple of (passed_all, list of filter results)
        """
        self._logger.info(f"Applying filters for {symbol} (alpha={alpha_score:.2f})")
        
        results = []
        
        # Apply each enabled filter
        for filter_type, config in self.filter_configs.items():
            if not config.enabled:
                continue
            
            result = self._apply_single_filter(
                filter_type,
                symbol,
                alpha_score,
                market_data,
                regime_data,
                config.params,
            )
            results.append(result)
        
        # Check if all filters passed
        passed_all = all(r.passed for r in results)
        
        if not passed_all:
            failed_filters = [r for r in results if not r.passed]
            self._logger.info(
                f"{symbol} failed {len(failed_filters)} filters",
                extra={"failed_filters": [r.reason for r in failed_filters]},
            )
        
        return passed_all, results
    
    def _apply_single_filter(
        self,
        filter_type: FilterType,
        symbol: str,
        alpha_score: float,
        market_data: Dict[str, Any],
        regime_data: Optional[Dict[str, Any]],
        params: Dict[str, Any],
    ) -> FilterResult:
        """
        Apply a single filter.
        
        Args:
            filter_type: Type of filter to apply
            symbol: Stock symbol
            alpha_score: Raw alpha score
            market_data: Market data
            regime_data: Regime data
            params: Filter parameters
            
        Returns:
            FilterResult
        """
        if filter_type == FilterType.LIQUIDITY:
            return self._filter_liquidity(symbol, market_data, params)
        elif filter_type == FilterType.RISK_REWARD:
            return self._filter_risk_reward(symbol, alpha_score, market_data, params)
        elif filter_type == FilterType.NEWS:
            return self._filter_news(symbol, market_data, params)
        elif filter_type == FilterType.VOLATILITY:
            return self._filter_volatility(symbol, market_data, params)
        elif filter_type == FilterType.REGIME:
            return self._filter_regime(symbol, regime_data, params)
        elif filter_type == FilterType.CIRCUIT:
            return self._filter_circuit(symbol, market_data, params)
        else:
            return FilterResult(
                filter_type=filter_type,
                passed=True,
                reason=f"Filter {filter_type.value} not implemented",
            )
    
    def _filter_liquidity(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        params: Dict[str, Any],
    ) -> FilterResult:
        """
        Filter based on liquidity metrics.
        
        Checks:
        - Average daily volume (ADV)
        - Delivery percentage (for Indian markets)
        - Spread
        
        Args:
            symbol: Stock symbol
            market_data: Market data
            params: Filter parameters
            
        Returns:
            FilterResult
        """
        min_adv = params.get("min_adv", 1000000)  # Default: 1M INR
        min_delivery_pct = params.get("min_delivery_pct", 20.0)  # Default: 20%
        max_spread_bps = params.get("max_spread_bps", 10)  # Default: 10 bps
        
        adv = market_data.get("adv", 0)
        delivery_pct = market_data.get("delivery_pct", 100)
        spread_bps = market_data.get("spread_bps", 0)
        
        reasons = []
        
        # Check ADV
        if adv < min_adv:
            reasons.append(f"ADV too low: {adv:,.0f} < {min_adv:,.0f}")
        
        # Check delivery percentage
        if delivery_pct < min_delivery_pct:
            reasons.append(f"Delivery % too low: {delivery_pct:.1f}% < {min_delivery_pct:.1f}%")
        
        # Check spread
        if spread_bps > max_spread_bps:
            reasons.append(f"Spread too wide: {spread_bps:.1f} bps > {max_spread_bps:.1f} bps")
        
        if reasons:
            return FilterResult(
                filter_type=FilterType.LIQUIDITY,
                passed=False,
                reason="; ".join(reasons),
                metadata={"adv": adv, "delivery_pct": delivery_pct, "spread_bps": spread_bps},
            )
        
        return FilterResult(
            filter_type=FilterType.LIQUIDITY,
            passed=True,
            reason="Liquidity requirements met",
            metadata={"adv": adv, "delivery_pct": delivery_pct, "spread_bps": spread_bps},
        )
    
    def _filter_risk_reward(
        self,
        symbol: str,
        alpha_score: float,
        market_data: Dict[str, Any],
        params: Dict[str, Any],
    ) -> FilterResult:
        """
        Filter based on risk/reward ratio.
        
        Calculates:
        - Expected return (from prediction model or alpha score)
        - Risk (ATR or volatility)
        - Risk/Reward ratio
        
        Args:
            symbol: Stock symbol
            alpha_score: Raw alpha score
            market_data: Market data
            params: Filter parameters
            
        Returns:
            FilterResult
        """
        min_rr_ratio = params.get("min_rr_ratio", 1.5)  # Default: 1.5:1
        max_risk_pct = params.get("max_risk_pct", 5.0)  # Default: 5%
        
        # Get expected return from market data or estimate from alpha score
        expected_return = market_data.get("expected_return")
        if expected_return is None:
            # Estimate from alpha score (0-100 -> 0-15%)
            expected_return = (alpha_score / 100) * 0.15
        
        # Get risk from ATR or volatility
        atr_pct = market_data.get("atr_pct")
        volatility = market_data.get("volatility")
        
        if atr_pct is not None:
            risk = atr_pct
        elif volatility is not None:
            risk = volatility
        else:
            # Default risk estimate
            risk = 0.03  # 3%
        
        # Calculate risk/reward ratio
        if risk > 0:
            rr_ratio = expected_return / risk
        else:
            rr_ratio = np.inf
        
        reasons = []
        
        # Check risk/reward ratio
        if rr_ratio < min_rr_ratio:
            reasons.append(f"RR ratio too low: {rr_ratio:.2f} < {min_rr_ratio:.2f}")
        
        # Check absolute risk
        if risk > max_risk_pct:
            reasons.append(f"Risk too high: {risk:.2%} > {max_risk_pct:.2%}")
        
        if reasons:
            return FilterResult(
                filter_type=FilterType.RISK_REWARD,
                passed=False,
                reason="; ".join(reasons),
                metadata={
                    "expected_return": expected_return,
                    "risk": risk,
                    "rr_ratio": rr_ratio,
                },
            )
        
        return FilterResult(
            filter_type=FilterType.RISK_REWARD,
            passed=True,
            reason=f"Risk/reward acceptable: {rr_ratio:.2f}:1",
            metadata={
                "expected_return": expected_return,
                "risk": risk,
                "rr_ratio": rr_ratio,
            },
        )
    
    def _filter_news(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        params: Dict[str, Any],
    ) -> FilterResult:
        """
        Filter based on news sentiment.
        
        Checks for:
        - Major negative news
        - Corporate actions (earnings, board meetings, etc.)
        - Regulatory actions
        
        Args:
            symbol: Stock symbol
            market_data: Market data
            params: Filter parameters
            
        Returns:
            FilterResult
        """
        news_sentiment = market_data.get("news_sentiment", 0)  # -1 to 1
        has_major_negative = market_data.get("has_major_negative_news", False)
        has_corporate_action = market_data.get("has_corporate_action", False)
        
        reasons = []
        
        # Check for major negative news
        if has_major_negative:
            reasons.append("Major negative news detected")
        
        # Check news sentiment
        min_sentiment = params.get("min_sentiment", -0.5)
        if news_sentiment < min_sentiment:
            reasons.append(f"News sentiment too negative: {news_sentiment:.2f}")
        
        # Corporate actions are informational, not necessarily a filter
        # But we track them
        if has_corporate_action:
            return FilterResult(
                filter_type=FilterType.NEWS,
                passed=True,
                reason="Corporate action detected (informational)",
                metadata={
                    "news_sentiment": news_sentiment,
                    "has_corporate_action": True,
                },
            )
        
        if reasons:
            return FilterResult(
                filter_type=FilterType.NEWS,
                passed=False,
                reason="; ".join(reasons),
                metadata={
                    "news_sentiment": news_sentiment,
                    "has_major_negative": has_major_negative,
                },
            )
        
        return FilterResult(
            filter_type=FilterType.NEWS,
            passed=True,
            reason="No negative news filters",
            metadata={"news_sentiment": news_sentiment},
        )
    
    def _filter_volatility(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        params: Dict[str, Any],
    ) -> FilterResult:
        """
        Filter based on volatility levels.
        
        Args:
            symbol: Stock symbol
            market_data: Market data
            params: Filter parameters
            
        Returns:
            FilterResult
        """
        max_volatility = params.get("max_volatility", 0.15)  # Default: 15%
        min_volatility = params.get("min_volatility", 0.005)  # Default: 0.5%
        
        volatility = market_data.get("volatility", 0.02)
        
        reasons = []
        
        # Check max volatility
        if volatility > max_volatility:
            reasons.append(f"Volatility too high: {volatility:.2%} > {max_volatility:.2%}")
        
        # Check min volatility (avoid dead stocks)
        if volatility < min_volatility:
            reasons.append(f"Volatility too low: {volatility:.2%} < {min_volatility:.2%}")
        
        if reasons:
            return FilterResult(
                filter_type=FilterType.VOLATILITY,
                passed=False,
                reason="; ".join(reasons),
                metadata={"volatility": volatility},
            )
        
        return FilterResult(
            filter_type=FilterType.VOLATILITY,
            passed=True,
            reason=f"Volatility acceptable: {volatility:.2%}",
            metadata={"volatility": volatility},
        )
    
    def _filter_regime(
        self,
        symbol: str,
        regime_data: Optional[Dict[str, Any]],
        params: Dict[str, Any],
    ) -> FilterResult:
        """
        Filter based on market regime.
        
        Args:
            symbol: Stock symbol
            regime_data: Regime data
            params: Filter parameters
            
        Returns:
            FilterResult
        """
        if regime_data is None:
            return FilterResult(
                filter_type=FilterType.REGIME,
                passed=True,
                reason="No regime data available",
            )
        
        regime_type = regime_data.get("regime", "unknown")
        confidence = regime_data.get("confidence", 0)
        
        min_confidence = params.get("min_confidence", 60)
        
        # Filter if regime confidence is too low
        if confidence < min_confidence:
            return FilterResult(
                filter_type=FilterType.REGIME,
                passed=False,
                reason=f"Regime confidence too low: {confidence:.1f}% < {min_confidence:.1f}%",
                metadata={"regime": regime_type, "confidence": confidence},
            )
        
        return FilterResult(
            filter_type=FilterType.REGIME,
            passed=True,
            reason=f"Regime acceptable: {regime_type} (confidence: {confidence:.1f}%)",
            metadata={"regime": regime_type, "confidence": confidence},
        )
    
    def _filter_circuit(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        params: Dict[str, Any],
    ) -> FilterResult:
        """
        Filter based on circuit limits (Indian market specific).
        
        Args:
            symbol: Stock symbol
            market_data: Market data
            params: Filter parameters
            
        Returns:
            FilterResult
        """
        near_circuit = market_data.get("near_circuit", False)
        circuit_hit = market_data.get("circuit_hit", False)
        
        if circuit_hit:
            return FilterResult(
                filter_type=FilterType.CIRCUIT,
                passed=False,
                reason="Stock hit circuit limit",
                metadata={"circuit_hit": True},
            )
        
        if near_circuit:
            return FilterResult(
                filter_type=FilterType.CIRCUIT,
                passed=False,
                reason="Stock near circuit limit",
                metadata={"near_circuit": True},
            )
        
        return FilterResult(
            filter_type=FilterType.CIRCUIT,
            passed=True,
            reason="No circuit issues",
            metadata={"circuit_hit": False, "near_circuit": False},
        )
    
    def _initialize_default_configs(self) -> Dict[FilterType, FilterConfig]:
        """
        Initialize default filter configurations.
        
        Returns:
            Dictionary of FilterType -> FilterConfig
        """
        return {
            FilterType.LIQUIDITY: FilterConfig(
                filter_type=FilterType.LIQUIDITY,
                enabled=True,
                params={
                    "min_adv": 1000000,  # 1M INR
                    "min_delivery_pct": 20.0,
                    "max_spread_bps": 10,
                },
            ),
            FilterType.RISK_REWARD: FilterConfig(
                filter_type=FilterType.RISK_REWARD,
                enabled=True,
                params={
                    "min_rr_ratio": 1.5,
                    "max_risk_pct": 5.0,
                },
            ),
            FilterType.NEWS: FilterConfig(
                filter_type=FilterType.NEWS,
                enabled=True,
                params={
                    "min_sentiment": -0.5,
                },
            ),
            FilterType.VOLATILITY: FilterConfig(
                filter_type=FilterType.VOLATILITY,
                enabled=True,
                params={
                    "max_volatility": 0.15,
                    "min_volatility": 0.005,
                },
            ),
            FilterType.REGIME: FilterConfig(
                filter_type=FilterType.REGIME,
                enabled=True,
                params={
                    "min_confidence": 60,
                },
            ),
            FilterType.CIRCUIT: FilterConfig(
                filter_type=FilterType.CIRCUIT,
                enabled=True,
                params={},
            ),
        }
    
    def update_filter_config(
        self,
        filter_type: FilterType,
        enabled: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update configuration for a specific filter.
        
        Args:
            filter_type: Type of filter to update
            enabled: Whether to enable/disable the filter
            params: New parameters for the filter
        """
        if filter_type not in self.filter_configs:
            self._logger.warning(f"Unknown filter type: {filter_type}")
            return
        
        config = self.filter_configs[filter_type]
        
        if enabled is not None:
            config.enabled = enabled
        
        if params is not None:
            config.params.update(params)
        
        self._logger.info(
            f"Updated filter config for {filter_type.value}",
            extra={"enabled": config.enabled, "params": config.params},
        )
    
    def get_filter_summary(self) -> Dict[str, Any]:
        """
        Get summary of all filter configurations.
        
        Returns:
            Dictionary with filter summary
        """
        return {
            filter_type.value: config.to_dict()
            for filter_type, config in self.filter_configs.items()
        }
