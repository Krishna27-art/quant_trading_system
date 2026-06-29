"""
Trading Engine Configuration

Dynamic configuration loaded from environment variables with sane defaults.
Supports runtime reload without restart.
"""

import os
from dataclasses import dataclass, field
from datetime import time as dtime
from enum import Enum


class ExecutionMode(str, Enum):
    """Trading execution modes."""

    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class MarketSession(str, Enum):
    """NSE/BSE market sessions."""

    PRE_OPEN = "pre_open"
    NORMAL = "normal"
    CLOSING = "closing"
    POST_CLOSE = "post_close"
    CLOSED = "closed"


@dataclass
class RiskLimits:
    """Risk management limits — adjustable at runtime."""

    max_daily_loss_pct: float = 0.02  # 2% of NAV
    max_weekly_loss_pct: float = 0.05  # 5% of NAV
    max_drawdown_pct: float = 0.10  # 10% MDD circuit breaker
    max_single_position_pct: float = 0.05  # 5% of NAV per position
    max_sector_exposure_pct: float = 0.25  # 25% per sector
    max_gross_exposure: float = 1.0  # 1.0 = no leverage
    max_trades_per_day: int = 50
    max_order_value_inr: float = 5_000_000  # ₹50 lakh fat-finger limit
    max_order_pct_of_adv: float = 0.10  # 10% of ADV
    max_price_deviation_atr: float = 3.0  # 3× ATR fat-finger
    min_reward_to_risk: float = 1.5  # Minimum R:R ratio
    overnight_position_reduction_pct: float = 0.50  # Cut 50% before close


@dataclass
class DataFeedConfig:
    """Data feed configuration."""

    primary_feed: str = "websocket"
    fallback_feed: str = "rest_polling"
    emergency_feed: str = "cached"
    staleness_threshold_seconds: float = 5.0
    tick_outlier_atr_multiple: float = 5.0
    reconnect_delay_seconds: float = 1.0
    max_reconnect_attempts: int = 10


@dataclass
class BrokerConfig:
    """Broker connection configuration."""

    primary_broker: str = os.environ.get("PRIMARY_BROKER", "zerodha")
    api_key: str = os.environ.get("BROKER_API_KEY", "")
    api_secret: str = os.environ.get("BROKER_API_SECRET", "")
    access_token: str = os.environ.get("BROKER_ACCESS_TOKEN", "")
    heartbeat_interval_seconds: float = 10.0
    order_timeout_seconds: float = 5.0
    max_retry_attempts: int = 3


@dataclass
class AlphaConfig:
    """Alpha signal configuration."""

    enabled_alphas: list[str] = field(
        default_factory=lambda: ["momentum", "mean_reversion", "microstructure", "intraday_setups"]
    )
    signal_decay_halflife_days: int = 5
    min_ic_threshold: float = 0.02
    rebalance_frequency: str = "daily"  # daily, weekly, intraday


@dataclass
class TradingConfig:
    """
    Master configuration for the trading engine.
    All values are sourced from environment variables with sane defaults.
    """

    # Core mode
    mode: ExecutionMode = ExecutionMode.PAPER

    # Market hours (IST) — NSE regular session
    market_open: dtime = dtime(9, 15)
    market_close: dtime = dtime(15, 30)
    pre_open_start: dtime = dtime(9, 0)
    pre_open_end: dtime = dtime(9, 8)
    position_cut_time: dtime = dtime(15, 15)  # Start reducing positions

    # Universe
    universe: str = os.environ.get("TRADING_UNIVERSE", "nifty200")
    instruments: list[str] = field(default_factory=lambda: ["NIFTY", "BANKNIFTY"])

    # Tick interval for the main loop
    tick_interval_seconds: float = 60.0  # 1-minute bars

    # Watchdog — kill process if main loop hangs
    watchdog_timeout_seconds: float = 300.0  # 5 minutes

    # Component heartbeat interval
    heartbeat_interval_seconds: float = 30.0

    # Sub-configs
    risk: RiskLimits = field(default_factory=RiskLimits)
    data_feed: DataFeedConfig = field(default_factory=DataFeedConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)

    # Logging
    log_level: str = os.environ.get("LOG_LEVEL", "INFO")

    # Database
    db_url: str = os.environ.get("DATABASE_URL", "")
    redis_url: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    @classmethod
    def from_env(cls) -> "TradingConfig":
        """Load configuration from environment variables."""
        mode_str = os.environ.get("TRADING_MODE", "paper").lower()
        mode = (
            ExecutionMode(mode_str)
            if mode_str in ExecutionMode.__members__.values()
            else ExecutionMode.PAPER
        )

        config = cls(mode=mode)

        # Override risk limits from env
        if os.environ.get("MAX_DAILY_LOSS_PCT"):
            config.risk.max_daily_loss_pct = float(os.environ["MAX_DAILY_LOSS_PCT"])
        if os.environ.get("MAX_DRAWDOWN_PCT"):
            config.risk.max_drawdown_pct = float(os.environ["MAX_DRAWDOWN_PCT"])
        if os.environ.get("MAX_GROSS_EXPOSURE"):
            config.risk.max_gross_exposure = float(os.environ["MAX_GROSS_EXPOSURE"])
        if os.environ.get("MAX_ORDER_VALUE_INR"):
            config.risk.max_order_value_inr = float(os.environ["MAX_ORDER_VALUE_INR"])

        # Override tick interval
        if os.environ.get("TICK_INTERVAL_SECONDS"):
            config.tick_interval_seconds = float(os.environ["TICK_INTERVAL_SECONDS"])

        return config

    def is_market_hours(self, current_time: dtime) -> bool:
        """Check if current time is within market hours."""
        return self.market_open <= current_time <= self.market_close

    def is_position_cut_time(self, current_time: dtime) -> bool:
        """Check if we should start reducing positions for close."""
        return current_time >= self.position_cut_time

    def validate(self) -> list[str]:
        """Validate configuration, return list of errors."""
        errors = []
        if self.mode == ExecutionMode.LIVE:
            if not self.broker.api_key:
                errors.append("BROKER_API_KEY is required for LIVE mode")
            if not self.broker.api_secret:
                errors.append("BROKER_API_SECRET is required for LIVE mode")
        if self.risk.max_daily_loss_pct <= 0:
            errors.append("max_daily_loss_pct must be positive")
        if self.risk.max_drawdown_pct <= 0:
            errors.append("max_drawdown_pct must be positive")
        if self.tick_interval_seconds < 1.0:
            errors.append("tick_interval_seconds must be >= 1.0")
        return errors
