from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ticker = Column(String, index=True)
    strategy = Column(String, index=True)
    entry_price = Column(Float)

    # Core multi-objective predictions
    predicted_direction = Column(String)  # BUY/SELL/HOLD
    predicted_return = Column(Float)
    predicted_mfe = Column(Float)
    predicted_mae = Column(Float)

    # Target / Limits
    stop_loss = Column(Float)
    tp1 = Column(Float)
    tp2 = Column(Float)
    tp3 = Column(Float)
    expected_holding_time = Column(Integer)  # Bars

    # Confidence and Context
    raw_probability = Column(Float)
    calibrated_confidence = Column(Float)
    risk_score = Column(Float)
    market_regime = Column(String)

    # Metadata
    feature_version = Column(String)
    model_version = Column(String)
    reason_codes = Column(String)


class LabelDatabase(Base):
    __tablename__ = "label_database"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    entry_time = Column(DateTime, index=True)
    entry_price = Column(Float)

    # Forward simulation outcome
    target_hit = Column(Boolean)
    stoploss_hit = Column(Boolean)
    time_expired = Column(Boolean)

    # True multi-objective values
    actual_return = Column(Float)
    actual_mfe = Column(Float)
    actual_mae = Column(Float)
    actual_duration = Column(Integer)  # Bars

    exit_time = Column(DateTime)
    exit_price = Column(Float)
    winner = Column(Boolean)


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    entry_time = Column(DateTime)
    entry_price = Column(Float)
    exit_time = Column(DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)


class SymbolMaster(Base):
    __tablename__ = "symbol_master"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True)
    isin = Column(String, unique=True, nullable=True)
    asset_class = Column(String, index=True)  # EQUITY, FUTURES, OPTIONS, INDEX
    exchange = Column(String, default="NSE")
    lot_size = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    added_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    data_quality_score = Column(Float, default=100.0)


class MarketCalendar(Base):
    __tablename__ = "market_calendar"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, unique=True, index=True)
    is_holiday = Column(Boolean, default=False)
    holiday_name = Column(String, nullable=True)
    is_half_day = Column(Boolean, default=False)
    market_open = Column(DateTime, nullable=True)
    market_close = Column(DateTime, nullable=True)


class FeatureRegistry(Base):
    __tablename__ = "feature_registry"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String, unique=True, index=True)
    version = Column(Integer, default=1)
    owner = Column(String, default="Research")
    input_dependencies = Column(String)  # JSON string of input columns
    parameters = Column(String)  # JSON string of parameters
    description = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    version = Column(Integer, default=1)
    model_type = Column(String)  # Tree, DeepLearning, Meta, RL
    target = Column(String)  # Return, Probability, HoldTime
    features_version = Column(String)
    git_commit = Column(String, nullable=True)
    hyperparameters = Column(String)  # JSON string
    metrics = Column(String)  # JSON string (Accuracy, Sharpe, etc.)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)
