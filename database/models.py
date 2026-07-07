from sqlalchemy import Boolean, Column, DateTime, Integer, Numeric, String, Text, UniqueConstraint, Date, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import synonym

Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String(50), primary_key=True)
    symbol = Column(String(20), nullable=False)
    model_version = Column(String(50))
    feature_version = Column(String(50))
    # Full JSON snapshot of every feature value used at prediction time.
    # Stored inline (avg ~2 KB). Required for post-hoc audits and reproducibility.
    feature_snapshot = Column(Text)
    # Canonical feature schema version tag (e.g. "v2.1").
    # Lets us track which feature set a saved model was trained on.
    feature_schema_version = Column(String(20))
    features_used = Column("features_used", Text)     # kept for legacy compat
    prediction = Column("prediction", String(20), nullable=False)
    horizon = Column("horizon", String(20), nullable=False)
    confidence = Column("confidence", Numeric(6, 4))
    entry_price = Column(Numeric(12, 2))
    stop_loss = Column(Numeric(12, 2))
    target_price = Column(Numeric(12, 2))
    expected_return = Column("expected_return", Numeric(10, 6))
    prediction_time = Column("prediction_time", DateTime, nullable=False)
    expiry_time = Column(DateTime)
    actual_outcome = Column("actual_outcome", String(20))
    target_hit = Column(Boolean, default=False)
    stop_hit = Column(Boolean, default=False)
    actual_return = Column(Numeric(10, 4))
    mfe = Column(Numeric(10, 4))
    mae = Column(Numeric(10, 4))
    # Timestamp when the trade was resolved (SL/target hit or timeout).
    exit_time = Column(DateTime)
    # Number of bars from entry to exit (for duration analysis).
    hold_bars = Column(Integer)
    latency_ms = Column(Integer)
    is_correct = Column(Boolean)
    regime = Column(String(50))
    reason = Column(Text)

    # Uniqueness constraint: one prediction per (symbol, timestamp, model).
    # Prevents duplicate rows when the pipeline reruns on the same bar.
    __table_args__ = (
        UniqueConstraint(
            "symbol", "prediction_time", "model_version",
            name="uq_prediction_symbol_time_model",
        ),
    )

    # SQLAlchemy synonyms to map the attributes accessed in scripts
    generated_at = synonym("prediction_time")
    timeframe = synonym("horizon")
    direction = synonym("prediction")
    predicted_probability = synonym("confidence")
    features_json = synonym("features_used")
    outcome = synonym("actual_outcome")


class Tick(Base):
    __tablename__ = "ticks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(DateTime, nullable=False)
    symbol = Column(String(20), nullable=False)
    ltp = Column(Numeric(12, 2), nullable=False)
    volume = Column(Numeric(12, 2), default=0)


class IndexTick(Base):
    __tablename__ = "index_ticks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    name = Column(String(50), nullable=False)
    value = Column(Numeric(12, 2), nullable=False)
    change = Column(Numeric(5, 2))


class PaperTrade(Base):
    __tablename__ = "paper_trades"

    id = Column(String(50), primary_key=True)
    user_id = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # BUY or SELL
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Numeric(12, 2), nullable=False)
    exit_price = Column(Numeric(12, 2))
    entry_timestamp = Column(DateTime, nullable=False)
    exit_timestamp = Column(DateTime)
    status = Column(String(20), default="OPEN")  # OPEN, CLOSED, CANCELLED
    strategy_id = Column(String(50))  # Model/strategy that generated the trade
    pnl = Column(Numeric(12, 2))
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime)


class AIMarketOutlook(Base):
    __tablename__ = "ai_market_outlook"

    id = Column(String(50), primary_key=True)
    date = Column(Date, unique=True, nullable=False)
    market_regime = Column(String(50), nullable=False)
    risk_level = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    sector_rotation = Column(Text)  # JSON serialized array
    top_themes = Column(Text)       # JSON serialized array
    watchlist = Column(Text)        # JSON serialized array
    warnings = Column(Text)         # JSON serialized array
    raw_json = Column(Text)
    created_at = Column(DateTime, nullable=False)


class ModelPostmortem(Base):
    __tablename__ = "model_postmortem"

    id = Column(String(50), primary_key=True)
    date = Column(Date, unique=True, nullable=False)
    total_trades = Column(Integer, nullable=False)
    total_losses = Column(Integer, nullable=False)
    win_rate = Column(Float, nullable=False)
    analysis_json = Column(Text, nullable=False)
    recommendations = Column(Text)
    created_at = Column(DateTime, nullable=False)


