from sqlalchemy import Boolean, Column, DateTime, Integer, Numeric, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import synonym

Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String(50), primary_key=True)
    symbol = Column(String(20), nullable=False)
    model_version = Column(String(50))
    feature_version = Column(String(50))
    features_used = Column("features_used", Text)
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
    latency_ms = Column(Integer)
    is_correct = Column(Boolean)
    regime = Column(String(50))
    reason = Column(Text)

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
