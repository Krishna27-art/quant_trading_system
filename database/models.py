from sqlalchemy import Boolean, Column, DateTime, Integer, Numeric, String, Text, UniqueConstraint, Date, Float, func, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import synonym
from datetime import datetime

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


class Feature(Base):
    """
    Feature Laboratory - Main feature storage table.
    
    Stores computed feature values for every symbol and timestamp.
    One row = one stock + one day + one feature value.
    """
    __tablename__ = "features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    feature_name = Column(String(100), nullable=False, index=True)
    feature_category = Column(String(50), nullable=False, index=True)
    feature_version = Column(String(20), default="1.0")
    feature_value = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    
    # Uniqueness constraint: one value per (symbol, date, feature_name, version)
    __table_args__ = (
        UniqueConstraint(
            "symbol", "date", "feature_name", "feature_version",
            name="uq_feature_symbol_date_name_version",
        ),
        Index('ix_features_symbol_date', 'symbol', 'date'),
        Index('ix_features_feature_name_date', 'feature_name', 'date'),
    )


class FeatureMetadata(Base):
    """
    Feature Laboratory - Feature metadata registry.
    
    Stores documentation and metadata for every feature definition.
    """
    __tablename__ = "feature_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    feature_name = Column(String(100), unique=True, nullable=False, index=True)
    feature_category = Column(String(50), nullable=False)
    description = Column(Text, nullable=False)
    timeframe = Column(String(10), nullable=False)
    required_columns = Column(Text, nullable=False)  # JSON array
    output_range = Column(String(50))
    version = Column(String(20), default="1.0")
    author = Column(String(100), default="system")
    computation_method = Column(Text)
    assumptions = Column(Text)
    limitations = Column(Text)
    references = Column(Text)
    is_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    last_updated = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)


class FeatureQuality(Base):
    """
    Feature Laboratory - Feature quality scores.
    
    Tracks predictive quality of each feature over time.
    """
    __tablename__ = "feature_quality"

    id = Column(Integer, primary_key=True, autoincrement=True)
    feature_name = Column(String(100), nullable=False, index=True)
    feature_version = Column(String(20), default="1.0")
    quality_score = Column(Float, nullable=False)  # 0-100
    win_rate = Column(Float)
    average_return = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    profit_factor = Column(Float)
    sample_size = Column(Integer, nullable=False)
    evaluation_period_start = Column(Date, nullable=False)
    evaluation_period_end = Column(Date, nullable=False)
    computed_at = Column(DateTime, nullable=False, default=datetime.now)
    
    __table_args__ = (
        UniqueConstraint(
            "feature_name", "feature_version", "evaluation_period_end",
            name="uq_feature_quality_version_period",
        ),
    )


class FeatureImportance(Base):
    """
    Feature Laboratory - ML feature importance tracking.
    
    Stores feature importance scores from ML models.
    """
    __tablename__ = "feature_importance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    feature_name = Column(String(100), nullable=False)
    importance_score = Column(Float, nullable=False)
    rank = Column(Integer)
    computed_at = Column(DateTime, nullable=False, default=datetime.now)
    
    __table_args__ = (
        UniqueConstraint(
            "model_name", "model_version", "feature_name",
            name="uq_feature_importance_model_feature",
        ),
    )


class FeatureCorrelation(Base):
    """
    Feature Laboratory - Feature correlation matrix.
    
    Stores pairwise correlations between features to identify redundancy.
    """
    __tablename__ = "feature_correlation"

    id = Column(Integer, primary_key=True, autoincrement=True)
    feature_1 = Column(String(100), nullable=False, index=True)
    feature_2 = Column(String(100), nullable=False, index=True)
    correlation_coefficient = Column(Float, nullable=False)
    p_value = Column(Float)
    sample_size = Column(Integer)
    computed_at = Column(DateTime, nullable=False, default=datetime.now)
    
    __table_args__ = (
        UniqueConstraint(
            "feature_1", "feature_2",
            name="uq_feature_correlation_pair",
        ),
    )


class FeatureCombination(Base):
    """
    Feature Laboratory - Feature combinations for alpha research.
    
    Stores tested feature combinations and their performance.
    """
    __tablename__ = "feature_combinations"

    id = Column(String(50), primary_key=True)
    combination_name = Column(String(100), nullable=False)
    features = Column(Text, nullable=False)  # JSON array of feature names
    conditions = Column(Text, nullable=False)  # JSON array of conditions
    win_rate = Column(Float)
    average_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    sample_size = Column(Integer)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    last_tested = Column(DateTime)
    notes = Column(Text)


