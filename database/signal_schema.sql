-- Signal Engine Database Schema
-- Stores generated signals, performance metrics, and combination rules

-- Signal History Table
-- Stores individual signals for each symbol at each timestamp
CREATE TABLE IF NOT EXISTS signal_history (
    symbol String,
    timestamp DateTime,
    category String,  -- technical, volume, options, fundamental, sentiment, macro, sector, market
    score Float32,
    direction String,  -- BULLISH, BEARISH, NEUTRAL
    confidence Float32,
    reason String,
    raw_values String,  -- JSON string of raw values
    local_recv_timestamp DateTime DEFAULT now(),
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (symbol, timestamp, category);

-- Signal Set Table
-- Stores complete signal sets for each symbol at each timestamp
CREATE TABLE IF NOT EXISTS signal_sets (
    symbol String,
    timestamp DateTime,
    technical_score Nullable(Float32),
    technical_direction Nullable(String),
    technical_confidence Nullable(Float32),
    volume_score Nullable(Float32),
    volume_direction Nullable(String),
    volume_confidence Nullable(Float32),
    options_score Nullable(Float32),
    options_direction Nullable(String),
    options_confidence Nullable(Float32),
    fundamental_score Nullable(Float32),
    fundamental_direction Nullable(String),
    fundamental_confidence Nullable(Float32),
    sentiment_score Nullable(Float32),
    sentiment_direction Nullable(String),
    sentiment_confidence Nullable(Float32),
    macro_score Nullable(Float32),
    macro_direction Nullable(String),
    macro_confidence Nullable(Float32),
    sector_score Nullable(Float32),
    sector_direction Nullable(String),
    sector_confidence Nullable(Float32),
    market_score Nullable(Float32),
    market_direction Nullable(String),
    market_confidence Nullable(Float32),
    average_score Float32,
    dominant_direction String,
    local_recv_timestamp DateTime DEFAULT now(),
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (symbol, timestamp);

-- Signal Performance Table
-- Stores performance metrics for each signal category
CREATE TABLE IF NOT EXISTS signal_performance (
    category String,
    total_trades Int32,
    winning_trades Int32,
    losing_trades Int32,
    win_rate Float32,
    average_return Float32,
    average_win Float32,
    average_loss Float32,
    profit_factor Float32,
    max_drawdown Float32,
    sharpe_ratio Float32,
    sortino_ratio Float32,
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (category);

-- Trade Records Table
-- Stores individual trade records for performance tracking
CREATE TABLE IF NOT EXISTS trade_records (
    symbol String,
    signal_category String,
    signal_direction String,
    entry_price Float32,
    exit_price Float32,
    return_pct Float32,
    entry_time DateTime,
    exit_time DateTime,
    holding_period_days Int32,
    local_recv_timestamp DateTime DEFAULT now(),
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (signal_category, exit_time);

-- Signal Combination Rules Table
-- Stores tested signal combination rules and their performance
CREATE TABLE IF NOT EXISTS signal_combination_rules (
    rule_name String,
    rule_config String,  -- JSON string of rule configuration
    total_trades Int32,
    winning_trades Int32,
    losing_trades Int32,
    win_rate Float32,
    average_return Float32,
    profit_factor Float32,
    sharpe_ratio Float32,
    max_drawdown Float32,
    is_active Bool DEFAULT false,
    local_recv_timestamp DateTime DEFAULT now(),
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (rule_name);

-- Signal Filter Results Table
-- Stores filter results for audit trail
CREATE TABLE IF NOT EXISTS signal_filter_results (
    symbol String,
    timestamp DateTime,
    passed Bool,
    reason String,
    overall_score Float32,
    filtered_categories String,  -- JSON string of categories that passed
    local_recv_timestamp DateTime DEFAULT now(),
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (symbol, timestamp);

-- Signal Ranking Table
-- Stores ranking results for each timestamp
CREATE TABLE IF NOT EXISTS signal_rankings (
    symbol String,
    timestamp DateTime,
    rank Int32,
    overall_score Float32,
    category_diversity Float32,
    direction_agreement Float32,
    average_confidence Float32,
    local_recv_timestamp DateTime DEFAULT now(),
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (timestamp, rank);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_signal_history_symbol_timestamp ON signal_history (symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_signal_history_category ON signal_history (category);
CREATE INDEX IF NOT EXISTS idx_signal_sets_symbol_timestamp ON signal_sets (symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_signal_performance_category ON signal_performance (category);
CREATE INDEX IF NOT EXISTS idx_trade_records_category_time ON trade_records (signal_category, exit_time);
CREATE INDEX IF NOT EXISTS idx_combination_rules_active ON signal_combination_rules (is_active);
CREATE INDEX IF NOT EXISTS idx_rankings_timestamp_rank ON signal_rankings (timestamp, rank);
