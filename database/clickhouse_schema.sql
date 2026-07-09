
-- Source: data_infra/storage/database/clickhouse_schema.sql
-- ============================================================================
-- CLICKHOUSE SCHEMA - Production Database
-- ============================================================================
-- High-performance, real-time analytics database
-- Architecture: NSE → Bronze Parquet → ClickHouse → Feature Store → Models
-- ============================================================================

-- Create database
CREATE DATABASE IF NOT EXISTS market;

USE market;

-- ============================================================================
-- SECURITY MASTER
-- ============================================================================

CREATE TABLE IF NOT EXISTS security_master (
    symbol String,
    isin String,
    company_name String,
    sector String,
    industry String,
    listing_date Date,
    delisting_date Date,
    face_value Float32,
    market_cap_bucket String,
    nse_symbol String,
    bse_symbol String,
    lot_size UInt32,
    index_memberships String,
    effective_from Date,
    effective_to Date,
    is_current UInt8,
    created_at DateTime DEFAULT now(),
    updated_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(effective_from)
ORDER BY (symbol, effective_from)
SETTINGS index_granularity = 8192;

-- ============================================================================
-- EQUITY HISTORY
-- ============================================================================

CREATE TABLE IF NOT EXISTS equity_history (
    symbol String,
    date Date,
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume UInt64,
    vwap Float64,
    trades UInt32,
    is_degraded UInt8 DEFAULT 0,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(date)
ORDER BY (symbol, date)
SETTINGS index_granularity = 8192;

-- ============================================================================
-- OPTIONS CHAIN
-- ============================================================================

CREATE TABLE IF NOT EXISTS options_chain (
    timestamp DateTime,
    symbol String,
    strike Float64,
    expiry Date,
    option_type String,
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume UInt64,
    oi UInt64,
    iv Float32,
    delta Float32,
    gamma Float32,
    theta Float32,
    vega Float32,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMonth(timestamp)
ORDER BY (symbol, expiry, strike, option_type, timestamp)
SETTINGS index_granularity = 8192;

-- ============================================================================
-- CORPORATE ACTIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol String,
    event_type String,
    announcement_date Date,
    record_date Date,
    ex_date Date,
    ratio Float32,
    dividend_amount Float64,
    rights_ratio Float32,
    rights_price Float64,
    buyback_price Float64,
    buyback_quantity UInt32,
    old_symbol String,
    new_symbol String,
    currency String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(announcement_date)
ORDER BY (symbol, announcement_date, event_type)
SETTINGS index_granularity = 8192;

-- ============================================================================
-- MARKET CALENDAR
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_calendar (
    date Date,
    is_trading_day UInt8,
    is_weekend UInt8,
    is_holiday UInt8,
    holiday_name String,
    holiday_type String,
    session_type String,
    session_start Time,
    session_end Time,
    is_expiry UInt8,
    shifted_expiry_date Date,
    is_revised UInt8,
    original_holiday_date Date,
    revision_reason String,
    description String,
    created_at DateTime DEFAULT now(),
    updated_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY date
SETTINGS index_granularity = 8192;

-- ============================================================================
-- UNIVERSE MEMBERS
-- ============================================================================

CREATE TABLE IF NOT EXISTS universe_members (
    id String,
    symbol String,
    isin String,
    company_name String,
    start_date Date,
    end_date Date,
    is_active UInt8,
    exit_reason String,
    sector String,
    market_cap Float64,
    listing_date Date,
    added_at DateTime,
    updated_at DateTime
) ENGINE = MergeTree()
ORDER BY (symbol, start_date)
SETTINGS index_granularity = 8192;

-- ============================================================================
-- UNIVERSE HISTORY
-- ============================================================================

CREATE TABLE IF NOT EXISTS universe_history (
    id String,
    snapshot_date Date,
    total_symbols UInt32,
    active_symbols UInt32,
    new_entries UInt32,
    exits UInt32,
    symbols String,
    created_at DateTime
) ENGINE = MergeTree()
ORDER BY snapshot_date
SETTINGS index_granularity = 8192;

-- ============================================================================
-- DATA LINEAGE
-- ============================================================================

CREATE TABLE IF NOT EXISTS data_lineage (
    id String,
    dataset String,
    source String,
    downloaded_at DateTime,
    pipeline_version String,
    git_commit String,
    checksum String,
    row_count UInt32,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (dataset, downloaded_at)
SETTINGS index_granularity = 8192;

-- ============================================================================
-- POINT-IN-TIME TABLES
-- ============================================================================

-- Fundamental PIT
CREATE TABLE IF NOT EXISTS fundamental_pit (
    id String,
    symbol String,
    metric_name String,
    metric_value Float64,
    period String,
    period_end Date,
    available_at DateTime,
    valid_from DateTime,
    valid_to DateTime,
    entry_reason String,
    exit_reason String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(available_at)
ORDER BY (symbol, metric_name, available_at)
SETTINGS index_granularity = 8192;

-- Promoter Holding PIT
CREATE TABLE IF NOT EXISTS promoter_holding_pit (
    id String,
    symbol String,
    promoter_name String,
    holding_percentage Float32,
    shares_held UInt64,
    period_end Date,
    available_at DateTime,
    valid_from DateTime,
    valid_to DateTime,
    entry_reason String,
    exit_reason String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(available_at)
ORDER BY (symbol, available_at)
SETTINGS index_granularity = 8192;

-- Corporate Actions PIT
CREATE TABLE IF NOT EXISTS corporate_actions_pit (
    id String,
    symbol String,
    event_type String,
    announcement_date Date,
    record_date Date,
    ex_date Date,
    split_ratio Float32,
    bonus_ratio Float32,
    dividend_amount Float64,
    rights_ratio Float32,
    rights_price Float64,
    buyback_price Float64,
    buyback_quantity UInt32,
    old_symbol String,
    new_symbol String,
    currency String,
    is_revised UInt8,
    revision_reason String,
    available_at DateTime,
    valid_from DateTime,
    valid_to DateTime,
    entry_reason String,
    exit_reason String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(available_at)
ORDER BY (symbol, event_type, available_at)
SETTINGS index_granularity = 8192;

-- Index Membership PIT
CREATE TABLE IF NOT EXISTS index_membership_pit (
    id String,
    symbol String,
    index_name String,
    weight Float32,
    entry_date Date,
    exit_date Date,
    available_at DateTime,
    valid_from DateTime,
    valid_to DateTime,
    entry_reason String,
    exit_reason String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(available_at)
ORDER BY (symbol, index_name, available_at)
SETTINGS index_granularity = 8192;

-- ============================================================================
-- FEATURE STORE TABLES
-- ============================================================================

-- Price Features
CREATE TABLE IF NOT EXISTS price_features (
    symbol String,
    date Date,
    feature_name String,
    feature_value Float64,
    lookback UInt16,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(date)
ORDER BY (symbol, date, feature_name)
SETTINGS index_granularity = 8192;

-- Volatility Features
CREATE TABLE IF NOT EXISTS volatility_features (
    symbol String,
    date Date,
    feature_name String,
    feature_value Float64,
    lookback UInt16,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(date)
ORDER BY (symbol, date, feature_name)
SETTINGS index_granularity = 8192;

-- Volume Features
CREATE TABLE IF NOT EXISTS volume_features (
    symbol String,
    date Date,
    feature_name String,
    feature_value Float64,
    lookback UInt16,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(date)
ORDER BY (symbol, date, feature_name)
SETTINGS index_granularity = 8192;

-- Options Features
CREATE TABLE IF NOT EXISTS options_features (
    symbol String,
    date Date,
    feature_name String,
    feature_value Float64,
    lookback UInt16,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(date)
ORDER BY (symbol, date, feature_name)
SETTINGS index_granularity = 8192;

-- Flow Features
CREATE TABLE IF NOT EXISTS flow_features (
    symbol String,
    date Date,
    feature_name String,
    feature_value Float64,
    lookback UInt16,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(date)
ORDER BY (symbol, date, feature_name)
SETTINGS index_granularity = 8192;

-- Fundamental Features
CREATE TABLE IF NOT EXISTS fundamental_features (
    symbol String,
    date Date,
    feature_name String,
    feature_value Float64,
    lookback UInt16,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(date)
ORDER BY (symbol, date, feature_name)
SETTINGS index_granularity = 8192;

-- Alternative Features
CREATE TABLE IF NOT EXISTS alternative_features (
    symbol String,
    date Date,
    feature_name String,
    feature_value Float64,
    lookback UInt16,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(date)
ORDER BY (symbol, date, feature_name)
SETTINGS index_granularity = 8192;


