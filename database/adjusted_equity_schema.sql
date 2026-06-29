-- Adjusted Equity History Schema
-- Contains corporate action adjusted OHLCV data for institutional feature stores

CREATE TABLE IF NOT EXISTS adjusted_equity_history (
    symbol String,
    date Date,
    adj_open Float32,
    adj_high Float32,
    adj_low Float32,
    adj_close Float32,
    adj_volume Int32,
    split_factor Float32 DEFAULT 1.0,
    dividend_yield Float32 DEFAULT 0.0,
    exchange_timestamp DateTime, -- BITEMPORAL: Actual time of the trade/close
    local_recv_timestamp DateTime DEFAULT now(), -- BITEMPORAL: Time our system logged it
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (symbol, date);

CREATE TABLE IF NOT EXISTS events_multiplier_table (
    symbol String,
    ex_date Date,
    multiplier Float32,
    action_type String, -- 'SPLIT', 'BONUS', 'DIVIDEND'
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (symbol, ex_date, action_type);
