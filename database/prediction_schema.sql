-- Simplified Prediction Schema
-- Focused on prediction tracking without execution dependencies

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    prediction_time TIMESTAMPTZ NOT NULL,
    entry REAL NOT NULL,
    stoploss REAL NOT NULL,
    target1 REAL NOT NULL,
    target2 REAL NOT NULL,
    probability REAL NOT NULL,
    confidence REAL NOT NULL,
    model_version TEXT NOT NULL,
    status TEXT DEFAULT 'OPEN',  -- OPEN, TARGET1_HIT, TARGET2_HIT, SL_HIT, EXPIRED
    actual_exit REAL,            -- Actual exit price when prediction is resolved
    profit REAL,                 -- Actual profit/loss when prediction is resolved
    result TEXT,                 -- WIN, LOSS, BREAKEVEN, PENDING
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol);
CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(prediction_time DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions(status);
CREATE INDEX IF NOT EXISTS idx_predictions_result ON predictions(result);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_version);
