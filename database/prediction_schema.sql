-- Prediction Table Schema
-- Single source of truth for all predictions emitted by the model pipeline.
-- Must stay in sync with database/models.py Prediction ORM class.
--
-- Status lifecycle:  OPEN -> WIN | LOSS | TIMEOUT
--
-- Every column maps 1:1 to a PredictionRecord field in validation/prediction_record.py.

CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    model_version TEXT NOT NULL,
    feature_version TEXT,
    -- feature_schema_version: e.g. "v2.1" — which canonical feature set was used
    feature_schema_version TEXT,
    -- Full JSON of every feature value at prediction time (~2 KB avg).
    -- Written once at prediction time, never updated.
    feature_snapshot TEXT,
    features_used TEXT,                     -- legacy compat
    prediction TEXT NOT NULL,               -- "BUY" | "SELL"
    horizon TEXT NOT NULL,                  -- "INTRADAY" | "SWING" | "LONGTERM"
    confidence REAL,                        -- calibrated win probability [0,1]
    expected_return REAL,
    entry_price REAL,
    stop_loss REAL,
    target_price REAL,
    prediction_time TIMESTAMPTZ NOT NULL,   -- IST-aware
    expiry_time TIMESTAMPTZ,
    exit_time TIMESTAMPTZ,                  -- when SL/target hit or timeout
    regime TEXT,
    reason TEXT,
    latency_ms INTEGER,
    actual_outcome TEXT DEFAULT 'OPEN',     -- "WIN" | "LOSS" | "TIMEOUT" | "OPEN"
    target_hit BOOLEAN DEFAULT FALSE,
    stop_hit BOOLEAN DEFAULT FALSE,
    actual_return REAL,
    mfe REAL,
    mae REAL,
    hold_bars INTEGER,                      -- bars from entry to exit
    is_correct BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_prediction_symbol_time_model
        UNIQUE (symbol, prediction_time, model_version)
);

CREATE INDEX IF NOT EXISTS idx_predictions_symbol   ON predictions(symbol);
CREATE INDEX IF NOT EXISTS idx_predictions_time     ON predictions(prediction_time DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_outcome  ON predictions(actual_outcome);
CREATE INDEX IF NOT EXISTS idx_predictions_model    ON predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_horizon  ON predictions(horizon);
CREATE INDEX IF NOT EXISTS idx_predictions_open
    ON predictions(actual_outcome, prediction_time) WHERE actual_outcome = 'OPEN';

-- Upgrade DDL for existing databases (idempotent):
--   ALTER TABLE predictions ADD COLUMN IF NOT EXISTS feature_snapshot TEXT;
--   ALTER TABLE predictions ADD COLUMN IF NOT EXISTS feature_schema_version TEXT;
--   ALTER TABLE predictions ADD COLUMN IF NOT EXISTS exit_time TIMESTAMPTZ;
--   ALTER TABLE predictions ADD COLUMN IF NOT EXISTS hold_bars INTEGER;

