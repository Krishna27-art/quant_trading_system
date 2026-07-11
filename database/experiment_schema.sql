-- Experiment Tracker Database Schema
-- Stores all experiment tracking data for the research platform

-- Research Projects Table
-- Stores research project metadata
CREATE TABLE IF NOT EXISTS research_projects (
    project_id String,
    name String,
    project_type String,
    description String,
    created_by String,
    created_at DateTime,
    status String,
    priority String,
    tags Array(String),
    notes String,
    local_recv_timestamp DateTime DEFAULT now(),
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (project_id);

-- Experiments Table
-- Stores experiment metadata
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id String,
    project_id String,
    name String,
    experiment_type String,
    purpose String,
    research_question String,
    created_by String,
    created_at DateTime,
    status String,
    priority String,
    decision Nullable(String),
    tags Array(String),
    notes String,
    local_recv_timestamp DateTime DEFAULT now(),
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (experiment_id);

-- Dataset Snapshots Table
-- Stores dataset version information
CREATE TABLE IF NOT EXISTS dataset_snapshots (
    snapshot_id String,
    experiment_id String,
    dataset_name String,
    version String,
    rows UInt32,
    features UInt32,
    date_range String,
    symbols Array(String),
    created_at DateTime,
    local_recv_timestamp DateTime DEFAULT now(),
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (snapshot_id);

-- Feature Snapshots Table
-- Stores feature snapshot information
CREATE TABLE IF NOT EXISTS feature_snapshots (
    snapshot_id String,
    experiment_id String,
    feature_names Array(String),
    feature_count UInt32,
    feature_types Map(String, String),
    created_at DateTime,
    local_recv_timestamp DateTime DEFAULT now(),
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (snapshot_id);

-- Signal Snapshots Table
-- Stores signal configuration information
CREATE TABLE IF NOT EXISTS signal_snapshots (
    snapshot_id String,
    experiment_id String,
    signal_versions Map(String, String),
    signal_weights Map(String, Float32),
    created_at DateTime,
    local_recv_timestamp DateTime DEFAULT now(),
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (snapshot_id);

-- Alpha Snapshots Table
-- Stores alpha configuration information
CREATE TABLE IF NOT EXISTS alpha_snapshots (
    snapshot_id String,
    experiment_id String,
    alpha_version String,
    alpha_weights Map(String, Float32),
    alpha_rules Array(String),
    created_at DateTime,
    local_recv_timestamp DateTime DEFAULT now(),
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (snapshot_id);

-- Hyperparameters Table
-- Stores hyperparameters used in experiments
CREATE TABLE IF NOT EXISTS hyperparameters (
    hyperparameters_id String,
    experiment_id String,
    parameters Map(String, String),
    created_at DateTime,
    local_recv_timestamp DateTime DEFAULT now(),
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (hyperparameters_id);

-- Training Metrics Table
-- Stores training metrics from model training
CREATE TABLE IF NOT EXISTS training_metrics (
    metrics_id String,
    experiment_id String,
    accuracy Float32,
    precision Float32,
    recall Float32,
    f1_score Float32,
    roc_auc Float32,
    log_loss Float32,
    created_at DateTime,
    local_recv_timestamp DateTime DEFAULT now(),
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (metrics_id);

-- Trading Metrics Table
-- Stores trading metrics from backtesting
CREATE TABLE IF NOT EXISTS trading_metrics (
    metrics_id String,
    experiment_id String,
    win_rate Float32,
    average_return Float32,
    average_loss Float32,
    profit_factor Float32,
    sharpe_ratio Float32,
    sortino_ratio Float32,
    max_drawdown Float32,
    expectancy Float32,
    calmar_ratio Float32,
    total_trades UInt32,
    created_at DateTime,
    local_recv_timestamp DateTime DEFAULT now(),
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (metrics_id);

-- Regime Performance Table
-- Stores performance by market regime
CREATE TABLE IF NOT EXISTS regime_performance (
    performance_id String,
    experiment_id String,
    regime String,
    win_rate Float32,
    sharpe_ratio Float32,
    total_trades UInt32,
    created_at DateTime,
    local_recv_timestamp DateTime DEFAULT now(),
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (performance_id);

-- Sector Performance Table
-- Stores performance by sector
CREATE TABLE IF NOT EXISTS sector_performance (
    performance_id String,
    experiment_id String,
    sector String,
    win_rate Float32,
    sharpe_ratio Float32,
    total_trades UInt32,
    created_at DateTime,
    local_recv_timestamp DateTime DEFAULT now(),
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (performance_id);

-- Feature Importance Table
-- Stores feature importance data
CREATE TABLE IF NOT EXISTS feature_importance (
    importance_id String,
    experiment_id String,
    top_features Map(String, Float32),
    worst_features Map(String, Float32),
    method String,
    created_at DateTime,
    local_recv_timestamp DateTime DEFAULT now(),
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (importance_id);

-- Research Notes Table
-- Stores research notes for experiments
CREATE TABLE IF NOT EXISTS research_notes (
    note_id String,
    experiment_id String,
    author String,
    content String,
    created_at DateTime,
    local_recv_timestamp DateTime DEFAULT now(),
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (note_id);

-- LLM Summaries Table
-- Stores LLM-generated summaries
CREATE TABLE IF NOT EXISTS llm_summaries (
    summary_id String,
    experiment_id String,
    best_model String,
    best_alpha String,
    weakest_feature String,
    recommendation String,
    insights Array(String),
    created_at DateTime,
    local_recv_timestamp DateTime DEFAULT now(),
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (summary_id);

-- Decision History Table
-- Stores decision history for experiments
CREATE TABLE IF NOT EXISTS decision_history (
    experiment_id String,
    decision String,
    reason String,
    timestamp DateTime,
    decided_by Nullable(String),
    manual Bool DEFAULT false,
    local_recv_timestamp DateTime DEFAULT now(),
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (experiment_id);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_research_projects_type ON research_projects (project_type);
CREATE INDEX IF NOT EXISTS idx_research_projects_status ON research_projects (status);
CREATE INDEX IF NOT EXISTS idx_experiments_project ON experiments (project_id);
CREATE INDEX IF NOT EXISTS idx_experiments_type ON experiments (experiment_type);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments (status);
CREATE INDEX IF NOT EXISTS idx_dataset_snapshots_experiment ON dataset_snapshots (experiment_id);
CREATE INDEX IF NOT EXISTS idx_feature_snapshots_experiment ON feature_snapshots (experiment_id);
CREATE INDEX IF NOT EXISTS idx_signal_snapshots_experiment ON signal_snapshots (experiment_id);
CREATE INDEX IF NOT EXISTS idx_alpha_snapshots_experiment ON alpha_snapshots (experiment_id);
CREATE INDEX IF NOT EXISTS idx_hyperparameters_experiment ON hyperparameters (experiment_id);
CREATE INDEX IF NOT EXISTS idx_training_metrics_experiment ON training_metrics (experiment_id);
CREATE INDEX IF NOT EXISTS idx_trading_metrics_experiment ON trading_metrics (experiment_id);
CREATE INDEX IF NOT EXISTS idx_regime_performance_experiment ON regime_performance (experiment_id);
CREATE INDEX IF NOT EXISTS idx_sector_performance_experiment ON sector_performance (experiment_id);
CREATE INDEX IF NOT EXISTS idx_feature_importance_experiment ON feature_importance (experiment_id);
CREATE INDEX IF NOT EXISTS idx_research_notes_experiment ON research_notes (experiment_id);
CREATE INDEX IF NOT EXISTS idx_llm_summaries_experiment ON llm_summaries (experiment_id);
CREATE INDEX IF NOT EXISTS idx_decision_history_experiment ON decision_history (experiment_id);
