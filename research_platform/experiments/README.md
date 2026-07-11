# Experiment Tracker

The research memory of the entire quant platform. Tracks every research attempt systematically.

## Architecture

The Experiment Tracker sits after backtesting and before the research database:

```
Market Data
      ↓
Feature Laboratory
      ↓
Signal Engine
      ↓
Alpha Engine
      ↓
Prediction
      ↓
Backtesting
      ↓
Experiment Tracker ⭐
      ↓
Research Database
      ↓
Learning Engine
```

## What is an Experiment?

Every research attempt is an experiment, not just model training:

- **Feature Research**: "Does Delivery % improve Swing Trading?"
- **Signal Research**: "Does OI Change improve Intraday?"
- **Model Research**: "CatBoost vs XGBoost"
- **Alpha Research**: "Should Alpha Weight of Volume be 25%?"
- **Regime Research**: "How does the model perform in bear markets?"

## Research Projects

Group experiments into projects:

```
projects/
├── Intraday
├── Swing
├── LongTerm
├── FeatureResearch
├── ModelResearch
├── AlphaResearch
├── SignalResearch
└── RiskResearch
```

## Usage

### Create a Research Project

```python
from research_platform.experiments import ProjectManager, ProjectType

project_manager = ProjectManager()

project = project_manager.create_project(
    name="Swing Trading Alpha Research",
    project_type=ProjectType.SWING,
    description="Research alpha combinations for swing trading",
    created_by="researcher",
    priority=ExperimentPriority.HIGH,
)
```

### Run an Experiment with MLflow

```python
from research_platform.experiments import (
    ExperimentRunner,
    Experiment,
    ExperimentType,
    ExperimentStatus,
)

runner = ExperimentRunner()

experiment = Experiment(
    experiment_id="EXP-001",
    project_id="PROJ-001",
    name="Volume Spike Test",
    experiment_type=ExperimentType.FEATURE_RESEARCH,
    purpose="Test if volume spike improves swing trading",
    research_question="Does volume spike improve win rate?",
    created_by="researcher",
)

with runner.run_experiment(experiment):
    # Your experiment code here
    runner.log_parameters({'volume_threshold': 2.0, 'lookback': 5})
    runner.log_trading_metrics(
        win_rate=0.68,
        sharpe_ratio=1.8,
        max_drawdown=12.0,
        # ... other metrics
    )
```

### Track Dataset Version

```python
from research_platform.experiments import DatasetVersionTracker

dataset_tracker = DatasetVersionTracker()

snapshot = dataset_tracker.create_snapshot(
    experiment_id="EXP-001",
    dataset_name="NSE_Equity",
    df=your_dataframe,
    date_column="date",
    symbol_column="symbol",
)
```

### Track Feature Snapshot

```python
from research_platform.experiments import FeatureSnapshotManager

feature_tracker = FeatureSnapshotManager()

snapshot = feature_tracker.create_snapshot(
    experiment_id="EXP-001",
    feature_names=['RSI', 'MACD', 'Volume', 'ATR', 'OBV'],
    df=your_dataframe,
)
```

### Track Signal Configuration

```python
from research_platform.experiments import SignalSnapshotManager

signal_tracker = SignalSnapshotManager()

snapshot = signal_tracker.create_snapshot(
    experiment_id="EXP-001",
    signal_versions={
        'Trend': 'V3',
        'Volume': 'V2',
        'Options': 'V5',
    },
    signal_weights={
        'Trend': 0.30,
        'Volume': 0.25,
        'Options': 0.20,
    },
)
```

### Track Alpha Configuration

```python
from research_platform.experiments import AlphaSnapshotManager

alpha_tracker = AlphaSnapshotManager()

snapshot = alpha_tracker.create_snapshot(
    experiment_id="EXP-001",
    alpha_version="V4",
    alpha_weights={
        'Trend': 0.30,
        'Volume': 0.25,
        'Options': 0.20,
        'Fundamental': 0.15,
        'Sentiment': 0.10,
    },
    alpha_rules=[
        'Trend > 70 AND Volume > 65',
        'Direction agreement required',
    ],
)
```

### Log Metrics

```python
from research_platform.experiments import MetricsLogger

metrics_logger = MetricsLogger()

# Training metrics
training_metrics = metrics_logger.log_training_metrics(
    experiment_id="EXP-001",
    accuracy=0.72,
    precision=0.68,
    recall=0.71,
    f1_score=0.69,
    roc_auc=0.78,
    log_loss=0.65,
)

# Trading metrics
trading_metrics = metrics_logger.log_trading_metrics(
    experiment_id="EXP-001",
    win_rate=0.68,
    average_return=5.2,
    average_loss=-3.1,
    profit_factor=1.8,
    sharpe_ratio=1.8,
    sortino_ratio=2.1,
    max_drawdown=12.0,
    expectancy=1.2,
    calmar_ratio=0.43,
    total_trades=150,
)
```

### Track Regime and Sector Performance

```python
from research_platform.experiments import RegimeSectorPerformanceTracker

tracker = RegimeSectorPerformanceTracker()

# Regime performance
tracker.log_regime_performance(
    experiment_id="EXP-001",
    regime="bull",
    returns=[5.2, 3.1, 7.8, 4.5, 6.2],
)

tracker.log_regime_performance(
    experiment_id="EXP-001",
    regime="bear",
    returns=[-2.1, -1.5, -3.2, -0.8, -2.5],
)

# Sector performance
tracker.log_sector_performance(
    experiment_id="EXP-001",
    sector="IT",
    returns=[6.2, 5.8, 7.1, 4.9, 6.5],
)
```

### Track Feature Importance

```python
from research_platform.experiments import FeatureImportanceTracker

importance_tracker = FeatureImportanceTracker()

importance_tracker.log_feature_importance(
    experiment_id="EXP-001",
    feature_importance={
        'RSI': 0.15,
        'MACD': 0.08,
        'Volume': 0.22,
        'ATR': 0.12,
        'OBV': 0.18,
    },
    method="gain",
)
```

### Generate Charts

```python
from research_platform.experiments import ChartsGenerator

charts = ChartsGenerator(output_dir="./experiment_charts")

# Equity curve
charts.generate_equity_curve(
    returns=[5.2, -1.5, 3.8, 2.1, -0.5],
    experiment_id="EXP-001",
)

# ROC curve
charts.generate_roc_curve(
    y_true=[1, 0, 1, 1, 0],
    y_scores=[0.8, 0.3, 0.7, 0.9, 0.4],
    experiment_id="EXP-001",
)

# Feature importance plot
charts.generate_feature_importance_plot(
    feature_importance={
        'RSI': 0.15,
        'MACD': 0.08,
        'Volume': 0.22,
    },
    experiment_id="EXP-001",
)
```

### Add Research Notes

```python
from research_platform.experiments import ResearchNotesManager

notes_manager = ResearchNotesManager()

note = notes_manager.create_note(
    experiment_id="EXP-001",
    author="researcher",
    content="Volume spike significantly improved swing trading performance. "
            "Win rate increased from 58% to 68%. Need to test with Bank Nifty.",
)
```

### Generate LLM Summary

```python
from research_platform.experiments import LLMSummaryGenerator

llm_generator = LLMSummaryGenerator()

summary = llm_generator.generate_summary(
    experiment_id="EXP-001",
    trading_metrics={
        'win_rate': 0.68,
        'sharpe_ratio': 1.8,
        'max_drawdown': 12.0,
    },
    feature_importance={
        'RSI': 0.15,
        'MACD': 0.08,
        'Volume': 0.22,
    },
)
```

### Compare Experiments

```python
from research_platform.experiments import ExperimentComparisonEngine

comparison_engine = ExperimentComparisonEngine()

comparison = comparison_engine.compare_experiments(
    experiments=[exp1, exp2, exp3],
    trading_metrics_map={
        'EXP-001': metrics1,
        'EXP-002': metrics2,
        'EXP-003': metrics3,
    },
)

print(comparison['rankings']['by_sharpe'])
```

### Make Decisions

```python
from research_platform.experiments import DecisionEngine

decision_engine = DecisionEngine()

decision = decision_engine.make_decision(
    experiment=experiment,
    trading_metrics=trading_metrics,
    min_sharpe=1.5,
    min_win_rate=0.55,
    max_drawdown=15.0,
)

# Or manual decision
decision_engine.manual_decision(
    experiment_id="EXP-001",
    decision=ExperimentDecision.ACCEPTED,
    reason="Excellent performance across all metrics",
    decided_by="lead_researcher",
)
```

## Database Schema

Initialize the database schema:

```bash
clickhouse-client --database quant < database/experiment_schema.sql
```

Tables:
- `research_projects` - Project metadata
- `experiments` - Experiment metadata
- `dataset_snapshots` - Dataset version tracking
- `feature_snapshots` - Feature tracking
- `signal_snapshots` - Signal configuration tracking
- `alpha_snapshots` - Alpha configuration tracking
- `hyperparameters` - Hyperparameter tracking
- `training_metrics` - Training metrics
- `trading_metrics` - Trading metrics
- `regime_performance` - Regime performance
- `sector_performance` - Sector performance
- `feature_importance` - Feature importance
- `research_notes` - Research notes
- `llm_summaries` - LLM-generated summaries
- `decision_history` - Decision tracking

## Key Features

1. **Reproducibility**: Every experiment tracks dataset version, features, signals, alpha, and hyperparameters
2. **MLflow Integration**: Automatic tracking with MLflow for parameters, metrics, and artifacts
3. **Comprehensive Metrics**: Both training metrics (accuracy, precision, etc.) and trading metrics (Sharpe, drawdown, etc.)
4. **Regime & Sector Analysis**: Track performance by market regime and sector
5. **Feature Importance**: Automatic tracking of feature importance
6. **Visualization**: Generate equity curves, ROC curves, SHAP plots, and more
7. **Research Notes**: Capture learnings and conclusions
8. **LLM Summaries**: AI-powered insights and recommendations
9. **Experiment Comparison**: Side-by-side comparison of experiments
10. **Decision Engine**: Automated or manual decision making

## Research Workflow

```
Research Idea
        ↓
Create Project
        ↓
Create Experiment
        ↓
Run Experiment (with MLflow)
        ↓
Log Dataset Version
        ↓
Log Feature Snapshot
        ↓
Log Signal Snapshot
        ↓
Log Alpha Snapshot
        ↓
Log Hyperparameters
        ↓
Log Training Metrics
        ↓
Log Trading Metrics
        ↓
Log Regime/Sector Performance
        ↓
Log Feature Importance
        ↓
Generate Charts
        ↓
Add Research Notes
        ↓
Generate LLM Summary
        ↓
Compare with Other Experiments
        ↓
Make Decision
        ↓
Knowledge Base
```

## Best Practices

1. **Always create projects** to group related experiments
2. **Track everything**: dataset, features, signals, alpha, hyperparameters
3. **Log both training and trading metrics** - trading metrics matter more
4. **Add research notes** after every experiment
5. **Compare experiments** before making decisions
6. **Use MLflow** for automatic tracking
7. **Generate charts** for visualization
8. **Track regime and sector performance** to understand where the model works
9. **Make decisions** - don't leave experiments unfinished
10. **Review LLM summaries** for AI-powered insights

## Next Steps

After building the Experiment Tracker, the next modules are:
- **Research Database**: Persistent storage of all research data
- **Learning Engine**: Continuous improvement from experiment results
