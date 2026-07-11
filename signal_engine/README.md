# Signal Engine

The brain of the quant system that converts features into meaningful trading signals.

## Architecture

The Signal Engine sits between the Feature Laboratory and the Prediction Engine:

```
Market Data
      ↓
Feature Laboratory
      ↓
Signal Engine
      ↓
Alpha Engine (next module)
      ↓
Prediction Engine
      ↓
Confidence Engine
```

## Pipeline

```
Market Data
        │
        ▼
Feature Laboratory
        │
        ▼
Generate Technical Signals
        │
        ▼
Generate Volume Signals
        │
        ▼
Generate Options Signals
        │
        ▼
Generate Fundamental Signals
        │
        ▼
Generate Sentiment Signals
        │
        ▼
Score Every Signal
        │
        ▼
Filter Weak Signals
        │
        ▼
Rank Stocks
        │
        ▼
Top Candidates
        │
        ▼
Prediction Model
```

## Signal Categories

### Technical Signal
Analyzes trend, momentum, and breakout patterns.
- **Inputs**: EMA, ADX, RSI, ATR, Higher Highs, Higher Lows
- **Output**: Bullish, Neutral, or Bearish with score (0-100)

### Volume Signal
Analyzes volume patterns and accumulation.
- **Inputs**: Volume Spike, OBV, Accumulation/Distribution
- **Output**: Strong Buying, Normal, or Weak with score (0-100)

### Options Signal
Analyzes options chain data for sentiment.
- **Inputs**: PCR, OI Change, IV Rank, Max Pain
- **Output**: Bullish, Neutral, or Bearish with score (0-100)

### Fundamental Signal
Analyzes company fundamentals.
- **Inputs**: ROE, ROCE, Sales Growth, Debt, Cash Flow, Valuation
- **Output**: Excellent, Average, or Poor with score (0-100)

### Sentiment Signal
Analyzes market sentiment from news and earnings.
- **Inputs**: News sentiment, Earnings surprises, Analyst ratings
- **Output**: Positive, Neutral, or Negative with score (0-100)

## Usage

### Basic Signal Generation

```python
from signal_engine import SignalGenerator
import pandas as pd

# Initialize signal generator
generator = SignalGenerator()

# Generate signals for a single symbol
signal_set = generator.generate_signals(
    symbol="INFY",
    ohlcva_data=ohlcva_df,
    options_data=options_dict,
    fundamental_data=fundamental_dict,
    sentiment_data=sentiment_dict,
)

# Get signal dashboard
dashboard = generator.get_signal_dashboard(signal_set)
print(dashboard)
```

### Batch Signal Generation

```python
# Generate signals for multiple symbols
data_map = {
    "INFY": {
        "ohlcva": infy_df,
        "options": infy_options,
        "fundamental": infy_fundamental,
        "sentiment": infy_sentiment,
    },
    "TCS": {
        "ohlcva": tcs_df,
        "options": tcs_options,
        "fundamental": tcs_fundamental,
        "sentiment": tcs_sentiment,
    },
}

signal_sets = generator.generate_signals_for_multiple(data_map)
```

### Signal Processing Pipeline

```python
# Process signals through scoring, filtering, and ranking
results = generator.process_signals(
    signal_sets=signal_sets,
    top_n=10,
)

# Access results
print(f"Passed symbols: {results['passed_symbols']}")
print(f"Top candidates: {[r.symbol for r in results['top_candidates']]}")
print(f"Rejection stats: {results['rejection_stats']}")
```

### Performance Tracking

```python
from datetime import datetime
from signal_engine import SignalCategory

# Record trade outcomes
generator.record_trade_outcome(
    symbol="INFY",
    signal_category=SignalCategory.TECHNICAL,
    signal_direction="BULLISH",
    entry_price=1450.0,
    exit_price=1480.0,
    entry_time=datetime(2024, 1, 1),
    exit_time=datetime(2024, 1, 5),
)

# Get performance summary
performance = generator.get_performance_summary()
print(performance)
```

### Custom Configuration

```python
from signal_engine import (
    ScoringConfig,
    FilterRule,
    MultiSignalConfirmation,
    RankingCriteria,
    SignalCategory,
)

# Custom scoring weights
scoring_config = ScoringConfig(
    category_weights={
        SignalCategory.TECHNICAL: 0.30,
        SignalCategory.VOLUME: 0.25,
        SignalCategory.OPTIONS: 0.25,
        SignalCategory.FUNDAMENTAL: 0.20,
    }
)

# Custom filter rules
filter_rules = [
    FilterRule(
        category=SignalCategory.TECHNICAL,
        min_score=70.0,
        min_confidence=70.0,
    ),
    FilterRule(
        category=SignalCategory.VOLUME,
        min_score=65.0,
        min_confidence=60.0,
    ),
]

# Custom confirmation
confirmation_config = MultiSignalConfirmation(
    required_categories=[SignalCategory.TECHNICAL, SignalCategory.VOLUME],
    min_categories=2,
    require_same_direction=True,
    min_overall_score=75.0,
)

# Initialize with custom config
generator = SignalGenerator(
    scorer_config=scoring_config,
    filter_rules=filter_rules,
    confirmation_config=confirmation_config,
)
```

## Signal Combination Testing

```python
from signal_engine import SignalCombinationTester, CombinationRule, SignalCategory

# Initialize tester
tester = SignalCombinationTester()

# Define a combination rule
rule = CombinationRule(
    name="trend_volume_options",
    category_thresholds={
        SignalCategory.TECHNICAL: 80.0,
        SignalCategory.VOLUME: 75.0,
        SignalCategory.OPTIONS: 70.0,
    },
    require_direction_agreement=True,
    min_categories=3,
)

# Test the rule on historical data
result = tester.test_combination(
    rule=rule,
    historical_signal_sets=historical_data,
)

print(f"Win rate: {result.win_rate:.2%}")
print(f"Average return: {result.average_return:.2f}%")
print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
```

## Database Schema

The Signal Engine uses ClickHouse for storage:

- `signal_history`: Individual signals for each symbol
- `signal_sets`: Complete signal sets for each symbol
- `signal_performance`: Performance metrics per category
- `trade_records`: Individual trade records
- `signal_combination_rules`: Tested combination rules
- `signal_filter_results`: Filter audit trail
- `signal_rankings`: Ranking results

Initialize the schema:

```bash
clickhouse-client --database quant < database/signal_schema.sql
```

## Key Features

1. **Multi-Category Signal Generation**: Generates signals from 5+ categories
2. **Signal Scoring**: Weighted scoring with configurable category weights
3. **Signal Filtering**: Multi-stage filtering with confirmation rules
4. **Signal Ranking**: Ranks stocks by signal quality
5. **Performance Tracking**: Tracks win rate, returns, Sharpe ratio, etc.
6. **Combination Testing**: Tests signal combinations to find alpha rules
7. **Explainable Signals**: Every signal explains itself with reasons

## Signal Quality

The Signal Engine is designed to:

- Convert 500 features → 10-20 high-quality signals
- Filter out weak opportunities before prediction
- Provide clear explanations for every signal
- Track historical performance of each signal type
- Enable systematic research on signal combinations

## Next Steps

After the Signal Engine, the next module is the **Alpha Engine**, which answers:
"Is this setup historically profitable enough to trade?"

This separation of concerns makes the system easier to improve over time.
