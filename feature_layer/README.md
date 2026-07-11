# Feature Laboratory

A production-grade feature store and management system for institutional quantitative trading.

## Overview

The Feature Laboratory provides a complete infrastructure for feature engineering, storage, analysis, and management. It follows institutional best practices for feature stores, ensuring reproducibility, versioning, and consistent feature usage across research, training, and production.

## Architecture

```
Raw Market Data
        ↓
Feature Generator
        ↓
Feature Database (PostgreSQL)
        ↓
Feature Analyzer
        ↓
Feature Ranking
        ↓
ML Models
```

## Components

### 1. Base Feature System (`base_feature.py`)

Standardized interface for all features with comprehensive metadata:

- **BaseFeature**: Abstract base class for all features
- **FeatureMetadata**: Complete documentation for each feature
- **FeatureResult**: Standardized output format
- **FeatureCategory**: 8 categories (technical, volume, options, fundamentals, macro, sentiment, sector, market)
- **Timeframe**: Support for multiple timeframes (tick to monthly)

Every feature includes:
- Description and computation method
- Required columns and output range
- Assumptions and limitations
- References and author
- Version tracking

### 2. Feature Generator (`feature_generator.py`)

Core engine that discovers, calculates, and stores features:

- Automatic feature discovery from category directories
- Enable/disable feature management
- Batch feature computation for symbols
- PostgreSQL storage with versioning
- Feature metadata registration
- Computation metrics tracking

### 3. Feature Analyzer (`feature_analyzer.py`)

Analyzes feature performance to determine predictive quality:

- Win rate calculation
- Average return analysis
- Sharpe ratio and Sortino ratio
- Max drawdown measurement
- Profit factor calculation
- Feature comparison
- Top feature ranking

### 4. Feature Importance Tracker (`feature_importance.py`)

Tracks ML model feature importance over time:

- Store importance from any ML model
- Track importance history
- Compare importance across models
- Aggregate importance scores
- Detect feature degradation

### 5. Feature Correlation Analyzer (`feature_correlation.py`)

Identifies redundant features through correlation analysis:

- Pearson, Spearman, Kendall correlations
- P-value calculation
- Highly correlated feature detection
- Redundancy reduction suggestions
- Correlation matrix storage

### 6. Feature Quality Scorer (`feature_quality.py`)

Comprehensive quality scoring (0-100) based on:

- **Predictive Power (40%)**: Win rate, Sharpe ratio, returns
- **Importance (25%)**: ML model feature importance
- **Uniqueness (20%)**: Low correlation with other features
- **Stability (15%)**: Consistent performance over time

Quality grades: A (90+), B (80+), C (70+), D (60+), F (<60)

### 7. Feature Version Manager (`feature_versioning.py`)

Manages feature versions for reproducibility:

- Create new versions (never overwrite)
- Version history tracking
- Version comparison
- Rollback capability
- Active version management

### 8. Feature Testing Framework (`feature_testing.py`)

Tests features and combinations to discover alpha:

- Single feature testing with thresholds
- Multi-feature combination testing
- Backtest metrics calculation
- Alpha candidate saving
- Strategy comparison

### 9. Feature Dashboard API (`feature_dashboard.py`)

FastAPI endpoints for feature management:

- `/api/features/summary` - Overall summary
- `/api/features/list` - List all features
- `/api/features/{name}` - Feature details
- `/api/features/{name}/quality` - Quality metrics
- `/api/features/quality/report` - Quality report
- `/api/features/importance/{model}` - Feature importance
- `/api/features/correlation/matrix` - Correlation matrix
- `/api/features/{name}/enable` - Enable feature
- `/api/features/{name}/disable` - Disable feature
- `/api/features/categories/summary` - Category summary

## Database Schema

### Tables

1. **features** - Main feature storage
   - symbol, date, feature_name, feature_category
   - feature_version, feature_value
   - Unique constraint: (symbol, date, feature_name, version)

2. **feature_metadata** - Feature documentation
   - feature_name, category, description
   - timeframe, required_columns, output_range
   - version, author, computation_method
   - assumptions, limitations, references
   - is_enabled flag

3. **feature_quality** - Quality scores
   - feature_name, feature_version
   - quality_score, win_rate, sharpe_ratio
   - profit_factor, max_drawdown
   - evaluation_period, sample_size

4. **feature_importance** - ML importance tracking
   - model_name, model_version, feature_name
   - importance_score, rank
   - computed_at

5. **feature_correlation** - Correlation matrix
   - feature_1, feature_2
   - correlation_coefficient, p_value
   - sample_size

6. **feature_combinations** - Alpha candidates
   - combination_name, features, conditions
   - win_rate, sharpe_ratio, returns
   - is_active, notes

## Feature Categories

### Technical (`technical/`)

Price-based technical indicators:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- ATR (Average True Range)
- EMA (Exponential Moving Average)
- VWAP (Volume Weighted Average Price)

### Volume (`volume/`)

Volume-based features (to be implemented):
- Volume spike detection
- OBV (On-Balance Volume)
- Volume profile
- Relative volume

### Options (`options/`)

Options-derived features (to be implemented):
- PCR (Put-Call Ratio)
- OI (Open Interest) change
- Max pain calculation
- IV (Implied Volatility)
- Option flow

### Fundamentals (`fundamentals/`)

Company fundamentals (to be implemented):
- PE ratio
- ROE (Return on Equity)
- EPS growth
- Debt ratios
- Delivery percentage

### Macro (`macro/`)

Macroeconomic indicators (to be implemented):
- USDINR
- Crude oil prices
- US futures
- Bond yields
- Gold prices

### Sentiment (`sentiment/`)

Market sentiment (to be implemented):
- News sentiment scores
- Social media sentiment
- Analyst ratings
- Institutional holdings

### Sector (`sector/`)

Sector-based features (to be implemented):
- Sector strength
- Relative strength vs sector
- Sector rotation
- Industry group performance

### Market (`market/`)

Market-wide features (to be implemented):
- Advance/Decline ratio
- Market breadth
- VIX
- Index performance

## Usage Examples

### Creating a New Feature

```python
from feature_layer.base_feature import BaseFeature, FeatureMetadata, FeatureCategory, Timeframe
import pandas as pd

class MyCustomFeature(BaseFeature):
    def _define_metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            feature_name="MyFeature",
            description="My custom feature description",
            category=FeatureCategory.TECHNICAL,
            timeframe=Timeframe.DAY_1,
            required_columns=["Close"],
            output_range="0-100",
            version="1.0",
            author="your_name",
            computation_method="Explanation of calculation",
            assumptions="Key assumptions",
            limitations="Known limitations",
            references="Source references"
        )
    
    def compute(self, data: pd.DataFrame) -> FeatureResult:
        # Implement feature calculation
        values = data['Close'].rolling(20).mean()
        return FeatureResult(
            feature_name=self.metadata.feature_name,
            values=values,
            metadata=self.metadata,
            computation_time_ms=0,
            warnings=[]
        )
```

### Generating Features

```python
from database.db_sync import get_db
from feature_layer import FeatureGenerator

db = next(get_db())
generator = FeatureGenerator(db)

# Generate all enabled features for a symbol
results = generator.generate_features(
    symbol="INFY",
    data=price_data,
    store=True
)
```

### Analyzing Feature Quality

```python
from feature_layer import FeatureAnalyzer
from datetime import date

analyzer = FeatureAnalyzer(db)

# Calculate performance metrics
metrics = analyzer.calculate_feature_performance(
    feature_name="RSI14",
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31)
)

# Get quality score
quality_score = analyzer.calculate_feature_quality_score(
    feature_name="RSI14",
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31)
)
```

### Testing Feature Combinations

```python
from feature_layer import FeatureTester

tester = FeatureTester(db)

# Test a combination
results = tester.test_feature_combination(
    features=["RSI14", "VWAP_Distance"],
    conditions=["< 30", "> -2"],
    combination_name="Oversold_Below_VWAP",
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31)
)

# Save as alpha candidate if successful
if results['win_rate'] > 0.7:
    tester.save_alpha_candidate(
        combination_name="Oversold_Below_VWAP",
        features=["RSI14", "VWAP_Distance"],
        conditions=["< 30", "> -2"],
        results=results,
        notes="Strong oversold signal with VWAP support"
    )
```

## Integration with MLflow

The Feature Laboratory is designed to work with MLflow for experiment tracking:

```python
import mlflow

# Log feature set used in training
with mlflow.start_run():
    mlflow.log_params({
        "feature_set_version": "1.0",
        "features_used": ["RSI14", "MACD", "ATR14", "VWAP_Distance"],
        "feature_count": 4
    })
    
    # Train model...
```

## Best Practices

1. **Always version features**: Never modify existing features - create new versions
2. **Document thoroughly**: Every feature must have complete metadata
3. **Test before enabling**: Validate feature quality before production use
4. **Monitor degradation**: Track feature importance and quality over time
5. **Remove redundancy**: Use correlation analysis to eliminate duplicate features
6. **Start simple**: Begin with basic features, add complexity gradually
7. **Validate assumptions**: Regularly test feature assumptions against market data

## Future Enhancements

- [ ] Integration with DuckDB for fast analytics
- [ ] Automated feature engineering
- [ ] Real-time feature computation
- [ ] Feature drift detection
- [ ] Automated feature selection
- [ ] Feature visualization dashboard
- [ ] Integration with Feast for distributed feature serving

## References

- [Feast - The Open Source Feature Store](https://feast.dev/)
- [Feature Stores for Machine Learning](https://www.featurestore.org/)
- [MLflow Feature Store Integration](https://feast.dev/blog/feast-mlflow-native-integration/)

## License

Part of the Quant Research OS project.
