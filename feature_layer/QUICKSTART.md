# Feature Laboratory - Quick Start Guide

This guide will help you get started with the Feature Laboratory in your existing quant trading system.

## Step 1: Database Migration

First, create the new tables in your PostgreSQL database:

```bash
# Using Alembic
alembic revision --autogenerate -m "Add Feature Laboratory tables"
alembic upgrade head
```

Or manually run the SQL to create the tables defined in `database/models.py`:
- `features`
- `feature_metadata`
- `feature_quality`
- `feature_importance`
- `feature_correlation`
- `feature_combinations`

## Step 2: Register Feature Dashboard API

Add the feature dashboard router to your FastAPI application in `api/main.py`:

```python
from feature_layer.feature_dashboard import router as feature_router

app.include_router(feature_router)
```

Now you can access the dashboard at:
- `http://localhost:8000/api/features/summary`
- `http://localhost:8000/api/features/list`
- etc.

## Step 3: Generate Features for Existing Data

Create a script to backfill features for your historical data:

```python
from database.db_sync import SessionLocal
from feature_layer import FeatureGenerator
import pandas as pd

# Load your historical price data
# This should have columns: Date, Open, High, Low, Close, Volume
price_data = pd.read_csv('your_historical_data.csv')
price_data['Date'] = pd.to_datetime(price_data['Date'])
price_data = price_data.set_index('Date')

# Initialize feature generator
db = SessionLocal()
generator = FeatureGenerator(db)

# Generate features for each symbol
symbols = price_data['Symbol'].unique()

for symbol in symbols:
    symbol_data = price_data[price_data['Symbol'] == symbol].copy()
    
    # Generate features
    results = generator.generate_features(
        symbol=symbol,
        data=symbol_data,
        store=True
    )
    
    print(f"Generated {len(results)} features for {symbol}")

db.close()
```

## Step 4: Analyze Feature Quality

Run feature analysis to understand which features work best:

```python
from database.db_sync import SessionLocal
from feature_layer import FeatureAnalyzer, FeatureQualityScorer
from datetime import date

db = SessionLocal()
analyzer = FeatureAnalyzer(db)
quality_scorer = FeatureQualityScorer(db)

# Analyze a specific feature
metrics = analyzer.calculate_feature_performance(
    feature_name="RSI14",
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31)
)

print(f"RSI14 Performance:")
print(f"Win Rate: {metrics['win_rate']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

# Update quality in database
analyzer.update_feature_quality_in_db(
    feature_name="RSI14",
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31)
)

# Get quality report
report = quality_scorer.get_quality_report()
print(report)

db.close()
```

## Step 5: Integrate with Prediction Pipeline

Modify your prediction pipeline to use features from the Feature Laboratory:

```python
from database.db_sync import SessionLocal
from feature_layer import FeatureGenerator

def get_features_for_prediction(symbol, current_date):
    """Get latest features for prediction."""
    db = SessionLocal()
    generator = FeatureGenerator(db)
    
    # Get latest features
    features = generator.get_latest_features(symbol)
    
    db.close()
    return features

# In your prediction function
features = get_features_for_prediction("INFY", date.today())
prediction = model.predict(features)
```

## Step 6: Track Feature Importance from ML Models

After training your models, log feature importance:

```python
from database.db_sync import SessionLocal
from feature_layer import FeatureImportanceTracker

# After training your model
importance_dict = {
    'RSI14': 0.15,
    'MACD': 0.08,
    'ATR14': 0.22,
    'VWAP_Distance': 0.31,
    # ... other features
}

db = SessionLocal()
tracker = FeatureImportanceTracker(db)

tracker.store_feature_importance(
    model_name="XGBoost_Classifier",
    model_version="1.0",
    feature_importance=importance_dict
)

db.close()
```

## Step 7: Test Feature Combinations

Discover alpha by testing feature combinations:

```python
from database.db_sync import SessionLocal
from feature_layer import FeatureTester
from datetime import date

db = SessionLocal()
tester = FeatureTester(db)

# Test a simple combination
results = tester.test_feature_combination(
    features=["RSI14", "VWAP_Distance"],
    conditions=["< 30", "> -2"],
    combination_name="Oversold_Below_VWAP",
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31)
)

print(f"Win Rate: {results['win_rate']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")

# Save if it looks good
if results['win_rate'] > 0.65:
    tester.save_alpha_candidate(
        combination_name="Oversold_Below_VWAP",
        features=["RSI14", "VWAP_Distance"],
        conditions=["< 30", "> -2"],
        results=results,
        notes="Strong oversold signal with VWAP support"
    )

db.close()
```

## Step 8: Monitor Feature Quality

Set up regular monitoring of feature quality:

```python
# Run this weekly or monthly
from database.db_sync import SessionLocal
from feature_layer import FeatureQualityScorer

db = SessionLocal()
scorer = FeatureQualityScorer(db)

# Get recommendations
recommendations = scorer.recommend_feature_actions()

print("Features to disable:", recommendations['disable'])
print("Features to monitor:", recommendations['monitor'])
print("Features to keep:", recommendations['keep'])

# Disable poor performing features
for feature in recommendations['disable']:
    generator.disable_feature(feature)

db.close()
```

## Daily Workflow

Once set up, your daily workflow should be:

1. **Download Market Data** → Your existing data pipeline
2. **Generate Features** → `FeatureGenerator.generate_features()`
3. **Run Predictions** → Use features from Feature Laboratory
4. **Track Results** → Update prediction outcomes
5. **Weekly**: Analyze feature quality and importance
6. **Monthly**: Test new feature combinations

## Integration with Existing Code

The Feature Laboratory is designed to work alongside your existing code. You don't need to rewrite everything:

### Option A: Gradual Migration

Keep your existing feature calculation for now, but:
- Store computed features in the Feature Laboratory
- Use the analyzer to understand which features work
- Gradually migrate to the new standardized format

### Option B: Full Migration

Replace existing feature calculation with the Feature Laboratory:
1. Migrate all indicators to the new format (see `technical/` folder)
2. Use `FeatureGenerator` for all feature computation
3. Remove old feature calculation code

## Troubleshooting

### Features not appearing in dashboard

Check that:
1. Database tables are created
2. Features are stored with correct metadata
3. `is_enabled` flag is set to True

### Feature generation is slow

- Use batch processing for multiple symbols
- Consider using DuckDB for faster analytics
- Optimize database indexes

### Quality scores are low

- Check if you have enough historical data
- Verify price data quality
- Ensure features are calculated correctly
- Adjust evaluation period

## Next Steps

1. **Add more features**: Implement features in other categories (volume, options, etc.)
2. **Set up automation**: Create scheduled jobs for feature generation
3. **Build dashboard UI**: Create a frontend for the API endpoints
4. **Integrate with MLflow**: Track feature sets used in each experiment
5. **Add real-time features**: Implement streaming feature computation

## Support

For issues or questions:
- Check the main README.md for detailed documentation
- Review the code documentation in each module
- Look at the example implementations in `technical/` folder
