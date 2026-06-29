# QuantResearchOS

An institutional-grade quantitative research operating system for the Indian equity markets (NSE/BSE).

## Architecture

```
QuantResearchOS/
├── apps/          # Dashboard (Streamlit), Scheduler (APScheduler)
├── services/      # Domain engines (Market Data, Features, Labels, Evaluation, News)
├── ml/            # Models (Trees, DL, RL), Ensemble, Inference/Calibration
├── database/      # PostgreSQL ORM schemas
├── shared/        # Configs, Pydantic contracts, Redis utilities
├── data/          # Parquet data lake
├── research/      # Notebooks, experiments, hypotheses
└── tests/         # Unit and integration tests
```

## Philosophy

- **Research-first**: The platform produces *knowledge*, not trades.
- **Immutable data**: Raw market data is never modified.
- **Multi-objective labeling**: Triple Barrier Method (MFE, MAE, holding time).
- **Meta-Ensemble**: Specialized models feed a stacking combiner.
- **Probability Calibration**: Raw probabilities are calibrated to true win rates.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run apps/dashboard/app.py
```

## License

Private — Not for redistribution.
