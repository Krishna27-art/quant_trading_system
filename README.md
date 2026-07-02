# Institutional Quantitative Trading Platform

An institutional-grade quantitative trading platform designed for high-performance, deterministic execution, and rigorous risk management in the Indian market (NSE/BSE).

---

## 1. Project Overview

This is a comprehensive quantitative research and trading operating system built for Indian equity markets. The platform integrates real-time market data ingestion, machine learning-based predictions, multi-broker execution, and institutional-grade risk management into a unified system.

**Core Capabilities:**
- **Real-time Market Data**: Live tick data ingestion via Upstox API with WebSocket fallback
- **ML Predictions**: LightGBM-based binary classifiers for INTRADAY, SWING, and LONGTERM timeframes
- **Multi-Broker Execution**: Smart order routing across Zerodha, Upstox, and Angel One with paper trading support
- **Risk Governance**: Pre-trade checks, drawdown limits, circuit breakers, and emergency kill switch
- **Research Platform**: Backtesting, walk-forward optimization, and performance analytics
- **Data Lake Architecture**: Bronze/Silver/Gold layers for data pipeline management
- **Low-Latency IPC**: Shared memory ring buffers for zero-copy inter-process communication
- **CQRS Pattern**: Separated read/write paths with PostgreSQL and DuckDB

**Target Market:** NSE (National Stock Exchange of India) with focus on NIFTY 50, BANKNIFTY, and liquid stocks

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   Upstox     │  │   NSE Lib    │  │   FRED API   │        │
│  │   Market Data│  │   (nsepython)│  │   (Macro)    │        │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │
└─────────┼──────────────────┼──────────────────┼────────────────┘
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼────────────────┐
│                    DATA PLATFORM (data_platform/)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Feed Manager │  │ Data Quality │  │ Ring Buffer  │        │
│  │ (WebSocket)  │  │   Gate       │  │  (SPSC IPC)  │        │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │
└─────────┼──────────────────┼──────────────────┼────────────────┘
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼────────────────┐
│              PREDICTION INTELLIGENCE (prediction_intelligence/)  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  LightGBM    │  │   XGBoost    │  │     LSTM     │        │
│  │ Classifiers  │  │   Models     │  │   (Optional) │        │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │
└─────────┼──────────────────┼──────────────────┼────────────────┘
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼────────────────┐
│           PORTFOLIO EXECUTION (portfolio_execution/)           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   OMS        │  │    EMS       │  │ Smart Router │        │
│  │ (Order Mgmt) │  │ (Execution)  │  │   (SOR)      │        │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │
└─────────┼──────────────────┼──────────────────┼────────────────┘
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼────────────────┐
│              RISK GOVERNANCE (risk_governance/)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Pre-Trade    │  │  Kill Switch │  │  Circuit     │        │
│  │   Checks     │  │  (Emergency) │  │  Breakers    │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
          │
┌─────────▼──────────────────────────────────────────────────────┐
│                    API & DATABASE LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  FastAPI     │  │ PostgreSQL   │  │    Redis     │        │
│  │  Backend     │  │ (Primary DB) │  │  (Streams)   │        │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │
└─────────┼──────────────────┼──────────────────┼────────────────┘
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼────────────────┐
│                      FRONTEND (frontend/)                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           Vanilla JS + HTML Dashboard                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

1. **Market Data Ingestion**: Upstox API provides real-time quotes, OHLCV candles, options chain, and fundamentals
2. **Data Validation**: Streaming outlier detection filters stale/corrupted prices
3. **Feature Engineering**: Technical indicators (RSI, EMA, ATR, VWAP) computed in real-time
4. **ML Inference**: LightGBM models generate win probabilities for each timeframe
5. **Signal Generation**: Composite alpha model combines momentum, mean reversion, and alternative signals
6. **Risk Checks**: Pre-trade validation enforces position limits, drawdowns, and SEBI regulations
7. **Order Execution**: Smart order routing places orders via broker adapters
8. **Fill Processing**: Execution worker handles fills and updates position state
9. **Persistence**: All state persisted to PostgreSQL with WAL for crash recovery

---

## 3. Repository Structure

```
quant/
├── agents/                          # LLM integration
│   └── llm_client.py               # Anthropic Claude client for analysis
├── api/                             # FastAPI backend
│   ├── auth.py                     # JWT authentication
│   └── main.py                     # REST API endpoints
├── auth/                            # Authentication utilities
│   └── upstox_token_refresher.py   # Upstox OAuth token management
├── config/                          # Configuration
│   ├── environment_config.py      # Environment-specific settings
│   ├── settings.py                 # Central configuration constants
│   └── universe.py                 # NSE universe definition
├── data_platform/                   # Data infrastructure
│   ├── feeds/                      # Market data feeds
│   │   ├── feed_manager.py         # WebSocket feed management
│   │   └── data_quality_gate.py    # Data validation
│   ├── feature_store/              # Feature engineering
│   │   ├── base.py                 # Feature store base classes
│   │   ├── macro.py                # Macro economic features
│   │   └── sentiment.py            # News sentiment features
│   ├── pipelines/                  # Data ingestion pipelines
│   │   ├── equity_history.py       # Historical equity data
│   │   ├── corporate_actions.py    # Corporate actions pipeline
│   │   ├── options_chain.py        # Options data pipeline
│   │   └── nse_options.py          # NSE options data
│   ├── sources/ingestion/          # Data source adapters
│   │   ├── nselib_source.py        # NSE library integration
│   │   ├── scraper_source.py       # Web scraping fallback
│   │   └── rate_limiter.py         # API rate limiting
│   ├── validation/                 # Data validation rules
│   │   ├── base_validator.py       # Base validation framework
│   │   ├── equity_rules.py         # Equity-specific rules
│   │   └── options_rules.py        # Options-specific rules
│   ├── ring_buffer.py              # Shared memory IPC
│   └── upstox_client.py            # Upstox API client
├── database/                        # Database layer
│   ├── connection.py               # PostgreSQL connection management
│   ├── db_async.py                 # Async database utilities
│   ├── db_sync.py                  # Sync database utilities
│   ├── models.py                   # SQLAlchemy ORM models
│   └── *.sql                       # Database schemas
├── frontend/                        # Web UI
│   ├── index.html                  # Main dashboard
│   └── favicon.svg                 # UI assets
├── india_specific/                  # India-specific utilities
│   └── circuit_limits.py           # NSE circuit limits
├── observability_mlops/             # Monitoring
│   ├── alerting.py                 # Alert management
│   ├── health_check.py             # System health monitoring
│   └── prometheus_metrics.py       # Prometheus metrics export
├── portfolio_execution/             # Trading execution
│   ├── core/                       # Core trading logic
│   ├── events/                     # Event definitions
│   ├── execution/                  # Execution management
│   │   ├── brokers/                # Broker adapters
│   │   ├── market_microstructure/  # Market microstructure analysis
│   │   ├── routing/                # Smart order routing
│   │   ├── advanced_algos.py       # Advanced execution algorithms
│   │   ├── base.py                 # Base execution classes
│   │   └── execution_sequencer.py   # Execution sequencing
│   ├── optimization/               # Portfolio optimization
│   │   ├── netting/                # Position netting
│   │   ├── hrp.py                  # Hierarchical Risk Parity
│   │   └── optimization.py         # Portfolio optimization
│   ├── config.py                   # Execution configuration
│   ├── drop_copy_reconciler.py     # Position reconciliation
│   ├── ems.py                      # Execution Management System
│   └── oms.py                      # Order Management System
├── prediction_intelligence/         # ML models
│   ├── base_lightgbm.py            # LightGBM base classifier
│   ├── base_logistic.py            # Logistic regression models
│   ├── base_lstm.py                # LSTM models
│   └── saved/                      # Saved model artifacts
├── research_platform/              # Research tools
│   ├── backtesting/               # Backtesting framework
│   │   ├── benchmarking.py         # Benchmark comparison
│   │   ├── cross_validation.py     # Cross-validation utilities
│   │   └── deflated_sharpe.py     # Deflated Sharpe ratio
│   ├── experiments/               # Experiment tracking
│   │   └── experiment_tracker.py   # MLflow experiment tracking
│   └── research/                  # Research utilities
│       ├── backtest/              # Backtest utilities
│       ├── evaluation/            # Model evaluation
│       ├── alpha_evaluator.py     # Alpha signal evaluation
│       └── deflated_sharpe.py     # Risk-adjusted returns
├── risk_governance/                # Risk management
│   └── pre_trade/                 # Pre-trade risk checks
│       ├── circuit_breakers/      # Circuit breaker logic
│       ├── sebi_margin/            # SEBI margin requirements
│       ├── beta_hedger.py         # Beta hedging
│       ├── borrow_manager.py       # Short sale borrow management
│       ├── capacity_measurement.py # Capacity measurement
│       ├── kill_switch.py          # Emergency kill switch
│       ├── portfolio_drawdown_limits.py  # Drawdown limits
│       └── pre_trade_checks.py     # Pre-trade validation
├── scripts/                         # Utility scripts
│   ├── execution_worker.py         # Async order execution worker
│   ├── check_no_production_mocks.py # Mock detection
│   └── evaluate_outcomes.py        # Prediction evaluation
├── shared/                          # Shared utilities
│   └── schemas/                    # Shared data schemas
│       └── oms_events.py           # OMS event schemas
├── tests/                           # Test suite
│   ├── backfill/                   # Data backfill tests
│   ├── data_quality/              # Data quality tests
│   ├── integration/               # Integration tests
│   └── mocks/                     # Test mocks
├── utils/                           # Utility functions
│   ├── api_circuit_breaker.py      # API circuit breaker
│   ├── api_helpers.py              # API helper functions
│   ├── blas_config.py             # BLAS configuration
│   ├── clickhouse_client.py       # ClickHouse client
│   ├── logger.py                   # Structured logging
│   └── time_utils.py              # Time zone utilities
├── main.py                          # Main entry point
├── requirements.txt                 # Python dependencies
├── pyproject.toml                  # Project configuration
├── docker-compose.yml              # Docker services
├── Dockerfile                       # Container definition
└── .env.example                    # Environment template
```

---

## 4. Data Sources & APIs

### 4.1 Primary Data Sources

**Upstox API** (Primary Market Data)
- Live quotes and OHLCV candles
- Options chain with Greeks
- FII/DII activity data
- Market holidays
- Company fundamentals (balance sheet, income statement, cash flow, key ratios)
- Corporate actions
- News feeds
- Smartlists (futures, options, MTF)

**NSE Python Library** (Secondary)
- NSE equity data
- Index data
- Options data
- Historical data

**FRED API** (Macro Data)
- US macroeconomic indicators
- RBI repo rates
- Inflation data (CPI)
- Industrial production (IIP)

### 4.2 Broker APIs

**Zerodha Kite**
- Order placement and execution
- Position management
- Account information

**Upstox**
- Market data (primary source)
- Order execution (configured)

**Angel One SmartAPI**
- Alternative broker integration
- Market data fallback

### 4.3 LLM Integration

**Anthropic Claude**
- Model: claude-3-haiku-20240307
- Use cases:
  - Pre-market analysis
  - News sentiment analysis
  - Post-trade analysis
  - Market regime detection
- Fallback to mock responses when API key not configured

---

## 5. Technology Stack

### 5.1 Core Technologies

**Backend**
- Python 3.10+
- FastAPI (web framework)
- Uvicorn (ASGI server)
- SQLAlchemy (ORM)
- Alembic (database migrations)

**Databases**
- PostgreSQL with TimescaleDB (primary database)
- Redis (message queues, caching)
- DuckDB (analytics, feature store)
- ClickHouse (optional, production analytics)

**Machine Learning**
- LightGBM (primary ML framework)
- XGBoost (alternative models)
- Scikit-learn (utilities)
- MLflow (experiment tracking)
- Transformers (NLP models)
- PyTorch (deep learning)

**Data Processing**
- Pandas (data manipulation)
- NumPy (numerical computing)
- SciPy (scientific computing)
- CVXPY (portfolio optimization)

**Monitoring & Observability**
- Prometheus (metrics collection)
- Grafana (visualization)
- Structlog (structured logging)

**Task Queue**
- Celery (async task processing)
- Prefect (workflow orchestration)

**Testing**
- Pytest (testing framework)
- Pytest-cov (coverage)
- Great Expectations (data quality testing)

**API Integration**
- Requests (HTTP client)
- WebSockets (real-time data)
- aiohttp (async HTTP)

### 5.2 Development Tools

- Black (code formatting)
- Ruff (linting)
- MyPy (type checking)
- Pre-commit (git hooks)
- Docker (containerization)
- Docker Compose (multi-container orchestration)

---

## 6. Key Components

### 6.1 Data Platform

**Feed Manager** (`data_platform/feeds/feed_manager.py`)
- Multi-tier feed management (primary, secondary, fallback)
- WebSocket connection handling
- Automatic failover between data sources
- Health monitoring and staleness detection

**Data Quality Gate** (`data_platform/feeds/data_quality_gate.py`)
- Streaming outlier detection
- ATR-based price validation
- Gap detection and reporting
- Quality statistics tracking

**Ring Buffer IPC** (`data_platform/ring_buffer.py`)
- Lock-free SPSC ring buffers
- Shared memory for zero-copy IPC
- Separate buffers for ticks and orders
- Multi-process topology support

**Feature Store** (`data_platform/feature_store/`)
- Feature registration and dependency resolution
- Point-in-time joins
- Feature validation
- Macro and sentiment features

**Data Pipelines** (`data_platform/pipelines/`)
- Historical equity data ingestion
- Corporate actions processing
- Options chain data collection
- Macro data pipelines
- NSE options data

### 6.2 Prediction Intelligence

**LightGBM Classifiers** (`prediction_intelligence/base_lightgbm.py`)
- Binary classifiers for INTRADAY, SWING, LONGTERM
- Calibrated win probabilities
- Feature importance tracking
- Early stopping and regularization
- Model persistence with metadata

**Features Used**
- Technical: RSI, EMA, ATR, VWAP, volume ratios
- Returns: 1-day, 5-day, 20-day returns
- Market: VIX level, market regime
- Macro: USD-INR changes, DOW changes

### 6.3 Portfolio Execution

**Order Management System** (`portfolio_execution/oms.py`)
- Order lifecycle management
- Write-ahead logging for durability
- Position tracking
- Order status updates

**Execution Management System** (`portfolio_execution/ems.py`)
- Smart order routing
- Broker adapter management
- Fill processing
- Connection health monitoring

**Smart Order Router** (`portfolio_execution/execution/routing/`)
- Multi-broker routing strategies
- Best price routing
- Failover logic
- Latency tracking

**Signal Models** (`portfolio_execution/signals/`)
- Time series momentum
- Bollinger mean reversion
- News sentiment alpha
- Options flow alpha
- Fundamental PIT alpha
- Index futures basis alpha
- Volatility surface alpha

### 6.4 Risk Governance

**Pre-Trade Checks** (`risk_governance/pre_trade/pre_trade_checks.py`)
- Position size limits
- Price deviation checks
- Fat-finger prevention
- Daily drawdown limits
- SEBI margin requirements

**Kill Switch** (`risk_governance/pre_trade/kill_switch.py`)
- Emergency position liquidation
- Database transaction handling
- Dry-run mode for testing
- Comprehensive logging

**Circuit Breakers** (`risk_governance/pre_trade/circuit_breakers/`)
- Exchange circuit limits
- Volatility-based circuit breakers
- Position-level circuit breakers

**Borrow Manager** (`risk_governance/pre_trade/borrow_manager.py`)
- Short sale borrow verification
- Inventory tracking
- Borrow limit enforcement

### 6.5 API Layer

**FastAPI Backend** (`api/main.py`)
- RESTful API endpoints
- WebSocket support for real-time data
- JWT authentication
- CORS configuration
- Prometheus metrics endpoint

**Key Endpoints**
- `GET /api/health` - System health check
- `GET /api/indices` - Index data
- `GET /api/stocks` - Stock data with filtering
- `GET /api/predictions` - ML predictions
- `GET /api/sectors` - Sector performance
- `POST /api/auth/login` - Authentication
- `GET /metrics` - Prometheus metrics

### 6.6 Research Platform

**Backtesting** (`research_platform/backtesting/`)
- Historical simulation
- Walk-forward validation
- Benchmark comparison
- Deflated Sharpe ratio calculation

**Experiment Tracking** (`research_platform/experiments/`)
- MLflow integration
- Hyperparameter tracking
- Model versioning
- Performance metrics

---

## 7. Configuration

### 7.1 Environment Variables

Required environment variables (see `.env.example`):

```bash
# Broker Credentials
UPSTOX_BROKER_ACCESS_TOKEN=your_upstox_token
ZERODHA_API_KEY=your_zerodha_key
ZERODHA_ACCESS_TOKEN=your_zerodha_token
ANGEL_ONE_API_KEY=your_angel_key
ANGEL_ONE_CLIENT_CODE=your_client_code
ANGEL_ONE_PASSWORD=your_password
ANGEL_ONE_TOTP=your_totp_secret

# Database
POSTGRES_USER=postgres_user
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=quant_terminal
DATABASE_URL=postgresql://user:pass@localhost:5432/quant_terminal
REDIS_URL=redis://localhost:6379/0

# API Security
ADMIN_USERNAME=admin
ADMIN_PASSWORD=secure_admin_password
JWT_SECRET_KEY=256_bit_secret_key

# LLM
ANTHROPIC_API_KEY=sk-ant-xxxxx

# Monitoring
GRAFANA_ADMIN_PASSWORD=secure_grafana_password
```

### 7.2 Configuration Files

**Settings** (`config/settings.py`)
- Data directory structure (Bronze/Silver/Gold layers)
- Database paths
- Trading calendar configuration
- Data quality thresholds
- NSE API rate limits

**Universe** (`config/universe.py`)
- NSE universe definition
- Sector classifications
- Index constituents

---

## 8. How to Run

### 8.1 Prerequisites

- Python 3.10 or higher
- PostgreSQL 14+ with TimescaleDB
- Redis 7+
- Docker and Docker Compose (for containerized deployment)
- Upstox API access token
- (Optional) Zerodha Kite API credentials

### 8.2 Installation

**Clone the repository**
```bash
git clone <repository-url>
cd quant
```

**Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Configure environment**
```bash
cp .env.example .env
# Edit .env with your credentials
```

**Initialize database**
```bash
# Using Docker Compose
docker-compose up -d db redis

# Or manually setup PostgreSQL
createdb quant_terminal
psql quant_terminal < database/adjusted_equity_schema.sql
```

**Run migrations**
```bash
alembic upgrade head
```

### 8.3 Running the System

**Option 1: Docker Compose (Recommended)**
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

**Option 2: Manual Startup**

Start the API server:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Start the main trading orchestrator:
```bash
python main.py --mode paper --symbols RELIANCE,TCS,INFY --duration 60
```

Start the execution worker:
```bash
python scripts/execution_worker.py
```

### 8.4 Running Modes

**Backtest Mode** (Historical simulation)
```bash
python main.py --mode backtest --date 2024-01-15 --symbols RELIANCE,TCS
```

**Paper Trading Mode** (Simulated live trading)
```bash
python main.py --mode paper --symbols RELIANCE,TCS,INFY --duration 3600
```

**Live Trading Mode** (Real broker execution)
```bash
python main.py --mode live --symbols RELIANCE,TCS,INFY
```

### 8.5 Accessing the Dashboard

- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Frontend**: http://localhost:3000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001 (admin/password from .env)

### 8.6 Training ML Models

```bash
# Train LightGBM models for all timeframes
python -m prediction_intelligence.train_models

# Train specific timeframe
python -m prediction_intelligence.train_models --timeframe INTRADAY
```

### 8.7 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test suite
pytest tests/data_quality/
pytest tests/integration/
```

---

## 9. Development Guidelines

### 9.1 Coding Standards

- Follow PEP 8 with Black formatting (line length: 100)
- Use type hints for all functions
- Add docstrings for all public functions
- Use dataclasses/Pydantic models for data structures
- Handle exceptions explicitly with logging
- Follow SOLID principles

### 9.2 Quant Standards

- Prevent lookahead bias in all calculations
- Prevent survivorship bias in backtesting
- Prevent data leakage in ML pipelines
- Validate all predictions against historical performance
- Calculate standard metrics: Sharpe, Sortino, Max Drawdown, CAGR, Win Rate, Profit Factor
- Explain assumptions before implementing strategies

### 9.3 Data Standards

- Validate dataframe shapes before calculations
- Log dataframe dimensions
- Reject malformed financial data
- Handle missing values explicitly
- Detect NaN and infinite values

### 9.4 ML Standards

- Scale features before training
- Use train/test split
- Avoid future leakage
- Log feature importance
- Save model metadata
- Generate evaluation reports

---

## 10. Monitoring & Maintenance

### 10.1 Health Checks

System health endpoint: `GET /api/health/status`

Checks:
- Database connectivity
- Model availability
- Data pipeline freshness
- API gateway status

### 10.2 Metrics

Prometheus metrics available at `/metrics`:
- Order submission rate
- Fill rate
- Latency percentiles
- Prediction accuracy
- System resource usage

### 10.3 Logging

Structured logs using `structlog`:
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Log location: `logs/` directory
- Log rotation: Configured in settings

### 10.4 Alerts

Alerting via:
- Slack webhooks (configure in .env)
- PagerDuty integration (configure in .env)
- Email notifications (optional)

---

## 11. Troubleshooting

### 11.1 Common Issues

**Database Connection Failed**
- Check PostgreSQL is running: `docker-compose ps db`
- Verify DATABASE_URL in .env
- Check database credentials

**Upstox API Errors**
- Verify UPSTOX_BROKER_ACCESS_TOKEN is valid
- Check token hasn't expired (tokens expire periodically)
- Run token refresher: `python auth/upstox_token_refresher.py`

**Redis Connection Failed**
- Check Redis is running: `docker-compose ps redis`
- Verify REDIS_URL in .env

**Model Not Found**
- Train models: `python -m prediction_intelligence.train_models`
- Check MODEL_DIR environment variable
- Verify model files exist in `models/saved/`

### 11.2 Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python main.py --mode paper
```

---

## 12. Contributing

### 12.1 Development Workflow

1. Create feature branch from `main`
2. Make changes with tests
3. Run linting: `ruff check .`
4. Run formatting: `black .`
5. Run type checking: `mypy .`
6. Run tests: `pytest`
7. Submit pull request

### 12.2 Pre-commit Hooks

Install pre-commit hooks:
```bash
pre-commit install
```

Hooks run automatically on commit.

---

## 13. License

MIT License - See LICENSE file for details

---

## 14. Support

For issues and questions:
- GitHub Issues: [repository-url]/issues
- Documentation: [repository-url]/wiki
- Email: quant@example.com

---

## 15. Acknowledgments

- Upstox for market data API
- NSE for market data
- LightGBM team for ML framework
- Open-source community for various libraries
