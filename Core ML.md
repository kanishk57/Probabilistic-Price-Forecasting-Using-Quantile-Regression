Core ML & Data Processing:
├── Python 3.10+                    # Main language
├── LightGBM 4.0+                   # Gradient boosting (PRIMARY MODEL)
├── scikit-learn 1.3+               # Preprocessing, metrics, CV
├── pandas 2.1+                     # Data manipulation
├── polars 0.19+                    # Fast alternative (10x pandas speed)
├── numpy 1.24+                     # Numerical operations
├── pyarrow 13.0+                   # Parquet files (columnar storage)
└── joblib 1.3+                     # Model serialization

Technical Analysis:
├── ta-lib 0.4.28                   # Traditional indicators (ATR, RSI)
└── pandas-ta 0.3.14b               # Extended indicators

Hyperparameter Optimization:
└── optuna 3.3+                     # Bayesian optimization

Data Collection:
├── MetaTrader5 5.0.45              # Forex/Gold data (MT5 API)
├── ccxt 4.0+                       # Crypto exchange APIs
└── requests 2.31+                  # HTTP client for REST APIs
```

---

### **Frontend/API Layer**
```
Production API:
├── FastAPI 0.103+                  # Modern async web framework
│   ├── Pydantic 2.3+              # Data validation
│   └── uvicorn 0.23+              # ASGI server
│
├── Streamlit 1.27+                 # Quick dashboards (optional)
└── Dash/Plotly 5.16+              # Interactive visualizations
```

**Why FastAPI?**
- Async support (handle multiple requests)
- Auto-generated docs (Swagger UI)
- Type validation with Pydantic
- Fast (comparable to Node.js/Go)

---

### **Database & Storage**
```
Time-Series Data:
├── TimescaleDB                     # PostgreSQL extension for time-series
│   └── PostgreSQL 14+             # Base relational DB
│
├── InfluxDB 2.7+                   # Alternative: pure time-series DB
│
└── Parquet Files                   # File-based (good for < 10M rows)
    └── Local filesystem

Caching Layer:
└── Redis 7.2+                      # In-memory cache
    ├── Features cache             # Store latest 1000 candles
    ├── Predictions cache          # Recent model outputs
    └── Session data               # User state

Model Storage:
├── Local filesystem (.pkl)         # Joblib serialized models
└── MLflow Model Registry          # Versioned model storage (optional)


-- OHLCV table (hypertable)
CREATE TABLE ohlcv (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    PRIMARY KEY (time, symbol, timeframe)
);

-- Convert to hypertable (TimescaleDB)
SELECT create_hypertable('ohlcv', 'time');

-- Features table
CREATE TABLE features (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    fvg_type TEXT,
    fvg_size DOUBLE PRECISION,
    pd_position DOUBLE PRECISION,
    atr DOUBLE PRECISION,
    -- ... 15+ feature columns
    PRIMARY KEY (time, symbol, timeframe)
);

-- Predictions table
CREATE TABLE predictions (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    fill_probability DOUBLE PRECISION,
    tp_probability DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    model_version TEXT,
    PRIMARY KEY (time, symbol)
);

-- Trades table
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    entry_time TIMESTAMPTZ,
    exit_time TIMESTAMPTZ,
    symbol TEXT,
    direction TEXT,
    entry_price DOUBLE PRECISION,
    exit_price DOUBLE PRECISION,
    pnl DOUBLE PRECISION,
    outcome TEXT
);
```

---

### **Visualization & Monitoring**
```
Development/Analysis:
├── Jupyter Lab 4.0+                # Interactive notebooks
├── matplotlib 3.7+                 # Static plots
├── seaborn 0.12+                   # Statistical visualizations
└── plotly 5.16+                    # Interactive charts

Production Monitoring:
├── Grafana 10.0+                   # Dashboard visualization
├── Prometheus 2.45+                # Metrics collection
└── Loki 2.8+                       # Log aggregation

Experiment Tracking:
├── Weights & Biases (wandb)        # ML experiment tracking
├── MLflow 2.6+                     # Alternative: open-source
└── TensorBoard 2.13+               # PyTorch native
```

---

### **Infrastructure & Deployment**
```
Containerization:
├── Docker 24.0+                    # Containerization
└── Docker Compose 2.20+            # Multi-container orchestration

Orchestration (Optional for scaling):
└── Kubernetes 1.27+                # Container orchestration

CI/CD:
├── GitHub Actions                  # Automated testing/deployment
└── GitLab CI/CD                    # Alternative

Process Management:
├── Supervisor 4.2+                 # Keep processes running
└── systemd                         # Linux service manager

Web Server:
└── Nginx 1.24+                     # Reverse proxy, load balancer
```

---

### **Development Tools**
```
IDE/Editors:
├── Cursor / VS Code                # AI-assisted coding
├── PyCharm Professional            # Full Python IDE
└── Jupyter Lab                     # Interactive development

Code Quality:
├── black 23.0+                     # Code formatter
├── pylint 2.17+                    # Linter
├── mypy 1.4+                       # Type checker
├── pytest 7.4+                     # Testing framework
└── pytest-cov 4.1+                 # Code coverage

Version Control:
├── Git 2.40+                       # Version control
└── DVC 3.0+                        # Data version control

Dependency Management:
├── Poetry 1.5+                     # Dependency manager (RECOMMENDED)
└── pip + virtualenv                # Alternative
```

---

### **Message Queue / Task Scheduler (Optional)**
```
Background Tasks:
├── Celery 5.3+                     # Distributed task queue
│   └── RabbitMQ 3.12+             # Message broker
│
└── Apache Airflow 2.7+             # Workflow orchestration
    ├── Schedule model retraining
    ├── Daily data collection
    └── Backtest automation
```

---

## 📊 Complete System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Streamlit  │  │   Grafana    │  │ Jupyter Lab  │     │
│  │  Dashboard   │  │  Monitoring  │  │  Analysis    │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼─────────────┐
│                    FASTAPI BACKEND                           │
│  ┌────────────────────────────────────────────────────┐     │
│  │  /predict      /backtest     /health   /metrics    │     │
│  │  (REST API endpoints)                              │     │
│  └─────────┬──────────────────────────────────────────┘     │
│            │                                                 │
│  ┌─────────▼──────────────────────────────────────────┐     │
│  │         Business Logic Layer                       │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐        │     │
│  │  │ Feature  │  │  Model   │  │ Backtest │        │     │
│  │  │ Engineer │  │ Predictor│  │  Engine  │        │     │
│  │  └──────────┘  └──────────┘  └──────────┘        │     │
│  └────────────────────────────────────────────────────┘     │
└──────────────────────────┬───────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
┌─────────▼──────┐  ┌──────▼──────┐  ┌─────▼──────┐
│     REDIS      │  │ TimescaleDB │  │  Parquet   │
│   (Cache)      │  │ (Time-Series│  │   Files    │
│                │  │     Data)   │  │            │
│ • Features     │  │             │  │ • Models   │
│ • Predictions  │  │ • OHLCV     │  │ • Backups  │
│ • Sessions     │  │ • Features  │  │            │
│                │  │ • Trades    │  │            │
└────────────────┘  └─────────────┘  └────────────┘
          │                │                │
          └────────────────┼────────────────┘
                           │
          ┌────────────────┴────────────────┐
          │                                 │
┌─────────▼──────┐              ┌───────────▼──────┐
│   Data Feed    │              │   Monitoring     │
│                │              │                  │
│ • MT5 API      │              │ • Prometheus     │
│ • CCXT         │              │ • Grafana        │
│ • WebSockets   │              │ • Alerting       │
└────────────────┘              └──────────────────┘

# Core ML
lightgbm==4.0.0
scikit-learn==1.3.0
optuna==3.3.0

# Data processing
pandas==2.1.0
polars==0.19.0
numpy==1.24.0
pyarrow==13.0.0

# Technical analysis
ta-lib==0.4.28
pandas-ta==0.3.14b

# API
fastapi==0.103.0
uvicorn[standard]==0.23.0
pydantic==2.3.0

# Data collection
MetaTrader5==5.0.45
ccxt==4.0.0
requests==2.31.0

# Database
psycopg2-binary==2.9.7
sqlalchemy==2.0.0
redis==4.6.0

# Visualization
matplotlib==3.7.0
seaborn==0.12.0
plotly==5.16.0
streamlit==1.27.0

# Utilities
python-dotenv==1.0.0
pyyaml==6.0.1
joblib==1.3.0

# Testing
pytest==7.4.0
pytest-cov==4.1.0

# Monitoring (optional)
prometheus-client==0.17.0

