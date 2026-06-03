
---
<!-- Badges -->
![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/github/license/Matshiselan/End-to-End-SMARD-Forecasting-and-Dispatch-Optimization-Platform)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Requirements](https://img.shields.io/badge/requirements-up%20to%20date-brightgreen)


# End-to-End SMARD Forecasting and Dispatch Optimization Platform

An advanced, modular platform for end-to-end forecasting and dispatch optimization in the German energy market, leveraging SMARD data, state-of-the-art machine learning (LightGBM, XGBoost, LSTM with quantile outputs), robust feature engineering, and degradation-aware battery dispatch optimization.



## Overview

This platform provides a full pipeline for:
- **Data extraction** from SMARD and storage in PostgreSQL
- **Advanced feature engineering** (calendar, cyclical, lagged, rolling, domain-specific)
- **Production-grade model training** (LightGBM, XGBoost, LSTM with probabilistic/quantile outputs)
- **Model interpretability** (SHAP for tree and deep models)
- **Robust evaluation and backtesting** (walk-forward validation, metrics, plots)
- **Degradation-aware dispatch optimization** (MILP with battery cycling penalty)
- **Reproducible, modular codebase** (scripts, notebooks, Docker)


## Folder Structure

```
.
End-to-End-SMARD-Forecasting-and-Dispatch-Optimization-Platform/
├── BESS_Automated_Report.md
├── docker-compose.yml
├── LICENSE
├── models/
│   └── lstm_model.pth
├── notebooks/
│   ├── eda.ipynb
│   └── modelling.ipynb
├── README.md
├── requirements.txt
├── src/
│   ├── __pycache__/
│   │   └── config.cpython-312.pyc
│   ├── config.py
│   ├── etl/
│   │   └── fetch_smard.py
│   ├── LLM/
│   │   ├── __pycache__/
│   │   │   └── ollama.cpython-312.pyc
│   │   └── ollama.py
│   ├── models/
│   │   ├── __pycache__/
│   │   │   ├── backtest.cpython-312.pyc
│   │   │   ├── dataset.cpython-312.pyc
│   │   │   ├── evaluate.cpython-312.pyc
│   │   │   ├── features.cpython-312.pyc
│   │   │   ├── train_cnn.cpython-312.pyc
│   │   │   └── train_lgbm.cpython-312.pyc
│   │   ├── dataset.py
│   │   ├── datasets.py
│   │   ├── features.py
│   │   ├── train_lgbm.py
│   │   ├── train_lstm.py
│   │   └── train_xgb.py
│   └── optimization/
│       └── dispatch_lp.py
└── tamp) AS latest_date 
```


## Getting Started



### 1. Environment Setup

Clone the repository and navigate to the project root.

Install dependencies:
```bash
pip install -r requirements.txt
```

Or use Docker for a reproducible environment:
```bash
docker-compose up
```



### 2. Database Setup

Ensure PostgreSQL is running (see docker-compose.yml or your local setup).
Connect to the database:
```bash
psql -h localhost -U postgres -d energy_db
```
Inspect tables:
```
\dt
\d smard_market_data
```


## Database Table Structure

The main table `smard_market_data` includes (not exhaustive):

| Column               | Type                      | Description                |
|----------------------|--------------------------|----------------------------|
| timestamp            | timestamp with time zone  | Data timestamp             |
| gen_nuclear          | numeric                   | Nuclear generation         |
| gen_lignite          | numeric                   | Lignite generation         |
| ...                  | ...                       | ...                        |
| price_de_lu          | numeric                   | German/Lux price           |
| proj_onshore         | numeric                   | Forecast onshore wind      |
| ...                  | ...                       | ...                        |




### 3. Data Extraction (ETL)

Fetch and load SMARD data:
```bash
python src/etl/fetch_smard.py
```
By default, the script fetches data for May 2020 to May 2025. Adjust the timestamp range in the script to change the period.



### 4. Exploratory Data Analysis (EDA)

Open and run the EDA notebook:
```
jupyter notebook notebooks/eda.ipynb
```


---


## Feature Engineering

Feature engineering is handled in `src/models/features.py` via `create_day_ahead_features(df)`, which:
- Extracts calendar, cyclical, and holiday features
- Adds daylight saving time (DST) indicator
- Computes domain features (renewable/fossil totals, penetration)
- Adds lagged features (24h, 48h, 168h) and rolling stats (mean, std, volatility)
- Calculates ramp rates and trends
- Handles missing values robustly

Edit `src/models/features.py` to customize or extend features.

---


## Dataset Preparation

`src/models/dataset.py` provides robust utilities:
- `prepare_modeling_matrices`: Extracts feature matrix X and target y, drops non-predictive columns, ensures no leakage
- `get_time_series_splits`: Walk-forward time series cross-validation splits

All model scripts use these functions for consistent, production-grade data handling.

---


## Modeling & Evaluation

### 1. Forecasting & Optimization
You can run open and run the Modelling notebook:
```
jupyter notebook notebooks/modelling.ipynb
```
See also these scripts

- **LightGBM** (walk-forward, multi-output):
  ```bash
  python -m src.models.train_lgbm
  ```
- **XGBoost** (walk-forward):
  ```bash
  python -m src.models.train_xgb
  ```
- **LSTM** (probabilistic/quantile, robust scaling):
  ```bash
  python -m src.models.train_lstm
  ```

All scripts support configuration of target, forecast horizon, and hyperparameters. Edit scripts or pass arguments as needed.

### 1.2. Model Evaluation & Backtesting

- Evaluate model performance:
  ```bash
  python -m src.models.evaluate
  ```
- Backtest models:
  ```bash
  python -m src.models.backtest
  ```

### 1.3. Testing

- Add unit tests under a `tests/` directory (not included by default).
- To test individual modules:
  ```bash
  pytest src/models/features.py
  ```


### 1.4. Dispatch Optimization

Degradation-aware battery dispatch optimization is implemented in `src/optimization/dispatch_lp.py`:
- MILP-based daily walk-forward backtest
- Incorporates cell degradation penalty (EUR/MWh cycled)
- Supports scenario-based (quantile) price forecasts
- Produces detailed ledger and operational plots

See `notebooks/modelling.ipynb` for example usage and visualization.

### 1.5. LLM Report Writing
To write the report we use a the Meta-Llama, run

```
python src/LLM/ollama.py
```

### 2. Causal Modelling

This model is governed by this DAG
<img width="1919" height="1446" alt="DAG" src="https://github.com/user-attachments/assets/c2aea5ee-1254-4c4d-b9bb-70ed11d48bda" />

You can run open and run the Causal Model notebook:
```
jupyter notebook notebooks/causal_modelling.ipynb

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

## License

MIT License

