
---

# End-to-End SMARD Forecasting and Dispatch Optimization Platform

A comprehensive platform for modeling, forecasting, and optimizing German energy production, consumption, and market prices using SMARD data.

## Project Overview

This platform enables:
- Extraction and ETL of SMARD market data
- Time series forecasting of generation, consumption, and prices
- Exploratory data analysis (EDA) and feature engineering
- Model training (LightGBM, LSTM, Transformers)
- Dispatch optimization and scenario analysis

## Features

- Modular ETL pipeline for SMARD data
- Flexible model training and evaluation scripts
- Jupyter notebooks for EDA and visualization
- Dockerized environment for reproducibility
- Example database schema and connection instructions

## Folder Structure

```
.
├── docker-compose.yml         # Docker orchestration for services
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── notebooks/                 # Jupyter notebooks for EDA and analysis
│   └── eda.ipynb
├── models/                    # Saved model artifacts
│   ├── lgbm_gen_onshore_wind_96steps.pkl
│   ├── lgbm_gen_solar_96steps.pkl
│   └── lgbm_price_de_lu_96steps.pkl
├── src/                       # Source code
│   ├── config.py              # Configuration (DB, paths, etc.)
│   ├── etl/                   # ETL scripts
│   │   └── fetch_smard.py
│   ├── models/                # Model training and feature engineering
│   │   ├── train_lgbm.py
│   │   ├── train_tft.py
│   │   ├── features.py
│   │   ├── backtest.py
│   │   ├── evaluate.py
│   │   ├── huggingface_transformer.py
│   │   ├── lstm.py
│   │   └── predict.py
│   └── optimization/          # (Optional) Dispatch optimization scripts
└── ...
```

## Setup & Usage

### 1. Environment Setup

- Clone the repository and navigate to the project root.
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- (Optional) Use Docker for a reproducible environment:
  ```bash
  docker-compose up
  ```

### 2. Database

- Ensure PostgreSQL is running (see docker-compose.yml or your local setup).
- Connect to the database:
  ```bash
  psql -h localhost -U postgres -d energy_db
  ```
- Inspect tables:
  ```
  \dt
  \d smard_market_data
  ```

### 3. Data ETL

- Fetch and load SMARD data:
  ```bash
  python src/etl/fetch_smard.py
  ```

### 4. Exploratory Data Analysis

- Open and run the EDA notebook:
  ```
  notebooks/eda.ipynb
  ```

### 5. Model Training

- Train models (example for LightGBM):
  ```bash
  python -m src.models.train_lgbm
  ```

## Example: Database Table Structure

The main table `smard_market_data` includes:

| Column               | Type                      | Description                |
|----------------------|--------------------------|----------------------------|
| timestamp            | timestamp with time zone  | Data timestamp             |
| gen_nuclear          | numeric(14,4)             | Nuclear generation         |
| gen_lignite          | numeric(14,4)             | Lignite generation         |
| ...                  | ...                       | ...                        |
| price_de_lu          | numeric(14,4)             | German/Lux price           |
| proj_onshore         | numeric(14,4)             | Forecast onshore wind      |
| ...                  | ...                       | ...                        |

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

## License

MIT License

---

Let me know if you want to add badges, contact info, or further customization!