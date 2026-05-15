
---

# End-to-End SMARD Forecasting and Dispatch Optimization Platform

A comprehensive platform for modeling, forecasting, and optimizing German energy production, consumption, and market prices using SMARD data.


## Overview

End-to-End SMARD Forecasting and Dispatch Optimization Platform is a comprehensive solution for modeling, forecasting, and optimizing German energy production, consumption, and market prices using SMARD data. The platform supports data extraction, feature engineering, model training, evaluation, and dispatch optimization in a modular, reproducible environment.

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


## Getting Started


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


### 2. Database Setup

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

## Database Table Structure

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



### 3. Data Extraction (ETL)

- Fetch and load SMARD data:
  ```bash
  python src/etl/fetch_smard.py
  ```

**Note:**
By default, the data extraction script fetches data for the period **May 2020 to May 2025**. This is controlled by the following line in `src/etl/fetch_smard.py`:

```python
target_timestamps = [ts for ts in available_timestamps if 1588284000000 <= ts <= 1746057600000]
```

You can adjust the date range by modifying the timestamp values in this line. The values represent milliseconds since epoch (Unix time). For other periods, update these values accordingly to fetch your desired date range.


### 4. Exploratory Data Analysis (EDA)

- Open and run the EDA notebook:
  ```
  notebooks/eda.ipynb
  ```


---

## Feature Engineering

Feature engineering is handled in `src/models/features.py` via the `create_features(df)` function. This script:

- Extracts temporal features (hour, day of week, month, quarter-hour)
- Flags weekends and German public holidays
- Adds daylight saving time (DST) indicator
- Creates cyclical features (sine/cosine transforms for hour, day, month)
- Computes domain features (total renewable generation, residual load)
- Adds lagged features and rolling statistics (mean, std) for key variables
- Calculates ramp rates (short-term changes)
- Handles missing values with forward fill

To customize or add features, edit `src/models/features.py`.

---

## Dataset Preparation

The script `src/models/dataset.py` provides the `build_direct_dataset` function, which:

- Selects features and target columns from the DataFrame
- Scales features using `StandardScaler`
- Constructs input sequences (X) and target sequences (Y) for time series models
- Supports configurable sequence length and forecast horizon
- Returns numpy arrays (float32) for efficient model training, along with timestamps and feature names

This function is used by all model training scripts to prepare data for supervised learning.

---

## Modeling & Testing

### 1. Model Training

- LightGBM:
  ```bash
  python -m src.models.train_lgbm
  ```
- LSTM:
  ```bash
  python -m src.models.lstm
  ```
- Temporal Fusion Transformer (TFT):
  ```bash
  python -m src.models.train_tft
  ```
- CNN:
  ```bash
  python -m src.models.train_cnn
  ```

Each script supports configuration of target variable, forecast horizon, and other hyperparameters. Edit the script or pass arguments as needed.


### 2. Model Evaluation

- Evaluate model performance:
  ```bash
  python -m src.models.evaluate
  ```
- Backtest models:
  ```bash
  python -m src.models.backtest
  ```


### 3. Prediction

- Generate predictions using a trained model:
  ```bash
  python -m src.models.predict
  ```


### 4. Testing

- Unit tests can be added under a `tests/` directory (not included by default).
- To test individual modules, run them as scripts or use pytest:
  ```bash
  pytest src/models/features.py
  ```



## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

## License

MIT License

---

Let me know if you want to add badges, contact info, or further customization!