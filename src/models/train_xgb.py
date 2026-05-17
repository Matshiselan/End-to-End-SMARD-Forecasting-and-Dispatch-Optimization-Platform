import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sqlalchemy import create_engine
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from src.config import DATABASE_URL
from src.models.features import create_day_ahead_features
from src.models.datasets import prepare_modeling_matrices

def train_xgb_walkforward():
    """
    Train an XGBoost regressor with walk-forward time series validation.
    Saves the best model and reports MAE/RMSE for each fold and overall.
    """
    engine = create_engine(DATABASE_URL)
    query = """
        SELECT * FROM smard_market_data ORDER BY timestamp
    """
    df = pd.read_sql(query, engine)
    df = create_day_ahead_features(df)
    X, y = prepare_modeling_matrices(df)

    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses = [], []
    best_model = None
    best_mae = float('inf')
    y_val_last, preds_last = None, None

    print("Starting XGBoost Time-Series Walk-Forward Validation...\n")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_xgb = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

        model_xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = model_xgb.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = root_mean_squared_error(y_val, preds)
        maes.append(mae)
        rmses.append(rmse)

        print(f"Fold {fold+1}: MAE = {mae:.2f} EUR/MWh | RMSE = {rmse:.2f} EUR/MWh")

        if mae < best_mae:
            best_mae = mae
            best_model = model_xgb

        if fold == n_splits - 1:
            y_val_last = y_val
            preds_last = preds

    print(f"\nFinal XGBoost Cross-Validation MAE: {np.mean(maes):.2f} EUR/MWh")
    print(f"Final XGBoost Cross-Validation RMSE: {np.mean(rmses):.2f} EUR/MWh")

    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_model, "models/xgb_model.pkl")
    print("Best XGBoost model saved to 'models/xgb_model.pkl'.")

    return y_val_last, preds_last

if __name__ == "__main__":
    train_xgb_walkforward()
