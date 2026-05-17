
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from src.config import DATABASE_URL
from src.models.features import create_day_ahead_features
from src.models.dataset import prepare_modeling_matrices, get_time_series_splits

def train_lgbm_walkforward():
    """
    Train a MultiOutput LightGBM model with walk-forward validation and save the best model and scaler.
    Reports MAE and RMSE for each fold and overall.
    """
    engine = create_engine(DATABASE_URL)
    query = """
        SELECT * FROM smard_market_data ORDER BY timestamp
    """
    df = pd.read_sql(query, engine)
    df = create_day_ahead_features(df)
    X, Y = prepare_modeling_matrices(df, target="price_de_lu")

    n_samples = len(X)
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses = [], []
    best_model = None
    best_mae = float('inf')
    y_val_last, preds_last = None, None

    print("Starting LightGBM Walk-Forward Validation...\n")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)

        model = MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        )

        model.fit(X_train_flat, Y_train)
        preds = model.predict(X_val_flat)
        mae = mean_absolute_error(Y_val, preds)
        rmse = root_mean_squared_error(Y_val, preds)
        maes.append(mae)
        rmses.append(rmse)

        print(f"Fold {fold+1}: MAE = {mae:.2f} | RMSE = {rmse:.2f}")

        # Save best model (lowest MAE)
        if mae < best_mae:
            best_mae = mae
            best_model = model

        # Save last fold for plotting
        if fold == n_splits - 1:
            y_val_last = Y_val
            preds_last = preds

    print(f"\nFinal Cross-Validation MAE: {np.mean(maes):.2f}")
    print(f"Final Cross-Validation RMSE: {np.mean(rmses):.2f}")

    # Save best model and scaler
    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_model, "models/lgbm_multi.pkl")
    joblib.dump(scaler, "models/lgbm_scaler.pkl")
    print("Best LGBM model and scaler saved to 'models/' directory.")

    # Optionally return last fold for downstream plotting
    return y_val_last, preds_last

if __name__ == "__main__":
    train_lgbm_walkforward()