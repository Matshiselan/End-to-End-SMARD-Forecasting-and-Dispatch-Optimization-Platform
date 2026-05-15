import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb

from pathlib import Path
from sqlalchemy import create_engine

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from src.config import DATABASE_URL
from src.models.features import create_features

# =========================================================
# CONFIG
# =========================================================

np.random.seed(42)

engine = create_engine(DATABASE_URL)

# =========================================================
# METRICS
# =========================================================

def evaluate(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mape = (
        np.mean(
            np.abs(
                (y_true - y_pred)
                / np.maximum(y_true, 1e-6)
            )
        ) * 100
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }

# =========================================================
# WALK FORWARD SPLIT
# =========================================================

def chronological_split(df):

    n = len(df)

    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    train = df.iloc[:train_end]
    valid = df.iloc[train_end:valid_end]
    test = df.iloc[valid_end:]

    return train, valid, test

# =========================================================
# TRAIN
# =========================================================

def train_lgbm(
    target="price_de_lu",
    horizon=96
):

    print("=" * 60)
    print(f"Training target: {target}")
    print("=" * 60)

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    query = """
    SELECT *
    FROM smard_market_data
    ORDER BY timestamp
    """

    df = pd.read_sql(query, engine)

    df = create_features(df)

    # -----------------------------------------------------
    # TARGET
    # -----------------------------------------------------

    df[f"{target}_target"] = (
        df[target]
        .shift(-horizon)
    )

    # Remove future leakage rows
    df = df.dropna(
        subset=[f"{target}_target"]
    )

    # -----------------------------------------------------
    # FEATURE FILTERING
    # -----------------------------------------------------

    leakage_cols = [
        "timestamp",
        f"{target}_target"
    ]

    feature_cols = [
        c for c in df.columns
        if c not in leakage_cols
    ]

    # -----------------------------------------------------
    # SPLIT
    # -----------------------------------------------------

    train_df, valid_df, test_df = (
        chronological_split(df)
    )

    X_train = train_df[feature_cols]
    y_train = train_df[f"{target}_target"]

    X_valid = valid_df[feature_cols]
    y_valid = valid_df[f"{target}_target"]

    X_test = test_df[feature_cols]
    y_test = test_df[f"{target}_target"]

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(
        X_train,
        y_train,

        eval_set=[
            (X_valid, y_valid)
        ],

        eval_metric="l1",

        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(50)
        ]
    )

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    preds = model.predict(X_test)

    metrics = evaluate(
        y_test,
        preds
    )

    print("\nTEST METRICS")
    print(metrics)

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    })

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
    )

    print("\nTOP FEATURES")
    print(importance.head(20))

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    model_dir = Path("models")

    model_dir.mkdir(exist_ok=True)

    model_path = (
        model_dir /
        f"lgbm_{target}_{horizon}.pkl"
    )

    feature_path = (
        model_dir /
        f"lgbm_{target}_{horizon}_features.pkl"
    )

    joblib.dump(model, model_path)
    joblib.dump(feature_cols, feature_path)

    print(f"\nSaved model -> {model_path}")

    return model

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    train_lgbm(
        target="price_de_lu",
        horizon=96
    )

    train_lgbm(
        target="gen_solar",
        horizon=96
    )

    train_lgbm(
        target="gen_onshore_wind",
        horizon=96
    )