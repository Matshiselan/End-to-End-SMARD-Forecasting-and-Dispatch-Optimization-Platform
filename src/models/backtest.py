import pandas as pd
import numpy as np
import joblib

from sqlalchemy import create_engine

from src.config import DATABASE_URL

from src.models.features import create_features
from src.models.dataset import build_direct_dataset
from src.models.evaluate import evaluate_forecast

engine = create_engine(DATABASE_URL)

def rolling_backtest():

    df = pd.read_sql(
        "SELECT * FROM smard_market_data ORDER BY timestamp",
        engine
    )

    df = create_features(df)

    X, Y, _, _, _ = build_direct_dataset(
        df,
        target="price_de_lu"
    )

    model = joblib.load(
        "models/lgbm_multi.pkl"
    )

    preds = []
    actuals = []

    split = int(len(X)*0.8)

    X_test = X[split:]
    Y_test = Y[split:]

    for i in range(len(X_test)):

        x = X_test[i:i+1]

        x = x.reshape(
            x.shape[0],
            -1
        )

        pred = model.predict(x)

        preds.extend(pred.flatten())
        actuals.extend(
            Y_test[i].flatten()
        )

    return evaluate_forecast(
        np.array(actuals),
        np.array(preds),
        name="LGBM Backtest"
    )

if __name__ == "__main__":

    rolling_backtest()