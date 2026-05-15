import joblib
import pandas as pd

from sqlalchemy import create_engine

from src.config import DATABASE_URL
from src.models.features import create_features
from src.models.dataset import build_direct_dataset

engine = create_engine(DATABASE_URL)

def predict_lgbm():

    model = joblib.load(
        "models/lgbm_multi.pkl"
    )

    scaler = joblib.load(
        "models/lgbm_scaler.pkl"
    )

    df = pd.read_sql(
        "SELECT * FROM smard_market_data ORDER BY timestamp",
        engine
    )

    df = create_features(df)

    X, Y, timestamps, _, _ = (
        build_direct_dataset(
            df,
            target="price_de_lu"
        )
    )

    latest = X[-1:]

    latest_flat = latest.reshape(
        latest.shape[0],
        -1
    )

    pred = model.predict(
        latest_flat
    )[0]

    future_dates = pd.date_range(
        start=timestamps[-1],
        periods=len(pred),
        freq="15min"
    )

    forecast_df = pd.DataFrame({
        "timestamp": future_dates,
        "prediction": pred
    })

    return forecast_df

if __name__ == "__main__":

    print(
        predict_lgbm()
    )