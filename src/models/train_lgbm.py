import joblib
import pandas as pd

from pathlib import Path

from sqlalchemy import create_engine

from sklearn.multioutput import MultiOutputRegressor

from lightgbm import LGBMRegressor

from src.config import DATABASE_URL
from src.models.features import create_features
from src.models.dataset import build_direct_dataset

engine = create_engine(DATABASE_URL)

def train_lgbm():

    query = """
    SELECT *
    FROM smard_market_data
    ORDER BY timestamp
    """

    df = pd.read_sql(query, engine)

    df = create_features(df)

    X, Y, _, scaler, feature_cols = (
        build_direct_dataset(
            df,
            target="price_de_lu"
        )
    )

    n = len(X)

    train_end = int(n*0.8)

    X_train = X[:train_end]
    Y_train = Y[:train_end]

    X_train_flat = X_train.reshape(
        X_train.shape[0],
        -1
    )

    model = MultiOutputRegressor(
        LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            random_state=42
        )
    )

    model.fit(
        X_train_flat,
        Y_train
    )

    Path("models").mkdir(
        exist_ok=True
    )

    joblib.dump(
        model,
        "models/lgbm_multi.pkl"
    )

    joblib.dump(
        scaler,
        "models/lgbm_scaler.pkl"
    )

    print("LGBM saved")

if __name__ == "__main__":

    train_lgbm()