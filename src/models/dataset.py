import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

def build_direct_dataset(
    df,
    target,
    horizon=96,
    sequence_length=96
):

    df = df.copy()

    feature_cols = [
        c for c in df.columns
        if c not in [
            "timestamp"
        ]
    ]

    X_data = df[feature_cols].values
    y_data = df[target].values

    scaler = StandardScaler()

    X_data = scaler.fit_transform(X_data)

    X = []
    Y = []
    timestamps = []

    for i in range(
        sequence_length,
        len(df)-horizon
    ):

        x = X_data[
            i-sequence_length:i
        ]

        y = y_data[
            i:i+horizon
        ]

        X.append(x)
        Y.append(y)

        timestamps.append(
            df.iloc[i]["timestamp"]
        )

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    return (
        X,
        Y,
        timestamps,
        scaler,
        feature_cols
    )