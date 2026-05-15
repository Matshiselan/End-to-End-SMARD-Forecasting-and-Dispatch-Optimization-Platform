import numpy as np
import pandas as pd
import holidays

KNOWN_FUTURE = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
    "is_dst"
]

BASE_OBSERVED = [
    "price_de_lu",
    "gen_solar",
    "gen_onshore_wind",
    "cons_total_grid"
]

def create_features(df):

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = df.sort_values("timestamp")

    # =====================================================
    # TEMPORAL
    # =====================================================

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    df["quarter_hour"] = (
        df["timestamp"].dt.minute // 15
    )

    df["is_weekend"] = (
        df["dayofweek"]
        .isin([5, 6])
        .astype(int)
    )

    # =====================================================
    # HOLIDAYS
    # =====================================================

    de_holidays = holidays.Germany(
        years=range(2020, 2030)
    )

    df["is_holiday"] = (
        df["timestamp"]
        .dt.date
        .astype(str)
        .isin([str(x) for x in de_holidays])
        .astype(int)
    )

    # =====================================================
    # DST
    # =====================================================

    berlin_time = (
        df["timestamp"]
        .dt.tz_convert("Europe/Berlin")
    )

    df["is_dst"] = (
        berlin_time.apply(lambda x: x.dst().total_seconds() if pd.notnull(x) and x.dst() is not None else 0)
        .fillna(0)
        .ne(0)
        .astype(int)
    )

    # =====================================================
    # CYCLICAL
    # =====================================================

    df["hour_sin"] = np.sin(
        2*np.pi*df["hour"]/24
    )

    df["hour_cos"] = np.cos(
        2*np.pi*df["hour"]/24
    )

    df["dow_sin"] = np.sin(
        2*np.pi*df["dayofweek"]/7
    )

    df["dow_cos"] = np.cos(
        2*np.pi*df["dayofweek"]/7
    )

    df["month_sin"] = np.sin(
        2*np.pi*df["month"]/12
    )

    df["month_cos"] = np.cos(
        2*np.pi*df["month"]/12
    )

    # =====================================================
    # DOMAIN FEATURES
    # =====================================================

    renewable_cols = [
        "gen_solar",
        "gen_onshore_wind"
    ]

    existing = [
        c for c in renewable_cols
        if c in df.columns
    ]

    df["total_renewable_gen"] = (
        df[existing]
        .sum(axis=1)
    )

    df["residual_load"] = (
        df["cons_total_grid"]
        - df["total_renewable_gen"]
    )

    # =====================================================
    # LAGS
    # =====================================================

    lag_cols = BASE_OBSERVED + [
        "residual_load"
    ]

    lags = [1, 4, 96, 672]

    for col in lag_cols:

        if col not in df.columns:
            continue

        for lag in lags:

            df[f"{col}_lag_{lag}"] = (
                df[col]
                .shift(lag)
            )

    # =====================================================
    # ROLLING
    # =====================================================

    rolling_windows = {
        "1h": 4,
        "24h": 96
    }

    for col in lag_cols:

        if col not in df.columns:
            continue

        shifted = df[col].shift(1)

        for name, window in rolling_windows.items():

            df[f"{col}_roll_mean_{name}"] = (
                shifted
                .rolling(window)
                .mean()
            )

            df[f"{col}_roll_std_{name}"] = (
                shifted
                .rolling(window)
                .std()
            )

    # =====================================================
    # RAMPS
    # =====================================================

    for col in lag_cols:

        if col not in df.columns:
            continue

        df[f"{col}_ramp_1"] = (
            df[col]
            .diff(1)
        )

        df[f"{col}_ramp_4"] = (
            df[col]
            .diff(4)
        )

    # =====================================================
    # CLEAN
    # =====================================================

    for col in df.columns:

        if col == "timestamp":
            continue

        if pd.api.types.is_numeric_dtype(df[col]):

            df[col] = (
                df[col]
                .ffill()
            )

    return df