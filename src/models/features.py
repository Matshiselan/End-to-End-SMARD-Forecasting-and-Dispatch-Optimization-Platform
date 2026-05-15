import numpy as np
import pandas as pd
import holidays

# =========================================================
# FEATURE AVAILABILITY REGISTRY
# =========================================================

FEATURE_AVAILABILITY = {
    "known_future": [
        "hour",
        "dayofweek",
        "month",
        "quarter_hour",
        "is_weekend",
        "is_holiday",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_dst"
    ],

    "observed_only": [
        "price_de_lu",
        "gen_solar",
        "gen_onshore_wind",
        "cons_total_grid",
        "gen_nuclear",
        "gen_hydro",
        "gen_biomass",
        "gen_offshore_wind"
    ]
}

# =========================================================
# TEMPORAL FEATURES
# =========================================================

def create_temporal_features(df: pd.DataFrame):

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    # -----------------------------------------------------
    # BASIC TIME FEATURES
    # -----------------------------------------------------

    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute

    df["quarter_hour"] = (
        df["minute"] // 15
    )

    df["dayofweek"] = (
        df["timestamp"].dt.dayofweek
    )

    df["dayofmonth"] = (
        df["timestamp"].dt.day
    )

    df["month"] = (
        df["timestamp"].dt.month
    )

    df["quarter"] = (
        df["timestamp"].dt.quarter
    )

    df["year"] = (
        df["timestamp"].dt.year
    )

    df["weekofyear"] = (
        df["timestamp"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    # -----------------------------------------------------
    # WEEKEND
    # -----------------------------------------------------

    df["is_weekend"] = (
        df["dayofweek"]
        .isin([5, 6])
        .astype(int)
    )

    # -----------------------------------------------------
    # GERMAN HOLIDAYS
    # -----------------------------------------------------

    de_holidays = holidays.Germany(
        years=range(2020, 2030)
    )

    df["is_holiday"] = (
        df["timestamp"]
        .dt.date
        .astype(str)
        .isin(
            [str(x) for x in de_holidays]
        )
        .astype(int)
    )

    # -----------------------------------------------------
    # DST
    # -----------------------------------------------------

    berlin_time = (
        df["timestamp"]
        .dt.tz_convert("Europe/Berlin")
    )

    df["is_dst"] = (
        berlin_time
        .dt.dst()
        .dt.total_seconds()
        .fillna(0)
        .ne(0)
        .astype(int)
    )

    # -----------------------------------------------------
    # CYCLICAL FEATURES
    # -----------------------------------------------------

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["dow_sin"] = np.sin(
        2 * np.pi * df["dayofweek"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["dayofweek"] / 7
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    return df

# =========================================================
# LAG FEATURES
# =========================================================

def create_lag_features(
    df: pd.DataFrame,
    columns: list
):

    df = df.copy()

    # Energy-domain relevant lags
    lags = [
        1,      # 15 mins
        4,      # 1 hour
        96,     # previous day
        672     # previous week
    ]

    for col in columns:

        if col not in df.columns:
            continue

        for lag in lags:

            df[f"{col}_lag_{lag}"] = (
                df[col].shift(lag)
            )

    return df

# =========================================================
# ROLLING FEATURES
# =========================================================

def create_rolling_features(
    df: pd.DataFrame,
    columns: list
):

    df = df.copy()

    windows = {
        "1h": 4,
        "24h": 96
    }

    for col in columns:

        if col not in df.columns:
            continue

        for name, window in windows.items():

            shifted = df[col].shift(1)

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

    return df

# =========================================================
# RAMP FEATURES
# =========================================================

def create_ramp_features(
    df: pd.DataFrame,
    columns: list
):

    df = df.copy()

    for col in columns:

        if col not in df.columns:
            continue

        df[f"{col}_ramp_1"] = (
            df[col].diff(1)
        )

        df[f"{col}_ramp_4"] = (
            df[col].diff(4)
        )

    return df

# =========================================================
# DOMAIN FEATURES
# =========================================================

def create_domain_features(df):

    df = df.copy()

    renewable_cols = [
        "gen_solar",
        "gen_onshore_wind",
        "gen_offshore_wind",
        "gen_biomass",
        "gen_hydro"
    ]

    available = [
        c for c in renewable_cols
        if c in df.columns
    ]

    df["total_renewable_gen"] = (
        df[available]
        .sum(axis=1)
    )

    if "cons_total_grid" in df.columns:

        df["residual_load"] = (
            df["cons_total_grid"]
            - df["total_renewable_gen"]
        )

        df["renewable_share"] = (
            df["total_renewable_gen"]
            / df["cons_total_grid"]
            .replace(0, np.nan)
        )

    return df

# =========================================================
# MISSING VALUES
# =========================================================

def handle_missing(df):

    df = df.copy()

    # NEVER bfill time series forecasting data

    for col in df.columns:

        if col == "timestamp":
            continue

        if df[col].dtype.kind in "biufc":

            df[col] = (
                df[col]
                .ffill()
            )

    return df

# =========================================================
# MAIN FEATURE PIPELINE
# =========================================================

def create_features(df: pd.DataFrame):

    df = df.copy()

    df = df.sort_values("timestamp")

    observed_cols = [
        "price_de_lu",
        "gen_solar",
        "gen_onshore_wind",
        "cons_total_grid",
        "gen_nuclear",
        "gen_hydro",
        "gen_biomass",
        "gen_offshore_wind"
    ]

    observed_cols = [
        c for c in observed_cols
        if c in df.columns
    ]

    df = create_temporal_features(df)

    df = create_lag_features(
        df,
        observed_cols
    )

    df = create_rolling_features(
        df,
        observed_cols
    )

    df = create_ramp_features(
        df,
        observed_cols
    )

    df = create_domain_features(df)

    df = handle_missing(df)

    return df