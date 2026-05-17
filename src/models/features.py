
import numpy as np
import pandas as pd
import holidays

def create_day_ahead_features(df):
	"""
	Prepares SMARD data for 24-hour block Day-Ahead electricity price forecasting.
	Downsamples all data to 1-hour resolution and enforces strict causal lag structures.
	Adds advanced, production-grade features for robust modeling.
	"""
	df = df.copy()

	# Ensure timestamp is datetime format and sorted chronologically
	df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
	df = df.sort_values("timestamp")
	df = df.set_index("timestamp")

	# 1. RESOLUTION ALIGNMENT (Hourly Resampling)
	df = df.resample("1h").mean()
	df = df.reset_index()

	# 2. CALENDAR & TEMPORAL FEATURES
	df["hour"] = df["timestamp"].dt.hour
	df["dayofweek"] = df["timestamp"].dt.dayofweek
	df["month"] = df["timestamp"].dt.month
	df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
	de_holidays = holidays.Germany(years=range(2020, 2030))
	df["is_holiday"] = df["timestamp"].dt.date.astype(str).isin([str(x) for x in de_holidays]).astype(int)
	berlin_time = df["timestamp"].dt.tz_convert("Europe/Berlin")
	df["is_dst"] = (berlin_time.apply(lambda x: x.utcoffset().total_seconds()) == 7200).astype(int)

	# 3. CYCLICAL TRANSFORMATIONS
	df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
	df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
	df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
	df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
	df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
	df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

	# 4. STRUCTURAL FORECAST FEATURES
	if "proj_solar" in df.columns and "proj_onshore" in df.columns and "proj_offshore" in df.columns:
		df["total_proj_renewables"] = df["proj_solar"] + df["proj_onshore"] + df["proj_offshore"]
	else:
		df["total_proj_renewables"] = df.get("proj_wind_solar", 0)

	# 5. HISTORICAL OBSERVED FEATURES
	df["actual_renewables_total"] = df.get("gen_solar", 0) + df.get("gen_onshore_wind", 0) + df.get("gen_offshore_wind", 0)
	df["actual_fossil_total"] = df.get("gen_natural_gas", 0) + df.get("gen_hard_coal", 0) + df.get("gen_lignite", 0)
	df["historical_renewable_penetration"] = (df["actual_renewables_total"] / df["cons_total_grid"].replace(0, np.nan)).fillna(0)

	# 6. CAUSAL LAG ENGINE
	lag_targets = [
		"price_de_lu", "cons_total_grid", "cons_residual",
		"actual_renewables_total", "actual_fossil_total", "price_at"
	]
	hourly_lags = [24, 48, 168]
	for col in lag_targets:
		if col not in df.columns:
			continue
		for lag in hourly_lags:
			df[f"{col}_lag_{lag}"] = df[col].shift(lag)

	# 7. ROLLING HISTORICAL VOLATILITY & TRENDS
	rolling_windows = {"24h": 24, "168h": 168}
	for col in ["price_de_lu", "cons_residual", "actual_renewables_total"]:
		if col not in df.columns:
			continue
		causal_base = df[col].shift(24)
		for name, window in rolling_windows.items():
			df[f"{col}_roll_mean_{name}"] = causal_base.rolling(window).mean()
			df[f"{col}_roll_std_{name}"] = causal_base.rolling(window).std()

	# 8. HISTORICAL PRICE RAMPS
	for col in ["price_de_lu", "cons_residual"]:
		if col not in df.columns:
			continue
		df[f"{col}_ramp_24h"] = df[col].shift(24).diff(24)

	# 9. MISSING DATA AND TARGET PROTECTION
	target_col = "price_de_lu"
	target_series = df[target_col].copy()
	for col in df.columns:
		if col == "timestamp" or col == target_col:
			continue
		if pd.api.types.is_numeric_dtype(df[col]):
			df[col] = df[col].ffill()
	df[target_col] = target_series

	# =========================================================================
	# 10. ADVANCED & ROBUST NEW FEATURES (Production Grade)
	# =========================================================================

	# Previous day price range and volatility
	df["prev_day_price_range"] = (
		df["price_de_lu"].shift(24).rolling(24, min_periods=12).apply(lambda x: x.max() - x.min(), raw=True)
	)
	df["prev_day_price_std"] = (
		df["price_de_lu"].shift(24).rolling(24, min_periods=12).std()
	)

	# Rolling min/max prices (last 7 days)
	df["price_min_7d"] = (
		df["price_de_lu"].shift(24).rolling(24*7, min_periods=24).min()
	)
	df["price_max_7d"] = (
		df["price_de_lu"].shift(24).rolling(24*7, min_periods=24).max()
	)

	# Proximity to next holiday (in days)
	holiday_dates = pd.to_datetime([str(x) for x in holidays.Germany(years=range(2020, 2030))])
	def days_to_next_holiday(ts):
		future = holiday_dates[holiday_dates >= ts.date()]
		if len(future) == 0:
			return np.nan
		return (future[0] - ts.date()).days
	df["days_to_next_holiday"] = df["timestamp"].apply(days_to_next_holiday)

	# Hour of year (captures annual seasonality)
	df["hour_of_year"] = df["timestamp"].dt.dayofyear * 24 + df["timestamp"].dt.hour

	# Lagged renewable share (yesterday's renewable penetration)
	if "historical_renewable_penetration" in df.columns:
		df["renewable_share_lag24"] = df["historical_renewable_penetration"].shift(24)
	else:
		df["renewable_share_lag24"] = np.nan

	# Price spike indicator (was there a spike yesterday?)
	df["price_spike_lag24"] = (df["price_de_lu"].shift(24).diff().abs() > 50).astype(int)

	# Weekend or holiday combined indicator
	df["is_weekend_or_holiday"] = ((df["is_weekend"] == 1) | (df["is_holiday"] == 1)).astype(int)

	# Drop early initialization rows containing NaN artifacts due to lag dependencies and new features
	required_cols = [f"price_de_lu_lag_168", "prev_day_price_range", "prev_day_price_std", "price_min_7d", "price_max_7d"]
	df = df.dropna(subset=required_cols)

	return df
