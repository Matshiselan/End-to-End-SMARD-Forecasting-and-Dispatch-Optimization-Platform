
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.model_selection import TimeSeriesSplit

def prepare_modeling_matrices(
	df: pd.DataFrame,
	target: str = "price_de_lu",
	exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
	"""
	Separates the processed dataframe into feature matrix X and target y.
	Drops non-predictive variables and ensures only valid features are used.

	Args:
		df: Input DataFrame with all features and target.
		target: Name of the target column.
		exclude_cols: List of columns to exclude from X. If None, uses default list.

	Returns:
		X: Feature matrix (pd.DataFrame)
		y: Target vector (pd.Series)
	"""
	if exclude_cols is None:
		exclude_cols = [
			"timestamp", target, "price_neighbors", "price_at",
			"gen_nuclear", "gen_lignite", "gen_offshore_wind", "gen_hydro",
			"gen_other_conv", "gen_other_renew", "gen_biomass", "gen_onshore_wind",
			"gen_solar", "gen_hard_coal", "gen_pumped_storage", "gen_natural_gas",
			"cons_total_grid", "cons_residual", "cons_pumped_storage",
			"actual_renewables_total", "actual_fossil_total", "historical_renewable_penetration"
		]
	y = df[target].copy()
	X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
	return X, y

def get_time_series_splits(
	X: pd.DataFrame,
	n_splits: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
	"""
	Utility to get time series cross-validation splits.

	Args:
		X: Feature matrix.
		n_splits: Number of splits.

	Returns:
		List of (train_idx, val_idx) tuples for each fold.
	"""
	tscv = TimeSeriesSplit(n_splits=n_splits)
	return list(tscv.split(X))
