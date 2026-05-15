import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray, name: str = "Model"):
    """Professional evaluation for energy time series"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE with protection against zero
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100
    
    # Energy-specific: Peak Error
    peak_error = np.abs(y_true.max() - y_pred.max())
    
    print(f"\n=== {name} Evaluation ===")
    print(f"MAE          : {mae:,.3f}")
    print(f"RMSE         : {rmse:,.3f}")
    print(f"MAPE         : {mape:.2f}%")
    print(f"R²           : {r2:.4f}")
    print(f"Peak Error   : {peak_error:,.2f}")
    
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "peak_error": peak_error
    }


def evaluate_multiple_targets(predictions: dict, actuals: pd.DataFrame):
    """Evaluate multiple targets at once"""
    results = {}
    for target, pred in predictions.items():
        if target in actuals.columns:
            results[target] = evaluate_forecast(actuals[target].values, pred, target)
    return results