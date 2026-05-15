import pandas as pd
import joblib
from datetime import datetime, timedelta
from src.models.features import create_features
from src.config import DATABASE_URL
from sqlalchemy import create_engine

engine = create_engine(DATABASE_URL)

def load_model(target: str, horizon: int = 96):
    model_path = Path("models") / f"lgbm_{target}_{horizon}steps.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def predict_next_days(days: int = 7, target: str = "price_de_lu"):
    """
    Predict next N days ahead
    """
    # Load historical data
    df = pd.read_sql("SELECT * FROM smard_market_data ORDER BY timestamp DESC LIMIT 1000", engine)
    df = df.sort_values('timestamp')
    
    df = create_features(df)
    
    model = load_model(target, horizon=96)
    
    # Get latest features
    latest = df.iloc[-1:].copy()
    
    predictions = []
    current_df = df.copy()
    
    for i in range(days * 96):   # 15-min intervals
        feat = create_features(current_df.iloc[-100:])  # use recent window
        X = feat.iloc[-1:].drop(columns=['timestamp', target], errors='ignore')
        
        pred = model.predict(X)[0]
        predictions.append(pred)
        
        # Simulate next timestep (append prediction)
        next_row = latest.copy()
        next_row['timestamp'] = next_row['timestamp'] + pd.Timedelta(minutes=15)
        next_row[target] = pred
        current_df = pd.concat([current_df, next_row], ignore_index=True)
    
    future_dates = pd.date_range(
        start=df['timestamp'].max() + pd.Timedelta(minutes=15),
        periods=len(predictions),
        freq='15T'
    )
    
    forecast_df = pd.DataFrame({
        'timestamp': future_dates,
        f'predicted_{target}': predictions
    })
    
    return forecast_df