import pandas as pd
import numpy as np
from pathlib import Path
from src.models.features import create_features
from src.models.evaluate import evaluate_forecast
from lightgbm import LGBMRegressor
from tqdm import tqdm

def rolling_backtest(target: str = "price_de_lu", 
                     horizon: int = 96, 
                     initial_train_size: float = 0.6,
                     step_size: int = 96*7):   # retrain every 7 days
    """
    Proper rolling window backtest (industry standard for energy)
    """
    from src.config import DATABASE_URL
    from sqlalchemy import create_engine
    
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("SELECT * FROM smard_market_data ORDER BY timestamp", engine)
    df = create_features(df)
    df = df.dropna(subset=[target])
    
    feature_cols = [col for col in df.columns if col not in ['timestamp', target]]
    
    predictions = []
    actuals = []
    
    train_size = int(len(df) * initial_train_size)
    
    print(f"Starting rolling backtest for {target} | Horizon: {horizon} steps")
    
    for start in tqdm(range(train_size, len(df) - horizon, step_size)):
        train = df.iloc[:start]
        test = df.iloc[start:start + horizon]
        
        model = LGBMRegressor(
            n_estimators=800, 
            learning_rate=0.05, 
            max_depth=8,
            num_leaves=64,
            random_state=42,
            verbose=-1
        )
        
        model.fit(train[feature_cols], train[target])
        pred = model.predict(test[feature_cols])
        
        predictions.extend(pred)
        actuals.extend(test[target].values)
    
    return evaluate_forecast(np.array(actuals), np.array(predictions), f"Rolling Backtest - {target}")