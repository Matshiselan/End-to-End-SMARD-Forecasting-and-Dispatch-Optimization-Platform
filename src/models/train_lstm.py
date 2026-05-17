import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from src.config import DATABASE_URL
from src.models.features import create_day_ahead_features
from src.models.datasets import prepare_modeling_matrices

def create_lstm_sequences(X_df, y_series):
    """
    Groups tabular hourly data into continuous 24-hour daily blocks.
    Returns: X_seq (N_days, 24, N_features), y_seq (N_days, 24)
    """
    n_days = len(X_df) // 24
    X_trimmed = X_df.iloc[:n_days * 24].values
    y_trimmed = y_series.iloc[:n_days * 24].values
    X_seq = X_trimmed.reshape(n_days, 24, X_df.shape[1])
    y_seq = y_trimmed.reshape(n_days, 24)
    return X_seq, y_seq

class DayAheadLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(DayAheadLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out.squeeze(-1)

def train_lstm():
    engine = create_engine(DATABASE_URL)
    query = """
        SELECT * FROM smard_market_data ORDER BY timestamp
    """
    df = pd.read_sql(query, engine)
    df = create_day_ahead_features(df)
    X, y = prepare_modeling_matrices(df)

    # Data cleaning
    valid_target_mask = ~y.isna()
    X_clean = X[valid_target_mask].copy().ffill().bfill()
    y_clean = y[valid_target_mask].copy()

    # Scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    y_reshaped = y_clean.values.reshape(-1, 1)
    X_scaled = pd.DataFrame(scaler_X.fit_transform(X_clean), columns=X_clean.columns)
    y_scaled = scaler_y.fit_transform(y_reshaped).flatten()

    # Sequence creation
    X_seq, y_seq = create_lstm_sequences(X_scaled, pd.Series(y_scaled))
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

    # DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_lstm = DayAheadLSTM(
        input_size=X_seq.shape[2],
        hidden_size=64,
        num_layers=2
    ).to(device)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.001)

    # Training
    print("Starting LSTM Training with Target Scaling and Huber Loss...\n")
    model_lstm.train()
    for epoch in range(30):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model_lstm(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/30] -> Current Step Training Huber Loss: {epoch_loss/len(train_loader):.4f}")

    # Inference
    model_lstm.eval()
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        preds_scaled = model_lstm(X_val_tensor).cpu().numpy()
    preds_lstm_actual = scaler_y.inverse_transform(preds_scaled).flatten()
    y_val_actual = scaler_y.inverse_transform(y_val).flatten()
    mae_lstm = mean_absolute_error(y_val_actual, preds_lstm_actual)
    rmse_lstm = root_mean_squared_error(y_val_actual, preds_lstm_actual)
    print(f"\nFinal Cleaned Validation LSTM Model Metrics:")
    print(f"➡ MAE: {mae_lstm:.2f} EUR/MWh")
    print(f"➡ RMSE: {rmse_lstm:.2f} EUR/MWh")

    # Save model and scalers
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model_lstm.state_dict(), "models/lstm_model_state.pth")
    joblib.dump(scaler_X, "models/lstm_scaler_X.pkl")
    joblib.dump(scaler_y, "models/lstm_scaler_y.pkl")
    print("LSTM model state and scalers saved to 'models/' directory.")

if __name__ == "__main__":
    train_lstm()
