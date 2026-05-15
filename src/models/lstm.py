import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from torch.utils.data import (
    Dataset,
    DataLoader
)

from src.config import DATABASE_URL
from src.models.features import create_features

# =========================================================
# CONFIG
# =========================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SEQ_LEN = 96
BATCH_SIZE = 64

engine = create_engine(DATABASE_URL)

# =========================================================
# DATASET
# =========================================================

class EnergyDataset(Dataset):

    def __init__(
        self,
        X,
        y,
        seq_len
    ):

        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):

        return (
            len(self.X)
            - self.seq_len
        )

    def __getitem__(self, idx):

        x = (
            self.X[
                idx:idx+self.seq_len
            ]
        )

        y = (
            self.y[
                idx+self.seq_len
            ]
        )

        return (
            torch.tensor(
                x,
                dtype=torch.float32
            ),
            torch.tensor(
                y,
                dtype=torch.float32
            )
        )

# =========================================================
# MODEL
# =========================================================

class EnergyLSTM(nn.Module):

    def __init__(
        self,
        input_size
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )

        self.fc = nn.Linear(
            64,
            1
        )

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        return self.fc(out)

# =========================================================
# TRAIN
# =========================================================

def train_lstm(
    target="price_de_lu"
):

    query = """
    SELECT *
    FROM smard_market_data
    ORDER BY timestamp
    """

    df = pd.read_sql(query, engine)

    df = create_features(df)

    df[f"{target}_target"] = (
        df[target]
        .shift(-96)
    )

    df = df.dropna()

    exclude = [
        "timestamp",
        f"{target}_target"
    ]

    features = [
        c for c in df.columns
        if c not in exclude
    ]

    X = df[features].values
    y = df[f"{target}_target"].values

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    split = int(len(df) * 0.8)

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]

    train_ds = EnergyDataset(
        X_train,
        y_train,
        SEQ_LEN
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = EnergyLSTM(
        input_size=X.shape[1]
    ).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # -----------------------------------------------------
    # TRAIN LOOP
    # -----------------------------------------------------

    for epoch in range(10):

        model.train()

        epoch_loss = 0

        for xb, yb in train_loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            preds = model(xb).squeeze()

            loss = criterion(
                preds,
                yb
            )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        print(
            f"Epoch {epoch+1} "
            f"Loss: {epoch_loss:.4f}"
        )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    Path("models").mkdir(
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        f"models/lstm_{target}.pth"
    )

    print("LSTM model saved")

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    train_lstm(
        target="price_de_lu"
    )