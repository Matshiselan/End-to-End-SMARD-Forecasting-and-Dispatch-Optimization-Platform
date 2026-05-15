import torch
import torch.nn as nn

import pandas as pd

from pathlib import Path
from sqlalchemy import create_engine

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from src.config import DATABASE_URL
from src.models.features import create_features
from src.models.dataset import build_direct_dataset

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

engine = create_engine(DATABASE_URL)

class CNNForecaster(nn.Module):

    def __init__(
        self,
        num_features,
        horizon
    ):

        super().__init__()

        self.conv1 = nn.Conv1d(
            num_features,
            64,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv1d(
            64,
            128,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(
            128,
            horizon
        )

    def forward(self, x):

        x = x.permute(0,2,1)

        x = self.relu(
            self.conv1(x)
        )

        x = self.relu(
            self.conv2(x)
        )

        x = self.pool(x)

        x = x.squeeze(-1)

        return self.fc(x)

def train_cnn():

    df = pd.read_sql(
        "SELECT * FROM smard_market_data ORDER BY timestamp",
        engine
    )

    df = create_features(df)

    X, Y, _, _, _ = build_direct_dataset(
        df,
        target="price_de_lu"
    )

    X = torch.tensor(
        X,
        dtype=torch.float32
    )

    Y = torch.tensor(
        Y,
        dtype=torch.float32
    )

    ds = TensorDataset(X,Y)

    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=False
    )

    model = CNNForecaster(
        X.shape[2],
        Y.shape[1]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.MSELoss()

    for epoch in range(10):

        model.train()

        total_loss = 0

        for xb, yb in loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            pred = model(xb)

            loss = criterion(
                pred,
                yb
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1} "
            f"Loss {total_loss:.4f}"
        )

    Path("models").mkdir(
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        "models/cnn.pth"
    )

    print("CNN saved")

if __name__ == "__main__":

    train_cnn()