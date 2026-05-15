import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from transformers import (
    PatchTSTConfig,
    PatchTSTForPrediction
)

from sklearn.preprocessing import StandardScaler
from torch.utils.data import (
    Dataset,
    DataLoader
)

from sqlalchemy import create_engine

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

CONTEXT = 96
HORIZON = 96
BATCH_SIZE = 32

engine = create_engine(DATABASE_URL)

# =========================================================
# DATASET
# =========================================================

class TransformerDataset(Dataset):

    def __init__(
        self,
        values,
        context,
        horizon
    ):

        self.values = values
        self.context = context
        self.horizon = horizon

    def __len__(self):

        return (
            len(self.values)
            - self.context
            - self.horizon
        )

    def __getitem__(self, idx):

        x = (
            self.values[
                idx:idx+self.context
            ]
        )

        y = (
            self.values[
                idx+self.context:
                idx+self.context+self.horizon
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
# LOAD DATA
# =========================================================

def load_series(
    target="price_de_lu"
):

    query = """
    SELECT *
    FROM smard_market_data
    ORDER BY timestamp
    """

    df = pd.read_sql(query, engine)

    df = create_features(df)

    series = df[target].values

    scaler = StandardScaler()

    series = scaler.fit_transform(
        series.reshape(-1,1)
    ).flatten()

    return series

# =========================================================
# TRAIN
# =========================================================

def train_transformer():

    values = load_series()

    dataset = TransformerDataset(
        values,
        CONTEXT,
        HORIZON
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    config = PatchTSTConfig(
        context_length=CONTEXT,
        prediction_length=HORIZON,

        num_input_channels=1,

        d_model=32,
        num_attention_heads=4,
        num_hidden_layers=2,

        patch_length=16,
        patch_stride=8
    )

    model = PatchTSTForPrediction(
        config
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # -----------------------------------------------------
    # TRAIN LOOP
    # -----------------------------------------------------

    for epoch in range(5):

        model.train()

        total_loss = 0

        for xb, yb in loader:

            xb = xb.unsqueeze(-1).to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(
                past_values=xb,
                future_values=yb
            )

            loss = outputs.loss

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1} "
            f"Loss {total_loss:.4f}"
        )

    torch.save(
        model.state_dict(),
        "models/patchtst.pth"
    )

    print("Transformer saved")

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    train_transformer()