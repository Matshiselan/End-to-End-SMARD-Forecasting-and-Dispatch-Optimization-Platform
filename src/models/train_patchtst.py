import torch
import pandas as pd

from transformers import (
    PatchTSTConfig,
    PatchTSTForPrediction
)

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sqlalchemy import create_engine

from src.config import DATABASE_URL
from src.models.features import create_features
from src.models.dataset import build_direct_dataset

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

engine = create_engine(DATABASE_URL)

def train_patchtst():

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
        X[:,:,0],
        dtype=torch.float32
    )

    Y = torch.tensor(
        Y,
        dtype=torch.float32
    )

    ds = TensorDataset(X,Y)

    loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=False
    )

    config = PatchTSTConfig(
        context_length=96,
        prediction_length=96,
        num_input_channels=1,
        d_model=32,
        num_attention_heads=4,
        num_hidden_layers=2
    )

    model = PatchTSTForPrediction(
        config
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    for epoch in range(5):

        total_loss = 0

        for xb, yb in loader:

            xb = xb.unsqueeze(-1).to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            out = model(
                past_values=xb,
                future_values=yb
            )

            loss = out.loss

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

if __name__ == "__main__":

    train_patchtst()