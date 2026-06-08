"""PyTorch MLP with categorical embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


class EmbeddingMLP(nn.Module):
    def __init__(
        self,
        cat_cardinalities: list[int],
        n_numeric: int,
        embedding_dim: int = 16,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        self.embeddings = nn.ModuleList([nn.Embedding(n, embedding_dim) for n in cat_cardinalities])
        in_dim = len(cat_cardinalities) * embedding_dim + n_numeric
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        emb = [self.embeddings[i](x_cat[:, i]) for i in range(x_cat.shape[1])]
        x = torch.cat([x_num] + emb, dim=1)
        return self.mlp(x).squeeze(-1)


def _encode_categoricals(
    X: pd.DataFrame, cat_cols: list[str]
) -> tuple[np.ndarray, list, list[LabelEncoder]]:
    encoders: list[LabelEncoder] = []
    arrs = []
    cardinalities = []
    for col in cat_cols:
        le = LabelEncoder()
        vals = X[col].astype(str).fillna("missing")
        arrs.append(le.fit_transform(vals))
        encoders.append(le)
        cardinalities.append(len(le.classes_))
    return np.stack(arrs, axis=1), cardinalities, encoders


def train_embedding_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    embedding_dim: int = 16,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.1,
    lr: float = 0.001,
    epochs: int = 30,
    batch_size: int = 256,
    seed: int = 42,
) -> tuple[EmbeddingMLP, dict[str, Any], list[LabelEncoder]]:
    torch.manual_seed(seed)
    X_num = X_train[numeric_cols].fillna(X_train[numeric_cols].median()).values.astype(np.float32)
    x_cat, cardinalities, encoders = _encode_categoricals(X_train, categorical_cols)
    y = y_train.values.astype(np.float32)

    model = EmbeddingMLP(
        cat_cardinalities=cardinalities,
        n_numeric=X_num.shape[1],
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    ds = TensorDataset(
        torch.tensor(X_num),
        torch.tensor(x_cat, dtype=torch.long),
        torch.tensor(y),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    history: dict[str, list[float]] = {"loss": []}

    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for xn, xc, yt in loader:
            opt.zero_grad()
            logits = model(xn, xc)
            loss = criterion(logits, yt)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        history["loss"].append(epoch_loss / len(loader))

    return model, history, encoders


def predict_proba_torch(
    model: EmbeddingMLP,
    X: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    encoders: list[LabelEncoder],
) -> np.ndarray:
    model.eval()
    X_num = X[numeric_cols].fillna(X[numeric_cols].median()).values.astype(np.float32)
    arrs = []
    for col, le in zip(categorical_cols, encoders):
        vals = X[col].astype(str).fillna("missing")
        known = set(le.classes_)
        vals = vals.apply(lambda x: x if x in known else le.classes_[0])
        arrs.append(le.transform(vals))
    x_cat = np.stack(arrs, axis=1)
    with torch.no_grad():
        logits = model(torch.tensor(X_num), torch.tensor(x_cat, dtype=torch.long))
        proba = torch.sigmoid(logits).numpy()
    return proba
