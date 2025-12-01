from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float


def train_regressor(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    config: TrainConfig,
) -> nn.Module:
    dataset = TensorDataset(features, targets)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(config.epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
    return model


def predict(model: nn.Module, features: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(features).detach().cpu().numpy()
