# ruff: noqa: N803, N806  — sklearn API convention uses uppercase X
"""Shared deep learning training utilities for PyTorch models."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _get_device() -> str:
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def auto_batch_size(n_features: int, default: int = 1024) -> int:
    """Pick batch size based on available VRAM. Conservative for unknown GPUs."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return min(default, 512)
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if vram_gb >= 16:
            return 2048
        if vram_gb >= 8:
            return 1024
        return 512
    except RuntimeError:
        return default


class TabularDataset(Dataset):  # type: ignore[misc]
    """Simple dataset wrapping numpy arrays as float32 tensors."""

    def __init__(self, X: NDArray[np.float64], y: NDArray[np.float64] | None = None) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return (self.X[idx],)


class EarlyStopping:
    """Track validation loss and restore best model weights."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-5) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None
        self.best_state: dict | None = None  # type: ignore[type-arg]

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def fit_pytorch_model(
    model: nn.Module,
    train_X: NDArray[np.float64],
    train_y: NDArray[np.float64],
    val_X: NDArray[np.float64] | None = None,
    val_y: NDArray[np.float64] | None = None,
    *,
    epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
) -> nn.Module:
    """Full training loop with early stopping, LR scheduling, and mixed precision."""
    device = _get_device()
    model = model.to(device)

    train_ds = TabularDataset(train_X, train_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    val_loader = None
    if val_X is not None and val_y is not None:
        val_ds = TabularDataset(val_X, val_y)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    early_stop = EarlyStopping(patience=patience)

    for _epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            X_b, y_b = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
                pred = model(X_b).squeeze(-1)
                loss = criterion(pred, y_b)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            n_batches += 1

        avg_train = train_loss / max(n_batches, 1)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    X_b, y_b = batch[0].to(device), batch[1].to(device)
                    pred = model(X_b).squeeze(-1)
                    val_loss += criterion(pred, y_b).item()
                    val_n += 1
            avg_val = val_loss / max(val_n, 1)
            scheduler.step(avg_val)
            if early_stop.step(avg_val, model):
                early_stop.restore(model)
                break
        else:
            scheduler.step(avg_train)

    if val_loader is not None and early_stop.best_state is not None:
        early_stop.restore(model)

    return model


def predict_pytorch(
    model: nn.Module,
    X: NDArray[np.float64],
    batch_size: int = 2048,
) -> NDArray[np.float64]:
    """Run inference on numpy array, return numpy predictions."""
    device = _get_device()
    model = model.to(device).eval()
    ds = TabularDataset(X)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            X_b = batch[0].to(device)
            pred = model(X_b).squeeze(-1)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds)
