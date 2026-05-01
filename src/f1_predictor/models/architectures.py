# ruff: noqa: N803, N806  — sklearn API convention uses uppercase X
"""PyTorch model architectures with sklearn-compatible API for the training pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from f1_predictor.models.dl_utils import auto_batch_size, fit_pytorch_model, predict_pytorch

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Raw PyTorch modules
# ---------------------------------------------------------------------------


class _GRUNet(nn.Module):  # type: ignore[misc]
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class _MLPNet(nn.Module):  # type: ignore[misc]
    def __init__(self, input_dim: int, hidden1: int, hidden2: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Sklearn-compatible wrappers (work with sklearn.base.clone)
# ---------------------------------------------------------------------------


class GRU2Layer(BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """Bidirectional GRU regressor for lap-level (Models A/B)."""

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 0,
        epochs: int = 80,
        patience: int = 15,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

    def fit(self, X: Any, y: Any) -> GRU2Layer:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()
        X_clean = self._scaler.fit_transform(self._imputer.fit_transform(X_arr))

        bs = self.batch_size if self.batch_size > 0 else auto_batch_size(X_clean.shape[1])
        n = len(X_clean)
        val_size = max(int(n * 0.1), 256)
        train_X, val_X = X_clean[:-val_size], X_clean[-val_size:]
        train_y, val_y = y_arr[:-val_size], y_arr[-val_size:]

        self._model = _GRUNet(
            input_dim=X_clean.shape[1],
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self._model = fit_pytorch_model(
            self._model,
            train_X,
            train_y,
            val_X,
            val_y,
            epochs=self.epochs,
            batch_size=bs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            patience=self.patience,
        )
        return self

    def predict(self, X: Any) -> NDArray[np.float64]:
        X_arr = np.asarray(X, dtype=np.float64)
        X_clean = self._scaler.transform(self._imputer.transform(X_arr))
        return predict_pytorch(self._model, X_clean)


class FTTransformerWrapper(BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """Feature Tokenizer + Transformer for tabular data (Models A/B)."""

    def __init__(
        self,
        n_features: int = 9,
        d_token: int = 64,
        n_blocks: int = 3,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 0,
        epochs: int = 80,
        patience: int = 15,
    ) -> None:
        self.n_features = n_features
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

    def _build_model(self, n_features: int) -> nn.Module:
        try:
            from rtdl_revisiting_models import FTTransformer

            return FTTransformer(
                n_cont_features=n_features,
                cat_cardinalities=[],
                d_out=1,
                n_blocks=self.n_blocks,
                d_block=self.d_token * 4,
                attention_n_heads=max(1, self.d_token // 16),
                attention_dropout=self.attention_dropout,
                ffn_d_hidden=None,
                ffn_d_hidden_multiplier=4 / 3,
                ffn_dropout=self.ffn_dropout,
                residual_dropout=0.0,
                linformer_compression_ratio=None,
                linformer_sharing_policy=None,
            )
        except (ImportError, TypeError, ValueError):
            return _MLPNet(n_features, self.d_token * 2, self.d_token, self.ffn_dropout)

    def fit(self, X: Any, y: Any) -> FTTransformerWrapper:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()
        X_clean = self._scaler.fit_transform(self._imputer.fit_transform(X_arr))

        bs = self.batch_size if self.batch_size > 0 else auto_batch_size(X_clean.shape[1])
        n = len(X_clean)
        val_size = max(int(n * 0.1), 256)
        train_X, val_X = X_clean[:-val_size], X_clean[-val_size:]
        train_y, val_y = y_arr[:-val_size], y_arr[-val_size:]

        self._model = self._build_model(X_clean.shape[1])
        self._model = fit_pytorch_model(
            self._model,
            train_X,
            train_y,
            val_X,
            val_y,
            epochs=self.epochs,
            batch_size=bs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            patience=self.patience,
        )
        return self

    def predict(self, X: Any) -> NDArray[np.float64]:
        X_arr = np.asarray(X, dtype=np.float64)
        X_clean = self._scaler.transform(self._imputer.transform(X_arr))
        return predict_pytorch(self._model, X_clean)


class MLP3Layer(BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """3-layer MLP regressor for race-level data (Model C, ~3500 rows)."""

    def __init__(
        self,
        input_dim: int = 15,
        hidden1: int = 64,
        hidden2: int = 32,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 150,
        patience: int = 20,
    ) -> None:
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

    def fit(self, X: Any, y: Any) -> MLP3Layer:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()
        X_clean = self._scaler.fit_transform(self._imputer.fit_transform(X_arr))

        n = len(X_clean)
        val_size = max(int(n * 0.15), 100)
        train_X, val_X = X_clean[:-val_size], X_clean[-val_size:]
        train_y, val_y = y_arr[:-val_size], y_arr[-val_size:]

        self._model = _MLPNet(
            input_dim=X_clean.shape[1],
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            dropout=self.dropout,
        )
        self._model = fit_pytorch_model(
            self._model,
            train_X,
            train_y,
            val_X,
            val_y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            patience=self.patience,
        )
        return self

    def predict(self, X: Any) -> NDArray[np.float64]:
        X_arr = np.asarray(X, dtype=np.float64)
        X_clean = self._scaler.transform(self._imputer.transform(X_arr))
        return predict_pytorch(self._model, X_clean)
