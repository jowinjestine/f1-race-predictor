# ruff: noqa: N801, N803, N806  — sklearn API uses uppercase X; underscore class names are candidate IDs
"""PyTorch sequence model architectures for Model G (temporal simulation).

All models follow the sklearn-compatible wrapper pattern:
    model.fit(X_3d, y)  — X_3d shape: (n_samples, window, features)
    model.predict(X_3d) — returns (n_samples,) array
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

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

if TORCH_AVAILABLE:

    class _SeqGRUNet(nn.Module):  # type: ignore[misc]
        def __init__(
            self,
            n_features: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
            bidirectional: bool = False,
        ) -> None:
            super().__init__()
            self.gru = nn.GRU(
                n_features,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            out_dim = hidden_dim * (2 if bidirectional else 1)
            self.head = nn.Linear(out_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.gru(x)
            return self.head(out[:, -1, :])

    class _SeqLSTMNet(nn.Module):  # type: ignore[misc]
        def __init__(
            self,
            n_features: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
            bidirectional: bool = False,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                n_features,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            out_dim = hidden_dim * (2 if bidirectional else 1)
            self.head = nn.Linear(out_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    class _TCNBlock(nn.Module):  # type: ignore[misc]
        def __init__(
            self,
            in_ch: int,
            out_ch: int,
            kernel_size: int,
            dilation: int,
            dropout: float,
        ) -> None:
            super().__init__()
            padding = (kernel_size - 1) * dilation
            self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
            self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.dropout(self.relu(self.conv1(x)))
            out = out[:, :, : x.size(2)]  # causal trim
            out = self.dropout(self.relu(self.conv2(out)))
            out = out[:, :, : x.size(2)]
            return self.relu(out + self.downsample(x))

    class _SeqTCNNet(nn.Module):  # type: ignore[misc]
        def __init__(
            self,
            n_features: int,
            hidden_dim: int,
            num_layers: int,
            kernel_size: int,
            dropout: float,
        ) -> None:
            super().__init__()
            layers = []
            for i in range(num_layers):
                in_ch = n_features if i == 0 else hidden_dim
                layers.append(
                    _TCNBlock(
                        in_ch,
                        hidden_dim,
                        kernel_size,
                        dilation=2**i,
                        dropout=dropout,
                    )
                )
            self.tcn = nn.Sequential(*layers)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq, features) -> (batch, features, seq) for Conv1d
            out = self.tcn(x.transpose(1, 2))
            return self.head(out[:, :, -1])

    class _PositionalEncoding(nn.Module):  # type: ignore[misc]
        def __init__(self, d_model: int, max_len: int = 100) -> None:
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model > 1:
                pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, : x.size(1), :]

    class _SeqTransformerNet(nn.Module):  # type: ignore[misc]
        def __init__(
            self,
            n_features: int,
            d_model: int,
            n_heads: int,
            num_layers: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos_enc = _PositionalEncoding(d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.input_proj(x)
            x = self.pos_enc(x)
            x = self.encoder(x)
            return self.head(x[:, -1, :])

    class _SeqGRUAttnNet(nn.Module):  # type: ignore[misc]
        def __init__(
            self,
            n_features: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.gru = nn.GRU(
                n_features,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.attn = nn.Linear(hidden_dim, 1)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.gru(x)  # (batch, seq, hidden)
            weights = torch.softmax(self.attn(out), dim=1)  # (batch, seq, 1)
            context = (out * weights).sum(dim=1)  # (batch, hidden)
            return self.head(context)

    class _SeqCNN1DNet(nn.Module):  # type: ignore[misc]
        def __init__(
            self,
            n_features: int,
            hidden_dim: int,
            kernel_size: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(n_features, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.conv(x.transpose(1, 2))
            out = self.pool(out).squeeze(-1)
            return self.head(out)


# ---------------------------------------------------------------------------
# Sklearn-compatible wrappers
# ---------------------------------------------------------------------------


class _SeqBaseWrapper(BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """Base class for sequence model wrappers."""

    def __init__(
        self,
        n_features: int = 26,
        window_size: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 0,
        epochs: int = 80,
        patience: int = 15,
        n_heads: int = 4,
        kernel_size: int = 3,
    ) -> None:
        self.n_features = n_features
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.n_heads = n_heads
        self.kernel_size = kernel_size

    def _build_net(self) -> nn.Module:
        raise NotImplementedError

    def _prepare(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Scale features across the feature dimension, keeping temporal structure."""
        n_samples, seq_len, n_feat = X.shape
        flat = X.reshape(-1, n_feat)
        flat_scaled = self.scaler_.transform(flat)
        return flat_scaled.reshape(n_samples, seq_len, n_feat).astype(np.float32)

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> _SeqBaseWrapper:
        from f1_predictor.models.dl_utils import (
            SequenceDataset,
            auto_batch_size,
            fit_pytorch_model,
        )

        # Slice to window_size
        if X.ndim == 3 and X.shape[1] > self.window_size:
            X = X[:, -self.window_size :, :]

        _n_samples, _seq_len, n_feat = X.shape
        self.n_features = n_feat

        # Fit scaler on flattened features
        self.scaler_ = StandardScaler()
        flat = X.reshape(-1, n_feat)
        self.scaler_.fit(flat)

        X_scaled = self._prepare(X)
        y_np = np.asarray(y, dtype=np.float64)

        # Train/val split (last 20%)
        split = int(len(X_scaled) * 0.8)
        X_tr, X_va = X_scaled[:split], X_scaled[split:]
        y_tr, y_va = y_np[:split], y_np[split:]

        bs = self.batch_size or auto_batch_size(n_feat)
        self.model_ = self._build_net()
        self.model_ = fit_pytorch_model(
            self.model_,
            X_tr,
            y_tr,
            X_va,
            y_va,
            epochs=self.epochs,
            batch_size=bs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            patience=self.patience,
            dataset_cls=SequenceDataset,
        )
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        from f1_predictor.models.dl_utils import SequenceDataset, predict_pytorch

        if X.ndim == 3 and X.shape[1] > self.window_size:
            X = X[:, -self.window_size :, :]
        X_scaled = self._prepare(X)
        return predict_pytorch(self.model_, X_scaled, dataset_cls=SequenceDataset)


class SeqGRU_Shallow(_SeqBaseWrapper):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**{**{"hidden_dim": 64, "num_layers": 1}, **kwargs})

    def _build_net(self) -> nn.Module:
        return _SeqGRUNet(self.n_features, self.hidden_dim, self.num_layers, self.dropout)


class SeqGRU_Deep(_SeqBaseWrapper):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**{**{"hidden_dim": 128, "num_layers": 3}, **kwargs})

    def _build_net(self) -> nn.Module:
        return _SeqGRUNet(self.n_features, self.hidden_dim, self.num_layers, self.dropout)


class SeqGRU_Bidir(_SeqBaseWrapper):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**{**{"hidden_dim": 64, "num_layers": 2}, **kwargs})

    def _build_net(self) -> nn.Module:
        return _SeqGRUNet(
            self.n_features,
            self.hidden_dim,
            self.num_layers,
            self.dropout,
            bidirectional=True,
        )


class SeqLSTM_Shallow(_SeqBaseWrapper):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**{**{"hidden_dim": 64, "num_layers": 1}, **kwargs})

    def _build_net(self) -> nn.Module:
        return _SeqLSTMNet(self.n_features, self.hidden_dim, self.num_layers, self.dropout)


class SeqLSTM_Deep(_SeqBaseWrapper):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**{**{"hidden_dim": 128, "num_layers": 3}, **kwargs})

    def _build_net(self) -> nn.Module:
        return _SeqLSTMNet(self.n_features, self.hidden_dim, self.num_layers, self.dropout)


class SeqLSTM_Bidir(_SeqBaseWrapper):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**{**{"hidden_dim": 64, "num_layers": 2}, **kwargs})

    def _build_net(self) -> nn.Module:
        return _SeqLSTMNet(
            self.n_features,
            self.hidden_dim,
            self.num_layers,
            self.dropout,
            bidirectional=True,
        )


class SeqTCN(_SeqBaseWrapper):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**{**{"hidden_dim": 64, "num_layers": 4, "kernel_size": 3}, **kwargs})

    def _build_net(self) -> nn.Module:
        return _SeqTCNNet(
            self.n_features,
            self.hidden_dim,
            self.num_layers,
            self.kernel_size,
            self.dropout,
        )


class SeqTransformer(_SeqBaseWrapper):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**{**{"hidden_dim": 64, "num_layers": 3, "n_heads": 4}, **kwargs})

    def _build_net(self) -> nn.Module:
        return _SeqTransformerNet(
            self.n_features,
            self.hidden_dim,
            self.n_heads,
            self.num_layers,
            self.dropout,
        )


class SeqGRU_Attn(_SeqBaseWrapper):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**{**{"hidden_dim": 64, "num_layers": 2}, **kwargs})

    def _build_net(self) -> nn.Module:
        return _SeqGRUAttnNet(self.n_features, self.hidden_dim, self.num_layers, self.dropout)


class SeqCNN1D(_SeqBaseWrapper):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**{**{"hidden_dim": 64, "kernel_size": 3}, **kwargs})

    def _build_net(self) -> nn.Module:
        return _SeqCNN1DNet(self.n_features, self.hidden_dim, self.kernel_size, self.dropout)
