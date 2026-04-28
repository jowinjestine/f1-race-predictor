# ruff: noqa: N801, N803, N806  — sklearn API uses uppercase X; underscore class names are candidate IDs
"""Quantile and uncertainty-aware model architectures for Model I.

All models implement predict() (returns q50 median) and predict_quantiles()
(returns 5 quantiles: q10, q25, q50, q75, q90).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import _MLPNet from architectures for DeepEnsemble
try:
    from f1_predictor.models.architectures import _MLPNet
except ImportError:
    _MLPNet = None  # type: ignore[assignment, misc]

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


# ---------------------------------------------------------------------------
# Mixin for quantile models
# ---------------------------------------------------------------------------


class QuantileRegressorMixin:
    """Adds predict_quantiles() to any model that outputs 5 quantiles."""

    def predict_quantiles(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Raw PyTorch modules (multi-head quantile output)
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class _MLPQuantileNet(nn.Module):  # type: ignore[misc]
        def __init__(
            self,
            input_dim: int,
            hidden1: int,
            hidden2: int,
            dropout: float,
            n_quantiles: int = 5,
        ) -> None:
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
                nn.Linear(hidden2, n_quantiles),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class _GRUQuantileNet(nn.Module):  # type: ignore[misc]
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
            n_quantiles: int = 5,
        ) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden_dim * 2, n_quantiles)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            out, _ = self.gru(x)
            return self.head(out[:, -1, :])

    class _FTTQuantileNet(nn.Module):  # type: ignore[misc]
        def __init__(
            self,
            n_features: int,
            d_token: int,
            n_blocks: int,
            dropout: float,
            n_quantiles: int = 5,
        ) -> None:
            super().__init__()
            try:
                from rtdl_revisiting_models import FTTransformer

                self.ft = FTTransformer(
                    n_cont_features=n_features,
                    cat_cardinalities=[],
                    d_out=n_quantiles,
                    n_blocks=n_blocks,
                    d_block=d_token * 4,
                    attention_n_heads=max(1, d_token // 16),
                    attention_dropout=dropout,
                    ffn_d_hidden=None,
                    ffn_d_hidden_multiplier=4 / 3,
                    ffn_dropout=dropout,
                    residual_dropout=0.0,
                    linformer_compression_ratio=None,
                    linformer_sharing_policy=None,
                )
            except (ImportError, TypeError, ValueError):
                self.ft = nn.Sequential(
                    nn.Linear(n_features, d_token * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_token * 2, d_token),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_token, n_quantiles),
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.ft(x)

    class _MDNNet(nn.Module):  # type: ignore[misc]
        def __init__(
            self,
            input_dim: int,
            hidden1: int,
            hidden2: int,
            dropout: float,
            n_components: int = 3,
        ) -> None:
            super().__init__()
            self.n_components = n_components
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.BatchNorm1d(hidden1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden1, hidden2),
                nn.BatchNorm1d(hidden2),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            # Output: pi_logits, mu, log_sigma (3 * K)
            self.mdn_head = nn.Linear(hidden2, 3 * n_components)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.backbone(x)
            return self.mdn_head(h)

    class _GRUMDNNet(nn.Module):  # type: ignore[misc]
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
            n_components: int = 3,
        ) -> None:
            super().__init__()
            self.n_components = n_components
            self.gru = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.mdn_head = nn.Linear(
                hidden_dim * 2,
                3 * n_components,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            out, _ = self.gru(x)
            return self.mdn_head(out[:, -1, :])


# ---------------------------------------------------------------------------
# Sklearn-compatible wrappers
# ---------------------------------------------------------------------------


class _QuantileBaseWrapper(
    BaseEstimator,
    RegressorMixin,
    QuantileRegressorMixin,  # type: ignore[misc]
):
    """Base class for PyTorch quantile model wrappers."""

    def __init__(
        self,
        input_dim: int = 25,
        hidden1: int = 128,
        hidden2: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        d_token: int = 64,
        n_blocks: int = 3,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 0,
        epochs: int = 80,
        patience: int = 15,
    ) -> None:
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

    def _build_net(self) -> nn.Module:
        raise NotImplementedError

    def _prepare(self, X: Any) -> NDArray[np.float64]:
        X_arr = np.asarray(X, dtype=np.float64)
        return self.scaler_.transform(self.imputer_.transform(X_arr)).astype(np.float32)

    def fit(
        self,
        X: Any,
        y: Any,
    ) -> _QuantileBaseWrapper:
        from f1_predictor.models.dl_utils import (
            auto_batch_size,
            fit_pytorch_model_quantile,
        )

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        self.imputer_ = SimpleImputer(strategy="median")
        self.scaler_ = StandardScaler()
        X_clean = self.scaler_.fit_transform(
            self.imputer_.fit_transform(X_arr),
        )
        self.input_dim = X_clean.shape[1]

        bs = self.batch_size or auto_batch_size(X_clean.shape[1])
        split = int(len(X_clean) * 0.8)
        X_tr, X_va = X_clean[:split], X_clean[split:]
        y_tr, y_va = y_arr[:split], y_arr[split:]

        self.model_ = self._build_net()
        self.model_ = fit_pytorch_model_quantile(
            self.model_,
            X_tr,
            y_tr,
            X_va,
            y_va,
            quantiles=QUANTILES,
            epochs=self.epochs,
            batch_size=bs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            patience=self.patience,
        )
        return self

    def predict(self, X: Any) -> NDArray[np.float64]:
        q = self.predict_quantiles(X)
        return q[:, 2]  # q50

    def predict_quantiles(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        from f1_predictor.models.dl_utils import predict_pytorch_multi

        X_clean = self._prepare(X)
        raw = predict_pytorch_multi(self.model_, X_clean)
        # Enforce monotonicity (sort quantiles)
        return np.sort(raw, axis=1)


class _MDNBaseWrapper(
    BaseEstimator,
    RegressorMixin,
    QuantileRegressorMixin,  # type: ignore[misc]
):
    """Base class for Mixture Density Network wrappers."""

    def __init__(
        self,
        input_dim: int = 25,
        hidden1: int = 128,
        hidden2: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 0,
        epochs: int = 80,
        patience: int = 15,
        n_components: int = 3,
    ) -> None:
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.n_components = n_components

    def _build_net(self) -> nn.Module:
        raise NotImplementedError

    def _prepare(self, X: Any) -> NDArray[np.float64]:
        X_arr = np.asarray(X, dtype=np.float64)
        return self.scaler_.transform(self.imputer_.transform(X_arr)).astype(np.float32)

    def fit(self, X: Any, y: Any) -> _MDNBaseWrapper:
        from f1_predictor.models.dl_utils import (
            auto_batch_size,
            fit_pytorch_model_mdn,
        )

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        self.imputer_ = SimpleImputer(strategy="median")
        self.scaler_ = StandardScaler()
        X_clean = self.scaler_.fit_transform(
            self.imputer_.fit_transform(X_arr),
        )
        self.input_dim = X_clean.shape[1]

        bs = self.batch_size or auto_batch_size(X_clean.shape[1])
        split = int(len(X_clean) * 0.8)
        X_tr, X_va = X_clean[:split], X_clean[split:]
        y_tr, y_va = y_arr[:split], y_arr[split:]

        self.model_ = self._build_net()
        self.model_ = fit_pytorch_model_mdn(
            self.model_,
            X_tr,
            y_tr,
            X_va,
            y_va,
            n_components=self.n_components,
            epochs=self.epochs,
            batch_size=bs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            patience=self.patience,
        )
        return self

    def predict(self, X: Any) -> NDArray[np.float64]:
        q = self.predict_quantiles(X)
        return q[:, 2]  # q50

    def predict_quantiles(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        from f1_predictor.models.dl_utils import predict_pytorch_multi

        X_clean = self._prepare(X)
        params = predict_pytorch_multi(self.model_, X_clean)

        K = self.n_components
        pi_logits = params[:, :K]
        mu = params[:, K : 2 * K]
        log_sigma = params[:, 2 * K : 3 * K]

        # Softmax for weights
        pi = np.exp(pi_logits - pi_logits.max(axis=1, keepdims=True))
        pi = pi / pi.sum(axis=1, keepdims=True)
        sigma = np.exp(log_sigma).clip(min=1e-6)

        # Sample quantiles from mixture via inverse CDF approximation
        n = len(params)
        quantile_vals = np.zeros((n, len(QUANTILES)))
        for qi, q in enumerate(QUANTILES):
            # Weighted quantile: for each component, compute
            # mu + sigma * Phi^{-1}(q), then take weighted mean
            from scipy.stats import norm

            z = norm.ppf(q)
            component_vals = mu + sigma * z  # (n, K)
            quantile_vals[:, qi] = (pi * component_vals).sum(axis=1)

        return np.sort(quantile_vals, axis=1)


# ---------------------------------------------------------------------------
# Tree-based quantile models (one model per quantile)
# ---------------------------------------------------------------------------


class _TreeQuantileWrapper(
    BaseEstimator,
    RegressorMixin,
    QuantileRegressorMixin,  # type: ignore[misc]
):
    """Trains one tree model per quantile level."""

    def __init__(self, **kwargs: Any) -> None:
        self._tree_kwargs = kwargs

    def _make_tree(
        self,
        quantile: float,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError

    def fit(self, X: Any, y: Any) -> _TreeQuantileWrapper:
        self.models_: dict[float, Any] = {}
        for q in QUANTILES:
            m = self._make_tree(q, **self._tree_kwargs)
            m.fit(X, y)
            self.models_[q] = m
        return self

    def predict(self, X: Any) -> NDArray[np.float64]:
        return self.models_[0.5].predict(X)

    def predict_quantiles(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        cols = []
        for q in QUANTILES:
            cols.append(self.models_[q].predict(X))
        raw = np.column_stack(cols)
        return np.sort(raw, axis=1)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return dict(self._tree_kwargs)

    def set_params(self, **params: Any) -> _TreeQuantileWrapper:
        self._tree_kwargs.update(params)
        return self


class LightGBM_Quantile(_TreeQuantileWrapper):
    def _make_tree(
        self,
        quantile: float,
        **kwargs: Any,
    ) -> Any:
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            **kwargs,
        )


class XGBoost_Quantile(_TreeQuantileWrapper):
    def _make_tree(
        self,
        quantile: float,
        **kwargs: Any,
    ) -> Any:
        import xgboost as xgb

        return xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=quantile,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# DL quantile models
# ---------------------------------------------------------------------------


class MLP_MultiQuantile(_QuantileBaseWrapper):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(
            **{
                **{"hidden1": 128, "hidden2": 64},
                **kwargs,
            }
        )

    def _build_net(self) -> nn.Module:
        return _MLPQuantileNet(
            self.input_dim,
            self.hidden1,
            self.hidden2,
            self.dropout,
        )


class GRU_MultiQuantile(_QuantileBaseWrapper):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(
            **{
                **{"hidden_dim": 64, "num_layers": 2},
                **kwargs,
            }
        )

    def _build_net(self) -> nn.Module:
        return _GRUQuantileNet(
            self.input_dim,
            self.hidden_dim,
            self.num_layers,
            self.dropout,
        )


class FTTransformer_Quantile(_QuantileBaseWrapper):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(
            **{
                **{"d_token": 64, "n_blocks": 3},
                **kwargs,
            }
        )

    def _build_net(self) -> nn.Module:
        return _FTTQuantileNet(
            self.input_dim,
            self.d_token,
            self.n_blocks,
            self.dropout,
        )


# ---------------------------------------------------------------------------
# MDN models
# ---------------------------------------------------------------------------


class MDN_MLP(_MDNBaseWrapper):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(
            **{
                **{"hidden1": 128, "hidden2": 64, "n_components": 3},
                **kwargs,
            }
        )

    def _build_net(self) -> nn.Module:
        return _MDNNet(
            self.input_dim,
            self.hidden1,
            self.hidden2,
            self.dropout,
            self.n_components,
        )


class MDN_GRU(_MDNBaseWrapper):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(
            **{
                **{"hidden_dim": 64, "num_layers": 2, "n_components": 3},
                **kwargs,
            }
        )

    def _build_net(self) -> nn.Module:
        return _GRUMDNNet(
            self.input_dim,
            self.hidden_dim,
            self.num_layers,
            self.dropout,
            self.n_components,
        )


# ---------------------------------------------------------------------------
# Deep Ensemble (5 independent MLPs)
# ---------------------------------------------------------------------------


class DeepEnsemble(
    BaseEstimator,
    RegressorMixin,
    QuantileRegressorMixin,  # type: ignore[misc]
):
    """Ensemble of N independently trained MLPs for uncertainty."""

    def __init__(
        self,
        input_dim: int = 25,
        hidden1: int = 128,
        hidden2: int = 64,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 0,
        epochs: int = 80,
        patience: int = 15,
        n_members: int = 5,
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
        self.n_members = n_members

    def fit(self, X: Any, y: Any) -> DeepEnsemble:
        from f1_predictor.models.dl_utils import (
            auto_batch_size,
            fit_pytorch_model,
        )

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        self.imputer_ = SimpleImputer(strategy="median")
        self.scaler_ = StandardScaler()
        X_clean = self.scaler_.fit_transform(
            self.imputer_.fit_transform(X_arr),
        )
        self.input_dim = X_clean.shape[1]

        bs = self.batch_size or auto_batch_size(X_clean.shape[1])
        split = int(len(X_clean) * 0.8)
        X_tr, X_va = X_clean[:split], X_clean[split:]
        y_tr, y_va = y_arr[:split], y_arr[split:]

        self.models_: list[nn.Module] = []
        for i in range(self.n_members):
            # Bootstrap sample for diversity
            rng = np.random.RandomState(42 + i)
            idx = rng.choice(len(X_tr), len(X_tr), replace=True)
            net = _MLPNet(
                self.input_dim,
                self.hidden1,
                self.hidden2,
                self.dropout,
            )
            net = fit_pytorch_model(
                net,
                X_tr[idx],
                y_tr[idx],
                X_va,
                y_va,
                epochs=self.epochs,
                batch_size=bs,
                lr=self.lr,
                weight_decay=self.weight_decay,
                patience=self.patience,
            )
            self.models_.append(net)
        return self

    def predict(self, X: Any) -> NDArray[np.float64]:
        all_preds = self._predict_all(X)
        return np.mean(all_preds, axis=0)

    def predict_quantiles(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        all_preds = self._predict_all(X)  # (n_members, n_samples)
        quantile_vals = np.percentile(
            all_preds,
            [q * 100 for q in QUANTILES],
            axis=0,
        )
        return quantile_vals.T  # (n_samples, n_quantiles)

    def _predict_all(
        self,
        X: Any,
    ) -> NDArray[np.float64]:
        from f1_predictor.models.dl_utils import predict_pytorch

        X_arr = np.asarray(X, dtype=np.float64)
        X_clean = self.scaler_.transform(
            self.imputer_.transform(X_arr),
        )
        preds = []
        for net in self.models_:
            preds.append(predict_pytorch(net, X_clean))
        return np.array(preds)
