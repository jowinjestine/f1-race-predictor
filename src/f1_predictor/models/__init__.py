"""Models package — GPU detection and deep learning architectures."""

from f1_predictor.models.gpu import (
    detect_gpu_backend,
    get_device_summary,
    get_lightgbm_device,
    get_torch_device,
    get_xgboost_device,
)

__all__ = [
    "detect_gpu_backend",
    "get_device_summary",
    "get_lightgbm_device",
    "get_torch_device",
    "get_xgboost_device",
]
