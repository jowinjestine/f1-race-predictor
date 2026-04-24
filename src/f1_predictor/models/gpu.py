"""Unified GPU backend detection for NVIDIA CUDA and AMD ROCm."""

from __future__ import annotations

import shutil
import subprocess


def detect_gpu_backend() -> tuple[str, str | None]:
    """Detect GPU vendor. Returns (backend, device_name).

    Checks ROCm first (AMD), then NVIDIA. Returns ('cpu', None) if neither found.
    """
    if shutil.which("rocm-smi"):
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if line and not line.startswith("=") and "GPU" not in line.upper().split()[0:1]:
                        return "rocm", line
                return "rocm", "AMD GPU"
        except (subprocess.TimeoutExpired, OSError):
            pass

    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return "cuda", result.stdout.strip().split("\n")[0]
        except (subprocess.TimeoutExpired, OSError):
            pass

    return "cpu", None


def get_torch_device() -> str:
    """Return 'cuda' if PyTorch GPU is available (works for both CUDA and ROCm HIP), else 'cpu'."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_xgboost_device(backend: str) -> dict[str, str]:
    """Return XGBoost device kwargs. CUDA only — XGBoost GPU requires NVIDIA."""
    if backend == "cuda":
        return {"device": "cuda"}
    return {}


def get_lightgbm_device(backend: str) -> dict[str, str]:
    """Return LightGBM device kwargs. Works on both CUDA and ROCm (OpenCL backend)."""
    if backend in ("cuda", "rocm"):
        return {"device": "gpu"}
    return {}


def get_device_summary() -> dict[str, object]:
    """Return a summary dict for notebook printing."""
    backend, name = detect_gpu_backend()
    torch_device = get_torch_device()
    summary: dict[str, object] = {
        "backend": backend,
        "gpu_name": name,
        "torch_device": torch_device,
        "xgboost_device": get_xgboost_device(backend),
        "lightgbm_device": get_lightgbm_device(backend),
    }
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            summary["vram_gb"] = round(props.total_mem / 1024**3, 1)
            summary["torch_gpu_name"] = props.name
    except (ImportError, RuntimeError):
        pass
    return summary
