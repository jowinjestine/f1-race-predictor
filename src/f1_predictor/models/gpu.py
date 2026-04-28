"""Unified GPU backend detection for NVIDIA CUDA and AMD ROCm."""

from __future__ import annotations

import shutil
import subprocess


def _rocminfo_gpu_name() -> str | None:
    """Parse `rocminfo` for the first GPU agent's Marketing Name. Works on WSL."""
    if not shutil.which("rocminfo"):
        return None
    try:
        result = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    # In each Agent block, "Marketing Name:" appears before "Device Type:".
    # Hold the last seen name and commit it when we confirm Device Type is GPU.
    pending_name: str | None = None
    for raw in result.stdout.splitlines():
        line = raw.strip()
        if line.startswith("Agent "):
            pending_name = None
        elif line.startswith("Marketing Name:"):
            pending_name = line.split(":", 1)[1].strip() or None
        elif line.startswith("Device Type:") and "GPU" in line and pending_name:
            return pending_name
    return None


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
            # rocm-smi exits 0 on WSL even when it fails (errors go to stderr);
            # treat empty/unparseable stdout as "no answer" and try rocminfo next.
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if line and not line.startswith("=") and "GPU" not in line.upper().split()[0:1]:
                        return "rocm", line
        except (subprocess.TimeoutExpired, OSError):
            pass

    # WSL path: rocm-smi can't read /sys/module/amdgpu; rocminfo uses /dev/dxg.
    rocm_name = _rocminfo_gpu_name()
    if rocm_name:
        return "rocm", rocm_name

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
            summary["vram_gb"] = round(props.total_memory / 1024**3, 1)
            summary["torch_gpu_name"] = props.name
    except (ImportError, RuntimeError):
        pass
    return summary
