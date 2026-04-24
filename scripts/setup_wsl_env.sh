#!/usr/bin/env bash
# One-time WSL2 environment setup for F1 Race Predictor GPU training.
# Supports AMD ROCm and NVIDIA CUDA. Idempotent — safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "  F1 Race Predictor — WSL2 Environment Setup"
echo "============================================================"
echo ""

# ---------------------------------------------------------------
# 1. Validate WSL2
# ---------------------------------------------------------------
if uname -r | grep -qi microsoft; then
    echo "[OK] Running on WSL2"
else
    echo "[WARN] Not detected as WSL2 (uname -r: $(uname -r))"
    echo "       Continuing anyway — this script works on native Linux too."
fi

# ---------------------------------------------------------------
# 2. Detect GPU
# ---------------------------------------------------------------
GPU_BACKEND="cpu"
GPU_NAME="none"

if command -v rocm-smi &>/dev/null; then
    GPU_BACKEND="rocm"
    GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -v "^=" | head -1 || echo "AMD GPU")
    ROCM_VER=$(rocm-smi --showversion 2>/dev/null | grep -oP '\d+\.\d+' | head -1 || echo "6.2")
    echo "[OK] AMD GPU detected: $GPU_NAME"
    echo "     ROCm version: $ROCM_VER"
elif command -v nvidia-smi &>/dev/null; then
    GPU_BACKEND="cuda"
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "NVIDIA GPU")
    echo "[OK] NVIDIA GPU detected: $GPU_NAME"
else
    echo "[WARN] No GPU detected. DL models will be skipped during training."
fi

# ---------------------------------------------------------------
# 3. Install uv (if not present)
# ---------------------------------------------------------------
if command -v uv &>/dev/null; then
    echo "[OK] uv already installed: $(uv --version)"
else
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "[OK] uv installed: $(uv --version)"
fi

# ---------------------------------------------------------------
# 4. Install Python + project deps
# ---------------------------------------------------------------
echo ""
echo ">>> Installing project dependencies..."
uv sync --frozen --group dev --extra training --extra notebooks --extra dl

# ---------------------------------------------------------------
# 5. Install PyTorch with correct GPU backend
# ---------------------------------------------------------------
echo ""
echo ">>> Installing PyTorch..."
# --reinstall so we override any torch that uv sync pulled in (e.g. CUDA build from PyPI).
if [ "$GPU_BACKEND" = "rocm" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/rocm${ROCM_VER:-6.2}"
    echo "    Using ROCm index: $TORCH_INDEX"
    uv pip install --reinstall torch --index-url "$TORCH_INDEX"
elif [ "$GPU_BACKEND" = "cuda" ]; then
    echo "    Using default PyPI (CUDA)"
    uv pip install --reinstall torch
else
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    echo "    Using CPU-only index: $TORCH_INDEX"
    uv pip install --reinstall torch --index-url "$TORCH_INDEX"
fi

# ---------------------------------------------------------------
# 6. Install LightGBM with OpenCL (AMD only)
# ---------------------------------------------------------------
if [ "$GPU_BACKEND" = "rocm" ]; then
    echo ""
    echo ">>> Installing LightGBM with OpenCL GPU support..."
    if ! dpkg -l | grep -q ocl-icd-opencl-dev; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq ocl-icd-opencl-dev
    fi
    uv pip install --force-reinstall lightgbm --config-settings=cmake.define.USE_GPU=ON || {
        echo "[WARN] LightGBM GPU build failed. Using CPU-only LightGBM."
        echo "       Tree models will still work, just slower on LightGBM."
    }
fi

# ---------------------------------------------------------------
# 7. Validate GCS credentials
# ---------------------------------------------------------------
echo ""
echo ">>> Checking GCS credentials..."
if command -v gcloud &>/dev/null; then
    ACTIVE_ACCOUNT=$(gcloud auth list --filter="status:ACTIVE" --format="value(account)" 2>/dev/null | head -1)
    if [ -n "$ACTIVE_ACCOUNT" ]; then
        echo "[OK] GCS authenticated as: $ACTIVE_ACCOUNT"
    else
        echo "[ERROR] gcloud installed but no active account."
        echo "        Run: gcloud auth login"
        echo "        Then: gcloud auth application-default login"
    fi
else
    echo "[WARN] gcloud CLI not installed."
    echo "       Install: https://cloud.google.com/sdk/docs/install"
    echo "       Or set GOOGLE_APPLICATION_CREDENTIALS to a service account key."
fi

# ---------------------------------------------------------------
# 8. Validate installation
# ---------------------------------------------------------------
echo ""
echo ">>> Validating installation..."
# WSL+ROCm: preload WSL-aware libhsa so torch.cuda sees the GPU during validation.
if [ "$GPU_BACKEND" = "rocm" ] && grep -qi microsoft /proc/version 2>/dev/null; then
    for _hsa in /opt/rocm-*/lib/libhsa-runtime64.so.1; do
        [ -e "$_hsa" ] && export LD_PRELOAD="$_hsa${LD_PRELOAD:+:$LD_PRELOAD}" && break
    done
    unset _hsa
fi
uv run python -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        props = torch.cuda.get_device_properties(0)
        print(f'  VRAM: {props.total_memory / 1024**3:.1f} GB')
    else:
        print('  GPU: not available (CPU only)')
except ImportError:
    print('PyTorch: NOT INSTALLED')

try:
    import xgboost
    print(f'XGBoost: {xgboost.__version__}')
except ImportError:
    print('XGBoost: NOT INSTALLED')

try:
    import lightgbm
    print(f'LightGBM: {lightgbm.__version__}')
except ImportError:
    print('LightGBM: NOT INSTALLED')

try:
    import rtdl_revisiting_models
    print(f'rtdl: available')
except ImportError:
    print('rtdl: NOT INSTALLED')

try:
    from f1_predictor.models.gpu import detect_gpu_backend
    backend, name = detect_gpu_backend()
    print(f'GPU detection: backend={backend}, name={name}')
except Exception as e:
    print(f'GPU detection: FAILED ({e})')
"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  GPU backend: $GPU_BACKEND ($GPU_NAME)"
echo ""
echo "  To train all models:"
echo "    bash scripts/run_training_wsl.sh"
echo "============================================================"
