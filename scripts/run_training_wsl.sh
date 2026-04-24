#!/usr/bin/env bash
# Run the full F1 Race Predictor training pipeline locally on WSL2.
# Supports AMD ROCm and NVIDIA CUDA GPUs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# On WSL + ROCm, PyTorch's bundled libhsa-runtime64.so is the native-Linux build
# and cannot talk to /dev/dxg. Preload the WSL-aware system lib from the
# hsa-runtime-rocr4wsl-amdgpu package so torch.cuda sees the GPU.
if grep -qi microsoft /proc/version 2>/dev/null; then
    for _hsa in /opt/rocm-*/lib/libhsa-runtime64.so.1; do
        [ -e "$_hsa" ] && export LD_PRELOAD="$_hsa${LD_PRELOAD:+:$LD_PRELOAD}" && break
    done
    unset _hsa
fi

TIMEOUT_AB=7200   # 2 hours for lap-level models
TIMEOUT_CD=3600   # 1 hour for race-level / stacking
TIMEOUT_CMP=1800  # 30 min for comparison

echo "============================================================"
echo "  F1 Race Predictor — Local GPU Training Pipeline"
echo "============================================================"
echo "  Started: $(date)"
echo ""

# ---------------------------------------------------------------
# 1. Validate environment
# ---------------------------------------------------------------
echo ">>> [1/8] Validating environment..."
uv run python --version

# GPU check
uv run python -c "
from f1_predictor.models.gpu import detect_gpu_backend, get_torch_device
backend, name = detect_gpu_backend()
torch_dev = get_torch_device()
print(f'GPU backend: {backend} ({name})')
print(f'PyTorch device: {torch_dev}')
if backend == 'cpu':
    print('WARNING: No GPU detected. Tree models will run on CPU, DL models will be skipped.')
"

# GCS check (required)
echo ""
if command -v gcloud &>/dev/null; then
    ACTIVE=$(gcloud auth list --filter="status:ACTIVE" --format="value(account)" 2>/dev/null | head -1)
    if [ -n "$ACTIVE" ]; then
        echo "[OK] GCS authenticated as: $ACTIVE"
    else
        echo "[ERROR] No active GCS credentials. Run: gcloud auth login"
        exit 1
    fi
else
    echo "[WARN] gcloud not installed. GCS upload will use GOOGLE_APPLICATION_CREDENTIALS if set."
fi

# ---------------------------------------------------------------
# 2. Generate notebooks
# ---------------------------------------------------------------
echo ""
echo ">>> [2/8] Generating training notebooks..."
uv run python scripts/make_training_notebooks.py

# ---------------------------------------------------------------
# 3. Train Model A
# ---------------------------------------------------------------
echo ""
echo ">>> [3/8] Training Model A (lap + tyre)..."
echo "    Timeout: ${TIMEOUT_AB}s"
uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=$TIMEOUT_AB \
    notebooks/05a_model_A_training.ipynb \
    --output 05a_model_A_training.ipynb
echo "    Model A complete."

# ---------------------------------------------------------------
# 4. Train Model B
# ---------------------------------------------------------------
echo ""
echo ">>> [4/8] Training Model B (lap, no tyre)..."
echo "    Timeout: ${TIMEOUT_AB}s"
uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=$TIMEOUT_AB \
    notebooks/05b_model_B_training.ipynb \
    --output 05b_model_B_training.ipynb
echo "    Model B complete."

# ---------------------------------------------------------------
# 5. Train Model C
# ---------------------------------------------------------------
echo ""
echo ">>> [5/8] Training Model C (pre-race features)..."
echo "    Timeout: ${TIMEOUT_CD}s"
uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=$TIMEOUT_CD \
    notebooks/05c_model_C_training.ipynb \
    --output 05c_model_C_training.ipynb
echo "    Model C complete."

# ---------------------------------------------------------------
# 6. Train Model D (all-combinations stacking)
# ---------------------------------------------------------------
echo ""
echo ">>> [6/8] Training Model D (all-combinations stacking)..."
echo "    Timeout: ${TIMEOUT_CD}s"
uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=$TIMEOUT_CD \
    notebooks/05d_model_D_stacking.ipynb \
    --output 05d_model_D_stacking.ipynb
echo "    Model D complete."

# ---------------------------------------------------------------
# 7. Model comparison
# ---------------------------------------------------------------
echo ""
echo ">>> [7/8] Running model comparison..."
echo "    Timeout: ${TIMEOUT_CMP}s"
uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=$TIMEOUT_CMP \
    notebooks/06_model_comparison.ipynb \
    --output 06_model_comparison.ipynb
echo "    Comparison complete."

# ---------------------------------------------------------------
# 8. Upload to GCS (required)
# ---------------------------------------------------------------
echo ""
echo ">>> [8/8] Uploading artifacts to GCS..."
BUCKET="f1-predictor-artifacts-jowin"

# Predictions
gsutil -m -q cp data/training/model_*.parquet "gs://$BUCKET/data/training/" 2>/dev/null && \
    echo "    Uploaded prediction parquets." || echo "    [WARN] Prediction upload failed."

# Model pickles
gsutil -m -q cp data/raw/model/Model_*.pkl "gs://$BUCKET/data/raw/model/" 2>/dev/null && \
    echo "    Uploaded model pickles." || echo "    [WARN] Pickle upload failed."

# Executed notebooks
gsutil -m -q cp notebooks/05*.ipynb notebooks/06*.ipynb "gs://$BUCKET/data/notebooks/" 2>/dev/null && \
    echo "    Uploaded executed notebooks." || echo "    [WARN] Notebook upload failed."

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Training Pipeline Complete!"
echo "  Finished: $(date)"
echo ""
echo "  Artifacts:"
N_PKL=$(ls data/raw/model/Model_*.pkl 2>/dev/null | wc -l)
N_PQ=$(ls data/training/model_*.parquet 2>/dev/null | wc -l)
echo "    Model pickles: $N_PKL (in data/raw/model/)"
echo "    Prediction parquets: $N_PQ (in data/training/)"
echo "    Notebooks: notebooks/05*.ipynb, notebooks/06*.ipynb"
echo ""
echo "  GCS bucket: gs://$BUCKET/"
echo "============================================================"
