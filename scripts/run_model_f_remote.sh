#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Train Model F only on a GCE GPU VM, then upload results and self-delete.
#
# Usage:  bash scripts/run_model_f_remote.sh [--cpu]
# ---------------------------------------------------------------------------

PROJECT="jowin-personal-2026"
ZONE="${ZONE:-us-central1-a}"
BUCKET="f1-predictor-artifacts-jowin"
VM_NAME="f1-model-f-$(date +%s)"
STAGING_PREFIX="staging/training-run"
RESULTS_PREFIX="data"

USE_GPU=true
if [[ "${1:-}" == "--cpu" ]]; then
    USE_GPU=false
    MACHINE_TYPE="e2-standard-8"
    ACCELERATOR=""
    IMAGE_FAMILY="debian-12"
    IMAGE_PROJECT="debian-cloud"
    MAINT_POLICY=""
    echo ">>> CPU-only mode (e2-standard-8)"
else
    MACHINE_TYPE="g2-standard-8"
    ACCELERATOR="--accelerator=type=nvidia-l4,count=1"
    IMAGE_FAMILY="common-cu129-ubuntu-2204-nvidia-580"
    IMAGE_PROJECT="deeplearning-platform-release"
    MAINT_POLICY="--maintenance-policy=TERMINATE"
    echo ">>> GPU mode (g2-standard-8 + L4)"
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. Package repo and upload to GCS staging
# ---------------------------------------------------------------------------
echo ">>> Packaging repo..."
TARBALL="/tmp/f1-training-repo.tar.gz"
tar czf "$TARBALL" \
    --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
    --exclude='data/raw/*.parquet' --exclude='data/training' \
    --exclude='*.pkl' --exclude='.fastf1_cache' \
    --exclude='node_modules' --exclude='.mypy_cache' \
    -C "$REPO_ROOT" .

echo ">>> Uploading to gs://$BUCKET/$STAGING_PREFIX/"
gcloud storage cp "$TARBALL" "gs://$BUCKET/$STAGING_PREFIX/repo.tar.gz" --quiet

echo ">>> Staging complete."

# ---------------------------------------------------------------------------
# 2. Create the startup script (Model F only)
# ---------------------------------------------------------------------------
STARTUP_SCRIPT=$(cat <<'STARTUP_EOF'
#!/bin/bash
set -euo pipefail
exec > /var/log/f1-training.log 2>&1
echo "=== Model F training begin: $(date) ==="

BUCKET="__BUCKET__"
STAGING="__STAGING_PREFIX__"
RESULTS="__RESULTS_PREFIX__"
WORK="/opt/f1-training"

mkdir -p "$WORK" && cd "$WORK"

export HOME="${HOME:-/root}"
export GCE_METADATA_MTLS_MODE=none

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
export PATH="/root/.local/bin:$HOME/.local/bin:$PATH"

# Download repo
pip3 install gsutil 2>/dev/null || true
gsutil -q cp "gs://$BUCKET/$STAGING/repo.tar.gz" repo.tar.gz
tar xzf repo.tar.gz
rm repo.tar.gz

# Create required dirs
mkdir -p data/processed/simulation data/training data/raw/model

# Install Python + deps
uv python install 3.11
uv sync --frozen --group dev
uv pip install xgboost lightgbm optuna

# Check GPU
if nvidia-smi > /dev/null 2>&1; then
    echo "GPU detected — installing GPU-enabled packages"
    uv pip install xgboost --upgrade
fi

# Regenerate notebooks
uv run python scripts/make_training_notebooks.py

# Run Model F only
echo "=== Running Model F: $(date) ==="
uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=7200 \
    notebooks/05f_model_F_lap_simulation.ipynb \
    --output 05f_model_F_lap_simulation.ipynb 2>&1 || true

# Upload results to GCS
echo "=== Uploading results: $(date) ==="
gsutil -m -q cp data/training/model_F_*.parquet "gs://$BUCKET/$RESULTS/training/" 2>/dev/null || true
gsutil -m -q cp data/raw/model/Model_F_*.pkl "gs://$BUCKET/$RESULTS/raw/model/" 2>/dev/null || true
gsutil -q cp notebooks/05f_model_F_lap_simulation.ipynb "gs://$BUCKET/$RESULTS/notebooks/" 2>/dev/null || true
gsutil -m -q cp data/processed/simulation/*.parquet "gs://$BUCKET/$RESULTS/processed/simulation/" 2>/dev/null || true

# Signal completion
echo "DONE" | gsutil -q cp - "gs://$BUCKET/$STAGING/MODEL_F_DONE"

echo "=== Model F training complete: $(date) ==="

# Self-delete
VM_NAME=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/name)
VM_ZONE=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone | rev | cut -d'/' -f1 | rev)
gcloud compute instances delete "$VM_NAME" --zone="$VM_ZONE" --quiet &
STARTUP_EOF
)

# Substitute variables
STARTUP_SCRIPT="${STARTUP_SCRIPT//__BUCKET__/$BUCKET}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__STAGING_PREFIX__/$STAGING_PREFIX}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__RESULTS_PREFIX__/$RESULTS_PREFIX}"

STARTUP_FILE="/tmp/f1-model-f-startup.sh"
echo "$STARTUP_SCRIPT" > "$STARTUP_FILE"

# ---------------------------------------------------------------------------
# 3. Create the VM
# ---------------------------------------------------------------------------
echo ">>> Creating VM: $VM_NAME in $ZONE..."

gcloud compute instances create "$VM_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    $ACCELERATOR \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size=50GB \
    --scopes=storage-full,compute-rw \
    --metadata-from-file=startup-script="$STARTUP_FILE" \
    $MAINT_POLICY \
    --no-restart-on-failure

echo ""
echo "============================================================"
echo "  VM '$VM_NAME' created — training Model F only"
echo "============================================================"
echo ""
echo "  Monitor:"
echo "    gcloud compute ssh $VM_NAME --zone=$ZONE --command='tail -f /var/log/f1-training.log'"
echo ""
echo "  Check if done:"
echo "    gsutil stat gs://$BUCKET/$STAGING_PREFIX/MODEL_F_DONE"
echo ""
echo "  Manual cleanup:"
echo "    gcloud compute instances delete $VM_NAME --zone=$ZONE --quiet"
echo ""
echo "  Estimated time: ~30-60 min"
echo "============================================================"
