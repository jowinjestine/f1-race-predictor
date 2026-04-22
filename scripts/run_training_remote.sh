#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Run Model A & B training on a GCP VM with T4 GPU, then pull results back.
# Model C is already trained locally. Model D runs locally after A+B finish.
#
# Usage:
#   bash scripts/run_training_remote.sh [--cpu]
#
# Prerequisites:
#   - gcloud CLI authenticated (`gcloud auth login`)
#   - Project set: gcloud config set project jowin-personal-2026
#   - GPU quota in the target zone (or use --cpu flag)
# ---------------------------------------------------------------------------

PROJECT="jowin-personal-2026"
ZONE="us-central1-a"
BUCKET="f1-predictor-artifacts-jowin"
VM_NAME="f1-training-$(date +%s)"
STAGING_PREFIX="staging/training-run"
RESULTS_PREFIX="data"

# Default: GPU. Use --cpu flag to skip GPU.
USE_GPU=true
MACHINE_TYPE="n1-standard-8"
ACCELERATOR="--accelerator=type=nvidia-tesla-t4,count=1"
IMAGE_FAMILY="common-cu124-debian-11"
IMAGE_PROJECT="deeplearning-platform-release"

if [[ "${1:-}" == "--cpu" ]]; then
    USE_GPU=false
    MACHINE_TYPE="c2-standard-16"
    ACCELERATOR=""
    IMAGE_FAMILY="debian-12"
    IMAGE_PROJECT="debian-cloud"
    echo ">>> CPU-only mode (c2-standard-16, 16 vCPUs)"
else
    echo ">>> GPU mode (n1-standard-8 + T4)"
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
gsutil -q cp "$TARBALL" "gs://$BUCKET/$STAGING_PREFIX/repo.tar.gz"

# Upload data files (notebooks load from GCS, but also provide local fallback)
gsutil -q cp \
    data/processed/lap_tyre/features_laps_tyre.parquet \
    "gs://$BUCKET/$STAGING_PREFIX/features_laps_tyre.parquet"
gsutil -q cp \
    data/processed/lap_notyre/features_laps_notyre.parquet \
    "gs://$BUCKET/$STAGING_PREFIX/features_laps_notyre.parquet"

echo ">>> Staging complete."

# ---------------------------------------------------------------------------
# 2. Create the startup script
# ---------------------------------------------------------------------------
STARTUP_SCRIPT=$(cat <<'STARTUP_EOF'
#!/bin/bash
set -euo pipefail
exec > /var/log/f1-training.log 2>&1
echo "=== Startup script begin: $(date) ==="

BUCKET="__BUCKET__"
STAGING="__STAGING_PREFIX__"
RESULTS="__RESULTS_PREFIX__"
WORK="/opt/f1-training"

mkdir -p "$WORK" && cd "$WORK"

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Download repo
gsutil -q cp "gs://$BUCKET/$STAGING/repo.tar.gz" repo.tar.gz
tar xzf repo.tar.gz
rm repo.tar.gz

# Place data files locally as fallback
mkdir -p data/processed/lap_tyre data/processed/lap_notyre data/training data/raw/model
gsutil -q cp "gs://$BUCKET/$STAGING/features_laps_tyre.parquet" \
    data/processed/lap_tyre/features_laps_tyre.parquet
gsutil -q cp "gs://$BUCKET/$STAGING/features_laps_notyre.parquet" \
    data/processed/lap_notyre/features_laps_notyre.parquet

# Install Python + deps
uv python install 3.11
uv sync --frozen --group dev
uv pip install xgboost lightgbm optuna

# Check GPU
if nvidia-smi > /dev/null 2>&1; then
    echo "GPU detected — installing GPU-enabled XGBoost/LightGBM"
    uv pip install xgboost --upgrade
fi

# Regenerate notebooks (picks up GPU detection at runtime)
uv run python scripts/make_training_notebooks.py

# Run Model A
echo "=== Running Model A: $(date) ==="
uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=7200 \
    notebooks/05a_model_A_training.ipynb \
    --output 05a_model_A_training.ipynb 2>&1 || true

# Run Model B
echo "=== Running Model B: $(date) ==="
uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=7200 \
    notebooks/05b_model_B_training.ipynb \
    --output 05b_model_B_training.ipynb 2>&1 || true

# Upload results to GCS
echo "=== Uploading results: $(date) ==="
gsutil -m -q cp data/training/model_A_*.parquet "gs://$BUCKET/$RESULTS/training/"
gsutil -m -q cp data/training/model_B_*.parquet "gs://$BUCKET/$RESULTS/training/"
gsutil -m -q cp data/raw/model/Model_A_*.pkl "gs://$BUCKET/$RESULTS/raw/model/"
gsutil -m -q cp data/raw/model/Model_B_*.pkl "gs://$BUCKET/$RESULTS/raw/model/"

# Upload executed notebooks
gsutil -q cp notebooks/05a_model_A_training.ipynb "gs://$BUCKET/$RESULTS/notebooks/"
gsutil -q cp notebooks/05b_model_B_training.ipynb "gs://$BUCKET/$RESULTS/notebooks/"

# Signal completion
echo "DONE" | gsutil -q cp - "gs://$BUCKET/$STAGING/DONE"

echo "=== Training complete: $(date) ==="

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

STARTUP_FILE="/tmp/f1-startup.sh"
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
    --scopes=storage-full \
    --metadata-from-file=startup-script="$STARTUP_FILE" \
    --maintenance-policy=TERMINATE \
    --no-restart-on-failure

echo ""
echo "============================================================"
echo "  VM '$VM_NAME' created successfully!"
echo "============================================================"
echo ""
echo "  Monitor progress:"
echo "    gcloud compute ssh $VM_NAME --zone=$ZONE --command='tail -f /var/log/f1-training.log'"
echo ""
echo "  Check if done:"
echo "    gsutil stat gs://$BUCKET/$STAGING_PREFIX/DONE"
echo ""
echo "  Pull results when done:"
echo "    bash scripts/fetch_training_results.sh"
echo ""
echo "  Manual cleanup (VM self-deletes, but just in case):"
echo "    gcloud compute instances delete $VM_NAME --zone=$ZONE --quiet"
echo ""
echo "  Estimated time: 15-30 min (GPU) / 30-60 min (CPU)"
echo "============================================================"
