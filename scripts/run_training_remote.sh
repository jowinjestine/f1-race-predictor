#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Run all model training (A, B, C, D, E, F) + comparison on a GCP VM.
# The VM self-deletes after uploading results to GCS.
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
ZONE="${ZONE:-us-central1-a}"
BUCKET="f1-predictor-artifacts-jowin"
VM_NAME="f1-training-$(date +%s)"
STAGING_PREFIX="staging/training-run"
RESULTS_PREFIX="data"

# Default: L4 GPU. Use --t4 for T4, --cpu for CPU-only.
USE_GPU=true
GPU_TYPE="${GPU_TYPE:-l4}"
IMAGE_FAMILY="common-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT="deeplearning-platform-release"

if [[ "${1:-}" == "--cpu" ]]; then
    USE_GPU=false
    MACHINE_TYPE="e2-standard-8"
    ACCELERATOR=""
    IMAGE_FAMILY="debian-12"
    IMAGE_PROJECT="debian-cloud"
    echo ">>> CPU-only mode (e2-standard-8, 8 vCPUs)"
elif [[ "$GPU_TYPE" == "t4" || "${1:-}" == "--t4" ]]; then
    MACHINE_TYPE="n1-standard-8"
    ACCELERATOR="--accelerator=type=nvidia-tesla-t4,count=1"
    echo ">>> GPU mode (n1-standard-8 + T4, 16GB VRAM)"
else
    MACHINE_TYPE="g2-standard-8"
    ACCELERATOR="--accelerator=type=nvidia-l4,count=1"
    echo ">>> GPU mode (g2-standard-8 + L4, 24GB VRAM)"
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

# Upload data files (notebooks load from GCS, but also provide local fallback)
gcloud storage cp \
    data/processed/lap_tyre/features_laps_tyre.parquet \
    "gs://$BUCKET/$STAGING_PREFIX/features_laps_tyre.parquet" --quiet
gcloud storage cp \
    data/processed/lap_notyre/features_laps_notyre.parquet \
    "gs://$BUCKET/$STAGING_PREFIX/features_laps_notyre.parquet" --quiet
gcloud storage cp \
    data/processed/race/features_race.parquet \
    "gs://$BUCKET/$STAGING_PREFIX/features_race.parquet" --quiet

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

export HOME="${HOME:-/root}"

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Download repo
gsutil -q cp "gs://$BUCKET/$STAGING/repo.tar.gz" repo.tar.gz
tar xzf repo.tar.gz
rm repo.tar.gz

# Place data files locally as fallback
mkdir -p data/processed/lap_tyre data/processed/lap_notyre data/processed/race \
    data/training data/raw/model
gsutil -q cp "gs://$BUCKET/$STAGING/features_laps_tyre.parquet" \
    data/processed/lap_tyre/features_laps_tyre.parquet
gsutil -q cp "gs://$BUCKET/$STAGING/features_laps_notyre.parquet" \
    data/processed/lap_notyre/features_laps_notyre.parquet
gsutil -q cp "gs://$BUCKET/$STAGING/features_race.parquet" \
    data/processed/race/features_race.parquet

# Install Python + deps
uv python install 3.11
uv sync --frozen --group dev
uv pip install xgboost lightgbm optuna

# Check GPU and install PyTorch + DL deps
if nvidia-smi > /dev/null 2>&1; then
    echo "GPU detected — installing GPU-enabled packages"
    uv pip install xgboost --upgrade
    uv pip install torch --index-url https://download.pytorch.org/whl/cu126
    uv pip install rtdl-revisiting-models
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

# Run Model C
echo "=== Running Model C: $(date) ==="
uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=3600 \
    notebooks/05c_model_C_training.ipynb \
    --output 05c_model_C_training.ipynb 2>&1 || true

# Run Model D (depends on A, B, C outputs)
echo "=== Running Model D: $(date) ==="
uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=3600 \
    notebooks/05d_model_D_stacking.ipynb \
    --output 05d_model_D_stacking.ipynb 2>&1 || true

# Run Model E (depends on A, B, C outputs)
echo "=== Running Model E: $(date) ==="
uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=3600 \
    notebooks/05e_model_E_rich_stacking.ipynb \
    --output 05e_model_E_rich_stacking.ipynb 2>&1 || true

# Run Model F (lap simulation — independent of A-E)
echo "=== Running Model F: $(date) ==="
uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=7200 \
    notebooks/05f_model_F_lap_simulation.ipynb \
    --output 05f_model_F_lap_simulation.ipynb 2>&1 || true

# Run comparison notebook
echo "=== Running Comparison: $(date) ==="
uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=1800 \
    notebooks/06_model_comparison.ipynb \
    --output 06_model_comparison.ipynb 2>&1 || true

# Upload results to GCS
echo "=== Uploading results: $(date) ==="
gsutil -m -q cp data/training/model_A_*.parquet "gs://$BUCKET/$RESULTS/training/"
gsutil -m -q cp data/training/model_B_*.parquet "gs://$BUCKET/$RESULTS/training/"
gsutil -m -q cp data/raw/model/Model_A_*.pkl "gs://$BUCKET/$RESULTS/raw/model/"
gsutil -m -q cp data/raw/model/Model_B_*.pkl "gs://$BUCKET/$RESULTS/raw/model/"
gsutil -m -q cp data/training/model_C_*.parquet "gs://$BUCKET/$RESULTS/training/"
gsutil -m -q cp data/training/model_D_*.parquet "gs://$BUCKET/$RESULTS/training/"
gsutil -m -q cp data/raw/model/Model_C_*.pkl "gs://$BUCKET/$RESULTS/raw/model/"
gsutil -m -q cp data/raw/model/Model_D_*.pkl "gs://$BUCKET/$RESULTS/raw/model/"
gsutil -m -q cp data/training/model_E_*.parquet "gs://$BUCKET/$RESULTS/training/"
gsutil -m -q cp data/raw/model/Model_E_*.pkl "gs://$BUCKET/$RESULTS/raw/model/"
gsutil -m -q cp data/training/model_F_*.parquet "gs://$BUCKET/$RESULTS/training/"
gsutil -m -q cp data/raw/model/Model_F_*.pkl "gs://$BUCKET/$RESULTS/raw/model/"

# Upload executed notebooks
gsutil -q cp notebooks/05a_model_A_training.ipynb "gs://$BUCKET/$RESULTS/notebooks/"
gsutil -q cp notebooks/05b_model_B_training.ipynb "gs://$BUCKET/$RESULTS/notebooks/"
gsutil -q cp notebooks/05c_model_C_training.ipynb "gs://$BUCKET/$RESULTS/notebooks/"
gsutil -q cp notebooks/05d_model_D_stacking.ipynb "gs://$BUCKET/$RESULTS/notebooks/"
gsutil -q cp notebooks/05e_model_E_rich_stacking.ipynb "gs://$BUCKET/$RESULTS/notebooks/"
gsutil -q cp notebooks/05f_model_F_lap_simulation.ipynb "gs://$BUCKET/$RESULTS/notebooks/"
gsutil -q cp notebooks/06_model_comparison.ipynb "gs://$BUCKET/$RESULTS/notebooks/"

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

MAINT_POLICY=""
if $USE_GPU; then
    MAINT_POLICY="--maintenance-policy=TERMINATE"
fi

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
    $MAINT_POLICY \
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
echo "  Estimated time: 45-90 min (GPU) / 120-180 min (CPU)"
echo "============================================================"
