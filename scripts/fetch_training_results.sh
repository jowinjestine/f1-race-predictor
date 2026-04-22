#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Download training results from GCS to local paths.
# Run after the remote VM finishes Models A & B.
#
# Usage: bash scripts/fetch_training_results.sh
# ---------------------------------------------------------------------------

BUCKET="f1-predictor-artifacts-jowin"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo ">>> Downloading Model A & B artifacts from GCS..."

mkdir -p data/training data/raw/model

# Prediction parquets
gsutil -m cp "gs://$BUCKET/data/training/model_A_*.parquet" data/training/
gsutil -m cp "gs://$BUCKET/data/training/model_B_*.parquet" data/training/

# Model pickles
gsutil -m cp "gs://$BUCKET/data/raw/model/Model_A_*.pkl" data/raw/model/
gsutil -m cp "gs://$BUCKET/data/raw/model/Model_B_*.pkl" data/raw/model/

# Executed notebooks
gsutil -m cp "gs://$BUCKET/data/notebooks/05a_model_A_training.ipynb" notebooks/
gsutil -m cp "gs://$BUCKET/data/notebooks/05b_model_B_training.ipynb" notebooks/

echo ""
echo ">>> Downloaded artifacts:"
ls -la data/training/model_A_* data/training/model_B_* 2>/dev/null || echo "  (no prediction parquets found)"
ls -la data/raw/model/Model_A_* data/raw/model/Model_B_* 2>/dev/null || echo "  (no model pickles found)"

echo ""
echo ">>> Results saved to:"
echo "  Predictions: data/training/"
echo "  Models:      data/raw/model/"
echo "  Notebooks:   notebooks/"
echo ""
echo ">>> Next step: Run Model D locally"
echo "  uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 notebooks/05d_model_D_stacking.ipynb --output 05d_model_D_stacking.ipynb"
