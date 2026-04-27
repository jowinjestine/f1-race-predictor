#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Download training results from GCS to local paths.
# Run after the remote VM finishes all models (A, B, C, D) + comparison.
#
# Usage: bash scripts/fetch_training_results.sh
# ---------------------------------------------------------------------------

BUCKET="f1-predictor-artifacts-jowin"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo ">>> Downloading all model artifacts from GCS..."

mkdir -p data/training data/raw/model notebooks

# Prediction parquets
gcloud storage cp "gs://$BUCKET/data/training/model_A_*.parquet" data/training/
gcloud storage cp "gs://$BUCKET/data/training/model_B_*.parquet" data/training/
gcloud storage cp "gs://$BUCKET/data/training/model_C_*.parquet" data/training/
gcloud storage cp "gs://$BUCKET/data/training/model_D_*.parquet" data/training/
gcloud storage cp "gs://$BUCKET/data/training/model_E_*.parquet" data/training/

# Model pickles
gcloud storage cp "gs://$BUCKET/data/raw/model/Model_A_*.pkl" data/raw/model/
gcloud storage cp "gs://$BUCKET/data/raw/model/Model_B_*.pkl" data/raw/model/
gcloud storage cp "gs://$BUCKET/data/raw/model/Model_C_*.pkl" data/raw/model/
gcloud storage cp "gs://$BUCKET/data/raw/model/Model_D_*.pkl" data/raw/model/
gcloud storage cp "gs://$BUCKET/data/raw/model/Model_E_*.pkl" data/raw/model/

# Executed notebooks
gcloud storage cp "gs://$BUCKET/data/notebooks/05a_model_A_training.ipynb" notebooks/
gcloud storage cp "gs://$BUCKET/data/notebooks/05b_model_B_training.ipynb" notebooks/
gcloud storage cp "gs://$BUCKET/data/notebooks/05c_model_C_training.ipynb" notebooks/
gcloud storage cp "gs://$BUCKET/data/notebooks/05d_model_D_stacking.ipynb" notebooks/
gcloud storage cp "gs://$BUCKET/data/notebooks/05e_model_E_rich_stacking.ipynb" notebooks/
gcloud storage cp "gs://$BUCKET/data/notebooks/06_model_comparison.ipynb" notebooks/

echo ""
echo ">>> Downloaded artifacts:"
ls -la data/training/model_*.parquet 2>/dev/null | wc -l | xargs -I{} echo "  {} prediction parquets"
ls -la data/raw/model/Model_*.pkl 2>/dev/null | wc -l | xargs -I{} echo "  {} model pickles"

echo ""
echo ">>> Results saved to:"
echo "  Predictions: data/training/"
echo "  Models:      data/raw/model/"
echo "  Notebooks:   notebooks/"
echo ""
echo ">>> All training complete. Review notebooks for results."
