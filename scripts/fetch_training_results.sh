#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Download training results from GCS to local paths.
# Run after the remote VM finishes all models (A-I) + comparison.
#
# Usage: bash scripts/fetch_training_results.sh
# ---------------------------------------------------------------------------

BUCKET="f1-predictor-artifacts-jowin"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo ">>> Downloading all model artifacts from GCS..."

mkdir -p data/training data/raw/model notebooks

# Prediction parquets
for m in A B C D E F G H I; do
    gcloud storage cp "gs://$BUCKET/data/training/model_${m}_*.parquet" data/training/ 2>/dev/null || true
done

# Model pickles
for m in A B C D E F G H I; do
    gcloud storage cp "gs://$BUCKET/data/raw/model/Model_${m}_*.pkl" data/raw/model/ 2>/dev/null || true
done

# Executed notebooks
for nb in \
    05a_model_A_training 05b_model_B_training 05c_model_C_training \
    05d_model_D_stacking 05e_model_E_rich_stacking 05f_model_F_lap_simulation \
    05g_model_G_temporal 05h_model_H_delta_mc 05i_model_I_quantile \
    06_model_comparison; do
    gcloud storage cp "gs://$BUCKET/data/notebooks/${nb}.ipynb" notebooks/ 2>/dev/null || true
done

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
