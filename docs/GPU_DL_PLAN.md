# GPU Deep Learning Models — GCE CUDA Training

## Overview

This document describes the GPU training architecture for the F1 Race Predictor.
The primary training target is a **GCE VM with NVIDIA T4 + CUDA 12.4**.

## Hardware Target

| Environment | GPU | VRAM | Backend | Status |
|-------------|-----|------|---------|--------|
| GCE VM (primary) | NVIDIA T4 | 16 GB | CUDA 12.4 | Active |
| WSL2 (deprecated) | AMD RX 7900 XT | 20 GB | ROCm 6.2 | Not recommended |

## GPU Backend Compatibility

| Library | NVIDIA CUDA | AMD ROCm | Notes |
|---------|-------------|----------|-------|
| PyTorch | `torch.cuda.*` | `torch.cuda.*` (HIP) | Identical API |
| XGBoost | `device='cuda'` | CPU only | XGBoost GPU requires CUDA |
| LightGBM | `device='gpu'` | `device='gpu'` | OpenCL backend, works on AMD |
| rtdl (FT-Transformer) | Works | Works | Pure PyTorch |

## Deep Learning Models

### Models A/B (lap-level, ~130K rows)

| Model | Architecture | Key HPs |
|-------|-------------|---------|
| GRU_2layer | 2-layer bidirectional GRU, hidden=64, per-timestep head | hidden_dim, num_layers, dropout |
| FT_Transformer | rtdl FTTransformer, d_token=64, 3 blocks | d_token, n_blocks, attention_dropout |

### Model C (race-level, ~3500 rows)

| Model | Architecture | Key HPs |
|-------|-------------|---------|
| MLP_3layer | 64->32->1, BatchNorm, Dropout(0.3), weight_decay=1e-3 | hidden1, hidden2, dropout |

### Model D (stacking)

All combinations of A x B x C variants, two-phase approach:
- **Phase 1:** RidgeCV screen of all combinations (~10 min)
- **Phase 2:** Full tournament with Optuna on top 20 (~20 min)

## Key Files

| File | Purpose |
|------|---------|
| `src/f1_predictor/models/gpu.py` | Unified GPU detection (ROCm/CUDA/CPU) |
| `src/f1_predictor/models/architectures.py` | GRU, FT-Transformer, MLP wrappers |
| `src/f1_predictor/models/dl_utils.py` | Training loop, early stopping, datasets |
| `scripts/run_training_remote.sh` | GCE VM training (primary) |
| `scripts/fetch_training_results.sh` | Download results from GCS |
| `scripts/setup_wsl_env.sh` | WSL2 environment bootstrap (deprecated) |
| `scripts/run_training_wsl.sh` | Local WSL2 training (deprecated) |

## Quick Start (GCE VM)

```bash
# Launch training on GCE VM (runs all models + comparison, self-deletes)
bash scripts/run_training_remote.sh

# Monitor progress
gcloud compute ssh f1-training-XXXXX --zone=us-central1-a \
    --command='tail -f /var/log/f1-training.log'

# Check completion
gsutil stat gs://f1-predictor-artifacts-jowin/staging/training-run/DONE

# Download results
bash scripts/fetch_training_results.sh
```
