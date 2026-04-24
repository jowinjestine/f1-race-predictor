# GPU Deep Learning Models + WSL2/ROCm Training Plan

## Overview

This document describes the GPU training architecture for the F1 Race Predictor.
The pipeline supports both **AMD ROCm** (WSL2) and **NVIDIA CUDA** (GCE VM) backends.

## Hardware Targets

| Environment | GPU | VRAM | Backend |
|-------------|-----|------|---------|
| WSL2 (primary) | AMD RX 7900 XT | 20 GB | ROCm 6.2 |
| GCE VM (fallback) | NVIDIA T4 | 16 GB | CUDA |

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
| `scripts/setup_wsl_env.sh` | One-time WSL2 environment bootstrap |
| `scripts/run_training_wsl.sh` | Local training orchestrator |
| `scripts/run_training_remote.sh` | GCE VM training (preserved, fallback) |

## Quick Start (WSL2)

```bash
# One-time setup
bash scripts/setup_wsl_env.sh

# Run full training pipeline
bash scripts/run_training_wsl.sh
```
