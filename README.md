# F1 Race Predictor

ML-powered Formula 1 race outcome predictions using historical telemetry, weather data, and gradient-boosted models with explainable AI.

## Overview

Predicts F1 race finishing positions by combining:

- **Historical race data** from [FastF1](https://docs.fastf1.dev/) (2018-2025 seasons)
- **Weather conditions** from [Open-Meteo](https://open-meteo.com/)
- **XGBoost** gradient-boosted models for prediction
- **SHAP** values for model explainability
- **FastAPI** serving layer deployed on Google Cloud Run

## Architecture

```
FastF1 API ──┐
             ├──▶ Feature Engineering ──▶ XGBoost Model ──▶ FastAPI ──▶ Cloud Run
Open-Meteo ──┘         │                      │
                       ▼                      ▼
                   Parquet Store          MLflow (local)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Package Manager | [uv](https://docs.astral.sh/uv/) |
| ML Model | XGBoost |
| Explainability | SHAP |
| API Framework | FastAPI |
| Data Sources | FastF1, Open-Meteo |
| Cloud | GCP (Cloud Run, GCS, BigQuery) |
| CI/CD | GitHub Actions |
| Experiment Tracking | MLflow |

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) (for deployment)
- Git

### Local Development

```bash
git clone https://github.com/jowinjestine/f1-race-predictor.git
cd f1-race-predictor

# Install dependencies (uv downloads Python 3.11 automatically)
make install

# Run all CI checks locally
make ci

# Individual commands
make lint        # ruff linter
make format      # auto-format code
make typecheck   # mypy strict mode
make test        # pytest with coverage
```

## Project Structure

```
f1-race-predictor/
├── src/f1_predictor/
│   ├── data/          # Data ingestion (FastF1 + Open-Meteo)
│   ├── features/      # Feature engineering pipeline
│   ├── models/        # XGBoost training + MLflow logging
│   ├── explain/       # SHAP explanations
│   └── api/           # FastAPI application
├── tests/             # Test suite
├── notebooks/         # Exploration notebooks
├── data/
│   ├── raw/           # Raw data (git-ignored)
│   └── processed/     # Processed features (git-ignored)
└── pyproject.toml     # Project configuration
```

## GPU Training (WSL2 / ROCm)

```bash
# One-time setup (auto-detects AMD ROCm or NVIDIA CUDA)
bash scripts/setup_wsl_env.sh

# Run full training pipeline (all 4 models + comparison)
bash scripts/run_training_wsl.sh
```

Also supports NVIDIA CUDA — the setup script auto-detects the GPU vendor.
See [docs/GPU_DL_PLAN.md](docs/GPU_DL_PLAN.md) for architecture details.

## Development

### Commits

[Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `ci:`, `chore:`

### Branching

- `main` is protected — all changes via squash-merge PRs
- Feature branches: `feat/<short-name>`

## License

MIT — see [LICENSE](LICENSE).
