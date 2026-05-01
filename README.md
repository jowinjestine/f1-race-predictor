# F1 Race Predictor

ML-powered Formula 1 race outcome predictions using historical telemetry, gradient-boosted ensembles, and autoregressive lap-by-lap simulation.

## Overview

Predicts F1 race finishing positions through a pipeline of nine models that progress from single-lap predictions through multi-model stacking to full-race simulation with uncertainty quantification.

**Data sources:** [FastF1](https://docs.fastf1.dev/) (2018-2024 telemetry), [Jolpica API](https://github.com/jolpica/jolpica-f1) (2025 results), [Open-Meteo](https://open-meteo.com/) (weather)

**Live API:** Deployed on Google Cloud Run at `f1-race-predictor-yqe7tpf66a-uc.a.run.app`

## Architecture

```
                          ┌─────────────────────────────────────────┐
                          │          Training Pipeline              │
                          │                                         │
FastF1 ──┐                │  Model A (lap+tyre) ──┐                │
         ├──▶ Features ──▶│  Model B (lap-only)  ──┼──▶ Model D/E  │
Open-Meteo┘               │  Model C (pre-race)  ──┘   (stacking)  │
                          │                                         │
                          │  Model F (lap-time ratio) ──▶ Simulator │
                          │  Model G (temporal GRU)                 │
                          │  Model H (delta + MC)     ──▶ Ensemble  │
                          │  Model I (quantile MC)                  │
                          └────────────────┬────────────────────────┘
                                           │
                                           ▼
                                    FastAPI on Cloud Run
                                   POST /api/v1/simulate
```

### Model Hierarchy

| Layer | Models | Purpose |
|-------|--------|---------|
| Base predictors | A, B, C | Lap-level and pre-race position predictions |
| Stacking ensembles | D, E | Combine base predictions for race-level accuracy |
| Simulation | F, G | Autoregressive lap-time-ratio prediction |
| Production simulators | H, I | Delta-baseline and quantile-based race simulation |
| Serving ensemble | H+E | H's trajectories refined by E's final-position stacker |

See [docs/models.md](docs/models.md) for detailed documentation on each model.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| ML frameworks | LightGBM, XGBoost, PyTorch (GRU, FT-Transformer, MLP) |
| API framework | FastAPI |
| Data sources | FastF1, Jolpica, Open-Meteo |
| Cloud | GCP (Cloud Run, GCS, Artifact Registry) |
| CI/CD | GitHub Actions (lint, format, typecheck, test, deploy) |
| GPU training | GCE VM with NVIDIA L4 (CUDA) |

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) (for deployment)

### Local Development

```bash
git clone https://github.com/jowinjestine/f1-race-predictor.git
cd f1-race-predictor

# Install all dependencies
uv sync --extra dev --extra serve

# Run CI checks
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy
uv run pytest
```

### Running the API Locally

```bash
uv sync --extra serve
uv run uvicorn f1_predictor.api.main:app --reload
```

Models are loaded from `data/raw/model/` by default. Set `F1_LOAD_FROM_GCS=true` and `F1_GCS_BUCKET=f1-predictor-artifacts-jowin` to download from GCS at startup.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/simulate` | Run H+E ensemble race simulation |
| POST | `/api/v1/simulate/monte-carlo` | Monte Carlo simulation with position distributions |
| GET | `/api/v1/circuits` | List available circuits with default strategies |
| GET | `/api/v1/drivers/{season}` | List drivers for a season |
| GET | `/api/v1/races/{season}` | List races for a season |
| GET | `/healthz` | Liveness check |
| GET | `/readyz` | Readiness check (models loaded) |

### Example Request

```bash
curl -X POST https://f1-race-predictor-yqe7tpf66a-uc.a.run.app/api/v1/simulate \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "circuit": "Monza",
    "drivers": [
      {"driver": "VER", "grid_position": 1, "q1": 80.5, "q2": 80.1, "q3": 79.8},
      {"driver": "NOR", "grid_position": 2, "q1": 80.7, "q2": 80.3, "q3": 80.0}
    ]
  }'
```

## Project Structure

```
f1-race-predictor/
├── src/f1_predictor/
│   ├── api/               # FastAPI application (config, schemas, routers)
│   ├── data/              # Data ingestion (FastF1, Jolpica, Open-Meteo, GCS)
│   ├── features/          # Feature engineering (lap, race, simulation, delta, sequence)
│   ├── models/            # Training utilities (GPU detection, DL architectures)
│   ├── simulation/        # Race simulators (engine, delta, quantile, ensemble)
│   └── explain/           # SHAP explanations
├── notebooks/
│   ├── 01-04_*.ipynb      # Data collection and EDA
│   ├── 05a-05i_*.ipynb    # Model training (A through I)
│   ├── 06-07_*.ipynb      # SHAP analysis and comparison
│   ├── 08_*.ipynb         # Simulation comparison
│   └── 09_*.ipynb         # Ensemble validation
├── docs/
│   ├── models.md          # Model documentation
│   ├── data_dictionary.md # Field definitions
│   └── GPU_DL_PLAN.md     # GPU training architecture
├── scripts/               # GCE training scripts
├── tests/                 # Test suite
├── Dockerfile             # Production container
└── pyproject.toml         # Project configuration
```

## GPU Training

```bash
# Launch training on GCE VM with L4 GPU
bash scripts/run_training_remote.sh

# Download results
bash scripts/fetch_training_results.sh
```

All trained models are stored in `gs://f1-predictor-artifacts-jowin/data/raw/model/`.

## Deployment

The API auto-deploys to Cloud Run on push to `main` (when `src/`, `Dockerfile`, `pyproject.toml`, or `uv.lock` change).

- **Image registry:** `us-docker.pkg.dev/jowin-personal-2026/f1-race-predictor/f1-race-predictor`
- **Resources:** 2 CPU, 2 GiB RAM, 1-5 instances
- **Auth:** Requires GCP identity token (not publicly accessible)
- **CI/CD:** GitHub Actions with Workload Identity Federation (no long-lived keys)

## Development

### Commits

[Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `ci:`, `chore:`

### Branching

- `main` is protected — all changes via squash-merge PRs
- Feature branches: `feat/<short-name>`

## License

MIT — see [LICENSE](LICENSE).
