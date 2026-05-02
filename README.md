# F1 Race Predictor

ML-powered Formula 1 race outcome predictions using historical telemetry, gradient-boosted ensembles, and autoregressive lap-by-lap simulation.

## Overview

Predicts F1 race finishing positions through a pipeline of nine models that progress from single-lap predictions through multi-model stacking to full-race simulation with uncertainty quantification.

**Data sources:** [FastF1](https://docs.fastf1.dev/) (2018-2024 telemetry), [Jolpica API](https://github.com/jolpica/jolpica-f1) (2025 results), [Open-Meteo](https://open-meteo.com/) (weather)

**Live API:** Deployed on Google Cloud Run — requires GCP identity token for access.

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
                           ┌────────────────────────┐
                           │ POST /api/v1/simulate   │
                           │ POST /simulate/monte-   │
                           │      carlo              │
                           │ POST /optimize-strategy │
                           └────────────────────────┘
```

### Model Hierarchy

| Layer | Models | Purpose |
|-------|--------|---------|
| Base predictors | A, B, C | Lap-level and pre-race position predictions |
| Stacking ensembles | D, E | Combine base predictions for race-level accuracy |
| Simulation | F, G | Autoregressive lap-time-ratio prediction |
| Production simulators | H, I | Delta-baseline and quantile-based race simulation |
| Serving ensemble | H+E | H's trajectories refined by E's final-position stacker |

### Key Features

- **Lap-by-lap simulation** — full race telemetry with position, lap time, gap, tyre state at every lap
- **Monte Carlo uncertainty** — position distributions with percentiles and confidence intervals
- **DNF prediction** — per-driver retirement probability with per-lap hazard sampling
- **Strategy optimization** — find optimal pit timing and compound selection for any driver
- **Compound differentiation** — physics-informed tyre pace offsets and degradation rates
- **Decorrelated ensemble** — Models A, B, and H feed independent signals to Model E

## Documentation

| Document | Description |
|----------|-------------|
| [Methodology](docs/methodology.md) | How the predictor works — delta decomposition, compound corrections, Monte Carlo, cross-validation, evaluation |
| [Models](docs/models.md) | Detailed documentation of all nine models (A-I) — features, training, performance |
| [API Reference](docs/api.md) | Complete endpoint documentation with request/response examples |
| [Simulation Engine](docs/simulation.md) | Internals of the lap-by-lap simulator — DriverState, feature construction, blending, strategy optimization |
| [Data Dictionary](docs/data_dictionary.md) | Field definitions for race-level and lap-level datasets |
| [Deployment](docs/deployment.md) | Cloud Run setup, CI/CD pipeline, GPU training, container build |

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

## Quick Start

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

Models are loaded from `data/raw/model/` by default. Set `F1_LOAD_FROM_GCS=true` to download from GCS at startup.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/simulate` | Run H+E ensemble race simulation with lap-by-lap telemetry |
| POST | `/api/v1/simulate/monte-carlo` | Monte Carlo simulation with position distributions and DNF rates |
| POST | `/api/v1/optimize-strategy` | Find optimal pit strategy for a target driver |
| GET | `/api/v1/circuits` | List available circuits with default strategies |
| GET | `/api/v1/drivers/{season}` | List drivers for a season (2018-2025) |
| GET | `/api/v1/races/{season}` | List races for a season |
| GET | `/api/health` | Liveness check |
| GET | `/api/ready` | Readiness check (models loaded) |

See [API Reference](docs/api.md) for full request/response documentation.

### Example: Race Simulation

```bash
curl -X POST https://f1-race-predictor-yqe7tpf66a-uc.a.run.app/api/v1/simulate \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "circuit": "Monaco Grand Prix",
    "drivers": [
      {"driver": "VER", "grid_position": 1, "q3": 70.2},
      {"driver": "LEC", "grid_position": 2, "q3": 70.4, "dnf_probability": 0.15},
      {"driver": "NOR", "grid_position": 3, "q3": 70.6}
    ],
    "blend_laps": 10
  }'
```

### Example: Strategy Optimization

```bash
curl -X POST https://f1-race-predictor-yqe7tpf66a-uc.a.run.app/api/v1/optimize-strategy \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "circuit": "Monaco Grand Prix",
    "drivers": [
      {"driver": "VER", "grid_position": 1, "q3": 70.2},
      {"driver": "LEC", "grid_position": 2, "q3": 70.4},
      {"driver": "NOR", "grid_position": 3, "q3": 70.6}
    ],
    "target_driver": "NOR",
    "use_monte_carlo": false
  }'
```

## Project Structure

```
f1-race-predictor/
├── src/f1_predictor/
│   ├── api/               # FastAPI application
│   │   ├── main.py        # App setup, lifespan, CORS
│   │   ├── config.py      # Settings (env vars with F1_ prefix)
│   │   ├── schemas.py     # Pydantic request/response models
│   │   ├── dependencies.py # Model registry, startup loading
│   │   └── routers/       # health, simulation, data endpoints
│   ├── data/              # Data ingestion
│   │   ├── collect.py     # FastF1 race collection + weather
│   │   ├── collect_laps.py # Lap-by-lap telemetry collection
│   │   ├── jolpica.py     # Jolpica API client (2025 season)
│   │   └── storage.py     # GCS upload/download helpers
│   ├── features/          # Feature engineering
│   │   ├── common.py      # Shared utilities (rolling means, one-hot)
│   │   ├── lap_features.py # Lap-level features (Models A, B)
│   │   ├── race_features.py # Pre-race features (Model C)
│   │   ├── simulation_features.py # Simulation features (Model F)
│   │   ├── delta_features.py # Delta-ratio features (Model H)
│   │   ├── sequence_features.py # Windowed sequences (Model G)
│   │   └── splits.py      # CV strategies (LeaveOneSeason, ExpandingWindow)
│   ├── simulation/        # Race simulators
│   │   ├── engine.py      # Base RaceSimulator, DriverState, LapRecord
│   │   ├── delta_simulator.py # DeltaRaceSimulator (H), MonteCarloSimulator
│   │   ├── ensemble_simulator.py # EnsembleSimulator (H+E), A/B integration
│   │   ├── strategy.py    # Candidate generation, strategy optimization
│   │   ├── defaults.py    # Circuit defaults, default strategy builder
│   │   ├── quantile_simulator.py # QuantileRaceSimulator (I)
│   │   └── sequence_simulator.py # SequenceRaceSimulator (G)
│   ├── models/            # Training utilities
│   │   ├── gpu.py         # GPU detection (CUDA / ROCm)
│   │   ├── architectures.py # DL models (GRU, FT-Transformer, MLP)
│   │   └── dl_utils.py    # Training loop, early stopping
│   └── explain/           # SHAP explanations
├── notebooks/
│   ├── 01-04_*.ipynb      # Data collection and EDA
│   ├── 05a-05i_*.ipynb    # Model training (A through I)
│   ├── 06-07_*.ipynb      # SHAP analysis and comparison
│   ├── 08_*.ipynb         # Simulation comparison
│   └── 09_*.ipynb         # Ensemble validation
├── docs/                  # Documentation (see table above)
├── scripts/               # GCE training scripts
├── tests/                 # Test suite
├── Dockerfile             # Production container
└── pyproject.toml         # Project configuration
```

## Performance

### Simulation Accuracy (4 held-out 2024 races)

| Configuration | RMSE | Spearman |
|---------------|------|----------|
| Model H only (blend=0) | 3.46 | 0.82 |
| Model E standalone | 2.60 | 0.93 |
| H+E ensemble (blend=10) | 3.50 | 0.80 |

### Strategy Optimizer

On a 78-lap Monaco simulation with 5 drivers:
- 20 candidate strategies evaluated in <2 seconds
- 30-second time spread across strategies with compound differentiation
- Position differentiation (P3 vs P4) for suboptimal compound choices

## Deployment

Auto-deploys to Cloud Run on push to `main` when `src/`, `Dockerfile`, `pyproject.toml`, or `uv.lock` change.

See [Deployment](docs/deployment.md) for full infrastructure details.

## Development

### Commits

[Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `ci:`, `chore:`

### Branching

- `main` is protected — all changes via squash-merge PRs
- Feature branches: `feat/<short-name>`, `fix/<short-name>`

## License

MIT — see [LICENSE](LICENSE).
