# Deployment

The F1 Race Predictor API runs on Google Cloud Run with automated CI/CD via GitHub Actions.

## Infrastructure

```
GitHub (main branch)
    │
    ├──▶ CI Workflow (.github/workflows/ci.yml)
    │      ruff check → ruff format → mypy → pytest
    │
    └──▶ Deploy Workflow (.github/workflows/deploy.yml)
           Docker build → Artifact Registry → Cloud Run
```

| Component | Detail |
|-----------|--------|
| **Container registry** | `us-docker.pkg.dev/jowin-personal-2026/f1-race-predictor/f1-race-predictor` |
| **Cloud Run service** | `f1-race-predictor` in `us-central1` |
| **CPU / Memory** | 2 vCPU, 2 GiB RAM |
| **Instances** | Min 1, Max 5 |
| **Concurrency** | 10 requests per instance |
| **Request timeout** | 300 seconds |
| **Auth** | Google Cloud IAM (identity token required) |
| **GCS bucket** | `f1-predictor-artifacts-jowin` (models + data) |

## Container

The `Dockerfile` builds a Python 3.11-slim image:

1. Install `libgomp1` (required by LightGBM)
2. Copy `uv` from the official image
3. Install dependencies with `uv sync --frozen --no-dev --extra serve`
4. Install the package in editable mode
5. Start Uvicorn on port 8080

At startup, the container downloads ~50MB of model and data files from GCS:
- Model pickles: `Model_H_LightGBM_GOSS_Delta.pkl`, `Model_E_*.pkl`, `Model_A_*.pkl`, `Model_B_*.pkl`
- Lookup tables: `field_medians.pkl`, `circuit_defaults.pkl`
- Parquet data: `all_races.parquet`, `all_laps.parquet`

Cold start takes ~30-60 seconds (GCS download + model deserialization).

## CI/CD Pipeline

### CI (.github/workflows/ci.yml)

Triggers on every push and PR to `main`.

| Step | Command |
|------|---------|
| Lint | `uv run ruff check src/ tests/` |
| Format | `uv run ruff format --check src/ tests/` |
| Type check | `uv run mypy` |
| Test | `uv run pytest` |

### Deploy (.github/workflows/deploy.yml)

Triggers on push to `main` when these paths change:
- `src/**`
- `Dockerfile`
- `pyproject.toml`
- `uv.lock`

| Step | Detail |
|------|--------|
| Auth | Workload Identity Federation (no long-lived service account keys) |
| Build | `docker build -t $IMAGE:$SHA -t $IMAGE:latest .` |
| Push | To Artifact Registry |
| Deploy | `gcloud run deploy` with environment variables |

Environment variables set on Cloud Run:
- `F1_LOAD_FROM_GCS=true`
- `F1_GCS_BUCKET=f1-predictor-artifacts-jowin`

### Claude Code Action (.github/workflows/claude.yml)

Allows Claude Code to respond to `@claude` mentions in issues and PR comments. Has write access to contents, PRs, and issues.

## Local Development

```bash
# Install all dependencies
uv sync --extra dev --extra serve

# Run the API locally (loads models from data/raw/model/)
uv run uvicorn f1_predictor.api.main:app --reload

# Or load from GCS
F1_LOAD_FROM_GCS=true uv run uvicorn f1_predictor.api.main:app --reload
```

## GPU Training

Model training runs on a GCE VM with an NVIDIA L4 GPU (16GB VRAM, CUDA 12.4).

```bash
# Launch training on GCE
bash scripts/run_training_remote.sh

# Fetch results to local machine
bash scripts/fetch_training_results.sh
```

All trained models are stored in `gs://f1-predictor-artifacts-jowin/data/raw/model/`.

| Model File | Size | Algorithm |
|------------|------|-----------|
| `Model_A_LightGBM_GOSS.pkl` | ~2 MB | LightGBM GOSS |
| `Model_B_LightGBM_GOSS.pkl` | ~2 MB | LightGBM GOSS |
| `Model_E_LightGBM_shallow.pkl` | ~500 KB | LightGBM shallow |
| `Model_H_LightGBM_GOSS_Delta.pkl` | ~3 MB | LightGBM GOSS |
| `field_medians.pkl` | ~100 KB | Lookup table |
| `circuit_defaults.pkl` | ~50 KB | Lookup table |

## Monitoring

- **Liveness:** `GET /api/health` — always returns 200
- **Readiness:** `GET /api/ready` — returns 503 until all models are loaded
- **Cloud Run logs:** `gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=f1-race-predictor"`
- **Cloud Run metrics:** CPU, memory, request latency, instance count available in Cloud Console
