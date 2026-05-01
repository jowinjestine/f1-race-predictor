"""Health check endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from f1_predictor.api.dependencies import registry

router = APIRouter(prefix="/api", tags=["health"])


@router.get(
    "/health",
    summary="Liveness probe",
    description="Returns 200 if the service process is running. Does not check model state.",
)
def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/ready",
    summary="Readiness probe",
    description="""Returns 200 with `models_loaded: true` once all ML models and circuit data
have been downloaded from GCS and loaded into memory. During startup (typically 10-30s),
returns `models_loaded: false` — simulation endpoints will return 503 until ready.""",
)
def readiness() -> dict[str, str | bool]:
    return {
        "status": "ready" if registry.ready else "loading",
        "models_loaded": registry.ready,
    }
