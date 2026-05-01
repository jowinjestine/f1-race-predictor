"""Health check endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from f1_predictor.api.dependencies import registry

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def readiness() -> dict[str, str | bool]:
    return {
        "status": "ready" if registry.ready else "loading",
        "models_loaded": registry.ready,
    }
