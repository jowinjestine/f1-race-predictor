"""FastAPI application — F1 race prediction API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from f1_predictor.api.config import Settings
from f1_predictor.api.dependencies import registry
from f1_predictor.api.routers import data, health, simulation

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    registry.load(settings)
    yield


DESCRIPTION = """\
ML-powered Formula 1 race outcome predictions using an ensemble of nine
gradient-boosted and deep learning models trained on 2018-2025 telemetry data.

## How it works

The API runs an **autoregressive lap-by-lap race simulation** using Model H
(LightGBM GOSS with delta-baseline decomposition). At each simulated lap, the
model predicts how much faster or slower each driver is than the historical
field median for that circuit and lap, then converts to absolute lap times
using qualifying pace as the baseline.

Optionally, final positions can be refined by Model E — a rich feature stacker
that aggregates lap-level trajectory statistics into 13 meta-features.

## Endpoints

- **Simulate** — deterministic race simulation with full lap-by-lap telemetry
- **Monte Carlo** — run N simulations with Gaussian noise injection to produce
  position distributions (percentiles, standard deviations)
- **Data** — look up available circuits, drivers, and races

## Models

| Model | Role | Algorithm |
|-------|------|-----------|
| **H** | Lap-by-lap simulation | LightGBM GOSS (delta-ratio target) |
| **E** | Final position refinement | LightGBM shallow (13 meta-features) |
| **Monte Carlo** | Uncertainty quantification | 200 noise-injected H simulations |

Built with [FastF1](https://docs.fastf1.dev/), LightGBM, and FastAPI.
"""

app = FastAPI(
    title="F1 Race Predictor",
    summary="Pre-race simulation API — predict finishing positions for any F1 grid",
    description=DESCRIPTION,
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(simulation.router)
app.include_router(data.router)
