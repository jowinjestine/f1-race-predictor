"""FastAPI application — F1 race prediction API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from f1_predictor.api.config import Settings
from f1_predictor.api.dependencies import registry
from f1_predictor.api.routers import data, health, simulation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    registry.load(settings)
    yield


app = FastAPI(
    title="F1 Race Predictor",
    description="Pre-race simulation using H+E ensemble (Model H trajectories + Model E final position refinement)",
    version="0.1.0",
    lifespan=lifespan,
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
