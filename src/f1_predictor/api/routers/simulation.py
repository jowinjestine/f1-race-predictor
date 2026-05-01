"""Simulation endpoints — race prediction using H+E ensemble."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from f1_predictor.api.dependencies import registry
from f1_predictor.api.schemas import (
    FinalStanding,
    LapRecordOut,
    MonteCarloResponse,
    MonteCarloStanding,
    SimulationRequest,
    SimulationResponse,
)
from f1_predictor.simulation.delta_simulator import MonteCarloSimulator
from f1_predictor.simulation.ensemble_simulator import EnsembleSimulator

router = APIRouter(prefix="/api/v1", tags=["simulation"])


def _to_driver_dicts(req: SimulationRequest) -> list[dict[str, Any]]:
    return [
        {
            "driver": d.driver,
            "grid_position": d.grid_position,
            "q1": d.q1,
            "q2": d.q2,
            "q3": d.q3,
            "initial_tyre": d.initial_tyre,
        }
        for d in req.drivers
    ]


def _to_strategies(req: SimulationRequest) -> dict[str, list[tuple[str, int | None]]] | None:
    if not req.strategies:
        return None
    return {
        drv: [(leg.compound, leg.pit_on_lap) for leg in legs]
        for drv, legs in req.strategies.items()
    }


@router.post("/simulate", response_model=SimulationResponse)
def simulate(req: SimulationRequest) -> SimulationResponse:
    if not registry.ready or registry.h_simulator is None:
        raise HTTPException(503, "Models not loaded yet")

    if req.circuit not in registry.circuit_defaults:
        available = sorted(registry.circuit_defaults.keys())
        raise HTTPException(400, f"Unknown circuit '{req.circuit}'. Available: {available}")

    drivers = _to_driver_dicts(req)
    strategies = _to_strategies(req)

    sim = EnsembleSimulator(registry.h_simulator, registry.model_e, blend_laps=req.blend_laps)
    result = sim.simulate(req.circuit, drivers, strategies)

    model_label = "H+E ensemble" if req.blend_laps > 0 else "H only"

    lap_records = [
        LapRecordOut(
            lap_number=rec.lap,
            driver=rec.driver,
            position=rec.position,
            lap_time=rec.lap_time,
            cum_time=rec.cum_time,
            gap_to_leader=rec.gap_to_leader,
            compound=rec.compound,
            tire_life=rec.tire_life,
            stint=rec.stint,
        )
        for rec in result.lap_records
    ]

    final_standings = [FinalStanding(**fr) for fr in result.final_results]

    return SimulationResponse(
        circuit=result.circuit,
        total_laps=result.total_laps,
        model=model_label,
        blend_laps=req.blend_laps,
        lap_records=lap_records,
        final_standings=final_standings,
    )


@router.post("/simulate/monte-carlo", response_model=MonteCarloResponse)
def simulate_monte_carlo(req: SimulationRequest) -> MonteCarloResponse:
    if not registry.ready or registry.h_simulator is None:
        raise HTTPException(503, "Models not loaded yet")

    if req.circuit not in registry.circuit_defaults:
        available = sorted(registry.circuit_defaults.keys())
        raise HTTPException(400, f"Unknown circuit '{req.circuit}'. Available: {available}")

    drivers = _to_driver_dicts(req)
    strategies = _to_strategies(req)

    mc = MonteCarloSimulator(
        registry.h_simulator,
        n_simulations=req.n_simulations,
    )
    mc_result = mc.simulate(req.circuit, drivers, strategies)

    return MonteCarloResponse(
        circuit=mc_result.circuit,
        n_simulations=mc_result.n_simulations,
        model="H Monte Carlo",
        standings=[MonteCarloStanding(**r) for r in mc_result.results],
    )
