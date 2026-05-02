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
    OptimizeStrategyRequest,
    OptimizeStrategyResponse,
    SimulationRequest,
    SimulationResponse,
    StrategyLeg,
    StrategyOption,
)
from f1_predictor.simulation.delta_simulator import MonteCarloSimulator
from f1_predictor.simulation.ensemble_simulator import EnsembleSimulator
from f1_predictor.simulation.strategy import generate_candidates, optimize_strategy

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


def _to_dnf_probs(req: SimulationRequest) -> dict[str, float] | None:
    probs = {d.driver: d.dnf_probability for d in req.drivers if d.dnf_probability > 0}
    return probs or None


@router.post(
    "/simulate",
    response_model=SimulationResponse,
    summary="Run a deterministic race simulation",
    description="""Simulates a full race lap-by-lap using Model H (LightGBM delta-baseline).

**How it works:**

1. Each driver starts with their qualifying time as a pace baseline
2. At every lap, the model predicts each driver's deviation from the historical
   field median lap-time-ratio for that circuit
3. Predictions are clamped (normal: 1.01-1.15x, pit-in: 1.10-1.50x, pit-out: 1.03-1.25x)
   and converted to absolute lap times
4. Positions are derived from cumulative race time after each lap
5. Pit stops are executed according to the provided or default strategy
6. If `dnf_probability` is set for any driver, they may retire mid-race

**Pit strategies:** If no strategy is provided, circuit-default strategies are used
(based on historical data). Custom strategies let you explore "what if" scenarios
like an early undercut or a 1-stop vs 2-stop gamble.

**Blend laps:** When `blend_laps > 0`, the last N laps interpolate H's trajectory
positions toward Model E's final-position predictions. Set to 0 (recommended) for
pure Model H simulation.

Returns full lap-by-lap telemetry (position, lap time, gap, tyre state) plus
final standings.""",
    responses={
        400: {"description": "Unknown circuit name — use GET /api/v1/circuits for valid names"},
        503: {"description": "Models not loaded yet — the service is still starting up"},
    },
)
def simulate(req: SimulationRequest) -> SimulationResponse:
    if not registry.ready or registry.h_simulator is None:
        raise HTTPException(503, "Models not loaded yet")

    if req.circuit not in registry.circuit_defaults:
        available = sorted(registry.circuit_defaults.keys())
        raise HTTPException(400, f"Unknown circuit '{req.circuit}'. Available: {available}")

    drivers = _to_driver_dicts(req)
    strategies = _to_strategies(req)
    dnf_probs = _to_dnf_probs(req)

    sim = EnsembleSimulator(
        registry.h_simulator,
        registry.model_e,
        model_a=registry.model_a,
        model_b=registry.model_b,
        blend_laps=req.blend_laps,
    )
    result = sim.simulate(req.circuit, drivers, strategies, dnf_probs=dnf_probs)

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


@router.post(
    "/simulate/monte-carlo",
    response_model=MonteCarloResponse,
    summary="Run Monte Carlo simulation for position distributions",
    description="""Runs N independent simulations with Gaussian noise (std=0.01) injected into
Model H's lap-time-ratio predictions. Each simulation produces slightly different
trajectories, capturing the inherent uncertainty in race outcomes.

**Use cases:**

- Estimate how likely a driver is to finish on the podium
- Compare strategy risk — a 1-stop may have a higher ceiling but wider position spread
- Quantify the "safety" of a grid position (low std = predictable, high std = volatile)
- When `dnf_probability` is set, each simulation independently samples retirements

**Output:** For each driver, returns the median position, mean, 10th/25th/75th/90th
percentiles, standard deviation, and DNF rate across all simulations.

**Performance:** 50 simulations take ~5s, 200 simulations ~15s. The default (200) provides
stable percentile estimates.""",
    responses={
        400: {"description": "Unknown circuit name"},
        503: {"description": "Models not loaded yet"},
    },
)
def simulate_monte_carlo(req: SimulationRequest) -> MonteCarloResponse:
    if not registry.ready or registry.h_simulator is None:
        raise HTTPException(503, "Models not loaded yet")

    if req.circuit not in registry.circuit_defaults:
        available = sorted(registry.circuit_defaults.keys())
        raise HTTPException(400, f"Unknown circuit '{req.circuit}'. Available: {available}")

    drivers = _to_driver_dicts(req)
    strategies = _to_strategies(req)
    dnf_probs = _to_dnf_probs(req)

    mc = MonteCarloSimulator(
        registry.h_simulator,
        n_simulations=req.n_simulations,
    )
    mc_result = mc.simulate(req.circuit, drivers, strategies, dnf_probs=dnf_probs)

    return MonteCarloResponse(
        circuit=mc_result.circuit,
        n_simulations=mc_result.n_simulations,
        model="H Monte Carlo",
        standings=[MonteCarloStanding(**r) for r in mc_result.results],
    )


@router.post(
    "/optimize-strategy",
    response_model=OptimizeStrategyResponse,
    summary="Find optimal pit strategy for a target driver",
    description="""Generates candidate pit strategies by varying pit lap timing, stop counts,
and compound sequences around the circuit's historical defaults. Each candidate
is evaluated by running a full Model H simulation.

**Two-phase approach:**

1. **Deterministic phase:** Simulates all candidates (~30) in ~1-2 seconds.
   Ranks by finishing position, then total time for tie-breaking.
2. **Monte Carlo phase (optional):** Runs N noisy simulations on the top 5
   strategies for confidence intervals (~10-15 seconds).

Other drivers use circuit-default strategies while the target driver's strategy
is varied. This answers "what's the best strategy for me, given everyone else
drives normally?"

Returns ranked strategies with position predictions and optional uncertainty.""",
    responses={
        400: {"description": "Unknown circuit or target driver not in grid"},
        503: {"description": "Models not loaded yet"},
    },
)
def optimize_pit_strategy(req: OptimizeStrategyRequest) -> OptimizeStrategyResponse:
    if not registry.ready or registry.h_simulator is None:
        raise HTTPException(503, "Models not loaded yet")

    if req.circuit not in registry.circuit_defaults:
        available = sorted(registry.circuit_defaults.keys())
        raise HTTPException(400, f"Unknown circuit '{req.circuit}'. Available: {available}")

    driver_abbrevs = {d.driver for d in req.drivers}
    if req.target_driver not in driver_abbrevs:
        raise HTTPException(
            400,
            f"Target driver '{req.target_driver}' not in grid. "
            f"Available: {sorted(driver_abbrevs)}",
        )

    drivers = [
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

    circuit_info = registry.circuit_defaults[req.circuit]
    candidates = generate_candidates(
        circuit_info,
        pit_lap_delta=req.pit_lap_delta,
        max_candidates=req.max_candidates,
    )

    opt_result = optimize_strategy(
        registry.h_simulator,
        req.circuit,
        drivers,
        req.target_driver,
        candidates,
        use_monte_carlo=req.use_monte_carlo,
        n_simulations=req.n_simulations,
    )

    strategy_options = []
    for rank, sr in enumerate(opt_result.results, 1):
        legs = [
            StrategyLeg(compound=compound, pit_on_lap=pit_lap)
            for compound, pit_lap in sr.candidate.strategy
        ]
        strategy_options.append(
            StrategyOption(
                rank=rank,
                strategy=legs,
                description=sr.candidate.description,
                predicted_position=sr.position,
                predicted_time=sr.total_time,
                gap_to_leader=sr.gap_to_leader,
                position_mean=sr.position_mean,
                position_std=sr.position_std,
            )
        )

    return OptimizeStrategyResponse(
        circuit=req.circuit,
        target_driver=req.target_driver,
        n_candidates_tested=opt_result.n_candidates,
        strategies=strategy_options,
    )
