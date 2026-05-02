"""Pit strategy generation and optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any

from f1_predictor.simulation.defaults import get_default_strategy
from f1_predictor.simulation.delta_simulator import DeltaRaceSimulator, MonteCarloSimulator

DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
MIN_STINT_DEFAULT = 8


@dataclass
class CandidateStrategy:
    strategy: list[tuple[str, int | None]]
    description: str


@dataclass
class StrategyResult:
    candidate: CandidateStrategy
    position: int
    total_time: float
    gap_to_leader: float
    position_mean: float | None = None
    position_std: float | None = None


@dataclass
class OptimizationResult:
    target_driver: str
    circuit: str
    n_candidates: int
    results: list[StrategyResult] = field(default_factory=list)


def _describe_strategy(strat: list[tuple[str, int | None]], total_laps: int) -> str:
    """Human-readable strategy description like '2-stop: SOFT(1-20) HARD(21-45) MEDIUM(46-57)'."""
    n_stops = sum(1 for _, pit in strat if pit is not None)
    parts = []
    start = 1
    for compound, pit_lap in strat:
        end = pit_lap if pit_lap is not None else total_laps
        parts.append(f"{compound}({start}-{end})")
        start = end + 1
    return f"{n_stops}-stop: {' '.join(parts)}"


def generate_candidates(
    circuit_defaults: dict[str, Any],
    *,
    compounds: list[str] | None = None,
    min_stint: int = MIN_STINT_DEFAULT,
    max_stint: int | None = None,
    pit_lap_delta: int = 5,
    max_candidates: int = 30,
) -> list[CandidateStrategy]:
    """Generate candidate pit strategies from circuit defaults.

    Varies pit lap timing, stop count, and compound sequences.
    """
    compounds = compounds or DRY_COMPOUNDS
    total_laps = circuit_defaults["total_laps"]
    typical_stops = circuit_defaults["typical_stops"]
    pit_windows = circuit_defaults["pit_windows"]

    if max_stint is None:
        max_stint = total_laps - min_stint

    candidates: list[CandidateStrategy] = []
    seen: set[tuple[tuple[str, int | None], ...]] = set()

    def _add(strat: list[tuple[str, int | None]]) -> None:
        key = tuple(strat)
        if key in seen:
            return
        if not _is_valid(strat, total_laps, min_stint, max_stint):
            return
        seen.add(key)
        candidates.append(
            CandidateStrategy(
                strategy=strat,
                description=_describe_strategy(strat, total_laps),
            )
        )

    # 1-stop candidates
    for c1, c2 in permutations(compounds, 2):
        for delta in range(-pit_lap_delta, pit_lap_delta + 1, 2):
            base_pit = pit_windows[0] if pit_windows else total_laps // 2
            pit_lap = base_pit + delta
            _add([(c1, pit_lap), (c2, None)])

    # 2-stop candidates
    if typical_stops >= 2 or len(pit_windows) >= 2:
        base_p1 = pit_windows[0] if len(pit_windows) >= 1 else total_laps // 3
        base_p2 = pit_windows[1] if len(pit_windows) >= 2 else 2 * total_laps // 3
        for seq in permutations(compounds, 3):
            for d1 in range(-pit_lap_delta, pit_lap_delta + 1, 3):
                for d2 in range(-pit_lap_delta, pit_lap_delta + 1, 3):
                    p1 = base_p1 + d1
                    p2 = base_p2 + d2
                    if p1 < p2:
                        _add([(seq[0], p1), (seq[1], p2), (seq[2], None)])

    # Default strategy as baseline
    default = get_default_strategy(circuit_defaults, "MEDIUM")
    _add(default)

    if len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]

    return candidates


def _is_valid(
    strat: list[tuple[str, int | None]],
    total_laps: int,
    min_stint: int,
    max_stint: int,
) -> bool:
    """Check that all stints respect length constraints and use 2+ compounds."""
    if len(strat) < 2:
        return False

    unique_compounds = {c for c, _ in strat}
    if len(unique_compounds) < 2:
        return False

    start = 1
    for _, pit_lap in strat:
        end = pit_lap if pit_lap is not None else total_laps
        stint_len = end - start + 1
        if stint_len < min_stint or stint_len > max_stint:
            return False
        start = end + 1

    return True


def optimize_strategy(
    simulator: DeltaRaceSimulator,
    circuit: str,
    drivers: list[dict[str, Any]],
    target_driver: str,
    candidates: list[CandidateStrategy],
    *,
    use_monte_carlo: bool = False,
    n_simulations: int = 50,
    mc_top_n: int = 5,
) -> OptimizationResult:
    """Evaluate candidate strategies for a target driver.

    Phase 1: Deterministic simulation for all candidates.
    Phase 2 (optional): Monte Carlo on top mc_top_n candidates.
    """
    phase1_results: list[StrategyResult] = []

    for candidate in candidates:
        strategies = {target_driver: candidate.strategy}
        result = simulator.simulate(circuit, drivers, strategies)

        target_fr = next(
            (fr for fr in result.final_results if fr["driver"] == target_driver),
            None,
        )
        if target_fr is None:
            continue

        phase1_results.append(
            StrategyResult(
                candidate=candidate,
                position=target_fr["position"],
                total_time=target_fr["total_time"],
                gap_to_leader=target_fr["gap_to_leader"],
            )
        )

    phase1_results.sort(key=lambda r: (r.position, r.total_time))

    if not use_monte_carlo:
        return OptimizationResult(
            target_driver=target_driver,
            circuit=circuit,
            n_candidates=len(candidates),
            results=phase1_results,
        )

    # Phase 2: Monte Carlo on top N
    top_results = phase1_results[:mc_top_n]
    mc_results: list[StrategyResult] = []

    for sr in top_results:
        strategies = {target_driver: sr.candidate.strategy}
        mc = MonteCarloSimulator(simulator, n_simulations=n_simulations)
        mc_result = mc.simulate(circuit, drivers, strategies)

        target_mc = next(
            (r for r in mc_result.results if r["driver"] == target_driver),
            None,
        )
        if target_mc is None:
            continue

        mc_results.append(
            StrategyResult(
                candidate=sr.candidate,
                position=target_mc["position"],
                total_time=sr.total_time,
                gap_to_leader=sr.gap_to_leader,
                position_mean=target_mc["position_mean"],
                position_std=target_mc["position_std"],
            )
        )

    mc_results.sort(key=lambda r: (r.position_mean or r.position, r.total_time))

    remaining = [r for r in phase1_results[mc_top_n:]]
    return OptimizationResult(
        target_driver=target_driver,
        circuit=circuit,
        n_candidates=len(candidates),
        results=mc_results + remaining,
    )
