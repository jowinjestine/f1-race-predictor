# ruff: noqa: N803  — sklearn convention uses uppercase X
"""Delta-target race simulator and Monte Carlo wrapper for Model H."""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from f1_predictor.simulation.engine import RaceSimulator, SimulationResult


@dataclass
class MonteCarloResult:
    """Aggregated Monte Carlo simulation output."""

    circuit: str
    n_simulations: int
    results: list[dict[str, Any]]


class NoisyModelWrapper:
    """Wraps a model and adds Gaussian noise to predictions."""

    def __init__(
        self,
        model: Any,
        noise_std: float,
        rng: np.random.RandomState,
    ) -> None:
        self.model = model
        self.noise_std = noise_std
        self.rng = rng

    def predict(self, X: Any) -> Any:
        preds = self.model.predict(X)
        noise = self.rng.normal(0, self.noise_std, size=preds.shape)
        return preds + noise


class DeltaRaceSimulator(RaceSimulator):
    """Simulator that predicts delta_ratio, adds back field_median baseline."""

    def __init__(
        self,
        model: Any,
        circuit_defaults: dict[str, dict[str, Any]],
        field_medians: dict[str, dict[int, float]],
    ) -> None:
        super().__init__(model, circuit_defaults)
        self.field_medians = field_medians
        self._last_pit_in: set[str] = set()

    def _get_baseline(self, circuit: str, lap: int) -> float:
        """Look up the historical median ratio for this circuit and lap."""
        curve = self.field_medians.get(circuit, {})
        if lap in curve:
            return curve[lap]
        if curve:
            closest = min(curve.keys(), key=lambda k: abs(k - lap))
            return curve[closest]
        return 1.05

    def simulate(
        self,
        circuit: str,
        drivers: list[dict[str, Any]],
        strategies: dict[str, list[tuple[str, int | None]]] | None = None,
        *,
        dnf_probs: dict[str, float] | None = None,
        rng: np.random.RandomState | None = None,
    ) -> SimulationResult:
        """Run simulation with delta prediction + baseline reconstruction.

        Args:
            dnf_probs: Per-driver race-level DNF probability (0.0 to 1.0).
                Converted to per-lap hazard rate internally.
            rng: Random state for DNF sampling (deterministic if None and
                no DNF probs provided).
        """
        from f1_predictor.features.race_features import HYBRID_CIRCUITS, STREET_CIRCUITS
        from f1_predictor.features.simulation_features import SIMULATION_FEATURE_COLS
        from f1_predictor.simulation.defaults import get_default_strategy
        from f1_predictor.simulation.engine import DriverState, LapRecord

        if circuit not in self.circuit_defaults:
            msg = f"Unknown circuit: {circuit}. Available: {list(self.circuit_defaults.keys())}"
            raise ValueError(msg)

        info = self.circuit_defaults[circuit]
        total_laps = info["total_laps"]
        is_street = circuit in STREET_CIRCUITS
        is_hybrid = circuit in HYBRID_CIRCUITS
        is_permanent = not is_street and not is_hybrid

        # Compute per-lap hazard rates from race-level DNF probabilities
        hazard_rates: dict[str, float] = {}
        if dnf_probs:
            for drv_name, p in dnf_probs.items():
                if p > 0:
                    hazard_rates[drv_name] = 1.0 - (1.0 - p) ** (1.0 / total_laps)

        if rng is None and hazard_rates:
            rng = np.random.RandomState(42)

        # Initialize driver states
        states: list[DriverState] = []
        for drv in sorted(drivers, key=lambda d: d["grid_position"]):
            q_times = [drv.get(f"q{i}") for i in (1, 2, 3)]
            q_valid = [t for t in q_times if t is not None]
            best_q = min(q_valid) if q_valid else 90.0

            initial_tyre = drv.get("initial_tyre", "MEDIUM")
            strat = None
            if strategies and drv["driver"] in strategies:
                strat = strategies[drv["driver"]]
            if strat is None:
                strat = get_default_strategy(info, initial_tyre)

            states.append(
                DriverState(
                    driver=drv["driver"],
                    grid_position=drv["grid_position"],
                    best_quali_sec=best_q,
                    compound=strat[0][0] if strat else initial_tyre,
                    position=drv["grid_position"],
                    strategy=strat,
                )
            )

        lap_records: list[LapRecord] = []

        for lap in range(1, total_laps + 1):
            active_states = [s for s in states if not s.is_retired]
            if not active_states:
                break

            pit_in_drivers = set()
            pit_out_drivers = set()
            for st in active_states:
                if st.current_stint_idx < len(st.strategy):
                    _, pit_lap = st.strategy[st.current_stint_idx]
                    if pit_lap is not None and lap == pit_lap:
                        pit_in_drivers.add(st.driver)
                if lap > 1 and st.driver in getattr(self, "_last_pit_in", set()):
                    pit_out_drivers.add(st.driver)

            features_df = self._build_features(
                lap,
                total_laps,
                active_states,
                is_street,
                is_hybrid,
                is_permanent,
                pit_in_drivers,
                pit_out_drivers,
            )

            # Predict delta and add baseline
            deltas = self.model.predict(features_df[SIMULATION_FEATURE_COLS])
            baseline = self._get_baseline(circuit, lap)
            ratios = deltas + baseline

            # Clamp
            for i, st in enumerate(active_states):
                if st.driver in pit_in_drivers:
                    ratios[i] = np.clip(ratios[i], 1.10, 1.50)
                elif st.driver in pit_out_drivers:
                    ratios[i] = np.clip(ratios[i], 1.03, 1.25)
                else:
                    ratios[i] = np.clip(ratios[i], 1.01, 1.15)

            # Update state
            for i, st in enumerate(active_states):
                lt = float(ratios[i]) * st.best_quali_sec
                st.cum_time += lt
                st.lap_times.append(lt)
                st.stint_times.append(lt)
                st.stint_lives.append(st.tire_life)
                st.tire_life += 1
                st.laps_since_last_pit += 1

            # Rank active drivers by cumulative time
            sorted_active = sorted(active_states, key=lambda s: s.cum_time)
            leader_time = sorted_active[0].cum_time
            for pos_idx, st in enumerate(sorted_active):
                st.position = pos_idx + 1

            for st in active_states:
                lap_records.append(
                    LapRecord(
                        lap=lap,
                        driver=st.driver,
                        position=st.position,
                        lap_time=st.lap_times[-1],
                        cum_time=st.cum_time,
                        gap_to_leader=st.cum_time - leader_time,
                        compound=st.compound,
                        tire_life=st.tire_life - 1,
                        stint=st.stint,
                    )
                )

            # Handle pit stops
            self._last_pit_in = pit_in_drivers
            for st in active_states:
                if st.driver in pit_in_drivers:
                    st.pit_stop_count += 1
                    st.tire_life = 1
                    st.stint += 1
                    st.stint_times = []
                    st.stint_lives = []
                    st.laps_since_last_pit = 0
                    st.current_stint_idx += 1
                    if st.current_stint_idx < len(st.strategy):
                        st.compound = st.strategy[st.current_stint_idx][0]

            # DNF sampling: retire drivers probabilistically after lap processing
            if rng is not None and hazard_rates:
                for st in active_states:
                    h = hazard_rates.get(st.driver, 0.0)
                    if h > 0 and rng.random() < h:
                        st.is_retired = True
                        st.retired_on_lap = lap
                        st.retirement_status = "DNF"

        # Build final results — finishers by cum_time, then DNFs by retirement lap
        finishers = [s for s in states if not s.is_retired]
        retired = [s for s in states if s.is_retired]
        finishers.sort(key=lambda s: s.cum_time)
        retired.sort(key=lambda s: -(s.retired_on_lap or 0))

        final_results = []
        leader_time = finishers[0].cum_time if finishers else 0.0
        for pos_idx, st in enumerate(finishers + retired):
            final_results.append(
                {
                    "driver": st.driver,
                    "position": pos_idx + 1,
                    "total_time": st.cum_time,
                    "gap_to_leader": st.cum_time - leader_time,
                    "pit_stops": st.pit_stop_count,
                    "status": st.retirement_status,
                    "laps_completed": st.retired_on_lap or total_laps,
                }
            )

        return SimulationResult(
            circuit=circuit,
            total_laps=total_laps,
            lap_records=lap_records,
            final_results=final_results,
        )


class MonteCarloSimulator:
    """Runs N simulations with noise injection and aggregates results."""

    def __init__(
        self,
        base_simulator: RaceSimulator,
        n_simulations: int = 200,
        noise_std: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.base_simulator = base_simulator
        self.n_simulations = n_simulations
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

    def simulate(
        self,
        circuit: str,
        drivers: list[dict[str, Any]],
        strategies: dict[str, list[tuple[str, int | None]]] | None = None,
        *,
        dnf_probs: dict[str, float] | None = None,
    ) -> MonteCarloResult:
        all_positions: dict[str, list[int]] = defaultdict(list)
        all_statuses: dict[str, list[str]] = defaultdict(list)

        for _ in range(self.n_simulations):
            noisy_model = NoisyModelWrapper(self.base_simulator.model, self.noise_std, self.rng)
            sim = _copy_simulator_with_model(self.base_simulator, noisy_model)
            mc_rng = np.random.RandomState(self.rng.randint(0, 2**31))
            result = sim.simulate(
                circuit, drivers, strategies, dnf_probs=dnf_probs, rng=mc_rng
            )
            for fr in result.final_results:
                all_positions[fr["driver"]].append(fr["position"])
                all_statuses[fr["driver"]].append(fr.get("status", "Finished"))

        results = []
        for driver, positions in all_positions.items():
            arr = np.array(positions)
            statuses = all_statuses[driver]
            dnf_count = sum(1 for s in statuses if s == "DNF")
            results.append(
                {
                    "driver": driver,
                    "position": int(np.median(arr)),
                    "position_mean": float(np.mean(arr)),
                    "position_p10": int(np.percentile(arr, 10)),
                    "position_p25": int(np.percentile(arr, 25)),
                    "position_p75": int(np.percentile(arr, 75)),
                    "position_p90": int(np.percentile(arr, 90)),
                    "position_std": float(np.std(arr)),
                    "dnf_rate": dnf_count / len(statuses),
                }
            )

        results.sort(key=lambda r: r["position"])  # type: ignore[arg-type, return-value]

        return MonteCarloResult(
            circuit=circuit,
            n_simulations=self.n_simulations,
            results=results,
        )


def _copy_simulator_with_model(
    simulator: RaceSimulator,
    new_model: Any,
) -> RaceSimulator:
    """Create a copy of a simulator with a different model."""
    sim = copy.copy(simulator)
    sim.model = new_model
    return sim
