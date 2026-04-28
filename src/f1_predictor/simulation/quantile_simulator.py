# ruff: noqa: N806  — numpy convention uses uppercase X_seq, W
"""Quantile-sampling Monte Carlo race simulator for Model I.

Samples from the predicted quantile distribution at each lap
to produce diverse but calibrated simulation trajectories.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from f1_predictor.simulation.engine import (
    CLAMP_NORMAL,
    CLAMP_PIT_IN,
    CLAMP_PIT_OUT,
    DriverState,
    RaceSimulator,
)


@dataclass
class QuantileMCResult:
    """Aggregated quantile-based Monte Carlo simulation output."""

    circuit: str
    n_simulations: int
    results: list[dict[str, Any]]


class QuantileRaceSimulator(RaceSimulator):  # type: ignore[override]
    """Monte Carlo simulator that samples from predicted quantiles.

    At each lap for each driver, draws a random quantile level
    and interpolates between predicted quantile values to get
    the sampled lap_time_ratio.
    """

    def __init__(
        self,
        model: Any,
        circuit_defaults: dict[str, dict[str, Any]],
        n_simulations: int = 200,
        seed: int = 42,
    ) -> None:
        super().__init__(model, circuit_defaults)
        self.n_simulations = n_simulations
        self.rng = np.random.RandomState(seed)

    def simulate(
        self,
        circuit: str,
        drivers: list[dict[str, Any]],
        strategies: (dict[str, list[tuple[str, int | None]]] | None) = None,
    ) -> QuantileMCResult:
        from f1_predictor.features.race_features import (
            HYBRID_CIRCUITS,
            STREET_CIRCUITS,
        )
        from f1_predictor.features.simulation_features import (
            SIMULATION_FEATURE_COLS,
        )
        from f1_predictor.simulation.defaults import get_default_strategy

        if circuit not in self.circuit_defaults:
            msg = f"Unknown circuit: {circuit}. Available: {list(self.circuit_defaults.keys())}"
            raise ValueError(msg)

        info = self.circuit_defaults[circuit]
        total_laps = info["total_laps"]
        is_street = int(circuit in STREET_CIRCUITS)
        is_hybrid = int(circuit in HYBRID_CIRCUITS)
        is_permanent = int(not is_street and not is_hybrid)

        all_positions: dict[str, list[int]] = defaultdict(list)

        for _sim_i in range(self.n_simulations):
            # Initialise fresh driver states each run
            states: list[DriverState] = []
            for drv in sorted(
                drivers,
                key=lambda d: d["grid_position"],
            ):
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
                        compound=(strat[0][0] if strat else initial_tyre),
                        position=drv["grid_position"],
                        strategy=strat,
                    )
                )

            for lap in range(1, total_laps + 1):
                pit_in_drivers: set[str] = set()
                pit_out_drivers: set[str] = set()
                for st in states:
                    if st.current_stint_idx < len(st.strategy):
                        _, pit_lap = st.strategy[st.current_stint_idx]
                        if pit_lap is not None and lap == pit_lap:
                            pit_in_drivers.add(st.driver)
                    if st.laps_since_last_pit == 0 and st.pit_stop_count > 0 and lap > 1:
                        pit_out_drivers.add(st.driver)

                features_df = self._build_features(
                    lap,
                    total_laps,
                    states,
                    is_street,
                    is_hybrid,
                    is_permanent,
                    pit_in_drivers,
                    pit_out_drivers,
                )

                # Get quantile predictions
                X_feat = features_df[SIMULATION_FEATURE_COLS]
                quantiles = self.model.predict_quantiles(X_feat)
                # quantiles: (n_drivers, 5) for q10, q25, q50, q75, q90

                # Sample random quantile level per driver
                u = self.rng.uniform(0, 1, size=len(states))
                ratios = _interpolate_quantiles(quantiles, u)

                # Clamp
                for i, st in enumerate(states):
                    if st.driver in pit_in_drivers:
                        ratios[i] = np.clip(
                            ratios[i],
                            *CLAMP_PIT_IN,
                        )
                    elif st.driver in pit_out_drivers:
                        ratios[i] = np.clip(
                            ratios[i],
                            *CLAMP_PIT_OUT,
                        )
                    else:
                        ratios[i] = np.clip(
                            ratios[i],
                            *CLAMP_NORMAL,
                        )

                # Update state
                for i, st in enumerate(states):
                    lt = float(ratios[i]) * st.best_quali_sec
                    st.cum_time += lt
                    st.lap_times.append(lt)
                    st.stint_times.append(lt)
                    st.stint_lives.append(st.tire_life)
                    st.tire_life += 1
                    st.laps_since_last_pit += 1

                # Rank
                sorted_states = sorted(
                    states,
                    key=lambda s: s.cum_time,
                )
                for pos_idx, st in enumerate(sorted_states):
                    st.position = pos_idx + 1

                # Pit stops
                for st in states:
                    if st.driver in pit_in_drivers:
                        st.pit_stop_count += 1
                        st.tire_life = 1
                        st.stint += 1
                        st.stint_times = []
                        st.stint_lives = []
                        st.laps_since_last_pit = 0
                        st.current_stint_idx += 1
                        if st.current_stint_idx < len(
                            st.strategy,
                        ):
                            st.compound = st.strategy[st.current_stint_idx][0]

            # Record final positions for this run
            final_sorted = sorted(
                states,
                key=lambda s: s.cum_time,
            )
            for pos_idx, st in enumerate(final_sorted):
                all_positions[st.driver].append(pos_idx + 1)

        # Aggregate across all runs
        results = []
        for driver, positions in all_positions.items():
            arr = np.array(positions)
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
                }
            )

        results.sort(key=lambda r: r["position"])  # type: ignore[arg-type, return-value]

        return QuantileMCResult(
            circuit=circuit,
            n_simulations=self.n_simulations,
            results=results,
        )


def _interpolate_quantiles(
    quantiles: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate between quantile predictions.

    Args:
        quantiles: (n, 5) array for q10, q25, q50, q75, q90
        u: (n,) uniform samples in [0, 1]

    Returns:
        (n,) interpolated values
    """
    q_levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    n = len(u)
    result = np.zeros(n)
    for i in range(n):
        result[i] = np.interp(u[i], q_levels, quantiles[i])
    return result
