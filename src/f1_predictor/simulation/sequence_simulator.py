# ruff: noqa: N806  — numpy convention uses uppercase X_seq, X_list, W
"""Sequence-aware race simulator for Model G.

Maintains a per-driver sliding window of feature vectors so that
sequence models (GRU, LSTM, TCN, Transformer, etc.) can consume
temporal context during autoregressive simulation.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from f1_predictor.features.simulation_features import SIMULATION_FEATURE_COLS
from f1_predictor.simulation.engine import (
    CLAMP_NORMAL,
    CLAMP_PIT_IN,
    CLAMP_PIT_OUT,
    DriverState,
    LapRecord,
    RaceSimulator,
    SimulationResult,
)


class SequenceRaceSimulator(RaceSimulator):
    """Simulator that feeds a sliding window to a sequence model."""

    def __init__(
        self,
        model: Any,
        circuit_defaults: dict[str, dict[str, Any]],
        window_size: int = 5,
    ) -> None:
        super().__init__(model, circuit_defaults)
        self.window_size = window_size

    def simulate(
        self,
        circuit: str,
        drivers: list[dict[str, Any]],
        strategies: (dict[str, list[tuple[str, int | None]]] | None) = None,
        *,
        dnf_probs: dict[str, float] | None = None,
        rng: np.random.RandomState | None = None,
    ) -> SimulationResult:
        from f1_predictor.features.race_features import (
            HYBRID_CIRCUITS,
            STREET_CIRCUITS,
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

        # Initialise driver states
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

        # Per-driver sliding window buffers
        n_features = len(SIMULATION_FEATURE_COLS)
        buffers: dict[str, deque[np.ndarray]] = {
            st.driver: deque(maxlen=self.window_size) for st in states
        }

        lap_records: list[LapRecord] = []

        for lap in range(1, total_laps + 1):
            # Determine pit events
            pit_in_drivers: set[str] = set()
            pit_out_drivers: set[str] = set()
            for st in states:
                if st.current_stint_idx < len(st.strategy):
                    _, pit_lap = st.strategy[st.current_stint_idx]
                    if pit_lap is not None and lap == pit_lap:
                        pit_in_drivers.add(st.driver)
                if st.laps_since_last_pit == 0 and st.pit_stop_count > 0 and lap > 1:
                    pit_out_drivers.add(st.driver)

            # Build tabular features for this lap
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
            feature_rows = features_df[SIMULATION_FEATURE_COLS].values.astype(np.float64)

            # Append to per-driver buffers and build 3D input
            for i, st in enumerate(states):
                buffers[st.driver].append(feature_rows[i])

            # Stack into (n_drivers, window, n_features + 1)
            X_seq = self._build_sequence_input(
                states,
                buffers,
                n_features,
            )

            # Predict
            ratios = self.model.predict(X_seq)

            # Clamp
            for i, st in enumerate(states):
                if st.driver in pit_in_drivers:
                    ratios[i] = np.clip(ratios[i], *CLAMP_PIT_IN)
                elif st.driver in pit_out_drivers:
                    ratios[i] = np.clip(ratios[i], *CLAMP_PIT_OUT)
                else:
                    ratios[i] = np.clip(ratios[i], *CLAMP_NORMAL)

            # Update state
            for i, st in enumerate(states):
                lt = float(ratios[i]) * st.best_quali_sec
                st.cum_time += lt
                st.lap_times.append(lt)
                st.stint_times.append(lt)
                st.stint_lives.append(st.tire_life)
                st.tire_life += 1
                st.laps_since_last_pit += 1

            # Rank by cumulative time
            sorted_states = sorted(states, key=lambda s: s.cum_time)
            leader_time = sorted_states[0].cum_time
            for pos_idx, st in enumerate(sorted_states):
                st.position = pos_idx + 1

            for st in states:
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
            for st in states:
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

        # Build final results
        final_sorted = sorted(states, key=lambda s: s.cum_time)
        leader_time = final_sorted[0].cum_time
        final_results = []
        for pos_idx, st in enumerate(final_sorted):
            final_results.append(
                {
                    "driver": st.driver,
                    "position": pos_idx + 1,
                    "total_time": st.cum_time,
                    "gap_to_leader": st.cum_time - leader_time,
                    "pit_stops": st.pit_stop_count,
                }
            )

        return SimulationResult(
            circuit=circuit,
            total_laps=total_laps,
            lap_records=lap_records,
            final_results=final_results,
        )

    def _build_sequence_input(
        self,
        states: list[DriverState],
        buffers: dict[str, deque[np.ndarray]],
        n_features: int,
    ) -> np.ndarray:
        """Build (n_drivers, window_size, n_features+1) array.

        Left-pads with zeros for early laps and appends a
        seq_valid_mask channel (0=pad, 1=real).
        """
        W = self.window_size
        X_list = []
        for st in states:
            buf = list(buffers[st.driver])
            n_valid = len(buf)
            if n_valid < W:
                pad = [np.zeros(n_features)] * (W - n_valid)
                buf = pad + buf
            window = np.stack(buf)  # (W, n_features)

            mask = np.zeros((W, 1))
            mask[-n_valid:] = 1.0
            window_with_mask = np.hstack([window, mask])
            X_list.append(window_with_mask)

        return np.array(X_list, dtype=np.float64)
