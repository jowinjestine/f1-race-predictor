"""Autoregressive race simulation engine for Model F."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from f1_predictor.features.race_features import HYBRID_CIRCUITS, STREET_CIRCUITS
from f1_predictor.features.simulation_features import SIMULATION_FEATURE_COLS
from f1_predictor.simulation.defaults import get_default_strategy


@dataclass
class DriverState:
    """Mutable state for one driver during simulation."""

    driver: str
    grid_position: int
    best_quali_sec: float
    compound: str
    tire_life: int = 1
    stint: int = 1
    pit_stop_count: int = 0
    laps_since_last_pit: int = 0
    cum_time: float = 0.0
    position: int = 0
    lap_times: list[float] = field(default_factory=list)
    stint_times: list[float] = field(default_factory=list)
    stint_lives: list[int] = field(default_factory=list)

    # Pit strategy: list of (compound, pit_on_lap | None)
    strategy: list[tuple[str, int | None]] = field(default_factory=list)
    current_stint_idx: int = 0


@dataclass
class LapRecord:
    """Record for one driver at one lap."""

    lap: int
    driver: str
    position: int
    lap_time: float
    cum_time: float
    gap_to_leader: float
    compound: str
    tire_life: int
    stint: int


@dataclass
class SimulationResult:
    """Complete simulation output."""

    circuit: str
    total_laps: int
    lap_records: list[LapRecord]
    final_results: list[dict[str, Any]]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert lap records to a DataFrame."""
        rows = []
        for rec in self.lap_records:
            rows.append({
                "lap_number": rec.lap,
                "driver": rec.driver,
                "position": rec.position,
                "lap_time": rec.lap_time,
                "cum_time": rec.cum_time,
                "gap_to_leader": rec.gap_to_leader,
                "compound": rec.compound,
                "tire_life": rec.tire_life,
                "stint": rec.stint,
            })
        return pd.DataFrame(rows)


# Prediction clamps (lap_time_ratio bounds)
CLAMP_NORMAL = (1.01, 1.15)
CLAMP_PIT_IN = (1.10, 1.50)
CLAMP_PIT_OUT = (1.03, 1.25)


class RaceSimulator:
    """Autoregressive race simulator using a Model F lap-time-ratio predictor."""

    def __init__(
        self,
        model: Any,
        circuit_defaults: dict[str, dict[str, Any]],
    ) -> None:
        self.model = model
        self.circuit_defaults = circuit_defaults

    def simulate(
        self,
        circuit: str,
        drivers: list[dict[str, Any]],
        strategies: dict[str, list[tuple[str, int | None]]] | None = None,
    ) -> SimulationResult:
        """Run a full race simulation.

        Args:
            circuit: Circuit name (e.g., "Monza", "Monaco").
            drivers: List of dicts with keys:
                driver, grid_position, q1, q2, q3, initial_tyre
            strategies: Optional dict mapping driver -> pit strategy.
                Each strategy is a list of (compound, pit_on_lap) tuples.
                If not provided, uses circuit-default strategy.

        Returns:
            SimulationResult with lap-by-lap records and final standings.
        """
        if circuit not in self.circuit_defaults:
            raise ValueError(
                f"Unknown circuit '{circuit}'. "
                f"Available: {sorted(self.circuit_defaults.keys())}"
            )

        cinfo = self.circuit_defaults[circuit]
        total_laps = cinfo["total_laps"]
        strategies = strategies or {}

        # Circuit type features
        is_street = int(circuit in STREET_CIRCUITS)
        is_hybrid = int(circuit in HYBRID_CIRCUITS)
        is_permanent = int(not is_street and not is_hybrid)

        # Initialize driver states
        states: list[DriverState] = []
        for d in sorted(drivers, key=lambda x: x["grid_position"]):
            best_q = min(
                v for v in [d.get("q1"), d.get("q2"), d.get("q3")]
                if v is not None and not np.isnan(v)
            )
            drv = DriverState(
                driver=d["driver"],
                grid_position=d["grid_position"],
                best_quali_sec=best_q,
                compound=d.get("initial_tyre", "MEDIUM"),
                position=d["grid_position"],
            )
            if d["driver"] in strategies:
                drv.strategy = strategies[d["driver"]]
            else:
                drv.strategy = get_default_strategy(cinfo, drv.compound)
            states.append(drv)

        all_records: list[LapRecord] = []

        for lap in range(1, total_laps + 1):
            # Check for pit-in on this lap (before prediction)
            pit_in_drivers = set()
            pit_out_drivers = set()
            for s in states:
                if s.current_stint_idx < len(s.strategy):
                    _, pit_lap = s.strategy[s.current_stint_idx]
                    if pit_lap is not None and lap == pit_lap:
                        pit_in_drivers.add(s.driver)

            # Check for pit-out (pitted on previous lap)
            for s in states:
                if s.laps_since_last_pit == 0 and s.pit_stop_count > 0 and lap > 1:
                    pit_out_drivers.add(s.driver)

            # Build feature matrix for all drivers
            features = self._build_features(
                lap, total_laps, states,
                is_street, is_hybrid, is_permanent,
                pit_in_drivers, pit_out_drivers,
            )

            # Predict lap time ratios
            ratios = self.model.predict(features)

            # Clamp predictions
            for i, s in enumerate(states):
                if s.driver in pit_in_drivers:
                    ratios[i] = np.clip(ratios[i], *CLAMP_PIT_IN)
                elif s.driver in pit_out_drivers:
                    ratios[i] = np.clip(ratios[i], *CLAMP_PIT_OUT)
                else:
                    ratios[i] = np.clip(ratios[i], *CLAMP_NORMAL)

            # Convert to lap times and update state
            for i, s in enumerate(states):
                lap_time = ratios[i] * s.best_quali_sec
                s.cum_time += lap_time
                s.lap_times.append(lap_time)
                s.stint_times.append(lap_time)
                s.stint_lives.append(s.tire_life)
                s.tire_life += 1
                s.laps_since_last_pit += 1

            # Derive positions from cumulative times
            sorted_states = sorted(states, key=lambda x: x.cum_time)
            leader_time = sorted_states[0].cum_time
            for pos_idx, s in enumerate(sorted_states):
                s.position = pos_idx + 1

            # Record results
            for s in states:
                all_records.append(LapRecord(
                    lap=lap,
                    driver=s.driver,
                    position=s.position,
                    lap_time=s.lap_times[-1],
                    cum_time=s.cum_time,
                    gap_to_leader=s.cum_time - leader_time,
                    compound=s.compound,
                    tire_life=s.tire_life - 1,
                    stint=s.stint,
                ))

            # Handle pit stops (execute after recording lap)
            for s in states:
                if s.driver in pit_in_drivers:
                    s.current_stint_idx += 1
                    s.pit_stop_count += 1
                    s.laps_since_last_pit = 0
                    s.stint += 1
                    s.stint_times = []
                    s.stint_lives = []
                    s.tire_life = 1
                    if s.current_stint_idx < len(s.strategy):
                        s.compound = s.strategy[s.current_stint_idx][0]

        # Build final results
        final_sorted = sorted(states, key=lambda x: x.cum_time)
        final_results = []
        for pos, s in enumerate(final_sorted, 1):
            final_results.append({
                "driver": s.driver,
                "position": pos,
                "total_time": s.cum_time,
                "gap_to_leader": s.cum_time - final_sorted[0].cum_time,
                "pit_stops": s.pit_stop_count,
            })

        return SimulationResult(
            circuit=circuit,
            total_laps=total_laps,
            lap_records=all_records,
            final_results=final_results,
        )

    def _build_features(
        self,
        lap: int,
        total_laps: int,
        states: list[DriverState],
        is_street: int,
        is_hybrid: int,
        is_permanent: int,
        pit_in_drivers: set[str],
        pit_out_drivers: set[str],
    ) -> pd.DataFrame:
        """Build the feature matrix for all drivers at a given lap."""
        rows = []
        for s in states:
            # Rolling pace (EWMA-style)
            rolling_3 = self._ewma(s.lap_times, 3, alpha=0.5)
            rolling_5 = self._ewma(s.lap_times, 5, alpha=0.4)

            # Degradation rate: OLS of stint lap times vs tire_life
            deg_rate = self._compute_deg_rate(s.stint_times, s.stint_lives)

            # Gap to leader
            all_cum = [st.cum_time for st in states]
            leader_cum = min(all_cum) if all_cum else 0.0
            gap_to_leader = s.cum_time - leader_cum

            # Position change from grid
            pos_change = s.grid_position - s.position

            # Compound one-hot
            compounds = {
                "compound_HARD": int(s.compound == "HARD"),
                "compound_INTERMEDIATE": int(s.compound == "INTERMEDIATE"),
                "compound_MEDIUM": int(s.compound == "MEDIUM"),
                "compound_SOFT": int(s.compound == "SOFT"),
                "compound_WET": int(s.compound == "WET"),
            }

            row = {
                "grid_position": s.grid_position,
                "best_quali_sec": s.best_quali_sec,
                "circuit_street": is_street,
                "circuit_hybrid": is_hybrid,
                "circuit_permanent": is_permanent,
                "lap_number": lap,
                "race_progress_pct": lap / total_laps,
                **compounds,
                "tire_life": s.tire_life,
                "stint": s.stint,
                "is_pit_in_lap": int(s.driver in pit_in_drivers),
                "is_pit_out_lap": int(s.driver in pit_out_drivers),
                "pit_stop_count": s.pit_stop_count,
                "laps_since_last_pit": s.laps_since_last_pit,
                "lap_time_rolling_3": rolling_3,
                "lap_time_rolling_5": rolling_5,
                "degradation_rate": deg_rate,
                "gap_to_leader": gap_to_leader,
                "position": s.position,
                "position_change_from_lap1": pos_change,
                "is_caution": 0,
            }
            rows.append(row)

        return pd.DataFrame(rows)[SIMULATION_FEATURE_COLS]

    @staticmethod
    def _ewma(values: list[float], window: int, alpha: float) -> float:
        """Exponentially weighted moving average over the last `window` values."""
        if not values:
            return np.nan
        recent = values[-window:]
        if len(recent) == 1:
            return recent[0]
        weights = np.array([(1 - alpha) ** i for i in range(len(recent) - 1, -1, -1)])
        return float(np.average(recent, weights=weights))

    @staticmethod
    def _compute_deg_rate(stint_times: list[float], stint_lives: list[int]) -> float:
        """OLS slope of lap times vs tire life within current stint."""
        if len(stint_times) < 3:
            return np.nan
        times = np.array(stint_times, dtype=np.float64)
        lives = np.array(stint_lives, dtype=np.float64)
        valid = np.isfinite(times) & np.isfinite(lives)
        if valid.sum() < 3:
            return np.nan
        coeffs = np.polyfit(lives[valid], times[valid], 1)
        return float(coeffs[0])
