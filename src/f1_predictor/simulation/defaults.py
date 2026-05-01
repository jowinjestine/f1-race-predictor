"""Circuit metadata and default pit strategies computed from historical data."""

from __future__ import annotations

from typing import Any

import pandas as pd  # noqa: TC002


def build_circuit_defaults(laps: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Compute per-circuit defaults from historical lap data.

    Returns dict keyed by event_name with:
        total_laps: int
        typical_stops: int
        pit_windows: list[int]  (median pit laps)
        common_sequence: list[str]  (most common compound order)
    """
    from f1_predictor.features.race_features import LOCATION_ALIASES

    df = laps.copy()
    if "event_name" not in df.columns:
        return {}

    df["event_norm"] = df["event_name"].map(lambda x: LOCATION_ALIASES.get(x, x))

    defaults: dict[str, dict[str, Any]] = {}

    for event, grp in df.groupby("event_norm"):
        race_laps = grp.groupby(["season", "round"])["lap_number"].max()
        total_laps = int(race_laps.median())

        # Pit stop count: median number of pit-in laps per driver per race
        if "is_pit_in_lap" in grp.columns:
            pit_counts = (
                grp[grp["is_pit_in_lap"] == True]  # noqa: E712
                .groupby(["season", "round", "driver_abbrev"])
                .size()
            )
            all_drivers = grp.groupby(["season", "round", "driver_abbrev"]).ngroups
            if len(pit_counts) > 0:
                all_keys = grp.groupby(["season", "round", "driver_abbrev"]).size()
                pit_counts = pit_counts.reindex(all_keys.index, fill_value=0)
                typical_stops = int(pit_counts.median())
            elif all_drivers > 0:
                typical_stops = 1
            else:
                typical_stops = 1
        else:
            typical_stops = 1

        # Pit window: median lap numbers where pits happen
        pit_laps_data = grp[grp["is_pit_in_lap"] == True]  # noqa: E712
        pit_windows: list[int] = []
        if len(pit_laps_data) > 0:
            for stint_num in range(1, typical_stops + 1):
                stint_pits = pit_laps_data.groupby(["season", "round", "driver_abbrev"]).nth(
                    stint_num - 1
                )
                if len(stint_pits) > 0 and "lap_number" in stint_pits.columns:
                    pit_windows.append(int(stint_pits["lap_number"].median()))
                elif len(stint_pits) > 0:
                    pit_windows.append(int(total_laps * stint_num / (typical_stops + 1)))

        if not pit_windows and typical_stops > 0:
            for i in range(1, typical_stops + 1):
                pit_windows.append(int(total_laps * i / (typical_stops + 1)))

        # Most common compound sequence
        common_sequence = _get_common_compound_sequence(grp, typical_stops)

        defaults[str(event)] = {
            "total_laps": total_laps,
            "typical_stops": typical_stops,
            "pit_windows": pit_windows,
            "common_sequence": common_sequence,
        }

    return defaults


def _get_common_compound_sequence(race_laps: pd.DataFrame, n_stops: int) -> list[str]:
    """Find the most common compound sequence for a circuit."""
    if "tire_compound" not in race_laps.columns:
        return ["MEDIUM", "HARD"] if n_stops <= 1 else ["SOFT", "MEDIUM", "HARD"]

    sequences: list[tuple[str, ...]] = []
    for (_, _, _), drv_grp in race_laps.groupby(["season", "round", "driver_abbrev"]):
        compounds = (
            drv_grp.sort_values("lap_number").groupby("stint")["tire_compound"].first().tolist()
        )
        compounds = [c for c in compounds if c is not None]
        if len(compounds) >= 2:
            sequences.append(tuple(compounds))

    if not sequences:
        return ["MEDIUM", "HARD"] if n_stops <= 1 else ["SOFT", "MEDIUM", "HARD"]

    from collections import Counter

    most_common = Counter(sequences).most_common(1)[0][0]
    return list(most_common)


def build_field_median_curves(
    laps: pd.DataFrame,
    races: pd.DataFrame,
) -> dict[str, dict[int, float]]:
    """Compute per-circuit median lap_time_ratio at each lap number.

    Convenience re-export from delta_features. Use at simulation time
    alongside build_circuit_defaults.
    """
    from f1_predictor.features.delta_features import build_field_median_curves as _build

    return _build(laps, races)


def get_default_strategy(
    circuit_defaults: dict[str, Any],
    initial_tyre: str,
) -> list[tuple[str, int | None]]:
    """Build a default pit strategy from circuit defaults.

    Returns list of (compound, pit_on_lap) tuples.
    The last entry has pit_on_lap=None (run to end).
    """
    pit_windows = circuit_defaults["pit_windows"]
    common_seq = circuit_defaults["common_sequence"]

    strategy: list[tuple[str, int | None]] = []

    if not pit_windows:
        strategy.append((initial_tyre, None))
        return strategy

    # Use historical compound sequence, starting with user's initial tyre
    compounds = list(common_seq)
    if compounds and compounds[0] != initial_tyre:
        compounds[0] = initial_tyre

    # Ensure we have enough compounds for all stints
    while len(compounds) < len(pit_windows) + 1:
        compounds.append("HARD" if initial_tyre != "HARD" else "MEDIUM")

    for i, pit_lap in enumerate(pit_windows):
        strategy.append((compounds[i], pit_lap))
    strategy.append((compounds[len(pit_windows)], None))

    return strategy
