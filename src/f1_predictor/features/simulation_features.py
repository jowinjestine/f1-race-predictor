"""Feature engineering for Model F: lap time simulation training data."""

from __future__ import annotations

import pandas as pd  # noqa: TC002

from f1_predictor.features.common import encode_compound_onehot
from f1_predictor.features.lap_features import (
    _add_gap_features,
    _add_pit_features,
    _add_race_progress,
    _add_rolling_pace,
    _compute_degradation_rate,
)
from f1_predictor.features.race_features import (
    HYBRID_CIRCUITS,
    LOCATION_ALIASES,
    STREET_CIRCUITS,
)

RACE_KEY = ["season", "round"]
DRIVER_RACE_KEY = ["season", "round", "driver_abbrev"]


def _normalise_location(loc: str) -> str:
    return LOCATION_ALIASES.get(loc, loc)


def build_simulation_training_data(
    laps: pd.DataFrame,
    races: pd.DataFrame,
) -> pd.DataFrame:
    """Build training data for Model F (lap time ratio prediction).

    Merges lap-level data with qualifying times and circuit info.
    Target: lap_time_ratio = lap_time_sec / best_quali_sec.
    """
    df = laps.copy()
    df = df[df["season"].between(2019, 2024)].copy()
    df = df.dropna(subset=["tire_compound", "lap_time_sec"])
    df = df.sort_values([*DRIVER_RACE_KEY, "lap_number"]).reset_index(drop=True)

    # --- Merge qualifying times from race data ---
    races = races.copy()
    races["location_norm"] = races["location"].map(_normalise_location)
    quali_cols = ["season", "round", "driver_abbrev", "q1_time_sec", "q2_time_sec", "q3_time_sec"]
    quali = races[[c for c in quali_cols if c in races.columns]].copy()
    quali["best_quali_sec"] = quali[["q1_time_sec", "q2_time_sec", "q3_time_sec"]].min(axis=1)
    df = df.merge(
        quali[["season", "round", "driver_abbrev", "best_quali_sec"]],
        on=DRIVER_RACE_KEY,
        how="left",
    )

    # Merge grid position
    if "grid_position" in races.columns:
        grid = races[["season", "round", "driver_abbrev", "grid_position"]].copy()
        df = df.merge(grid, on=DRIVER_RACE_KEY, how="left")

    # --- Compute target ---
    df["lap_time_ratio"] = df["lap_time_sec"] / df["best_quali_sec"]

    # Drop rows without qualifying data or with extreme outliers
    df = df.dropna(subset=["lap_time_ratio", "best_quali_sec"])
    df = df[(df["lap_time_ratio"] >= 0.95) & (df["lap_time_ratio"] <= 1.6)]

    # --- Caution flag from track status ---
    if "track_status" in df.columns:
        caution_codes = {"4", "5", "6", "7", "41", "45", "46", "47", "56", "67"}
        df["is_caution"] = df["track_status"].astype(str).isin(caution_codes).astype(int)
    else:
        df["is_caution"] = 0

    # --- Circuit type features ---
    loc_map = races[["season", "round", "location"]].drop_duplicates()
    loc_map["location_norm"] = loc_map["location"].map(_normalise_location)
    df = df.merge(loc_map[["season", "round", "location_norm"]], on=RACE_KEY, how="left")
    df["circuit_street"] = df["location_norm"].isin(STREET_CIRCUITS).astype(int)
    df["circuit_hybrid"] = df["location_norm"].isin(HYBRID_CIRCUITS).astype(int)
    df["circuit_permanent"] = ((df["circuit_street"] == 0) & (df["circuit_hybrid"] == 0)).astype(
        int
    )

    # --- Lap dynamics features (reuse from lap_features.py) ---
    df = _add_rolling_pace(df)
    df = _add_gap_features(df)
    df = _add_pit_features(df)
    df = _add_race_progress(df)

    # --- Tyre features ---
    onehot = encode_compound_onehot(df["tire_compound"])
    for col in onehot.columns:
        df[col] = onehot[col].values
    df["degradation_rate"] = _compute_degradation_rate(df)

    # --- Position change from lap 1 ---
    lap1_pos = (
        df[df["lap_number"] == 1].set_index(DRIVER_RACE_KEY)["position"].rename("lap1_position")
    )
    df = df.join(lap1_pos, on=DRIVER_RACE_KEY)
    df["position_change_from_lap1"] = df["lap1_position"] - df["position"]
    df = df.drop(columns=["lap1_position"], errors="ignore")

    # --- Select output columns ---
    id_cols = ["season", "round", "event_name", "driver_abbrev", "team"]

    feature_cols = [
        # Static
        "grid_position",
        "best_quali_sec",
        "circuit_street",
        "circuit_hybrid",
        "circuit_permanent",
        # Deterministic dynamic
        "lap_number",
        "race_progress_pct",
        "compound_HARD",
        "compound_INTERMEDIATE",
        "compound_MEDIUM",
        "compound_SOFT",
        "compound_WET",
        "tire_life",
        "stint",
        "is_pit_in_lap",
        "is_pit_out_lap",
        "pit_stop_count",
        "laps_since_last_pit",
        # Feedback dynamic
        "lap_time_rolling_3",
        "lap_time_rolling_5",
        "degradation_rate",
        "gap_to_leader",
        "position",
        "position_change_from_lap1",
        # Context
        "is_caution",
    ]

    target_col = ["lap_time_ratio"]

    output_cols = id_cols + feature_cols + target_col
    present = [c for c in output_cols if c in df.columns]
    result = df[present].copy()

    # Drop rows where target is NaN
    result = result.dropna(subset=["lap_time_ratio"]).reset_index(drop=True)

    return result


SIMULATION_FEATURE_COLS = [
    "grid_position",
    "best_quali_sec",
    "circuit_street",
    "circuit_hybrid",
    "circuit_permanent",
    "lap_number",
    "race_progress_pct",
    "compound_HARD",
    "compound_INTERMEDIATE",
    "compound_MEDIUM",
    "compound_SOFT",
    "compound_WET",
    "tire_life",
    "stint",
    "is_pit_in_lap",
    "is_pit_out_lap",
    "pit_stop_count",
    "laps_since_last_pit",
    "lap_time_rolling_3",
    "lap_time_rolling_5",
    "degradation_rate",
    "gap_to_leader",
    "position",
    "position_change_from_lap1",
    "is_caution",
]
