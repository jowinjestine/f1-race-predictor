"""Feature engineering for Model H: delta lap-time-ratio target."""

from __future__ import annotations

from typing import TYPE_CHECKING

from f1_predictor.features.race_features import LOCATION_ALIASES

if TYPE_CHECKING:
    import pandas as pd


def _normalise_location(loc: str) -> str:
    return LOCATION_ALIASES.get(loc, loc)


def build_field_median_curves(
    laps: pd.DataFrame,
    races: pd.DataFrame,
) -> dict[str, dict[int, float]]:
    """Compute per-circuit median lap_time_ratio at each lap number.

    Returns:
        {"Monza": {1: 1.085, 2: 1.062, ...}, ...}
    """
    df = laps.copy()
    df = df[df["season"].between(2019, 2024)].copy()
    df = df.dropna(subset=["lap_time_sec", "tire_compound"])

    # Merge qualifying times
    races_c = races.copy()
    quali_cols = ["season", "round", "driver_abbrev", "q1_time_sec", "q2_time_sec", "q3_time_sec"]
    quali = races_c[[c for c in quali_cols if c in races_c.columns]].copy()
    quali["best_quali_sec"] = quali[["q1_time_sec", "q2_time_sec", "q3_time_sec"]].min(axis=1)
    df = df.merge(
        quali[["season", "round", "driver_abbrev", "best_quali_sec"]],
        on=["season", "round", "driver_abbrev"],
        how="left",
    )

    df["lap_time_ratio"] = df["lap_time_sec"] / df["best_quali_sec"]
    df = df.dropna(subset=["lap_time_ratio"])
    df = df[(df["lap_time_ratio"] >= 0.95) & (df["lap_time_ratio"] <= 1.6)]

    # Normalise circuit names
    loc_map = races_c[["season", "round", "event_name"]].drop_duplicates()
    loc_map["circuit"] = loc_map["event_name"].map(_normalise_location)
    df = df.merge(loc_map[["season", "round", "circuit"]], on=["season", "round"], how="left")

    curves: dict[str, dict[int, float]] = {}
    for circuit, grp in df.groupby("circuit"):
        lap_medians = grp.groupby("lap_number")["lap_time_ratio"].median()
        curves[str(circuit)] = {int(k): float(v) for k, v in lap_medians.items()}  # type: ignore[call-overload]

    return curves


def build_delta_training_data(
    df: pd.DataFrame,
    field_medians: dict[str, dict[int, float]],
    races: pd.DataFrame,
) -> pd.DataFrame:
    """Add field_median_ratio and delta_ratio columns to simulation training data.

    Args:
        df: Output of build_simulation_training_data() with lap_time_ratio target.
        field_medians: Output of build_field_median_curves().
        races: Race DataFrame for circuit name lookup.

    Returns:
        DataFrame with original columns plus field_median_ratio, delta_ratio.
    """
    result = df.copy()

    # Map circuit names
    loc_map = races[["season", "round", "event_name"]].drop_duplicates()
    loc_map["circuit"] = loc_map["event_name"].map(_normalise_location)

    if "circuit" not in result.columns:
        result = result.merge(
            loc_map[["season", "round", "circuit"]], on=["season", "round"], how="left"
        )

    # Look up field median for each row
    def _lookup_median(row: pd.Series) -> float:
        circuit = row.get("circuit", "")
        lap = int(row.get("lap_number", 0))
        curve = field_medians.get(str(circuit), {})
        if lap in curve:
            return curve[lap]
        # Fallback: closest lap or global median
        if curve:
            closest = min(curve.keys(), key=lambda k: abs(k - lap))
            return curve[closest]
        return 1.05  # global fallback

    result["field_median_ratio"] = result.apply(_lookup_median, axis=1)
    result["delta_ratio"] = result["lap_time_ratio"] - result["field_median_ratio"]

    return result
