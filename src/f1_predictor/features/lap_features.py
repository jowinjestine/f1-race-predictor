"""Lap-level feature engineering for Models A and B."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1_predictor.features.common import (
    encode_compound_onehot,
    rolling_mean_by_group,
)

RACE_KEY = ["season", "round"]
DRIVER_RACE_KEY = ["season", "round", "driver_abbrev"]


def build_lap_tyre_features(laps: pd.DataFrame) -> pd.DataFrame:
    """Build lap-level features WITH tyre data (Model A, 2019-2024)."""
    df = laps[laps["season"].between(2019, 2024)].copy()
    df = df.dropna(subset=["tire_compound"])
    return _build_lap_features(df, include_tyre=True)


def build_lap_notyre_features(laps: pd.DataFrame) -> pd.DataFrame:
    """Build lap-level features WITHOUT tyre data (Model B, 2018-2025)."""
    df = laps.copy()
    return _build_lap_features(df, include_tyre=False)


def _build_lap_features(df: pd.DataFrame, *, include_tyre: bool) -> pd.DataFrame:
    """Core feature builder shared by Model A and Model B."""
    df = df.sort_values([*DRIVER_RACE_KEY, "lap_number"]).reset_index(drop=True)

    df = _add_race_normalization(df)
    df = _add_rolling_pace(df)
    df = _add_position_features(df)
    df = _add_gap_features(df)
    df = _add_pit_features(df)
    df = _add_race_progress(df)

    if include_tyre:
        df = _add_tyre_features(df)

    id_cols = ["season", "round", "event_name", "driver_abbrev", "team", "lap_number"]
    shared_feature_cols = [
        "lap_time_delta_race_median",
        "lap_time_rolling_3",
        "lap_time_rolling_5",
        "position_change_from_lap1",
        "gap_to_leader",
        "gap_to_ahead",
        "laps_since_last_pit",
        "pit_stop_count",
        "race_progress_pct",
        "is_pit_in_lap",
        "is_pit_out_lap",
    ]
    tyre_cols = [
        "compound_HARD",
        "compound_INTERMEDIATE",
        "compound_MEDIUM",
        "compound_SOFT",
        "compound_WET",
        "tire_life",
        "stint",
        "degradation_rate",
        "compound_pace_delta",
    ]
    target_col = ["position"]

    feature_cols = shared_feature_cols + (tyre_cols if include_tyre else [])
    output_cols = id_cols + feature_cols + target_col
    present = [c for c in output_cols if c in df.columns]
    return df[present].copy()


def _add_race_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Add lap time delta from race median."""
    race_median = df.groupby(RACE_KEY)["lap_time_sec"].transform("median")
    df["lap_time_delta_race_median"] = df["lap_time_sec"] - race_median
    return df


def _add_rolling_pace(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling average lap times (3-lap and 5-lap)."""
    df["lap_time_rolling_3"] = rolling_mean_by_group(df, DRIVER_RACE_KEY, "lap_time_sec", window=3)
    df["lap_time_rolling_5"] = rolling_mean_by_group(df, DRIVER_RACE_KEY, "lap_time_sec", window=5)
    return df


def _add_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add position change from lap 1."""
    lap1_pos = (
        df[df["lap_number"] == 1]
        .set_index(["season", "round", "driver_abbrev"])["position"]
        .rename("lap1_position")
    )
    df = df.join(lap1_pos, on=["season", "round", "driver_abbrev"])
    df["position_change_from_lap1"] = df["lap1_position"] - df["position"]
    df = df.drop(columns=["lap1_position"])
    return df


def _add_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add gap to leader and gap to car ahead."""
    df["cum_time"] = df.groupby(DRIVER_RACE_KEY)["lap_time_sec"].cumsum()

    leader_time = df.groupby([*RACE_KEY, "lap_number"])["cum_time"].min().rename("leader_cum_time")
    df = df.join(leader_time, on=[*RACE_KEY, "lap_number"])
    df["gap_to_leader"] = df["cum_time"] - df["leader_cum_time"]

    sorted_lap = df.sort_values([*RACE_KEY, "lap_number", "position"])
    ahead_time = sorted_lap.groupby([*RACE_KEY, "lap_number"])["cum_time"].shift(1)
    df.loc[sorted_lap.index, "ahead_cum_time"] = ahead_time.values
    df["gap_to_ahead"] = df["cum_time"] - df["ahead_cum_time"]
    df.loc[df["position"] == 1, "gap_to_ahead"] = 0.0

    df = df.drop(columns=["cum_time", "leader_cum_time", "ahead_cum_time"])
    return df


def _add_pit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add laps since last pit and pit stop count."""
    df["is_pit_in_lap"] = df["is_pit_in_lap"].astype(int)
    df["is_pit_out_lap"] = df["is_pit_out_lap"].astype(int)

    pit_counts: list[int] = []
    laps_since: list[int] = []
    prev_key = ("", 0, "")
    count = 0
    since = 0

    for _, row in df.iterrows():
        key = (row["season"], row["round"], row["driver_abbrev"])
        if key != prev_key:
            count = 0
            since = 0
            prev_key = key
        else:
            since += 1

        if row["is_pit_out_lap"] == 1:
            since = 0
            count += 1

        pit_counts.append(count)
        laps_since.append(since)

    df["pit_stop_count"] = pit_counts
    df["laps_since_last_pit"] = laps_since
    return df


def _add_race_progress(df: pd.DataFrame) -> pd.DataFrame:
    """Add race progress as fraction of total laps."""
    max_laps = df.groupby(RACE_KEY)["lap_number"].transform("max")
    df["race_progress_pct"] = df["lap_number"] / max_laps
    return df


def _add_tyre_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add tyre-specific features (Model A only)."""
    onehot = encode_compound_onehot(df["tire_compound"])
    for col in onehot.columns:
        df[col] = onehot[col].values

    df["degradation_rate"] = _compute_degradation_rate(df)
    df["compound_pace_delta"] = _compute_compound_pace_delta(df)
    return df


def _compute_degradation_rate(df: pd.DataFrame) -> pd.Series:
    """OLS slope of lap_time_sec over tire_life within each stint."""
    stint_key = ["season", "round", "driver_abbrev", "stint"]
    deg = pd.Series(0.0, index=df.index)

    for _, group in df.groupby(stint_key, sort=False):
        times = group["lap_time_sec"].values
        lives = group["tire_life"].values
        for i in range(3, len(times)):
            valid = np.isfinite(times[:i]) & np.isfinite(lives[:i])
            if valid.sum() >= 3:
                coeffs = np.polyfit(lives[:i][valid], times[:i][valid], 1)
                deg.iloc[group.index[i]] = coeffs[0]

    return deg


def _compute_compound_pace_delta(df: pd.DataFrame) -> pd.Series:
    """Expanding median of compound lap time minus race median, shifted."""
    race_median = df.groupby(RACE_KEY)["lap_time_sec"].transform("median")
    compound_key = ["season", "round", "tire_compound"]

    compound_median = (
        df.groupby(compound_key, sort=False)["lap_time_sec"]
        .expanding()
        .median()
        .droplevel(list(range(len(compound_key))))
        .sort_index()
    )
    shifted = compound_median.groupby(df[compound_key].apply(tuple, axis=1), sort=False).shift(1)

    return shifted - race_median
