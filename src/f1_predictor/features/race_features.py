"""Pre-race feature engineering for Model C."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1_predictor.features.common import (
    expanding_count_by_group,
    expanding_mean_by_group,
    expanding_sum_by_group,
    rolling_mean_by_group,
    rolling_sum_by_group,
    safe_divide,
)

RACE_KEY = ["season", "round"]

LOCATION_ALIASES: dict[str, str] = {
    "Monte Carlo": "Monaco",
    "Marina Bay": "Singapore",
    "Spa-Francorchamps": "Spa",
    "Yas Marina": "Abu Dhabi",
    "Yas Island": "Abu Dhabi",
    "Montréal": "Montreal",
    "São Paulo": "Sao Paulo",
    "Nürburgring": "Nurburgring",
    "Portimão": "Portimao",
}

STREET_CIRCUITS: frozenset[str] = frozenset({
    "Monaco", "Singapore", "Baku", "Jeddah", "Las Vegas",
})
HYBRID_CIRCUITS: frozenset[str] = frozenset({
    "Melbourne", "Miami", "Montreal",
})


def build_race_features(races: pd.DataFrame) -> pd.DataFrame:
    """Build pre-race features for Model C (2018-2025)."""
    df = races.copy()
    df = df.sort_values([*RACE_KEY, "driver_abbrev"]).reset_index(drop=True)
    df["location_norm"] = df["location"].replace(LOCATION_ALIASES)
    df["is_dnf_int"] = df["is_dnf"].astype(int)
    df["is_podium_int"] = df["is_podium"].astype(int)

    df = _add_qualifying_features(df)
    df = _add_season_form_features(df)
    df = _add_driver_circuit_features(df)
    df = _add_team_form_features(df)
    df = _add_circuit_features(df)
    df = _add_weather_features(df)

    id_cols = ["season", "round", "event_name", "driver_abbrev", "team"]
    feature_cols = [
        "best_quali_sec",
        "quali_delta_to_pole",
        "grid_position",
        "quali_position_vs_teammate",
        "avg_finish_last_3",
        "avg_finish_last_5",
        "points_last_3",
        "points_cumulative_season",
        "dnf_rate_season",
        "position_trend",
        "driver_circuit_avg_finish",
        "driver_circuit_races",
        "driver_circuit_podium_rate",
        "driver_circuit_dnf_rate",
        "team_avg_finish_last_3",
        "team_points_cumulative_season",
        "circuit_street",
        "circuit_permanent",
        "circuit_hybrid",
        "circuit_avg_dnf_rate",
        "weather_temp_max",
        "weather_precip_mm",
        "weather_wind_max_kph",
        "is_wet_race",
    ]
    target_cols = ["finish_position", "is_podium", "is_points_finish", "is_dnf"]
    output_cols = id_cols + feature_cols + target_cols
    present = [c for c in output_cols if c in df.columns]
    return df[present].copy()


def _add_qualifying_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add qualifying-derived features (not leakage — qualifying is pre-race)."""
    df["best_quali_sec"] = df[["q1_time_sec", "q2_time_sec", "q3_time_sec"]].min(
        axis=1
    )

    pole_time = df.groupby(RACE_KEY)["best_quali_sec"].transform("min")
    df["quali_delta_to_pole"] = df["best_quali_sec"] - pole_time

    team_grid_sum = df.groupby([*RACE_KEY, "team"])["grid_position"].transform("sum")
    team_count = df.groupby([*RACE_KEY, "team"])["grid_position"].transform("count")
    teammate_grid = safe_divide(
        team_grid_sum - df["grid_position"],
        team_count - 1,
    )
    df["quali_position_vs_teammate"] = df["grid_position"] - teammate_grid
    return df


def _add_season_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add season form features (all shifted via common helpers)."""
    driver_key = ["driver_abbrev"]
    season_driver_key = ["season", "driver_abbrev"]

    df["avg_finish_last_3"] = rolling_mean_by_group(
        df, driver_key, "finish_position", window=3
    )
    df["avg_finish_last_5"] = rolling_mean_by_group(
        df, driver_key, "finish_position", window=5
    )
    df["points_last_3"] = rolling_sum_by_group(
        df, driver_key, "points", window=3
    )
    df["points_cumulative_season"] = expanding_sum_by_group(
        df, season_driver_key, "points"
    )
    df["dnf_rate_season"] = expanding_mean_by_group(
        df, season_driver_key, "is_dnf_int"
    )
    df["position_trend"] = _compute_position_trend(df)
    return df


def _compute_position_trend(df: pd.DataFrame) -> pd.Series:
    """OLS slope of finish_position over last 5 races per driver."""
    trend = pd.Series(np.nan, index=df.index)
    for _, group in df.groupby("driver_abbrev", sort=False):
        positions = group["finish_position"].values
        for i in range(len(positions)):
            start = max(0, i - 5)
            window = positions[start:i]
            valid = window[np.isfinite(window)]
            if len(valid) >= 3:
                x = np.arange(len(valid), dtype=float)
                coeffs = np.polyfit(x, valid, 1)
                trend.loc[group.index[i]] = coeffs[0]
    return trend


def _add_driver_circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add driver circuit history features (expanding, shifted)."""
    dc_key = ["driver_abbrev", "location_norm"]

    df["driver_circuit_avg_finish"] = expanding_mean_by_group(
        df, dc_key, "finish_position"
    )
    df["driver_circuit_races"] = expanding_count_by_group(
        df, dc_key, "finish_position"
    )
    df["driver_circuit_podium_rate"] = expanding_mean_by_group(
        df, dc_key, "is_podium_int"
    )
    df["driver_circuit_dnf_rate"] = expanding_mean_by_group(
        df, dc_key, "is_dnf_int"
    )
    return df


def _add_team_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add team form features (aggregated at team-race level)."""
    team_race = (
        df.groupby([*RACE_KEY, "team"])
        .agg(
            team_finish=("finish_position", "mean"),
            team_points=("points", "sum"),
        )
        .reset_index()
    )
    team_race = team_race.sort_values([*RACE_KEY]).reset_index(drop=True)

    team_race["team_avg_finish_last_3"] = rolling_mean_by_group(
        team_race, ["team"], "team_finish", window=3
    )
    team_race["team_points_cumulative_season"] = expanding_sum_by_group(
        team_race, ["season", "team"], "team_points"
    )

    df = df.merge(
        team_race[
            [*RACE_KEY, "team", "team_avg_finish_last_3", "team_points_cumulative_season"]
        ],
        on=[*RACE_KEY, "team"],
        how="left",
    )
    return df


def _add_circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add circuit type and historical DNF rate."""
    df["circuit_street"] = df["location_norm"].isin(STREET_CIRCUITS).astype(int)
    df["circuit_hybrid"] = df["location_norm"].isin(HYBRID_CIRCUITS).astype(int)
    df["circuit_permanent"] = (
        ~df["location_norm"].isin(STREET_CIRCUITS | HYBRID_CIRCUITS)
    ).astype(int)

    race_dnf = (
        df.groupby([*RACE_KEY, "location_norm"])["is_dnf_int"]
        .mean()
        .reset_index()
        .rename(columns={"is_dnf_int": "_race_dnf_rate"})
    )
    race_dnf = race_dnf.sort_values([*RACE_KEY]).reset_index(drop=True)
    race_dnf["circuit_avg_dnf_rate"] = expanding_mean_by_group(
        race_dnf, ["location_norm"], "_race_dnf_rate"
    )

    df = df.merge(
        race_dnf[[*RACE_KEY, "location_norm", "circuit_avg_dnf_rate"]],
        on=[*RACE_KEY, "location_norm"],
        how="left",
    )
    return df


def _add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weather features with circuit-mean imputation for nulls."""
    circuit_temp = df.groupby("location_norm")["weather_temp_max"].transform("mean")
    df["weather_temp_max"] = df["weather_temp_max"].fillna(circuit_temp)
    df["weather_temp_max"] = df["weather_temp_max"].fillna(df["weather_temp_max"].mean())

    df["weather_precip_mm"] = df["weather_precip_mm"].fillna(0.0)

    circuit_wind = df.groupby("location_norm")["weather_wind_max_kph"].transform("mean")
    df["weather_wind_max_kph"] = df["weather_wind_max_kph"].fillna(circuit_wind)
    df["weather_wind_max_kph"] = df["weather_wind_max_kph"].fillna(
        df["weather_wind_max_kph"].mean()
    )

    is_rainfall = df["f1_rainfall"].eq(True).fillna(False)
    df["is_wet_race"] = (is_rainfall | (df["weather_precip_mm"] > 1.0)).astype(int)
    return df
