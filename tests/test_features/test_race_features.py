"""Tests for pre-race feature engineering (Model C)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1_predictor.features.race_features import (
    LOCATION_ALIASES,
    build_race_features,
)


def _make_races(
    n_seasons: int = 2,
    rounds_per_season: int = 3,
    drivers_per_round: int = 4,
    start_season: int = 2022,
) -> pd.DataFrame:
    """Create minimal race data for testing."""
    teams = ["TeamA", "TeamA", "TeamB", "TeamB"]
    locations = ["Silverstone", "Monaco", "Monza"]
    rows = []
    for s in range(n_seasons):
        season = start_season + s
        for r in range(1, rounds_per_season + 1):
            for d in range(drivers_per_round):
                driver = f"DR{d}"
                finish = d + 1
                rows.append(
                    {
                        "season": season,
                        "round": r,
                        "event_name": f"Race {r}",
                        "location": locations[(r - 1) % len(locations)],
                        "country": "UK",
                        "event_date": f"{season}-03-{r:02d}",
                        "driver_number": str(d),
                        "driver_abbrev": driver,
                        "driver_id": f"driver{d}",
                        "first_name": f"First{d}",
                        "last_name": f"Last{d}",
                        "team": teams[d],
                        "team_id": f"team{d // 2}",
                        "finish_position": float(finish),
                        "grid_position": float(finish),
                        "status": "Finished",
                        "points": max(0.0, 10.0 - d * 3.0),
                        "laps_completed": 50,
                        "is_classified": True,
                        "q1_time_sec": 90.0 + d * 0.5,
                        "q2_time_sec": 89.0 + d * 0.5 if d < 3 else np.nan,
                        "q3_time_sec": 88.0 + d * 0.5 if d < 2 else np.nan,
                        "race_time_sec": 5400.0 + d * 10.0,
                        "f1_air_temp_mean": 25.0,
                        "f1_track_temp_mean": 35.0,
                        "f1_humidity_mean": 50.0,
                        "f1_pressure_mean": 1013.0,
                        "f1_wind_speed_mean": 10.0,
                        "f1_rainfall": False,
                        "weather_temp_max": 28.0,
                        "weather_temp_min": 18.0,
                        "weather_precip_mm": 0.0,
                        "weather_wind_max_kph": 20.0,
                        "is_podium": finish <= 3,
                        "is_points_finish": finish <= 10,
                        "is_dnf": False,
                    }
                )
    return pd.DataFrame(rows)


class TestBuildRaceFeatures:
    def test_output_has_all_feature_columns(self) -> None:
        races = _make_races()
        result = build_race_features(races)
        expected = [
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
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_output_has_target_columns(self) -> None:
        races = _make_races()
        result = build_race_features(races)
        for col in ["finish_position", "is_podium", "is_points_finish", "is_dnf"]:
            assert col in result.columns

    def test_output_has_id_columns(self) -> None:
        races = _make_races()
        result = build_race_features(races)
        for col in ["season", "round", "event_name", "driver_abbrev", "team"]:
            assert col in result.columns

    def test_row_count_preserved(self) -> None:
        races = _make_races()
        result = build_race_features(races)
        assert len(result) == len(races)


class TestQualifyingFeatures:
    def test_best_quali_is_min_of_q1_q2_q3(self) -> None:
        races = _make_races(n_seasons=1, rounds_per_season=1, drivers_per_round=1)
        races["q1_time_sec"] = 92.0
        races["q2_time_sec"] = 91.0
        races["q3_time_sec"] = 90.0
        result = build_race_features(races)
        assert result["best_quali_sec"].iloc[0] == pytest.approx(90.0)

    def test_quali_delta_to_pole_for_pole_is_zero(self) -> None:
        races = _make_races(n_seasons=1, rounds_per_season=1)
        result = build_race_features(races)
        pole = result.loc[result["grid_position"] == 1.0]
        assert (pole["quali_delta_to_pole"] == 0.0).all()

    def test_teammate_comparison(self) -> None:
        races = _make_races(n_seasons=1, rounds_per_season=1, drivers_per_round=2)
        races.loc[races["driver_abbrev"] == "DR0", "grid_position"] = 3.0
        races.loc[races["driver_abbrev"] == "DR1", "grid_position"] = 7.0
        result = build_race_features(races)
        dr0 = result[result["driver_abbrev"] == "DR0"]
        assert dr0["quali_position_vs_teammate"].iloc[0] == pytest.approx(-4.0)


class TestSeasonFormFeatures:
    def test_avg_finish_last_3_is_nan_for_first_race(self) -> None:
        races = _make_races()
        result = build_race_features(races)
        first_race = result[(result["season"] == 2022) & (result["round"] == 1)]
        assert first_race["avg_finish_last_3"].isna().all()

    def test_points_cumulative_season_first_race_is_nan(self) -> None:
        races = _make_races()
        result = build_race_features(races)
        first_race = result[(result["season"] == 2022) & (result["round"] == 1)]
        assert first_race["points_cumulative_season"].isna().all()

    def test_dnf_rate_season_first_race_is_nan(self) -> None:
        races = _make_races()
        result = build_race_features(races)
        first_race = result[(result["season"] == 2022) & (result["round"] == 1)]
        assert first_race["dnf_rate_season"].isna().all()

    def test_position_trend_needs_3_races(self) -> None:
        races = _make_races(n_seasons=1, rounds_per_season=2, drivers_per_round=1)
        result = build_race_features(races)
        assert result["position_trend"].isna().all()


class TestDriverCircuitFeatures:
    def test_first_visit_is_nan(self) -> None:
        races = _make_races(n_seasons=1, rounds_per_season=1)
        result = build_race_features(races)
        assert result["driver_circuit_avg_finish"].isna().all()
        assert result["driver_circuit_races"].isna().all()

    def test_second_visit_uses_first(self) -> None:
        races = _make_races(n_seasons=2, rounds_per_season=1, drivers_per_round=1)
        result = build_race_features(races)
        second_visit = result[result["season"] == 2023]
        assert second_visit["driver_circuit_avg_finish"].notna().all()
        assert second_visit["driver_circuit_races"].iloc[0] == pytest.approx(1.0)


class TestTeamFormFeatures:
    def test_team_avg_finish_first_race_is_nan(self) -> None:
        races = _make_races(n_seasons=1, rounds_per_season=1)
        result = build_race_features(races)
        assert result["team_avg_finish_last_3"].isna().all()

    def test_team_points_cumulative_first_race_is_nan(self) -> None:
        races = _make_races(n_seasons=1, rounds_per_season=1)
        result = build_race_features(races)
        assert result["team_points_cumulative_season"].isna().all()


class TestCircuitFeatures:
    def test_monaco_is_street(self) -> None:
        races = _make_races()
        result = build_race_features(races)
        monaco = result[result["event_name"].str.contains("Race 2")]
        assert (monaco["circuit_street"] == 1).all()
        assert (monaco["circuit_permanent"] == 0).all()

    def test_silverstone_is_permanent(self) -> None:
        races = _make_races()
        result = build_race_features(races)
        silverstone = result[result["event_name"].str.contains("Race 1")]
        assert (silverstone["circuit_permanent"] == 1).all()
        assert (silverstone["circuit_street"] == 0).all()

    def test_circuit_type_mutually_exclusive(self) -> None:
        races = _make_races()
        result = build_race_features(races)
        type_sum = result["circuit_street"] + result["circuit_permanent"] + result["circuit_hybrid"]
        assert (type_sum == 1).all()

    def test_location_aliases_normalize(self) -> None:
        assert LOCATION_ALIASES["Monte Carlo"] == "Monaco"
        assert LOCATION_ALIASES["Marina Bay"] == "Singapore"
        assert LOCATION_ALIASES["Spa-Francorchamps"] == "Spa"


class TestWeatherFeatures:
    def test_is_wet_race_from_rainfall(self) -> None:
        races = _make_races(n_seasons=1, rounds_per_season=1, drivers_per_round=1)
        races["f1_rainfall"] = True
        races["weather_precip_mm"] = 0.0
        result = build_race_features(races)
        assert result["is_wet_race"].iloc[0] == 1

    def test_is_wet_race_from_precip(self) -> None:
        races = _make_races(n_seasons=1, rounds_per_season=1, drivers_per_round=1)
        races["f1_rainfall"] = False
        races["weather_precip_mm"] = 5.0
        result = build_race_features(races)
        assert result["is_wet_race"].iloc[0] == 1

    def test_dry_race(self) -> None:
        races = _make_races(n_seasons=1, rounds_per_season=1, drivers_per_round=1)
        races["f1_rainfall"] = False
        races["weather_precip_mm"] = 0.0
        result = build_race_features(races)
        assert result["is_wet_race"].iloc[0] == 0

    def test_null_weather_imputed(self) -> None:
        races = _make_races(n_seasons=1, rounds_per_season=2, drivers_per_round=1)
        races.loc[races["round"] == 2, "weather_temp_max"] = np.nan
        result = build_race_features(races)
        assert result["weather_temp_max"].notna().all()

    def test_null_precip_defaults_to_zero(self) -> None:
        races = _make_races(n_seasons=1, rounds_per_season=1, drivers_per_round=1)
        races["weather_precip_mm"] = np.nan
        result = build_race_features(races)
        assert result["weather_precip_mm"].iloc[0] == 0.0


class TestLeakagePrevention:
    def test_no_future_data_in_season_form(self) -> None:
        """Verify that removing a future race doesn't change past features."""
        races_full = _make_races(n_seasons=1, rounds_per_season=5, drivers_per_round=2)
        races_truncated = races_full[races_full["round"] <= 4].copy()

        result_full = build_race_features(races_full)
        result_truncated = build_race_features(races_truncated)

        r3_full = result_full[result_full["round"] == 3].sort_values("driver_abbrev")
        r3_trunc = result_truncated[result_truncated["round"] == 3].sort_values("driver_abbrev")

        check_cols = [
            "avg_finish_last_3",
            "points_last_3",
            "points_cumulative_season",
            "dnf_rate_season",
        ]
        for col in check_cols:
            pd.testing.assert_series_equal(
                r3_full[col].reset_index(drop=True),
                r3_trunc[col].reset_index(drop=True),
                check_names=False,
                obj=col,
            )

    def test_no_future_data_in_weather_imputation(self) -> None:
        """Removing a future race shouldn't change past weather values."""
        races_full = _make_races(n_seasons=1, rounds_per_season=3, drivers_per_round=1)
        races_full.loc[races_full["round"] == 2, "weather_temp_max"] = np.nan

        races_trunc = races_full[races_full["round"] <= 2].copy()

        result_full = build_race_features(races_full)
        result_trunc = build_race_features(races_trunc)

        r2_full = result_full[result_full["round"] == 2]
        r2_trunc = result_trunc[result_trunc["round"] == 2]

        pd.testing.assert_series_equal(
            r2_full["weather_temp_max"].reset_index(drop=True),
            r2_trunc["weather_temp_max"].reset_index(drop=True),
            check_names=False,
        )
