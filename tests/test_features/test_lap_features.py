"""Tests for lap-level feature engineering (Models A and B)."""

from __future__ import annotations

import pandas as pd
import pytest

from f1_predictor.features.lap_features import (
    build_lap_notyre_features,
    build_lap_tyre_features,
)


def _make_laps(
    season: int = 2022,
    n_drivers: int = 2,
    n_laps: int = 5,
    compound: str = "SOFT",
) -> pd.DataFrame:
    """Create minimal lap data for testing."""
    rows = []
    for d in range(n_drivers):
        driver = f"DR{d}"
        for lap in range(1, n_laps + 1):
            rows.append(
                {
                    "season": season,
                    "round": 1,
                    "event_name": "Test GP",
                    "driver_abbrev": driver,
                    "team": f"Team{d}",
                    "lap_number": lap,
                    "lap_time_sec": 90.0 + d * 2.0 + lap * 0.1,
                    "sector_1_sec": 28.0,
                    "sector_2_sec": 32.0,
                    "sector_3_sec": 30.0,
                    "position": d + 1,
                    "tire_compound": compound,
                    "tire_life": lap,
                    "stint": 1,
                    "is_pit_in_lap": lap == 3,
                    "is_pit_out_lap": lap == 4,
                    "track_status": "1",
                    "is_personal_best": False,
                    "pit_duration_sec": 23.0 if lap == 3 else None,
                }
            )
    return pd.DataFrame(rows)


class TestBuildLapTyreFeatures:
    def test_filters_to_2019_2024(self) -> None:
        laps_2018 = _make_laps(season=2018)
        laps_2022 = _make_laps(season=2022)
        laps_2025 = _make_laps(season=2025)
        combined = pd.concat([laps_2018, laps_2022, laps_2025], ignore_index=True)
        result = build_lap_tyre_features(combined)
        assert set(result["season"].unique()) == {2022}

    def test_output_has_tyre_columns(self) -> None:
        laps = _make_laps(season=2022)
        result = build_lap_tyre_features(laps)
        assert "compound_SOFT" in result.columns
        assert "compound_MEDIUM" in result.columns
        assert "tire_life" in result.columns
        assert "stint" in result.columns
        assert "degradation_rate" in result.columns
        assert "compound_pace_delta" in result.columns

    def test_output_has_shared_features(self) -> None:
        laps = _make_laps(season=2022)
        result = build_lap_tyre_features(laps)
        expected = [
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
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_drops_null_tyre_rows(self) -> None:
        laps = _make_laps(season=2022)
        laps.loc[0, "tire_compound"] = None
        result = build_lap_tyre_features(laps)
        assert len(result) == len(laps) - 1

    def test_position_is_target(self) -> None:
        laps = _make_laps(season=2022)
        result = build_lap_tyre_features(laps)
        assert "position" in result.columns

    def test_rolling_3_leakage_prevention(self) -> None:
        laps = _make_laps(season=2022, n_drivers=1, n_laps=5)
        result = build_lap_tyre_features(laps)
        first_lap = result[result["lap_number"] == 1]
        assert first_lap["lap_time_rolling_3"].isna().all()


class TestBuildLapNotyreFeatures:
    def test_includes_all_seasons(self) -> None:
        laps_2018 = _make_laps(season=2018)
        laps_2025 = _make_laps(season=2025, compound="SOFT")
        combined = pd.concat([laps_2018, laps_2025], ignore_index=True)
        result = build_lap_notyre_features(combined)
        assert set(result["season"].unique()) == {2018, 2025}

    def test_no_tyre_columns(self) -> None:
        laps = _make_laps(season=2022)
        result = build_lap_notyre_features(laps)
        tyre_cols = [
            "compound_SOFT",
            "compound_MEDIUM",
            "compound_HARD",
            "tire_life",
            "stint",
            "degradation_rate",
            "compound_pace_delta",
        ]
        for col in tyre_cols:
            assert col not in result.columns

    def test_has_shared_features(self) -> None:
        laps = _make_laps(season=2022)
        result = build_lap_notyre_features(laps)
        assert "lap_time_delta_race_median" in result.columns
        assert "gap_to_leader" in result.columns
        assert "race_progress_pct" in result.columns

    def test_gap_to_leader_for_leader_is_zero(self) -> None:
        laps = _make_laps(season=2022, n_drivers=2, n_laps=3)
        result = build_lap_notyre_features(laps)
        leader = result[result["position"] == 1]
        assert (leader["gap_to_leader"] == 0.0).all()

    def test_race_progress_at_last_lap_is_one(self) -> None:
        laps = _make_laps(season=2022, n_drivers=1, n_laps=5)
        result = build_lap_notyre_features(laps)
        last_lap = result[result["lap_number"] == 5]
        assert last_lap["race_progress_pct"].iloc[0] == pytest.approx(1.0)

    def test_pit_count_increments(self) -> None:
        laps = _make_laps(season=2022, n_drivers=1, n_laps=5)
        result = build_lap_notyre_features(laps)
        dr = result[result["driver_abbrev"] == "DR0"]
        assert dr[dr["lap_number"] == 1]["pit_stop_count"].iloc[0] == 0
        assert dr[dr["lap_number"] == 4]["pit_stop_count"].iloc[0] == 1

    def test_position_change_from_lap1(self) -> None:
        laps = _make_laps(season=2022, n_drivers=1, n_laps=3)
        laps.loc[laps["lap_number"] == 1, "position"] = 5
        laps.loc[laps["lap_number"] == 3, "position"] = 3
        result = build_lap_notyre_features(laps)
        row = result[result["lap_number"] == 3]
        assert row["position_change_from_lap1"].iloc[0] == 2
