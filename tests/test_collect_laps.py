"""Tests for lap-level data collection module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from f1_predictor.data.collect_laps import (
    _build_driver_id_to_code,
    _build_driver_id_to_team,
    _build_pitstop_map,
    _fastf1_lap_row,
    _normalize_compound,
    _safe_int,
    add_pit_duration,
)


class TestSafeInt:
    def test_converts_int(self) -> None:
        assert _safe_int(5) == 5

    def test_converts_float(self) -> None:
        assert _safe_int(3.0) == 3

    def test_returns_none_for_nan(self) -> None:
        assert _safe_int(float("nan")) is None

    def test_returns_none_for_nat(self) -> None:
        assert _safe_int(pd.NaT) is None

    def test_returns_none_for_none(self) -> None:
        assert _safe_int(None) is None

    def test_returns_none_for_invalid(self) -> None:
        assert _safe_int("abc") is None


class TestNormalizeCompound:
    def test_normalizes_soft(self) -> None:
        assert _normalize_compound("SOFT") == "SOFT"

    def test_normalizes_lowercase(self) -> None:
        assert _normalize_compound("medium") == "MEDIUM"

    def test_normalizes_with_whitespace(self) -> None:
        assert _normalize_compound(" Hard ") == "HARD"

    def test_returns_none_for_unknown(self) -> None:
        assert _normalize_compound("HYPERSOFT") is None

    def test_returns_none_for_nan(self) -> None:
        assert _normalize_compound(float("nan")) is None

    def test_returns_none_for_none(self) -> None:
        assert _normalize_compound(None) is None

    def test_returns_none_for_empty(self) -> None:
        assert _normalize_compound("") is None

    def test_intermediate(self) -> None:
        assert _normalize_compound("Intermediate") == "INTERMEDIATE"

    def test_wet(self) -> None:
        assert _normalize_compound("WET") == "WET"


class TestBuildPitstopMap:
    def test_builds_map(self) -> None:
        pitstops = [
            {"driverId": "max_verstappen", "lap": "15", "stop": "1", "duration": "23.500"},
            {"driverId": "norris", "lap": "20", "stop": "1", "duration": "24.100"},
        ]
        result = _build_pitstop_map(pitstops)
        assert ("max_verstappen", 15) in result
        assert result[("max_verstappen", 15)] is not None
        assert abs(result[("max_verstappen", 15)] - 23.5) < 0.01  # type: ignore[operator]
        assert ("norris", 20) in result

    def test_handles_empty(self) -> None:
        assert _build_pitstop_map([]) == {}

    def test_handles_null_duration(self) -> None:
        pitstops = [{"driverId": "ver", "lap": "10", "stop": "1", "duration": None}]
        result = _build_pitstop_map(pitstops)
        assert result[("ver", 10)] is None

    def test_skips_missing_driver_id(self) -> None:
        pitstops = [{"lap": "10", "stop": "1", "duration": "23.0"}]
        assert _build_pitstop_map(pitstops) == {}

    def test_skips_invalid_lap(self) -> None:
        pitstops = [{"driverId": "ver", "lap": "abc", "stop": "1", "duration": "23.0"}]
        assert _build_pitstop_map(pitstops) == {}


class TestBuildDriverIdToCode:
    def test_builds_mapping(self) -> None:
        results = [
            {"Driver": {"driverId": "max_verstappen", "code": "VER"}, "Constructor": {}},
            {"Driver": {"driverId": "norris", "code": "NOR"}, "Constructor": {}},
        ]
        mapping = _build_driver_id_to_code(results)
        assert mapping["max_verstappen"] == "VER"
        assert mapping["norris"] == "NOR"

    def test_handles_empty(self) -> None:
        assert _build_driver_id_to_code([]) == {}


class TestBuildDriverIdToTeam:
    def test_builds_mapping(self) -> None:
        results = [
            {
                "Driver": {"driverId": "max_verstappen"},
                "Constructor": {"name": "Red Bull"},
            },
        ]
        mapping = _build_driver_id_to_team(results)
        assert mapping["max_verstappen"] == "Red Bull"


class TestFastf1LapRow:
    def test_extracts_fields(self) -> None:
        lap = MagicMock()
        lap.get = MagicMock(
            side_effect=lambda k, default="": {
                "Driver": "VER",
                "Team": "Red Bull Racing",
                "LapNumber": 1,
                "LapTime": pd.Timedelta(seconds=92.5),
                "Sector1Time": pd.Timedelta(seconds=28.1),
                "Sector2Time": pd.Timedelta(seconds=34.2),
                "Sector3Time": pd.Timedelta(seconds=30.2),
                "Position": 1,
                "Compound": "SOFT",
                "TyreLife": 5,
                "Stint": 1,
                "PitInTime": pd.NaT,
                "PitOutTime": pd.NaT,
                "TrackStatus": "1",
                "IsPersonalBest": True,
            }.get(k, default)
        )

        row = _fastf1_lap_row(2024, 1, "Bahrain GP", lap)
        assert row["season"] == 2024
        assert row["round"] == 1
        assert row["driver_abbrev"] == "VER"
        assert row["lap_number"] == 1
        assert row["lap_time_sec"] is not None
        assert abs(row["lap_time_sec"] - 92.5) < 0.01
        assert row["tire_compound"] == "SOFT"
        assert row["tire_life"] == 5
        assert row["stint"] == 1
        assert row["is_pit_in_lap"] is False
        assert row["is_pit_out_lap"] is False
        assert row["is_personal_best"] is True

    def test_pit_in_lap(self) -> None:
        lap = MagicMock()
        pit_in_td = pd.Timedelta(seconds=3600)
        lap.get = MagicMock(
            side_effect=lambda k, default="": {
                "Driver": "NOR",
                "Team": "McLaren",
                "LapNumber": 15,
                "LapTime": pd.Timedelta(seconds=95.0),
                "Sector1Time": pd.NaT,
                "Sector2Time": pd.NaT,
                "Sector3Time": pd.NaT,
                "Position": 3,
                "Compound": "MEDIUM",
                "TyreLife": 15,
                "Stint": 1,
                "PitInTime": pit_in_td,
                "PitOutTime": pd.NaT,
                "TrackStatus": "1",
                "IsPersonalBest": False,
            }.get(k, default)
        )

        row = _fastf1_lap_row(2024, 1, "Bahrain GP", lap)
        assert row["is_pit_in_lap"] is True
        assert row["is_pit_out_lap"] is False


class TestAddPitDuration:
    def test_adds_pit_duration(self) -> None:
        df = pd.DataFrame(
            {
                "driver_abbrev": ["VER", "VER", "VER"],
                "lap_number": [14, 15, 16],
                "lap_time_sec": [90.0, 115.0, 91.0],
            }
        )
        pit_map = {("VER", 15): 23.5}
        result = add_pit_duration(df, pit_map)
        assert result.at[1, "pit_duration_sec"] == 23.5
        assert result.at[0, "pit_duration_sec"] is None

    def test_skips_if_column_exists(self) -> None:
        df = pd.DataFrame(
            {
                "driver_abbrev": ["VER"],
                "lap_number": [1],
                "lap_time_sec": [90.0],
                "pit_duration_sec": [None],
            }
        )
        result = add_pit_duration(df, {})
        assert "pit_duration_sec" in result.columns
