"""Tests for Jolpica API client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from f1_predictor.data.jolpica import (
    _parse_race_time_millis,
    get_laps,
    get_pitstops,
    get_qualifying_results,
    get_race_results,
    get_season_schedule,
    parse_lap_time,
)


class TestParseLapTime:
    def test_parses_minutes_seconds(self) -> None:
        assert parse_lap_time("1:22.167") is not None
        result = parse_lap_time("1:22.167")
        assert result is not None
        assert abs(result - 82.167) < 0.001

    def test_parses_seconds_only(self) -> None:
        result = parse_lap_time("82.167")
        assert result is not None
        assert abs(result - 82.167) < 0.001

    def test_returns_none_for_empty(self) -> None:
        assert parse_lap_time("") is None
        assert parse_lap_time(None) is None

    def test_returns_none_for_invalid(self) -> None:
        assert parse_lap_time("abc") is None


class TestParseRaceTimeMillis:
    def test_parses_millis(self) -> None:
        result = _parse_race_time_millis("6126304")
        assert result is not None
        assert abs(result - 6126.304) < 0.001

    def test_returns_none_for_empty(self) -> None:
        assert _parse_race_time_millis(None) is None
        assert _parse_race_time_millis("") is None


class TestGetSeasonSchedule:
    @patch("f1_predictor.data.jolpica._get_json")
    def test_returns_races(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {
            "MRData": {
                "RaceTable": {
                    "Races": [
                        {"round": "1", "raceName": "Australian GP"},
                        {"round": "2", "raceName": "Bahrain GP"},
                    ]
                }
            }
        }
        races = get_season_schedule(2025)
        assert len(races) == 2
        assert races[0]["raceName"] == "Australian GP"

    @patch("f1_predictor.data.jolpica._get_json")
    def test_returns_empty_on_failure(self, mock_get: MagicMock) -> None:
        mock_get.return_value = None
        assert get_season_schedule(2025) == []


class TestGetRaceResults:
    @patch("f1_predictor.data.jolpica._get_json")
    def test_returns_results(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {
            "MRData": {
                "RaceTable": {
                    "Races": [
                        {
                            "Results": [
                                {"position": "1", "Driver": {"code": "NOR"}},
                                {"position": "2", "Driver": {"code": "PIA"}},
                            ]
                        }
                    ]
                }
            }
        }
        results = get_race_results(2025, 1)
        assert len(results) == 2

    @patch("f1_predictor.data.jolpica._get_json")
    def test_returns_empty_for_no_races(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {"MRData": {"RaceTable": {"Races": []}}}
        assert get_race_results(2025, 99) == []

    @patch("f1_predictor.data.jolpica._get_json")
    def test_returns_empty_on_failure(self, mock_get: MagicMock) -> None:
        mock_get.return_value = None
        assert get_race_results(2025, 1) == []


class TestGetQualifyingResults:
    @patch("f1_predictor.data.jolpica._get_json")
    def test_returns_qualifying(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {
            "MRData": {
                "RaceTable": {
                    "Races": [
                        {
                            "QualifyingResults": [
                                {"position": "1", "Driver": {"code": "NOR"}, "Q1": "1:22.000"},
                            ]
                        }
                    ]
                }
            }
        }
        results = get_qualifying_results(2025, 1)
        assert len(results) == 1
        assert results[0]["Q1"] == "1:22.000"

    @patch("f1_predictor.data.jolpica._get_json")
    def test_returns_empty_on_failure(self, mock_get: MagicMock) -> None:
        mock_get.return_value = None
        assert get_qualifying_results(2025, 1) == []

    @patch("f1_predictor.data.jolpica._get_json")
    def test_returns_empty_for_no_races(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {"MRData": {"RaceTable": {"Races": []}}}
        assert get_qualifying_results(2025, 1) == []


class TestGetLaps:
    @patch("f1_predictor.data.jolpica._get_json")
    def test_returns_laps(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {
            "MRData": {
                "RaceTable": {
                    "Races": [
                        {
                            "Laps": [
                                {
                                    "number": "1",
                                    "Timings": [
                                        {"driverId": "ver", "position": "1", "time": "1:34.523"},
                                    ],
                                }
                            ]
                        }
                    ]
                }
            }
        }
        laps = get_laps(2025, 1)
        assert len(laps) == 1
        assert laps[0]["number"] == "1"
        assert len(laps[0]["Timings"]) == 1

    @patch("f1_predictor.data.jolpica._get_json")
    def test_returns_empty_on_failure(self, mock_get: MagicMock) -> None:
        mock_get.return_value = None
        assert get_laps(2025, 1) == []

    @patch("f1_predictor.data.jolpica._get_json")
    def test_returns_empty_for_no_races(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {"MRData": {"RaceTable": {"Races": []}}}
        assert get_laps(2025, 1) == []


class TestGetPitstops:
    @patch("f1_predictor.data.jolpica._get_json")
    def test_returns_pitstops(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {
            "MRData": {
                "total": "1",
                "limit": "100",
                "offset": "0",
                "RaceTable": {
                    "Races": [
                        {
                            "PitStops": [
                                {
                                    "driverId": "ver",
                                    "lap": "15",
                                    "stop": "1",
                                    "duration": "23.500",
                                },
                            ]
                        }
                    ]
                },
            }
        }
        stops = get_pitstops(2025, 1)
        assert len(stops) == 1
        assert stops[0]["driverId"] == "ver"
        assert stops[0]["duration"] == "23.500"

    @patch("f1_predictor.data.jolpica._get_json")
    def test_paginates_pitstops(self, mock_get: MagicMock) -> None:
        page1 = {
            "MRData": {
                "total": "101",
                "limit": "100",
                "offset": "0",
                "RaceTable": {"Races": [{"PitStops": [{"driverId": "ver", "lap": "15"}]}]},
            }
        }
        page2 = {
            "MRData": {
                "total": "101",
                "limit": "100",
                "offset": "100",
                "RaceTable": {"Races": [{"PitStops": [{"driverId": "nor", "lap": "20"}]}]},
            }
        }
        mock_get.side_effect = [page1, page2]
        stops = get_pitstops(2025, 1)
        assert len(stops) == 2
        assert stops[0]["driverId"] == "ver"
        assert stops[1]["driverId"] == "nor"

    @patch("f1_predictor.data.jolpica._get_json")
    def test_returns_empty_on_failure(self, mock_get: MagicMock) -> None:
        mock_get.return_value = None
        assert get_pitstops(2025, 1) == []

    @patch("f1_predictor.data.jolpica._get_json")
    def test_returns_empty_for_no_races(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {"MRData": {"total": "0", "RaceTable": {"Races": []}}}
        assert get_pitstops(2025, 1) == []


class TestParseRaceTimeMillisEdge:
    def test_returns_none_for_invalid(self) -> None:
        assert _parse_race_time_millis("not_a_number") is None
