"""Tests for Jolpica API client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from f1_predictor.data.jolpica import (
    _parse_lap_time,
    _parse_race_time_millis,
    get_race_results,
    get_season_schedule,
)


class TestParseLapTime:
    def test_parses_minutes_seconds(self) -> None:
        assert _parse_lap_time("1:22.167") is not None
        result = _parse_lap_time("1:22.167")
        assert result is not None
        assert abs(result - 82.167) < 0.001

    def test_parses_seconds_only(self) -> None:
        result = _parse_lap_time("82.167")
        assert result is not None
        assert abs(result - 82.167) < 0.001

    def test_returns_none_for_empty(self) -> None:
        assert _parse_lap_time("") is None
        assert _parse_lap_time(None) is None

    def test_returns_none_for_invalid(self) -> None:
        assert _parse_lap_time("abc") is None


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
