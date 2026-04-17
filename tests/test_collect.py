"""Tests for data collection module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests as req_lib

from f1_predictor.data.collect import (
    _aggregate_fastf1_weather,
    _first,
    _td_to_seconds,
    add_target_variables,
    backfill_qualifying,
    get_openmeteo_weather,
)


class TestFirst:
    def test_returns_first_element(self) -> None:
        assert _first([42.5, 10.0]) == 42.5

    def test_returns_none_for_empty_list(self) -> None:
        assert _first([]) is None

    def test_returns_none_for_none_input(self) -> None:
        assert _first(None) is None

    def test_returns_none_for_none_element(self) -> None:
        assert _first([None]) is None


class TestTdToSeconds:
    def test_converts_timedelta(self) -> None:
        td = pd.Timedelta(hours=1, minutes=31, seconds=44.742)
        result = _td_to_seconds(td)
        assert result is not None
        assert abs(result - 5504.742) < 0.001

    def test_returns_none_for_nat(self) -> None:
        assert _td_to_seconds(pd.NaT) is None

    def test_returns_none_for_none(self) -> None:
        assert _td_to_seconds(None) is None

    def test_returns_none_for_zero_timedelta(self) -> None:
        assert _td_to_seconds(pd.Timedelta(0)) is None


class TestAddTargetVariables:
    @pytest.fixture()
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "finish_position": [1, 3, 10, 15, None],
                "status": ["Finished", "Finished", "+1 Lap", "Retired", "Collision"],
            }
        )

    def test_is_podium(self, sample_df: pd.DataFrame) -> None:
        result = add_target_variables(sample_df)
        assert list(result["is_podium"]) == [True, True, False, False, False]

    def test_is_points_finish(self, sample_df: pd.DataFrame) -> None:
        result = add_target_variables(sample_df)
        assert list(result["is_points_finish"]) == [True, True, True, False, False]

    def test_is_dnf(self, sample_df: pd.DataFrame) -> None:
        result = add_target_variables(sample_df)
        assert list(result["is_dnf"]) == [False, False, False, True, True]

    def test_does_not_mutate_input(self, sample_df: pd.DataFrame) -> None:
        original_cols = list(sample_df.columns)
        add_target_variables(sample_df)
        assert list(sample_df.columns) == original_cols


class TestAggregateFastf1Weather:
    def test_aggregates_weather_data(self) -> None:
        weather_df = pd.DataFrame(
            {
                "AirTemp": [25.0, 27.0, 26.0],
                "TrackTemp": [40.0, 42.0, 41.0],
                "Humidity": [50.0, 55.0, 52.5],
                "Pressure": [1013.0, 1013.5, 1013.25],
                "WindSpeed": [3.0, 5.0, 4.0],
                "Rainfall": [False, False, False],
            }
        )
        session = MagicMock()
        session.weather_data = weather_df
        result = _aggregate_fastf1_weather(session)
        assert result["f1_air_temp_mean"] == pytest.approx(26.0)
        assert result["f1_track_temp_mean"] == pytest.approx(41.0)
        assert result["f1_humidity_mean"] == pytest.approx(52.5)
        assert result["f1_wind_speed_mean"] == pytest.approx(4.0)
        assert result["f1_rainfall"] is False

    def test_detects_rainfall(self) -> None:
        weather_df = pd.DataFrame(
            {
                "AirTemp": [20.0],
                "TrackTemp": [30.0],
                "Humidity": [80.0],
                "Pressure": [1010.0],
                "WindSpeed": [6.0],
                "Rainfall": [True],
            }
        )
        session = MagicMock()
        session.weather_data = weather_df
        result = _aggregate_fastf1_weather(session)
        assert result["f1_rainfall"] is True

    def test_returns_nones_for_empty_weather(self) -> None:
        session = MagicMock()
        session.weather_data = None
        result = _aggregate_fastf1_weather(session)
        assert result["f1_air_temp_mean"] is None
        assert result["f1_rainfall"] is None

    def test_returns_nones_on_exception(self) -> None:
        session = MagicMock()
        weather_mock = MagicMock()
        weather_mock.__len__ = MagicMock(return_value=1)
        weather_mock.__getitem__ = MagicMock(side_effect=KeyError("AirTemp"))
        session.weather_data = weather_mock
        result = _aggregate_fastf1_weather(session)
        assert result["f1_air_temp_mean"] is None


class TestGetOpenmeteoWeather:
    @patch("f1_predictor.data.collect.requests.get")
    def test_returns_weather_data(self, mock_get: MagicMock) -> None:
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = MagicMock()
        mock_get.return_value.json.return_value = {
            "daily": {
                "temperature_2m_max": [32.5],
                "temperature_2m_min": [22.1],
                "precipitation_sum": [0.0],
                "windspeed_10m_max": [15.3],
            }
        }
        result = get_openmeteo_weather(26.0, 50.5, "2024-03-02")
        assert result["weather_temp_max"] == 32.5
        assert result["weather_precip_mm"] == 0.0

    @patch("f1_predictor.data.collect.requests.get")
    def test_returns_nones_on_failure(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = req_lib.ConnectionError("Network error")
        result = get_openmeteo_weather(26.0, 50.5, "2024-03-02")
        assert result["weather_temp_max"] is None
        assert result["weather_precip_mm"] is None

    @patch("f1_predictor.data.collect.requests.get")
    def test_returns_nones_on_json_decode_error(self, mock_get: MagicMock) -> None:
        mock_get.return_value.raise_for_status = MagicMock()
        mock_get.return_value.json.side_effect = ValueError("Invalid JSON")
        result = get_openmeteo_weather(26.0, 50.5, "2024-03-02")
        assert result["weather_temp_max"] is None
        assert result["weather_precip_mm"] is None


class TestBackfillQualifying:
    @patch("f1_predictor.data.collect.get_qualifying_results")
    def test_backfills_null_qualifying(self, mock_quali: MagicMock) -> None:
        df = pd.DataFrame(
            {
                "season": [2024, 2024],
                "round": [1, 1],
                "driver_abbrev": ["VER", "NOR"],
                "q1_time_sec": [None, None],
                "q2_time_sec": [None, None],
                "q3_time_sec": [None, None],
            }
        )
        mock_quali.return_value = [
            {"Driver": {"code": "VER"}, "Q1": "1:30.000", "Q2": "1:29.000", "Q3": "1:28.500"},
            {"Driver": {"code": "NOR"}, "Q1": "1:30.100", "Q2": "1:29.200", "Q3": None},
        ]
        result = backfill_qualifying(df)
        assert result.at[0, "q1_time_sec"] == pytest.approx(90.0)
        assert result.at[0, "q3_time_sec"] == pytest.approx(88.5)
        assert result.at[1, "q1_time_sec"] == pytest.approx(90.1)
        assert pd.isna(result.at[1, "q3_time_sec"])

    def test_skips_when_not_all_null(self) -> None:
        df = pd.DataFrame(
            {
                "season": [2024],
                "round": [1],
                "driver_abbrev": ["VER"],
                "q1_time_sec": [90.0],
                "q2_time_sec": [None],
                "q3_time_sec": [None],
            }
        )
        result = backfill_qualifying(df)
        assert result.at[0, "q1_time_sec"] == 90.0

    def test_returns_empty_for_empty_df(self) -> None:
        result = backfill_qualifying(pd.DataFrame())
        assert len(result) == 0
