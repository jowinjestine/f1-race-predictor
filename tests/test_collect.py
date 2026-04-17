"""Tests for data collection module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests as req_lib

from f1_predictor.data.collect import (
    _first,
    _td_to_seconds,
    add_target_variables,
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
