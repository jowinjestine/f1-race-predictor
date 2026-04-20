"""Tests for shared feature engineering utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1_predictor.features.common import (
    encode_compound_onehot,
    expanding_count_by_group,
    expanding_mean_by_group,
    expanding_sum_by_group,
    rolling_mean_by_group,
    rolling_sum_by_group,
    safe_divide,
)


class TestSafeDivide:
    def test_normal_division(self) -> None:
        num = pd.Series([10.0, 20.0, 30.0])
        denom = pd.Series([2.0, 5.0, 10.0])
        result = safe_divide(num, denom)
        expected = pd.Series([5.0, 4.0, 3.0])
        pd.testing.assert_series_equal(result, expected)

    def test_zero_denominator_returns_nan(self) -> None:
        num = pd.Series([10.0, 20.0])
        denom = pd.Series([0.0, 5.0])
        result = safe_divide(num, denom)
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(4.0)

    def test_nan_numerator(self) -> None:
        num = pd.Series([np.nan, 20.0])
        denom = pd.Series([2.0, 5.0])
        result = safe_divide(num, denom)
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(4.0)


class TestRollingMeanByGroup:
    def test_basic_rolling_mean(self) -> None:
        df = pd.DataFrame({
            "group": ["A", "A", "A", "A"],
            "value": [10.0, 20.0, 30.0, 40.0],
        })
        result = rolling_mean_by_group(df, ["group"], "value", window=2)
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(10.0)
        assert result.iloc[2] == pytest.approx(15.0)
        assert result.iloc[3] == pytest.approx(25.0)

    def test_shift_prevents_leakage(self) -> None:
        df = pd.DataFrame({
            "group": ["A", "A", "A"],
            "value": [100.0, 200.0, 300.0],
        })
        result = rolling_mean_by_group(df, ["group"], "value", window=3)
        assert np.isnan(result.iloc[0])

    def test_multiple_groups(self) -> None:
        df = pd.DataFrame({
            "group": ["A", "A", "B", "B"],
            "value": [10.0, 20.0, 100.0, 200.0],
        })
        result = rolling_mean_by_group(df, ["group"], "value", window=2)
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(10.0)
        assert np.isnan(result.iloc[2])
        assert result.iloc[3] == pytest.approx(100.0)


class TestExpandingMeanByGroup:
    def test_expanding_mean(self) -> None:
        df = pd.DataFrame({
            "group": ["A", "A", "A"],
            "value": [10.0, 20.0, 30.0],
        })
        result = expanding_mean_by_group(df, ["group"], "value")
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(10.0)
        assert result.iloc[2] == pytest.approx(15.0)

    def test_groups_are_independent(self) -> None:
        df = pd.DataFrame({
            "group": ["A", "B", "A", "B"],
            "value": [10.0, 100.0, 20.0, 200.0],
        })
        result = expanding_mean_by_group(df, ["group"], "value")
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(10.0)
        assert result.iloc[3] == pytest.approx(100.0)


class TestExpandingSumByGroup:
    def test_expanding_sum(self) -> None:
        df = pd.DataFrame({
            "group": ["A", "A", "A"],
            "value": [10.0, 20.0, 30.0],
        })
        result = expanding_sum_by_group(df, ["group"], "value")
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(10.0)
        assert result.iloc[2] == pytest.approx(30.0)


class TestExpandingCountByGroup:
    def test_expanding_count(self) -> None:
        df = pd.DataFrame({
            "group": ["A", "A", "A"],
            "value": [1.0, 1.0, 1.0],
        })
        result = expanding_count_by_group(df, ["group"], "value")
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(1.0)
        assert result.iloc[2] == pytest.approx(2.0)


class TestRollingSumByGroup:
    def test_rolling_sum(self) -> None:
        df = pd.DataFrame({
            "group": ["A", "A", "A", "A"],
            "value": [10.0, 20.0, 30.0, 40.0],
        })
        result = rolling_sum_by_group(df, ["group"], "value", window=2)
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(10.0)
        assert result.iloc[2] == pytest.approx(30.0)
        assert result.iloc[3] == pytest.approx(50.0)


class TestEncodeCompoundOnehot:
    def test_all_compounds(self) -> None:
        series = pd.Series(["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"])
        result = encode_compound_onehot(series)
        assert list(result.columns) == [
            "compound_HARD",
            "compound_INTERMEDIATE",
            "compound_MEDIUM",
            "compound_SOFT",
            "compound_WET",
        ]
        assert result["compound_SOFT"].iloc[0] == 1
        assert result["compound_MEDIUM"].iloc[1] == 1
        assert result["compound_HARD"].iloc[2] == 1
        assert result["compound_INTERMEDIATE"].iloc[3] == 1
        assert result["compound_WET"].iloc[4] == 1

    def test_unknown_compound_is_all_zeros(self) -> None:
        series = pd.Series(["UNKNOWN"])
        result = encode_compound_onehot(series)
        assert result.sum(axis=1).iloc[0] == 0

    def test_null_compound(self) -> None:
        series = pd.Series([None])
        result = encode_compound_onehot(series)
        assert result.sum(axis=1).iloc[0] == 0
