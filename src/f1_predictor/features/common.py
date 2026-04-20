"""Shared utilities for feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Element-wise division returning NaN when denominator is zero."""
    result: pd.Series = numerator / denominator.replace(0, np.nan)
    return result


def rolling_mean_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    window: int,
    min_periods: int = 1,
) -> pd.Series:
    """Grouped rolling mean with shift(1) to prevent data leakage.

    The current row is never included in its own rolling window.
    """
    group_codes = df.groupby(group_cols, sort=False).ngroup()
    shifted: pd.Series = df.groupby(group_cols, sort=False)[value_col].shift(1)
    result: pd.Series = (
        shifted.groupby(group_codes, sort=False)
        .rolling(window=window, min_periods=min_periods)
        .mean()
        .droplevel(0)
        .sort_index()
    )
    return result


def expanding_mean_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
) -> pd.Series:
    """Grouped expanding mean with shift(1) to prevent data leakage."""
    group_codes = df.groupby(group_cols, sort=False).ngroup()
    result: pd.Series = (
        df.groupby(group_cols, sort=False)[value_col]
        .expanding()
        .mean()
        .droplevel(list(range(len(group_cols))))
        .sort_index()
    )
    return result.groupby(group_codes, sort=False).shift(1)


def expanding_sum_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
) -> pd.Series:
    """Grouped expanding sum with shift(1) to prevent data leakage."""
    group_codes = df.groupby(group_cols, sort=False).ngroup()
    result: pd.Series = (
        df.groupby(group_cols, sort=False)[value_col]
        .expanding()
        .sum()
        .droplevel(list(range(len(group_cols))))
        .sort_index()
    )
    return result.groupby(group_codes, sort=False).shift(1)


def expanding_count_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
) -> pd.Series:
    """Grouped expanding count with shift(1) to prevent data leakage."""
    group_codes = df.groupby(group_cols, sort=False).ngroup()
    result: pd.Series = (
        df.groupby(group_cols, sort=False)[value_col]
        .expanding()
        .count()
        .droplevel(list(range(len(group_cols))))
        .sort_index()
    )
    return result.groupby(group_codes, sort=False).shift(1)


def rolling_sum_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    window: int,
    min_periods: int = 1,
) -> pd.Series:
    """Grouped rolling sum with shift(1) to prevent data leakage."""
    group_codes = df.groupby(group_cols, sort=False).ngroup()
    shifted: pd.Series = df.groupby(group_cols, sort=False)[value_col].shift(1)
    result: pd.Series = (
        shifted.groupby(group_codes, sort=False)
        .rolling(window=window, min_periods=min_periods)
        .sum()
        .droplevel(0)
        .sort_index()
    )
    return result


VALID_COMPOUNDS = frozenset({"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"})


def encode_compound_onehot(series: pd.Series) -> pd.DataFrame:
    """One-hot encode tire compound into 5 boolean columns."""
    result = pd.DataFrame(index=series.index)
    for compound in sorted(VALID_COMPOUNDS):
        result[f"compound_{compound}"] = (series == compound).astype(int)
    return result
