"""Cross-validation splitters for temporal F1 data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator
import numpy.typing as npt


class LeaveOneSeasonOut:
    """Leave-one-season-out CV for Model A (2019-2024).

    Each fold holds out one season for validation. The test season is
    excluded from all folds entirely.
    """

    def __init__(
        self,
        val_seasons: list[int] | None = None,
        test_season: int = 2024,
    ) -> None:
        self.val_seasons = val_seasons or [2019, 2020, 2021, 2022, 2023]
        self.test_season = test_season

    def get_n_splits(self) -> int:
        return len(self.val_seasons)

    def split(
        self,
        groups: npt.NDArray[np.int_],
    ) -> Iterator[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
        """Yield (train_indices, val_indices) for each fold."""
        for val_season in self.val_seasons:
            train_mask = (groups != val_season) & (groups != self.test_season)
            val_mask = groups == val_season
            yield (
                np.where(train_mask)[0],
                np.where(val_mask)[0],
            )

    def get_test_split(
        self,
        groups: npt.NDArray[np.int_],
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """Return (train_indices, test_indices) for final evaluation."""
        train_mask = groups != self.test_season
        test_mask = groups == self.test_season
        return np.where(train_mask)[0], np.where(test_mask)[0]


class ExpandingWindowSplit:
    """Expanding-window CV for Models B and C (2018-2025).

    Each fold uses a growing training window. The test season is
    excluded from all folds.
    """

    def __init__(
        self,
        fold_definitions: list[tuple[list[int], int]] | None = None,
        test_season: int = 2025,
    ) -> None:
        self.fold_definitions = fold_definitions or [
            ([2018], 2019),
            ([2018, 2019], 2020),
            ([2018, 2019, 2020], 2021),
            ([2018, 2019, 2020, 2021], 2022),
            ([2018, 2019, 2020, 2021, 2022], 2023),
        ]
        self.test_season = test_season

    def get_n_splits(self) -> int:
        return len(self.fold_definitions)

    def split(
        self,
        groups: npt.NDArray[np.int_],
    ) -> Iterator[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
        """Yield (train_indices, val_indices) for each fold."""
        for train_seasons, val_season in self.fold_definitions:
            train_mask = np.isin(groups, train_seasons)
            val_mask = groups == val_season
            yield (
                np.where(train_mask)[0],
                np.where(val_mask)[0],
            )

    def get_test_split(
        self,
        groups: npt.NDArray[np.int_],
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """Return (train_indices, test_indices) for final evaluation."""
        train_mask = groups != self.test_season
        test_mask = groups == self.test_season
        return np.where(train_mask)[0], np.where(test_mask)[0]
