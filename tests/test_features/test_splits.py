"""Tests for cross-validation splitters."""

from __future__ import annotations

import numpy as np

from f1_predictor.features.splits import ExpandingWindowSplit, LeaveOneSeasonOut


class TestLeaveOneSeasonOut:
    def setup_method(self) -> None:
        self.groups = np.array(
            [2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024]
        )
        self.splitter = LeaveOneSeasonOut()

    def test_yields_five_folds(self) -> None:
        folds = list(self.splitter.split(self.groups))
        assert len(folds) == 5

    def test_each_val_is_one_season(self) -> None:
        for i, (_, val_idx) in enumerate(self.splitter.split(self.groups)):
            val_seasons = set(self.groups[val_idx])
            assert len(val_seasons) == 1
            assert val_seasons == {self.splitter.val_seasons[i]}

    def test_test_season_never_in_train_or_val(self) -> None:
        for train_idx, val_idx in self.splitter.split(self.groups):
            assert self.splitter.test_season not in self.groups[train_idx]
            assert self.splitter.test_season not in self.groups[val_idx]

    def test_get_test_split(self) -> None:
        train_idx, test_idx = self.splitter.get_test_split(self.groups)
        assert set(self.groups[test_idx]) == {2024}
        assert 2024 not in self.groups[train_idx]

    def test_get_n_splits(self) -> None:
        assert self.splitter.get_n_splits() == 5

    def test_custom_val_seasons(self) -> None:
        splitter = LeaveOneSeasonOut(val_seasons=[2020, 2021], test_season=2024)
        folds = list(splitter.split(self.groups))
        assert len(folds) == 2


class TestExpandingWindowSplit:
    def setup_method(self) -> None:
        self.groups = np.array(
            [
                2018,
                2018,
                2019,
                2019,
                2020,
                2020,
                2021,
                2021,
                2022,
                2022,
                2023,
                2023,
                2024,
                2024,
                2025,
                2025,
            ]
        )
        self.splitter = ExpandingWindowSplit()

    def test_yields_five_folds(self) -> None:
        folds = list(self.splitter.split(self.groups))
        assert len(folds) == 5

    def test_fold_1_train_is_2018_val_is_2019(self) -> None:
        folds = list(self.splitter.split(self.groups))
        train_idx, val_idx = folds[0]
        assert set(self.groups[train_idx]) == {2018}
        assert set(self.groups[val_idx]) == {2019}

    def test_fold_5_train_is_2018_to_2022_val_is_2023(self) -> None:
        folds = list(self.splitter.split(self.groups))
        train_idx, val_idx = folds[4]
        assert set(self.groups[train_idx]) == {2018, 2019, 2020, 2021, 2022}
        assert set(self.groups[val_idx]) == {2023}

    def test_test_season_never_in_any_fold(self) -> None:
        for train_idx, val_idx in self.splitter.split(self.groups):
            assert self.splitter.test_season not in self.groups[train_idx]
            assert self.splitter.test_season not in self.groups[val_idx]

    def test_get_test_split(self) -> None:
        train_idx, test_idx = self.splitter.get_test_split(self.groups)
        assert set(self.groups[test_idx]) == {2025}
        assert 2025 not in self.groups[train_idx]

    def test_get_n_splits(self) -> None:
        assert self.splitter.get_n_splits() == 5

    def test_expanding_window_grows(self) -> None:
        prev_train_size = 0
        for train_idx, _ in self.splitter.split(self.groups):
            assert len(train_idx) > prev_train_size
            prev_train_size = len(train_idx)
