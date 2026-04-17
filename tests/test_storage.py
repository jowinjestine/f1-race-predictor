"""Tests for GCS storage module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pandas as pd

from f1_predictor.data.storage import ensure_latest


class TestEnsureLatest:
    @patch("f1_predictor.data.storage.sync_to_local")
    def test_reads_combined_file(self, mock_sync: MagicMock, tmp_path: Path) -> None:
        df = pd.DataFrame({"season": [2024], "round": [1], "driver": ["VER"]})
        combined = tmp_path / "all_races.parquet"
        df.to_parquet(combined, index=False)

        result = ensure_latest(tmp_path)
        assert len(result) == 1
        assert result["driver"].iloc[0] == "VER"

    @patch("f1_predictor.data.storage.sync_to_local")
    def test_falls_back_to_season_files(self, mock_sync: MagicMock, tmp_path: Path) -> None:
        df1 = pd.DataFrame({"season": [2023], "round": [1]})
        df2 = pd.DataFrame({"season": [2024], "round": [1]})
        df1.to_parquet(tmp_path / "season_2023.parquet", index=False)
        df2.to_parquet(tmp_path / "season_2024.parquet", index=False)

        result = ensure_latest(tmp_path)
        assert len(result) == 2
        assert set(result["season"]) == {2023, 2024}

    @patch("f1_predictor.data.storage.sync_to_local", side_effect=Exception("no GCS"))
    def test_handles_gcs_failure(self, mock_sync: MagicMock, tmp_path: Path) -> None:
        df = pd.DataFrame({"season": [2024], "round": [1]})
        (tmp_path / "all_races.parquet").parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(tmp_path / "all_races.parquet", index=False)

        result = ensure_latest(tmp_path)
        assert len(result) == 1

    @patch("f1_predictor.data.storage.sync_to_local", side_effect=Exception("no GCS"))
    def test_returns_empty_when_nothing_available(
        self, mock_sync: MagicMock, tmp_path: Path
    ) -> None:
        result = ensure_latest(tmp_path)
        assert len(result) == 0
