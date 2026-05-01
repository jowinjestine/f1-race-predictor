"""Model registry — loads simulators at startup, provides FastAPI dependencies."""

from __future__ import annotations

import io
import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from f1_predictor.api.config import Settings
from f1_predictor.simulation.defaults import build_circuit_defaults, build_field_median_curves
from f1_predictor.simulation.delta_simulator import DeltaRaceSimulator
from f1_predictor.simulation.ensemble_simulator import EnsembleSimulator

logger = logging.getLogger(__name__)


class CPUUnpickler(pickle.Unpickler):
    """Remap CUDA tensors to CPU when loading pkl files trained on GPU."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch

            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def _load_pkl(path: Path) -> Any:
    try:
        with open(path, "rb") as f:
            return CPUUnpickler(f).load()
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301


class ModelRegistry:
    """Holds all loaded models and simulators."""

    def __init__(self) -> None:
        self.h_simulator: DeltaRaceSimulator | None = None
        self.ensemble_simulator: EnsembleSimulator | None = None
        self.model_e: Any = None
        self.circuit_defaults: dict[str, Any] = {}
        self.laps_df: pd.DataFrame | None = None
        self.races_df: pd.DataFrame | None = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self, settings: Settings) -> None:
        """Load all models and data at startup."""
        model_dir = settings.model_dir
        data_dir = settings.data_dir

        if settings.load_from_gcs:
            self._download_from_gcs(settings)

        logger.info("Loading parquet data...")
        self.laps_df = pd.read_parquet(data_dir / "raw/laps/all_laps.parquet")
        self.races_df = pd.read_parquet(data_dir / "raw/race/all_races.parquet")

        logger.info("Building circuit defaults and field medians...")
        self.circuit_defaults = build_circuit_defaults(self.laps_df)
        field_medians = build_field_median_curves(self.laps_df, self.races_df)

        logger.info("Loading Model H...")
        model_h = _load_pkl(model_dir / "Model_H_LightGBM_GOSS_Delta.pkl")
        self.h_simulator = DeltaRaceSimulator(model_h, self.circuit_defaults, field_medians)

        logger.info("Loading Model E...")
        self.model_e = _load_pkl(model_dir / "Model_E_LightGBM_shallow.pkl")
        self.ensemble_simulator = EnsembleSimulator(
            self.h_simulator, self.model_e, blend_laps=settings.default_blend_laps
        )

        self._ready = True
        logger.info("All models loaded successfully.")

    def _download_from_gcs(self, settings: Settings) -> None:
        from f1_predictor.data.storage import download_blob

        files = [
            ("data/raw/model/Model_H_LightGBM_GOSS_Delta.pkl", settings.model_dir / "Model_H_LightGBM_GOSS_Delta.pkl"),
            ("data/raw/model/Model_E_LightGBM_shallow.pkl", settings.model_dir / "Model_E_LightGBM_shallow.pkl"),
            ("data/raw/laps/all_laps.parquet", settings.data_dir / "raw/laps/all_laps.parquet"),
            ("data/raw/race/all_races.parquet", settings.data_dir / "raw/race/all_races.parquet"),
        ]
        for blob_name, local_path in files:
            if not local_path.exists():
                logger.info("Downloading %s from GCS...", blob_name)
                download_blob(blob_name, local_path)


registry = ModelRegistry()
