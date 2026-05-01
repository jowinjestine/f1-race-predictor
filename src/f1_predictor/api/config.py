"""API configuration via pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_dir: Path = Path("data/raw/model")
    data_dir: Path = Path("data")
    load_from_gcs: bool = False
    gcs_bucket: str = "f1-predictor-artifacts-jowin"
    port: int = 8080
    default_blend_laps: int = 10

    model_config = {"env_prefix": "F1_"}
