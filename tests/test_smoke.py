"""Smoke tests to verify the project skeleton works."""

import importlib

from f1_predictor import __version__


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_import_subpackages() -> None:
    for name in ["api", "data", "explain", "features", "models"]:
        importlib.import_module(f"f1_predictor.{name}")
