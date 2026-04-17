"""Persist and retrieve F1 data from Google Cloud Storage."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
from google.cloud import storage

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

BUCKET_NAME = "f1-predictor-artifacts-jowin"
GCS_PREFIX = "data/raw"
GCP_PROJECT = "jowin-personal-2026"


def get_client() -> storage.Client:
    return storage.Client(project=GCP_PROJECT)


def upload_parquet(local_path: Path, blob_name: str | None = None) -> str:
    """Upload a local parquet file to GCS. Returns the gs:// URI."""
    client = get_client()
    bucket = client.bucket(BUCKET_NAME)
    name = blob_name or f"{GCS_PREFIX}/{local_path.name}"
    blob = bucket.blob(name)
    blob.upload_from_filename(str(local_path))
    uri = f"gs://{BUCKET_NAME}/{name}"
    logger.info("Uploaded %s -> %s", local_path, uri)
    return uri


def download_parquet(blob_name: str, local_path: Path) -> Path:
    """Download a parquet file from GCS to a local path."""
    client = get_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    logger.info("Downloaded %s -> %s", blob_name, local_path)
    return local_path


def read_parquet_from_gcs(blob_name: str) -> pd.DataFrame:
    """Read a parquet file directly from GCS into a DataFrame."""
    uri = f"gs://{BUCKET_NAME}/{blob_name}"
    logger.info("Reading %s", uri)
    return pd.read_parquet(uri)


def upload_season_files(data_dir: Path) -> list[str]:
    """Upload all season parquet files and the combined file to GCS."""
    uploaded: list[str] = []
    for path in sorted(data_dir.glob("*.parquet")):
        uri = upload_parquet(path)
        uploaded.append(uri)
    return uploaded


def list_remote_seasons() -> list[str]:
    """List all parquet blobs under the data/raw prefix."""
    client = get_client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=GCS_PREFIX)
    return [b.name for b in blobs if b.name.endswith(".parquet")]


def sync_to_local(local_dir: Path) -> list[Path]:
    """Download all remote parquet files that don't exist locally."""
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_blobs = list_remote_seasons()
    downloaded: list[Path] = []
    for blob_name in remote_blobs:
        filename = blob_name.split("/")[-1]
        local_path = local_dir / filename
        if not local_path.exists():
            download_parquet(blob_name, local_path)
            downloaded.append(local_path)
        else:
            logger.info("Already exists locally: %s", local_path)
    return downloaded


def ensure_latest(local_dir: Path) -> pd.DataFrame:
    """Sync from GCS and return the combined DataFrame.

    Falls back to reading local files if GCS is unavailable.
    """
    try:
        sync_to_local(local_dir)
    except Exception:
        logger.warning("Could not sync from GCS, using local files", exc_info=True)

    combined_path = local_dir / "all_races.parquet"
    if combined_path.exists():
        return pd.read_parquet(combined_path)

    season_files = sorted(local_dir.glob("season_*.parquet"))
    if season_files:
        frames: list[pd.DataFrame] = [pd.read_parquet(f) for f in season_files]
        combined: pd.DataFrame = pd.concat(frames, ignore_index=True)
        return combined

    return pd.DataFrame()
