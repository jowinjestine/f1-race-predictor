#!/usr/bin/env python3
"""Poll GCS for model training completion signals. Prints status lines to stdout."""

from __future__ import annotations

import sys
import time

from google.cloud import compute_v1, storage

PROJECT = "jowin-personal-2026"
BUCKET = "f1-predictor-artifacts-jowin"
STAGING_PREFIX = "staging/training-run"

MODELS = ["G", "H", "I"]
ARTIFACT_PREFIXES = {
    "G": ("data/training/model_G", "data/raw/model/Model_G"),
    "H": ("data/training/model_H", "data/raw/model/Model_H"),
    "I": ("data/training/model_I", "data/raw/model/Model_I"),
}


def check_status() -> dict[str, dict]:
    storage_client = storage.Client(project=PROJECT)
    bucket = storage_client.bucket(BUCKET)
    compute_client = compute_v1.InstancesClient()

    status = {}
    for m in MODELS:
        done_blob = bucket.blob(f"{STAGING_PREFIX}/MODEL_{m}_DONE")
        is_done = done_blob.exists()

        parquet_count = sum(1 for _ in bucket.list_blobs(prefix=ARTIFACT_PREFIXES[m][0]))
        pkl_count = sum(1 for _ in bucket.list_blobs(prefix=ARTIFACT_PREFIXES[m][1]))

        vm_running = False
        vm_name = None
        vm_zone = None
        for zone in [
            "us-central1-a", "us-central1-b", "us-central1-c",
            "us-east4-a", "us-east4-c",
            "us-west1-a", "us-west1-b",
            "us-east1-b", "us-east1-c",
        ]:
            try:
                for inst in compute_client.list(project=PROJECT, zone=zone):
                    if f"f1-model-{m.lower()}" in inst.name and inst.status == "RUNNING":
                        vm_running = True
                        vm_name = inst.name
                        vm_zone = zone
                        break
            except Exception:
                pass
            if vm_running:
                break

        if is_done and (parquet_count > 0 or pkl_count > 0):
            state = "DONE"
        elif is_done and parquet_count == 0:
            state = "FAILED (no artifacts)"
        elif vm_running:
            state = f"TRAINING ({vm_name} in {vm_zone})"
        else:
            state = "WAITING"

        status[m] = {
            "state": state,
            "done": is_done,
            "parquets": parquet_count,
            "pkls": pkl_count,
            "vm": vm_name,
        }

    return status


def format_status(status: dict) -> str:
    lines = []
    ts = time.strftime("%H:%M UTC", time.gmtime())
    lines.append(f"[{ts}] Training Status:")
    for m in MODELS:
        s = status[m]
        lines.append(f"  Model {m}: {s['state']} | parquets={s['parquets']} pkl={s['pkls']}")
    return "\n".join(lines)


if __name__ == "__main__":
    status = check_status()
    print(format_status(status))
    sys.stdout.flush()
