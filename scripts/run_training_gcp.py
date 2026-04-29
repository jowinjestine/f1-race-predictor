#!/usr/bin/env python3
"""Train models on GCE GPU VMs using the Python SDK (replaces gcloud CLI scripts)."""

from __future__ import annotations

import argparse
import io
import os
import tarfile
import time
from pathlib import Path

from google.cloud import compute_v1, storage

PROJECT = "jowin-personal-2026"
ZONE = "us-central1-a"
BUCKET = "f1-predictor-artifacts-jowin"
STAGING_PREFIX = "staging/training-run"
RESULTS_PREFIX = "data"
CPU_IMAGE_FAMILY = "debian-12"
CPU_IMAGE_PROJECT = "debian-cloud"

GPU_CONFIGS = {
    "l4": {
        "accelerator": "nvidia-l4",
        "machine_type": "g2-standard-8",
        "image_family": "common-cu129-ubuntu-2204-nvidia-580",
        "image_project": "deeplearning-platform-release",
        "zones": [
            "us-central1-a",
            "us-central1-b",
            "us-central1-c",
            "us-east4-a",
            "us-east4-c",
            "us-west1-a",
            "us-west1-b",
            "us-east1-b",
            "us-east1-c",
        ],
    },
    "v100": {
        "accelerator": "nvidia-tesla-v100",
        "machine_type": "n1-standard-8",
        "image_family": "ubuntu-2204-lts",
        "image_project": "ubuntu-os-cloud",
        "zones": [
            "us-central1-a",
            "us-central1-b",
            "us-central1-c",
            "us-central1-f",
            "us-east1-b",
            "us-west1-b",
        ],
    },
    "a100": {
        "accelerator": "nvidia-tesla-a100",
        "machine_type": "a2-highgpu-1g",
        "image_family": "common-cu129-ubuntu-2204-nvidia-580",
        "image_project": "deeplearning-platform-release",
        "zones": [
            "us-central1-a",
            "us-central1-b",
            "us-central1-c",
            "us-central1-f",
            "us-east1-b",
            "us-west1-b",
        ],
    },
}

REPO_ROOT = Path(__file__).resolve().parent.parent

EXCLUDE_PATTERNS = {
    ".venv",
    "__pycache__",
    ".git",
    ".fastf1_cache",
    "node_modules",
    ".mypy_cache",
}

MODELS = {
    "G": {
        "notebook": "05g_model_G_temporal.ipynb",
        "timeout": 10800,
        "extra_pip": "",
        "artifact_prefix": "model_G",
        "model_prefix": "Model_G",
        "est_minutes": "120-180",
    },
    "H": {
        "notebook": "05h_model_H_delta_mc.ipynb",
        "timeout": 7200,
        "extra_pip": "",
        "artifact_prefix": "model_H",
        "model_prefix": "Model_H",
        "est_minutes": "60-120",
    },
    "I": {
        "notebook": "05i_model_I_quantile.ipynb",
        "timeout": 7200,
        "extra_pip": "scipy",
        "artifact_prefix": "model_I",
        "model_prefix": "Model_I",
        "est_minutes": "60-120",
    },
}


def _should_exclude(path: Path) -> bool:
    return any(part in EXCLUDE_PATTERNS for part in path.parts)


SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")


def _make_startup_script(model_keys: list[str]) -> str:
    extra_pips = set()
    for k in model_keys:
        if MODELS[k]["extra_pip"]:
            extra_pips.add(MODELS[k]["extra_pip"])
    extra_pip_line = f"uv pip install {' '.join(extra_pips)}" if extra_pips else ""

    model_blocks = ""
    for k in model_keys:
        m = MODELS[k]
        model_blocks += f"""
echo "=== Running Model {k}: $(date) ==="
slack_notify ":racing_car: *Model {k} starting* on $(hostname)"

uv run jupyter nbconvert --to notebook --execute \\
    --ExecutePreprocessor.timeout=-1 \\
    notebooks/{m["notebook"]} \\
    --output {m["notebook"]} 2>&1
NBCONVERT_EXIT=$?
echo ">>> Model {k} nbconvert exit code: $NBCONVERT_EXIT"

gsutil -m -q cp data/training/{m["artifact_prefix"]}_*.parquet "gs://$BUCKET/$RESULTS/training/" 2>/dev/null || true
gsutil -m -q cp data/raw/model/{m["model_prefix"]}_*.pkl "gs://$BUCKET/$RESULTS/raw/model/" 2>/dev/null || true
gsutil -q cp notebooks/{m["notebook"]} "gs://$BUCKET/$RESULTS/notebooks/" 2>/dev/null || true
gsutil -q cp /var/log/f1-model-{k.lower()}-progress.log "gs://$BUCKET/$STAGING/model_{k.lower()}_progress.log" 2>/dev/null || true

if [ "$NBCONVERT_EXIT" -eq 0 ] && ls data/training/{m["artifact_prefix"]}_*.parquet 1>/dev/null 2>&1; then
    echo "DONE" | gsutil -q cp - "gs://$BUCKET/$STAGING/MODEL_{k}_DONE"
    slack_notify ":tada: *Model {k} SUCCESS* on $(hostname)"
else
    echo "FAILED" | gsutil -q cp - "gs://$BUCKET/$STAGING/MODEL_{k}_DONE"
    slack_notify ":rotating_light: *Model {k} FAILED* on $(hostname)"
fi

upload_log
"""

    return f"""#!/bin/bash
exec > /var/log/f1-training.log 2>&1
set -euo pipefail

BUCKET="{BUCKET}"
STAGING="{STAGING_PREFIX}"
RESULTS="{RESULTS_PREFIX}"
WORK="/opt/f1-training"
export SLACK_WEBHOOK_URL="{SLACK_WEBHOOK_URL}"

slack_notify() {{
    curl -s -X POST "$SLACK_WEBHOOK_URL" \\
        -H 'Content-Type: application/json' \\
        -d "{{\\"text\\": \\"$1\\"}}" 2>/dev/null || true
}}

upload_log() {{
    gsutil -q cp /var/log/f1-training.log "gs://$BUCKET/$STAGING/training_log.txt" 2>/dev/null || true
}}
trap upload_log EXIT

echo "=== Training begin: $(date) ==="
slack_notify ":computer: *VM $(hostname) started* -- training Models {" ".join(model_keys)} sequentially"

mkdir -p "$WORK" && cd "$WORK"

export HOME="${{HOME:-/root}}"
export GCE_METADATA_MTLS_MODE=none

echo ">>> Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
export PATH="/root/.local/bin:$HOME/.local/bin:$PATH"

echo ">>> Downloading repo..."
pip3 install gsutil 2>/dev/null || true
gsutil -q cp "gs://$BUCKET/$STAGING/repo.tar.gz" repo.tar.gz
tar xzf repo.tar.gz
rm repo.tar.gz

mkdir -p data/processed/simulation data/training data/raw/model checkpoints

echo ">>> Installing Python deps..."
uv python install 3.11
uv sync --frozen --group dev
uv pip install xgboost lightgbm optuna
{extra_pip_line}

if ! nvidia-smi > /dev/null 2>&1; then
    echo ">>> nvidia-smi not found — installing NVIDIA driver..."
    apt-get update -qq && apt-get install -y -qq python3-pip pciutils 2>/dev/null
    curl -fsSL https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py -o /tmp/install_gpu_driver.py
    python3 /tmp/install_gpu_driver.py 2>&1 || true
fi

echo ">>> Waiting for GPU driver (up to 5 min)..."
GPU_READY=false
for i in $(seq 1 60); do
    if nvidia-smi > /dev/null 2>&1; then
        GPU_READY=true
        break
    fi
    echo "    nvidia-smi not ready yet (attempt $i/60)..."
    sleep 5
done

if [ "$GPU_READY" = true ]; then
    echo ">>> GPU detected — installing PyTorch CUDA"
    nvidia-smi
    uv pip install torch --index-url https://download.pytorch.org/whl/cu124
    uv pip install xgboost --upgrade
else
    echo ">>> No GPU after 5 min — installing CPU PyTorch"
    slack_notify ":warning: *GPU driver failed to load* on $(hostname) — falling back to CPU"
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo ">>> Verifying torch..."
uv run python -c "import torch; print(f'torch={{torch.__version__}}, cuda={{torch.cuda.is_available()}}')"

echo ">>> Regenerating notebooks..."
uv run python scripts/make_training_notebooks.py

echo ">>> Testing imports..."
uv run python -c "
from f1_predictor.features.simulation_features import build_simulation_training_data
print('simulation_features OK')
from f1_predictor.models.sequence_architectures import SeqGRU_Shallow
print('sequence_architectures OK')
"

slack_notify ":white_check_mark: VM setup complete -- starting training"

# Start background health monitor
chmod +x scripts/vm_health_monitor.sh
nohup bash scripts/vm_health_monitor.sh > /tmp/health_monitor.log 2>&1 &
HEALTH_PID=$!
echo ">>> Health monitor started (PID $HEALTH_PID)"
{model_blocks}
# Stop health monitor
kill $HEALTH_PID 2>/dev/null || true

echo "=== All models complete: $(date) ==="
slack_notify ":checkered_flag: *All training complete* on $(hostname)"
echo "ALL_DONE" | gsutil -q cp - "gs://$BUCKET/$STAGING/ALL_TRAINING_DONE"
upload_log

# VM stays running — user will manually stop it
"""


def package_and_upload() -> None:
    print(">>> Packaging repo...")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(REPO_ROOT.rglob("*")):
            if p.is_file() and not _should_exclude(p.relative_to(REPO_ROOT)):
                rel = p.relative_to(REPO_ROOT)
                if rel.suffix == ".parquet" and "data/raw" in str(rel):
                    continue
                if rel.suffix == ".pkl":
                    continue
                if "data/training" in str(rel):
                    continue
                tar.add(p, arcname=str(rel))
    buf.seek(0)
    size_mb = len(buf.getvalue()) / (1024 * 1024)
    print(f">>> Tarball size: {size_mb:.1f} MB")

    print(f">>> Uploading to gs://{BUCKET}/{STAGING_PREFIX}/repo.tar.gz")
    client = storage.Client(project=PROJECT)
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(f"{STAGING_PREFIX}/repo.tar.gz")
    blob.upload_from_file(buf, content_type="application/gzip")
    print(">>> Upload complete.")


def create_vm(model_keys: list[str], *, cpu: bool = False, gpu: str = "l4") -> tuple[str, str]:
    """Create a GCE VM, trying zones in order. Returns (vm_name, zone)."""
    label = "-".join(k.lower() for k in model_keys)
    vm_name = f"f1-model-{label}-{int(time.time())}"
    startup_script = _make_startup_script(model_keys)

    instance_client = compute_v1.InstancesClient()
    image_client = compute_v1.ImagesClient()

    gpu_cfg = GPU_CONFIGS[gpu]

    if cpu:
        image_family = CPU_IMAGE_FAMILY
        image_project = CPU_IMAGE_PROJECT
    else:
        image_family = gpu_cfg["image_family"]
        image_project = gpu_cfg["image_project"]

    image = image_client.get_from_family(project=image_project, family=image_family)
    zones_to_try = [ZONE] if cpu else gpu_cfg["zones"]

    for zone in zones_to_try:
        if cpu:
            mt = "e2-standard-8"
        else:
            mt = gpu_cfg["machine_type"]
        machine_type = f"zones/{zone}/machineTypes/{mt}"

        disk = compute_v1.AttachedDisk(
            auto_delete=True,
            boot=True,
            initialize_params=compute_v1.AttachedDiskInitializeParams(
                disk_size_gb=50,
                source_image=image.self_link,
            ),
        )

        instance = compute_v1.Instance(
            name=vm_name,
            machine_type=machine_type,
            disks=[disk],
            network_interfaces=[
                compute_v1.NetworkInterface(
                    access_configs=[compute_v1.AccessConfig(name="External NAT")],
                )
            ],
            metadata=compute_v1.Metadata(
                items=[
                    compute_v1.Items(key="startup-script", value=startup_script),
                    compute_v1.Items(key="install-nvidia-driver", value="True"),
                ]
            ),
            service_accounts=[
                compute_v1.ServiceAccount(
                    email="default",
                    scopes=[
                        "https://www.googleapis.com/auth/devstorage.full_control",
                        "https://www.googleapis.com/auth/compute",
                    ],
                )
            ],
            scheduling=compute_v1.Scheduling(
                on_host_maintenance="TERMINATE" if not cpu else "MIGRATE",
                automatic_restart=False,
            ),
        )

        if not cpu:
            instance.guest_accelerators = [
                compute_v1.AcceleratorConfig(
                    accelerator_type=f"zones/{zone}/acceleratorTypes/{gpu_cfg['accelerator']}",
                    accelerator_count=1,
                )
            ]

        print(f">>> Creating VM: {vm_name} in {zone} ({gpu.upper()} GPU)...")
        try:
            op = instance_client.insert(project=PROJECT, zone=zone, instance_resource=instance)
            op.result()
            print(f">>> VM {vm_name} created in {zone}.")
            return vm_name, zone
        except Exception as e:
            if "ZONE_RESOURCE_POOL_EXHAUSTED" in str(e) or "STOCKOUT" in str(e):
                print(f"    {zone} stockout, trying next zone...")
                continue
            raise

    raise RuntimeError(f"All zones exhausted — could not create VM for Models {model_keys}")


def wait_for_completion(model_keys: list[str], poll_seconds: int = 120) -> bool:
    client = storage.Client(project=PROJECT)
    bucket = client.bucket(BUCKET)
    remaining = set(model_keys)

    print(f">>> Waiting for Models {model_keys} to finish (polling every {poll_seconds}s)...")
    while remaining:
        for m in list(remaining):
            blob = bucket.blob(f"{STAGING_PREFIX}/MODEL_{m}_DONE")
            if blob.exists():
                content = blob.download_as_text().strip()
                print(f">>> Model {m} training {content}.")
                remaining.discard(m)
        if remaining:
            time.sleep(poll_seconds)
            print(f"    ... still training: {sorted(remaining)}")

    all_blob = bucket.blob(f"{STAGING_PREFIX}/ALL_TRAINING_DONE")
    if all_blob.exists():
        print(">>> ALL models complete.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Train F1 models on GCE GPU VMs")
    parser.add_argument(
        "models",
        nargs="*",
        default=["G", "H", "I"],
        choices=["G", "H", "I"],
        help="Which models to train (default: G H I)",
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPU-only VMs")
    parser.add_argument(
        "--gpu",
        default="l4",
        choices=list(GPU_CONFIGS),
        help="GPU type: l4 (default), v100, a100",
    )
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for completion")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume: don't clear DONE signals for successfully completed models",
    )
    args = parser.parse_args()

    if not SLACK_WEBHOOK_URL:
        print("WARNING: SLACK_WEBHOOK_URL not set — Slack notifications will be disabled on the VM")

    # Clear previous completion signals (unless --resume)
    client = storage.Client(project=PROJECT)
    bucket = client.bucket(BUCKET)
    if not args.resume:
        for m in args.models:
            blob = bucket.blob(f"{STAGING_PREFIX}/MODEL_{m}_DONE")
            if blob.exists():
                blob.delete()
                print(f">>> Cleared previous MODEL_{m}_DONE signal")
        all_blob = bucket.blob(f"{STAGING_PREFIX}/ALL_TRAINING_DONE")
        if all_blob.exists():
            all_blob.delete()
            print(">>> Cleared ALL_TRAINING_DONE signal")

    package_and_upload()

    vm_name, zone = create_vm(args.models, cpu=args.cpu, gpu=args.gpu)
    print(f"\n  VM: {vm_name} in {zone}")
    print(f"  SSH: ssh -i ~/.ssh/google_compute_engine jchiriyankandath@<EXTERNAL_IP>")
    print(f"  Progress: tail -f /var/log/f1-model-g-progress.log")
    print(f"  Slack updates: every 5 min via webhook\n")

    if not args.no_wait:
        wait_for_completion(args.models)


if __name__ == "__main__":
    main()
