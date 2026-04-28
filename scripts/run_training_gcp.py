#!/usr/bin/env python3
"""Train models on GCE GPU VMs using the Python SDK (replaces gcloud CLI scripts)."""

from __future__ import annotations

import argparse
import io
import tarfile
import time
from pathlib import Path

from google.cloud import compute_v1, storage

PROJECT = "jowin-personal-2026"
ZONE = "us-central1-a"
FALLBACK_ZONES = [
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-east4-a",
    "us-east4-c",
    "us-west1-a",
    "us-west1-b",
    "us-east1-b",
    "us-east1-c",
]
BUCKET = "f1-predictor-artifacts-jowin"
STAGING_PREFIX = "staging/training-run"
RESULTS_PREFIX = "data"
GPU_IMAGE_FAMILY = "common-cu129-ubuntu-2204-nvidia-580"
GPU_IMAGE_PROJECT = "deeplearning-platform-release"
CPU_IMAGE_FAMILY = "debian-12"
CPU_IMAGE_PROJECT = "debian-cloud"

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


def _make_startup_script(model_key: str) -> str:
    m = MODELS[model_key]
    extra_pip = f"uv pip install {m['extra_pip']}" if m["extra_pip"] else ""
    return f"""#!/bin/bash
exec > /var/log/f1-training.log 2>&1
set -euo pipefail

upload_log() {{
    gsutil -q cp /var/log/f1-training.log "gs://$BUCKET/$STAGING/model_{model_key.lower()}_log.txt" 2>/dev/null || true
}}
trap upload_log EXIT

echo "=== Model {model_key} training begin: $(date) ==="

BUCKET="{BUCKET}"
STAGING="{STAGING_PREFIX}"
RESULTS="{RESULTS_PREFIX}"
WORK="/opt/f1-training"

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

mkdir -p data/processed/simulation data/training data/raw/model

echo ">>> Installing Python deps..."
uv python install 3.11
uv sync --frozen --group dev
uv pip install xgboost lightgbm optuna
{extra_pip}

if nvidia-smi > /dev/null 2>&1; then
    echo ">>> GPU detected — installing PyTorch CUDA"
    nvidia-smi
    uv pip install torch --index-url https://download.pytorch.org/whl/cu124
    uv pip install xgboost --upgrade
else
    echo ">>> No GPU — installing CPU PyTorch"
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo ">>> Verifying torch..."
uv run python -c "import torch; print(f'torch={{torch.__version__}}, cuda={{torch.cuda.is_available()}}')"

echo ">>> Regenerating notebooks..."
uv run python scripts/make_training_notebooks.py

echo ">>> Verifying notebook exists..."
ls -la notebooks/{m['notebook']}

echo ">>> Testing imports..."
uv run python -c "
from f1_predictor.features.simulation_features import build_simulation_training_data
print('simulation_features OK')
from f1_predictor.models.sequence_architectures import SeqGRU_Shallow
print('sequence_architectures OK')
"

echo "=== Running Model {model_key}: $(date) ==="
uv run jupyter nbconvert --to notebook --execute \\
    --ExecutePreprocessor.timeout={m['timeout']} \\
    notebooks/{m['notebook']} \\
    --output {m['notebook']} 2>&1
NBCONVERT_EXIT=$?
echo ">>> nbconvert exit code: $NBCONVERT_EXIT"

upload_log

echo "=== Uploading results: $(date) ==="
gsutil -m -q cp data/training/{m['artifact_prefix']}_*.parquet "gs://$BUCKET/$RESULTS/training/" 2>/dev/null || true
gsutil -m -q cp data/raw/model/{m['model_prefix']}_*.pkl "gs://$BUCKET/$RESULTS/raw/model/" 2>/dev/null || true
gsutil -q cp notebooks/{m['notebook']} "gs://$BUCKET/$RESULTS/notebooks/" 2>/dev/null || true

if [ "$NBCONVERT_EXIT" -eq 0 ] && ls data/training/{m['artifact_prefix']}_*.parquet 1>/dev/null 2>&1; then
    echo "DONE" | gsutil -q cp - "gs://$BUCKET/$STAGING/MODEL_{model_key}_DONE"
    echo "=== Model {model_key} training SUCCESS: $(date) ==="
else
    echo "FAILED" | gsutil -q cp - "gs://$BUCKET/$STAGING/MODEL_{model_key}_DONE"
    echo "=== Model {model_key} training FAILED: $(date) ==="
fi

echo "=== Model {model_key} training complete: $(date) ==="

VM_NAME=$(curl -s -H "Metadata-Flavor: Google" \\
    http://metadata.google.internal/computeMetadata/v1/instance/name)
VM_ZONE=$(curl -s -H "Metadata-Flavor: Google" \\
    http://metadata.google.internal/computeMetadata/v1/instance/zone | rev | cut -d'/' -f1 | rev)
gcloud compute instances delete "$VM_NAME" --zone="$VM_ZONE" --quiet &
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


def create_vm(model_key: str, *, cpu: bool = False) -> tuple[str, str]:
    """Create a GCE VM, trying FALLBACK_ZONES in order. Returns (vm_name, zone)."""
    vm_name = f"f1-model-{model_key.lower()}-{int(time.time())}"
    startup_script = _make_startup_script(model_key)

    instance_client = compute_v1.InstancesClient()
    image_client = compute_v1.ImagesClient()

    if cpu:
        image_family = CPU_IMAGE_FAMILY
        image_project = CPU_IMAGE_PROJECT
    else:
        image_family = GPU_IMAGE_FAMILY
        image_project = GPU_IMAGE_PROJECT

    image = image_client.get_from_family(project=image_project, family=image_family)

    zones_to_try = [ZONE] if cpu else FALLBACK_ZONES

    for zone in zones_to_try:
        machine_type = f"zones/{zone}/machineTypes/{'e2-standard-8' if cpu else 'g2-standard-8'}"

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
                    accelerator_type=f"zones/{zone}/acceleratorTypes/nvidia-l4",
                    accelerator_count=1,
                )
            ]

        print(f">>> Creating VM: {vm_name} in {zone} ({'GPU' if not cpu else 'CPU'})...")
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

    raise RuntimeError(f"All zones exhausted — could not create VM for Model {model_key}")


def wait_for_completion(model_key: str, poll_seconds: int = 120) -> bool:
    client = storage.Client(project=PROJECT)
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(f"{STAGING_PREFIX}/MODEL_{model_key}_DONE")

    print(f">>> Waiting for Model {model_key} to finish (polling every {poll_seconds}s)...")
    while True:
        if blob.exists():
            print(f">>> Model {model_key} training COMPLETE.")
            return True
        time.sleep(poll_seconds)
        print(f"    ... still training Model {model_key}")


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
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for completion")
    parser.add_argument(
        "--parallel", action="store_true", help="Create all VMs at once instead of sequentially"
    )
    args = parser.parse_args()

    # Clear any previous completion signals
    client = storage.Client(project=PROJECT)
    bucket = client.bucket(BUCKET)
    for m in args.models:
        blob = bucket.blob(f"{STAGING_PREFIX}/MODEL_{m}_DONE")
        if blob.exists():
            blob.delete()
            print(f">>> Cleared previous MODEL_{m}_DONE signal")

    package_and_upload()

    if args.parallel:
        vms = {}
        for m in args.models:
            vms[m] = create_vm(m, cpu=args.cpu)
        if not args.no_wait:
            for m in args.models:
                wait_for_completion(m)
    else:
        for m in args.models:
            info = MODELS[m]
            vm_name, zone = create_vm(m, cpu=args.cpu)
            print(f"\n  Monitor: gcloud compute ssh {vm_name} --zone={zone} --command='tail -f /var/log/f1-training.log'")
            print(f"  Estimated time: ~{info['est_minutes']} min\n")
            if not args.no_wait:
                wait_for_completion(m)
            else:
                print(f"  Check: gsutil stat gs://{BUCKET}/{STAGING_PREFIX}/MODEL_{m}_DONE\n")

    print("\n>>> All done.")


if __name__ == "__main__":
    main()
