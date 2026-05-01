#!/usr/bin/env python3
"""Monitor GCE training VMs for Models G, H, I and post updates to Slack.

Usage:
    uv run python scripts/monitor_training.py [--interval 120]

Polls GCS for completion signals and VM logs, sends updates to
#claude-updates Slack channel via Composio.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime

BUCKET = "f1-predictor-artifacts-jowin"
STAGING_PREFIX = "staging/training-run"
SLACK_CHANNEL = "C0AV35RHKF1"  # #claude-updates

MODELS = {
    "G": {
        "signal": f"gs://{BUCKET}/{STAGING_PREFIX}/MODEL_G_DONE",
        "artifacts": [
            f"gs://{BUCKET}/data/training/model_G_*.parquet",
            f"gs://{BUCKET}/data/raw/model/Model_G_*.pkl",
        ],
        "name": "Model G (Temporal Sequence)",
    },
    "H": {
        "signal": f"gs://{BUCKET}/{STAGING_PREFIX}/MODEL_H_DONE",
        "artifacts": [
            f"gs://{BUCKET}/data/training/model_H_*.parquet",
            f"gs://{BUCKET}/data/raw/model/Model_H_*.pkl",
        ],
        "name": "Model H (Delta + Monte Carlo)",
    },
    "I": {
        "signal": f"gs://{BUCKET}/{STAGING_PREFIX}/MODEL_I_DONE",
        "artifacts": [
            f"gs://{BUCKET}/data/training/model_I_*.parquet",
            f"gs://{BUCKET}/data/raw/model/Model_I_*.pkl",
        ],
        "name": "Model I (Quantile/Uncertainty)",
    },
}


def _run(cmd: list[str], timeout: int = 30) -> tuple[bool, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, r.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, ""


def check_signal(model_key: str) -> bool:
    signal_path = MODELS[model_key]["signal"]
    ok, _ = _run(["gsutil", "stat", signal_path])
    return ok


def count_artifacts(model_key: str) -> dict[str, int]:
    counts = {}
    for pattern in MODELS[model_key]["artifacts"]:
        ok, out = _run(["gsutil", "ls", pattern], timeout=15)
        if ok and out:
            counts[pattern.split("/")[-1]] = len(out.strip().split("\n"))
        else:
            counts[pattern.split("/")[-1]] = 0
    return counts


def find_running_vms() -> list[str]:
    ok, out = _run([
        "gcloud", "compute", "instances", "list",
        "--filter", "name~f1-model-",
        "--format", "value(name,zone,status)",
    ], timeout=60)
    if ok and out:
        return out.strip().split("\n")
    return []


def send_slack(message: str) -> bool:
    try:
        from composio import ComposioToolSet
        toolset = ComposioToolSet()
        result = toolset.execute_action(
            action="SLACK_SEND_MESSAGE",
            params={
                "channel": SLACK_CHANNEL,
                "markdown_text": message,
            },
        )
        return result.get("successful", False)
    except Exception:
        pass

    # Fallback: try using composio CLI
    try:
        payload = json.dumps({
            "channel": SLACK_CHANNEL,
            "markdown_text": message,
        })
        ok, _ = _run([
            "composio", "execute", "SLACK_SEND_MESSAGE",
            "--params", payload,
        ], timeout=30)
        return ok
    except Exception:
        print("[WARN] Could not send Slack message")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor GCE training")
    parser.add_argument("--interval", type=int, default=120, help="Poll interval in seconds")
    args = parser.parse_args()

    completed: set[str] = set()
    notified_started: set[str] = set()
    start_time = time.time()

    print(f"Monitoring Models G, H, I (interval={args.interval}s)")
    print(f"Slack channel: #claude-updates ({SLACK_CHANNEL})")

    while len(completed) < len(MODELS):
        now = datetime.now(UTC).strftime("%H:%M UTC")
        elapsed = int((time.time() - start_time) / 60)

        for key, info in MODELS.items():
            if key in completed:
                continue

            if check_signal(key):
                completed.add(key)
                artifacts = count_artifacts(key)
                art_str = ", ".join(f"{v} {k}" for k, v in artifacts.items())

                msg = (
                    f"## {info['name']} - COMPLETE\n\n"
                    f"Training finished at {now} ({elapsed} min elapsed).\n\n"
                    f"**Artifacts:** {art_str}\n\n"
                    f"**Progress:** {len(completed)}/{len(MODELS)} models done"
                )
                if len(completed) == len(MODELS):
                    msg += (
                        "\n\n**All training complete!** "
                        "Run `bash scripts/fetch_training_results.sh`"
                    )

                send_slack(msg)
                print(f"[{now}] {info['name']} COMPLETE ({art_str})")

        # Check for running VMs (first iteration only to report start)
        if not notified_started:
            vms = find_running_vms()
            if vms:
                notified_started.update(MODELS.keys())

        remaining = [MODELS[k]["name"] for k in MODELS if k not in completed]
        if remaining:
            print(f"[{now}] Waiting... {len(completed)}/{len(MODELS)} done. "
                  f"Remaining: {', '.join(remaining)}")
            time.sleep(args.interval)

    # Final summary
    total_min = int((time.time() - start_time) / 60)
    summary = (
        f"## Training Complete\n\n"
        f"All 3 models finished in **{total_min} minutes**.\n\n"
        f"**Next steps:**\n"
        f"1. `bash scripts/fetch_training_results.sh`\n"
        f"2. Review notebooks in `notebooks/`\n"
        f"3. Run comparison notebook"
    )
    send_slack(summary)
    print(f"\nAll training complete in {total_min} minutes!")


if __name__ == "__main__":
    main()
