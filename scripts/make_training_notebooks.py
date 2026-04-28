# ruff: noqa: RUF001  — notebook display strings use Unicode
"""Generate training notebooks for Models A, B, C, D, E, F, G, H, I."""

import json
import uuid
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def make_cell(cell_type: str, source: str, **kwargs) -> dict:
    lines = source.split("\n")
    source_lines = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        source_lines.append(lines[-1])
    cell = {
        "cell_type": cell_type,
        "metadata": kwargs.get("metadata", {}),
        "source": source_lines,
        "id": uuid.uuid4().hex[:8],
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def md(text: str) -> dict:
    return make_cell("markdown", text)


def code(text: str) -> dict:
    return make_cell("code", text)


def make_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# Shared code blocks
# ---------------------------------------------------------------------------

CHDIR = """\
import os
from pathlib import Path

if not (Path.cwd() / "pyproject.toml").exists():
    # We're likely in notebooks/ — go up to repo root
    for p in [Path.cwd().parent, Path.cwd().parent.parent]:
        if (p / "pyproject.toml").exists():
            os.chdir(p)
            break

print(f"Working directory: {Path.cwd()}")"""

IMPORTS = """\
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

from f1_predictor.features.splits import ExpandingWindowSplit, LeaveOneSeasonOut
from f1_predictor.data.storage import (
    load_from_gcs_or_local,
    load_training_parquet,
    save_training_parquet,
    save_model_pickle as gcs_save_model_pickle,
    save_notebook,
    sync_training_from_gcs,
)
from f1_predictor.models.gpu import (
    detect_gpu_backend, get_lightgbm_device, get_torch_device, get_xgboost_device,
)

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

TRAINING_DIR = Path("data/training")
MODEL_DIR = Path("data/raw/model")
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# GPU detection (supports NVIDIA CUDA and AMD ROCm)
GPU_BACKEND, GPU_NAME = detect_gpu_backend()
TORCH_DEVICE = get_torch_device()
print(f"GPU backend: {GPU_BACKEND} ({GPU_NAME})")
print(f"PyTorch device: {TORCH_DEVICE}")

# Deep learning models (PyTorch — works on both CUDA and ROCm via HIP)
DL_AVAILABLE = False
try:
    from f1_predictor.models.architectures import GRU2Layer, FTTransformerWrapper, MLP3Layer
    DL_AVAILABLE = TORCH_DEVICE != "cpu"
    print(f"DL models available: {DL_AVAILABLE}")
except (ImportError, NameError):
    print("DL models not available (torch/rtdl not installed)")"""

HELPERS = '''\
NAN_TOLERANT = {
    "XGBoost", "XGBoost_DART", "XGBoost_Linear",
    "LightGBM", "LightGBM_DART", "LightGBM_GOSS",
    "XGBoost_Conservative", "XGBoost_Deep",
    "LightGBM_Shallow", "LightGBM_Deep",
}

DL_SKIP_OPTUNA = {"GRU_2layer", "FT_Transformer", "MLP_3layer"}


def get_candidates():
    """Return dict of model_name -> model instance. GPU-accelerated where possible."""
    xgb_device = get_xgboost_device(GPU_BACKEND)
    lgb_device = get_lightgbm_device(GPU_BACKEND)

    candidates = {
        # XGBoost variants (GPU on CUDA, CPU on ROCm)
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300, n_jobs=-1, random_state=42, verbosity=0, **xgb_device),
        "XGBoost_DART": xgb.XGBRegressor(
            n_estimators=300, booster="dart", n_jobs=-1,
            random_state=42, verbosity=0, **xgb_device),
        "XGBoost_Linear": xgb.XGBRegressor(
            n_estimators=300, booster="gblinear", n_jobs=-1,
            random_state=42, verbosity=0, **xgb_device),
        # LightGBM variants (GPU via OpenCL on both CUDA and ROCm)
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=300, n_jobs=-1, random_state=42, verbose=-1, **lgb_device),
        "LightGBM_DART": lgb.LGBMRegressor(
            n_estimators=300, boosting_type="dart", n_jobs=-1,
            random_state=42, verbose=-1, **lgb_device),
        "LightGBM_GOSS": lgb.LGBMRegressor(
            n_estimators=300, boosting_type="goss", n_jobs=-1,
            random_state=42, verbose=-1, **lgb_device),
        # Extra tree-based variants
        "XGBoost_Conservative": xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            n_jobs=-1, random_state=42, verbosity=0, **xgb_device),
        "LightGBM_Shallow": lgb.LGBMRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            n_jobs=-1, random_state=42, verbose=-1, **lgb_device),
        "XGBoost_Deep": xgb.XGBRegressor(
            n_estimators=300, max_depth=10, learning_rate=0.1,
            n_jobs=-1, random_state=42, verbosity=0, **xgb_device),
        "LightGBM_Deep": lgb.LGBMRegressor(
            n_estimators=300, max_depth=10, learning_rate=0.1,
            n_jobs=-1, random_state=42, verbose=-1, **lgb_device),
    }

    # Deep learning candidates (PyTorch — works on both CUDA and ROCm via HIP)
    if DL_AVAILABLE:
        n_feat = len(FEATURE_COLS)
        candidates["GRU_2layer"] = GRU2Layer(input_dim=n_feat)
        candidates["FT_Transformer"] = FTTransformerWrapper(n_features=n_feat)

    print(f"Candidates ({len(candidates)}): {list(candidates.keys())}")
    return candidates


def get_candidates_c():
    """Return dict of candidates for Model C (race-level, includes MLP)."""
    xgb_device = get_xgboost_device(GPU_BACKEND)
    lgb_device = get_lightgbm_device(GPU_BACKEND)

    candidates = {
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300, n_jobs=-1, random_state=42, verbosity=0, **xgb_device),
        "XGBoost_DART": xgb.XGBRegressor(
            n_estimators=300, booster="dart", n_jobs=-1,
            random_state=42, verbosity=0, **xgb_device),
        "XGBoost_Linear": xgb.XGBRegressor(
            n_estimators=300, booster="gblinear", n_jobs=-1,
            random_state=42, verbosity=0, **xgb_device),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=300, n_jobs=-1, random_state=42, verbose=-1, **lgb_device),
        "LightGBM_DART": lgb.LGBMRegressor(
            n_estimators=300, boosting_type="dart", n_jobs=-1,
            random_state=42, verbose=-1, **lgb_device),
        "LightGBM_GOSS": lgb.LGBMRegressor(
            n_estimators=300, boosting_type="goss", n_jobs=-1,
            random_state=42, verbose=-1, **lgb_device),
        "XGBoost_Conservative": xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            n_jobs=-1, random_state=42, verbosity=0, **xgb_device),
        "LightGBM_Shallow": lgb.LGBMRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            n_jobs=-1, random_state=42, verbose=-1, **lgb_device),
        "XGBoost_Deep": xgb.XGBRegressor(
            n_estimators=300, max_depth=10, learning_rate=0.1,
            n_jobs=-1, random_state=42, verbosity=0, **xgb_device),
        "LightGBM_Deep": lgb.LGBMRegressor(
            n_estimators=300, max_depth=10, learning_rate=0.1,
            n_jobs=-1, random_state=42, verbose=-1, **lgb_device),
    }

    if DL_AVAILABLE:
        n_feat = len(FEATURE_COLS)
        candidates["MLP_3layer"] = MLP3Layer(input_dim=n_feat)

    print(f"Candidates ({len(candidates)}): {list(candidates.keys())}")
    return candidates


def cv_evaluate(model, X, y, splitter, groups):
    """Evaluate model across CV folds. Returns dict with fold and mean metrics."""
    fold_rmse, fold_mae = [], []
    for train_idx, val_idx in splitter.split(groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        import sklearn.base
        m = sklearn.base.clone(model)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_val)
        fold_rmse.append(np.sqrt(mean_squared_error(y_val, preds)))
        fold_mae.append(mean_absolute_error(y_val, preds))
    return {
        "fold_rmse": fold_rmse,
        "fold_mae": fold_mae,
        "mean_rmse": np.mean(fold_rmse),
        "std_rmse": np.std(fold_rmse),
        "mean_mae": np.mean(fold_mae),
    }


def screen_models(candidates, X, y, splitter, groups):
    """Screen all candidates via CV. Returns sorted DataFrame."""
    rows = []
    for name, model in candidates.items():
        print(f"  Screening {name}...")
        result = cv_evaluate(model, X, y, splitter, groups)
        rows.append({"model": name, **result})
    df = pd.DataFrame(rows).sort_values("mean_rmse").reset_index(drop=True)
    return df'''

OPTUNA_HELPERS = '''\
def _xgb_base_space(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1500),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )

def _lgb_base_space(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1500),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )

def get_optuna_param_space(name, trial):
    """Return HP dict for a given model name and Optuna trial."""
    xgb_device = get_xgboost_device(GPU_BACKEND)
    lgb_device = get_lightgbm_device(GPU_BACKEND)

    if name == "XGBoost":
        params = _xgb_base_space(trial)
        params.update(n_jobs=-1, random_state=42, verbosity=0, **xgb_device)
        return params
    elif name == "XGBoost_DART":
        params = _xgb_base_space(trial)
        params.update(booster="dart", n_jobs=-1, random_state=42, verbosity=0, **xgb_device)
        params["rate_drop"] = trial.suggest_float("rate_drop", 0.01, 0.5)
        return params
    elif name == "XGBoost_Linear":
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 1500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            booster="gblinear", n_jobs=-1, random_state=42, verbosity=0, **xgb_device,
        )
        return params
    elif name in ("XGBoost_Conservative", "XGBoost_Deep"):
        params = _xgb_base_space(trial)
        params.update(n_jobs=-1, random_state=42, verbosity=0, **xgb_device)
        return params
    elif name == "LightGBM":
        params = _lgb_base_space(trial)
        params.update(n_jobs=-1, random_state=42, verbose=-1, **lgb_device)
        return params
    elif name == "LightGBM_DART":
        params = _lgb_base_space(trial)
        params.update(boosting_type="dart", n_jobs=-1, random_state=42, verbose=-1, **lgb_device)
        params["drop_rate"] = trial.suggest_float("drop_rate", 0.01, 0.5)
        return params
    elif name == "LightGBM_GOSS":
        params = _lgb_base_space(trial)
        params.pop("subsample", None)
        params.update(boosting_type="goss", n_jobs=-1, random_state=42, verbose=-1, **lgb_device)
        return params
    elif name in ("LightGBM_Shallow", "LightGBM_Deep"):
        params = _lgb_base_space(trial)
        params.update(n_jobs=-1, random_state=42, verbose=-1, **lgb_device)
        return params
    return {}


MODEL_CLASSES = {
    "XGBoost": xgb.XGBRegressor,
    "XGBoost_DART": xgb.XGBRegressor,
    "XGBoost_Linear": xgb.XGBRegressor,
    "XGBoost_Conservative": xgb.XGBRegressor,
    "XGBoost_Deep": xgb.XGBRegressor,
    "LightGBM": lgb.LGBMRegressor,
    "LightGBM_DART": lgb.LGBMRegressor,
    "LightGBM_GOSS": lgb.LGBMRegressor,
    "LightGBM_Shallow": lgb.LGBMRegressor,
    "LightGBM_Deep": lgb.LGBMRegressor,
}
if DL_AVAILABLE:
    MODEL_CLASSES.update({
        "GRU_2layer": GRU2Layer,
        "FT_Transformer": FTTransformerWrapper,
        "MLP_3layer": MLP3Layer,
    })


def reconstruct_params(name, best_params):
    """Translate flat Optuna best_params back to model constructor args."""
    params = dict(best_params)
    xgb_device = get_xgboost_device(GPU_BACKEND)
    lgb_device = get_lightgbm_device(GPU_BACKEND)

    if name == "XGBoost":
        params.update(n_jobs=-1, random_state=42, verbosity=0, **xgb_device)
    elif name == "XGBoost_DART":
        params.update(booster="dart", n_jobs=-1, random_state=42, verbosity=0, **xgb_device)
    elif name == "XGBoost_Linear":
        params.update(booster="gblinear", n_jobs=-1, random_state=42, verbosity=0, **xgb_device)
    elif name in ("XGBoost_Conservative", "XGBoost_Deep"):
        params.update(n_jobs=-1, random_state=42, verbosity=0, **xgb_device)
    elif name == "LightGBM":
        params.update(n_jobs=-1, random_state=42, verbose=-1, **lgb_device)
    elif name == "LightGBM_DART":
        params.update(boosting_type="dart", n_jobs=-1, random_state=42, verbose=-1, **lgb_device)
    elif name == "LightGBM_GOSS":
        params.pop("subsample", None)
        params.update(boosting_type="goss", n_jobs=-1, random_state=42, verbose=-1, **lgb_device)
    elif name in ("LightGBM_Shallow", "LightGBM_Deep"):
        params.update(n_jobs=-1, random_state=42, verbose=-1, **lgb_device)
    return params


def run_optuna_round(name, X, y, splitter, groups, n_trials):
    """Run Optuna study for a single model. Returns best params and best RMSE.
    DL models skip Optuna (fixed HPs) — returns empty params and screening RMSE."""
    if name in DL_SKIP_OPTUNA:
        result = cv_evaluate(MODEL_CLASSES[name](), X, y, splitter, groups)
        return {}, result["mean_rmse"]

    def objective(trial):
        params = get_optuna_param_space(name, trial)
        model_cls = MODEL_CLASSES[name]
        model = model_cls(**params)
        result = cv_evaluate(model, X, y, splitter, groups)
        return result["mean_rmse"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, catch=(Exception,), show_progress_bar=False)
    return study.best_params, study.best_value'''

SAVE_ARTIFACTS = '''\
def save_predictions(model, X, y, id_df, model_type, model_name, split_name):
    """Save prediction parquet locally and to GCS."""
    preds = model.predict(X)
    out = id_df.copy()
    out["y_true"] = y.values
    out["y_pred"] = preds
    fname = f"model_{model_type}_{model_name}_{split_name}.parquet"
    uri = save_training_parquet(out, fname, TRAINING_DIR)
    print(f"  Saved {fname} -> {uri}")
    return preds


def save_model_pkl(model, model_type, model_name):
    """Save model pickle locally and to GCS."""
    fname = f"Model_{model_type}_{model_name}.pkl"
    uri = gcs_save_model_pickle(model, fname, MODEL_DIR)
    print(f"  Saved {fname} -> {uri}")'''


PROGRESS_LOGGER = """\
import sys
from datetime import datetime, timezone

class ProgressLogger:
    def __init__(self, model_key, log_dir="/var/log"):
        self.model_key = model_key
        self.log_path = f"{{log_dir}}/f1-model-{{model_key.lower()}}-progress.log"
        try:
            self._f = open(self.log_path, "a", buffering=1)
        except OSError:
            self._f = None

    def log(self, msg):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"[Model {{self.model_key}}] [{{ts}}] {{msg}}"
        print(line, flush=True)
        if self._f:
            self._f.write(line + "\\n")
            self._f.flush()

    def round_header(self, round_num, desc):
        self.log(f"========== ROUND {{round_num}}: {{desc}} ==========")

    def screening(self, name, idx, total, rmse=None, error=None):
        if error:
            self.log(f"Screening {{idx}}/{{total}} {{name}} -- FAILED: {{error}}")
        else:
            self.log(f"Screening {{idx}}/{{total}} {{name}} -- RMSE: {{rmse:.6f}}")

    def optuna_trial(self, name, trial_num, total, rmse, best_rmse):
        self.log(f"{{name}} trial {{trial_num}}/{{total}} -- RMSE: {{rmse:.6f}} (best: {{best_rmse:.6f}})")

    def model_complete(self, name, round_num, rmse):
        self.log(f"{{name}} Round {{round_num}} COMPLETE -- best RMSE: {{rmse:.6f}}")

    def close(self):
        if self._f:
            self._f.close()

progress = ProgressLogger("{model_key}")"""


SLACK_NOTIFIER = """\
import json as _json_slack
import os as _os_slack
from urllib.request import Request as _SlackReq, urlopen as _slack_urlopen

class SlackNotifier:
    def __init__(self, model_key):
        self.model_key = model_key
        self.webhook_url = _os_slack.environ.get("SLACK_WEBHOOK_URL", "")
        self.enabled = bool(self.webhook_url)

    def send(self, text):
        if not self.enabled:
            return
        try:
            data = _json_slack.dumps({{"text": text}}).encode()
            req = _SlackReq(self.webhook_url, data=data,
                            headers={{"Content-Type": "application/json"}})
            _slack_urlopen(req, timeout=10)
        except Exception:
            pass

    def round_start(self, round_num, desc, n_models):
        self.send(f":racing_car: *Model {{self.model_key}} -- Round {{round_num}}*\\n{{desc}} ({{n_models}} models)")

    def round_complete(self, round_num, summary):
        self.send(f":checkered_flag: *Model {{self.model_key}} -- Round {{round_num}} complete*\\n{{summary}}")

    def model_start(self):
        self.send(f":rocket: *Model {{self.model_key}} training STARTED*")

    def model_complete(self, best_model, best_rmse):
        self.send(f":tada: *Model {{self.model_key}} training COMPLETE*\\nBest: {{best_model}} (RMSE: {{best_rmse:.6f}})")

    def architecture_done(self, name, round_num, rmse):
        self.send(f":gear: Model {{self.model_key}} R{{round_num}} -- {{name}} done (RMSE: {{rmse:.6f}})")

    def error(self, context, error_msg):
        self.send(f":rotating_light: *Model {{self.model_key}} ERROR* -- {{context}}: {{error_msg}}")

slack = SlackNotifier("{model_key}")"""


CHECKPOINT_MANAGER = """\
import json as _json_ckpt
import subprocess as _sp_ckpt
from datetime import datetime as _dt_ckpt, timezone as _tz_ckpt
from pathlib import Path as _Path_ckpt

class CheckpointManager:
    def __init__(self, model_key, local_base="/opt/f1-training/checkpoints",
                 bucket="f1-predictor-artifacts-jowin",
                 gcs_prefix="staging/training-run/checkpoints"):
        self.model_key = model_key
        self.local_dir = _Path_ckpt(local_base) / f"model_{{model_key}}"
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.gcs_prefix = f"gs://{{bucket}}/{{gcs_prefix}}/model_{{model_key}}"
        self._sync_from_gcs()

    def _sync_from_gcs(self):
        try:
            _sp_ckpt.run(
                ["gsutil", "-m", "-q", "cp", "-r",
                 f"{{self.gcs_prefix}}/*", str(self.local_dir) + "/"],
                capture_output=True, timeout=60)
        except Exception:
            pass

    def _upload(self, local_path):
        name = _Path_ckpt(local_path).name
        try:
            _sp_ckpt.run(
                ["gsutil", "-q", "cp", str(local_path),
                 f"{{self.gcs_prefix}}/{{name}}"],
                capture_output=True, timeout=30)
        except Exception:
            pass

    def save_checkpoint(self, round_num, arch_name, rmse, best_params, **extra):
        data = {{
            "model_key": self.model_key,
            "round": round_num,
            "architecture": arch_name,
            "rmse": rmse,
            "best_params": best_params,
            "timestamp": _dt_ckpt.now(_tz_ckpt.utc).isoformat(),
            **extra,
        }}
        path = self.local_dir / f"round_{{round_num}}_{{arch_name}}.json"
        path.write_text(_json_ckpt.dumps(data, indent=2, default=str))
        self._upload(path)

    def load_checkpoint(self, round_num, arch_name):
        path = self.local_dir / f"round_{{round_num}}_{{arch_name}}.json"
        if path.exists():
            return _json_ckpt.loads(path.read_text())
        return None

    def get_completed(self, round_num):
        result = {{}}
        for p in sorted(self.local_dir.glob(f"round_{{round_num}}_*.json")):
            if p.stem.endswith("_summary"):
                continue
            data = _json_ckpt.loads(p.read_text())
            result[data["architecture"]] = data
        return result

    def save_round_summary(self, round_num, results_list, top_names):
        data = {{"round": round_num, "results": results_list, "top_names": top_names,
                "timestamp": _dt_ckpt.now(_tz_ckpt.utc).isoformat()}}
        path = self.local_dir / f"round_{{round_num}}_summary.json"
        path.write_text(_json_ckpt.dumps(data, indent=2, default=str))
        self._upload(path)

    def load_round_summary(self, round_num):
        path = self.local_dir / f"round_{{round_num}}_summary.json"
        if path.exists():
            return _json_ckpt.loads(path.read_text())
        return None

ckpt = CheckpointManager("{model_key}")"""


# ---------------------------------------------------------------------------
# Model A notebook
# ---------------------------------------------------------------------------


def make_model_a() -> list[dict]:
    cells = [
        md(
            "# 05a — Model A Training: Lap + Tyre (2019-2024)\n\n"
            "Predicts **lap-level position** using 9 features including tyre data.\n"
            "CV: LeaveOneSeasonOut (test season = 2024)."
        ),
        md("## 0. Setup"),
        code(CHDIR),
        code(IMPORTS),
        code(HELPERS),
        code(OPTUNA_HELPERS),
        code(SAVE_ARTIFACTS),
        md("## 1. Load Features"),
        code("""\
df = load_from_gcs_or_local(
    "data/processed/lap_tyre/features_laps_tyre.parquet",
    Path("data/processed/lap_tyre/features_laps_tyre.parquet"),
)
print(f"Shape: {df.shape}")
df.head()"""),
        code("""\
FEATURE_COLS = [
    "gap_to_leader", "lap_time_delta_race_median", "gap_to_ahead",
    "position_change_from_lap1", "tire_life", "race_progress_pct",
    "degradation_rate", "compound_pace_delta", "pit_stop_count",
]
TARGET = "position"
ID_COLS = ["season", "round", "event_name", "driver_abbrev", "team", "lap_number"]

df = df.dropna(subset=[TARGET]).reset_index(drop=True)

X = df[FEATURE_COLS]
y = df[TARGET]
groups = df["season"].values
print(f"Features: {X.shape}, Target: {y.shape}")
print(f"NaN counts:\\n{X.isna().sum()}")"""),
        md("## 2. CV Splitter"),
        code("""\
splitter = LeaveOneSeasonOut(test_season=2024)
print(f"CV folds: {splitter.get_n_splits()}")
for i, (tr, va) in enumerate(splitter.split(groups)):
    tr_seasons = sorted(set(groups[tr]))
    va_seasons = sorted(set(groups[va]))
    print(f"  Fold {i}: train seasons={tr_seasons}, val seasons={va_seasons}, "
          f"train={len(tr):,}, val={len(va):,}")"""),
        md("## 3. Round 1 — Screen Models (default params)"),
        code("""\
candidates = get_candidates()
r1_results = screen_models(candidates, X, y, splitter, groups)
r1_results[["model", "mean_rmse", "std_rmse", "mean_mae"]]"""),
        code("""\
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(r1_results["model"], r1_results["mean_rmse"], xerr=r1_results["std_rmse"])
ax.set_xlabel("Mean RMSE (CV)")
ax.set_title("Round 1: Model Screening — Model A")
ax.invert_yaxis()
plt.tight_layout()
plt.show()"""),
        code("""\
top7_names = r1_results["model"].head(7).tolist()
print(f"Advancing to Round 2: {top7_names}")
eliminated = r1_results["model"].iloc[7:].tolist()
print(f"Eliminated: {eliminated}")"""),
        md("## 4. Round 2 — Optuna HP Tuning (top 7, 10 trials each)"),
        code("""\
r2_results = []
for name in top7_names:
    print(f"Tuning {name}...")
    best_params, best_rmse = run_optuna_round(name, X, y, splitter, groups, n_trials=10)
    r2_results.append({"model": name, "best_rmse": best_rmse, "best_params": best_params})
    print(f"  Best RMSE: {best_rmse:.4f}")

r2_df = pd.DataFrame(r2_results).sort_values("best_rmse").reset_index(drop=True)
r2_df[["model", "best_rmse"]]"""),
        code("""\
top5_names = r2_df["model"].head(5).tolist()
r2_best_params = {row["model"]: row["best_params"] for _, row in r2_df.iterrows()}
print(f"Advancing to Round 3: {top5_names}")"""),
        md("## 5. Round 3 — Final HP Tuning (top 5, 15 trials each)"),
        code("""\
r3_results = []
for name in top5_names:
    print(f"Fine-tuning {name}...")
    best_params, best_rmse = run_optuna_round(name, X, y, splitter, groups, n_trials=15)
    r3_results.append({"model": name, "best_rmse": best_rmse, "best_params": best_params})
    print(f"  Best RMSE: {best_rmse:.4f}")

r3_df = pd.DataFrame(r3_results).sort_values("best_rmse").reset_index(drop=True)
r3_best_params = {row["model"]: row["best_params"] for _, row in r3_df.iterrows()}
r3_df[["model", "best_rmse"]]"""),
        md("## 6. Test Set Evaluation"),
        code("""\
train_idx, test_idx = splitter.get_test_split(groups)
X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train_full, y_test = y.iloc[train_idx], y.iloc[test_idx]
id_train, id_test = df[ID_COLS].iloc[train_idx], df[ID_COLS].iloc[test_idx]

print(f"Train: {X_train_full.shape}, Test: {X_test.shape}")
print(f"Test season(s): {sorted(df['season'].iloc[test_idx].unique())}")"""),
        code("""\
final_results = []
for name in top5_names:
    if name in DL_SKIP_OPTUNA:
        model_cls = MODEL_CLASSES[name]
        model = model_cls(input_dim=len(FEATURE_COLS))
    else:
        params = reconstruct_params(name, r3_best_params[name])
        model_cls = MODEL_CLASSES[name]
        model = model_cls(**params)
    model.fit(X_train_full, y_train_full)

    train_preds = model.predict(X_train_full)
    train_rmse = np.sqrt(mean_squared_error(y_train_full, train_preds))
    train_mae = mean_absolute_error(y_train_full, train_preds)

    test_preds = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)

    val_rmse = r3_df.loc[r3_df["model"] == name, "best_rmse"].values[0]

    final_results.append({
        "model": name,
        "train_rmse": train_rmse, "train_mae": train_mae,
        "val_rmse": val_rmse,
        "test_rmse": test_rmse, "test_mae": test_mae,
        "overfit_gap": test_rmse - val_rmse,
    })

    print(f"{name}: train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}, "
          f"test_rmse={test_rmse:.4f}, gap={test_rmse - val_rmse:.4f}")

final_df = pd.DataFrame(final_results).sort_values("test_rmse").reset_index(drop=True)
final_df"""),
        code("""\
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(final_df))
w = 0.25
ax.bar(x - w, final_df["train_rmse"], w, label="Train RMSE")
ax.bar(x, final_df["val_rmse"], w, label="Val RMSE")
ax.bar(x + w, final_df["test_rmse"], w, label="Test RMSE")
ax.set_xticks(x)
ax.set_xticklabels(final_df["model"], rotation=30, ha="right")
ax.set_ylabel("RMSE")
ax.set_title("Model A — Final 5 Models: Train / Val / Test RMSE")
ax.legend()
plt.tight_layout()
plt.show()"""),
        md("## 7. Save Artifacts"),
        code("""\
for name in top5_names:
    if name in DL_SKIP_OPTUNA:
        model_cls = MODEL_CLASSES[name]
        model = model_cls(input_dim=len(FEATURE_COLS))
    else:
        params = reconstruct_params(name, r3_best_params[name])
        model_cls = MODEL_CLASSES[name]
        model = model_cls(**params)
    model.fit(X_train_full, y_train_full)

    save_predictions(model, X_train_full, y_train_full, id_train, "A", name, "Training")
    save_predictions(model, X_test, y_test, id_test, "A", name, "Test")

    # OOF validation predictions (for Model D stacking)
    oof_preds = np.full(len(X), np.nan)
    for tr_idx, va_idx in splitter.split(groups):
        import sklearn.base
        fold_model = sklearn.base.clone(model)
        fold_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof_preds[va_idx] = fold_model.predict(X.iloc[va_idx])

    val_mask = ~np.isnan(oof_preds)
    val_out = df[ID_COLS].loc[val_mask].copy()
    val_out["y_true"] = y.loc[val_mask].values
    val_out["y_pred"] = oof_preds[val_mask]
    fname = f"model_A_{name}_Validation.parquet"
    uri = save_training_parquet(val_out, fname, TRAINING_DIR)
    print(f"  Saved {fname} -> {uri}")

    save_model_pkl(model, "A", name)

print("\\nDone! All Model A artifacts saved.")"""),
        md("## Summary"),
        code("""\
print("=" * 60)
print("MODEL A TRAINING COMPLETE")
print("=" * 60)
print(f"\\nFinal 5 models (sorted by test RMSE):")
for _, row in final_df.iterrows():
    print(f"  {row['model']:20s}  test_rmse={row['test_rmse']:.4f}  gap={row['overfit_gap']:.4f}")
print(f"\\nArtifacts saved to:")
print(f"  Predictions: {TRAINING_DIR}")
print(f"  Models: {MODEL_DIR}")"""),
    ]
    return cells


# ---------------------------------------------------------------------------
# Model B notebook
# ---------------------------------------------------------------------------


def make_model_b() -> list[dict]:
    cells = [
        md(
            "# 05b — Model B Training: Lap, No Tyre (2018-2025)\n\n"
            "Predicts **lap-level position** using 8 features (no tyre data).\n"
            "CV: ExpandingWindowSplit (test season = 2025)."
        ),
        md("## 0. Setup"),
        code(CHDIR),
        code(IMPORTS),
        code(HELPERS),
        code(OPTUNA_HELPERS),
        code(SAVE_ARTIFACTS),
        md("## 1. Load Features"),
        code("""\
df = load_from_gcs_or_local(
    "data/processed/lap_notyre/features_laps_notyre.parquet",
    Path("data/processed/lap_notyre/features_laps_notyre.parquet"),
)
print(f"Shape: {df.shape}")
df.head()"""),
        code("""\
FEATURE_COLS = [
    "gap_to_leader", "lap_time_delta_race_median", "gap_to_ahead",
    "race_progress_pct", "position_change_from_lap1", "laps_since_last_pit",
    "pit_stop_count", "lap_time_rolling_3",
]
TARGET = "position"
ID_COLS = ["season", "round", "event_name", "driver_abbrev", "team", "lap_number"]

df = df.dropna(subset=[TARGET]).reset_index(drop=True)

X = df[FEATURE_COLS]
y = df[TARGET]
groups = df["season"].values
print(f"Features: {X.shape}, Target: {y.shape}")
print(f"NaN counts:\\n{X.isna().sum()}")"""),
        md("## 2. CV Splitter"),
        code("""\
splitter = ExpandingWindowSplit(test_season=2025)
print(f"CV folds: {splitter.get_n_splits()}")
for i, (tr, va) in enumerate(splitter.split(groups)):
    tr_seasons = sorted(set(groups[tr]))
    va_seasons = sorted(set(groups[va]))
    print(f"  Fold {i}: train seasons={tr_seasons}, val season={va_seasons}, "
          f"train={len(tr):,}, val={len(va):,}")"""),
        md("## 3. Round 1 — Screen Models (default params)"),
        code("""\
candidates = get_candidates()
r1_results = screen_models(candidates, X, y, splitter, groups)
r1_results[["model", "mean_rmse", "std_rmse", "mean_mae"]]"""),
        code("""\
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(r1_results["model"], r1_results["mean_rmse"], xerr=r1_results["std_rmse"])
ax.set_xlabel("Mean RMSE (CV)")
ax.set_title("Round 1: Model Screening — Model B")
ax.invert_yaxis()
plt.tight_layout()
plt.show()"""),
        code("""\
top7_names = r1_results["model"].head(7).tolist()
print(f"Advancing to Round 2: {top7_names}")
eliminated = r1_results["model"].iloc[7:].tolist()
print(f"Eliminated: {eliminated}")"""),
        md("## 4. Round 2 — Optuna HP Tuning (top 7, 10 trials each)"),
        code("""\
r2_results = []
for name in top7_names:
    print(f"Tuning {name}...")
    best_params, best_rmse = run_optuna_round(name, X, y, splitter, groups, n_trials=10)
    r2_results.append({"model": name, "best_rmse": best_rmse, "best_params": best_params})
    print(f"  Best RMSE: {best_rmse:.4f}")

r2_df = pd.DataFrame(r2_results).sort_values("best_rmse").reset_index(drop=True)
r2_df[["model", "best_rmse"]]"""),
        code("""\
top5_names = r2_df["model"].head(5).tolist()
r2_best_params = {row["model"]: row["best_params"] for _, row in r2_df.iterrows()}
print(f"Advancing to Round 3: {top5_names}")"""),
        md("## 5. Round 3 — Final HP Tuning (top 5, 15 trials each)"),
        code("""\
r3_results = []
for name in top5_names:
    print(f"Fine-tuning {name}...")
    best_params, best_rmse = run_optuna_round(name, X, y, splitter, groups, n_trials=15)
    r3_results.append({"model": name, "best_rmse": best_rmse, "best_params": best_params})
    print(f"  Best RMSE: {best_rmse:.4f}")

r3_df = pd.DataFrame(r3_results).sort_values("best_rmse").reset_index(drop=True)
r3_best_params = {row["model"]: row["best_params"] for _, row in r3_df.iterrows()}
r3_df[["model", "best_rmse"]]"""),
        md("## 6. Test Set Evaluation"),
        code("""\
train_idx, test_idx = splitter.get_test_split(groups)
X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train_full, y_test = y.iloc[train_idx], y.iloc[test_idx]
id_train, id_test = df[ID_COLS].iloc[train_idx], df[ID_COLS].iloc[test_idx]

print(f"Train: {X_train_full.shape}, Test: {X_test.shape}")
print(f"Test season(s): {sorted(df['season'].iloc[test_idx].unique())}")"""),
        code("""\
final_results = []
for name in top5_names:
    if name in DL_SKIP_OPTUNA:
        model_cls = MODEL_CLASSES[name]
        model = model_cls(input_dim=len(FEATURE_COLS))
    else:
        params = reconstruct_params(name, r3_best_params[name])
        model_cls = MODEL_CLASSES[name]
        model = model_cls(**params)
    model.fit(X_train_full, y_train_full)

    train_preds = model.predict(X_train_full)
    train_rmse = np.sqrt(mean_squared_error(y_train_full, train_preds))
    train_mae = mean_absolute_error(y_train_full, train_preds)

    test_preds = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)

    val_rmse = r3_df.loc[r3_df["model"] == name, "best_rmse"].values[0]

    final_results.append({
        "model": name,
        "train_rmse": train_rmse, "train_mae": train_mae,
        "val_rmse": val_rmse,
        "test_rmse": test_rmse, "test_mae": test_mae,
        "overfit_gap": test_rmse - val_rmse,
    })

    print(f"{name}: train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}, "
          f"test_rmse={test_rmse:.4f}, gap={test_rmse - val_rmse:.4f}")

final_df = pd.DataFrame(final_results).sort_values("test_rmse").reset_index(drop=True)
final_df"""),
        code("""\
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(final_df))
w = 0.25
ax.bar(x - w, final_df["train_rmse"], w, label="Train RMSE")
ax.bar(x, final_df["val_rmse"], w, label="Val RMSE")
ax.bar(x + w, final_df["test_rmse"], w, label="Test RMSE")
ax.set_xticks(x)
ax.set_xticklabels(final_df["model"], rotation=30, ha="right")
ax.set_ylabel("RMSE")
ax.set_title("Model B — Final 5 Models: Train / Val / Test RMSE")
ax.legend()
plt.tight_layout()
plt.show()"""),
        md("## 7. Save Artifacts"),
        code("""\
for name in top5_names:
    if name in DL_SKIP_OPTUNA:
        model_cls = MODEL_CLASSES[name]
        model = model_cls(input_dim=len(FEATURE_COLS))
    else:
        params = reconstruct_params(name, r3_best_params[name])
        model_cls = MODEL_CLASSES[name]
        model = model_cls(**params)
    model.fit(X_train_full, y_train_full)

    save_predictions(model, X_train_full, y_train_full, id_train, "B", name, "Training")
    save_predictions(model, X_test, y_test, id_test, "B", name, "Test")

    # OOF validation predictions (for Model D stacking)
    oof_preds = np.full(len(X), np.nan)
    for tr_idx, va_idx in splitter.split(groups):
        import sklearn.base
        fold_model = sklearn.base.clone(model)
        fold_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof_preds[va_idx] = fold_model.predict(X.iloc[va_idx])

    val_mask = ~np.isnan(oof_preds)
    val_out = df[ID_COLS].loc[val_mask].copy()
    val_out["y_true"] = y.loc[val_mask].values
    val_out["y_pred"] = oof_preds[val_mask]
    fname = f"model_B_{name}_Validation.parquet"
    uri = save_training_parquet(val_out, fname, TRAINING_DIR)
    print(f"  Saved {fname} -> {uri}")

    save_model_pkl(model, "B", name)

print("\\nDone! All Model B artifacts saved.")"""),
        md("## Summary"),
        code("""\
print("=" * 60)
print("MODEL B TRAINING COMPLETE")
print("=" * 60)
print(f"\\nFinal 5 models (sorted by test RMSE):")
for _, row in final_df.iterrows():
    print(f"  {row['model']:20s}  test_rmse={row['test_rmse']:.4f}  gap={row['overfit_gap']:.4f}")
print(f"\\nArtifacts saved to:")
print(f"  Predictions: {TRAINING_DIR}")
print(f"  Models: {MODEL_DIR}")"""),
    ]
    return cells


# ---------------------------------------------------------------------------
# Model C notebook
# ---------------------------------------------------------------------------


def make_model_c() -> list[dict]:
    cells = [
        md(
            "# 05c — Model C Training: Pre-Race Features (2018-2025)\n\n"
            "Predicts **race-level finish_position** using 15 pre-race features.\n"
            "CV: ExpandingWindowSplit (test season = 2025)."
        ),
        md("## 0. Setup"),
        code(CHDIR),
        code(IMPORTS),
        code(HELPERS),
        code(OPTUNA_HELPERS),
        code(SAVE_ARTIFACTS),
        md("## 1. Load Features"),
        code("""\
df = load_from_gcs_or_local(
    "data/processed/race/features_race.parquet",
    Path("data/processed/race/features_race.parquet"),
)
print(f"Shape: {df.shape}")
df.head()"""),
        code("""\
FEATURE_COLS = [
    "grid_position", "quali_delta_to_pole", "team_avg_finish_last_3",
    "best_quali_sec", "position_trend", "team_points_cumulative_season",
    "points_last_3", "weather_wind_max_kph", "driver_circuit_avg_finish",
    "weather_temp_max", "quali_position_vs_teammate", "dnf_rate_season",
    "circuit_avg_dnf_rate", "weather_precip_mm", "driver_circuit_races",
]
TARGET = "finish_position"
ID_COLS = ["season", "round", "event_name", "driver_abbrev", "team"]

df = df.dropna(subset=[TARGET]).reset_index(drop=True)

X = df[FEATURE_COLS]
y = df[TARGET]
groups = df["season"].values
print(f"Features: {X.shape}, Target: {y.shape}")
print(f"NaN counts:\\n{X.isna().sum()}")"""),
        md("## 2. CV Splitter"),
        code("""\
splitter = ExpandingWindowSplit(test_season=2025)
print(f"CV folds: {splitter.get_n_splits()}")
for i, (tr, va) in enumerate(splitter.split(groups)):
    tr_seasons = sorted(set(groups[tr]))
    va_seasons = sorted(set(groups[va]))
    print(f"  Fold {i}: train seasons={tr_seasons}, val season={va_seasons}, "
          f"train={len(tr):,}, val={len(va):,}")"""),
        md("## 3. Round 1 — Screen Models (default params)"),
        code("""\
candidates = get_candidates_c()
r1_results = screen_models(candidates, X, y, splitter, groups)
r1_results[["model", "mean_rmse", "std_rmse", "mean_mae"]]"""),
        code("""\
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(r1_results["model"], r1_results["mean_rmse"], xerr=r1_results["std_rmse"])
ax.set_xlabel("Mean RMSE (CV)")
ax.set_title("Round 1: Model Screening — Model C")
ax.invert_yaxis()
plt.tight_layout()
plt.show()"""),
        code("""\
top7_names = r1_results["model"].head(7).tolist()
print(f"Advancing to Round 2: {top7_names}")
eliminated = r1_results["model"].iloc[7:].tolist()
print(f"Eliminated: {eliminated}")"""),
        md("## 4. Round 2 — Optuna HP Tuning (top 7, 10 trials each)"),
        code("""\
r2_results = []
for name in top7_names:
    print(f"Tuning {name}...")
    best_params, best_rmse = run_optuna_round(name, X, y, splitter, groups, n_trials=10)
    r2_results.append({"model": name, "best_rmse": best_rmse, "best_params": best_params})
    print(f"  Best RMSE: {best_rmse:.4f}")

r2_df = pd.DataFrame(r2_results).sort_values("best_rmse").reset_index(drop=True)
r2_df[["model", "best_rmse"]]"""),
        code("""\
top5_names = r2_df["model"].head(5).tolist()
r2_best_params = {row["model"]: row["best_params"] for _, row in r2_df.iterrows()}
print(f"Advancing to Round 3: {top5_names}")"""),
        md("## 5. Round 3 — Final HP Tuning (top 5, 15 trials each)"),
        code("""\
r3_results = []
for name in top5_names:
    print(f"Fine-tuning {name}...")
    best_params, best_rmse = run_optuna_round(name, X, y, splitter, groups, n_trials=15)
    r3_results.append({"model": name, "best_rmse": best_rmse, "best_params": best_params})
    print(f"  Best RMSE: {best_rmse:.4f}")

r3_df = pd.DataFrame(r3_results).sort_values("best_rmse").reset_index(drop=True)
r3_best_params = {row["model"]: row["best_params"] for _, row in r3_df.iterrows()}
r3_df[["model", "best_rmse"]]"""),
        md("## 6. Test Set Evaluation"),
        code("""\
train_idx, test_idx = splitter.get_test_split(groups)
X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train_full, y_test = y.iloc[train_idx], y.iloc[test_idx]
id_train, id_test = df[ID_COLS].iloc[train_idx], df[ID_COLS].iloc[test_idx]

print(f"Train: {X_train_full.shape}, Test: {X_test.shape}")
print(f"Test season(s): {sorted(df['season'].iloc[test_idx].unique())}")"""),
        code("""\
final_results = []
for name in top5_names:
    if name in DL_SKIP_OPTUNA:
        model_cls = MODEL_CLASSES[name]
        model = model_cls(input_dim=len(FEATURE_COLS))
    else:
        params = reconstruct_params(name, r3_best_params[name])
        model_cls = MODEL_CLASSES[name]
        model = model_cls(**params)
    model.fit(X_train_full, y_train_full)

    train_preds = model.predict(X_train_full)
    train_rmse = np.sqrt(mean_squared_error(y_train_full, train_preds))
    train_mae = mean_absolute_error(y_train_full, train_preds)

    test_preds = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)

    val_rmse = r3_df.loc[r3_df["model"] == name, "best_rmse"].values[0]

    final_results.append({
        "model": name,
        "train_rmse": train_rmse, "train_mae": train_mae,
        "val_rmse": val_rmse,
        "test_rmse": test_rmse, "test_mae": test_mae,
        "overfit_gap": test_rmse - val_rmse,
    })

    print(f"{name}: train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}, "
          f"test_rmse={test_rmse:.4f}, gap={test_rmse - val_rmse:.4f}")

final_df = pd.DataFrame(final_results).sort_values("test_rmse").reset_index(drop=True)
final_df"""),
        code("""\
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(final_df))
w = 0.25
ax.bar(x - w, final_df["train_rmse"], w, label="Train RMSE")
ax.bar(x, final_df["val_rmse"], w, label="Val RMSE")
ax.bar(x + w, final_df["test_rmse"], w, label="Test RMSE")
ax.set_xticks(x)
ax.set_xticklabels(final_df["model"], rotation=30, ha="right")
ax.set_ylabel("RMSE")
ax.set_title("Model C — Final 5 Models: Train / Val / Test RMSE")
ax.legend()
plt.tight_layout()
plt.show()"""),
        md("## 7. Save Artifacts"),
        code("""\
for name in top5_names:
    if name in DL_SKIP_OPTUNA:
        model_cls = MODEL_CLASSES[name]
        model = model_cls(input_dim=len(FEATURE_COLS))
    else:
        params = reconstruct_params(name, r3_best_params[name])
        model_cls = MODEL_CLASSES[name]
        model = model_cls(**params)
    model.fit(X_train_full, y_train_full)

    save_predictions(model, X_train_full, y_train_full, id_train, "C", name, "Training")
    save_predictions(model, X_test, y_test, id_test, "C", name, "Test")

    # OOF validation predictions (for Model D stacking)
    oof_preds = np.full(len(X), np.nan)
    for tr_idx, va_idx in splitter.split(groups):
        import sklearn.base
        fold_model = sklearn.base.clone(model)
        fold_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof_preds[va_idx] = fold_model.predict(X.iloc[va_idx])

    val_mask = ~np.isnan(oof_preds)
    val_out = df[ID_COLS].loc[val_mask].copy()
    val_out["y_true"] = y.loc[val_mask].values
    val_out["y_pred"] = oof_preds[val_mask]
    fname = f"model_C_{name}_Validation.parquet"
    uri = save_training_parquet(val_out, fname, TRAINING_DIR)
    print(f"  Saved {fname} -> {uri}")

    save_model_pkl(model, "C", name)

print("\\nDone! All Model C artifacts saved.")"""),
        md("## Summary"),
        code("""\
print("=" * 60)
print("MODEL C TRAINING COMPLETE")
print("=" * 60)
print(f"\\nFinal 5 models (sorted by test RMSE):")
for _, row in final_df.iterrows():
    print(f"  {row['model']:20s}  test_rmse={row['test_rmse']:.4f}  gap={row['overfit_gap']:.4f}")
print(f"\\nArtifacts saved to:")
print(f"  Predictions: {TRAINING_DIR}")
print(f"  Models: {MODEL_DIR}")"""),
    ]
    return cells


# ---------------------------------------------------------------------------
# Model D notebook (stacking)
# ---------------------------------------------------------------------------


def make_model_d() -> list[dict]:
    cells = [
        md(
            "# 05d — Model D Training: All-Combinations Stacking\n\n"
            "Combines race-level predictions from **all** variants of Models A, B, C\n"
            "into stacking ensembles. Two-phase approach:\n"
            "- **Phase 1:** RidgeCV screen of all A x B x C combinations\n"
            "- **Phase 2:** Full tournament on top 20 combinations\n\n"
            "Uses out-of-fold (OOF) predictions as meta-features to prevent leakage."
        ),
        md("## 0. Setup"),
        code(CHDIR),
        code("""\
import itertools
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb

from f1_predictor.features.splits import LeaveOneSeasonOut
from f1_predictor.data.storage import (
    load_training_parquet,
    save_training_parquet,
    save_model_pickle as gcs_save_model_pickle,
    sync_training_from_gcs,
)

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

TRAINING_DIR = Path("data/training")
MODEL_DIR = Path("data/raw/model")
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Sync base model artifacts from GCS
for mt in ["A", "B", "C"]:
    sync_training_from_gcs(mt, TRAINING_DIR)
print("Synced base model artifacts from GCS.")"""),
        code("""\
def wrap_imputer(model):
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])

NAN_TOLERANT_D = {"XGBoost_shallow", "LightGBM_shallow"}

def cv_evaluate(model, X, y, splitter, groups):
    fold_rmse, fold_mae = [], []
    for train_idx, val_idx in splitter.split(groups):
        import sklearn.base
        m = sklearn.base.clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = m.predict(X.iloc[val_idx])
        fold_rmse.append(np.sqrt(mean_squared_error(y.iloc[val_idx], preds)))
        fold_mae.append(mean_absolute_error(y.iloc[val_idx], preds))
    return {
        "fold_rmse": fold_rmse, "fold_mae": fold_mae,
        "mean_rmse": np.mean(fold_rmse), "std_rmse": np.std(fold_rmse),
        "mean_mae": np.mean(fold_mae),
    }"""),
        md(
            "## 1. Discover All Variant Predictions\n\n"
            "Scan `data/training/` for all Validation parquets from each model type."
        ),
        code("""\
def discover_variants(model_type):
    \"\"\"Find all variant names that have Validation parquets.\"\"\"
    files = sorted(TRAINING_DIR.glob(f"model_{model_type}_*_Validation.parquet"))
    variants = []
    for f in files:
        name = f.stem.replace(f"model_{model_type}_", "").replace("_Validation", "")
        variants.append(name)
    return variants

variants_A = discover_variants("A")
variants_B = discover_variants("B")
variants_C = discover_variants("C")
print(f"Model A variants ({len(variants_A)}): {variants_A}")
print(f"Model B variants ({len(variants_B)}): {variants_B}")
print(f"Model C variants ({len(variants_C)}): {variants_C}")
print(f"Total combinations: {len(variants_A)} × {len(variants_B)} × {len(variants_C)} = "
      f"{len(variants_A) * len(variants_B) * len(variants_C)}")"""),
        code("""\
def aggregate_lap_to_race(val_df):
    \"\"\"Aggregate lap-level predictions to race level using the last lap per driver-race.\"\"\"
    val_df = val_df.sort_values(["season", "round", "driver_abbrev", "lap_number"])
    last_laps = val_df.groupby(["season", "round", "driver_abbrev"]).tail(1)
    return last_laps[["season", "round", "driver_abbrev", "y_true", "y_pred"]].copy()

# Load and cache all variant predictions
merge_key = ["season", "round", "driver_abbrev"]
preds_A, preds_B, preds_C = {}, {}, {}

for v in variants_A:
    df = load_training_parquet(f"model_A_{v}_Validation.parquet", TRAINING_DIR)
    race_df = aggregate_lap_to_race(df)
    preds_A[v] = race_df.rename(columns={"y_pred": f"pred_A_{v}", "y_true": "true_pos"})

for v in variants_B:
    df = load_training_parquet(f"model_B_{v}_Validation.parquet", TRAINING_DIR)
    race_df = aggregate_lap_to_race(df)
    preds_B[v] = race_df.rename(columns={"y_pred": f"pred_B_{v}", "y_true": "true_pos"})

for v in variants_C:
    df = load_training_parquet(f"model_C_{v}_Validation.parquet", TRAINING_DIR)
    preds_C[v] = df.rename(columns={"y_pred": f"pred_C_{v}", "y_true": "true_pos"})

print(f"Loaded {len(preds_A)} A, {len(preds_B)} B, {len(preds_C)} C variant predictions")"""),
        md("## 2. Build All-Variants Meta Matrix"),
        code("""\
# Start with first C variant to get the base structure
first_c = variants_C[0]
base = preds_C[first_c][merge_key + ["true_pos", f"pred_C_{first_c}"]].copy()

# Merge all other C variants
for v in variants_C[1:]:
    base = base.merge(preds_C[v][merge_key + [f"pred_C_{v}"]], on=merge_key, how="outer")

# Merge all A variants
for v in variants_A:
    base = base.merge(preds_A[v][merge_key + [f"pred_A_{v}"]], on=merge_key, how="left")

# Merge all B variants
for v in variants_B:
    base = base.merge(preds_B[v][merge_key + [f"pred_B_{v}"]], on=merge_key, how="left")

base = base.dropna(subset=["true_pos"]).reset_index(drop=True)
y_all = base["true_pos"]
groups_all = base["season"].values
print(f"All-variants matrix: {base.shape}")
print(f"Seasons: {sorted(base['season'].unique())}")"""),
        md("## 3. CV Splitter"),
        code("""\
available_seasons = sorted(base["season"].unique())
val_seasons = [s for s in available_seasons if s != 2023]
splitter = LeaveOneSeasonOut(val_seasons=val_seasons, test_season=2023)
print(f"Val seasons: {val_seasons}, Test: 2023")
print(f"CV folds: {splitter.get_n_splits()}")"""),
        md(
            "## 4. Phase 1 — RidgeCV Screen of ALL Combinations\n\n"
            "Fast screen using RidgeCV on every A x B x C combination."
        ),
        code("""\
all_combos = list(itertools.product(variants_A, variants_B, variants_C))
print(f"Screening {len(all_combos)} combinations with RidgeCV...")

phase1_results = []
for i, (va, vb, vc) in enumerate(all_combos):
    feat_cols = [f"pred_A_{va}", f"pred_B_{vb}", f"pred_C_{vc}"]
    X_combo = base[feat_cols]

    model = wrap_imputer(RidgeCV())
    result = cv_evaluate(model, X_combo, y_all, splitter, groups_all)
    phase1_results.append({
        "A": va, "B": vb, "C": vc,
        "combo": f"{va}__{vb}__{vc}",
        "mean_rmse": result["mean_rmse"],
        "std_rmse": result["std_rmse"],
    })
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(all_combos)} done...")

p1_df = pd.DataFrame(phase1_results).sort_values("mean_rmse").reset_index(drop=True)
print(f"\\nPhase 1 complete. Top 10 combinations:")
print(p1_df.head(10)[["combo", "mean_rmse", "std_rmse"]].to_string(index=False))"""),
        code("""\
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(p1_df["mean_rmse"], bins=40, edgecolor="black", alpha=0.7)
best_val = p1_df["mean_rmse"].iloc[0]
top20_val = p1_df["mean_rmse"].iloc[19]
ax.axvline(best_val, color="red", linestyle="--", label=f"Best: {best_val:.4f}")
ax.axvline(top20_val, color="orange", linestyle="--", label=f"Top-20: {top20_val:.4f}")
ax.set_xlabel("CV RMSE (RidgeCV)")
ax.set_ylabel("Count")
ax.set_title(f"Phase 1: Distribution of {len(all_combos)} Combination RMSEs")
ax.legend()
plt.tight_layout()
plt.show()"""),
        md("## 5. Phase 2 — Full Tournament on Top 20"),
        code("""\
TOP_N = 20
top_combos = p1_df.head(TOP_N)
print(f"Phase 2: Full tournament on top {TOP_N} combinations\\n")

D_MODEL_CLASSES = {
    "RidgeCV": RidgeCV,
    "LassoCV": LassoCV,
    "ElasticNetCV": ElasticNetCV,
    "XGBoost_shallow": xgb.XGBRegressor,
    "LightGBM_shallow": lgb.LGBMRegressor,
}

def get_d_param_space(name, trial):
    if name == "XGBoost_shallow":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 2, 4),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            random_state=42, verbosity=0,
        )
    elif name == "LightGBM_shallow":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 2, 4),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            random_state=42, verbose=-1,
        )
    elif name == "RidgeCV":
        alphas = [trial.suggest_float(f"alpha_{i}", 0.001, 100.0, log=True) for i in range(5)]
        return dict(alphas=sorted(alphas))
    elif name == "LassoCV":
        return dict(n_alphas=trial.suggest_int("n_alphas", 50, 200), random_state=42, max_iter=5000)
    elif name == "ElasticNetCV":
        return dict(
            l1_ratio=[trial.suggest_float(f"l1_{i}", 0.1, 0.9) for i in range(3)],
            n_alphas=trial.suggest_int("n_alphas", 50, 200), random_state=42, max_iter=5000,
        )
    return {}

def reconstruct_d_params(name, best_params):
    params = dict(best_params)
    if name == "XGBoost_shallow":
        params.update(random_state=42, verbosity=0)
    elif name == "LightGBM_shallow":
        params.update(random_state=42, verbose=-1)
    elif name == "RidgeCV":
        alpha_keys = sorted(k for k in params if k.startswith("alpha_"))
        alphas = sorted(params.pop(k) for k in alpha_keys)
        params["alphas"] = alphas
    elif name == "LassoCV":
        params.update(random_state=42, max_iter=5000)
    elif name == "ElasticNetCV":
        l1_keys = sorted(k for k in params if k.startswith("l1_"))
        l1_ratio = [params.pop(k) for k in l1_keys]
        params["l1_ratio"] = l1_ratio
        params.update(random_state=42, max_iter=5000)
    return params

phase2_results = []
for _, row in top_combos.iterrows():
    va, vb, vc = row["A"], row["B"], row["C"]
    combo_name = row["combo"]
    feat_cols = [f"pred_A_{va}", f"pred_B_{vb}", f"pred_C_{vc}"]
    X_combo = base[feat_cols]

    # Screen 5 meta-learners
    combo_rows = []
    meta_candidates = [
        ("RidgeCV", wrap_imputer(RidgeCV())),
        ("LassoCV", wrap_imputer(LassoCV(random_state=42, max_iter=5000))),
        ("ElasticNetCV", wrap_imputer(
            ElasticNetCV(random_state=42, max_iter=5000))),
        ("XGBoost_shallow", xgb.XGBRegressor(
            n_estimators=100, max_depth=3, random_state=42, verbosity=0)),
        ("LightGBM_shallow", lgb.LGBMRegressor(
            n_estimators=100, max_depth=3, random_state=42, verbose=-1)),
    ]
    for meta_name, meta_model in meta_candidates:
        result = cv_evaluate(meta_model, X_combo, y_all, splitter, groups_all)
        combo_rows.append({"meta": meta_name, "mean_rmse": result["mean_rmse"]})

    best_meta = sorted(combo_rows, key=lambda r: r["mean_rmse"])[0]

    # Optuna tune the best meta-learner (10 trials)
    best_meta_name = best_meta["meta"]
    def objective(trial):
        params = get_d_param_space(best_meta_name, trial)
        model = D_MODEL_CLASSES[best_meta_name](**params)
        if best_meta_name not in NAN_TOLERANT_D:
            model = wrap_imputer(model)
        return cv_evaluate(model, X_combo, y_all, splitter, groups_all)["mean_rmse"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=10, show_progress_bar=False)

    phase2_results.append({
        "A": va, "B": vb, "C": vc, "combo": combo_name,
        "meta": best_meta_name, "best_rmse": study.best_value,
        "best_params": study.best_params,
        "phase1_rmse": row["mean_rmse"],
    })
    print(f"  {combo_name}: meta={best_meta_name}, RMSE={study.best_value:.4f}")

p2_df = pd.DataFrame(phase2_results).sort_values("best_rmse").reset_index(drop=True)
print(f"\\nPhase 2 complete. Top 5:")
print(p2_df.head()[["combo", "meta", "best_rmse"]].to_string(index=False))"""),
        code("""\
fig, ax = plt.subplots(figsize=(14, 6))
ax.barh(range(TOP_N), p2_df["best_rmse"].values)
ax.set_yticks(range(TOP_N))
ax.set_yticklabels([f"{r['combo']}\\n({r['meta']})" for _, r in p2_df.iterrows()], fontsize=7)
ax.set_xlabel("CV RMSE (tuned)")
ax.set_title(f"Phase 2: Top {TOP_N} Combinations After Optuna Tuning")
ax.invert_yaxis()
plt.tight_layout()
plt.show()"""),
        md("## 6. Test Set Evaluation (Top 5 Combinations)"),
        code("""\
train_idx, test_idx = splitter.get_test_split(groups_all)
id_train = base[merge_key].iloc[train_idx]
id_test = base[merge_key].iloc[test_idx]
y_train = y_all.iloc[train_idx]
y_test = y_all.iloc[test_idx]

print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
print(f"Test season: {sorted(base['season'].iloc[test_idx].unique())}")

final_results = []
for _, row in p2_df.head(5).iterrows():
    va, vb, vc = row["A"], row["B"], row["C"]
    feat_cols = [f"pred_A_{va}", f"pred_B_{vb}", f"pred_C_{vc}"]
    X_train_combo = base[feat_cols].iloc[train_idx]
    X_test_combo = base[feat_cols].iloc[test_idx]

    params = reconstruct_d_params(row["meta"], row["best_params"])
    model = D_MODEL_CLASSES[row["meta"]](**params)
    if row["meta"] not in NAN_TOLERANT_D:
        model = wrap_imputer(model)

    model.fit(X_train_combo, y_train)

    test_preds = model.predict(X_test_combo)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)

    final_results.append({
        "combo": row["combo"], "meta": row["meta"],
        "val_rmse": row["best_rmse"], "test_rmse": test_rmse, "test_mae": test_mae,
        "overfit_gap": test_rmse - row["best_rmse"],
    })
    print(f"  {row['combo']} ({row['meta']}): val={row['best_rmse']:.4f}, test={test_rmse:.4f}")

final_df = pd.DataFrame(final_results).sort_values("test_rmse").reset_index(drop=True)
final_df"""),
        md("## 7. Save Artifacts (Top 5 Combinations)"),
        code("""\
for _, row in p2_df.head(5).iterrows():
    va, vb, vc = row["A"], row["B"], row["C"]
    feat_cols = [f"pred_A_{va}", f"pred_B_{vb}", f"pred_C_{vc}"]
    X_train_combo = base[feat_cols].iloc[train_idx]
    X_test_combo = base[feat_cols].iloc[test_idx]

    params = reconstruct_d_params(row["meta"], row["best_params"])
    model = D_MODEL_CLASSES[row["meta"]](**params)
    if row["meta"] not in NAN_TOLERANT_D:
        model = wrap_imputer(model)
    model.fit(X_train_combo, y_train)

    combo_tag = f"{va}__{vb}__{vc}__{row['meta']}"
    for split_name, X_s, y_s, id_s in [
        ("Training", X_train_combo, y_train, id_train),
        ("Test", X_test_combo, y_test, id_test),
    ]:
        out = id_s.copy()
        out["y_true"] = y_s.values
        out["y_pred"] = model.predict(X_s)
        fname = f"model_D_{combo_tag}_{split_name}.parquet"
        uri = save_training_parquet(out, fname, TRAINING_DIR)
        print(f"  Saved {fname} -> {uri}")

    pkl_name = f"Model_D_{combo_tag}.pkl"
    uri = gcs_save_model_pickle(model, pkl_name, MODEL_DIR)
    print(f"  Saved {pkl_name} -> {uri}")

print("\\nDone! All Model D artifacts saved.")"""),
        md("## Summary"),
        code("""\
print("=" * 60)
print("MODEL D (ALL-COMBINATIONS STACKING) COMPLETE")
print("=" * 60)
print(f"\\nTotal combinations screened: {len(all_combos)}")
print(f"Phase 1 (RidgeCV): {len(all_combos)} combos -> top {TOP_N}")
print(f"Phase 2 (tournament + Optuna): top {TOP_N} -> final 5")
print(f"\\nFinal 5 combinations:")
for _, row in final_df.iterrows():
    print(f"  {row['combo']:40s}  meta={row['meta']:15s}  "
          f"test_rmse={row['test_rmse']:.4f}  gap={row['overfit_gap']:.4f}")"""),
    ]
    return cells


# ---------------------------------------------------------------------------
# Model E notebook — Rich Feature Stacking
# ---------------------------------------------------------------------------


def make_model_e() -> list[dict]:
    cells = [
        md(
            "# 05e — Model E Training: Rich Feature Stacking\n\n"
            "Combines **Model A's lap-level richness** with "
            "**Model D's stacking stability**.\n\n"
            "Key improvements over Model D (which uses only 3 prediction features):\n"
            "- **Rich lap-level aggregations** — mean, std, last-5, min, range "
            "(not just last lap)\n"
            "- **Pre-race context** — grid position and qualifying gap\n"
            "- **Auto-selection** — picks the best variant of each base model\n\n"
            "| Feature Source | Count | Signal |\n"
            "|---|---|---|\n"
            "| Model A lap-level agg | 6 | In-race position trajectory |\n"
            "| Model B lap-level agg | 4 | Long-term form trajectory |\n"
            "| Model C prediction | 1 | Pre-race baseline |\n"
            "| Race features | 2 | Grid position, qualifying pace |\n"
            "| **Total** | **13** | |"
        ),
        md("## 0. Setup"),
        code(CHDIR),
        code("""\
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, median_absolute_error,
    max_error, r2_score, mean_absolute_percentage_error,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb

from f1_predictor.features.splits import LeaveOneSeasonOut
from f1_predictor.data.storage import (
    load_from_gcs_or_local,
    load_training_parquet,
    save_training_parquet,
    save_model_pickle as gcs_save_model_pickle,
    sync_training_from_gcs,
)

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

TRAINING_DIR = Path("data/training")
MODEL_DIR = Path("data/raw/model")
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

for mt in ["A", "B", "C"]:
    sync_training_from_gcs(mt, TRAINING_DIR)
print("Synced base model artifacts from GCS.")"""),
        code('''\
merge_key = ["season", "round", "driver_abbrev"]

NAN_TOLERANT_E = {"XGBoost_shallow", "LightGBM_shallow"}


def wrap_imputer(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def cv_evaluate(model, X, y, splitter, groups):
    """CV evaluation returning fold and mean metrics."""
    fold_rmse, fold_mae = [], []
    for train_idx, val_idx in splitter.split(groups):
        import sklearn.base
        m = sklearn.base.clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = m.predict(X.iloc[val_idx])
        fold_rmse.append(np.sqrt(mean_squared_error(
            y.iloc[val_idx], preds)))
        fold_mae.append(mean_absolute_error(y.iloc[val_idx], preds))
    return {
        "fold_rmse": fold_rmse, "fold_mae": fold_mae,
        "mean_rmse": np.mean(fold_rmse),
        "std_rmse": np.std(fold_rmse),
        "mean_mae": np.mean(fold_mae),
    }


def aggregate_lap_rich(df, prefix):
    """Compute rich aggregation stats from lap-level predictions."""
    df = df.sort_values(merge_key + ["lap_number"])
    grp = df.groupby(merge_key)["y_pred"]

    last_vals = grp.last()
    mean_vals = grp.mean()
    std_vals = grp.std().fillna(0)
    min_vals = grp.min()
    max_vals = grp.max()

    agg = pd.DataFrame({
        f"{prefix}_last": last_vals,
        f"{prefix}_mean": mean_vals,
        f"{prefix}_std": std_vals,
        f"{prefix}_min": min_vals,
        f"{prefix}_range": max_vals - min_vals,
    }).reset_index()

    last5 = (df.groupby(merge_key).tail(5)
             .groupby(merge_key)["y_pred"].mean()
             .rename(f"{prefix}_last5").reset_index())
    agg = agg.merge(last5, on=merge_key)

    y_true = (df.groupby(merge_key)["y_true"]
              .last().rename("y_true").reset_index())
    agg = agg.merge(y_true, on=merge_key)
    return agg


def select_best_variant(model_type, is_lap_level=False):
    """Pick best variant by validation RMSE (race-level)."""
    files = sorted(TRAINING_DIR.glob(
        f"model_{model_type}_*_Validation.parquet"))
    best_name, best_rmse = None, float("inf")
    for f in files:
        stem = (f.stem
                .replace(f"model_{model_type}_", "")
                .replace("_Validation", ""))
        df = pd.read_parquet(f)
        if is_lap_level and "lap_number" in df.columns:
            df = df.sort_values(merge_key + ["lap_number"])
            df = df.groupby(merge_key).tail(1)
        rmse = np.sqrt(mean_squared_error(df["y_true"], df["y_pred"]))
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = stem
    return best_name, best_rmse'''),
        md("## 1. Auto-Select Best Base Model Variants"),
        code("""\
best_A, rmse_A = select_best_variant("A", is_lap_level=True)
best_B, rmse_B = select_best_variant("B", is_lap_level=True)
best_C, rmse_C = select_best_variant("C", is_lap_level=False)

print(f"Best Model A variant: {best_A} (val RMSE={rmse_A:.4f})")
print(f"Best Model B variant: {best_B} (val RMSE={rmse_B:.4f})")
print(f"Best Model C variant: {best_C} (val RMSE={rmse_C:.4f})")"""),
        md(
            "## 2. Build Rich Meta-Features\n\n"
            "Extract 6 statistics from Model A's lap-level predictions, "
            "4 from Model B, 1 from Model C."
        ),
        code("""\
df_A = load_training_parquet(
    f"model_A_{best_A}_Validation.parquet", TRAINING_DIR)
df_B = load_training_parquet(
    f"model_B_{best_B}_Validation.parquet", TRAINING_DIR)
print(f"Model A laps: {len(df_A):,}, Model B laps: {len(df_B):,}")

agg_A = aggregate_lap_rich(df_A, "A")
print(f"Model A race-level: {len(agg_A)} rows, features: "
      f"{[c for c in agg_A.columns if c.startswith('A_')]}")

agg_B = aggregate_lap_rich(df_B, "B")
B_keep = [c for c in agg_B.columns
          if c.startswith("B_") and
          any(s in c for s in ["last", "mean", "std", "last5"])]
agg_B = agg_B[merge_key + B_keep]
print(f"Model B race-level: {len(agg_B)} rows, features: {B_keep}")

df_C = load_training_parquet(
    f"model_C_{best_C}_Validation.parquet", TRAINING_DIR)
df_C = df_C.rename(
    columns={"y_pred": "C_pred"})[merge_key + ["C_pred"]]
print(f"Model C: {len(df_C)} rows")"""),
        md("## 3. Merge + Pre-Race Context Features"),
        code("""\
base = agg_A.merge(agg_B, on=merge_key, how="left")
base = base.merge(df_C, on=merge_key, how="left")

race_feat = load_from_gcs_or_local(
    "data/processed/race/features_race.parquet",
    Path("data/processed/race/features_race.parquet"),
)
prerace_cols = ["grid_position", "quali_delta_to_pole"]
race_merge = race_feat[merge_key + prerace_cols].copy()
base = base.merge(race_merge, on=merge_key, how="left")

base = base.dropna(subset=["y_true"]).reset_index(drop=True)
y_all = base["y_true"]
groups_all = base["season"].values

FEATURE_COLS = [c for c in base.columns
                if c not in merge_key + ["y_true"]]
X_all = base[FEATURE_COLS]

print(f"Meta-feature matrix: {base.shape}")
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Seasons: {sorted(base['season'].unique())}")
print(f"NaN counts:\\n{base[FEATURE_COLS].isna().sum()}")"""),
        md("## 4. CV Splitter"),
        code("""\
available_seasons = sorted(base["season"].unique())
val_seasons = [s for s in available_seasons if s != 2023]
splitter = LeaveOneSeasonOut(
    val_seasons=val_seasons, test_season=2023)
print(f"Val seasons: {val_seasons}, Test: 2023")
print(f"CV folds: {splitter.get_n_splits()}")"""),
        md("## 5. Screen Meta-Learners"),
        code("""\
meta_candidates = [
    ("RidgeCV", wrap_imputer(RidgeCV())),
    ("LassoCV", wrap_imputer(
        LassoCV(random_state=42, max_iter=5000))),
    ("ElasticNetCV", wrap_imputer(
        ElasticNetCV(random_state=42, max_iter=5000))),
    ("XGBoost_shallow", xgb.XGBRegressor(
        n_estimators=100, max_depth=3,
        random_state=42, verbosity=0)),
    ("LightGBM_shallow", lgb.LGBMRegressor(
        n_estimators=100, max_depth=3,
        random_state=42, verbose=-1)),
]

screen_rows = []
for name, model in meta_candidates:
    result = cv_evaluate(model, X_all, y_all, splitter, groups_all)
    screen_rows.append({
        "meta": name, "mean_rmse": result["mean_rmse"],
        "std_rmse": result["std_rmse"],
    })
    print(f"  {name}: RMSE={result['mean_rmse']:.4f} "
          f"+/- {result['std_rmse']:.4f}")

screen_df = pd.DataFrame(screen_rows).sort_values("mean_rmse")
print(f"\\nBest meta-learner: {screen_df.iloc[0]['meta']}")"""),
        md("## 6. Optuna Tuning (Top 3 Meta-Learners)"),
        code('''\
E_MODEL_CLASSES = {
    "RidgeCV": RidgeCV,
    "LassoCV": LassoCV,
    "ElasticNetCV": ElasticNetCV,
    "XGBoost_shallow": xgb.XGBRegressor,
    "LightGBM_shallow": lgb.LGBMRegressor,
}


def get_e_param_space(name, trial):
    """Optuna param space for Model E meta-learners."""
    if name == "XGBoost_shallow":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 2, 5),
            learning_rate=trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float(
                "colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float(
                "reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float(
                "reg_lambda", 1e-8, 10.0, log=True),
            random_state=42, verbosity=0,
        )
    elif name == "LightGBM_shallow":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 2, 5),
            learning_rate=trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float(
                "colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float(
                "reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float(
                "reg_lambda", 1e-8, 10.0, log=True),
            random_state=42, verbose=-1,
        )
    elif name == "RidgeCV":
        alphas = [trial.suggest_float(
            f"alpha_{i}", 0.001, 100.0, log=True) for i in range(5)]
        return dict(alphas=sorted(alphas))
    elif name == "LassoCV":
        return dict(
            n_alphas=trial.suggest_int("n_alphas", 50, 200),
            random_state=42, max_iter=5000)
    elif name == "ElasticNetCV":
        return dict(
            l1_ratio=[trial.suggest_float(
                f"l1_{i}", 0.1, 0.9) for i in range(3)],
            n_alphas=trial.suggest_int("n_alphas", 50, 200),
            random_state=42, max_iter=5000,
        )
    return {}


def reconstruct_e_params(name, best_params):
    """Rebuild model constructor args from Optuna best_params."""
    params = dict(best_params)
    if name == "XGBoost_shallow":
        params.update(random_state=42, verbosity=0)
    elif name == "LightGBM_shallow":
        params.update(random_state=42, verbose=-1)
    elif name == "RidgeCV":
        alpha_keys = sorted(k for k in params if k.startswith("alpha_"))
        alphas = sorted(params.pop(k) for k in alpha_keys)
        params["alphas"] = alphas
    elif name == "LassoCV":
        params.update(random_state=42, max_iter=5000)
    elif name == "ElasticNetCV":
        l1_keys = sorted(k for k in params if k.startswith("l1_"))
        l1_ratio = [params.pop(k) for k in l1_keys]
        params["l1_ratio"] = l1_ratio
        params.update(random_state=42, max_iter=5000)
    return params'''),
        code("""\
top3 = screen_df.head(3)["meta"].values.tolist()
print(f"Tuning top 3: {top3}\\n")

tune_results = []
for meta_name in top3:
    def objective(trial, _name=meta_name):
        params = get_e_param_space(_name, trial)
        model = E_MODEL_CLASSES[_name](**params)
        if _name not in NAN_TOLERANT_E:
            model = wrap_imputer(model)
        return cv_evaluate(
            model, X_all, y_all, splitter, groups_all
        )["mean_rmse"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=15, show_progress_bar=False)
    tune_results.append({
        "meta": meta_name, "best_rmse": study.best_value,
        "best_params": study.best_params,
    })
    print(f"  {meta_name}: RMSE={study.best_value:.4f}")

tune_df = (pd.DataFrame(tune_results)
           .sort_values("best_rmse").reset_index(drop=True))
best_meta = tune_df.iloc[0]
print(f"\\nBest: {best_meta['meta']} "
      f"RMSE={best_meta['best_rmse']:.4f}")"""),
        md("## 7. Test Set Evaluation"),
        code("""\
train_idx, test_idx = splitter.get_test_split(groups_all)
X_train = X_all.iloc[train_idx]
X_test = X_all.iloc[test_idx]
y_train = y_all.iloc[train_idx]
y_test = y_all.iloc[test_idx]
id_train = base[merge_key].iloc[train_idx]
id_test = base[merge_key].iloc[test_idx]

print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
print(f"Test season: "
      f"{sorted(base['season'].iloc[test_idx].unique())}\\n")

test_results = []
for _, row in tune_df.iterrows():
    params = reconstruct_e_params(row["meta"], row["best_params"])
    model = E_MODEL_CLASSES[row["meta"]](**params)
    if row["meta"] not in NAN_TOLERANT_E:
        model = wrap_imputer(model)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    med_ae = median_absolute_error(y_test, preds)
    mx_err = max_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds) * 100
    w1 = np.mean(np.abs(y_test.values - preds) <= 1.0) * 100
    w3 = np.mean(np.abs(y_test.values - preds) <= 3.0) * 100

    test_with_preds = base.iloc[test_idx][merge_key].copy()
    test_with_preds["y_true"] = y_test.values
    test_with_preds["y_pred"] = preds
    spear_vals = []
    for _, grp in test_with_preds.groupby(["season", "round"]):
        if (len(grp) >= 3
                and grp["y_true"].std() > 0
                and grp["y_pred"].std() > 0):
            spear_vals.append(
                spearmanr(grp["y_true"], grp["y_pred"]).statistic)
    spear = np.mean(spear_vals) if spear_vals else np.nan

    test_results.append({
        "Meta": row["meta"], "Val_RMSE": row["best_rmse"],
        "Test_RMSE": rmse, "MAE": mae, "R2": r2,
        "Median_AE": med_ae, "Max_Error": mx_err,
        "MAPE": mape, "Within_1": w1, "Within_3": w3,
        "Spearman": spear,
        "Overfit_Gap": rmse - row["best_rmse"],
    })
    print(f"  {row['meta']:20s}  RMSE={rmse:.4f}  "
          f"R2={r2:.4f}  Spearman={spear:.4f}  "
          f"Within-3={w3:.1f}%")

test_df = (pd.DataFrame(test_results)
           .sort_values("Test_RMSE").reset_index(drop=True))
test_df"""),
        md("## 8. Compare to Models A and D"),
        code("""\
best_E = test_df.iloc[0]
print("=" * 70)
print("MODEL E vs A vs D (Test Set)")
print("=" * 70)

ref_A = {"RMSE": 3.1028, "R2": 0.6743,
         "Spearman": 0.9037, "Within_3": 70.3}
ref_D = {"RMSE": 3.1803, "R2": 0.6946,
         "Spearman": 0.8477, "Within_3": 73.6}

for metric, fmt in [("RMSE", ".4f"), ("R2", ".4f"),
                    ("Spearman", ".4f"), ("Within_3", ".1f")]:
    e_key = "Test_RMSE" if metric == "RMSE" else metric
    e_val = best_E[e_key]
    a_val = ref_A[metric]
    d_val = ref_D[metric]
    sfx = "%" if metric == "Within_3" else ""
    print(f"  {metric:12s}  A={a_val:{fmt}}{sfx}  "
          f"D={d_val:{fmt}}{sfx}  E={e_val:{fmt}}{sfx}")

print(f"\\nModel E meta-learner: {best_E['Meta']}")
print(f"Overfit gap: {best_E['Overfit_Gap']:.4f}")"""),
        md("## 9. Feature Importance"),
        code("""\
params = reconstruct_e_params(
    best_meta["meta"], best_meta["best_params"])
final_model = E_MODEL_CLASSES[best_meta["meta"]](**params)
if best_meta["meta"] not in NAN_TOLERANT_E:
    final_model = wrap_imputer(final_model)
final_model.fit(X_train, y_train)

try:
    if hasattr(final_model, "feature_importances_"):
        importances = final_model.feature_importances_
    elif hasattr(final_model, "named_steps"):
        inner = final_model.named_steps["model"]
        if hasattr(inner, "feature_importances_"):
            importances = inner.feature_importances_
        elif hasattr(inner, "coef_"):
            importances = np.abs(inner.coef_)
        else:
            importances = None
    else:
        importances = None

    if importances is not None:
        fi_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance": importances,
        }).sort_values("Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(fi_df)),
                fi_df["Importance"].values, color="#1976D2")
        ax.set_yticks(range(len(fi_df)))
        ax.set_yticklabels(fi_df["Feature"].values)
        ax.set_xlabel("Importance")
        ax.set_title(
            f"Model E Feature Importance ({best_meta['meta']})")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()

        print("Feature ranking:")
        for _, row in fi_df.iterrows():
            print(f"  {row['Feature']:25s}  "
                  f"{row['Importance']:.4f}")
except Exception as e:
    print(f"Could not extract feature importance: {e}")"""),
        md("## 10. Save Artifacts"),
        code("""\
for _, row in tune_df.iterrows():
    params = reconstruct_e_params(row["meta"], row["best_params"])
    model = E_MODEL_CLASSES[row["meta"]](**params)
    if row["meta"] not in NAN_TOLERANT_E:
        model = wrap_imputer(model)
    model.fit(X_train, y_train)

    model_tag = row["meta"]
    for split_name, X_s, y_s, id_s in [
        ("Training", X_train, y_train, id_train),
        ("Test", X_test, y_test, id_test),
    ]:
        out = id_s.copy()
        out["y_true"] = y_s.values
        out["y_pred"] = model.predict(X_s)
        fname = f"model_E_{model_tag}_{split_name}.parquet"
        uri = save_training_parquet(out, fname, TRAINING_DIR)
        print(f"  Saved {fname} -> {uri}")

    pkl_name = f"Model_E_{model_tag}.pkl"
    uri = gcs_save_model_pickle(model, pkl_name, MODEL_DIR)
    print(f"  Saved {pkl_name} -> {uri}")

print("\\nDone! All Model E artifacts saved.")"""),
        md("## Summary"),
        code("""\
print("=" * 70)
print("MODEL E (RICH FEATURE STACKING) COMPLETE")
print("=" * 70)
print(f"\\nBase models: A={best_A}, B={best_B}, C={best_C}")
print(f"Meta-features: {len(FEATURE_COLS)}")
for f in FEATURE_COLS:
    print(f"  - {f}")
print(f"\\nResults (test set):")
for _, row in test_df.iterrows():
    print(f"  {row['Meta']:20s}  RMSE={row['Test_RMSE']:.4f}  "
          f"R2={row['R2']:.4f}  Spearman={row['Spearman']:.4f}  "
          f"Within-3={row['Within_3']:.1f}%")
best = test_df.iloc[0]
print(f"\\nBest: {best['Meta']}  RMSE={best['Test_RMSE']:.4f}  "
      f"Within-3={best['Within_3']:.1f}%  "
      f"Spearman={best['Spearman']:.4f}")"""),
    ]
    return cells


# ---------------------------------------------------------------------------
# Model F notebook: Lap Time Simulation
# ---------------------------------------------------------------------------


def make_model_f() -> list[dict]:
    cells = [
        md(
            "# 05f — Model F Training: Lap Time Simulation\n\n"
            "Predicts **lap_time_ratio** (lap_time / best_quali_time) for autoregressive\n"
            "race simulation. Positions derived by ranking cumulative predicted times.\n\n"
            "| Feature Group | Count | Description |\n"
            "|---|---|---|\n"
            "| Static | 5 | Grid, quali pace, circuit type |\n"
            "| Deterministic | 10 | Lap, tyre, pit state |\n"
            "| Feedback | 6 | Rolling pace, gaps, position |\n"
            "| Context | 1 | Caution flag |\n"
            "| **Total** | **22** | |\n\n"
            "CV: ExpandingWindowSplit (2019→2020, 2019-20→2021, ..., 2019-23→2024 test)."
        ),
        md("## 0. Setup"),
        code(CHDIR),
        code(IMPORTS),
        code(HELPERS),
        code(OPTUNA_HELPERS),
        code(SAVE_ARTIFACTS),
        md("## 1. Build Training Data"),
        code("""\
from f1_predictor.features.simulation_features import (
    build_simulation_training_data,
    SIMULATION_FEATURE_COLS,
)

laps = load_from_gcs_or_local(
    "data/raw/laps/all_laps.parquet",
    Path("data/raw/laps/all_laps.parquet"),
)
races = load_from_gcs_or_local(
    "data/raw/race/all_races.parquet",
    Path("data/raw/race/all_races.parquet"),
)

df = build_simulation_training_data(laps, races)

# Save processed artifact
sim_dir = Path("data/processed/simulation")
sim_dir.mkdir(parents=True, exist_ok=True)
df.to_parquet(sim_dir / "features_simulation.parquet", index=False)

print(f"Shape: {df.shape}")
print(f"Seasons: {sorted(df['season'].unique())}")
print(f"Races: {df.groupby(['season', 'round']).ngroups}")
print(f"Target stats:\\n{df['lap_time_ratio'].describe()}")
df.head()"""),
        code("""\
FEATURE_COLS = SIMULATION_FEATURE_COLS
TARGET = "lap_time_ratio"
ID_COLS = ["season", "round", "event_name", "driver_abbrev", "team"]

df = df.dropna(subset=[TARGET]).reset_index(drop=True)

X = df[FEATURE_COLS]
y = df[TARGET]
groups = df["season"].values
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"NaN counts:\\n{X.isna().sum()}")"""),
        md("## 2. CV Splitter"),
        code("""\
splitter = ExpandingWindowSplit(
    fold_definitions=[
        ([2019], 2020),
        ([2019, 2020], 2021),
        ([2019, 2020, 2021], 2022),
        ([2019, 2020, 2021, 2022], 2023),
    ],
    test_season=2024,
)
print(f"CV folds: {splitter.get_n_splits()}")
for i, (tr, va) in enumerate(splitter.split(groups)):
    tr_seasons = sorted(set(groups[tr]))
    va_seasons = sorted(set(groups[va]))
    print(f"  Fold {i}: train seasons={tr_seasons}, val seasons={va_seasons}, "
          f"train={len(tr):,}, val={len(va):,}")"""),
        md("## 3. Round 1 — Screen Models (default params)"),
        code("""\
candidates = get_candidates()
r1_results = screen_models(candidates, X, y, splitter, groups)
r1_results[["model", "mean_rmse", "std_rmse", "mean_mae"]]"""),
        code("""\
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(r1_results["model"], r1_results["mean_rmse"], xerr=r1_results["std_rmse"])
ax.set_xlabel("Mean RMSE (CV)")
ax.set_title("Round 1: Model Screening — Model F (Lap Time Ratio)")
ax.invert_yaxis()
plt.tight_layout()
plt.show()"""),
        code("""\
top7_names = r1_results["model"].head(7).tolist()
print(f"Advancing to Round 2: {top7_names}")
eliminated = r1_results["model"].iloc[7:].tolist()
print(f"Eliminated: {eliminated}")"""),
        md("## 4. Round 2 — Optuna HP Tuning (top 7, 10 trials each)"),
        code("""\
r2_results = []
for name in top7_names:
    print(f"Tuning {name}...")
    best_params, best_rmse = run_optuna_round(name, X, y, splitter, groups, n_trials=10)
    r2_results.append({"model": name, "best_rmse": best_rmse, "best_params": best_params})
    print(f"  Best RMSE: {best_rmse:.4f}")

r2_df = pd.DataFrame(r2_results).sort_values("best_rmse").reset_index(drop=True)
r2_df[["model", "best_rmse"]]"""),
        code("""\
top5_names = r2_df["model"].head(5).tolist()
r2_best_params = {row["model"]: row["best_params"] for _, row in r2_df.iterrows()}
print(f"Advancing to Round 3: {top5_names}")"""),
        md("## 5. Round 3 — Final HP Tuning (top 5, 15 trials each)"),
        code("""\
r3_results = []
for name in top5_names:
    print(f"Fine-tuning {name}...")
    best_params, best_rmse = run_optuna_round(name, X, y, splitter, groups, n_trials=15)
    r3_results.append({"model": name, "best_rmse": best_rmse, "best_params": best_params})
    print(f"  Best RMSE: {best_rmse:.4f}")

r3_df = pd.DataFrame(r3_results).sort_values("best_rmse").reset_index(drop=True)
r3_best_params = {row["model"]: row["best_params"] for _, row in r3_df.iterrows()}
r3_df[["model", "best_rmse"]]"""),
        md("## 6. Test Set Evaluation (Per-Lap)"),
        code("""\
train_idx, test_idx = splitter.get_test_split(groups)
X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train_full, y_test = y.iloc[train_idx], y.iloc[test_idx]
id_train, id_test = df[ID_COLS].iloc[train_idx], df[ID_COLS].iloc[test_idx]

print(f"Train: {X_train_full.shape}, Test: {X_test.shape}")
print(f"Test season(s): {sorted(df['season'].iloc[test_idx].unique())}")"""),
        code("""\
final_results = []
for name in top5_names:
    if name in DL_SKIP_OPTUNA:
        model_cls = MODEL_CLASSES[name]
        model = model_cls(input_dim=len(FEATURE_COLS))
    else:
        params = reconstruct_params(name, r3_best_params[name])
        model_cls = MODEL_CLASSES[name]
        model = model_cls(**params)
    model.fit(X_train_full, y_train_full)

    train_preds = model.predict(X_train_full)
    train_rmse = np.sqrt(mean_squared_error(y_train_full, train_preds))
    train_mae = mean_absolute_error(y_train_full, train_preds)

    test_preds = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)

    val_rmse = r3_df.loc[r3_df["model"] == name, "best_rmse"].values[0]

    final_results.append({
        "model": name,
        "train_rmse": train_rmse, "train_mae": train_mae,
        "val_rmse": val_rmse,
        "test_rmse": test_rmse, "test_mae": test_mae,
        "overfit_gap": test_rmse - val_rmse,
    })

    print(f"{name}: train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}, "
          f"test_rmse={test_rmse:.4f}, gap={test_rmse - val_rmse:.4f}")

final_df = pd.DataFrame(final_results).sort_values("test_rmse").reset_index(drop=True)
final_df"""),
        code("""\
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(final_df))
w = 0.25
ax.bar(x - w, final_df["train_rmse"], w, label="Train RMSE")
ax.bar(x, final_df["val_rmse"], w, label="Val RMSE")
ax.bar(x + w, final_df["test_rmse"], w, label="Test RMSE")
ax.set_xticks(x)
ax.set_xticklabels(final_df["model"], rotation=30, ha="right")
ax.set_ylabel("RMSE (lap_time_ratio)")
ax.set_title("Model F — Final 5 Models: Train / Val / Test RMSE")
ax.legend()
plt.tight_layout()
plt.show()"""),
        md(
            "## 7. Full Simulation Evaluation (2024 Test Races)\n\n"
            "Run autoregressive simulation on test season races and compare\n"
            "predicted final positions to actual finishing positions."
        ),
        code("""\
from scipy.stats import spearmanr
from f1_predictor.simulation.engine import RaceSimulator
from f1_predictor.simulation.defaults import build_circuit_defaults

circuit_defaults = build_circuit_defaults(laps)

# Use best per-lap model for simulation
best_model_name = final_df.iloc[0]["model"]
if best_model_name in DL_SKIP_OPTUNA:
    best_model = MODEL_CLASSES[best_model_name](input_dim=len(FEATURE_COLS))
else:
    best_params = reconstruct_params(best_model_name, r3_best_params[best_model_name])
    best_model = MODEL_CLASSES[best_model_name](**best_params)
best_model.fit(X_train_full, y_train_full)

simulator = RaceSimulator(best_model, circuit_defaults)
print(f"Simulator ready with {best_model_name}")
print(f"Circuits available: {len(circuit_defaults)}")"""),
        code("""\
# Get 2024 test races with qualifying data
test_races = races[races["season"] == 2024].copy()
test_race_list = test_races.groupby(["season", "round", "event_name"]).first().reset_index()

sim_results = []
for _, race_row in test_race_list.iterrows():
    event = race_row["event_name"]
    from f1_predictor.features.race_features import LOCATION_ALIASES
    event_norm = LOCATION_ALIASES.get(event, event)

    if event_norm not in circuit_defaults:
        print(f"  Skipping {event} (no circuit data)")
        continue

    # Get drivers for this race
    race_drivers = test_races[
        (test_races["season"] == race_row["season"])
        & (test_races["round"] == race_row["round"])
    ].copy()

    drivers_input = []
    actual_positions = {}
    for _, drv in race_drivers.iterrows():
        q1 = drv.get("q1_time_sec")
        q2 = drv.get("q2_time_sec")
        q3 = drv.get("q3_time_sec")
        q_times = [t for t in [q1, q2, q3] if pd.notna(t)]
        if not q_times or pd.isna(drv.get("grid_position")):
            continue

        drivers_input.append({
            "driver": drv["driver_abbrev"],
            "grid_position": int(drv["grid_position"]),
            "q1": q1 if pd.notna(q1) else None,
            "q2": q2 if pd.notna(q2) else None,
            "q3": q3 if pd.notna(q3) else None,
            "initial_tyre": "MEDIUM",
        })
        if pd.notna(drv.get("finish_position")):
            actual_positions[drv["driver_abbrev"]] = int(drv["finish_position"])

    if len(drivers_input) < 10:
        print(f"  Skipping {event} (only {len(drivers_input)} drivers with quali data)")
        continue

    try:
        result = simulator.simulate(event_norm, drivers_input)
        for fr in result.final_results:
            if fr["driver"] in actual_positions:
                sim_results.append({
                    "event": event,
                    "driver": fr["driver"],
                    "predicted_pos": fr["position"],
                    "actual_pos": actual_positions[fr["driver"]],
                })
        print(f"  {event}: simulated {len(drivers_input)} drivers")
    except Exception as e:
        print(f"  {event}: simulation failed — {e}")

sim_df = pd.DataFrame(sim_results)
print(f"\\nTotal driver-race predictions: {len(sim_df)}")"""),
        code("""\
# Compute simulation-level metrics
from sklearn.metrics import r2_score

if len(sim_df) > 0:
    sim_rmse = np.sqrt(mean_squared_error(sim_df["actual_pos"], sim_df["predicted_pos"]))
    sim_mae = mean_absolute_error(sim_df["actual_pos"], sim_df["predicted_pos"])
    sim_r2 = r2_score(sim_df["actual_pos"], sim_df["predicted_pos"])
    w1 = np.mean(np.abs(sim_df["actual_pos"] - sim_df["predicted_pos"]) <= 1) * 100
    w3 = np.mean(np.abs(sim_df["actual_pos"] - sim_df["predicted_pos"]) <= 3) * 100

    # Per-race Spearman
    spear_vals = []
    for event, grp in sim_df.groupby("event"):
        if len(grp) >= 3 and grp["actual_pos"].std() > 0 and grp["predicted_pos"].std() > 0:
            rho, _ = spearmanr(grp["actual_pos"], grp["predicted_pos"])
            spear_vals.append(rho)
    spearman = np.mean(spear_vals) if spear_vals else float("nan")

    print("=" * 60)
    print("MODEL F SIMULATION RESULTS (2024 Test Season)")
    print("=" * 60)
    print(f"  RMSE:        {sim_rmse:.4f}")
    print(f"  MAE:         {sim_mae:.4f}")
    print(f"  R2:          {sim_r2:.4f}")
    print(f"  Within-1:    {w1:.1f}%")
    print(f"  Within-3:    {w3:.1f}%")
    print(f"  Spearman:    {spearman:.4f}")
    print(f"  Races:       {sim_df['event'].nunique()}")
else:
    print("No simulation results to evaluate.")"""),
        code("""\
# Scatter plot: predicted vs actual positions
if len(sim_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(sim_df["actual_pos"], sim_df["predicted_pos"], alpha=0.3, s=20)
    ax.plot([1, 20], [1, 20], "r--", lw=1)
    ax.set_xlabel("Actual Position")
    ax.set_ylabel("Predicted Position")
    ax.set_title(f"Model F Simulation: Predicted vs Actual (RMSE={sim_rmse:.2f})")

    ax = axes[1]
    errors = sim_df["predicted_pos"] - sim_df["actual_pos"]
    ax.hist(errors, bins=range(-15, 16), edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Position Error (predicted - actual)")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution (Within-3: {w3:.1f}%)")

    plt.tight_layout()
    plt.show()"""),
        md("## 8. Save Artifacts"),
        code("""\
for name in top5_names:
    if name in DL_SKIP_OPTUNA:
        model_cls = MODEL_CLASSES[name]
        model = model_cls(input_dim=len(FEATURE_COLS))
    else:
        params = reconstruct_params(name, r3_best_params[name])
        model_cls = MODEL_CLASSES[name]
        model = model_cls(**params)
    model.fit(X_train_full, y_train_full)

    save_predictions(model, X_train_full, y_train_full, id_train, "F", name, "Training")
    save_predictions(model, X_test, y_test, id_test, "F", name, "Test")

    # OOF validation predictions
    oof_preds = np.full(len(X), np.nan)
    for tr_idx, va_idx in splitter.split(groups):
        import sklearn.base
        fold_model = sklearn.base.clone(model)
        fold_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof_preds[va_idx] = fold_model.predict(X.iloc[va_idx])

    val_mask = ~np.isnan(oof_preds)
    val_out = df[ID_COLS].loc[val_mask].copy()
    val_out["y_true"] = y.loc[val_mask].values
    val_out["y_pred"] = oof_preds[val_mask]
    fname = f"model_F_{name}_Validation.parquet"
    uri = save_training_parquet(val_out, fname, TRAINING_DIR)
    print(f"  Saved {fname} -> {uri}")

    save_model_pkl(model, "F", name)

print("\\nDone! All Model F artifacts saved.")"""),
        md("## Summary"),
        code("""\
print("=" * 60)
print("MODEL F (LAP TIME SIMULATION) TRAINING COMPLETE")
print("=" * 60)
print(f"\\nPer-lap evaluation (top 5, sorted by test RMSE):")
for _, row in final_df.iterrows():
    print(f"  {row['model']:20s}  test_rmse={row['test_rmse']:.6f}  "
          f"gap={row['overfit_gap']:.6f}")
if len(sim_df) > 0:
    print(f"\\nFull simulation (2024 test season):")
    print(f"  Position RMSE: {sim_rmse:.4f}")
    print(f"  Spearman:      {spearman:.4f}")
    print(f"  Within-3:      {w3:.1f}%")
print(f"\\nArtifacts saved to:")
print(f"  Predictions: {TRAINING_DIR}")
print(f"  Models: {MODEL_DIR}")
print(f"  Features: data/processed/simulation/")"""),
    ]
    return cells


# ---------------------------------------------------------------------------
# Model G notebook: Temporal Sequence Models
# ---------------------------------------------------------------------------


def make_model_g() -> list[dict]:
    cells = [
        md(
            "# 05g — Model G Training: Temporal Sequence Models\n\n"
            "Treats each driver's race as a **time series**. A sliding window of\n"
            "the last W laps' features predicts the current lap's `lap_time_ratio`.\n\n"
            "10 GPU candidates: GRU/LSTM/TCN/Transformer/CNN variants.\n"
            "All models Optuna-tunable (no skip).\n\n"
            "CV: ExpandingWindowSplit (2019→2020, ..., 2019-23→2024 test)."
        ),
        md("## 0. Setup"),
        code(CHDIR),
        code(IMPORTS),
        code(SAVE_ARTIFACTS),
        code(PROGRESS_LOGGER.format(model_key="G")),
        code(SLACK_NOTIFIER.format(model_key="G")),
        code(CHECKPOINT_MANAGER.format(model_key="G")),
        code("slack.model_start()\nprogress.log('Starting Model G training')"),
        md("## 1. Build Sequence Training Data"),
        code("""\
from f1_predictor.features.simulation_features import (
    build_simulation_training_data,
    SIMULATION_FEATURE_COLS,
)
from f1_predictor.features.sequence_features import (
    build_sequence_training_data,
    slice_window,
)

laps = load_from_gcs_or_local(
    "data/raw/laps/all_laps.parquet",
    Path("data/raw/laps/all_laps.parquet"),
)
races = load_from_gcs_or_local(
    "data/raw/race/all_races.parquet",
    Path("data/raw/race/all_races.parquet"),
)

# Build tabular simulation data first
df_sim = build_simulation_training_data(laps, races)
print(f"Tabular shape: {df_sim.shape}")

# Reshape into windowed sequences (max_window=10, sliced per candidate)
MAX_WINDOW = 10
X_seq, y_seq, id_df = build_sequence_training_data(df_sim, max_window=MAX_WINDOW)
print(f"X_seq shape: {X_seq.shape}  (samples, window, features+mask)")
print(f"y shape: {y_seq.shape}")
print(f"Target stats:\\n{pd.Series(y_seq).describe()}")"""),
        code("""\
# seasons for CV split — use id_df
groups = id_df["season"].values
n_features_seq = X_seq.shape[2]  # includes mask channel
print(f"Sequence features (incl mask): {n_features_seq}")
print(f"Seasons: {sorted(set(groups))}")"""),
        md("## 2. CV Splitter"),
        code("""\
splitter = ExpandingWindowSplit(
    fold_definitions=[
        ([2019], 2020),
        ([2019, 2020], 2021),
        ([2019, 2020, 2021], 2022),
        ([2019, 2020, 2021, 2022], 2023),
    ],
    test_season=2024,
)
print(f"CV folds: {splitter.get_n_splits()}")
for i, (tr, va) in enumerate(splitter.split(groups)):
    tr_seasons = sorted(set(groups[tr]))
    va_seasons = sorted(set(groups[va]))
    print(f"  Fold {i}: train seasons={tr_seasons}, val seasons={va_seasons}, "
          f"train={len(tr):,}, val={len(va):,}")"""),
        md(
            "## 3. Model Candidates and Helpers\n\n"
            "10 sequence model candidates, all Optuna-tunable."
        ),
        code("""\
from f1_predictor.models.sequence_architectures import (
    SeqGRU_Shallow, SeqGRU_Deep, SeqGRU_Bidir,
    SeqLSTM_Shallow, SeqLSTM_Deep, SeqLSTM_Bidir,
    SeqTCN, SeqTransformer, SeqGRU_Attn, SeqCNN1D,
)

DL_SKIP_OPTUNA = set()  # all models tunable
NAN_TOLERANT = set()  # no tree-based models

MODEL_CLASSES_G = {
    "SeqGRU_Shallow": SeqGRU_Shallow,
    "SeqGRU_Deep": SeqGRU_Deep,
    "SeqGRU_Bidir": SeqGRU_Bidir,
    "SeqLSTM_Shallow": SeqLSTM_Shallow,
    "SeqLSTM_Deep": SeqLSTM_Deep,
    "SeqLSTM_Bidir": SeqLSTM_Bidir,
    "SeqTCN": SeqTCN,
    "SeqTransformer": SeqTransformer,
    "SeqGRU_Attn": SeqGRU_Attn,
    "SeqCNN1D": SeqCNN1D,
}


def get_candidates_g():
    candidates = {}
    if not DL_AVAILABLE:
        print("WARNING: No GPU — sequence models require PyTorch with CUDA/ROCm")
        return candidates
    for name, cls in MODEL_CLASSES_G.items():
        candidates[name] = cls(n_features=n_features_seq)
    print(f"Candidates ({len(candidates)}): {list(candidates.keys())}")
    return candidates


def _seq_optuna_space(trial, model_name):
    params = {
        "n_features": n_features_seq,
        "window_size": trial.suggest_int("window_size", 3, MAX_WINDOW),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048]),
    }
    if model_name in ("SeqGRU_Shallow", "SeqGRU_Deep", "SeqGRU_Bidir",
                       "SeqLSTM_Shallow", "SeqLSTM_Deep", "SeqLSTM_Bidir",
                       "SeqGRU_Attn"):
        params["num_layers"] = trial.suggest_int("num_layers", 1, 4)
    if model_name in ("SeqTCN", "SeqCNN1D"):
        params["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5, 7])
    if model_name == "SeqTCN":
        params["num_layers"] = trial.suggest_int("num_layers", 2, 6)
    if model_name == "SeqTransformer":
        params["num_layers"] = trial.suggest_int("num_layers", 1, 4)
        params["n_heads"] = trial.suggest_categorical("n_heads", [2, 4, 8])
        # d_model must be divisible by n_heads
        hd = params["hidden_dim"]
        nh = params["n_heads"]
        if hd % nh != 0:
            params["hidden_dim"] = max(nh, (hd // nh) * nh)
    return params


def get_optuna_param_space_g(name, trial):
    return _seq_optuna_space(trial, name)


def reconstruct_params_g(name, best_params):
    params = dict(best_params)
    params["n_features"] = n_features_seq
    return params"""),
        code("""\
def cv_evaluate_g(model, X_seq_full, y_full, splitter, groups):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmses, maes = [], []
    for tr_idx, va_idx in splitter.split(groups):
        import sklearn.base
        m = sklearn.base.clone(model)
        X_tr, y_tr = X_seq_full[tr_idx], y_full[tr_idx]
        X_va, y_va = X_seq_full[va_idx], y_full[va_idx]
        m.fit(X_tr, y_tr)
        preds = m.predict(X_va)
        rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
        maes.append(mean_absolute_error(y_va, preds))
    return {"mean_rmse": np.mean(rmses), "std_rmse": np.std(rmses), "mean_mae": np.mean(maes)}


def screen_models_g(candidates, X_seq_full, y_full, splitter, groups):
    rows = []
    completed = ckpt.get_completed(1)
    total = len(candidates)
    for idx, (name, model) in enumerate(candidates.items(), 1):
        if name in completed:
            cp = completed[name]
            rows.append({"model": name, "mean_rmse": cp["rmse"],
                         "std_rmse": cp.get("std_rmse", 0), "mean_mae": cp.get("mean_mae", 0)})
            progress.log(f"Screening {idx}/{total} {name} -- RESUMED (RMSE: {cp['rmse']:.6f})")
            continue
        try:
            result = cv_evaluate_g(model, X_seq_full, y_full, splitter, groups)
            rows.append({"model": name, **result})
            progress.screening(name, idx, total, rmse=result["mean_rmse"])
            ckpt.save_checkpoint(1, name, result["mean_rmse"], {},
                                 std_rmse=result["std_rmse"], mean_mae=result["mean_mae"])
        except Exception as e:
            progress.screening(name, idx, total, error=str(e))
            slack.error(f"R1 screening {name}", str(e))
    if not rows:
        raise RuntimeError("All candidates failed — check data for NaN or shape issues")
    return pd.DataFrame(rows).sort_values("mean_rmse").reset_index(drop=True)


def run_optuna_round_g(name, X_seq_full, y_full, splitter, groups, n_trials):
    def objective(trial):
        params = get_optuna_param_space_g(name, trial)
        ws = params.pop("window_size", MAX_WINDOW)
        model_cls = MODEL_CLASSES_G[name]
        model = model_cls(**params, window_size=ws)
        X_sliced = slice_window(X_seq_full, ws)
        result = cv_evaluate_g(model, X_sliced, y_full, splitter, groups)
        return result["mean_rmse"]
    def _trial_cb(study, trial):
        if trial.value is not None:
            progress.optuna_trial(name, trial.number + 1, n_trials, trial.value, study.best_value)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        objective, n_trials=n_trials,
        catch=(Exception,), show_progress_bar=False,
        callbacks=[_trial_cb],
    )
    return study.best_params, study.best_value"""),
        md("## 4. Round 1 — Screen Models"),
        code("""\
progress.round_header(1, "Screen all candidates")
slack.round_start(1, "Screening all candidates", len(get_candidates_g()))
candidates = get_candidates_g()
r1_results = screen_models_g(candidates, X_seq, y_seq, splitter, groups)
ckpt.save_round_summary(1, r1_results.to_dict("records"), r1_results["model"].head(7).tolist())
slack.round_complete(1, f"Best: {r1_results.iloc[0]['model']} (RMSE: {r1_results.iloc[0]['mean_rmse']:.6f})")
r1_results[["model", "mean_rmse", "std_rmse", "mean_mae"]]"""),
        code("""\
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(r1_results["model"], r1_results["mean_rmse"], xerr=r1_results["std_rmse"])
ax.set_xlabel("Mean RMSE (CV)")
ax.set_title("Round 1: Model Screening — Model G (Temporal Sequences)")
ax.invert_yaxis()
plt.tight_layout()
plt.show()"""),
        code("""\
r1_summary = ckpt.load_round_summary(1)
if r1_summary:
    top7_names = r1_summary["top_names"]
else:
    top7_names = r1_results["model"].head(7).tolist()
print(f"Advancing to Round 2: {top7_names}")
eliminated = [n for n in r1_results["model"].tolist() if n not in top7_names]
print(f"Eliminated: {eliminated}")"""),
        md("## 5. Round 2 — Optuna (top 7, 10 trials each)"),
        code("""\
progress.round_header(2, "Optuna HP tuning (top 7, 10 trials each)")
slack.round_start(2, "Optuna HP tuning", len(top7_names))
r2_results = []
completed_r2 = ckpt.get_completed(2)
for idx, name in enumerate(top7_names, 1):
    if name in completed_r2:
        cp = completed_r2[name]
        r2_results.append({"model": name, "best_rmse": cp["rmse"], "best_params": cp["best_params"]})
        progress.log(f"{name} -- RESUMED from checkpoint (RMSE: {cp['rmse']:.6f})")
        continue
    progress.log(f"Tuning {name} ({idx}/{len(top7_names)})...")
    try:
        best_params, best_rmse = run_optuna_round_g(
            name, X_seq, y_seq, splitter, groups, n_trials=10)
        r2_results.append({"model": name, "best_rmse": best_rmse, "best_params": best_params})
        ckpt.save_checkpoint(2, name, best_rmse, best_params)
        progress.model_complete(name, 2, best_rmse)
        slack.architecture_done(name, 2, best_rmse)
    except Exception as e:
        progress.log(f"{name} FAILED in Round 2: {e}")
        slack.error(f"R2 {name}", str(e))

r2_df = pd.DataFrame(r2_results).sort_values("best_rmse").reset_index(drop=True)
ckpt.save_round_summary(2, r2_results, r2_df["model"].head(5).tolist())
slack.round_complete(2, f"Top 5: {r2_df['model'].head(5).tolist()}")
r2_df[["model", "best_rmse"]]"""),
        code("""\
r2_summary = ckpt.load_round_summary(2)
if r2_summary:
    top5_names = r2_summary["top_names"]
else:
    top5_names = r2_df["model"].head(5).tolist()
r2_best_params = {row["model"]: row["best_params"] for _, row in r2_df.iterrows() if "best_params" in row}
print(f"Advancing to Round 3: {top5_names}")"""),
        md("## 6. Round 3 — Final Tuning (top 5, 15 trials each)"),
        code("""\
progress.round_header(3, "Final tuning (top 5, 15 trials each)")
slack.round_start(3, "Final Optuna tuning", len(top5_names))
r3_results = []
completed_r3 = ckpt.get_completed(3)
for idx, name in enumerate(top5_names, 1):
    if name in completed_r3:
        cp = completed_r3[name]
        r3_results.append({"model": name, "best_rmse": cp["rmse"], "best_params": cp["best_params"]})
        progress.log(f"{name} -- RESUMED from checkpoint (RMSE: {cp['rmse']:.6f})")
        continue
    progress.log(f"Fine-tuning {name} ({idx}/{len(top5_names)})...")
    try:
        best_params, best_rmse = run_optuna_round_g(
            name, X_seq, y_seq, splitter, groups, n_trials=15)
        r3_results.append({"model": name, "best_rmse": best_rmse, "best_params": best_params})
        ckpt.save_checkpoint(3, name, best_rmse, best_params)
        progress.model_complete(name, 3, best_rmse)
        slack.architecture_done(name, 3, best_rmse)
    except Exception as e:
        progress.log(f"{name} FAILED in Round 3: {e}")
        slack.error(f"R3 {name}", str(e))

r3_df = pd.DataFrame(r3_results).sort_values("best_rmse").reset_index(drop=True)
r3_best_params = {row["model"]: row["best_params"] for _, row in r3_df.iterrows() if "best_params" in row}
ckpt.save_round_summary(3, r3_results, r3_df["model"].head(5).tolist())
slack.round_complete(3, f"Best: {r3_df.iloc[0]['model']} (RMSE: {r3_df.iloc[0]['best_rmse']:.6f})")
r3_df[["model", "best_rmse"]]"""),
        md("## 7. Test Set Evaluation (Per-Lap)"),
        code("""\
train_idx, test_idx = splitter.get_test_split(groups)
X_seq_train, X_seq_test = X_seq[train_idx], X_seq[test_idx]
y_train_full, y_test = y_seq[train_idx], y_seq[test_idx]
id_train, id_test = id_df.iloc[train_idx], id_df.iloc[test_idx]

print(f"Train: {X_seq_train.shape}, Test: {X_seq_test.shape}")
print(f"Test season(s): {sorted(set(groups[test_idx]))}")"""),
        code("""\
final_results = []
for name in top5_names:
    params = reconstruct_params_g(name, r3_best_params[name])
    ws = params.pop("window_size", MAX_WINDOW)
    model_cls = MODEL_CLASSES_G[name]
    model = model_cls(**params, window_size=ws)

    X_tr_sliced = slice_window(X_seq_train, ws)
    X_te_sliced = slice_window(X_seq_test, ws)

    model.fit(X_tr_sliced, y_train_full)

    train_preds = model.predict(X_tr_sliced)
    train_rmse = np.sqrt(mean_squared_error(y_train_full, train_preds))

    test_preds = model.predict(X_te_sliced)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

    val_rmse = r3_df.loc[r3_df["model"] == name, "best_rmse"].values[0]

    final_results.append({
        "model": name,
        "train_rmse": train_rmse, "val_rmse": val_rmse,
        "test_rmse": test_rmse, "overfit_gap": test_rmse - val_rmse,
        "window_size": ws,
    })
    print(f"{name} (ws={ws}): train_rmse={train_rmse:.4f}, "
          f"val_rmse={val_rmse:.4f}, test_rmse={test_rmse:.4f}")

final_df = pd.DataFrame(final_results).sort_values("test_rmse").reset_index(drop=True)
final_df"""),
        md("## 8. Sequence Simulation (2024 Test Races)"),
        code("""\
from f1_predictor.simulation.sequence_simulator import SequenceRaceSimulator
from f1_predictor.simulation.defaults import build_circuit_defaults
from f1_predictor.simulation.evaluation import evaluate_simulation
from f1_predictor.features.race_features import LOCATION_ALIASES

circuit_defaults = build_circuit_defaults(laps)

# Retrain best model on full training set
best_row = final_df.iloc[0]
best_model_name = best_row["model"]
best_ws = int(best_row["window_size"])
best_params = reconstruct_params_g(best_model_name, r3_best_params[best_model_name])
best_params.pop("window_size", None)
best_model = MODEL_CLASSES_G[best_model_name](**best_params, window_size=best_ws)
X_tr_sliced = slice_window(X_seq_train, best_ws)
best_model.fit(X_tr_sliced, y_train_full)

seq_sim = SequenceRaceSimulator(best_model, circuit_defaults, window_size=best_ws)
print(f"Sequence simulator: {best_model_name} (window={best_ws})")"""),
        code("""\
test_races = races[races["season"] == 2024].copy()
test_race_list = test_races.groupby(
    ["season", "round", "event_name"]
).first().reset_index()

sim_results = []
for _, race_row in test_race_list.iterrows():
    event = race_row["event_name"]
    event_norm = LOCATION_ALIASES.get(event, event)

    if event_norm not in circuit_defaults:
        print(f"  Skipping {event} (no circuit data)")
        continue

    race_drivers = test_races[
        (test_races["season"] == race_row["season"])
        & (test_races["round"] == race_row["round"])
    ].copy()

    drivers_input = []
    actual_positions = {}
    for _, drv in race_drivers.iterrows():
        q1 = drv.get("q1_time_sec")
        q2 = drv.get("q2_time_sec")
        q3 = drv.get("q3_time_sec")
        q_times = [t for t in [q1, q2, q3] if pd.notna(t)]
        if not q_times or pd.isna(drv.get("grid_position")):
            continue
        drivers_input.append({
            "driver": drv["driver_abbrev"],
            "grid_position": int(drv["grid_position"]),
            "q1": q1 if pd.notna(q1) else None,
            "q2": q2 if pd.notna(q2) else None,
            "q3": q3 if pd.notna(q3) else None,
            "initial_tyre": "MEDIUM",
        })
        if pd.notna(drv.get("finish_position")):
            actual_positions[drv["driver_abbrev"]] = int(drv["finish_position"])

    if len(drivers_input) < 10:
        continue

    try:
        result = seq_sim.simulate(event_norm, drivers_input)
        for fr in result.final_results:
            if fr["driver"] in actual_positions:
                sim_results.append({
                    "event": event, "driver": fr["driver"],
                    "predicted_pos": fr["position"],
                    "actual_pos": actual_positions[fr["driver"]],
                })
        print(f"  {event}: simulated {len(drivers_input)} drivers")
    except Exception as e:
        print(f"  {event}: failed — {e}")

sim_df = pd.DataFrame(sim_results)
print(f"\\nSimulation results: {len(sim_df)} driver-race predictions")"""),
        code("""\
if len(sim_df) > 0:
    sim_metrics = evaluate_simulation(sim_df)
    print("=" * 60)
    print("MODEL G — SEQUENCE SIMULATION (2024)")
    print("=" * 60)
    for k, v in sim_metrics.items():
        print(f"  {k:20s}: {v}")"""),
        md("## 9. Save Artifacts"),
        code("""\
ID_COLS_SEQ = ["season", "round", "driver_abbrev", "lap_number"]

for name in top5_names:
    params = reconstruct_params_g(name, r3_best_params[name])
    ws = params.pop("window_size", MAX_WINDOW)
    model_cls = MODEL_CLASSES_G[name]
    model = model_cls(**params, window_size=ws)

    X_tr_s = slice_window(X_seq_train, ws)
    X_te_s = slice_window(X_seq_test, ws)
    model.fit(X_tr_s, y_train_full)

    # Training predictions
    train_preds = model.predict(X_tr_s)
    out = id_train.copy()
    out["y_true"] = y_train_full
    out["y_pred"] = train_preds
    fname = f"model_G_{name}_Training.parquet"
    uri = save_training_parquet(out, fname, TRAINING_DIR)
    print(f"  Saved {fname} -> {uri}")

    # Test predictions
    test_preds = model.predict(X_te_s)
    out = id_test.copy()
    out["y_true"] = y_test
    out["y_pred"] = test_preds
    fname = f"model_G_{name}_Test.parquet"
    uri = save_training_parquet(out, fname, TRAINING_DIR)
    print(f"  Saved {fname} -> {uri}")

    # OOF validation predictions
    oof_preds = np.full(len(y_seq), np.nan)
    for tr_idx, va_idx in splitter.split(groups):
        import sklearn.base
        fold_model = sklearn.base.clone(model)
        X_tr_fold = slice_window(X_seq[tr_idx], ws)
        fold_model.fit(X_tr_fold, y_seq[tr_idx])
        X_va_fold = slice_window(X_seq[va_idx], ws)
        oof_preds[va_idx] = fold_model.predict(X_va_fold)

    val_mask = ~np.isnan(oof_preds)
    val_out = id_df.loc[val_mask].copy()
    val_out["y_true"] = y_seq[val_mask]
    val_out["y_pred"] = oof_preds[val_mask]
    fname = f"model_G_{name}_Validation.parquet"
    uri = save_training_parquet(val_out, fname, TRAINING_DIR)
    print(f"  Saved {fname} -> {uri}")

    save_model_pkl(model, "G", name)

print("\\nDone! All Model G artifacts saved.")"""),
        md("## Summary"),
        code("""\
print("=" * 60)
print("MODEL G (TEMPORAL SEQUENCE) TRAINING COMPLETE")
print("=" * 60)
print(f"\\nPer-lap evaluation (top 5, sorted by test RMSE):")
for _, row in final_df.iterrows():
    print(f"  {row['model']:25s} ws={int(row['window_size'])} "
          f"test_rmse={row['test_rmse']:.6f} gap={row['overfit_gap']:.6f}")

if len(sim_df) > 0:
    print(f"\\nSequence simulation (2024):")
    print(f"  Position RMSE: {sim_metrics['position_rmse']:.4f}")
    print(f"  Spearman:      {sim_metrics['spearman_mean']:.4f}")
    print(f"  Within-3:      {sim_metrics['within_3']:.1f}%")
print(f"\\nArtifacts saved to:")
print(f"  Predictions: {TRAINING_DIR}")
print(f"  Models: {MODEL_DIR}")

best_name = final_df.iloc[0]["model"]
best_rmse = final_df.iloc[0]["test_rmse"]
progress.log(f"MODEL G TRAINING COMPLETE -- best: {best_name} (RMSE: {best_rmse:.6f})")
slack.model_complete(best_name, best_rmse)"""),
    ]
    return cells


# ---------------------------------------------------------------------------
# Model H notebook: Delta + Monte Carlo
# ---------------------------------------------------------------------------


def make_model_h() -> list[dict]:
    cells = [
        md(
            "# 05h — Model H Training: Delta + Monte Carlo Simulation\n\n"
            "Predicts **delta_ratio** (lap_time_ratio - field_median_ratio) to cancel\n"
            "systematic bias. Monte Carlo (N=200) averages out random errors.\n\n"
            "Same 25 features as Model F, different target + MC inference.\n\n"
            "CV: ExpandingWindowSplit (2019→2020 ... 2019-23→2024 test)."
        ),
        md("## 0. Setup"),
        code(CHDIR),
        code(IMPORTS),
        code(SAVE_ARTIFACTS),
        code(PROGRESS_LOGGER.format(model_key="H")),
        code(SLACK_NOTIFIER.format(model_key="H")),
        code(CHECKPOINT_MANAGER.format(model_key="H")),
        code("slack.model_start()\nprogress.log('Starting Model H training')"),
        md("## 1. Build Training Data (Delta Target)"),
        code("""\
from f1_predictor.features.simulation_features import (
    build_simulation_training_data,
    SIMULATION_FEATURE_COLS,
)
from f1_predictor.features.delta_features import (
    build_field_median_curves,
    build_delta_training_data,
)

laps = load_from_gcs_or_local(
    "data/raw/laps/all_laps.parquet",
    Path("data/raw/laps/all_laps.parquet"),
)
races = load_from_gcs_or_local(
    "data/raw/race/all_races.parquet",
    Path("data/raw/race/all_races.parquet"),
)

# Build base simulation features
df_base = build_simulation_training_data(laps, races)
print(f"Base shape: {df_base.shape}")

# Build field median curves and delta target
field_medians = build_field_median_curves(laps, races)
print(f"Circuits with median curves: {len(field_medians)}")

df = build_delta_training_data(df_base, field_medians, races)
print(f"Delta shape: {df.shape}")
print(f"delta_ratio stats:\\n{df['delta_ratio'].describe()}")"""),
        code("""\
FEATURE_COLS = SIMULATION_FEATURE_COLS
TARGET = "delta_ratio"
ID_COLS = ["season", "round", "event_name", "driver_abbrev", "team"]

df = df.dropna(subset=[TARGET]).reset_index(drop=True)

X = df[FEATURE_COLS]
y = df[TARGET]
groups = df["season"].values
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"NaN counts:\\n{X.isna().sum()}")"""),
        md("## 2. CV Splitter"),
        code("""\
splitter = ExpandingWindowSplit(
    fold_definitions=[
        ([2019], 2020),
        ([2019, 2020], 2021),
        ([2019, 2020, 2021], 2022),
        ([2019, 2020, 2021, 2022], 2023),
    ],
    test_season=2024,
)
print(f"CV folds: {splitter.get_n_splits()}")
for i, (tr, va) in enumerate(splitter.split(groups)):
    tr_seasons = sorted(set(groups[tr]))
    va_seasons = sorted(set(groups[va]))
    print(f"  Fold {i}: train seasons={tr_seasons}, val seasons={va_seasons}, "
          f"train={len(tr):,}, val={len(va):,}")"""),
        md(
            "## 3. Model Candidates and Helpers\n\n"
            "9 GPU candidates: 6 tree-based + 3 DL, all on delta target.\n"
            "DL models are Optuna-tunable (no skip)."
        ),
        code("""\
NAN_TOLERANT = {
    "XGBoost_Delta", "XGBoost_DART_Delta", "XGBoost_Deep_Delta",
    "LightGBM_Delta", "LightGBM_GOSS_Delta", "LightGBM_Shallow_Delta",
}

DL_SKIP_OPTUNA = set()  # all models tunable

def get_candidates_h():
    xgb_device = get_xgboost_device(GPU_BACKEND)
    lgb_device = get_lightgbm_device(GPU_BACKEND)
    candidates = {
        "XGBoost_Delta": xgb.XGBRegressor(
            n_estimators=300, n_jobs=-1, random_state=42, verbosity=0, **xgb_device),
        "XGBoost_DART_Delta": xgb.XGBRegressor(
            n_estimators=300, booster="dart", n_jobs=-1,
            random_state=42, verbosity=0, **xgb_device),
        "XGBoost_Deep_Delta": xgb.XGBRegressor(
            n_estimators=300, max_depth=10, learning_rate=0.1,
            n_jobs=-1, random_state=42, verbosity=0, **xgb_device),
        "LightGBM_Delta": lgb.LGBMRegressor(
            n_estimators=300, n_jobs=-1, random_state=42, verbose=-1, **lgb_device),
        "LightGBM_GOSS_Delta": lgb.LGBMRegressor(
            n_estimators=300, boosting_type="goss", n_jobs=-1,
            random_state=42, verbose=-1, **lgb_device),
        "LightGBM_Shallow_Delta": lgb.LGBMRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            n_jobs=-1, random_state=42, verbose=-1, **lgb_device),
    }
    if DL_AVAILABLE:
        n_feat = len(FEATURE_COLS)
        candidates["GRU_Delta"] = GRU2Layer(input_dim=n_feat)
        candidates["FTTransformer_Delta"] = FTTransformerWrapper(n_features=n_feat)
        candidates["MLP_Delta"] = MLP3Layer(input_dim=n_feat, hidden1=128, hidden2=64)
    print(f"Candidates ({len(candidates)}): {list(candidates.keys())}")
    return candidates

MODEL_CLASSES_H = {
    "XGBoost_Delta": xgb.XGBRegressor,
    "XGBoost_DART_Delta": xgb.XGBRegressor,
    "XGBoost_Deep_Delta": xgb.XGBRegressor,
    "LightGBM_Delta": lgb.LGBMRegressor,
    "LightGBM_GOSS_Delta": lgb.LGBMRegressor,
    "LightGBM_Shallow_Delta": lgb.LGBMRegressor,
}
if DL_AVAILABLE:
    MODEL_CLASSES_H.update({
        "GRU_Delta": GRU2Layer,
        "FTTransformer_Delta": FTTransformerWrapper,
        "MLP_Delta": MLP3Layer,
    })

def _xgb_base_space_h(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1500),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )

def _lgb_base_space_h(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1500),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )

def _dl_base_space_h(trial, model_cls):
    params = dict(
        input_dim=len(FEATURE_COLS),
    )
    if model_cls == MLP3Layer:
        params["hidden1"] = trial.suggest_categorical("hidden1", [64, 128, 256])
        params["hidden2"] = trial.suggest_categorical("hidden2", [32, 64, 128])
        params["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
        params["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        params["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    else:
        params["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
        params["num_layers"] = trial.suggest_int("num_layers", 1, 3)
        params["dropout"] = trial.suggest_float("dropout", 0.05, 0.5)
        params["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        params["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    if model_cls == FTTransformerWrapper:
        params["n_features"] = len(FEATURE_COLS)
        params.pop("input_dim", None)
    return params

def get_optuna_param_space_h(name, trial):
    xgb_device = get_xgboost_device(GPU_BACKEND)
    lgb_device = get_lightgbm_device(GPU_BACKEND)
    if name.startswith("XGBoost"):
        params = _xgb_base_space_h(trial)
        if "DART" in name:
            params["booster"] = "dart"
            params["rate_drop"] = trial.suggest_float("rate_drop", 0.01, 0.5)
        if "Deep" in name:
            pass  # max_depth already in search space
        params.update(n_jobs=-1, random_state=42, verbosity=0, **xgb_device)
        return params
    elif name.startswith("LightGBM"):
        params = _lgb_base_space_h(trial)
        if "GOSS" in name:
            params.pop("subsample", None)
            params["boosting_type"] = "goss"
        if "Shallow" in name:
            pass
        params.update(n_jobs=-1, random_state=42, verbose=-1, **lgb_device)
        return params
    elif name in ("GRU_Delta", "FTTransformer_Delta", "MLP_Delta"):
        return _dl_base_space_h(trial, MODEL_CLASSES_H[name])
    return {}

def reconstruct_params_h(name, best_params):
    params = dict(best_params)
    xgb_device = get_xgboost_device(GPU_BACKEND)
    lgb_device = get_lightgbm_device(GPU_BACKEND)
    if name.startswith("XGBoost"):
        if "DART" in name:
            params.setdefault("booster", "dart")
        params.update(n_jobs=-1, random_state=42, verbosity=0, **xgb_device)
    elif name.startswith("LightGBM"):
        if "GOSS" in name:
            params.pop("subsample", None)
            params["boosting_type"] = "goss"
        params.update(n_jobs=-1, random_state=42, verbose=-1, **lgb_device)
    return params"""),
        code("""\
def cv_evaluate_h(model, X, y, splitter, groups):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmses, maes = [], []
    for tr_idx, va_idx in splitter.split(groups):
        import sklearn.base
        m = sklearn.base.clone(model)
        X_tr = X.iloc[tr_idx] if hasattr(X, "iloc") else X[tr_idx]
        y_tr = y.iloc[tr_idx] if hasattr(y, "iloc") else y[tr_idx]
        X_va = X.iloc[va_idx] if hasattr(X, "iloc") else X[va_idx]
        y_va = y.iloc[va_idx] if hasattr(y, "iloc") else y[va_idx]
        name_str = type(model).__name__
        if name_str in str(NAN_TOLERANT) or any(n in str(type(model)) for n in ["XGB", "LGB"]):
            m.fit(X_tr, y_tr)
        else:
            X_tr_clean = X_tr.fillna(0) if hasattr(X_tr, "fillna") else np.nan_to_num(X_tr)
            X_va_clean = X_va.fillna(0) if hasattr(X_va, "fillna") else np.nan_to_num(X_va)
            m.fit(X_tr_clean, y_tr)
            X_va = X_va_clean
        preds = m.predict(X_va)
        rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
        maes.append(mean_absolute_error(y_va, preds))
    return {"mean_rmse": np.mean(rmses), "std_rmse": np.std(rmses), "mean_mae": np.mean(maes)}

def screen_models_h(candidates, X, y, splitter, groups):
    rows = []
    completed = ckpt.get_completed(1)
    total = len(candidates)
    for idx, (name, model) in enumerate(candidates.items(), 1):
        if name in completed:
            cp = completed[name]
            rows.append({"model": name, "mean_rmse": cp["rmse"],
                         "std_rmse": cp.get("std_rmse", 0), "mean_mae": cp.get("mean_mae", 0)})
            progress.log(f"Screening {idx}/{total} {name} -- RESUMED (RMSE: {cp['rmse']:.6f})")
            continue
        try:
            result = cv_evaluate_h(model, X, y, splitter, groups)
            rows.append({"model": name, **result})
            progress.screening(name, idx, total, rmse=result["mean_rmse"])
            ckpt.save_checkpoint(1, name, result["mean_rmse"], {},
                                 std_rmse=result["std_rmse"], mean_mae=result["mean_mae"])
        except Exception as e:
            progress.screening(name, idx, total, error=str(e))
            slack.error(f"R1 screening {name}", str(e))
    if not rows:
        raise RuntimeError("All candidates failed — check data for NaN or shape issues")
    return pd.DataFrame(rows).sort_values("mean_rmse").reset_index(drop=True)

def run_optuna_round_h(name, X, y, splitter, groups, n_trials):
    def objective(trial):
        params = get_optuna_param_space_h(name, trial)
        model_cls = MODEL_CLASSES_H[name]
        model = model_cls(**params)
        result = cv_evaluate_h(model, X, y, splitter, groups)
        return result["mean_rmse"]
    def _trial_cb(study, trial):
        if trial.value is not None:
            progress.optuna_trial(name, trial.number + 1, n_trials, trial.value, study.best_value)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, catch=(Exception,), show_progress_bar=False,
                   callbacks=[_trial_cb])
    return study.best_params, study.best_value"""),
        md("## 4. Round 1 — Screen Models"),
        code("""\
progress.round_header(1, "Screen all candidates")
slack.round_start(1, "Screening all candidates", len(get_candidates_h()))
candidates = get_candidates_h()
r1_results = screen_models_h(candidates, X, y, splitter, groups)
ckpt.save_round_summary(1, r1_results.to_dict("records"), r1_results["model"].head(7).tolist())
slack.round_complete(1, f"Best: {r1_results.iloc[0]['model']} (RMSE: {r1_results.iloc[0]['mean_rmse']:.6f})")
r1_results[["model", "mean_rmse", "std_rmse", "mean_mae"]]"""),
        code("""\
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(r1_results["model"], r1_results["mean_rmse"], xerr=r1_results["std_rmse"])
ax.set_xlabel("Mean RMSE (CV)")
ax.set_title("Round 1: Model Screening — Model H (Delta Ratio)")
ax.invert_yaxis()
plt.tight_layout()
plt.show()"""),
        code("""\
r1_summary = ckpt.load_round_summary(1)
if r1_summary:
    top7_names = r1_summary["top_names"]
else:
    top7_names = r1_results["model"].head(7).tolist()
print(f"Advancing to Round 2: {top7_names}")
eliminated = [n for n in r1_results["model"].tolist() if n not in top7_names]
print(f"Eliminated: {eliminated}")"""),
        md("## 5. Round 2 — Optuna (top 7, 10 trials each)"),
        code("""\
progress.round_header(2, "Optuna HP tuning (top 7, 10 trials each)")
slack.round_start(2, "Optuna HP tuning", len(top7_names))
r2_results = []
completed_r2 = ckpt.get_completed(2)
for idx, name in enumerate(top7_names, 1):
    if name in completed_r2:
        cp = completed_r2[name]
        r2_results.append({"model": name, "best_rmse": cp["rmse"], "best_params": cp["best_params"]})
        progress.log(f"{name} -- RESUMED from checkpoint (RMSE: {cp['rmse']:.6f})")
        continue
    progress.log(f"Tuning {name} ({idx}/{len(top7_names)})...")
    try:
        best_params, best_rmse = run_optuna_round_h(name, X, y, splitter, groups, n_trials=10)
        r2_results.append({"model": name, "best_rmse": best_rmse, "best_params": best_params})
        ckpt.save_checkpoint(2, name, best_rmse, best_params)
        progress.model_complete(name, 2, best_rmse)
        slack.architecture_done(name, 2, best_rmse)
    except Exception as e:
        progress.log(f"{name} FAILED in Round 2: {e}")
        slack.error(f"R2 {name}", str(e))

r2_df = pd.DataFrame(r2_results).sort_values("best_rmse").reset_index(drop=True)
ckpt.save_round_summary(2, r2_results, r2_df["model"].head(5).tolist())
slack.round_complete(2, f"Top 5: {r2_df['model'].head(5).tolist()}")
r2_df[["model", "best_rmse"]]"""),
        code("""\
r2_summary = ckpt.load_round_summary(2)
if r2_summary:
    top5_names = r2_summary["top_names"]
else:
    top5_names = r2_df["model"].head(5).tolist()
r2_best_params = {row["model"]: row["best_params"] for _, row in r2_df.iterrows() if "best_params" in row}
print(f"Advancing to Round 3: {top5_names}")"""),
        md("## 6. Round 3 — Final Tuning (top 5, 15 trials each)"),
        code("""\
progress.round_header(3, "Final tuning (top 5, 15 trials each)")
slack.round_start(3, "Final Optuna tuning", len(top5_names))
r3_results = []
completed_r3 = ckpt.get_completed(3)
for idx, name in enumerate(top5_names, 1):
    if name in completed_r3:
        cp = completed_r3[name]
        r3_results.append({"model": name, "best_rmse": cp["rmse"], "best_params": cp["best_params"]})
        progress.log(f"{name} -- RESUMED from checkpoint (RMSE: {cp['rmse']:.6f})")
        continue
    progress.log(f"Fine-tuning {name} ({idx}/{len(top5_names)})...")
    try:
        best_params, best_rmse = run_optuna_round_h(name, X, y, splitter, groups, n_trials=15)
        r3_results.append({"model": name, "best_rmse": best_rmse, "best_params": best_params})
        ckpt.save_checkpoint(3, name, best_rmse, best_params)
        progress.model_complete(name, 3, best_rmse)
        slack.architecture_done(name, 3, best_rmse)
    except Exception as e:
        progress.log(f"{name} FAILED in Round 3: {e}")
        slack.error(f"R3 {name}", str(e))

r3_df = pd.DataFrame(r3_results).sort_values("best_rmse").reset_index(drop=True)
r3_best_params = {row["model"]: row["best_params"] for _, row in r3_df.iterrows() if "best_params" in row}
ckpt.save_round_summary(3, r3_results, r3_df["model"].head(5).tolist())
slack.round_complete(3, f"Best: {r3_df.iloc[0]['model']} (RMSE: {r3_df.iloc[0]['best_rmse']:.6f})")
r3_df[["model", "best_rmse"]]"""),
        md("## 7. Test Set Evaluation (Per-Lap Delta)"),
        code("""\
train_idx, test_idx = splitter.get_test_split(groups)
X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train_full, y_test = y.iloc[train_idx], y.iloc[test_idx]
id_train, id_test = df[ID_COLS].iloc[train_idx], df[ID_COLS].iloc[test_idx]

print(f"Train: {X_train_full.shape}, Test: {X_test.shape}")
print(f"Test season(s): {sorted(df['season'].iloc[test_idx].unique())}")"""),
        code("""\
final_results = []
for name in top5_names:
    params = reconstruct_params_h(name, r3_best_params[name])
    model_cls = MODEL_CLASSES_H[name]
    model = model_cls(**params)
    model.fit(X_train_full, y_train_full)

    train_preds = model.predict(X_train_full)
    train_rmse = np.sqrt(mean_squared_error(y_train_full, train_preds))

    test_preds = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

    val_rmse = r3_df.loc[r3_df["model"] == name, "best_rmse"].values[0]

    final_results.append({
        "model": name,
        "train_rmse": train_rmse, "val_rmse": val_rmse,
        "test_rmse": test_rmse, "overfit_gap": test_rmse - val_rmse,
    })

    print(f"{name}: train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}, "
          f"test_rmse={test_rmse:.4f}, gap={test_rmse - val_rmse:.4f}")

final_df = pd.DataFrame(final_results).sort_values("test_rmse").reset_index(drop=True)
final_df"""),
        md("## 8. Fit Residual Distribution for Monte Carlo"),
        code("""\
# Compute OOF residuals to calibrate MC noise
best_model_name = final_df.iloc[0]["model"]
best_params = reconstruct_params_h(best_model_name, r3_best_params[best_model_name])
best_model = MODEL_CLASSES_H[best_model_name](**best_params)

oof_residuals = np.full(len(X), np.nan)
for tr_idx, va_idx in splitter.split(groups):
    import sklearn.base
    fold_model = sklearn.base.clone(best_model)
    fold_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    preds = fold_model.predict(X.iloc[va_idx])
    oof_residuals[va_idx] = y.iloc[va_idx].values - preds

valid_mask = ~np.isnan(oof_residuals)
residual_std = float(np.std(oof_residuals[valid_mask]))
print(f"OOF residual std: {residual_std:.6f}")
print(f"This will be used as noise_std for Monte Carlo simulation")"""),
        md("## 9. Full Simulation — Delta + Monte Carlo (2024 Test Races)"),
        code("""\
from f1_predictor.simulation.delta_simulator import DeltaRaceSimulator, MonteCarloSimulator
from f1_predictor.simulation.defaults import build_circuit_defaults
from f1_predictor.simulation.evaluation import evaluate_simulation, evaluate_monte_carlo_calibration
from f1_predictor.features.race_features import LOCATION_ALIASES

circuit_defaults = build_circuit_defaults(laps)

# Retrain best model on full training set
best_model = MODEL_CLASSES_H[best_model_name](**best_params)
best_model.fit(X_train_full, y_train_full)

# Create delta simulator
delta_sim = DeltaRaceSimulator(best_model, circuit_defaults, field_medians)
mc_sim = MonteCarloSimulator(delta_sim, n_simulations=200, noise_std=residual_std, seed=42)

print(f"Simulator ready: {best_model_name}")
print(f"MC noise_std: {residual_std:.6f}, N=200 runs")"""),
        code("""\
test_races = races[races["season"] == 2024].copy()
test_race_list = test_races.groupby(["season", "round", "event_name"]).first().reset_index()

# Single-run delta simulation
sim_results = []
# MC simulation
mc_results_all = []

for _, race_row in test_race_list.iterrows():
    event = race_row["event_name"]
    event_norm = LOCATION_ALIASES.get(event, event)

    if event_norm not in circuit_defaults:
        print(f"  Skipping {event} (no circuit data)")
        continue

    race_drivers = test_races[
        (test_races["season"] == race_row["season"])
        & (test_races["round"] == race_row["round"])
    ].copy()

    drivers_input = []
    actual_positions = {}
    for _, drv in race_drivers.iterrows():
        q1 = drv.get("q1_time_sec")
        q2 = drv.get("q2_time_sec")
        q3 = drv.get("q3_time_sec")
        q_times = [t for t in [q1, q2, q3] if pd.notna(t)]
        if not q_times or pd.isna(drv.get("grid_position")):
            continue
        drivers_input.append({
            "driver": drv["driver_abbrev"],
            "grid_position": int(drv["grid_position"]),
            "q1": q1 if pd.notna(q1) else None,
            "q2": q2 if pd.notna(q2) else None,
            "q3": q3 if pd.notna(q3) else None,
            "initial_tyre": "MEDIUM",
        })
        if pd.notna(drv.get("finish_position")):
            actual_positions[drv["driver_abbrev"]] = int(drv["finish_position"])

    if len(drivers_input) < 10:
        continue

    try:
        # Single-run delta
        result = delta_sim.simulate(event_norm, drivers_input)
        for fr in result.final_results:
            if fr["driver"] in actual_positions:
                sim_results.append({
                    "event": event, "driver": fr["driver"],
                    "predicted_pos": fr["position"],
                    "actual_pos": actual_positions[fr["driver"]],
                })

        # Monte Carlo
        mc_result = mc_sim.simulate(event_norm, drivers_input)
        for r in mc_result.results:
            if r["driver"] in actual_positions:
                mc_results_all.append({
                    **r, "event": event,
                    "actual_pos": actual_positions[r["driver"]],
                    "predicted_pos": r["position"],
                })
        print(f"  {event}: simulated {len(drivers_input)} drivers")
    except Exception as e:
        print(f"  {event}: failed — {e}")

sim_df = pd.DataFrame(sim_results)
mc_df = pd.DataFrame(mc_results_all)
print(f"\\nSingle-run results: {len(sim_df)} driver-race predictions")
print(f"MC results: {len(mc_df)} driver-race predictions")"""),
        code("""\
from f1_predictor.simulation.evaluation import evaluate_simulation, evaluate_monte_carlo_calibration

# Single-run delta metrics
if len(sim_df) > 0:
    delta_metrics = evaluate_simulation(sim_df)
    print("=" * 60)
    print("MODEL H — SINGLE-RUN DELTA SIMULATION (2024)")
    print("=" * 60)
    for k, v in delta_metrics.items():
        print(f"  {k:20s}: {v}")

# Monte Carlo metrics
if len(mc_df) > 0:
    mc_sim_metrics = evaluate_simulation(mc_df)
    mc_cal_metrics = evaluate_monte_carlo_calibration(mc_df.to_dict("records"))
    print()
    print("=" * 60)
    print("MODEL H — MONTE CARLO SIMULATION (N=200, 2024)")
    print("=" * 60)
    for k, v in mc_sim_metrics.items():
        print(f"  {k:20s}: {v}")
    print()
    print("Monte Carlo Calibration:")
    for k, v in mc_cal_metrics.items():
        print(f"  {k:20s}: {v}")"""),
        md("## 10. Save Artifacts"),
        code("""\
for name in top5_names:
    params = reconstruct_params_h(name, r3_best_params[name])
    model = MODEL_CLASSES_H[name](**params)
    model.fit(X_train_full, y_train_full)

    save_predictions(model, X_train_full, y_train_full, id_train, "H", name, "Training")
    save_predictions(model, X_test, y_test, id_test, "H", name, "Test")

    oof_preds = np.full(len(X), np.nan)
    for tr_idx, va_idx in splitter.split(groups):
        import sklearn.base
        fold_model = sklearn.base.clone(model)
        fold_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof_preds[va_idx] = fold_model.predict(X.iloc[va_idx])

    val_mask = ~np.isnan(oof_preds)
    val_out = df[ID_COLS].loc[val_mask].copy()
    val_out["y_true"] = y.loc[val_mask].values
    val_out["y_pred"] = oof_preds[val_mask]
    fname = f"model_H_{name}_Validation.parquet"
    uri = save_training_parquet(val_out, fname, TRAINING_DIR)
    print(f"  Saved {fname} -> {uri}")

    save_model_pkl(model, "H", name)

print("\\nDone! All Model H artifacts saved.")"""),
        md("## Summary"),
        code("""\
print("=" * 60)
print("MODEL H (DELTA + MONTE CARLO) TRAINING COMPLETE")
print("=" * 60)
print(f"\\nPer-lap evaluation (top 5, sorted by test RMSE):")
for _, row in final_df.iterrows():
    print(f"  {row['model']:30s}  test_rmse={row['test_rmse']:.6f}  gap={row['overfit_gap']:.6f}")

if len(sim_df) > 0:
    print(f"\\nSingle-run delta simulation (2024):")
    print(f"  Position RMSE: {delta_metrics['position_rmse']:.4f}")
    print(f"  Spearman:      {delta_metrics['spearman_mean']:.4f}")
    print(f"  Within-3:      {delta_metrics['within_3']:.1f}%")

if len(mc_df) > 0:
    print(f"\\nMonte Carlo simulation (N=200, 2024):")
    print(f"  Position RMSE: {mc_sim_metrics['position_rmse']:.4f}")
    print(f"  Spearman:      {mc_sim_metrics['spearman_mean']:.4f}")
    print(f"  Within-3:      {mc_sim_metrics['within_3']:.1f}%")
    print(f"  Coverage@80%:  {mc_cal_metrics['coverage_80']:.1f}%")

print(f"\\nArtifacts saved to:")
print(f"  Predictions: {TRAINING_DIR}")
print(f"  Models: {MODEL_DIR}")

best_name = final_df.iloc[0]["model"]
best_rmse = final_df.iloc[0]["test_rmse"]
progress.log(f"MODEL H TRAINING COMPLETE -- best: {best_name} (RMSE: {best_rmse:.6f})")
slack.model_complete(best_name, best_rmse)"""),
    ]
    return cells


# ---------------------------------------------------------------------------
# Model I notebook: Uncertainty / Quantile Models
# ---------------------------------------------------------------------------


def make_model_i() -> list[dict]:
    cells = [
        md(
            "# 05i — Model I Training: Uncertainty-Aware / Quantile Models\n\n"
            "Predicts the **distribution** of `lap_time_ratio` via quantile regression,\n"
            "mixture density networks, or deep ensembles.\n\n"
            "8 GPU candidates: 2 tree-quantile + 3 DL-quantile + 2 MDN + 1 ensemble.\n"
            "Quantile Monte Carlo: sample from predicted distribution for principled MC.\n\n"
            "CV: ExpandingWindowSplit (2019→2020, ..., 2019-23→2024 test)."
        ),
        md("## 0. Setup"),
        code(CHDIR),
        code(IMPORTS),
        code(SAVE_ARTIFACTS),
        code(PROGRESS_LOGGER.format(model_key="I")),
        code(SLACK_NOTIFIER.format(model_key="I")),
        code(CHECKPOINT_MANAGER.format(model_key="I")),
        code("slack.model_start()\nprogress.log('Starting Model I training')"),
        md("## 1. Build Training Data"),
        code("""\
from f1_predictor.features.simulation_features import (
    build_simulation_training_data,
    SIMULATION_FEATURE_COLS,
)

laps = load_from_gcs_or_local(
    "data/raw/laps/all_laps.parquet",
    Path("data/raw/laps/all_laps.parquet"),
)
races = load_from_gcs_or_local(
    "data/raw/race/all_races.parquet",
    Path("data/raw/race/all_races.parquet"),
)

df = build_simulation_training_data(laps, races)
print(f"Shape: {df.shape}")
print(f"Target stats:\\n{df['lap_time_ratio'].describe()}")"""),
        code("""\
FEATURE_COLS = SIMULATION_FEATURE_COLS
TARGET = "lap_time_ratio"
ID_COLS = ["season", "round", "event_name", "driver_abbrev", "team"]

df = df.dropna(subset=[TARGET]).reset_index(drop=True)

X = df[FEATURE_COLS]
y = df[TARGET]
groups = df["season"].values
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"X shape: {X.shape}, y shape: {y.shape}")"""),
        md("## 2. CV Splitter"),
        code("""\
splitter = ExpandingWindowSplit(
    fold_definitions=[
        ([2019], 2020),
        ([2019, 2020], 2021),
        ([2019, 2020, 2021], 2022),
        ([2019, 2020, 2021, 2022], 2023),
    ],
    test_season=2024,
)
print(f"CV folds: {splitter.get_n_splits()}")
for i, (tr, va) in enumerate(splitter.split(groups)):
    tr_seasons = sorted(set(groups[tr]))
    va_seasons = sorted(set(groups[va]))
    print(f"  Fold {i}: train={tr_seasons}, val={va_seasons}, "
          f"train={len(tr):,}, val={len(va):,}")"""),
        md(
            "## 3. Model Candidates and Helpers\n\n"
            "8 uncertainty-aware candidates, all Optuna-tunable."
        ),
        code("""\
from f1_predictor.models.quantile_architectures import (
    LightGBM_Quantile, XGBoost_Quantile,
    MLP_MultiQuantile, GRU_MultiQuantile, FTTransformer_Quantile,
    MDN_MLP, MDN_GRU, DeepEnsemble,
)

DL_SKIP_OPTUNA = set()  # all models tunable
NAN_TOLERANT = {"LightGBM_Quantile", "XGBoost_Quantile"}

MODEL_CLASSES_I = {
    "LightGBM_Quantile": LightGBM_Quantile,
    "XGBoost_Quantile": XGBoost_Quantile,
}
if DL_AVAILABLE:
    MODEL_CLASSES_I.update({
        "MLP_MultiQuantile": MLP_MultiQuantile,
        "GRU_MultiQuantile": GRU_MultiQuantile,
        "FTTransformer_Quantile": FTTransformer_Quantile,
        "MDN_MLP": MDN_MLP,
        "MDN_GRU": MDN_GRU,
        "DeepEnsemble": DeepEnsemble,
    })


def get_candidates_i():
    xgb_device = get_xgboost_device(GPU_BACKEND)
    lgb_device = get_lightgbm_device(GPU_BACKEND)
    candidates = {
        "LightGBM_Quantile": LightGBM_Quantile(
            n_estimators=300, n_jobs=-1, random_state=42, verbose=-1, **lgb_device),
        "XGBoost_Quantile": XGBoost_Quantile(
            n_estimators=300, n_jobs=-1, random_state=42, verbosity=0, **xgb_device),
    }
    if DL_AVAILABLE:
        n_feat = len(FEATURE_COLS)
        candidates["MLP_MultiQuantile"] = MLP_MultiQuantile(input_dim=n_feat)
        candidates["GRU_MultiQuantile"] = GRU_MultiQuantile(input_dim=n_feat)
        candidates["FTTransformer_Quantile"] = FTTransformer_Quantile(input_dim=n_feat)
        candidates["MDN_MLP"] = MDN_MLP(input_dim=n_feat)
        candidates["MDN_GRU"] = MDN_GRU(input_dim=n_feat)
        candidates["DeepEnsemble"] = DeepEnsemble(input_dim=n_feat)
    print(f"Candidates ({len(candidates)}): {list(candidates.keys())}")
    return candidates


def _tree_quantile_space(trial, tree_type):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1500),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )
    if tree_type == "xgb":
        xgb_device = get_xgboost_device(GPU_BACKEND)
        params.update(n_jobs=-1, random_state=42, verbosity=0, **xgb_device)
    else:
        lgb_device = get_lightgbm_device(GPU_BACKEND)
        params.update(n_jobs=-1, random_state=42, verbose=-1, **lgb_device)
    return params


def _dl_quantile_space(trial, model_cls):
    params = {"input_dim": len(FEATURE_COLS)}
    if model_cls in (MLP_MultiQuantile, MDN_MLP, DeepEnsemble):
        params["hidden1"] = trial.suggest_categorical("hidden1", [64, 128, 256])
        params["hidden2"] = trial.suggest_categorical("hidden2", [32, 64, 128])
    if model_cls in (GRU_MultiQuantile, MDN_GRU):
        params["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
        params["num_layers"] = trial.suggest_int("num_layers", 1, 3)
    if model_cls == FTTransformer_Quantile:
        params["d_token"] = trial.suggest_categorical("d_token", [32, 64, 128])
        params["n_blocks"] = trial.suggest_int("n_blocks", 1, 4)
    params["dropout"] = trial.suggest_float("dropout", 0.05, 0.5)
    params["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    params["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    if model_cls in (MDN_MLP, MDN_GRU):
        params["n_components"] = trial.suggest_int("n_components", 2, 5)
    if model_cls == DeepEnsemble:
        params["n_members"] = trial.suggest_int("n_members", 3, 7)
    return params


def get_optuna_param_space_i(name, trial):
    if name == "LightGBM_Quantile":
        return _tree_quantile_space(trial, "lgb")
    elif name == "XGBoost_Quantile":
        return _tree_quantile_space(trial, "xgb")
    else:
        return _dl_quantile_space(trial, MODEL_CLASSES_I[name])


def reconstruct_params_i(name, best_params):
    params = dict(best_params)
    if name == "XGBoost_Quantile":
        xgb_device = get_xgboost_device(GPU_BACKEND)
        params.update(n_jobs=-1, random_state=42, verbosity=0, **xgb_device)
    elif name == "LightGBM_Quantile":
        lgb_device = get_lightgbm_device(GPU_BACKEND)
        params.update(n_jobs=-1, random_state=42, verbose=-1, **lgb_device)
    else:
        params.setdefault("input_dim", len(FEATURE_COLS))
    return params"""),
        code("""\
def cv_evaluate_i(model, X, y, splitter, groups):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmses, maes = [], []
    for tr_idx, va_idx in splitter.split(groups):
        import sklearn.base
        m = sklearn.base.clone(model)
        X_tr = X.iloc[tr_idx] if hasattr(X, "iloc") else X[tr_idx]
        y_tr = y.iloc[tr_idx] if hasattr(y, "iloc") else y[tr_idx]
        X_va = X.iloc[va_idx] if hasattr(X, "iloc") else X[va_idx]
        y_va = y.iloc[va_idx] if hasattr(y, "iloc") else y[va_idx]
        name_str = type(model).__name__
        if name_str in str(NAN_TOLERANT) or any(n in str(type(model)) for n in ["XGB", "LGB"]):
            m.fit(X_tr, y_tr)
        else:
            X_tr_clean = X_tr.fillna(0) if hasattr(X_tr, "fillna") else np.nan_to_num(X_tr)
            X_va_clean = X_va.fillna(0) if hasattr(X_va, "fillna") else np.nan_to_num(X_va)
            m.fit(X_tr_clean, y_tr)
            X_va = X_va_clean
        # Use median prediction (q50) for RMSE
        preds = m.predict(X_va)
        rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
        maes.append(mean_absolute_error(y_va, preds))
    return {"mean_rmse": np.mean(rmses), "std_rmse": np.std(rmses), "mean_mae": np.mean(maes)}


def screen_models_i(candidates, X, y, splitter, groups):
    rows = []
    completed = ckpt.get_completed(1)
    total = len(candidates)
    for idx, (name, model) in enumerate(candidates.items(), 1):
        if name in completed:
            cp = completed[name]
            rows.append({"model": name, "mean_rmse": cp["rmse"],
                         "std_rmse": cp.get("std_rmse", 0), "mean_mae": cp.get("mean_mae", 0)})
            progress.log(f"Screening {idx}/{total} {name} -- RESUMED (RMSE: {cp['rmse']:.6f})")
            continue
        try:
            result = cv_evaluate_i(model, X, y, splitter, groups)
            rows.append({"model": name, **result})
            progress.screening(name, idx, total, rmse=result["mean_rmse"])
            ckpt.save_checkpoint(1, name, result["mean_rmse"], {},
                                 std_rmse=result["std_rmse"], mean_mae=result["mean_mae"])
        except Exception as e:
            progress.screening(name, idx, total, error=str(e))
            slack.error(f"R1 screening {name}", str(e))
    if not rows:
        raise RuntimeError("All candidates failed — check data for NaN or shape issues")
    return pd.DataFrame(rows).sort_values("mean_rmse").reset_index(drop=True)


def run_optuna_round_i(name, X, y, splitter, groups, n_trials):
    def objective(trial):
        params = get_optuna_param_space_i(name, trial)
        model_cls = MODEL_CLASSES_I[name]
        model = model_cls(**params)
        result = cv_evaluate_i(model, X, y, splitter, groups)
        return result["mean_rmse"]
    def _trial_cb(study, trial):
        if trial.value is not None:
            progress.optuna_trial(name, trial.number + 1, n_trials, trial.value, study.best_value)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        objective, n_trials=n_trials,
        catch=(Exception,), show_progress_bar=False,
        callbacks=[_trial_cb],
    )
    return study.best_params, study.best_value"""),
        md("## 4. Round 1 — Screen Models"),
        code("""\
progress.round_header(1, "Screen all candidates")
slack.round_start(1, "Screening all candidates", len(get_candidates_i()))
candidates = get_candidates_i()
r1_results = screen_models_i(candidates, X, y, splitter, groups)
ckpt.save_round_summary(1, r1_results.to_dict("records"), r1_results["model"].head(min(7, len(r1_results))).tolist())
slack.round_complete(1, f"Best: {r1_results.iloc[0]['model']} (RMSE: {r1_results.iloc[0]['mean_rmse']:.6f})")
r1_results[["model", "mean_rmse", "std_rmse", "mean_mae"]]"""),
        code("""\
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(r1_results["model"], r1_results["mean_rmse"], xerr=r1_results["std_rmse"])
ax.set_xlabel("Mean RMSE (CV)")
ax.set_title("Round 1: Model Screening — Model I (Quantile/Uncertainty)")
ax.invert_yaxis()
plt.tight_layout()
plt.show()"""),
        code("""\
r1_summary = ckpt.load_round_summary(1)
if r1_summary:
    top7_names = r1_summary["top_names"]
else:
    top7_names = r1_results["model"].head(min(7, len(r1_results))).tolist()
print(f"Advancing to Round 2: {top7_names}")
eliminated = [n for n in r1_results["model"].tolist() if n not in top7_names]
if eliminated:
    print(f"Eliminated: {eliminated}")"""),
        md("## 5. Round 2 — Optuna (top 7, 10 trials each)"),
        code("""\
progress.round_header(2, "Optuna HP tuning (top 7, 10 trials each)")
slack.round_start(2, "Optuna HP tuning", len(top7_names))
r2_results = []
completed_r2 = ckpt.get_completed(2)
for idx, name in enumerate(top7_names, 1):
    if name in completed_r2:
        cp = completed_r2[name]
        r2_results.append({"model": name, "best_rmse": cp["rmse"], "best_params": cp["best_params"]})
        progress.log(f"{name} -- RESUMED from checkpoint (RMSE: {cp['rmse']:.6f})")
        continue
    progress.log(f"Tuning {name} ({idx}/{len(top7_names)})...")
    try:
        best_params, best_rmse = run_optuna_round_i(
            name, X, y, splitter, groups, n_trials=10)
        r2_results.append({"model": name, "best_rmse": best_rmse, "best_params": best_params})
        ckpt.save_checkpoint(2, name, best_rmse, best_params)
        progress.model_complete(name, 2, best_rmse)
        slack.architecture_done(name, 2, best_rmse)
    except Exception as e:
        progress.log(f"{name} FAILED in Round 2: {e}")
        slack.error(f"R2 {name}", str(e))

r2_df = pd.DataFrame(r2_results).sort_values("best_rmse").reset_index(drop=True)
ckpt.save_round_summary(2, r2_results, r2_df["model"].head(5).tolist())
slack.round_complete(2, f"Top 5: {r2_df['model'].head(5).tolist()}")
r2_df[["model", "best_rmse"]]"""),
        code("""\
r2_summary = ckpt.load_round_summary(2)
if r2_summary:
    top5_names = r2_summary["top_names"]
else:
    top5_names = r2_df["model"].head(5).tolist()
r2_best_params = {row["model"]: row["best_params"] for _, row in r2_df.iterrows() if "best_params" in row}
print(f"Advancing to Round 3: {top5_names}")"""),
        md("## 6. Round 3 — Final Tuning (top 5, 15 trials each)"),
        code("""\
progress.round_header(3, "Final tuning (top 5, 15 trials each)")
slack.round_start(3, "Final Optuna tuning", len(top5_names))
r3_results = []
completed_r3 = ckpt.get_completed(3)
for idx, name in enumerate(top5_names, 1):
    if name in completed_r3:
        cp = completed_r3[name]
        r3_results.append({"model": name, "best_rmse": cp["rmse"], "best_params": cp["best_params"]})
        progress.log(f"{name} -- RESUMED from checkpoint (RMSE: {cp['rmse']:.6f})")
        continue
    progress.log(f"Fine-tuning {name} ({idx}/{len(top5_names)})...")
    try:
        best_params, best_rmse = run_optuna_round_i(
            name, X, y, splitter, groups, n_trials=15)
        r3_results.append({"model": name, "best_rmse": best_rmse, "best_params": best_params})
        ckpt.save_checkpoint(3, name, best_rmse, best_params)
        progress.model_complete(name, 3, best_rmse)
        slack.architecture_done(name, 3, best_rmse)
    except Exception as e:
        progress.log(f"{name} FAILED in Round 3: {e}")
        slack.error(f"R3 {name}", str(e))

r3_df = pd.DataFrame(r3_results).sort_values("best_rmse").reset_index(drop=True)
r3_best_params = {row["model"]: row["best_params"] for _, row in r3_df.iterrows() if "best_params" in row}
ckpt.save_round_summary(3, r3_results, r3_df["model"].head(5).tolist())
slack.round_complete(3, f"Best: {r3_df.iloc[0]['model']} (RMSE: {r3_df.iloc[0]['best_rmse']:.6f})")
r3_df[["model", "best_rmse"]]"""),
        md("## 7. Test Set Evaluation (Per-Lap)"),
        code("""\
train_idx, test_idx = splitter.get_test_split(groups)
X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train_full, y_test = y.iloc[train_idx], y.iloc[test_idx]
id_train, id_test = df[ID_COLS].iloc[train_idx], df[ID_COLS].iloc[test_idx]

print(f"Train: {X_train_full.shape}, Test: {X_test.shape}")
print(f"Test season(s): {sorted(df['season'].iloc[test_idx].unique())}")"""),
        code("""\
final_results = []
for name in top5_names:
    params = reconstruct_params_i(name, r3_best_params[name])
    model_cls = MODEL_CLASSES_I[name]
    model = model_cls(**params)
    model.fit(X_train_full, y_train_full)

    train_preds = model.predict(X_train_full)
    train_rmse = np.sqrt(mean_squared_error(y_train_full, train_preds))

    test_preds = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

    val_rmse = r3_df.loc[r3_df["model"] == name, "best_rmse"].values[0]

    final_results.append({
        "model": name,
        "train_rmse": train_rmse, "val_rmse": val_rmse,
        "test_rmse": test_rmse, "overfit_gap": test_rmse - val_rmse,
    })
    print(f"{name}: train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}, "
          f"test_rmse={test_rmse:.4f}")

final_df = pd.DataFrame(final_results).sort_values("test_rmse").reset_index(drop=True)
final_df"""),
        md("## 8. Quantile Monte Carlo Simulation (2024 Test Races)"),
        code("""\
from f1_predictor.simulation.quantile_simulator import QuantileRaceSimulator
from f1_predictor.simulation.defaults import build_circuit_defaults
from f1_predictor.simulation.evaluation import evaluate_simulation, evaluate_monte_carlo_calibration
from f1_predictor.features.race_features import LOCATION_ALIASES

circuit_defaults = build_circuit_defaults(laps)

# Retrain best quantile model on full training set
best_model_name = final_df.iloc[0]["model"]
best_params = reconstruct_params_i(best_model_name, r3_best_params[best_model_name])
best_model = MODEL_CLASSES_I[best_model_name](**best_params)
best_model.fit(X_train_full, y_train_full)

qmc_sim = QuantileRaceSimulator(
    best_model, circuit_defaults, n_simulations=200, seed=42)
print(f"Quantile MC simulator: {best_model_name}, N=200 runs")"""),
        code("""\
test_races = races[races["season"] == 2024].copy()
test_race_list = test_races.groupby(
    ["season", "round", "event_name"]).first().reset_index()

# Single-run (q50 only)
from f1_predictor.simulation.engine import RaceSimulator
single_sim = RaceSimulator(best_model, circuit_defaults)

sim_results = []
mc_results_all = []

for _, race_row in test_race_list.iterrows():
    event = race_row["event_name"]
    event_norm = LOCATION_ALIASES.get(event, event)

    if event_norm not in circuit_defaults:
        print(f"  Skipping {event} (no circuit data)")
        continue

    race_drivers = test_races[
        (test_races["season"] == race_row["season"])
        & (test_races["round"] == race_row["round"])
    ].copy()

    drivers_input = []
    actual_positions = {}
    for _, drv in race_drivers.iterrows():
        q1 = drv.get("q1_time_sec")
        q2 = drv.get("q2_time_sec")
        q3 = drv.get("q3_time_sec")
        q_times = [t for t in [q1, q2, q3] if pd.notna(t)]
        if not q_times or pd.isna(drv.get("grid_position")):
            continue
        drivers_input.append({
            "driver": drv["driver_abbrev"],
            "grid_position": int(drv["grid_position"]),
            "q1": q1 if pd.notna(q1) else None,
            "q2": q2 if pd.notna(q2) else None,
            "q3": q3 if pd.notna(q3) else None,
            "initial_tyre": "MEDIUM",
        })
        if pd.notna(drv.get("finish_position")):
            actual_positions[drv["driver_abbrev"]] = int(drv["finish_position"])

    if len(drivers_input) < 10:
        continue

    try:
        # Single-run (median) simulation
        result = single_sim.simulate(event_norm, drivers_input)
        for fr in result.final_results:
            if fr["driver"] in actual_positions:
                sim_results.append({
                    "event": event, "driver": fr["driver"],
                    "predicted_pos": fr["position"],
                    "actual_pos": actual_positions[fr["driver"]],
                })

        # Quantile Monte Carlo
        mc_result = qmc_sim.simulate(event_norm, drivers_input)
        for r in mc_result.results:
            if r["driver"] in actual_positions:
                mc_results_all.append({
                    **r, "event": event,
                    "actual_pos": actual_positions[r["driver"]],
                    "predicted_pos": r["position"],
                })
        print(f"  {event}: simulated {len(drivers_input)} drivers")
    except Exception as e:
        print(f"  {event}: failed — {e}")

sim_df = pd.DataFrame(sim_results)
mc_df = pd.DataFrame(mc_results_all)
print(f"\\nSingle-run results: {len(sim_df)} driver-race predictions")
print(f"Quantile MC results: {len(mc_df)} driver-race predictions")"""),
        code("""\
# Single-run metrics
if len(sim_df) > 0:
    single_metrics = evaluate_simulation(sim_df)
    print("=" * 60)
    print("MODEL I — SINGLE-RUN (MEDIAN) SIMULATION (2024)")
    print("=" * 60)
    for k, v in single_metrics.items():
        print(f"  {k:20s}: {v}")

# Quantile MC metrics
if len(mc_df) > 0:
    mc_sim_metrics = evaluate_simulation(mc_df)
    mc_cal_metrics = evaluate_monte_carlo_calibration(mc_df.to_dict("records"))
    print()
    print("=" * 60)
    print("MODEL I — QUANTILE MC SIMULATION (N=200, 2024)")
    print("=" * 60)
    for k, v in mc_sim_metrics.items():
        print(f"  {k:20s}: {v}")
    print()
    print("Calibration:")
    for k, v in mc_cal_metrics.items():
        print(f"  {k:20s}: {v}")"""),
        md("## 9. Save Artifacts"),
        code("""\
for name in top5_names:
    params = reconstruct_params_i(name, r3_best_params[name])
    model = MODEL_CLASSES_I[name](**params)
    model.fit(X_train_full, y_train_full)

    save_predictions(model, X_train_full, y_train_full, id_train, "I", name, "Training")
    save_predictions(model, X_test, y_test, id_test, "I", name, "Test")

    oof_preds = np.full(len(X), np.nan)
    for tr_idx, va_idx in splitter.split(groups):
        import sklearn.base
        fold_model = sklearn.base.clone(model)
        fold_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof_preds[va_idx] = fold_model.predict(X.iloc[va_idx])

    val_mask = ~np.isnan(oof_preds)
    val_out = df[ID_COLS].loc[val_mask].copy()
    val_out["y_true"] = y.loc[val_mask].values
    val_out["y_pred"] = oof_preds[val_mask]
    fname = f"model_I_{name}_Validation.parquet"
    uri = save_training_parquet(val_out, fname, TRAINING_DIR)
    print(f"  Saved {fname} -> {uri}")

    save_model_pkl(model, "I", name)

print("\\nDone! All Model I artifacts saved.")"""),
        md("## Summary"),
        code("""\
print("=" * 60)
print("MODEL I (UNCERTAINTY / QUANTILE) TRAINING COMPLETE")
print("=" * 60)
print(f"\\nPer-lap evaluation (top 5, sorted by test RMSE):")
for _, row in final_df.iterrows():
    print(f"  {row['model']:30s}  test_rmse={row['test_rmse']:.6f}  gap={row['overfit_gap']:.6f}")

if len(sim_df) > 0:
    print(f"\\nSingle-run (median) simulation (2024):")
    print(f"  Position RMSE: {single_metrics['position_rmse']:.4f}")
    print(f"  Spearman:      {single_metrics['spearman_mean']:.4f}")
    print(f"  Within-3:      {single_metrics['within_3']:.1f}%")

if len(mc_df) > 0:
    print(f"\\nQuantile MC simulation (N=200, 2024):")
    print(f"  Position RMSE: {mc_sim_metrics['position_rmse']:.4f}")
    print(f"  Spearman:      {mc_sim_metrics['spearman_mean']:.4f}")
    print(f"  Within-3:      {mc_sim_metrics['within_3']:.1f}%")
    print(f"  Coverage@80%:  {mc_cal_metrics['coverage_80']:.1f}%")

print(f"\\nArtifacts saved to:")
print(f"  Predictions: {TRAINING_DIR}")
print(f"  Models: {MODEL_DIR}")

best_name = final_df.iloc[0]["model"]
best_rmse = final_df.iloc[0]["test_rmse"]
progress.log(f"MODEL I TRAINING COMPLETE -- best: {best_name} (RMSE: {best_rmse:.6f})")
slack.model_complete(best_name, best_rmse)"""),
    ]
    return cells


# ---------------------------------------------------------------------------
# Notebook 06: Model Comparison
# ---------------------------------------------------------------------------


def make_model_comparison() -> list[dict]:
    cells = [
        md(
            "# 06 — Model Comparison: All Models × All Sets × 10 Criteria\n\n"
            "Comprehensive evaluation of every trained model variant across "
            "Training, Validation, and Test sets using 10 metrics:\n\n"
            "| # | Criterion | What it measures |\n"
            "|---|-----------|------------------|\n"
            "| 1 | RMSE | Penalises large errors |\n"
            "| 2 | MAE | Average absolute error |\n"
            "| 3 | R² | Proportion of variance explained |\n"
            "| 4 | Median AE | Robust central error (ignores outliers) |\n"
            "| 5 | Max Error | Worst single prediction |\n"
            "| 6 | MAPE | Percentage-based error |\n"
            "| 7 | Within-1 | % predictions within 1 position |\n"
            "| 8 | Within-3 | % predictions within 3 positions |\n"
            "| 9 | Spearman ρ | Per-race ranking correlation |\n"
            "| 10 | Kendall τ | Per-race pairwise ordering agreement |"
        ),
        md("## 0. Setup"),
        code(CHDIR),
        code("""\
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    max_error,
    r2_score,
)

from f1_predictor.data.storage import sync_training_from_gcs

warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.rcParams.update({"figure.dpi": 110, "figure.facecolor": "white"})

TRAINING_DIR = Path("data/training")
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

# Sync all model artifacts from GCS
for mt in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
    sync_training_from_gcs(mt, TRAINING_DIR)
print("Synced all training artifacts from GCS.")"""),
        md("## 1. Load All Prediction Parquets"),
        code("""\
records = []
for pq in sorted(TRAINING_DIR.glob("model_*_*.parquet")):
    stem = pq.stem
    # Parse: model_{type}_{variant}_{split}
    parts = stem.split("_", 2)  # ['model', type, rest]
    model_type = parts[1]
    rest = parts[2]
    split_name = rest.rsplit("_", 1)[-1]  # Training / Validation / Test
    variant = rest.rsplit("_", 1)[0]
    records.append({
        "model_type": model_type, "variant": variant,
        "split": split_name, "path": str(pq),
    })

inventory = pd.DataFrame(records)
print(f"Found {len(inventory)} parquet files")
print(inventory.groupby(["model_type", "split"]).size().unstack(fill_value=0))"""),
        md("## 2. Evaluation Functions"),
        code("""\
def compute_10_criteria(y_true, y_pred):
    \"\"\"Compute all 10 evaluation criteria for a set of predictions.\"\"\"
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
        "Median_AE": median_absolute_error(y_true, y_pred),
        "Max_Error": max_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "Within_1": np.mean(np.abs(y_true - y_pred) <= 1.0) * 100,
        "Within_3": np.mean(np.abs(y_true - y_pred) <= 3.0) * 100,
    }


def race_rank_correlations(df, y_true_col="y_true", y_pred_col="y_pred"):
    \"\"\"Compute per-race Spearman and Kendall correlations, then average.\"\"\"
    group_cols = ["season", "round"]
    available = [c for c in group_cols if c in df.columns]
    if not available:
        return {"Spearman_ρ": np.nan, "Kendall_τ": np.nan}
    spear, kend = [], []
    for _, grp in df.groupby(available):
        if len(grp) < 3:
            continue
        yt, yp = grp[y_true_col].values, grp[y_pred_col].values
        if np.std(yt) == 0 or np.std(yp) == 0:
            continue
        spear.append(spearmanr(yt, yp).statistic)
        kend.append(kendalltau(yt, yp).statistic)
    return {
        "Spearman_ρ": np.mean(spear) if spear else np.nan,
        "Kendall_τ": np.mean(kend) if kend else np.nan,
    }


def aggregate_laps_to_race(df):
    \"\"\"For lap-level models (A, B): take last lap per driver-race.\"\"\"
    if "lap_number" not in df.columns:
        return df
    df = df.sort_values(["season", "round", "driver_abbrev", "lap_number"])
    return df.groupby(["season", "round", "driver_abbrev"]).tail(1).copy()


CRITERIA_COLS = [
    "RMSE", "MAE", "R²", "Median_AE", "Max_Error",
    "MAPE", "Within_1", "Within_3", "Spearman_ρ", "Kendall_τ",
]"""),
        md("## 3. Evaluate Every Model × Variant × Split"),
        code("""\
all_results = []

for _, row in inventory.iterrows():
    df = pd.read_parquet(row["path"])
    # Aggregate lap-level models to race level
    df_race = aggregate_laps_to_race(df)

    y_true = df_race["y_true"].values
    y_pred = df_race["y_pred"].values

    metrics = compute_10_criteria(y_true, y_pred)
    rank_metrics = race_rank_correlations(df_race)
    metrics.update(rank_metrics)

    all_results.append({
        "Model": row["model_type"],
        "Variant": row["variant"],
        "Split": row["split"],
        "N": len(df_race),
        **metrics,
    })

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(["Model", "Variant", "Split"])
print(f"Evaluated {len(results_df)} model-variant-split combinations")
results_df.head(10)"""),
        md("## 4. Full Results Table\n\nEvery model × variant × split with all 10 criteria."),
        code("""\
display_cols = ["Model", "Variant", "Split", "N"] + CRITERIA_COLS

fmt = {
    "RMSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.4f}",
    "Median_AE": "{:.4f}", "Max_Error": "{:.2f}",
    "MAPE": "{:.1f}%", "Within_1": "{:.1f}%", "Within_3": "{:.1f}%",
    "Spearman_ρ": "{:.4f}", "Kendall_τ": "{:.4f}",
}

styled = (
    results_df[display_cols]
    .style.format({k: v.replace("%", "") for k, v in fmt.items()})
    .background_gradient(subset=["RMSE", "MAE", "Max_Error", "MAPE"], cmap="RdYlGn_r")
    .background_gradient(
        subset=["R²", "Within_1", "Within_3", "Spearman_ρ", "Kendall_τ"],
        cmap="RdYlGn")
)
styled"""),
        md(
            "## 5. Best Variant per Model (Test Set)\n\n"
            "Selecting each model type's best variant by test RMSE."
        ),
        code("""\
test_df = results_df[results_df["Split"] == "Test"].copy()
best_idx = test_df.groupby("Model")["RMSE"].idxmin()
best_test = test_df.loc[best_idx].sort_values("RMSE")

print("Best variant per model type on TEST set:\\n")
best_test[display_cols].to_string(index=False)
best_test[display_cols]"""),
        md(
            "## 6. Train / Val / Test Comparison (Best Variants)\n\n"
            "Check for overfitting: are train metrics much better than test?"
        ),
        code("""\
best_variants = best_test[["Model", "Variant"]].values.tolist()
overfit_rows = []
for model, variant in best_variants:
    for split in ["Training", "Validation", "Test"]:
        mask = (
            (results_df["Model"] == model)
            & (results_df["Variant"] == variant)
            & (results_df["Split"] == split)
        )
        row = results_df[mask]
        if not row.empty:
            overfit_rows.append(row.iloc[0])

overfit_df = pd.DataFrame(overfit_rows)
overfit_df = overfit_df.sort_values(["Model", "Split"])

pivot_rmse = overfit_df.pivot_table(index=["Model", "Variant"], columns="Split", values="RMSE")
pivot_rmse = pivot_rmse.reindex(columns=["Training", "Validation", "Test"])
print("RMSE across splits (best variants):\\n")
print(pivot_rmse.to_string())

if "Training" in pivot_rmse.columns and "Test" in pivot_rmse.columns:
    pivot_rmse["Overfit_Gap"] = pivot_rmse["Test"] - pivot_rmse["Training"]
    print("\\nOverfit gap (Test - Train): positive = underfit, negative = overfit")
    print(pivot_rmse["Overfit_Gap"].to_string())"""),
        code("""\
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

split_order = ["Training", "Validation", "Test"]
color_map = {"Training": "#2196F3", "Validation": "#FF9800", "Test": "#4CAF50"}

for idx, (model, variant) in enumerate(best_variants):
    if idx >= 4:
        break
    ax = axes[idx]
    subset = overfit_df[(overfit_df["Model"] == model) & (overfit_df["Variant"] == variant)]
    valid = [s for s in split_order if s in subset["Split"].values]
    subset = subset.set_index("Split").reindex(valid)

    x = np.arange(len(CRITERIA_COLS))
    for i, split in enumerate(split_order):
        if split not in subset.index:
            continue
        vals = subset.loc[split, CRITERIA_COLS].values.astype(float)
        # Normalise to [0,1] for radar-like grouped bar
        ax.bar(x + i * 0.25, vals, 0.25, label=split, color=color_map[split], alpha=0.85)

    ax.set_xticks(x + 0.25)
    ax.set_xticklabels(CRITERIA_COLS, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Model {model} — {variant}", fontsize=11)
    ax.legend(fontsize=8)

plt.suptitle("All 10 Criteria: Train / Val / Test (Best Variants)", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()"""),
        md(
            "## 7. Head-to-Head: Best Models on Test Set\n\n"
            "Grouped bar chart comparing the 4 model types side-by-side."
        ),
        code("""\
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for i, metric in enumerate(CRITERIA_COLS):
    ax = axes[i // 5, i % 5]
    models = best_test["Model"].values
    vals = best_test[metric].values.astype(float)
    colours = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2"][:len(models)]
    bars = ax.bar(models, vals, color=colours, alpha=0.85)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_title(metric, fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)

    # Mark "good" direction
    lower_better = metric in ("RMSE", "MAE", "Median_AE", "Max_Error", "MAPE")
    best_val = min(vals) if lower_better else max(vals)
    best_idx_arr = np.where(vals == best_val)[0]
    for bi in best_idx_arr:
        bars[bi].set_edgecolor("gold")
        bars[bi].set_linewidth(2.5)

plt.suptitle("Head-to-Head: Best Variant per Model (Test Set) — gold border = best",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.show()"""),
        md(
            "## 8. Per-Model Variant Rankings\n\n"
            "Within each model type, how do the variants compare on the test set?"
        ),
        code("""\
for model_type in sorted(test_df["Model"].unique()):
    subset = test_df[test_df["Model"] == model_type].sort_values("RMSE")
    print(f"\\n{'='*70}")
    print(f"Model {model_type} — Test Set Rankings ({len(subset)} variants)")
    print(f"{'='*70}")
    print(subset[["Variant"] + CRITERIA_COLS].to_string(index=False))"""),
        md(
            "## 9. Ranking Quality Deep Dive\n\n"
            "Spearman ρ and Kendall τ measure how well predicted positions "
            "preserve the true finishing order within each race. Values close to 1.0 "
            "mean the model ranks drivers correctly even if absolute positions are off."
        ),
        code("""\
rank_cols = ["Model", "Variant", "Split", "Spearman_ρ", "Kendall_τ"]
test_df = results_df[results_df["Split"] == "Test"]
rank_df = test_df[rank_cols].sort_values("Spearman_ρ", ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

labels = rank_df.apply(lambda r: f"{r['Model']}/{r['Variant']}", axis=1).values
x = np.arange(len(labels))

ax1.barh(x, rank_df["Spearman_ρ"].values, color="#1976D2", alpha=0.8)
ax1.set_yticks(x)
ax1.set_yticklabels(labels, fontsize=7)
ax1.set_xlabel("Spearman ρ")
ax1.set_title("Spearman Rank Correlation (Test)")
ax1.invert_yaxis()
ax1.axvline(x=0.8, color="green", linestyle="--", alpha=0.5, label="ρ=0.8 (strong)")
ax1.legend(fontsize=8)

ax2.barh(x, rank_df["Kendall_τ"].values, color="#F57C00", alpha=0.8)
ax2.set_yticks(x)
ax2.set_yticklabels(labels, fontsize=7)
ax2.set_xlabel("Kendall τ")
ax2.set_title("Kendall Tau (Test)")
ax2.invert_yaxis()
ax2.axvline(x=0.6, color="green", linestyle="--", alpha=0.5, label="τ=0.6 (strong)")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.show()"""),
        md(
            "## 10. Position Accuracy Breakdown\n\n"
            "What percentage of predictions land exactly right, within 1, 2, or 3 positions?"
        ),
        code("""\
accuracy_rows = []
for _, row in inventory[inventory["split"] == "Test"].iterrows():
    df = pd.read_parquet(row["path"])
    df = aggregate_laps_to_race(df)
    errs = np.abs(df["y_true"].values - df["y_pred"].values)
    accuracy_rows.append({
        "Model": row["model_type"], "Variant": row["variant"],
        "Exact": np.mean(errs < 0.5) * 100,
        "Within_1": np.mean(errs <= 1.0) * 100,
        "Within_2": np.mean(errs <= 2.0) * 100,
        "Within_3": np.mean(errs <= 3.0) * 100,
        "Within_5": np.mean(errs <= 5.0) * 100,
        ">5_off": np.mean(errs > 5.0) * 100,
    })

acc_df = pd.DataFrame(accuracy_rows).sort_values("Within_3", ascending=False)

# Show best variant per model
best_acc = acc_df.groupby("Model").first().reset_index().sort_values("Within_3", ascending=False)
print("Position accuracy (best variant per model, test set):\\n")
print(best_acc.to_string(index=False, float_format="{:.1f}%".format))"""),
        code("""\
fig, ax = plt.subplots(figsize=(10, 5))
thresholds = ["Exact", "Within_1", "Within_2", "Within_3", "Within_5"]
colours = ["#0D47A1", "#1976D2", "#42A5F5", "#90CAF9", "#BBDEFB"]

x = np.arange(len(best_acc))
width = 0.15
for i, (thresh, col) in enumerate(zip(thresholds, colours)):
    ax.bar(x + i * width, best_acc[thresh].values, width, label=thresh, color=col)

ax.set_xticks(x + width * 2)
ax.set_xticklabels([f"Model {r['Model']}\\n{r['Variant']}" for _, r in best_acc.iterrows()],
                   fontsize=9)
ax.set_ylabel("% of predictions")
ax.set_title("Position Accuracy Thresholds (Test Set, Best Variants)")
ax.legend()
ax.set_ylim(0, 105)
plt.tight_layout()
plt.show()"""),
        md("## Summary"),
        code("""\
print("=" * 70)
print("MODEL COMPARISON SUMMARY — 10 CRITERIA")
print("=" * 70)

print("\\nBest variant per model (by test RMSE):")
for _, row in best_test.iterrows():
    print(f"  Model {row['Model']:2s}  {row['Variant']:25s}  "
          f"RMSE={row['RMSE']:.4f}  R²={row['R²']:.4f}  "
          f"Spearman={row['Spearman_ρ']:.4f}  Within-3={row['Within_3']:.1f}%")

overall_best = best_test.iloc[0]
print(f"\\nOverall best: Model {overall_best['Model']} / {overall_best['Variant']}")
print(f"  RMSE={overall_best['RMSE']:.4f}, R²={overall_best['R²']:.4f}, "
      f"Within-3={overall_best['Within_3']:.1f}%, "
      f"Spearman ρ={overall_best['Spearman_ρ']:.4f}")"""),
    ]
    return cells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    notebooks = {
        "05a_model_A_training.ipynb": make_model_a(),
        "05b_model_B_training.ipynb": make_model_b(),
        "05c_model_C_training.ipynb": make_model_c(),
        "05d_model_D_stacking.ipynb": make_model_d(),
        "05e_model_E_rich_stacking.ipynb": make_model_e(),
        "05f_model_F_lap_simulation.ipynb": make_model_f(),
        "05g_model_G_temporal.ipynb": make_model_g(),
        "05h_model_H_delta_mc.ipynb": make_model_h(),
        "05i_model_I_quantile.ipynb": make_model_i(),
        "06_model_comparison.ipynb": make_model_comparison(),
    }
    for name, cells in notebooks.items():
        nb = make_notebook(cells)
        path = NOTEBOOKS_DIR / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Generated {path}")


if __name__ == "__main__":
    main()
