"""Generate training notebooks for Models A, B, C, D."""

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
from f1_predictor.data.storage import load_from_gcs_or_local

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

TRAINING_DIR = Path("data/training")
MODEL_DIR = Path("data/raw/model")
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# GPU detection
try:
    import subprocess as _sp
    GPU_AVAILABLE = _sp.run(["nvidia-smi"], capture_output=True).returncode == 0
except FileNotFoundError:
    GPU_AVAILABLE = False
print(f"GPU available: {GPU_AVAILABLE}")

# cuML GPU models (RAPIDS) — optional, fall back to sklearn if not available
CUML_AVAILABLE = False
try:
    from cuml.ensemble import RandomForestRegressor as cuRF
    from cuml.linear_model import Ridge as cuRidge, Lasso as cuLasso, ElasticNet as cuElasticNet
    CUML_AVAILABLE = True
    print("cuML available: True")
except ImportError:
    print("cuML not available — using XGBoost/LightGBM variants only")"""

HELPERS = '''\
NAN_TOLERANT = {
    "XGBoost", "XGBoost_DART", "XGBoost_Linear",
    "LightGBM", "LightGBM_DART", "LightGBM_GOSS",
}


def get_candidates():
    """Return dict of model_name -> model instance. All GPU-accelerated."""
    xgb_device = {"device": "cuda"} if GPU_AVAILABLE else {}
    lgb_device = {"device": "gpu"} if GPU_AVAILABLE else {}

    candidates = {
        # XGBoost variants — all use device="cuda"
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300, n_jobs=-1, random_state=42, verbosity=0, **xgb_device),
        "XGBoost_DART": xgb.XGBRegressor(
            n_estimators=300, booster="dart", n_jobs=-1, random_state=42, verbosity=0, **xgb_device),
        "XGBoost_Linear": xgb.XGBRegressor(
            n_estimators=300, booster="gblinear", n_jobs=-1, random_state=42, verbosity=0, **xgb_device),
        # LightGBM variants — all use device="gpu"
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=300, n_jobs=-1, random_state=42, verbose=-1, **lgb_device),
        "LightGBM_DART": lgb.LGBMRegressor(
            n_estimators=300, boosting_type="dart", n_jobs=-1, random_state=42, verbose=-1, **lgb_device),
        "LightGBM_GOSS": lgb.LGBMRegressor(
            n_estimators=300, boosting_type="goss", n_jobs=-1, random_state=42, verbose=-1, **lgb_device),
    }

    # cuML GPU models — sklearn-compatible API, runs on CUDA
    if CUML_AVAILABLE:
        candidates["cuML_RF"] = cuRF(n_estimators=300, random_state=42)
        candidates["cuML_Ridge"] = cuRidge()
        candidates["cuML_Lasso"] = cuLasso()
        candidates["cuML_ElasticNet"] = cuElasticNet()
    else:
        # Fallback: more XGBoost/LightGBM variants to keep 10 candidates
        candidates["XGBoost_Conservative"] = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            n_jobs=-1, random_state=42, verbosity=0, **xgb_device)
        candidates["LightGBM_Shallow"] = lgb.LGBMRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            n_jobs=-1, random_state=42, verbose=-1, **lgb_device)
        candidates["XGBoost_Deep"] = xgb.XGBRegressor(
            n_estimators=300, max_depth=10, learning_rate=0.1,
            n_jobs=-1, random_state=42, verbosity=0, **xgb_device)
        candidates["LightGBM_Deep"] = lgb.LGBMRegressor(
            n_estimators=300, max_depth=10, learning_rate=0.1,
            n_jobs=-1, random_state=42, verbose=-1, **lgb_device)

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
    """Return HP dict for a given model name and Optuna trial. All GPU-enabled."""
    xgb_device = {"device": "cuda"} if GPU_AVAILABLE else {}
    lgb_device = {"device": "gpu"} if GPU_AVAILABLE else {}

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
        params.pop("subsample", None)  # GOSS doesn't use subsample
        params.update(boosting_type="goss", n_jobs=-1, random_state=42, verbose=-1, **lgb_device)
        return params
    elif name in ("LightGBM_Shallow", "LightGBM_Deep"):
        params = _lgb_base_space(trial)
        params.update(n_jobs=-1, random_state=42, verbose=-1, **lgb_device)
        return params
    elif name == "cuML_RF":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 1500),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
            random_state=42,
        )
    elif name == "cuML_Ridge":
        return dict(alpha=trial.suggest_float("alpha", 0.001, 100.0, log=True))
    elif name == "cuML_Lasso":
        return dict(alpha=trial.suggest_float("alpha", 0.001, 100.0, log=True))
    elif name == "cuML_ElasticNet":
        return dict(
            alpha=trial.suggest_float("alpha", 0.001, 100.0, log=True),
            l1_ratio=trial.suggest_float("l1_ratio", 0.1, 0.9),
        )
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
if CUML_AVAILABLE:
    MODEL_CLASSES.update({
        "cuML_RF": cuRF,
        "cuML_Ridge": cuRidge,
        "cuML_Lasso": cuLasso,
        "cuML_ElasticNet": cuElasticNet,
    })


def reconstruct_params(name, best_params):
    """Translate flat Optuna best_params back to model constructor args."""
    params = dict(best_params)
    xgb_device = {"device": "cuda"} if GPU_AVAILABLE else {}
    lgb_device = {"device": "gpu"} if GPU_AVAILABLE else {}

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
    elif name == "cuML_RF":
        params["random_state"] = 42
    # cuML_Ridge, cuML_Lasso, cuML_ElasticNet — no extra params needed
    return params


def run_optuna_round(name, X, y, splitter, groups, n_trials):
    """Run Optuna study for a single model. Returns best params and best RMSE."""
    def objective(trial):
        params = get_optuna_param_space(name, trial)
        model_cls = MODEL_CLASSES[name]
        model = model_cls(**params)
        result = cv_evaluate(model, X, y, splitter, groups)
        return result["mean_rmse"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value'''

SAVE_ARTIFACTS = '''\
def save_predictions(model, X, y, id_df, model_type, model_name, split_name):
    """Save prediction parquet for a given split."""
    preds = model.predict(X)
    out = id_df.copy()
    out["y_true"] = y.values
    out["y_pred"] = preds
    fname = f"model_{model_type}_{model_name}_{split_name}.parquet"
    out.to_parquet(TRAINING_DIR / fname, index=False)
    print(f"  Saved {fname}")
    return preds


def save_model_pickle(model, model_type, model_name):
    """Save model pickle."""
    fname = f"Model_{model_type}_{model_name}.pkl"
    with open(MODEL_DIR / fname, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved {fname}")'''


# ---------------------------------------------------------------------------
# Model A notebook
# ---------------------------------------------------------------------------

def make_model_a() -> list[dict]:
    cells = [
        md("# 05a — Model A Training: Lap + Tyre (2019-2024)\n\n"
           "Predicts **lap-level position** using 9 features including tyre data.\n"
           "CV: LeaveOneSeasonOut (test season = 2024)."),
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
        md("## 3. Round 1 — Screen 10 Models (default params)"),
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
eliminated = r1_results["model"].tail(3).tolist()
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
    val_out.to_parquet(TRAINING_DIR / fname, index=False)
    print(f"  Saved {fname}")

    save_model_pickle(model, "A", name)

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
        md("# 05b — Model B Training: Lap, No Tyre (2018-2025)\n\n"
           "Predicts **lap-level position** using 8 features (no tyre data).\n"
           "CV: ExpandingWindowSplit (test season = 2025)."),
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
        md("## 3. Round 1 — Screen 10 Models (default params)"),
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
eliminated = r1_results["model"].tail(3).tolist()
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
    val_out.to_parquet(TRAINING_DIR / fname, index=False)
    print(f"  Saved {fname}")

    save_model_pickle(model, "B", name)

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
        md("# 05c — Model C Training: Pre-Race Features (2018-2025)\n\n"
           "Predicts **race-level finish_position** using 15 pre-race features.\n"
           "CV: ExpandingWindowSplit (test season = 2025)."),
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
        md("## 3. Round 1 — Screen 10 Models (default params)"),
        code("""\
candidates = get_candidates()
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
eliminated = r1_results["model"].tail(3).tolist()
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
    val_out.to_parquet(TRAINING_DIR / fname, index=False)
    print(f"  Saved {fname}")

    save_model_pickle(model, "C", name)

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
        md("# 05d — Model D Training: Stacking Meta-Model\n\n"
           "Combines race-level predictions from Models A, B, C into a single ensemble.\n"
           "Uses out-of-fold (OOF) predictions as meta-features to prevent leakage.\n"
           "CV: LeaveOneSeasonOut (test season = 2023, last season with OOF from all models)."),
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb

from f1_predictor.features.splits import LeaveOneSeasonOut

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

TRAINING_DIR = Path("data/training")
MODEL_DIR = Path("data/raw/model")"""),
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
        md("## 1. Load OOF Predictions from Models A, B, C\n\n"
           "We load the **best** model's validation (OOF) predictions from each.\n"
           "For Models A and B (lap-level), we aggregate to race level using the last lap."),
        code("""\
def find_best_model(model_type):
    \"\"\"Find the best model for a type by reading test parquets and picking lowest RMSE.\"\"\"
    import glob
    test_files = sorted(TRAINING_DIR.glob(f"model_{model_type}_*_Test.parquet"))
    best_name, best_rmse = None, float("inf")
    for f in test_files:
        df_t = pd.read_parquet(f)
        rmse = np.sqrt(mean_squared_error(df_t["y_true"], df_t["y_pred"]))
        name = f.stem.replace(f"model_{model_type}_", "").replace("_Test", "")
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
    print(f"Best Model {model_type}: {best_name} (test RMSE={best_rmse:.4f})")
    return best_name

best_A = find_best_model("A")
best_B = find_best_model("B")
best_C = find_best_model("C")"""),
        code("""\
def aggregate_lap_to_race(val_df):
    \"\"\"Aggregate lap-level predictions to race level using the last lap per driver-race.\"\"\"
    val_df = val_df.sort_values(["season", "round", "driver_abbrev", "lap_number"])
    last_laps = val_df.groupby(["season", "round", "driver_abbrev"]).tail(1)
    return last_laps[["season", "round", "driver_abbrev", "y_true", "y_pred"]].copy()

# Load OOF validation predictions
val_A = pd.read_parquet(TRAINING_DIR / f"model_A_{best_A}_Validation.parquet")
val_B = pd.read_parquet(TRAINING_DIR / f"model_B_{best_B}_Validation.parquet")
val_C = pd.read_parquet(TRAINING_DIR / f"model_C_{best_C}_Validation.parquet")

# Aggregate A and B to race level
race_A = aggregate_lap_to_race(val_A).rename(columns={"y_pred": "pred_A", "y_true": "true_A"})
race_B = aggregate_lap_to_race(val_B).rename(columns={"y_pred": "pred_B", "y_true": "true_B"})
race_C = val_C.rename(columns={"y_pred": "pred_C", "y_true": "true_C"})

print(f"Race-level A: {race_A.shape}")
print(f"Race-level B: {race_B.shape}")
print(f"Race-level C: {race_C.shape}")"""),
        md("## 2. Build Meta-Feature Matrix"),
        code("""\
merge_key = ["season", "round", "driver_abbrev"]
meta = race_C[merge_key + ["pred_C", "true_C"]].copy()

# Merge A predictions (OOF covers 2019-2023)
meta = meta.merge(race_A[merge_key + ["pred_A"]], on=merge_key, how="left")
# Merge B predictions
meta = meta.merge(race_B[merge_key + ["pred_B"]], on=merge_key, how="left")

# Target is the race-level finish position from Model C's ground truth
meta["finish_position"] = meta["true_C"]
meta = meta.dropna(subset=["finish_position"])

print(f"Meta-feature matrix: {meta.shape}")
print(f"Seasons: {sorted(meta['season'].unique())}")
print(f"NaN counts:\\n{meta[['pred_A', 'pred_B', 'pred_C']].isna().sum()}")
meta.head()"""),
        code("""\
FEATURE_COLS = ["pred_A", "pred_B", "pred_C"]
TARGET = "finish_position"
ID_COLS = ["season", "round", "driver_abbrev"]

X = meta[FEATURE_COLS]
y = meta[TARGET]
groups = meta["season"].values
print(f"X: {X.shape}, y: {y.shape}")"""),
        md("## 3. CV Splitter\n\n"
           "Using LeaveOneSeasonOut with test_season=2023 — the latest season with "
           "genuine OOF predictions from all three base models.\n\n"
           "Season 2024 cannot be used: Model A excludes it (its own test season), "
           "and Models B/C include it in training (leakage)."),
        code("""\
# OOF predictions from all models cover 2019-2023
# 2024 is excluded: it's Model A's test season, and in B/C's training set (leakage)
available_seasons = sorted(meta["season"].unique())
val_seasons = [s for s in available_seasons if s != 2023]
splitter = LeaveOneSeasonOut(val_seasons=val_seasons, test_season=2023)
print(f"Val seasons: {val_seasons}")
print(f"CV folds: {splitter.get_n_splits()}")
for i, (tr, va) in enumerate(splitter.split(groups)):
    print(f"  Fold {i}: train={len(tr)}, val={len(va)}")"""),
        md("## 4. Round 1 — Screen 6 Candidates"),
        code("""\
candidates_d = {
    "RidgeCV": wrap_imputer(RidgeCV()),
    "LassoCV": wrap_imputer(LassoCV(random_state=42, max_iter=5000)),
    "ElasticNetCV": wrap_imputer(ElasticNetCV(random_state=42, max_iter=5000)),
    "XGBoost_shallow": xgb.XGBRegressor(
        n_estimators=100, max_depth=3, random_state=42, verbosity=0),
    "LightGBM_shallow": lgb.LGBMRegressor(
        n_estimators=100, max_depth=3, random_state=42, verbose=-1),
    "WeightedAvg": None,  # handled separately
}

# Evaluate non-WeightedAvg candidates
r1_rows = []
for name, model in candidates_d.items():
    if name == "WeightedAvg":
        continue
    print(f"  Screening {name}...")
    result = cv_evaluate(model, X, y, splitter, groups)
    r1_rows.append({"model": name, **result})

# Weighted average via grid search
best_wa_rmse = float("inf")
best_weights = None
for w_a in np.arange(0.0, 1.05, 0.1):
    for w_b in np.arange(0.0, 1.05 - w_a, 0.1):
        w_c = 1.0 - w_a - w_b
        if w_c < 0:
            continue
        preds = w_a * X["pred_A"].fillna(0) + w_b * X["pred_B"].fillna(0) + w_c * X["pred_C"]
        rmse = np.sqrt(mean_squared_error(y, preds))
        if rmse < best_wa_rmse:
            best_wa_rmse = rmse
            best_weights = (round(w_a, 1), round(w_b, 1), round(w_c, 1))

print(f"  WeightedAvg best weights: A={best_weights[0]}, B={best_weights[1]}, C={best_weights[2]}, RMSE={best_wa_rmse:.4f}")
r1_rows.append({"model": "WeightedAvg", "mean_rmse": best_wa_rmse, "std_rmse": 0.0, "mean_mae": 0.0, "fold_rmse": [], "fold_mae": []})

r1_df = pd.DataFrame(r1_rows).sort_values("mean_rmse").reset_index(drop=True)
r1_df[["model", "mean_rmse", "std_rmse"]]"""),
        code("""\
top4_names = r1_df["model"].head(4).tolist()
if "WeightedAvg" in top4_names:
    top4_names.remove("WeightedAvg")
    top4_names = r1_df[r1_df["model"] != "WeightedAvg"]["model"].head(4).tolist()
print(f"Advancing to Round 2 (excluding WeightedAvg): {top4_names}")"""),
        md("## 5. Round 2 — Optuna HP Tuning (top 4, 15 trials each)"),
        code("""\
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

def run_d_optuna(name, X, y, splitter, groups, n_trials):
    def objective(trial):
        params = get_d_param_space(name, trial)
        model = D_MODEL_CLASSES[name](**params)
        if name not in NAN_TOLERANT_D:
            model = wrap_imputer(model)
        return cv_evaluate(model, X, y, splitter, groups)["mean_rmse"]
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value

r2_results = []
for name in top4_names:
    print(f"Tuning {name}...")
    best_params, best_rmse = run_d_optuna(name, X, y, splitter, groups, n_trials=15)
    r2_results.append({"model": name, "best_rmse": best_rmse, "best_params": best_params})
    print(f"  Best RMSE: {best_rmse:.4f}")

r2_df = pd.DataFrame(r2_results).sort_values("best_rmse").reset_index(drop=True)
r2_df[["model", "best_rmse"]]"""),
        code("""\
top3_names = r2_df["model"].head(3).tolist()
print(f"Advancing to Round 3: {top3_names}")"""),
        md("## 6. Round 3 — Final HP Tuning (top 3, 20 trials each)"),
        code("""\
r3_results = []
for name in top3_names:
    print(f"Fine-tuning {name}...")
    best_params, best_rmse = run_d_optuna(name, X, y, splitter, groups, n_trials=20)
    r3_results.append({"model": name, "best_rmse": best_rmse, "best_params": best_params})
    print(f"  Best RMSE: {best_rmse:.4f}")

r3_df = pd.DataFrame(r3_results).sort_values("best_rmse").reset_index(drop=True)
r3_best_params = {row["model"]: row["best_params"] for _, row in r3_df.iterrows()}
r3_df[["model", "best_rmse"]]"""),
        md("## 7. Test Set Evaluation"),
        code("""\
train_idx, test_idx = splitter.get_test_split(groups)
X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train_full, y_test = y.iloc[train_idx], y.iloc[test_idx]
id_train = meta[ID_COLS].iloc[train_idx]
id_test = meta[ID_COLS].iloc[test_idx]

print(f"Train: {X_train_full.shape}, Test: {X_test.shape}")
print(f"Test season(s): {sorted(meta['season'].iloc[test_idx].unique())}")"""),
        code("""\
final_results = []
for name in top3_names:
    params = reconstruct_d_params(name, r3_best_params[name])
    model = D_MODEL_CLASSES[name](**params)
    if name not in NAN_TOLERANT_D:
        model = wrap_imputer(model)

    model.fit(X_train_full, y_train_full)

    train_preds = model.predict(X_train_full)
    train_rmse = np.sqrt(mean_squared_error(y_train_full, train_preds))

    test_preds = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)

    val_rmse = r3_df.loc[r3_df["model"] == name, "best_rmse"].values[0]

    final_results.append({
        "model": name, "train_rmse": train_rmse,
        "val_rmse": val_rmse, "test_rmse": test_rmse, "test_mae": test_mae,
        "overfit_gap": test_rmse - val_rmse,
    })
    print(f"{name}: train={train_rmse:.4f}, val={val_rmse:.4f}, test={test_rmse:.4f}")

final_df = pd.DataFrame(final_results).sort_values("test_rmse").reset_index(drop=True)
final_df"""),
        code("""\
# Also evaluate the weighted average on test set
wa_test_pred = best_weights[0] * X_test["pred_A"].fillna(0) + best_weights[1] * X_test["pred_B"].fillna(0) + best_weights[2] * X_test["pred_C"]
wa_test_rmse = np.sqrt(mean_squared_error(y_test, wa_test_pred))
print(f"WeightedAvg test RMSE: {wa_test_rmse:.4f} (weights: A={best_weights[0]}, B={best_weights[1]}, C={best_weights[2]})")"""),
        code("""\
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(final_df))
w = 0.25
ax.bar(x - w, final_df["train_rmse"], w, label="Train")
ax.bar(x, final_df["val_rmse"], w, label="Val")
ax.bar(x + w, final_df["test_rmse"], w, label="Test")
ax.set_xticks(x)
ax.set_xticklabels(final_df["model"], rotation=30, ha="right")
ax.set_ylabel("RMSE")
ax.set_title("Model D — Stacking Meta-Model: Train / Val / Test RMSE")
ax.legend()
plt.tight_layout()
plt.show()"""),
        md("## 8. Save Artifacts"),
        code("""\
for name in top3_names:
    params = reconstruct_d_params(name, r3_best_params[name])
    model = D_MODEL_CLASSES[name](**params)
    if name not in NAN_TOLERANT_D:
        model = wrap_imputer(model)
    model.fit(X_train_full, y_train_full)

    # Save predictions
    for split_name, X_s, y_s, id_s in [
        ("Training", X_train_full, y_train_full, id_train),
        ("Test", X_test, y_test, id_test),
    ]:
        out = id_s.copy()
        out["y_true"] = y_s.values
        out["y_pred"] = model.predict(X_s)
        fname = f"model_D_{name}_{split_name}.parquet"
        out.to_parquet(TRAINING_DIR / fname, index=False)
        print(f"  Saved {fname}")

    # Save pickle
    pkl_name = f"Model_D_{name}.pkl"
    with open(MODEL_DIR / pkl_name, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved {pkl_name}")

print("\\nDone! All Model D artifacts saved.")"""),
        md("## Summary"),
        code("""\
print("=" * 60)
print("MODEL D (STACKING) TRAINING COMPLETE")
print("=" * 60)
print(f"\\nBest base models: A={best_A}, B={best_B}, C={best_C}")
print(f"Weighted average: A={best_weights[0]}, B={best_weights[1]}, C={best_weights[2]}, RMSE={wa_test_rmse:.4f}")
print(f"\\nFinal meta-models:")
for _, row in final_df.iterrows():
    print(f"  {row['model']:20s}  test_rmse={row['test_rmse']:.4f}  gap={row['overfit_gap']:.4f}")"""),
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
    }
    for name, cells in notebooks.items():
        nb = make_notebook(cells)
        path = NOTEBOOKS_DIR / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Generated {path}")


if __name__ == "__main__":
    main()
