#!/usr/bin/env python
"""Test ensemble simulator: H alone vs E alone vs H+E across 4 test races."""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

MODEL_DIR = Path("data/raw/model")

TEST_EVENTS = [
    "Bahrain Grand Prix",
    "Emilia Romagna Grand Prix",
    "Hungarian Grand Prix",
    "Mexico City Grand Prix",
]

BLEND_VALUES = [5, 10, 15]


def load_model(fname):
    """Load a model pkl, remapping CUDA tensors to CPU if needed."""
    path = MODEL_DIR / fname
    try:
        import io
        import torch

        class CPUUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch.storage" and name == "_load_from_bytes":
                    return lambda b: torch.load(
                        io.BytesIO(b), map_location="cpu", weights_only=False
                    )
                return super().find_class(module, name)

        with open(path, "rb") as f:
            m = CPUUnpickler(f).load()
        if hasattr(m, "model_") and hasattr(m.model_, "cpu"):
            m.model_.cpu()
            m.model_.eval()
        return m
    except ImportError:
        with open(path, "rb") as f:
            return pickle.load(f)


def build_race_data(races, laps, event_name, season=2024):
    """Build driver inputs, actual finish, actual laps, pit laps for one race."""
    race = races[(races["season"] == season) & (races["event_name"] == event_name)]
    race_laps = laps[
        (laps["season"] == season) & (laps["event_name"] == event_name)
    ].copy()
    lap1 = race_laps[race_laps["lap_number"] == 1].set_index("driver_abbrev")

    drivers_input = []
    actual_finish = {}

    for _, row in race.iterrows():
        drv = row["driver_abbrev"]
        tyre = lap1.loc[drv, "tire_compound"] if drv in lap1.index else "MEDIUM"
        q_valid = [
            v
            for v in [
                row.get("q1_time_sec"),
                row.get("q2_time_sec"),
                row.get("q3_time_sec"),
            ]
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        ]
        if not q_valid:
            continue
        drivers_input.append(
            {
                "driver": drv,
                "grid_position": int(row["grid_position"]),
                "q1": row.get("q1_time_sec"),
                "q2": row.get("q2_time_sec"),
                "q3": row.get("q3_time_sec"),
                "initial_tyre": tyre,
            }
        )
        actual_finish[drv] = int(row["finish_position"])

    return drivers_input, actual_finish


def evaluate_final_positions(predicted, actual, event_name):
    """Compare predicted vs actual final positions."""
    rows = []
    for drv in predicted:
        if drv in actual:
            rows.append(
                {
                    "event": event_name,
                    "driver": drv,
                    "predicted_pos": predicted[drv],
                    "actual_pos": actual[drv],
                }
            )
    return rows


def main():
    from f1_predictor.simulation.defaults import (
        build_circuit_defaults,
        build_field_median_curves,
    )
    from f1_predictor.simulation.delta_simulator import DeltaRaceSimulator
    from f1_predictor.simulation.ensemble_simulator import EnsembleSimulator

    print("Loading data...")
    laps = pd.read_parquet("data/raw/laps/all_laps.parquet")
    races = pd.read_parquet("data/raw/race/all_races.parquet")
    circuit_defaults = build_circuit_defaults(laps)
    field_medians = build_field_median_curves(laps, races)

    print("Loading models...")
    model_h = load_model("Model_H_LightGBM_GOSS_Delta.pkl")
    model_e = load_model("Model_E_LightGBM_shallow.pkl")
    print("  H: Model_H_LightGBM_GOSS_Delta")
    print("  E: Model_E_LightGBM_shallow")

    h_sim = DeltaRaceSimulator(model_h, circuit_defaults, field_medians)

    # ===== H alone =====
    print("\n" + "=" * 70)
    print("MODEL H ALONE (baseline)")
    print("=" * 70)

    h_rows = []
    for ev in TEST_EVENTS:
        drivers_input, actual_finish = build_race_data(races, laps, ev)
        result = h_sim.simulate(ev, drivers_input)
        predicted = {r["driver"]: r["position"] for r in result.final_results}
        h_rows.extend(evaluate_final_positions(predicted, actual_finish, ev))

    h_df = pd.DataFrame(h_rows)
    h_metrics = _compute_metrics(h_df)
    _print_metrics("H alone", h_metrics)

    # ===== E alone (using H simulation to generate features, no blending) =====
    print("\n" + "=" * 70)
    print("MODEL E ALONE (H-proxy features, no trajectory blending)")
    print("=" * 70)

    e_rows = []
    for ev in TEST_EVENTS:
        drivers_input, actual_finish = build_race_data(races, laps, ev)
        ens = EnsembleSimulator(h_sim, model_e, blend_laps=0)
        result = ens.simulate(ev, drivers_input)
        # E's raw predictions (before blending)
        meta = ens.compute_meta_features(
            h_sim.simulate(ev, drivers_input),
            ens._build_qualifying_data(drivers_input),
        )
        feature_cols = [
            "A_last", "A_mean", "A_std", "A_min", "A_range", "A_last5",
            "B_last", "B_mean", "B_std", "B_last5",
            "C_pred", "grid_position", "quali_delta_to_pole",
        ]
        e_preds_raw = model_e.predict(meta[feature_cols])
        e_preds_raw = np.clip(e_preds_raw, 1.0, 20.0)
        # Rank them
        ranked_idx = np.argsort(e_preds_raw)
        e_ranked = {}
        for rank, idx in enumerate(ranked_idx, 1):
            e_ranked[meta.iloc[idx]["driver"]] = rank
        e_rows.extend(evaluate_final_positions(e_ranked, actual_finish, ev))

    e_df = pd.DataFrame(e_rows)
    e_metrics = _compute_metrics(e_df)
    _print_metrics("E alone (H-proxy features)", e_metrics)

    # ===== H+E ensemble with different blend_laps =====
    for bl in BLEND_VALUES:
        print("\n" + "=" * 70)
        print(f"ENSEMBLE H+E (blend_laps={bl})")
        print("=" * 70)

        ens_rows = []
        for ev in TEST_EVENTS:
            drivers_input, actual_finish = build_race_data(races, laps, ev)
            ens = EnsembleSimulator(h_sim, model_e, blend_laps=bl)
            result = ens.simulate(ev, drivers_input)
            predicted = {r["driver"]: r["position"] for r in result.final_results}
            ens_rows.extend(evaluate_final_positions(predicted, actual_finish, ev))

        ens_df = pd.DataFrame(ens_rows)
        ens_metrics = _compute_metrics(ens_df)
        _print_metrics(f"H+E (blend={bl})", ens_metrics)

        # Per-race breakdown
        print("\n  Per-race:")
        for ev in TEST_EVENTS:
            ev_df = ens_df[ens_df["event"] == ev]
            if len(ev_df) > 0:
                rmse = np.sqrt(np.mean((ev_df["actual_pos"] - ev_df["predicted_pos"]) ** 2))
                rho = spearmanr(ev_df["actual_pos"], ev_df["predicted_pos"])[0] if len(ev_df) >= 3 else np.nan
                print(f"    {ev:35s}  RMSE={rmse:.2f}  Spearman={rho:.3f}")

    # ===== Summary table =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<30s} {'RMSE':>6s} {'Spearman':>9s} {'Within 3':>9s} {'Within 5':>9s}")
    print("-" * 70)

    _print_summary_row("H alone", h_metrics)
    _print_summary_row("E alone (H-proxy)", e_metrics)

    for bl in BLEND_VALUES:
        ens_rows = []
        for ev in TEST_EVENTS:
            drivers_input, actual_finish = build_race_data(races, laps, ev)
            ens = EnsembleSimulator(h_sim, model_e, blend_laps=bl)
            result = ens.simulate(ev, drivers_input)
            predicted = {r["driver"]: r["position"] for r in result.final_results}
            ens_rows.extend(evaluate_final_positions(predicted, actual_finish, ev))
        ens_df = pd.DataFrame(ens_rows)
        m = _compute_metrics(ens_df)
        _print_summary_row(f"H+E (blend={bl})", m)


def _compute_metrics(df):
    actual = df["actual_pos"].values
    predicted = df["predicted_pos"].values
    abs_err = np.abs(actual - predicted)

    spear_vals = []
    for _, grp in df.groupby("event"):
        if len(grp) >= 3:
            rho, _ = spearmanr(grp["actual_pos"], grp["predicted_pos"])
            spear_vals.append(rho)

    return {
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "spearman": float(np.mean(spear_vals)) if spear_vals else float("nan"),
        "within_3": float(np.mean(abs_err <= 3) * 100),
        "within_5": float(np.mean(abs_err <= 5) * 100),
        "n": len(df),
    }


def _print_metrics(label, m):
    print(f"\n  {label}:")
    print(f"    RMSE        : {m['rmse']:.3f}")
    print(f"    Spearman    : {m['spearman']:.3f}")
    print(f"    Within 3    : {m['within_3']:.1f}%")
    print(f"    Within 5    : {m['within_5']:.1f}%")
    print(f"    N drivers   : {m['n']}")


def _print_summary_row(label, m):
    print(
        f"{label:<30s} {m['rmse']:6.2f} {m['spearman']:9.3f} "
        f"{m['within_3']:8.1f}% {m['within_5']:8.1f}%"
    )


if __name__ == "__main__":
    main()
