#!/usr/bin/env python
"""Train a bridge model: H simulation stats → actual finish position.

Runs H simulation on all 2019-2023 races, collects trajectory stats per driver,
trains a LightGBM to correct H's biases. Tests on 2024.
"""

import pickle
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

MODEL_DIR = Path("data/raw/model")

TEST_EVENTS_2024 = [
    "Bahrain Grand Prix",
    "Emilia Romagna Grand Prix",
    "Hungarian Grand Prix",
    "Mexico City Grand Prix",
]


def main():
    from f1_predictor.simulation.defaults import (
        build_circuit_defaults,
        build_field_median_curves,
    )
    from f1_predictor.simulation.delta_simulator import DeltaRaceSimulator

    print("Loading data...")
    laps = pd.read_parquet("data/raw/laps/all_laps.parquet")
    races = pd.read_parquet("data/raw/race/all_races.parquet")
    circuit_defaults = build_circuit_defaults(laps)
    field_medians = build_field_median_curves(laps, races)

    print("Loading Model H...")
    with open(MODEL_DIR / "Model_H_LightGBM_GOSS_Delta.pkl", "rb") as f:
        model_h = pickle.load(f)
    h_sim = DeltaRaceSimulator(model_h, circuit_defaults, field_medians)

    # Collect training data: simulate all available races
    all_rows = []
    train_seasons = [2019, 2020, 2021, 2022, 2023]
    test_season = 2024

    for season in train_seasons + [test_season]:
        season_races = races[races["season"] == season]
        events = season_races["event_name"].unique()
        print(f"\n  Season {season}: {len(events)} events")

        for ev in events:
            if ev not in circuit_defaults:
                continue

            race = season_races[season_races["event_name"] == ev]
            race_laps = laps[
                (laps["season"] == season) & (laps["event_name"] == ev)
            ]

            if len(race_laps) == 0:
                continue

            lap1 = race_laps[race_laps["lap_number"] == 1].set_index("driver_abbrev")
            drivers_input = []
            actual_finish = {}

            for _, row in race.iterrows():
                drv = row["driver_abbrev"]
                q_valid = [
                    v
                    for v in [
                        row.get("q1_time_sec"),
                        row.get("q2_time_sec"),
                        row.get("q3_time_sec"),
                    ]
                    if v is not None
                    and not (isinstance(v, float) and np.isnan(v))
                ]
                if not q_valid:
                    continue
                gp = row.get("grid_position")
                if gp is None or (isinstance(gp, float) and np.isnan(gp)):
                    continue
                tyre = (
                    lap1.loc[drv, "tire_compound"]
                    if drv in lap1.index
                    else "MEDIUM"
                )
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
                fp = row.get("finish_position")
                if fp is not None and not (isinstance(fp, float) and np.isnan(fp)):
                    actual_finish[drv] = int(fp)

            if len(drivers_input) < 10:
                continue

            try:
                result = h_sim.simulate(ev, drivers_input)
            except Exception as e:
                print(f"    SKIP {ev}: {e}")
                continue

            df_sim = result.to_dataframe()

            # Build qualifying data
            pole_times = []
            for d in drivers_input:
                qs = [d.get(f"q{i}") for i in (1, 2, 3)]
                valid = [t for t in qs if t is not None and not (isinstance(t, float) and np.isnan(t))]
                if valid:
                    pole_times.append(min(valid))
            pole = min(pole_times) if pole_times else 90.0

            for d_info in drivers_input:
                drv = d_info["driver"]
                if drv not in actual_finish:
                    continue

                d_laps = df_sim[df_sim["driver"] == drv].sort_values("lap_number")
                positions = d_laps["position"].values.astype(float)
                lap_times = d_laps["lap_time"].values

                if len(positions) < 5:
                    continue

                qs = [d_info.get(f"q{i}") for i in (1, 2, 3)]
                valid = [t for t in qs if t is not None and not (isinstance(t, float) and np.isnan(t))]
                best_q = min(valid) if valid else 90.0

                all_rows.append(
                    {
                        "season": season,
                        "event": ev,
                        "driver": drv,
                        "grid_position": d_info["grid_position"],
                        "quali_delta_to_pole": best_q - pole,
                        "h_final_pos": float(positions[-1]),
                        "h_mean_pos": float(np.mean(positions)),
                        "h_std_pos": float(np.std(positions)),
                        "h_min_pos": float(np.min(positions)),
                        "h_max_pos": float(np.max(positions)),
                        "h_range_pos": float(np.ptp(positions)),
                        "h_last5_pos": float(np.mean(positions[-5:])),
                        "h_first5_pos": float(np.mean(positions[:5])),
                        "h_trend": float(np.mean(positions[-5:]) - np.mean(positions[:5])),
                        "h_mean_laptime": float(np.mean(lap_times)),
                        "h_std_laptime": float(np.std(lap_times)),
                        "actual_finish": actual_finish[drv],
                    }
                )

            n_drivers = len([r for r in all_rows if r["event"] == ev and r["season"] == season])
            print(f"    {ev}: {n_drivers} drivers")

    full_df = pd.DataFrame(all_rows)
    print(f"\nTotal samples: {len(full_df)}")

    # Split
    train_df = full_df[full_df["season"].isin(train_seasons)]
    test_df = full_df[full_df["season"] == test_season]
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    feature_cols = [
        "grid_position",
        "quali_delta_to_pole",
        "h_final_pos",
        "h_mean_pos",
        "h_std_pos",
        "h_min_pos",
        "h_max_pos",
        "h_range_pos",
        "h_last5_pos",
        "h_first5_pos",
        "h_trend",
        "h_mean_laptime",
        "h_std_laptime",
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["actual_finish"]
    X_test = test_df[feature_cols]
    y_test = test_df["actual_finish"]

    # Train LightGBM
    print("\nTraining bridge model...")
    model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"\nTrain RMSE: {train_rmse:.3f}")
    print(f"Test RMSE:  {test_rmse:.3f}")

    # Compare H alone vs bridge on test set
    h_rmse = np.sqrt(mean_squared_error(y_test, test_df["h_final_pos"]))
    print(f"H alone RMSE: {h_rmse:.3f}")

    # Spearman per race
    print("\nPer-race comparison:")
    print(f"{'Event':<35s} {'H RMSE':>7s} {'Bridge':>7s} {'H Spear':>8s} {'Br Spear':>9s}")
    print("-" * 70)
    for ev in test_df["event"].unique():
        ev_df = test_df[test_df["event"] == ev]
        ev_pred = model.predict(ev_df[feature_cols])
        h_r = np.sqrt(mean_squared_error(ev_df["actual_finish"], ev_df["h_final_pos"]))
        b_r = np.sqrt(mean_squared_error(ev_df["actual_finish"], ev_pred))
        h_s = spearmanr(ev_df["actual_finish"], ev_df["h_final_pos"])[0]
        b_s = spearmanr(ev_df["actual_finish"], ev_pred)[0]
        print(f"{ev:<35s} {h_r:7.2f} {b_r:7.2f} {h_s:8.3f} {b_s:9.3f}")

    # Within-N
    test_err_h = np.abs(y_test.values - test_df["h_final_pos"].values)
    test_err_b = np.abs(y_test.values - y_pred_test)
    print(f"\nWithin 3: H={np.mean(test_err_h <= 3)*100:.1f}% | Bridge={np.mean(test_err_b <= 3)*100:.1f}%")
    print(f"Within 5: H={np.mean(test_err_h <= 5)*100:.1f}% | Bridge={np.mean(test_err_b <= 5)*100:.1f}%")

    # Feature importance
    print("\nFeature importance:")
    imp = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1])
    for fname, score in imp:
        print(f"  {fname:<25s} {score:5d}")

    # Save bridge model
    out_path = MODEL_DIR / "Model_Bridge_H_to_final.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)
    print(f"\nSaved bridge model to {out_path}")


if __name__ == "__main__":
    main()
