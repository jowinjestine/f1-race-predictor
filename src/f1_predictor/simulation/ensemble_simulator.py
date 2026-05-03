"""Ensemble simulator: Model H (lap-by-lap) + Model E (final position stacker)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from f1_predictor.simulation.engine import SimulationResult

MODEL_A_FEATURES = [
    "gap_to_leader",
    "lap_time_delta_race_median",
    "gap_to_ahead",
    "position_change_from_lap1",
    "tire_life",
    "race_progress_pct",
    "degradation_rate",
    "compound_pace_delta",
    "pit_stop_count",
]

MODEL_B_FEATURES = [
    "gap_to_leader",
    "lap_time_delta_race_median",
    "gap_to_ahead",
    "race_progress_pct",
    "position_change_from_lap1",
    "laps_since_last_pit",
    "pit_stop_count",
    "lap_time_rolling_3",
]


class EnsembleSimulator:
    """Combine Model H trajectories with Model E final-position refinement.

    When Models A and B are provided, their lap-level predictions are
    computed independently from H's simulated lap data, giving Model E
    decorrelated input features.  Otherwise falls back to using H's
    positions as a proxy.
    """

    def __init__(
        self,
        h_simulator: Any,
        model_e: Any,
        *,
        model_a: Any = None,
        model_b: Any = None,
        blend_laps: int = 10,
    ) -> None:
        self.h_simulator = h_simulator
        self.model_e = model_e
        self.model_a = model_a
        self.model_b = model_b
        self.blend_laps = blend_laps

    def simulate(
        self,
        circuit: str,
        drivers: list[dict[str, Any]],
        strategies: dict[str, list[tuple[str, int | None]]] | None = None,
        qualifying_data: dict[str, dict[str, float]] | None = None,
        *,
        dnf_probs: dict[str, float] | None = None,
    ) -> SimulationResult:
        h_result: SimulationResult = self.h_simulator.simulate(
            circuit, drivers, strategies, dnf_probs=dnf_probs
        )

        if qualifying_data is None:
            qualifying_data = self._build_qualifying_data(drivers)

        e_predictions = self._predict_final_positions(h_result, qualifying_data)

        if self.blend_laps <= 0:
            return h_result

        return self._blend_trajectories(h_result, e_predictions)

    # ------------------------------------------------------------------
    # Qualifying helpers
    # ------------------------------------------------------------------

    def _build_qualifying_data(self, drivers: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        pole_times = []
        for d in drivers:
            qs = [d.get(f"q{i}") for i in (1, 2, 3)]
            valid = [t for t in qs if t is not None and not (isinstance(t, float) and np.isnan(t))]
            if valid:
                pole_times.append(min(valid))
        pole = min(pole_times) if pole_times else 90.0

        result = {}
        for d in drivers:
            qs = [d.get(f"q{i}") for i in (1, 2, 3)]
            valid = [t for t in qs if t is not None and not (isinstance(t, float) and np.isnan(t))]
            best = min(valid) if valid else 90.0
            result[d["driver"]] = {
                "grid_position": d["grid_position"],
                "quali_delta_to_pole": best - pole,
            }
        return result

    # ------------------------------------------------------------------
    # Feature reconstruction from H's simulation output
    # ------------------------------------------------------------------

    def _build_lap_features(
        self,
        h_result: SimulationResult,
        qualifying_data: dict[str, dict[str, float]],
    ) -> pd.DataFrame:
        """Reconstruct Model A and B input features from H's lap records."""
        df = h_result.to_dataframe()
        total_laps = h_result.total_laps

        grid_map = {d: qd.get("grid_position", 1) for d, qd in qualifying_data.items()}
        df["grid_position"] = df["driver"].map(grid_map)

        # Direct features
        df["pit_stop_count"] = df["stint"] - 1
        df["laps_since_last_pit"] = (df["tire_life"] - 1).clip(lower=0)
        df["race_progress_pct"] = df["lap_number"] / total_laps
        df["position_change_from_lap1"] = df["grid_position"] - df["position"]

        # gap_to_ahead: time gap to the car one position ahead
        df = df.sort_values(["lap_number", "cum_time"])
        df["gap_to_ahead"] = df.groupby("lap_number")["cum_time"].diff().fillna(0.0)

        # lap_time_delta_race_median: expanding median baseline, shifted by 1 lap
        lap_medians = df.groupby("lap_number")["lap_time"].median()
        expanding_med = lap_medians.expanding().median().shift(1)
        df["lap_time_delta_race_median"] = df["lap_time"] - df["lap_number"].map(expanding_med)

        # lap_time_rolling_3: shifted rolling mean (Model B)
        df = df.sort_values(["driver", "lap_number"])
        df["lap_time_rolling_3"] = df.groupby("driver")["lap_time"].transform(
            lambda x: x.rolling(3, min_periods=1).mean().shift(1)
        )

        # degradation_rate: OLS slope of lap_time vs tire_life within stint
        df["degradation_rate"] = self._compute_stint_degradation(df)

        # compound_pace_delta: too complex to reconstruct; LightGBM handles NaN
        df["compound_pace_delta"] = np.nan

        return df

    @staticmethod
    def _compute_stint_degradation(df: pd.DataFrame) -> pd.Series:
        """OLS slope of lap_time vs tire_life within each driver-stint."""
        deg = pd.Series(np.nan, index=df.index)
        for _, group in df.groupby(["driver", "stint"], sort=False):
            times = group["lap_time"].values
            lives = group["tire_life"].values
            for i in range(3, len(times)):
                valid = np.isfinite(times[:i]) & np.isfinite(lives[:i])
                if valid.sum() >= 3:
                    coeffs = np.polyfit(lives[:i][valid], times[:i][valid], 1)
                    deg.iloc[group.index[i]] = coeffs[0]
        return deg

    # ------------------------------------------------------------------
    # Model E meta-feature computation
    # ------------------------------------------------------------------

    def compute_meta_features(
        self,
        h_result: SimulationResult,
        qualifying_data: dict[str, dict[str, float]],
    ) -> pd.DataFrame:
        """Compute Model E's 13 features from H simulation output.

        When Models A and B are loaded, runs them independently on
        reconstructed lap features for decorrelated predictions.
        Otherwise falls back to using H's positions as proxy.
        """
        if self.model_a is not None and self.model_b is not None:
            return self._compute_meta_real(h_result, qualifying_data)
        return self._compute_meta_proxy(h_result, qualifying_data)

    def _compute_meta_real(
        self,
        h_result: SimulationResult,
        qualifying_data: dict[str, dict[str, float]],
    ) -> pd.DataFrame:
        """Use real Model A/B predictions for decorrelated meta-features."""
        feat_df = self._build_lap_features(h_result, qualifying_data)

        a_preds = self.model_a.predict(feat_df[MODEL_A_FEATURES])
        b_preds = self.model_b.predict(feat_df[MODEL_B_FEATURES])
        feat_df["a_pred"] = a_preds
        feat_df["b_pred"] = b_preds

        rows = []
        for driver in feat_df["driver"].unique():
            d_laps = feat_df[feat_df["driver"] == driver].sort_values("lap_number")
            a_vals = d_laps["a_pred"].values
            b_vals = d_laps["b_pred"].values
            h_positions = d_laps["position"].values.astype(float)

            if len(a_vals) == 0:
                continue

            a_last5 = a_vals[-5:] if len(a_vals) >= 5 else a_vals
            b_last5 = b_vals[-5:] if len(b_vals) >= 5 else b_vals

            q_data = qualifying_data.get(driver, {})
            grid = q_data.get("grid_position", h_positions[0])
            quali_delta = q_data.get("quali_delta_to_pole", 0.0)

            rows.append(
                {
                    "driver": driver,
                    "A_last": float(a_vals[-1]),
                    "A_mean": float(np.mean(a_vals)),
                    "A_std": float(np.std(a_vals)) if len(a_vals) > 1 else 0.0,
                    "A_min": float(np.min(a_vals)),
                    "A_range": float(np.max(a_vals) - np.min(a_vals)),
                    "A_last5": float(np.mean(a_last5)),
                    "B_last": float(b_vals[-1]),
                    "B_mean": float(np.mean(b_vals)),
                    "B_std": float(np.std(b_vals)) if len(b_vals) > 1 else 0.0,
                    "B_last5": float(np.mean(b_last5)),
                    "C_pred": float(h_positions[-1]),
                    "grid_position": grid,
                    "quali_delta_to_pole": quali_delta,
                }
            )

        return pd.DataFrame(rows)

    def _compute_meta_proxy(
        self,
        h_result: SimulationResult,
        qualifying_data: dict[str, dict[str, float]],
    ) -> pd.DataFrame:
        """Fallback: use H's positions as proxy for A/B (original behaviour)."""
        df = h_result.to_dataframe()
        rows = []

        for driver in df["driver"].unique():
            d_laps = df[df["driver"] == driver].sort_values("lap_number")
            positions = d_laps["position"].values.astype(float)

            if len(positions) == 0:
                continue

            last5 = positions[-5:] if len(positions) >= 5 else positions
            pos_std = float(np.std(positions)) if len(positions) > 1 else 0.0

            q_data = qualifying_data.get(driver, {})
            grid = q_data.get("grid_position", positions[0])
            quali_delta = q_data.get("quali_delta_to_pole", 0.0)

            rows.append(
                {
                    "driver": driver,
                    "A_last": float(positions[-1]),
                    "A_mean": float(np.mean(positions)),
                    "A_std": pos_std,
                    "A_min": float(np.min(positions)),
                    "A_range": float(np.max(positions) - np.min(positions)),
                    "A_last5": float(np.mean(last5)),
                    "B_last": float(positions[-1]),
                    "B_mean": float(np.mean(positions)),
                    "B_std": pos_std,
                    "B_last5": float(np.mean(last5)),
                    "C_pred": float(positions[-1]),
                    "grid_position": grid,
                    "quali_delta_to_pole": quali_delta,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Model E prediction
    # ------------------------------------------------------------------

    def _predict_final_positions(
        self,
        h_result: SimulationResult,
        qualifying_data: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Run Model E on H-derived meta-features."""
        meta_df = self.compute_meta_features(h_result, qualifying_data)

        feature_cols = [
            "A_last",
            "A_mean",
            "A_std",
            "A_min",
            "A_range",
            "A_last5",
            "B_last",
            "B_mean",
            "B_std",
            "B_last5",
            "C_pred",
            "grid_position",
            "quali_delta_to_pole",
        ]

        e_preds = self.model_e.predict(meta_df[feature_cols])
        e_preds = np.clip(e_preds, 1.0, 20.0)

        return dict(zip(meta_df["driver"], e_preds, strict=True))

    # ------------------------------------------------------------------
    # Trajectory blending
    # ------------------------------------------------------------------

    def _blend_trajectories(
        self,
        h_result: SimulationResult,
        e_predictions: dict[str, float],
    ) -> SimulationResult:
        """Re-rank final standings using E's predictions; keep H's telemetry.

        H provides realistic lap-by-lap telemetry (times, gaps, tyre state).
        E provides more accurate final position predictions (RMSE 2.60 vs 3.46).
        We keep H's lap records unchanged and use E to determine the finishing order.
        """
        total_laps = h_result.total_laps
        h_final_map = {fr["driver"]: fr for fr in h_result.final_results}

        finishers = [
            fr["driver"]
            for fr in h_result.final_results
            if fr.get("status", "Finished") == "Finished"
        ]
        dnfs = [fr for fr in h_result.final_results if fr.get("status", "Finished") != "Finished"]

        ranked = sorted(finishers, key=lambda d: e_predictions.get(d, 20.0))

        final_results = []
        leader_time = min(h_final_map[d]["total_time"] for d in finishers) if finishers else 0.0
        for pos, d in enumerate(ranked, 1):
            h_fr = h_final_map[d]
            final_results.append(
                {
                    "driver": d,
                    "position": pos,
                    "total_time": h_fr["total_time"],
                    "gap_to_leader": h_fr["total_time"] - leader_time,
                    "pit_stops": h_fr.get("pit_stops", 0),
                    "status": "Finished",
                    "laps_completed": h_fr.get("laps_completed", total_laps),
                }
            )

        for dnf_fr in dnfs:
            final_results.append(
                {
                    **dnf_fr,
                    "position": len(final_results) + 1,
                }
            )

        return SimulationResult(
            circuit=h_result.circuit,
            total_laps=total_laps,
            lap_records=list(h_result.lap_records),
            final_results=final_results,
        )
