"""Ensemble simulator: Model H (lap-by-lap) + Model E (final position stacker)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from f1_predictor.simulation.engine import LapRecord, SimulationResult


class EnsembleSimulator:
    """Combine Model H trajectories with Model E final-position refinement.

    H's simulated positions proxy A/B's lap-level position predictions.
    C_pred uses H's final simulated position (Model C needs 24 historical
    features not available at inference time).
    """

    def __init__(
        self,
        h_simulator: Any,
        model_e: Any,
        blend_laps: int = 10,
    ) -> None:
        self.h_simulator = h_simulator
        self.model_e = model_e
        self.blend_laps = blend_laps

    def simulate(
        self,
        circuit: str,
        drivers: list[dict[str, Any]],
        strategies: dict[str, list[tuple[str, int | None]]] | None = None,
        qualifying_data: dict[str, dict[str, float]] | None = None,
    ) -> SimulationResult:
        h_result: SimulationResult = self.h_simulator.simulate(circuit, drivers, strategies)

        if qualifying_data is None:
            qualifying_data = self._build_qualifying_data(drivers)

        e_predictions = self._predict_final_positions(h_result, qualifying_data)

        if self.blend_laps <= 0:
            return h_result

        return self._blend_trajectories(h_result, e_predictions)

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

    def compute_meta_features(
        self,
        h_result: SimulationResult,
        qualifying_data: dict[str, dict[str, float]],
    ) -> pd.DataFrame:
        """Compute Model E's 13 features from H simulation output."""
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

    def _blend_trajectories(
        self,
        h_result: SimulationResult,
        e_predictions: dict[str, float],
    ) -> SimulationResult:
        """Blend H's lap-by-lap positions toward E's final predictions."""
        total_laps = h_result.total_laps
        blend_start = max(1, total_laps - self.blend_laps)

        old_by_lap: dict[int, dict[str, LapRecord]] = {}
        for rec in h_result.lap_records:
            old_by_lap.setdefault(rec.lap, {})[rec.driver] = rec

        drivers = sorted({rec.driver for rec in h_result.lap_records})

        h_positions: dict[str, dict[int, int]] = {}
        for d in drivers:
            h_positions[d] = {}
        for rec in h_result.lap_records:
            h_positions[rec.driver][rec.lap] = rec.position

        new_records: list[LapRecord] = []

        for lap in range(1, total_laps + 1):
            lap_recs = old_by_lap.get(lap, {})
            if lap <= blend_start:
                new_records.extend(lap_recs.values())
            else:
                alpha = (lap - blend_start) / self.blend_laps

                blended_scores = {}
                for d in drivers:
                    h_pos = h_positions[d].get(lap, 20)
                    e_pos = e_predictions.get(d, h_pos)
                    blended_scores[d] = (1 - alpha) * h_pos + alpha * e_pos

                ranked = sorted(blended_scores, key=lambda d: blended_scores[d])
                rank_map = {d: i + 1 for i, d in enumerate(ranked)}

                for d in drivers:
                    old_rec = lap_recs.get(d)
                    if old_rec is None:
                        continue
                    new_records.append(
                        LapRecord(
                            lap=old_rec.lap,
                            driver=old_rec.driver,
                            position=rank_map[d],
                            lap_time=old_rec.lap_time,
                            cum_time=old_rec.cum_time,
                            gap_to_leader=old_rec.gap_to_leader,
                            compound=old_rec.compound,
                            tire_life=old_rec.tire_life,
                            stint=old_rec.stint,
                        )
                    )

        final_results = []
        last_lap_recs = {r.driver: r for r in new_records if r.lap == total_laps}
        leader_time = min(r.cum_time for r in last_lap_recs.values()) if last_lap_recs else 0.0
        for d in sorted(last_lap_recs, key=lambda x: last_lap_recs[x].position):
            rec = last_lap_recs[d]
            final_results.append(
                {
                    "driver": d,
                    "position": rec.position,
                    "total_time": rec.cum_time,
                    "gap_to_leader": rec.cum_time - leader_time,
                    "pit_stops": rec.stint - 1,
                }
            )

        return SimulationResult(
            circuit=h_result.circuit,
            total_laps=total_laps,
            lap_records=new_records,
            final_results=final_results,
        )
