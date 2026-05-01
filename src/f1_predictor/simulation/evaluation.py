"""Shared simulation evaluation metrics for Models F-I."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def evaluate_simulation(sim_df: pd.DataFrame) -> dict[str, float]:
    """Compute standard simulation metrics.

    Args:
        sim_df: DataFrame with columns: event, driver, predicted_pos, actual_pos

    Returns:
        Dict with position_rmse, position_mae, r2, within_1, within_3,
        within_5, spearman_mean, n_races.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    if len(sim_df) == 0:
        return {
            "position_rmse": float("nan"),
            "position_mae": float("nan"),
            "r2": float("nan"),
            "within_1": float("nan"),
            "within_3": float("nan"),
            "within_5": float("nan"),
            "spearman_mean": float("nan"),
            "n_races": 0,
        }

    actual = sim_df["actual_pos"].values
    predicted = sim_df["predicted_pos"].values
    abs_err = np.abs(actual - predicted)  # type: ignore[operator]

    spear_vals = []
    for _event, grp in sim_df.groupby("event"):
        if len(grp) >= 3 and grp["actual_pos"].std() > 0 and grp["predicted_pos"].std() > 0:
            rho, _ = spearmanr(grp["actual_pos"], grp["predicted_pos"])
            spear_vals.append(rho)

    return {
        "position_rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "position_mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
        "within_1": float(np.mean(abs_err <= 1) * 100),
        "within_3": float(np.mean(abs_err <= 3) * 100),
        "within_5": float(np.mean(abs_err <= 5) * 100),
        "spearman_mean": float(np.mean(spear_vals)) if spear_vals else float("nan"),
        "n_races": int(sim_df["event"].nunique()),
    }


def evaluate_monte_carlo_calibration(mc_results: list[dict[str, float]]) -> dict[str, float]:
    """Evaluate calibration of Monte Carlo / quantile predictions.

    Args:
        mc_results: List of dicts with keys: driver, actual_pos, predicted_pos,
                    position_p10, position_p90, position_p25, position_p75.

    Returns:
        Dict with coverage_80, coverage_50, sharpness_80, sharpness_50.
    """
    if not mc_results:
        return {
            "coverage_80": float("nan"),
            "coverage_50": float("nan"),
            "sharpness_80": float("nan"),
            "sharpness_50": float("nan"),
        }

    in_80 = 0
    in_50 = 0
    width_80 = []
    width_50 = []
    n = len(mc_results)

    for r in mc_results:
        actual = r["actual_pos"]
        if "position_p10" in r and "position_p90" in r:
            lo_80, hi_80 = r["position_p10"], r["position_p90"]
            if lo_80 <= actual <= hi_80:
                in_80 += 1
            width_80.append(hi_80 - lo_80)
        if "position_p25" in r and "position_p75" in r:
            lo_50, hi_50 = r["position_p25"], r["position_p75"]
            if lo_50 <= actual <= hi_50:
                in_50 += 1
            width_50.append(hi_50 - lo_50)

    return {
        "coverage_80": float(in_80 / n * 100) if width_80 else float("nan"),
        "coverage_50": float(in_50 / n * 100) if width_50 else float("nan"),
        "sharpness_80": float(np.mean(width_80)) if width_80 else float("nan"),
        "sharpness_50": float(np.mean(width_50)) if width_50 else float("nan"),
    }
