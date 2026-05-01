# ruff: noqa: N803, N806  — sklearn convention uses uppercase X
"""Reshape tabular simulation data into windowed sequences for Model G."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

from f1_predictor.features.simulation_features import SIMULATION_FEATURE_COLS

DRIVER_RACE_KEY = ["season", "round", "driver_abbrev"]


def build_sequence_training_data(
    df: pd.DataFrame,
    max_window: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64], pd.DataFrame]:
    """Reshape tabular simulation data into windowed sequences.

    For each driver-race, creates sliding windows of length max_window.
    Early laps (< max_window history) are left-padded with zeros.
    A seq_valid_mask feature (0=pad, 1=real) is appended.

    Args:
        df: Output of build_simulation_training_data() with lap_time_ratio.
        max_window: Maximum window length.

    Returns:
        X_seq: (n_samples, max_window, n_features+1) — padded sequences
               with seq_valid_mask as last feature
        y: (n_samples,) — target for the current lap (last in window)
        id_df: DataFrame with season/round/driver/lap_number for each sample
    """
    import pandas as _pd

    feature_cols = list(SIMULATION_FEATURE_COLS)
    n_features = len(feature_cols)

    df = df.sort_values([*DRIVER_RACE_KEY, "lap_number"]).reset_index(drop=True)

    # Compute per-column medians for NaN imputation (forward-fill within
    # each driver-race, then fall back to global median for remaining NaN)
    _global_medians = df[feature_cols].median()

    X_list: list[NDArray[np.float64]] = []
    y_list: list[float] = []
    id_rows: list[dict[str, object]] = []

    for (_season, _round, _driver), grp in df.groupby(DRIVER_RACE_KEY):
        grp_filled = grp[feature_cols].ffill().fillna(_global_medians)
        features = grp_filled.values.astype(np.float64)
        targets = grp["lap_time_ratio"].values.astype(np.float64)
        n_laps = len(grp)

        id_cols_vals = grp[["season", "round", "driver_abbrev", "lap_number"]].values

        for i in range(n_laps):
            window_start = max(0, i - max_window + 1)
            window = features[window_start : i + 1]
            n_valid = len(window)

            # Left-pad if needed
            if n_valid < max_window:
                pad = np.zeros((max_window - n_valid, n_features))
                window = np.vstack([pad, window])

            # Add mask feature
            mask = np.zeros((max_window, 1))
            mask[-n_valid:] = 1.0
            window_with_mask = np.hstack([window, mask])

            X_list.append(window_with_mask)
            y_list.append(targets[i])
            id_rows.append(
                {
                    "season": int(id_cols_vals[i, 0]),
                    "round": int(id_cols_vals[i, 1]),
                    "driver_abbrev": str(id_cols_vals[i, 2]),
                    "lap_number": int(id_cols_vals[i, 3]),
                }
            )

    X_seq = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    id_df = _pd.DataFrame(id_rows)

    return X_seq, y, id_df


def slice_window(X_seq: NDArray[np.float64], window_size: int) -> NDArray[np.float64]:
    """Slice a pre-built max-window sequence array to a smaller window.

    Takes the last `window_size` time steps from each sample.
    """
    if window_size >= X_seq.shape[1]:
        return X_seq
    return X_seq[:, -window_size:, :]
