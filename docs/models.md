# Model Documentation

Nine models trained in sequence, each building on outputs from earlier stages. Models A-C are base predictors, D-E are stacking ensembles, and F-I power autoregressive race simulators.

## Model Summary

| Model | Algorithm | Target | Data | Test RMSE | CV Strategy |
|-------|-----------|--------|------|-----------|-------------|
| **A** | LightGBM | `position` (per lap) | 139K laps (2019-2024) | 2.71 | LeaveOneSeasonOut |
| **B** | LightGBM DART | `position` (per lap) | 165K laps (2018-2025) | 5.29 | ExpandingWindow |
| **C** | XGBoost Conservative | `finish_position` | 3.5K race-drivers (2018-2025) | 4.21 | ExpandingWindow |
| **D** | LightGBM shallow (meta) | `finish_position` | 2K race-drivers | 3.18 | LeaveOneSeasonOut |
| **E** | LightGBM shallow (meta) | `finish_position` | 2K race-drivers | 2.60 | LeaveOneSeasonOut |
| **F** | LightGBM Shallow | `lap_time_ratio` | 79K laps (2019-2024) | 0.021 | ExpandingWindow |
| **G** | GRU 2-layer | `lap_time_ratio` | 79K sequences | — | ExpandingWindow |
| **H** | DeltaRaceSimulator | `delta_ratio` | via F features | 3.46* | Simulation eval |
| **I** | QuantileRaceSimulator | 5 quantiles of `lap_time_ratio` | via F features | 6.19* | Simulation eval |

*H and I RMSE is measured on final race positions across 4 held-out 2024 races, not per-lap.

---

## Model A — Lap + Tyre

**Notebook:** `notebooks/05a_model_A_training.ipynb`
**Feature module:** `src/f1_predictor/features/lap_features.py` (`build_lap_tyre_features`)

Predicts a driver's position at each lap using tyre-specific features. Requires tyre compound data, so limited to 2019-2024 (FastF1 coverage).

### Features (9)

| Feature | Description |
|---------|-------------|
| `gap_to_leader` | Time gap to race leader (seconds) |
| `lap_time_delta_race_median` | Lap time minus race-wide median for that lap |
| `gap_to_ahead` | Time gap to car immediately ahead |
| `position_change_from_lap1` | Current position minus grid position |
| `tire_life` | Laps on current tyre set |
| `race_progress_pct` | Lap number / total laps |
| `degradation_rate` | OLS slope of lap times vs tyre life within stint |
| `compound_pace_delta` | Compound-specific pace offset |
| `pit_stop_count` | Number of pit stops completed |

### Training

- **CV:** LeaveOneSeasonOut — each fold holds out one season (2019-2023), test on 2024
- **Variants tested:** XGBoost (standard, DART, Linear, Conservative, Deep), LightGBM (standard, DART, GOSS, Shallow, Deep), GRU 2-layer, FT-Transformer
- **Best:** LightGBM standard (RMSE 2.71, MAE 2.01)
- **Tuning:** Optuna 3-round (screening, 10 trials, 15 trials)
- **OOF predictions** saved for downstream stacking in Models D and E

---

## Model B — Lap, No Tyre

**Notebook:** `notebooks/05b_model_B_training.ipynb`
**Feature module:** `src/f1_predictor/features/lap_features.py` (`build_lap_notyre_features`)

Same task as Model A but without tyre compound data. Trades feature richness for longer history (2018-2025 including 2025 via Jolpica API).

### Features (8)

| Feature | Description |
|---------|-------------|
| `gap_to_leader` | Time gap to race leader |
| `lap_time_delta_race_median` | Lap time minus race-wide median |
| `gap_to_ahead` | Time gap to car immediately ahead |
| `race_progress_pct` | Lap number / total laps |
| `position_change_from_lap1` | Position change from grid |
| `laps_since_last_pit` | Laps since most recent pit stop |
| `pit_stop_count` | Number of pit stops completed |
| `lap_time_rolling_3` | 3-lap rolling average of lap times |

### Training

- **CV:** ExpandingWindowSplit — growing windows (2018 -> 2018-19 -> ... -> 2018-23), test on 2025
- **Best:** LightGBM DART (RMSE 5.29)
- **Note:** Higher test RMSE than A due to distribution shift to 2025 season (larger overfit gap ~2.1)

---

## Model C — Pre-Race Features

**Notebook:** `notebooks/05c_model_C_training.ipynb`
**Feature module:** `src/f1_predictor/features/race_features.py` (`build_race_features`)

Predicts race-level finish position from pre-race context only — no lap-level information. One prediction per driver per race.

### Features (15)

| Group | Features |
|-------|----------|
| **Grid** | `grid_position`, `quali_delta_to_pole`, `best_quali_sec` |
| **Driver form** | `position_trend`, `points_last_3`, `driver_circuit_avg_finish`, `driver_circuit_races` |
| **Team** | `team_avg_finish_last_3`, `team_points_cumulative_season`, `quali_position_vs_teammate`, `dnf_rate_season` |
| **Circuit** | `circuit_avg_dnf_rate` |
| **Weather** | `weather_temp_max`, `weather_wind_max_kph`, `weather_precip_mm` |

### Training

- **CV:** ExpandingWindowSplit (test season 2025)
- **Variants tested:** XGBoost (Linear, Conservative), LightGBM (standard, GOSS, Deep), MLP 3-layer
- **Best:** XGBoost Conservative (RMSE 4.21, R^2 ~0.65)
- All aggregations use expanding means with `shift(1)` to prevent look-ahead leakage

---

## Model D — All-Combinations Stacking

**Notebook:** `notebooks/05d_model_D_stacking.ipynb`

Exhaustive ensemble of all A x B x C variant combinations using a two-phase tournament.

### Architecture

1. **Phase 1 — RidgeCV screening:** Test all 300 combinations (5 A-variants x 5 B-variants x 12 C-variants) with RidgeCV. Select top 20.
2. **Phase 2 — Optuna tuning:** Train LightGBM shallow meta-learner on each top-20 combination (10 trials each). Select final 5.

### Meta-Features (3)

| Feature | Description |
|---------|-------------|
| `pred_A` | Model A's OOF position prediction (aggregated to race level) |
| `pred_B` | Model B's OOF position prediction (aggregated to race level) |
| `pred_C` | Model C's OOF finish position prediction |

### Training

- **CV:** LeaveOneSeasonOut (val 2019-2022, test 2023)
- **Meta-learner:** LightGBM shallow (max_depth=3, n_estimators=100)
- **Best combination:** LightGBM_Deep (A) + LightGBM_GOSS (B) + ExtraTrees (C) -> RMSE 3.18
- **Negative overfit gap** (~-0.15), meaning test outperforms val — stacking regularises well

---

## Model E — Rich Feature Stacking

**Notebook:** `notebooks/05e_model_E_rich_stacking.ipynb`

Improved stacking that preserves lap-level richness through aggregation features. Auto-selects best A, B, C variants, then constructs rich meta-features from their OOF predictions.

### Meta-Features (13)

| Source | Features | Description |
|--------|----------|-------------|
| **Model A** (6) | `A_last`, `A_mean`, `A_std`, `A_min`, `A_range`, `A_last5` | Lap-level position aggregations |
| **Model B** (4) | `B_last`, `B_mean`, `B_std`, `B_last5` | Lap-level position aggregations |
| **Model C** (1) | `C_pred` | Pre-race finish position prediction |
| **Pre-race** (2) | `grid_position`, `quali_delta_to_pole` | Direct qualifying context |

### Training

- **CV:** LeaveOneSeasonOut (val 2019-2022, test 2023)
- **Auto-selected variants:** A=LightGBM_GOSS, B=LightGBM_GOSS, C=ExtraTrees
- **Best:** LightGBM shallow — RMSE 2.60, Spearman 0.93, within-3-positions 77%
- **Outperforms Model D by ~20% on RMSE** — the rich aggregations (std, range, last-5) capture information lost in D's single-point summaries
- **Used in production** as the final-position refinement layer in the H+E ensemble

---

## Model F — Lap Time Simulation

**Notebook:** `notebooks/05f_model_F_lap_simulation.ipynb`
**Feature module:** `src/f1_predictor/features/simulation_features.py` (`build_simulation_training_data`)

Predicts `lap_time_ratio = lap_time / best_quali_time` for autoregressive race simulation. This is the foundation model for the simulation engine.

### Features (25)

| Group | Features |
|-------|----------|
| **Static** (5) | `grid_position`, `best_quali_sec`, `circuit_street`, `circuit_hybrid`, `circuit_permanent` |
| **Deterministic** (10) | `lap_number`, `race_progress_pct`, 5x compound one-hot, `tire_life`, `stint` |
| **Pit flags** (2) | `is_pit_in_lap`, `is_pit_out_lap` |
| **Feedback** (6) | `pit_stop_count`, `laps_since_last_pit`, `lap_time_rolling_3`, `lap_time_rolling_5`, `degradation_rate`, `gap_to_leader` |
| **Position** (2) | `position`, `position_change_from_lap1` |

### Target

`lap_time_ratio` — normalised by qualifying time so the model learns relative pace rather than absolute lap times. This makes predictions transferable across circuits.

### Training

- **CV:** ExpandingWindowSplit (2019 -> ... -> 2024 test)
- **Best:** LightGBM Shallow (per-lap RMSE 0.021)
- **Simulation evaluation:** RMSE 6.19 on final positions (4 held-out 2024 races)

### Simulation Engine

`src/f1_predictor/simulation/engine.py` — `RaceSimulator` class

The simulator runs lap-by-lap, maintaining per-driver state (cumulative time, tyre life, stint, position). At each lap:
1. Build features from current state
2. Predict `lap_time_ratio` with Model F
3. Clamp predictions (normal: 1.01-1.15, pit-in: 1.10-1.50, pit-out: 1.03-1.25)
4. Convert to absolute time: `lap_time = ratio * best_quali_sec`
5. Rank drivers by cumulative time to derive positions

---

## Model G — Temporal Sequence

**Notebook:** `notebooks/05g_model_G_temporal.ipynb`
**Feature module:** `src/f1_predictor/features/sequence_features.py` (`build_sequence_training_data`)
**Simulator:** `src/f1_predictor/simulation/sequence_simulator.py` — `SequenceRaceSimulator`

Temporal variant of Model F using sequence models that consume a sliding window of past laps.

### Architecture

- **Input:** 3D tensor `(n_drivers, window_size, n_features+1)` — the `+1` is a `seq_valid_mask` channel (0=pad, 1=real) for early laps
- **Window:** Configurable sliding window (default 5-10 laps)
- **Models tested:** GRU 2-layer (bidirectional, hidden=64), LSTM, FT-Transformer
- **Left-padding:** Early laps with fewer than `window_size` history are zero-padded on the left

### Training

- **CV:** ExpandingWindowSplit
- **Note:** All GRU variants produced identical simulation results, suggesting the temporal context didn't add discriminative power beyond the tabular features' rolling aggregations

---

## Model H — Delta + Monte Carlo

**Notebook:** `notebooks/05h_model_H_delta_mc.ipynb`
**Feature module:** `src/f1_predictor/features/delta_features.py` (`build_delta_training_data`, `build_field_median_curves`)
**Simulator:** `src/f1_predictor/simulation/delta_simulator.py` — `DeltaRaceSimulator`, `MonteCarloSimulator`

Predicts `delta_ratio` (deviation from historical field median) instead of absolute `lap_time_ratio`. The baseline lookup table provides circuit-specific prior knowledge.

### Delta Decomposition

```
lap_time_ratio = field_median_ratio(circuit, lap) + delta_ratio
```

- `field_median_ratio`: Per-circuit, per-lap median from 2019-2024 historical data
- `delta_ratio`: The model's prediction — how much faster/slower a driver is than the field

This decomposition reduces the prediction range and lets the model focus on driver-specific deviations.

### Monte Carlo Wrapper

`MonteCarloSimulator` runs N simulations (default 200) with Gaussian noise injection (sigma=0.01) added to predictions. Aggregates position distributions:

| Output | Description |
|--------|-------------|
| `position` | Median position across simulations |
| `position_mean` | Mean position |
| `position_p10` / `position_p25` | Optimistic bounds (10th/25th percentile) |
| `position_p75` / `position_p90` | Pessimistic bounds |
| `position_std` | Position uncertainty |

### Performance

- **Best simulator:** RMSE 3.46, Spearman 0.82 on 4 held-out 2024 races
- **Used in production** as the core simulator in the H+E ensemble

---

## Model I — Quantile Regression

**Notebook:** `notebooks/05i_model_I_quantile.ipynb`
**Simulator:** `src/f1_predictor/simulation/quantile_simulator.py` — `QuantileRaceSimulator`

Multi-output quantile model that predicts the full distribution of `lap_time_ratio` at each lap, enabling probabilistic simulation without external noise injection.

### Architecture

- **Output:** 5 quantiles per prediction — q10, q25, q50, q75, q90
- **Sampling:** At each lap, draw uniform `u ~ [0, 1]` per driver and interpolate along the quantile curve
- **Result:** Each simulation run produces a different trajectory, reflecting inherent prediction uncertainty

### Quantile Interpolation

```python
q_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
sampled_ratio = np.interp(u, q_levels, predicted_quantiles)
```

### Performance

- **Simulation RMSE:** 6.19 on final positions
- Produces well-calibrated uncertainty bands but collapses autoregressively over many laps — error accumulation widens position distributions beyond useful range

---

## H+E Ensemble (Production)

**Simulator:** `src/f1_predictor/simulation/ensemble_simulator.py` — `EnsembleSimulator`

The production serving ensemble combines Model H's lap-by-lap simulation with Model E's final-position refinement.

### Architecture

1. Run Model H's `DeltaRaceSimulator` for a full race
2. Compute Model E's 13 meta-features from H's simulated positions (H's positions proxy A/B's lap-level predictions)
3. Predict final positions with Model E's LightGBM meta-learner
4. Optionally blend H's trajectory toward E's predictions over the last N laps

### A/B Proxy Collapse

In the current implementation, H produces a single position per driver per lap. When constructing E's meta-features, all A-derived and B-derived features come from the same source:
- `A_last = B_last = C_pred` (all equal H's final simulated position)
- `A_mean = B_mean`, `A_std = B_std`, etc.

This reduces E's 13 features to 9 unique values, limiting its discriminative power. As a result, `blend_laps` defaults to 0 (H-only), with E available as an optional refinement.

### Validation Results (4 held-out 2024 races)

| Configuration | RMSE | Spearman |
|---------------|------|----------|
| H only (blend=0) | 3.51 | 0.80 |
| H+E (blend=5) | 3.50 | 0.80 |
| H+E (blend=10) | 3.50 | 0.80 |

---

## Cross-Validation Strategies

### LeaveOneSeasonOut

Used by Models A, D, E. Each fold holds out one complete season for validation. The test season is excluded from all folds.

```
Fold 1: Train on 2020-2023, Val on 2019
Fold 2: Train on 2019,2021-2023, Val on 2020
...
Test:   Train on 2019-2023, Test on 2024
```

### ExpandingWindowSplit

Used by Models B, C, F, G. Growing training windows that respect temporal ordering.

```
Fold 1: Train on 2018, Val on 2019
Fold 2: Train on 2018-2019, Val on 2020
...
Test:   Train on 2018-2024, Test on 2025
```

Both strategies are implemented in `src/f1_predictor/features/splits.py`.

---

## Storage

All trained models and predictions are stored in GCS:

| Artifact | Path |
|----------|------|
| Trained models | `gs://f1-predictor-artifacts-jowin/data/raw/model/Model_{A-I}_*.pkl` |
| OOF predictions | `gs://f1-predictor-artifacts-jowin/data/training/model_{a-i}_*.parquet` |
| Field medians (H) | `gs://f1-predictor-artifacts-jowin/data/raw/model/field_medians.pkl` |
| Circuit defaults | `gs://f1-predictor-artifacts-jowin/data/raw/model/circuit_defaults.pkl` |

The Cloud Run container downloads these at startup when `F1_LOAD_FROM_GCS=true`.
