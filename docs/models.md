# Model Documentation

Nine models trained in sequence, each building on outputs from earlier stages. Models A-C are base predictors, D-E are stacking ensembles, and F-I power autoregressive race simulators.

## Model Summary

| Model | Algorithm | Target | Data | Test RMSE | Production Role |
|-------|-----------|--------|------|-----------|-----------------|
| **A** | LightGBM GOSS | `position` (per lap) | 139K laps (2019-2024) | 2.71 | Feeds decorrelated signal to Model E |
| **B** | LightGBM GOSS | `position` (per lap) | 165K laps (2018-2025) | 5.29 | Feeds decorrelated signal to Model E |
| **C** | XGBoost Conservative | `finish_position` | 3.5K race-drivers (2018-2025) | 4.21 | Not used directly (stacking input only) |
| **D** | LightGBM shallow (meta) | `finish_position` | 2K race-drivers | 3.18 | Not used — superseded by Model E |
| **E** | LightGBM shallow (meta) | `finish_position` | 2K race-drivers | 2.60 | Final position refinement in H+E ensemble |
| **F** | LightGBM Shallow | `lap_time_ratio` | 79K laps (2019-2024) | 0.021 | Not used — no delta correction, poor simulation |
| **G** | LSTM Bidir | `lap_time_ratio` | 79K sequences | 0.022 | Not used — all variants collapse to identical simulation |
| **H** | LightGBM GOSS Delta | `delta_ratio` | 79K laps (2019-2024) | 3.46* | Core lap-by-lap simulator |
| **I** | LightGBM Quantile | 5 quantiles of `lap_time_ratio` | 79K laps | 6.99* | Not used — collapses autoregressively |

*H and I: RMSE measured on final race positions across 4 held-out 2024 races, not per-lap.

---

## Model A — Lap + Tyre

**Notebook:** `notebooks/05a_model_A_training.ipynb`
**Feature module:** `src/f1_predictor/features/lap_features.py` (`build_lap_tyre_features`)
**Production role:** Runs inside the H+E ensemble to provide tyre-aware position predictions as decorrelated input for Model E.

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

### All Variants Tested (12)

**Round 1 — Screening (default hyperparameters):**

| Rank | Variant | CV RMSE | CV MAE | Notes |
|------|---------|---------|--------|-------|
| 1 | LightGBM | 3.034 | 2.232 | |
| 2 | LightGBM_Deep | 3.039 | 2.236 | |
| 3 | LightGBM_GOSS | 3.041 | 2.237 | |
| 4 | LightGBM_DART | 3.071 | 2.295 | |
| 5 | LightGBM_Shallow | 3.100 | 2.340 | |
| 6 | XGBoost_Conservative | 3.113 | 2.351 | |
| 7 | XGBoost_Deep | 3.141 | 2.278 | |
| 8 | XGBoost_DART | 3.208 | 2.359 | Eliminated |
| 9 | XGBoost | 3.208 | 2.359 | Eliminated (identical to DART) |
| 10 | FT_Transformer | 3.241 | 2.422 | Eliminated |
| 11 | GRU_2layer | 3.587 | 2.744 | Eliminated |
| 12 | XGBoost_Linear | 4.835 | 4.115 | Eliminated |

**Round 3 — Final tuning (Optuna, 15 trials):**

| Variant | Tuned CV RMSE |
|---------|---------------|
| LightGBM_GOSS | 2.983 |
| LightGBM_DART | 2.986 |
| LightGBM_Shallow | 3.002 |
| LightGBM_Deep | 3.002 |
| LightGBM | 3.002 |

**Final Test Set (2024 season):**

| Variant | Test RMSE | Test MAE | Overfit Gap |
|---------|-----------|----------|-------------|
| **LightGBM** | **2.712** | 2.017 | -0.290 |
| LightGBM_Shallow | 2.712 | 2.017 | -0.290 |
| LightGBM_Deep | 2.712 | 2.017 | -0.290 |
| LightGBM_GOSS | 2.721 | 2.023 | -0.262 |
| LightGBM_DART | 2.773 | 2.058 | -0.213 |

**Selected:** LightGBM (standard), though effectively a 3-way tie with Shallow and Deep. LightGBM_GOSS was auto-selected for the ensemble because it had the best validation RMSE on last-lap aggregation.

**Key findings:**
- Deep learning models (GRU, FT-Transformer) were eliminated in Round 1 — tabular GBMs dominated
- XGBoost and XGBoost_DART produced identical results (DART dropout had no effect at default params)
- Negative overfit gap (-0.29) means 2024 was easier to predict than the CV average
- OOF predictions saved for Models D and E stacking

---

## Model B — Lap, No Tyre

**Notebook:** `notebooks/05b_model_B_training.ipynb`
**Feature module:** `src/f1_predictor/features/lap_features.py` (`build_lap_notyre_features`)
**Production role:** Runs inside the H+E ensemble to provide pace-based position predictions (without tyre data) as a second decorrelated input for Model E.

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

### All Variants Tested (12)

**Round 1 — Screening:**

| Rank | Variant | CV RMSE | Notes |
|------|---------|---------|-------|
| 1 | LightGBM_DART | 3.201 | |
| 2 | LightGBM_Shallow | 3.206 | |
| 3 | XGBoost_Conservative | 3.216 | |
| 4 | LightGBM | 3.234 | |
| 5 | LightGBM_Deep | 3.245 | |
| 6 | LightGBM_GOSS | 3.249 | |
| 7 | XGBoost_Deep | 3.380 | |
| 8 | XGBoost | 3.499 | Eliminated |
| 9 | XGBoost_DART | 3.499 | Eliminated (identical to XGBoost) |
| 10 | FT_Transformer | 3.513 | Eliminated |
| 11 | GRU_2layer | 3.721 | Eliminated (high variance, std=0.43) |
| 12 | XGBoost_Linear | 4.897 | Eliminated |

**Final Test Set (2025 season, 2,400 rows):**

| Variant | Test RMSE | Test MAE | Overfit Gap |
|---------|-----------|----------|-------------|
| **LightGBM_DART** | **5.286** | 4.025 | +2.12 |
| LightGBM | 5.410 | 4.122 | +2.24 |
| LightGBM_Shallow | 5.410 | 4.122 | +2.24 |
| LightGBM_Deep | 5.411 | 4.123 | +2.24 |
| LightGBM_GOSS | 5.426 | 4.133 | +2.27 |

**Selected:** LightGBM_DART (lowest test RMSE). LightGBM_GOSS was auto-selected for the ensemble based on validation RMSE of the last-lap aggregation.

**Why the high test RMSE:** The 2025 test set is dramatically harder than historical seasons — an overfit gap of +2.1 (test is 2.1 RMSE worse than CV). This is the only model where test was much worse than CV, reflecting distribution shift in the 2025 season (new regulations, team reshuffling, partial-season data).

---

## Model C — Pre-Race Features

**Notebook:** `notebooks/05c_model_C_training.ipynb`
**Feature module:** `src/f1_predictor/features/race_features.py` (`build_race_features`)
**Production role:** Not used directly in the API. Provides OOF predictions as stacking input for Models D and E.

Predicts race-level finish position from pre-race context only — no lap-level information. One prediction per driver per race.

### Features (15)

| Group | Features |
|-------|----------|
| **Grid** | `grid_position`, `quali_delta_to_pole`, `best_quali_sec` |
| **Driver form** | `position_trend`, `points_last_3`, `driver_circuit_avg_finish`, `driver_circuit_races` |
| **Team** | `team_avg_finish_last_3`, `team_points_cumulative_season`, `quali_position_vs_teammate`, `dnf_rate_season` |
| **Circuit** | `circuit_avg_dnf_rate` |
| **Weather** | `weather_temp_max`, `weather_wind_max_kph`, `weather_precip_mm` |

### All Variants Tested (11)

**Round 1 — Screening:**

| Rank | Variant | CV RMSE | Notes |
|------|---------|---------|-------|
| 1 | XGBoost_Linear | 4.549 | Won CV but lost on test |
| 2 | LightGBM_Shallow | 4.752 | |
| 3 | XGBoost_Conservative | 4.812 | |
| 4 | LightGBM_DART | 4.838 | |
| 5 | LightGBM | 5.003 | |
| 6 | LightGBM_Deep | 5.036 | |
| 7 | LightGBM_GOSS | 5.087 | |
| 8 | XGBoost_Deep | 5.096 | Eliminated |
| 9 | XGBoost | 5.134 | Eliminated |
| 10 | XGBoost_DART | 5.134 | Eliminated |
| 11 | MLP_3layer | 5.184 | Eliminated (very high variance, std=0.74) |

**Final Test Set (2025 season, 479 rows):**

| Variant | Test RMSE | Test MAE | Overfit Gap |
|---------|-----------|----------|-------------|
| **XGBoost_Conservative** | **4.214** | 3.289 | -0.345 |
| LightGBM_Deep | 4.251 | 3.312 | -0.328 |
| LightGBM | 4.251 | 3.312 | -0.328 |
| LightGBM_GOSS | 4.254 | 3.351 | -0.311 |
| XGBoost_Linear | 4.357 | 3.394 | -0.162 |

**Selected:** XGBoost_Conservative. Notably, XGBoost_Linear won CV (RMSE 4.52) but lost on test (4.36) — the simpler linear model overfit to CV while the conservative tree generalized better.

**Why not used directly:** Model C predicts from pre-race features only. Its RMSE (4.21) is substantially worse than the lap-level models. Its value is as a stacking input — it captures pre-race context (team form, weather, circuit history) that lap-level models miss.

---

## Model D — All-Combinations Stacking

**Notebook:** `notebooks/05d_model_D_stacking.ipynb`
**Production role:** Not used — superseded by Model E which achieves 20% lower RMSE.

Exhaustive ensemble of all A x B x C variant combinations using a two-phase tournament.

### Architecture

1. **Phase 1 — RidgeCV screening:** Test all 300 combinations (5 A-variants x 5 B-variants x 12 C-variants) with RidgeCV. Select top 20.
2. **Phase 2 — Tournament:** Train 5 meta-learner types on each top-20 combination with Optuna. All 20 selected **LightGBM_shallow** as best meta-learner.

### Meta-Features (3)

| Feature | Description |
|---------|-------------|
| `pred_A` | Model A's OOF position prediction (aggregated to race level — last lap only) |
| `pred_B` | Model B's OOF position prediction (aggregated to race level — last lap only) |
| `pred_C` | Model C's OOF finish position prediction |

### Meta-Learners Tested

| Meta-Learner | Result |
|-------------|--------|
| LightGBM_shallow | Selected in all 20 combinations |
| RidgeCV | Screening only |
| LassoCV | Never won |
| ElasticNetCV | Never won |
| XGBoost_shallow | Never won |

### Top 5 Results (test on 2023 season)

| Combination (A / B / C) | Test RMSE | Test MAE | Overfit Gap |
|-------------------------|-----------|----------|-------------|
| **LightGBM_Deep / LightGBM_GOSS / ExtraTrees** | **3.180** | 2.264 | -0.162 |
| LightGBM / LightGBM_GOSS / ExtraTrees | 3.181 | 2.265 | -0.160 |
| LightGBM_Shallow / LightGBM_GOSS / ExtraTrees | 3.182 | 2.265 | -0.160 |
| LightGBM_DART / LightGBM_GOSS / ExtraTrees | 3.205 | 2.285 | -0.134 |
| LightGBM / LightGBM / ExtraTrees | 3.212 | 2.289 | -0.133 |

**Key findings:**
- ExtraTrees was the universal C-variant winner — all top 20 combinations used it
- A and B variant choice barely mattered — top 10 spanned only 0.01 RMSE
- Negative overfit gap (-0.16) shows stacking regularises well
- **Why superseded:** Model D uses only 3 features (one prediction per base model). Model E uses 13 features (6 aggregation stats from A, 4 from B, plus C_pred and qualifying context), capturing trajectory information lost in D's single-point summaries.

---

## Model E — Rich Feature Stacking

**Notebook:** `notebooks/05e_model_E_rich_stacking.ipynb`
**Production role:** Final position refinement layer in the H+E ensemble. Runs in production on every simulation request when `blend_laps > 0`.

Improved stacking that preserves lap-level richness through aggregation features.

### Meta-Features (13)

| Source | Features | Description |
|--------|----------|-------------|
| **Model A** (6) | `A_last`, `A_mean`, `A_std`, `A_min`, `A_range`, `A_last5` | Lap-level position trajectory statistics |
| **Model B** (4) | `B_last`, `B_mean`, `B_std`, `B_last5` | Lap-level position trajectory statistics |
| **Model C** (1) | `C_pred` | Pre-race finish position prediction |
| **Pre-race** (2) | `grid_position`, `quali_delta_to_pole` | Direct qualifying context |

### Meta-Learners Tested

| Rank | Meta-Learner | CV RMSE | Notes |
|------|-------------|---------|-------|
| 1 | **LightGBM_shallow** | 2.225 | Selected |
| 2 | LassoCV | 2.394 | |
| 3 | RidgeCV | 2.396 | |
| 4 | ElasticNetCV | 2.411 | |
| 5 | XGBoost_shallow | 2.416 | |

### Final Test Set (2023 season, 430 rows)

| Meta-Learner | Test RMSE | R^2 | Spearman | Within-3 | Overfit Gap |
|-------------|-----------|------|----------|----------|-------------|
| **LightGBM_shallow** | **2.603** | **0.764** | **0.928** | **77.4%** | +0.378 |
| LassoCV | 2.672 | 0.751 | 0.928 | 74.7% | +0.278 |
| RidgeCV | 2.675 | 0.751 | 0.926 | 74.9% | +0.282 |

### Cross-Model Comparison (all on 2023 test)

| Model | Test RMSE | R^2 | Spearman | Within-3 |
|-------|-----------|------|----------|----------|
| Model A (best) | 3.103 | 0.674 | 0.904 | 70.3% |
| Model D (best) | 3.180 | 0.695 | 0.848 | 73.6% |
| **Model E** | **2.603** | **0.764** | **0.928** | **77.4%** |

**Why E beats D by 20%:** The 6 aggregation statistics from Model A (std, range, last-5 mean) capture how volatile a driver's trajectory was — a driver who held P3 with low variance is more likely to finish P3 than one who oscillated between P1 and P6, even if both had the same last-lap prediction.

---

## Model F — Lap Time Simulation

**Notebook:** `notebooks/05f_model_F_lap_simulation.ipynb`
**Feature module:** `src/f1_predictor/features/simulation_features.py` (`build_simulation_training_data`)
**Production role:** Not used — replaced by Model H which adds delta correction and achieves far better simulation accuracy.

Predicts `lap_time_ratio = lap_time / best_quali_time` for autoregressive race simulation. This is the foundation model that established the feature set used by H and I.

### Features (25)

| Group | Features |
|-------|----------|
| **Static** (5) | `grid_position`, `best_quali_sec`, `circuit_street`, `circuit_hybrid`, `circuit_permanent` |
| **Deterministic** (10) | `lap_number`, `race_progress_pct`, 5x compound one-hot, `tire_life`, `stint` |
| **Pit flags** (2) | `is_pit_in_lap`, `is_pit_out_lap` |
| **Feedback** (6) | `pit_stop_count`, `laps_since_last_pit`, `lap_time_rolling_3`, `lap_time_rolling_5`, `degradation_rate`, `gap_to_leader` |
| **Position** (2) | `position`, `position_change_from_lap1` |
| **Safety** (1) | `is_caution` |

### All Variants Tested (10)

**Round 1 — Screening:**

| Rank | Variant | CV RMSE | Notes |
|------|---------|---------|-------|
| 1 | XGBoost_Conservative | 0.0584 | |
| 2 | LightGBM_Shallow | 0.0591 | |
| 3 | LightGBM_GOSS | 0.0599 | |
| 4 | LightGBM_Deep | 0.0610 | |
| 5 | LightGBM | 0.0612 | |
| 6 | XGBoost_DART | 0.0638 | |
| 7 | XGBoost | 0.0638 | |
| 8 | XGBoost_Deep | 0.0647 | Eliminated |
| 9 | LightGBM_DART | 0.0671 | Eliminated |
| 10 | XGBoost_Linear | 0.0690 | Eliminated |

**Final Test Set (2024, per-lap):**

| Variant | Test RMSE | Overfit Gap |
|---------|-----------|-------------|
| **LightGBM_Shallow** | **0.0211** | -0.036 |
| LightGBM_Deep | 0.0211 | -0.036 |
| LightGBM_GOSS | 0.0212 | -0.035 |
| XGBoost_Conservative | 0.0213 | -0.035 |
| XGBoost | 0.0213 | -0.035 |

**Simulation Evaluation (4 held-out 2024 races, 79 predictions):**

| Metric | Value |
|--------|-------|
| Position RMSE | **6.19** |
| Spearman | 0.42 |
| Within-3 | 43.0% |

**Why not used:** Despite good per-lap RMSE (0.021), Model F produces poor race simulations (position RMSE 6.19, Spearman 0.42). The absolute `lap_time_ratio` target includes circuit-specific baseline variance that accumulates over 50-78 autoregressive laps. Model H's delta decomposition subtracts this baseline, reducing the prediction range and producing much better simulations.

---

## Model G — Temporal Sequence

**Notebook:** `notebooks/05g_model_G_temporal.ipynb`
**Feature module:** `src/f1_predictor/features/sequence_features.py` (`build_sequence_training_data`)
**Simulator:** `src/f1_predictor/simulation/sequence_simulator.py` — `SequenceRaceSimulator`
**Production role:** Not used — all variants collapse to identical simulation results.

Temporal variant of Model F using sequence models that consume a sliding window of past laps.

### Architecture

- **Input:** 3D tensor `(n_drivers, window_size, n_features+1)` — the `+1` is a `seq_valid_mask` channel
- **Left-padding:** Early laps with fewer than `window_size` history are zero-padded on the left

### All Variants Tested (10)

**Round 1 — Screening:**

| Rank | Variant | CV RMSE | Notes |
|------|---------|---------|-------|
| 1 | SeqGRU_Deep | 0.0602 | |
| 2 | SeqLSTM_Deep | 0.0613 | |
| 3 | SeqGRU_Attn | 0.0622 | |
| 4 | SeqLSTM_Bidir | 0.0624 | |
| 5 | SeqTransformer | 0.0638 | |
| 6 | SeqGRU_Bidir | 0.0649 | |
| 7 | SeqGRU_Shallow | 0.0687 | |
| 8 | SeqLSTM_Shallow | 0.0760 | Eliminated |
| 9 | SeqTCN | 0.0791 | Eliminated |
| 10 | SeqCNN1D | 0.1085 | Eliminated |

**Final Test Set (2024, per-lap):**

| Variant | Test RMSE | Window Size |
|---------|-----------|-------------|
| **SeqLSTM_Bidir** | **0.0219** | 7 |
| SeqGRU_Attn | 0.0226 | 9 |
| SeqLSTM_Deep | 0.0231 | 3 |
| SeqGRU_Bidir | 0.0236 | 3 |
| SeqGRU_Deep | 0.0238 | 3 |

**Simulation Evaluation (4 held-out 2024 races):**

| Metric | Value |
|--------|-------|
| Position RMSE | **3.66** |
| Spearman | 0.80 |
| Within-3 | 77.2% |

**Why not used:** All 5 G variants produce **completely identical** simulation results (RMSE 3.661, Spearman 0.799 across the board). Despite different per-lap metrics and architectures (GRU vs LSTM, shallow vs deep, attention vs bidirectional), the autoregressive feedback loop causes all models to converge to the same trajectory. The temporal context adds no discriminative power beyond what the tabular features' rolling aggregations already provide. Model H achieves better accuracy (RMSE 3.46 vs 3.66) with a simpler architecture.

---

## Model H — Delta + Monte Carlo

**Notebook:** `notebooks/05h_model_H_delta_mc.ipynb`
**Feature module:** `src/f1_predictor/features/delta_features.py` (`build_delta_training_data`, `build_field_median_curves`)
**Simulator:** `src/f1_predictor/simulation/delta_simulator.py` — `DeltaRaceSimulator`, `MonteCarloSimulator`
**Production role:** Core simulator — powers all three API endpoints (`/simulate`, `/simulate/monte-carlo`, `/optimize-strategy`).

Predicts `delta_ratio` (deviation from historical field median) instead of absolute `lap_time_ratio`.

### Delta Decomposition

```
lap_time_ratio = field_median_ratio(circuit, lap) + delta_ratio
                 ───────────────────────────────     ───────────
                 Historical baseline (lookup table)   Model prediction
```

The `field_median_ratio` is a per-circuit, per-lap median from 2019-2024 historical data. This decomposition reduces the prediction range and lets the model focus on driver-specific deviations rather than circuit-level patterns.

### All Variants Tested (9)

**Round 1 — Screening (includes 3 deep learning variants):**

| Rank | Variant | CV RMSE | Notes |
|------|---------|---------|-------|
| 1 | LightGBM_Shallow_Delta | 0.0648 | |
| 2 | LightGBM_Delta | 0.0648 | |
| 3 | LightGBM_GOSS_Delta | 0.0650 | |
| 4 | GRU_Delta | 0.0665 | |
| 5 | XGBoost_DART_Delta | 0.0702 | |
| 6 | XGBoost_Delta | 0.0702 | |
| 7 | MLP_Delta | 0.0709 | |
| 8 | XGBoost_Deep_Delta | 0.0718 | Eliminated |
| 9 | FTTransformer_Delta | 0.0787 | Eliminated (high variance, std=0.026) |

**Final Test Set (2024, per-lap delta):**

| Variant | Test RMSE | Overfit Gap |
|---------|-----------|-------------|
| **LightGBM_GOSS_Delta** | **0.0220** | -0.041 |
| XGBoost_DART_Delta | 0.0221 | -0.042 |
| XGBoost_Delta | 0.0226 | -0.040 |
| LightGBM_Shallow_Delta | 0.0228 | -0.041 |
| LightGBM_Delta | 0.0228 | -0.041 |

### Simulation Evaluation (4 held-out 2024 races)

**Single-run (deterministic):**

| Variant | Position RMSE | Spearman | Within-3 | Within-5 |
|---------|--------------|----------|----------|----------|
| H_LightGBM_Delta | 3.462 | 0.820 | 75.9% | 88.6% |
| H_LightGBM_Shallow_Delta | 3.462 | 0.820 | 75.9% | 88.6% |
| **H_LightGBM_GOSS_Delta** | **3.499** | **0.816** | **75.9%** | **88.6%** |
| H_XGBoost_Delta | 3.703 | 0.793 | 63.3% | 84.8% |
| H_XGBoost_DART_Delta | 3.770 | 0.785 | 69.6% | 84.8% |

**Monte Carlo (N=200, using OOF residual std=0.060 as noise):**

| Metric | Value |
|--------|-------|
| Position RMSE | 3.634 |
| Spearman | 0.805 |
| Coverage@80% | 87.3% |
| Coverage@50% | 67.1% |
| Sharpness@80% | 9.47 |

**Per-race breakdown:**

| Race | RMSE | Spearman |
|------|------|----------|
| Bahrain GP | 2.74 | 0.887 |
| Emilia Romagna GP | 2.70 | 0.888 |
| Hungarian GP | 3.62 | 0.803 |
| Mexico City GP | 4.56 | 0.687 |

**Selected:** LightGBM_GOSS_Delta — best per-lap test RMSE, near-best simulation RMSE, and best MC calibration.

**Why H is the best simulator:** The delta decomposition cancels systematic circuit bias, reducing autoregressive error accumulation. H achieves position RMSE 3.46 vs F's 6.19 (+44% improvement) and G's 3.66 (+5% improvement), while also being the most robust on the hardest race (Mexico City: 4.56 vs G's 5.11).

---

## Model I — Quantile Regression

**Notebook:** `notebooks/05i_model_I_quantile.ipynb`
**Simulator:** `src/f1_predictor/simulation/quantile_simulator.py` — `QuantileRaceSimulator`
**Production role:** Not used — collapses catastrophically during autoregressive simulation despite having the best per-lap accuracy.

Multi-output quantile model that predicts the full distribution of `lap_time_ratio` at each lap.

### Architecture

- **Output:** 5 quantiles per prediction — q10, q25, q50, q75, q90
- **Sampling:** At each lap, draw `u ~ Uniform(0, 1)` per driver and interpolate along the quantile curve
- **Goal:** Probabilistic simulation without external noise injection

### All Variants Tested (8)

**Round 1 — Screening:**

| Rank | Variant | CV RMSE (q50) | Notes |
|------|---------|---------------|-------|
| 1 | LightGBM_Quantile | 0.0600 | |
| 2 | FTTransformer_Quantile | 0.0602 | |
| 3 | GRU_MultiQuantile | 0.0621 | |
| 4 | MDN_MLP | 0.0643 | Mixture Density Network |
| 5 | XGBoost_Quantile | 0.0644 | |
| 6 | MDN_GRU | 0.0652 | Mixture Density Network |
| 7 | DeepEnsemble | 0.0669 | 5 independent networks |
| 8 | MLP_MultiQuantile | 0.0669 | Eliminated |

**Final Test Set (2024, per-lap, median prediction):**

| Variant | Test RMSE | Overfit Gap |
|---------|-----------|-------------|
| **LightGBM_Quantile** | **0.0178** | -0.038 |
| XGBoost_Quantile | 0.0209 | -0.037 |
| GRU_MultiQuantile | 0.0215 | -0.038 |
| FTTransformer_Quantile | 0.0322 | -0.027 |
| MDN_GRU | 0.0375 | -0.026 |

**Note:** LightGBM_Quantile's per-lap RMSE (0.0178) is the **best of any model across all notebooks** — better than F (0.0211) and H (0.0220).

### Simulation Evaluation (4 held-out 2024 races)

**Median (deterministic) simulation:**

| Metric | LightGBM_Quantile |
|--------|-------------------|
| Position RMSE | **6.99** |
| Spearman | 0.26 |
| Within-3 | 40.5% |

**Quantile MC simulation (N=200):**

| Metric | Value |
|--------|-------|
| Position RMSE | 6.82 |
| Spearman | 0.16 |
| Coverage@80% | 54.4% |
| Coverage@50% | 35.4% |

**One surprise from notebook 08:** FTTransformer_Quantile — which had mediocre per-lap RMSE (0.0322) — somehow achieved position RMSE 3.59 and Spearman 0.81 in simulation, nearly matching Model H. This suggests the FT-Transformer's internal representations may be more stable under autoregressive feedback, but it wasn't explored further.

**Why not used:** Despite the best per-lap accuracy of any model, I produces the worst simulation results. The quantile sampling introduces noise that compounds over 50-70 autoregressive laps — small per-lap errors accumulate into large position divergences. The MC version is even worse (Spearman drops from 0.26 to 0.16). Model H's delta approach + external noise injection produces both better accuracy and better-calibrated uncertainty.

---

## H+E Ensemble (Production)

**Simulator:** `src/f1_predictor/simulation/ensemble_simulator.py` — `EnsembleSimulator`

The production serving ensemble combines Model H's lap-by-lap simulation with Model E's final-position refinement.

### Architecture

1. Run Model H's `DeltaRaceSimulator` for a full race
2. Reconstruct Model A (9 features) and Model B (8 features) inputs from H's simulated lap data
3. Run A and B independently to get per-lap position predictions
4. Aggregate into 10 meta-features per driver (A: last, mean, std, min, range, last5; B: last, mean, std, last5)
5. Add `C_pred` (H's final position), `grid_position`, `quali_delta_to_pole` → 13 features
6. Model E predicts final positions
7. Blend H's trajectory toward E's predictions over the last N laps

### Decorrelated Meta-Features

Models A and B use different feature subsets:

| Source | Unique Features | What it captures |
|--------|----------------|-----------------|
| Model A | degradation_rate, compound_pace_delta, tire_life | Tyre-aware position trajectory |
| Model B | laps_since_last_pit, lap_time_rolling_3 | Pace-based position trajectory |
| Model H | (simulation output) | Delta-baseline position |

This gives Model E three independent signals instead of the degenerate case where A_last = B_last = C_pred.

### Compound Pace Corrections

Model H's LightGBM assigns near-zero importance to dry compound one-hots (SOFT=2 splits, MEDIUM=2, HARD=8 out of ~3,600 total). The simulator applies physics-informed corrections:

| Compound | Pace Offset | Degradation Rate (per lap after lap 5) |
|----------|------------|---------------------------------------|
| SOFT | -0.006 (faster) | 0.0006 (high wear) |
| MEDIUM | 0.000 (baseline) | 0.0003 (medium wear) |
| HARD | +0.004 (slower) | 0.0001 (low wear) |

### DNF/Retirement Prediction

Drivers can retire mid-race based on a per-driver DNF probability. The race-level probability is converted to a per-lap hazard rate: `h = 1 - (1-p)^(1/N)`. At each lap, active drivers are sampled for retirement.

### Validation Results (4 held-out 2024 races)

| Configuration | RMSE | Spearman |
|---------------|------|----------|
| H only (blend=0) | 3.46 | 0.82 |
| E standalone | 2.60 | 0.93 |
| H+E (blend=10) | 3.50 | 0.80 |

---

## Head-to-Head Simulation Comparison (Notebook 08)

All simulators evaluated on the same 4 held-out 2024 races (Bahrain, Emilia Romagna, Hungary, Mexico City), 79 driver-race predictions.

### Best Variant per Model Type

| Model | Best Variant | Position RMSE | Spearman | Within-3 | Within-5 |
|-------|-------------|--------------|----------|----------|----------|
| **H** | LightGBM_Delta | **3.46** | **0.82** | 75.9% | **88.6%** |
| G | SeqLSTM_Bidir | 3.66 | 0.80 | **77.2%** | 87.3% |
| I | FTTransformer_Quantile* | 3.59 | 0.81 | 73.4% | 89.9% |
| F | LightGBM_Shallow | 6.19 | 0.42 | 43.0% | 62.0% |

*FTTransformer_Quantile was an anomaly — the primary I variant (LightGBM_Quantile) scored 6.99 RMSE.

### Why Each Model Was or Wasn't Used

| Model | Used? | Reason |
|-------|-------|--------|
| **A** | Yes | Provides tyre-aware position predictions as decorrelated input for E |
| **B** | Yes | Provides pace-based position predictions as decorrelated input for E |
| C | No | Only used as stacking input for D/E — too weak standalone (RMSE 4.21) |
| D | No | Superseded by E (RMSE 3.18 vs 2.60) — uses only 3 features vs E's 13 |
| **E** | Yes | Best position accuracy (RMSE 2.60, Spearman 0.93) — refines H's output |
| F | No | No delta correction — position RMSE 6.19, Spearman 0.42 |
| G | No | All variants collapse to identical results — no advantage over H |
| **H** | Yes | Best simulator (RMSE 3.46, Spearman 0.82) — delta decomposition works |
| I | No | Best per-lap RMSE but collapses autoregressively (position RMSE 6.99) |

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

Used by Models B, C, F, G, H, I. Growing training windows that respect temporal ordering.

```
Fold 1: Train on 2019, Val on 2020
Fold 2: Train on 2019-2020, Val on 2021
...
Test:   Train on 2019-2023, Test on 2024
```

Both strategies are implemented in `src/f1_predictor/features/splits.py`.

---

## Key Findings Across All Notebooks

1. **LightGBM dominates:** Won or tied for best in every model. XGBoost was consistently eliminated early. The gap was ~0.1 RMSE in screening, narrowing to ~0.01 after tuning.

2. **Deep learning consistently underperformed tabular GBMs:** GRU, LSTM, FT-Transformer, and MLP were eliminated in Round 1 for every model where they competed against GBMs. The dataset size (~80-165K rows) is in tabular ML's sweet spot, not deep learning's.

3. **Best per-lap accuracy ≠ best simulation accuracy:** Model I had the best per-lap RMSE (0.0178) but the worst simulation RMSE (6.99). Error accumulation over 50-78 autoregressive steps amplifies small biases that don't appear in single-step evaluation.

4. **Delta decomposition is the key innovation:** Subtracting the field median baseline reduced simulation RMSE from 6.19 (F) to 3.46 (H) — a 44% improvement — by removing circuit-level variance from the prediction target.

5. **Rich aggregation features are worth 20%:** Model E's 13 features (including std, range, last-5 stats from A/B trajectories) beat Model D's 3 features by 20% on RMSE, showing that trajectory shape information matters, not just the final prediction.

6. **2024 was an "easy" test season:** All models except B showed negative overfit gaps (test better than CV), suggesting 2024 race outcomes were more predictable than average.

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
