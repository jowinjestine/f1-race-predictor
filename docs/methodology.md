# Methodology

How the F1 Race Predictor turns historical telemetry into lap-by-lap race simulations.

## Problem Statement

Predict the finishing positions of all drivers in a Formula 1 race, given only qualifying results and circuit identity. The system must produce:

1. **Deterministic predictions** — a single predicted finishing order with lap-by-lap telemetry
2. **Probabilistic predictions** — position distributions with confidence intervals via Monte Carlo simulation
3. **Strategy recommendations** — optimal pit stop timing and compound selection for a target driver

## Data Pipeline

```
FastF1 (2018-2024)  ──┐
                      ├──▶  Race + Lap parquets  ──▶  Feature engineering  ──▶  Model training
Jolpica API (2025)  ──┘           │
                                  │
Open-Meteo (weather) ─────────────┘
```

### Sources

| Source | Coverage | Data |
|--------|----------|------|
| [FastF1](https://docs.fastf1.dev/) | 2018-2024 | Lap times, sector splits, tyre compounds, track status, weather telemetry |
| [Jolpica API](https://github.com/jolpica/jolpica-f1) | 2025 | Race results, qualifying, lap times, pit stops (no tyre data) |
| [Open-Meteo](https://open-meteo.com/) | All seasons | Race-day temperature, precipitation, wind speed |

### Data Volume

- **Race-level:** ~2,800 driver-race observations (140 races x 20 drivers)
- **Lap-level:** ~165,000 individual laps across 7 seasons
- **Circuits:** 30+ unique circuits with location-alias normalization

## Modelling Philosophy

### Why Nine Models?

A single model cannot simultaneously learn:
- **Lap-level dynamics** — how tyre degradation, pit stops, and track position interact over 50-78 laps
- **Race-level patterns** — how grid position, team form, and weather predict the final result
- **Temporal structure** — how a driver's pace trajectory evolves within a stint

The nine-model architecture separates these concerns into specialized layers:

| Layer | Models | What it learns |
|-------|--------|----------------|
| **Base predictors** | A, B, C | Lap-level position (with/without tyre data) and pre-race finishing position |
| **Stacking ensembles** | D, E | How to combine base predictions into a single race-level estimate |
| **Simulation models** | F, G, H, I | How to predict lap times for autoregressive simulation |

### Delta-Baseline Decomposition

The production simulator (Model H) doesn't predict absolute lap time ratios. Instead it decomposes:

```
lap_time_ratio = field_median_ratio(circuit, lap) + delta_ratio
                 ───────────────────────────────     ───────────
                 Historical baseline (lookup table)   Model prediction
```

**Why this helps:**
- The field median captures circuit-specific patterns (fuel load, tyre cliffs, safety car likelihood by lap)
- The model only needs to predict *deviations* from the field — a much smaller target range
- Reduces RMSE on final positions from 6.19 (Model F, absolute ratio) to 3.46 (Model H, delta ratio)

### Compound Pace Correction

LightGBM learned to predict lap pace from rolling averages, tyre life, and degradation rate — but assigned near-zero importance to the dry compound one-hot features (SOFT=2 splits, MEDIUM=2, HARD=8 out of ~3,600 total). The model absorbed compound effects into correlated features.

To restore compound sensitivity for strategy optimization, the simulator applies physics-informed corrections:

| Compound | Pace Offset | Degradation Rate (per lap after lap 5) |
|----------|------------|---------------------------------------|
| SOFT | -0.006 (faster) | 0.0006 (high) |
| MEDIUM | 0.000 (baseline) | 0.0003 (medium) |
| HARD | +0.004 (slower) | 0.0001 (low) |

On a 75-second qualifying lap, SOFT is ~0.45s/lap faster than MEDIUM initially, but degrades ~0.02s/lap faster after 5 laps — matching real F1 compound characteristics.

### H+E Ensemble

The production serving pipeline combines two models:

1. **Model H** runs a full lap-by-lap simulation, producing position trajectories and cumulative times
2. **Model E** takes H's trajectory as input, extracts statistical features (mean, std, last-5 average, range), and predicts a refined final position
3. Over the last N laps (`blend_laps`), H's positions are interpolated toward E's predictions

When Models A and B are loaded, E receives decorrelated inputs — A and B each predict positions from H's simulated lap data using different feature sets, giving E three independent signals instead of one.

## Cross-Validation Strategies

Temporal ordering is critical in sports prediction — future seasons cannot be used to predict past ones.

### LeaveOneSeasonOut (Models A, D, E)

Each fold holds out one complete season for validation. The test season is excluded from all folds.

```
Fold 1: Train 2020-2023  →  Val 2019
Fold 2: Train 2019,2021-2023  →  Val 2020
Fold 3: Train 2019-2020,2022-2023  →  Val 2021
Test:   Train 2019-2023  →  Test 2024
```

### ExpandingWindowSplit (Models B, C, F, G, H)

Growing training windows that strictly respect temporal ordering.

```
Fold 1: Train 2018  →  Val 2019
Fold 2: Train 2018-2019  →  Val 2020
...
Test:   Train 2018-2024  →  Test 2025
```

### Leakage Prevention

All aggregation features (rolling means, expanding averages, form metrics) use `shift(1)` to prevent look-ahead leakage. A feature at race R only uses data from races 1 to R-1.

## Monte Carlo Simulation

The `MonteCarloSimulator` runs N independent simulations (default 200) with Gaussian noise (sigma=0.01) injected into Model H's delta-ratio predictions. Each simulation produces slightly different trajectories.

**Aggregated outputs per driver:**
- Median position (most likely outcome)
- Mean position, standard deviation (volatility)
- 10th/25th/75th/90th percentiles (best-case to worst-case)
- DNF rate (fraction of simulations where the driver retired)

**Use cases:**
- Podium probability estimation
- Strategy risk comparison (1-stop vs 2-stop variance)
- Grid position safety analysis (low std = predictable outcome)

## DNF Prediction

Race-level DNF probability per driver is converted to a per-lap hazard rate:

```
hazard_rate = 1 - (1 - dnf_probability) ^ (1 / total_laps)
```

At each lap, the simulator samples `uniform(0, 1)` for each active driver. If the sample falls below the hazard rate, the driver retires. This produces:
- Realistic retirement timing (earlier retirements are possible but less likely per-lap)
- Correct aggregate DNF rates over Monte Carlo simulations
- Natural interaction with strategy — a DNF on lap 30 has different position impact than lap 70

## Strategy Optimization

### Candidate Generation

The optimizer generates 20-50 candidate strategies by varying:
1. **Pit lap timing** — default window +/- 1, 3, 5 laps
2. **Stop count** — 1-stop and 2-stop variants
3. **Compound sequences** — all permutations of {SOFT, MEDIUM, HARD}

Constraints:
- Minimum stint length: 8 laps (tyre warm-up)
- Maximum stint length: total_laps - 8
- At least 2 distinct dry compounds (F1 regulation)

### Two-Phase Evaluation

1. **Phase 1 (deterministic):** Run Model H simulation for all candidates (~1-2 seconds). Other drivers use circuit-default strategies. Rank by (position, total_time).

2. **Phase 2 (optional Monte Carlo):** Run N noisy simulations on the top 5 strategies (~10-15 seconds). Adds `position_mean` and `position_std` for confidence intervals.

## Evaluation

### Metrics

| Metric | What it measures |
|--------|-----------------|
| RMSE | Average position error magnitude |
| Spearman | Rank correlation (does the predicted order match?) |
| Within-3 | Fraction of drivers predicted within 3 positions of actual |

### Held-Out Results (4 races, 2024 season)

| Configuration | RMSE | Spearman |
|---------------|------|----------|
| Model H only (blend=0) | 3.46 | 0.82 |
| Model E standalone | 2.60 | 0.93 |
| H+E ensemble (blend=10) | 3.50 | 0.80 |

Model H excels at trajectory realism (lap-by-lap telemetry), while Model E excels at final position accuracy. The ensemble combines both strengths.
