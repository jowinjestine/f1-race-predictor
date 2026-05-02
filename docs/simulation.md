# Simulation Engine

The simulation engine runs lap-by-lap autoregressive race predictions. It maintains per-driver state, predicts lap times using Model H, handles pit stops, tyre degradation, and DNF retirements.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    EnsembleSimulator                      │
│                                                          │
│  ┌─────────────────────┐    ┌─────────────────────────┐  │
│  │  DeltaRaceSimulator  │    │     Model E (stacker)   │  │
│  │  (Model H)           │───▶│     13 meta-features    │  │
│  │                      │    │     Final position       │  │
│  │  Lap-by-lap delta    │    │     refinement           │  │
│  │  ratio prediction    │    └─────────────────────────┘  │
│  │  + field median      │              │                  │
│  │  baseline            │              ▼                  │
│  └─────────────────────┘    Blend H positions → E preds  │
│            │                over last N laps              │
│            ▼                                              │
│  ┌─────────────────────┐                                  │
│  │ MonteCarloSimulator  │                                  │
│  │ N runs with noise    │                                  │
│  │ + DNF sampling       │                                  │
│  └─────────────────────┘                                  │
└──────────────────────────────────────────────────────────┘
```

## Components

### DriverState

Per-driver mutable state maintained across laps.

| Field | Type | Description |
|-------|------|-------------|
| `driver` | str | Three-letter abbreviation |
| `grid_position` | int | Starting position |
| `best_quali_sec` | float | Best qualifying time — pace baseline |
| `compound` | str | Current tyre compound |
| `tire_life` | int | Laps on current tyre set |
| `stint` | int | Current stint number (1-indexed) |
| `pit_stop_count` | int | Total pit stops made |
| `laps_since_last_pit` | int | Laps since most recent pit |
| `cum_time` | float | Cumulative race time in seconds |
| `position` | int | Current race position |
| `lap_times` | list | All lap times recorded |
| `stint_times` | list | Lap times within current stint |
| `stint_lives` | list | Tyre ages within current stint |
| `strategy` | list | Pit strategy: list of (compound, pit_on_lap) |
| `current_stint_idx` | int | Index into strategy list |
| `is_retired` | bool | Whether driver has DNF'd |
| `retired_on_lap` | int? | Lap number of retirement |
| `retirement_status` | str | "Finished" or "DNF" |

### DeltaRaceSimulator (Model H)

The production simulator. Predicts `delta_ratio` — how much faster or slower a driver is than the historical field median at each lap.

**Lap-by-lap loop:**

```
For each lap 1..total_laps:
  1. Identify pit-in drivers (strategy says pit this lap)
  2. Identify pit-out drivers (pitted on previous lap)
  3. Build feature matrix (25 features per driver)
  4. Predict delta_ratio with Model H (LightGBM)
  5. Add field_median baseline: ratio = delta + baseline
  6. Apply compound pace offset and degradation
  7. Clamp ratios by scenario
  8. Convert to lap time: lap_time = ratio × best_quali_sec
  9. Update cumulative time, tyre state
  10. Rank by cumulative time → positions
  11. Record LapRecord telemetry
  12. Execute pit stops (reset tyre, advance strategy)
  13. Sample DNF retirements (if enabled)
```

**Field median baseline:**

Pre-computed per-circuit, per-lap median `lap_time_ratio` from 2019-2024 historical data. Stored as a lookup table: `{"Monaco Grand Prix": {1: 1.085, 2: 1.062, ...}}`.

**Compound corrections:**

Model H has near-zero feature importance on dry compound one-hots, so the simulator applies physics-informed corrections:

| Compound | Pace Offset | Degradation (per lap after lap 5) |
|----------|------------|----------------------------------|
| SOFT | -0.006 | 0.0006 |
| MEDIUM | 0.000 | 0.0003 |
| HARD | +0.004 | 0.0001 |

**Clamping ranges:**

| Scenario | Min Ratio | Max Ratio |
|----------|-----------|-----------|
| Normal lap | 1.01 | 1.15 |
| Pit-in lap | 1.10 | 1.50 |
| Pit-out lap | 1.03 | 1.25 |

### Feature Construction

The simulator builds 25 features per driver per lap:

| Group | Features | Source |
|-------|----------|--------|
| **Static** | grid_position, best_quali_sec, circuit_street/hybrid/permanent | Driver input + circuit classification |
| **Temporal** | lap_number, race_progress_pct | Lap counter |
| **Compound** | compound_SOFT/MEDIUM/HARD/INTERMEDIATE/WET (one-hot) | Current tyre state |
| **Tyre** | tire_life, stint | DriverState |
| **Pit** | is_pit_in_lap, is_pit_out_lap, pit_stop_count, laps_since_last_pit | Pit detection logic |
| **Pace** | lap_time_rolling_3 (EWMA α=0.5), lap_time_rolling_5 (EWMA α=0.4) | Rolling window over lap_times |
| **Degradation** | degradation_rate | OLS slope of stint lap times vs tyre life (requires 3+ laps) |
| **Position** | gap_to_leader, position, position_change_from_lap1 | Cumulative time ranking |
| **Safety** | is_caution | Always 0 (no safety car simulation) |

### MonteCarloSimulator

Wraps any `RaceSimulator` and runs N independent simulations with noise.

1. Creates a `NoisyModelWrapper` that adds `Normal(0, 0.01)` noise to predictions
2. Each simulation gets a unique random seed for both noise and DNF sampling
3. Aggregates position distributions: median, mean, std, percentiles (p10/p25/p75/p90)
4. Tracks DNF rate per driver (fraction of simulations where they retired)

### EnsembleSimulator (H+E)

Combines Model H's trajectories with Model E's final-position predictions.

**When Models A and B are loaded (production):**

1. Run Model H for the full race
2. Reconstruct A/B input features from H's simulated lap data:
   - `gap_to_leader`, `gap_to_ahead` (sorted cum_time diff per lap)
   - `lap_time_delta_race_median` (expanding median baseline)
   - `lap_time_rolling_3` (shifted rolling mean)
   - `degradation_rate` (OLS per stint), `compound_pace_delta` (NaN — LightGBM handles missing)
3. Run Model A (9 features) and Model B (8 features) independently
4. Aggregate A/B predictions per driver into 10 meta-features (last, mean, std, min, range, last5)
5. Add `C_pred` (H's final position), `grid_position`, `quali_delta_to_pole`
6. Model E predicts final positions from these 13 features
7. Blend H's trajectory toward E's predictions over the last N laps

**When A/B not loaded (fallback):**

Uses H's simulated positions as proxy for A and B predictions. This produces degenerate features (A_last = B_last = C_pred) and limits E's discriminative power.

### Trajectory Blending

For laps after `total_laps - blend_laps`:

```
alpha = (lap - blend_start) / blend_laps    # 0 → 1 linearly
blended_score = (1 - alpha) * h_position + alpha * e_position
```

Drivers are re-ranked by blended score at each blended lap. DNF'd drivers (no record at the final lap) are appended from H's results.

## Circuit Defaults

Built from historical data in `defaults.py`:

| Field | Source |
|-------|--------|
| `total_laps` | Median of max lap_number per race |
| `typical_stops` | Median pit stops per driver per race |
| `pit_windows` | Median lap number for each pit stop |
| `common_sequence` | Most frequent compound sequence |

**Default strategy generation:**

When no custom strategy is provided, `get_default_strategy()` builds one from the circuit's `pit_windows` and `common_sequence`, starting with the driver's `initial_tyre`.

## DNF/Retirement

Race-level DNF probability is converted to a per-lap hazard rate:

```
hazard = 1 - (1 - p) ^ (1 / total_laps)
```

Each lap, `uniform(0, 1)` is sampled per active driver. If below hazard, the driver retires. Retired drivers are removed from subsequent feature building and ranking, and appear at the end of final results with `status: "DNF"`.

## Strategy Optimization

`strategy.py` provides:

1. **`generate_candidates()`** — Produces 1-stop and 2-stop strategies by permuting compounds and varying pit lap timing around circuit defaults. Filters by min/max stint length and 2+ compound rule.

2. **`optimize_strategy()`** — Two-phase evaluation:
   - Phase 1: Deterministic simulation for all candidates (target driver varies, others use defaults)
   - Phase 2 (optional): Monte Carlo on top 5 for confidence intervals
   - Returns ranked `OptimizationResult` sorted by (position, total_time)

## Source Files

| File | Purpose |
|------|---------|
| `simulation/engine.py` | Base `RaceSimulator`, `DriverState`, `LapRecord`, feature construction |
| `simulation/delta_simulator.py` | `DeltaRaceSimulator` (Model H), `MonteCarloSimulator`, compound corrections |
| `simulation/ensemble_simulator.py` | `EnsembleSimulator` (H+E), A/B feature reconstruction, trajectory blending |
| `simulation/strategy.py` | Candidate generation, strategy optimization |
| `simulation/defaults.py` | Circuit defaults builder, default strategy generation |
| `simulation/quantile_simulator.py` | `QuantileRaceSimulator` (Model I, experimental) |
| `simulation/sequence_simulator.py` | `SequenceRaceSimulator` (Model G, experimental) |
