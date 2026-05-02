# API Reference

The F1 Race Predictor API is a FastAPI application deployed on Google Cloud Run. It provides deterministic and Monte Carlo race simulations, pit strategy optimization, and reference data queries.

**Base URL:** `https://f1-race-predictor-yqe7tpf66a-uc.a.run.app`

**Authentication:** Google Cloud identity token required.

```bash
TOKEN=$(gcloud auth print-identity-token)
curl -H "Authorization: Bearer $TOKEN" $BASE_URL/api/ready
```

**Interactive docs:** `GET /docs` (Swagger UI) or `GET /redoc` (ReDoc)

---

## Simulation Endpoints

### POST /api/v1/simulate

Run a deterministic lap-by-lap race simulation using the H+E ensemble.

**How it works:**
1. Each driver starts with their qualifying time as a pace baseline
2. At every lap, Model H predicts each driver's delta from the historical field median lap-time-ratio
3. Predictions are clamped (normal: 1.01-1.15x, pit-in: 1.10-1.50x, pit-out: 1.03-1.25x) and converted to absolute lap times
4. Compound pace offsets and degradation rates are applied
5. Positions are derived from cumulative race time after each lap
6. If `blend_laps > 0`, the last N laps interpolate H's positions toward Model E's final predictions
7. If `dnf_probability` is set, drivers may retire mid-race

**Request:**

```json
{
  "circuit": "Monaco Grand Prix",
  "drivers": [
    {
      "driver": "VER",
      "grid_position": 1,
      "q1": 71.5,
      "q2": 70.8,
      "q3": 70.2,
      "initial_tyre": "MEDIUM",
      "dnf_probability": 0.0
    },
    {
      "driver": "LEC",
      "grid_position": 2,
      "q3": 70.4,
      "initial_tyre": "SOFT",
      "dnf_probability": 0.15
    }
  ],
  "strategies": {
    "VER": [
      {"compound": "MEDIUM", "pit_on_lap": 25},
      {"compound": "HARD", "pit_on_lap": null}
    ]
  },
  "blend_laps": 0,
  "n_simulations": 200
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `circuit` | string | required | Circuit name as it appears in the F1 calendar. Use `GET /api/v1/circuits` to list options. |
| `drivers` | array | required | 2-20 drivers with qualifying data (see Driver Input below) |
| `strategies` | object | null | Per-driver pit strategies keyed by driver abbreviation. If omitted, circuit-default strategies are used. |
| `blend_laps` | int | 10 | Laps at the end to blend H toward E predictions. Set to 0 for pure Model H. |
| `n_simulations` | int | 200 | Number of Monte Carlo simulations (for /simulate/monte-carlo only) |

**Driver Input fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `driver` | string | required | Three-letter abbreviation (e.g. VER, NOR, LEC) |
| `grid_position` | int | required | Starting grid position (1-20) |
| `q1` | float | null | Q1 lap time in seconds |
| `q2` | float | null | Q2 lap time in seconds |
| `q3` | float | null | Q3 lap time in seconds. Best of q1/q2/q3 is used as pace baseline. |
| `initial_tyre` | string | "MEDIUM" | Starting tyre: SOFT, MEDIUM, HARD, INTERMEDIATE, WET |
| `dnf_probability` | float | 0.0 | Race-level retirement probability (0.0 to 1.0) |

**Response:**

```json
{
  "circuit": "Monaco Grand Prix",
  "total_laps": 78,
  "model": "H+E ensemble",
  "blend_laps": 10,
  "lap_records": [
    {
      "lap_number": 1,
      "driver": "VER",
      "position": 1,
      "lap_time": 76.42,
      "cum_time": 76.42,
      "gap_to_leader": 0.0,
      "compound": "MEDIUM",
      "tire_life": 1,
      "stint": 1
    }
  ],
  "final_standings": [
    {
      "driver": "VER",
      "position": 1,
      "total_time": 6007.7,
      "gap_to_leader": 0.0,
      "pit_stops": 1,
      "status": "Finished",
      "laps_completed": 78
    }
  ]
}
```

**Error responses:**
- `400` — Unknown circuit name
- `503` — Models not loaded yet (service still starting)

---

### POST /api/v1/simulate/monte-carlo

Run N independent simulations with noise injection for position distributions.

Uses the same request body as `/simulate`. The `n_simulations` field controls how many runs (default 200, range 10-1000).

Each simulation injects Gaussian noise (std=0.01) into Model H's predictions and independently samples DNF retirements.

**Response:**

```json
{
  "circuit": "Monaco Grand Prix",
  "n_simulations": 50,
  "model": "H Monte Carlo",
  "standings": [
    {
      "driver": "VER",
      "position": 1,
      "position_mean": 1.08,
      "position_p10": 1,
      "position_p25": 1,
      "position_p75": 1,
      "position_p90": 1,
      "position_std": 0.27,
      "dnf_rate": 0.0
    },
    {
      "driver": "LEC",
      "position": 2,
      "position_mean": 2.34,
      "position_p10": 2,
      "position_p25": 2,
      "position_p75": 2,
      "position_p90": 5,
      "position_std": 1.05,
      "dnf_rate": 0.12
    }
  ]
}
```

| Field | Description |
|-------|-------------|
| `position` | Median finishing position across all simulations |
| `position_mean` | Mean finishing position |
| `position_p10` | 10th percentile — optimistic scenario |
| `position_p25` | 25th percentile |
| `position_p75` | 75th percentile |
| `position_p90` | 90th percentile — pessimistic scenario |
| `position_std` | Standard deviation (lower = more predictable) |
| `dnf_rate` | Fraction of simulations where the driver retired |

**Performance:** ~5s for 50 simulations, ~15s for 200 simulations.

---

### POST /api/v1/optimize-strategy

Find the optimal pit strategy for a target driver.

Generates candidate strategies by varying pit lap timing, stop counts, and compound sequences around the circuit's historical defaults. Each candidate is evaluated by running a full Model H simulation.

**Request:**

```json
{
  "circuit": "Monaco Grand Prix",
  "drivers": [
    {"driver": "VER", "grid_position": 1, "q3": 70.2},
    {"driver": "LEC", "grid_position": 2, "q3": 70.4},
    {"driver": "NOR", "grid_position": 3, "q3": 70.6}
  ],
  "target_driver": "NOR",
  "use_monte_carlo": false,
  "n_simulations": 50,
  "max_candidates": 30,
  "pit_lap_delta": 5
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `circuit` | string | required | Circuit name |
| `drivers` | array | required | Starting grid (2-20 drivers) |
| `target_driver` | string | required | Driver abbreviation to optimize for |
| `use_monte_carlo` | bool | false | Run Monte Carlo on top 5 for confidence intervals |
| `n_simulations` | int | 50 | MC simulations per strategy (if enabled, 10-200) |
| `max_candidates` | int | 30 | Maximum candidate strategies to evaluate (5-100) |
| `pit_lap_delta` | int | 5 | Pit lap variation range (+/- laps, 1-10) |

**Response:**

```json
{
  "circuit": "Monaco Grand Prix",
  "target_driver": "NOR",
  "n_candidates_tested": 20,
  "strategies": [
    {
      "rank": 1,
      "strategy": [
        {"compound": "SOFT", "pit_on_lap": 24},
        {"compound": "HARD", "pit_on_lap": null}
      ],
      "description": "1-stop: SOFT(1-24) HARD(25-78)",
      "predicted_position": 3,
      "predicted_time": 6062.2,
      "gap_to_leader": 23.0,
      "position_mean": null,
      "position_std": null
    }
  ]
}
```

**Performance:** <2s deterministic, <15s with Monte Carlo.

---

## Data Endpoints

### GET /api/v1/circuits

List all available circuits with default pit strategies.

**Response:**

```json
[
  {
    "name": "Monaco Grand Prix",
    "total_laps": 78,
    "typical_stops": 1,
    "pit_windows": [18],
    "common_sequence": ["MEDIUM", "HARD"]
  }
]
```

### GET /api/v1/drivers/{season}

List drivers for a season (2018-2025).

```json
[
  {"driver_abbrev": "VER", "team": "Red Bull Racing"},
  {"driver_abbrev": "NOR", "team": "McLaren"}
]
```

### GET /api/v1/races/{season}

List races for a season.

```json
[
  {"season": 2024, "round": 1, "event_name": "Bahrain Grand Prix"}
]
```

---

## Health Endpoints

### GET /api/health

Liveness probe. Always returns `{"status": "ok"}`.

### GET /api/ready

Readiness probe. Returns `{"status": "ready", "models_loaded": true}` when all models are loaded and the service can accept simulation requests.

---

## Configuration

The API is configured via environment variables (prefix `F1_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `F1_MODEL_DIR` | `data/raw/model` | Path to model pickle files |
| `F1_DATA_DIR` | `data` | Path to parquet data files |
| `F1_LOAD_FROM_GCS` | `false` | Download models from GCS at startup |
| `F1_GCS_BUCKET` | `f1-predictor-artifacts-jowin` | GCS bucket name |
| `F1_PORT` | `8080` | Server port |
| `F1_DEFAULT_BLEND_LAPS` | `0` | Default blend laps for H+E ensemble |
