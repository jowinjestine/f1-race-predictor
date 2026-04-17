# Data Dictionary

Base fields in `data/raw/all_races.parquet` and per-season `data/raw/season_YYYY.parquet`.
Derived target columns (`is_podium`, `is_points_finish`, `is_dnf`) are only present in `all_races.parquet`.

## Identifiers

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `season` | int | Calendar year (2018-2025) | FastF1 / Jolpica |
| `round` | int | Round number within the season | FastF1 |
| `event_name` | str | Grand Prix name (e.g. "Bahrain Grand Prix") | FastF1 |
| `location` | str | Circuit location (e.g. "Sakhir") | FastF1 |
| `country` | str | Country name | FastF1 |
| `event_date` | str | Race date in YYYY-MM-DD format | FastF1 |

## Driver & Team

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `driver_number` | str | Car number | FastF1 |
| `driver_abbrev` | str | Three-letter abbreviation (e.g. "VER") | FastF1 |
| `driver_id` | str | FastF1 driver identifier | FastF1 |
| `first_name` | str | Driver's first name | FastF1 |
| `last_name` | str | Driver's last name | FastF1 |
| `team` | str | Constructor/team name | FastF1 |
| `team_id` | str | FastF1 team identifier | FastF1 |

## Race Results

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `finish_position` | int/null | Final classified position (null if not classified) | FastF1 |
| `grid_position` | int/null | Starting grid position | FastF1 |
| `status` | str | Finish status ("Finished", "+1 Lap", "Retired", etc.) | FastF1 |
| `points` | float | Championship points scored | FastF1 |
| `laps_completed` | int | Number of laps completed | FastF1 |
| `is_classified` | bool | Whether the driver was officially classified | FastF1 |

## Qualifying Times

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `q1_time_sec` | float/null | Q1 lap time in seconds (null if not set) | FastF1 |
| `q2_time_sec` | float/null | Q2 lap time in seconds (null if not set) | FastF1 |
| `q3_time_sec` | float/null | Q3 lap time in seconds (null if not set) | FastF1 |
| `race_time_sec` | float/null | Total race time in seconds. FastF1 provides absolute time for all drivers; Jolpica (2025+) provides milliseconds for the race winner only — null for other drivers | FastF1 / Jolpica |

## FastF1 Weather (race-session aggregates)

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `f1_air_temp_mean` | float/null | Mean air temperature during race (°C) | FastF1 telemetry |
| `f1_track_temp_mean` | float/null | Mean track surface temperature (°C) | FastF1 telemetry |
| `f1_humidity_mean` | float/null | Mean relative humidity (%) | FastF1 telemetry |
| `f1_pressure_mean` | float/null | Mean barometric pressure (mbar) | FastF1 telemetry |
| `f1_wind_speed_mean` | float/null | Mean wind speed (m/s) | FastF1 telemetry |
| `f1_rainfall` | bool/null | Whether any rainfall was recorded during the race | FastF1 telemetry |

## Open-Meteo Weather (daily observations)

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `weather_temp_max` | float/null | Maximum air temperature on race day (°C) | Open-Meteo |
| `weather_temp_min` | float/null | Minimum air temperature on race day (°C) | Open-Meteo |
| `weather_precip_mm` | float/null | Total precipitation on race day (mm) | Open-Meteo |
| `weather_wind_max_kph` | float/null | Maximum wind speed on race day (km/h) | Open-Meteo |

## Derived Target Variables

| Column | Type | Description | Logic |
|--------|------|-------------|-------|
| `is_podium` | bool | Finished in top 3 | `finish_position <= 3` |
| `is_points_finish` | bool | Finished in top 10 | `finish_position <= 10` |
| `is_dnf` | bool | Did not finish | `status` not in ("Finished", "+N Lap") |

## Notes

- One row per driver per race (typically 20 drivers per race).
- 2018-2024 collected via FastF1 (race results + weather telemetry); 2025 collected via Jolpica API.
- Qualifying times (q1/q2/q3) backfilled from Jolpica API (~70% coverage). Gaps are drivers eliminated before Q1 or missing API data.
- FastF1 weather is aggregated from per-minute telemetry during the race (not available for Jolpica-only seasons).
- Open-Meteo weather is a daily observation for the race location.
- Missing weather data (~8%) occurs when circuit coordinates are not mapped or the API is unavailable.
- Data is partitioned by season: `season_2018.parquet`, ..., `season_2025.parquet`.
- Combined file: `all_races.parquet` contains all seasons with derived target columns.
- All data is persisted to `gs://f1-predictor-artifacts-jowin/data/raw/`.
