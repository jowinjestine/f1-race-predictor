# Data Dictionary

Base fields in `data/raw/race/all_races.parquet` and per-season `data/raw/race/season_YYYY.parquet`.
Derived target columns (`is_podium`, `is_points_finish`, `is_dnf`) are only present in `all_races.parquet`.

## Identifiers

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `season` | int | Calendar year (2018-2025) | FastF1 / Jolpica |
| `round` | int | Round number within the season | FastF1 / Jolpica |
| `event_name` | str | Grand Prix name (e.g. "Bahrain Grand Prix") | FastF1 / Jolpica |
| `location` | str | Circuit location (e.g. "Sakhir") | FastF1 / Jolpica |
| `country` | str | Country name | FastF1 / Jolpica |
| `event_date` | str | Race date in YYYY-MM-DD format | FastF1 / Jolpica |

## Driver & Team

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `driver_number` | str | Car number | FastF1 / Jolpica |
| `driver_abbrev` | str | Three-letter abbreviation (e.g. "VER") | FastF1 / Jolpica |
| `driver_id` | str | Driver identifier | FastF1 / Jolpica |
| `first_name` | str | Driver's first name | FastF1 / Jolpica |
| `last_name` | str | Driver's last name | FastF1 / Jolpica |
| `team` | str | Constructor/team name | FastF1 / Jolpica |
| `team_id` | str | Team identifier | FastF1 / Jolpica |

## Race Results

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `finish_position` | int/null | Final classified position (null if not classified) | FastF1 / Jolpica |
| `grid_position` | int/null | Starting grid position | FastF1 / Jolpica |
| `status` | str | Finish status ("Finished", "+1 Lap", "Retired", etc.) | FastF1 / Jolpica |
| `points` | float | Championship points scored | FastF1 / Jolpica |
| `laps_completed` | int | Number of laps completed | FastF1 / Jolpica |
| `is_classified` | bool | Whether the driver was officially classified | FastF1 / Jolpica |

## Qualifying Times

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `q1_time_sec` | float/null | Q1 lap time in seconds (null if not set) | FastF1 / Jolpica |
| `q2_time_sec` | float/null | Q2 lap time in seconds (null if not set) | FastF1 / Jolpica |
| `q3_time_sec` | float/null | Q3 lap time in seconds (null if not set) | FastF1 / Jolpica |
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

---

# Lap-Level Data Dictionary

Base fields in `data/raw/laps/all_laps.parquet` and per-season `data/raw/laps/laps_YYYY.parquet`.

## Identifiers

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `season` | int | Calendar year (2018-2025) | FastF1 / Jolpica |
| `round` | int | Round number within the season | FastF1 / Jolpica |
| `event_name` | str | Grand Prix name | FastF1 / Jolpica |
| `driver_abbrev` | str | Usually a three-letter abbreviation (e.g. "VER"); may contain the Jolpica `driverId` when no code is available | FastF1 / Jolpica |
| `team` | str | Constructor/team name | FastF1 / Jolpica |

## Lap Data

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `lap_number` | int | Lap index (1-based) | FastF1 / Jolpica |
| `lap_time_sec` | float/null | Lap time in seconds | FastF1 / Jolpica |
| `sector_1_sec` | float/null | Sector 1 split time in seconds | FastF1 only (null for 2025) |
| `sector_2_sec` | float/null | Sector 2 split time in seconds | FastF1 only (null for 2025) |
| `sector_3_sec` | float/null | Sector 3 split time in seconds | FastF1 only (null for 2025) |
| `position` | int/null | Race position at end of lap | FastF1 / Jolpica |

## Tire & Stint

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `tire_compound` | str/null | Tyre compound: SOFT, MEDIUM, HARD, INTERMEDIATE, WET | FastF1 only (null for 2025) |
| `tire_life` | int/null | Number of laps on current tyre set | FastF1 / Jolpica (estimated for 2025) |
| `stint` | int/null | Stint number (increments on pit stop) | FastF1 / Jolpica (estimated for 2025) |

## Pit Stops

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `is_pit_in_lap` | bool | Driver pitted at end of this lap | FastF1 / Jolpica |
| `is_pit_out_lap` | bool | Driver exited pits at start of this lap | FastF1 / Jolpica |
| `pit_duration_sec` | float/null | Pit stop duration in seconds (only on pit laps) | Jolpica (null for FastF1 seasons unless backfilled) |

## Track Status & Flags

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `track_status` | str/null | Track status code (e.g. "1" = green, "4" = SC, "6" = VSC) | FastF1 only (null for 2025) |
| `is_personal_best` | bool | Whether this was the driver's fastest lap so far | FastF1 only (false for 2025) |

## Lap-Level Notes

- One row per driver per lap (typically 20 drivers x 50-78 laps per race).
- 2018-2024 collected via FastF1 `session.laps`; 2025 collected via Jolpica `/laps` + `/pitstops` endpoints.
- Sector times, tire compound, track status, and is_personal_best are only available from FastF1 (2018-2024).
- For 2025 (Jolpica), `tire_life` and `stint` are estimated from pit stop timing data.
- Data is partitioned by season: `laps_2018.parquet`, ..., `laps_2025.parquet`.
- Combined file: `all_laps.parquet`.
- All data is persisted to `gs://f1-predictor-artifacts-jowin/data/raw/laps/`.

---

## Notes

- One row per driver per race (typically 20 drivers per race).
- 2018-2024 collected via FastF1 (race results + weather telemetry); 2025 collected via Jolpica API.
- Qualifying times (q1/q2/q3) backfilled from Jolpica API (~70% coverage). Gaps reflect drivers eliminated in Q1, drivers who did not set a qualifying time / DNS, or missing API data.
- FastF1 weather is aggregated from per-minute telemetry during the race (not available for Jolpica-only seasons).
- Open-Meteo weather is a daily observation for the race location.
- Missing weather data (~8%) occurs when circuit coordinates are not mapped or the API is unavailable.
- Data is partitioned by season: `season_2018.parquet`, ..., `season_2025.parquet`.
- Combined file: `all_races.parquet` contains all seasons with derived target columns.
- All data is persisted to `gs://f1-predictor-artifacts-jowin/data/raw/race/`.
