# Jolpica API Client

## Overview
This community implements the HTTP client for the Jolpica (formerly Ergast-compatible) F1 REST API. It provides functions to fetch season schedules, race results, qualifying results, lap timing data, and pit stop records. It also includes time-string parsers and a high-level `collect_season_jolpica()` orchestrator that assembles a full season of race and qualifying data into a structured format.

## Key Components
| Component | Type | Source File |
|-----------|------|-------------|
| `jolpica.py` | Module | `src/f1_predictor/data/jolpica.py` |
| `_get_json()` | Function | `src/f1_predictor/data/jolpica.py` |
| `get_season_schedule()` | Function | `src/f1_predictor/data/jolpica.py` |
| `get_race_results()` | Function | `src/f1_predictor/data/jolpica.py` |
| `get_qualifying_results()` | Function | `src/f1_predictor/data/jolpica.py` |
| `parse_lap_time()` | Function | `src/f1_predictor/data/jolpica.py` |
| `_parse_race_time_millis()` | Function | `src/f1_predictor/data/jolpica.py` |
| `get_laps()` | Function | `src/f1_predictor/data/jolpica.py` |
| `get_pitstops()` | Function | `src/f1_predictor/data/jolpica.py` |
| `collect_season_jolpica()` | Function | `src/f1_predictor/data/jolpica.py` |

## Relationships

### Internal
- `_get_json()` --calls--> `get_season_schedule()`, `get_race_results()`, `get_qualifying_results()`, `get_laps()`, `get_pitstops()` [1.0]
- `collect_season_jolpica()` --calls--> `get_season_schedule()` [1.0]
- `collect_season_jolpica()` --calls--> `get_race_results()` [1.0]
- `collect_season_jolpica()` --calls--> `get_qualifying_results()` [1.0]
- `collect_season_jolpica()` --calls--> `parse_lap_time()` [1.0]
- `collect_season_jolpica()` --calls--> `_parse_race_time_millis()` [1.0]

### Cross-community
- Tested by the [Jolpica API Tests](Jolpica_API_Tests.md) community
- Tested by the [Collection Module Structure](Collection_Module_Structure.md) community (test_jolpica.py)
- Called by the [Season Collection Pipeline](Season_Collection_Pipeline.md) for qualifying backfill
- Called by the [Lap Collection Functions](Lap_Collection_Functions.md) for Jolpica-sourced lap data
- Part of the [Data Collection Core](Data_Collection_Core.md) subsystem

## Source Files
- `src/f1_predictor/data/jolpica.py`
