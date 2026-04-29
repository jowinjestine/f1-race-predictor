# Season Collection Pipeline

## Overview
This community contains the core race-level data collection pipeline in `collect.py`. It orchestrates fetching F1 race results and weather data for the 2018-2025 seasons using FastF1 for telemetry/weather and Open-Meteo for supplementary weather. The pipeline collects per-season data, backfills qualifying times from Jolpica, computes target variables, and uploads partitioned Parquet files to GCS.

## Key Components
| Component | Type | Source File |
|-----------|------|-------------|
| `collect.py` | Module | `src/f1_predictor/data/collect.py` |
| `collect_all()` | Function | `src/f1_predictor/data/collect.py` |
| `collect_season()` | Function | `src/f1_predictor/data/collect.py` |
| `get_openmeteo_weather()` | Function | `src/f1_predictor/data/collect.py` |
| `_aggregate_fastf1_weather()` | Function | `src/f1_predictor/data/collect.py` |
| `backfill_qualifying()` | Function | `src/f1_predictor/data/collect.py` |
| `add_target_variables()` | Function | `src/f1_predictor/data/collect.py` |
| `_upload_to_gcs()` | Function | `src/f1_predictor/data/collect.py` |
| `ensure_cache()` | Function | `src/f1_predictor/data/collect.py` |
| `_first()` | Utility | `src/f1_predictor/data/collect.py` |
| `_safe_float()` | Utility | `src/f1_predictor/data/collect.py` |
| `_empty_f1_weather()` | Utility | `src/f1_predictor/data/collect.py` |
| `_td_to_seconds()` | Utility | `src/f1_predictor/data/collect.py` |

## Relationships

### Internal
- `collect_all()` --calls--> `ensure_cache()` [1.0]
- `collect_all()` --calls--> `collect_season()` [1.0]
- `collect_all()` --calls--> `get_openmeteo_weather()` [1.0]
- `collect_all()` --calls--> `backfill_qualifying()` [1.0]
- `collect_all()` --calls--> `add_target_variables()` [1.0]
- `collect_all()` --calls--> `_upload_to_gcs()` [1.0]
- `collect_season()` --calls--> `get_openmeteo_weather()` [1.0]
- `collect_season()` --calls--> `_aggregate_fastf1_weather()` [1.0]
- `collect_season()` --calls--> `_td_to_seconds()` [1.0]
- `_aggregate_fastf1_weather()` --calls--> `_safe_float()` [1.0]
- `_aggregate_fastf1_weather()` --calls--> `_empty_f1_weather()` [1.0]
- `get_openmeteo_weather()` --calls--> `_first()` [1.0]

### Cross-community
- Tested by the [Collection Module Structure](Collection_Module_Structure.md) community (test_collect.py)
- Also tested by the [Data Collection Tests](Data_Collection_Tests.md) community
- `backfill_qualifying()` depends on the [Jolpica API Client](Jolpica_API_Client.md)
- GCS upload relies on the [GCS Storage Layer](GCS_Storage_Layer.md)
- Output data feeds into the [Race Feature Engineering](Race_Feature_Engineering.md) pipeline

## Source Files
- `src/f1_predictor/data/collect.py`
