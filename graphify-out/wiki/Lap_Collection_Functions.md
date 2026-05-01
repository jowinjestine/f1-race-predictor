# Lap Collection Functions

## Overview
This community encapsulates the lap-by-lap data collection pipeline for the F1 Race Predictor. It gathers detailed lap data from two sources -- FastF1 session data and the Jolpica REST API -- covering the 2018-2025 seasons. The module handles driver/team identification mapping, tire compound normalization, pit stop duration enrichment, and uploads the resulting Parquet files to Google Cloud Storage.

## Key Components
| Component | Type | Source File |
|-----------|------|-------------|
| `collect_laps.py` | Module | `src/f1_predictor/data/collect_laps.py` |
| `collect_all_laps()` | Function | `src/f1_predictor/data/collect_laps.py` |
| `collect_laps_fastf1()` | Function | `src/f1_predictor/data/collect_laps.py` |
| `collect_laps_jolpica()` | Function | `src/f1_predictor/data/collect_laps.py` |
| `_fastf1_lap_row()` | Function | `src/f1_predictor/data/collect_laps.py` |
| `_safe_int()` | Utility | `src/f1_predictor/data/collect_laps.py` |
| `_normalize_compound()` | Utility | `src/f1_predictor/data/collect_laps.py` |
| `_build_pitstop_map()` | Function | `src/f1_predictor/data/collect_laps.py` |
| `_build_driver_id_to_code()` | Function | `src/f1_predictor/data/collect_laps.py` |
| `_build_driver_id_to_team()` | Function | `src/f1_predictor/data/collect_laps.py` |
| `add_pit_duration()` | Function | `src/f1_predictor/data/collect_laps.py` |
| `_upload_laps_to_gcs()` | Function | `src/f1_predictor/data/collect_laps.py` |

## Relationships

### Internal
- `collect_all_laps()` --calls--> `collect_laps_fastf1()` [1.0]
- `collect_all_laps()` --calls--> `collect_laps_jolpica()` [1.0]
- `collect_all_laps()` --calls--> `_upload_laps_to_gcs()` [1.0]
- `collect_laps_fastf1()` --calls--> `_fastf1_lap_row()` [1.0]
- `collect_laps_jolpica()` --calls--> `_safe_int()` [1.0]
- `collect_laps_jolpica()` --calls--> `_build_pitstop_map()` [1.0]
- `collect_laps_jolpica()` --calls--> `_build_driver_id_to_code()` [1.0]
- `collect_laps_jolpica()` --calls--> `_build_driver_id_to_team()` [1.0]
- `_fastf1_lap_row()` --calls--> `_safe_int()` [1.0]
- `_fastf1_lap_row()` --calls--> `_normalize_compound()` [1.0]
- `_build_pitstop_map()` --calls--> `_safe_int()` [1.0]

### Cross-community
- Tested by the [Lap Collection Tests](Lap_Collection_Tests.md) community
- Tested by the [Collection Module Structure](Collection_Module_Structure.md) community (test_collect_laps.py)
- Lap data feeds into the [Lap Feature Engineering](Lap_Feature_Engineering.md) pipeline
- GCS upload depends on the [GCS Storage Layer](GCS_Storage_Layer.md)

## Source Files
- `src/f1_predictor/data/collect_laps.py`
