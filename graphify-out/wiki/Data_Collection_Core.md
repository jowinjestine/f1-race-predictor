# Data Collection Core

## Overview
This community represents the central data collection pipeline, project documentation, and foundational architecture of the F1 predictor. It includes race-level and lap-level data collection from FastF1 and Jolpica APIs, weather enrichment from Open-Meteo, qualifying backfill logic, GCS storage integration, smoke tests, and the README documentation covering the project overview, architecture, tech stack, and design rationale (XGBoost, SHAP, FastAPI on Cloud Run).

## Key Components
| Component | Type | Source File |
|-----------|------|-------------|
| Collect Season (FastF1) | Function | src/f1_predictor/data/collect.py |
| Collect All Seasons | Function | src/f1_predictor/data/collect.py |
| Backfill Qualifying from Jolpica | Function | src/f1_predictor/data/collect.py |
| Get Open-Meteo Weather | Function | src/f1_predictor/data/collect.py |
| Add Target Variables | Function | src/f1_predictor/data/collect.py |
| Circuit Coordinates Lookup | Data | src/f1_predictor/data/collect.py |
| Collect Laps FastF1 | Function | src/f1_predictor/data/collect_laps.py |
| Collect Laps Jolpica | Function | src/f1_predictor/data/collect_laps.py |
| Collect All Laps | Function | src/f1_predictor/data/collect_laps.py |
| Build Pitstop Map | Function | src/f1_predictor/data/collect_laps.py |
| GCS Storage Module | Module | src/f1_predictor/data/storage.py |
| test_smoke.py | Test Module | tests/test_smoke.py |
| Data Dictionary | Document | docs/data_dictionary.md |
| F1 Race Predictor Project Overview | Document | README.md |
| Architecture Diagram | Document | README.md |
| Tech Stack Table | Document | README.md |

## Relationships

### Internal
- `Collect All Seasons` --calls--> `Collect Season (FastF1)` [1.0]
- `Collect All Seasons` --calls--> `Backfill Qualifying from Jolpica` [1.0]
- `Collect All Seasons` --calls--> `Add Target Variables` [1.0]
- `Collect All Seasons` --calls--> `Upload Season Files to GCS` [1.0]
- `Collect Season (FastF1)` --calls--> `Get Open-Meteo Weather` [1.0]
- `Collect All Laps` --calls--> `Collect Laps FastF1` [1.0]
- `Collect All Laps` --calls--> `Collect Laps Jolpica` [1.0]
- `Collect All Laps` --calls--> `GCS Storage Module` [1.0]
- `Collect Laps Jolpica` --calls--> `Build Pitstop Map` [1.0]
- `Data Dictionary` --references--> `Race-Level Data Schema`, `Lap-Level Data Schema` [1.0]

### Cross-community
- `GCS Storage Module` is detailed in [GCS Storage Layer](GCS_Storage_Layer.md)
- `Backfill Qualifying from Jolpica` relies on the [Jolpica API Client](Jolpica_API_Client.md)
- Collected data feeds into [Lap Feature Engineering](Lap_Feature_Engineering.md) and [Race Feature Engineering](Race_Feature_Engineering.md)
- Storage module tests are in [Storage Layer Tests](Storage_Layer_Tests.md)
- Collection tests are in [Data Collection Tests](Data_Collection_Tests.md) and [Lap Collection Tests](Lap_Collection_Tests.md)
- Architecture Diagram references the [Feature Build Functions](Feature_Build_Functions.md) training pipeline

## Source Files
- `src/f1_predictor/data/collect.py`
- `src/f1_predictor/data/collect_laps.py`
- `src/f1_predictor/data/storage.py`
- `tests/test_smoke.py`
- `tests/test_storage.py`
- `docs/data_dictionary.md`
- `README.md`
