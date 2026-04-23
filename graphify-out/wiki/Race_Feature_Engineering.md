# Race Feature Engineering

## Overview
This community implements the pre-race feature engineering pipeline for Model C of the F1 predictor. It transforms raw race-level data into predictive features including qualifying performance, season form, driver circuit history, team performance trends, circuit characteristics, and weather conditions. All time-series features use shifted expanding/rolling windows to prevent data leakage.

## Key Components
| Component | Type | Source File |
|-----------|------|-------------|
| `race_features.py` | Module | `src/f1_predictor/features/race_features.py` |
| `build_race_features()` | Function | `src/f1_predictor/features/race_features.py` |
| `_add_qualifying_features()` | Function | `src/f1_predictor/features/race_features.py` |
| `_add_season_form_features()` | Function | `src/f1_predictor/features/race_features.py` |
| `_compute_position_trend()` | Function | `src/f1_predictor/features/race_features.py` |
| `_add_driver_circuit_features()` | Function | `src/f1_predictor/features/race_features.py` |
| `_add_team_form_features()` | Function | `src/f1_predictor/features/race_features.py` |
| `_add_circuit_features()` | Function | `src/f1_predictor/features/race_features.py` |
| `_impute_weather_historical()` | Function | `src/f1_predictor/features/race_features.py` |
| `_add_weather_features()` | Function | `src/f1_predictor/features/race_features.py` |

## Relationships

### Internal
- `build_race_features()` --calls--> `_add_qualifying_features()` [1.0]
- `build_race_features()` --calls--> `_add_season_form_features()` [1.0]
- `build_race_features()` --calls--> `_add_driver_circuit_features()` [1.0]
- `build_race_features()` --calls--> `_add_team_form_features()` [1.0]
- `build_race_features()` --calls--> `_add_circuit_features()` [1.0]
- `build_race_features()` --calls--> `_add_weather_features()` [1.0]
- `_add_season_form_features()` --calls--> `_compute_position_trend()` [1.0]
- `_add_weather_features()` --calls--> `_impute_weather_historical()` [1.0]

### Cross-community
- Tested by the [Race Feature Tests](Race_Feature_Tests.md) community
- Uses utilities from the [Common Feature Utilities](Common_Feature_Utilities.md) (rolling/expanding functions, safe_divide)
- Consumes data produced by the [Season Collection Pipeline](Season_Collection_Pipeline.md)
- Related function-level view in [Race Feature Functions](Race_Feature_Functions.md)
- Feature output feeds into the [Feature Build Functions](Feature_Build_Functions.md) and [Notebook Generation](Notebook_Generation.md) pipelines

## Source Files
- `src/f1_predictor/features/race_features.py`
