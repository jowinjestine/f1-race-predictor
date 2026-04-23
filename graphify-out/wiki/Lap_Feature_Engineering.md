# Lap Feature Engineering

## Overview
This community contains the lap-level feature engineering pipeline for Models A and B. The module transforms raw lap data into predictive features including race normalization, rolling pace, position changes, gap calculations, pit stop features, race progress, and tyre-specific features (degradation rate, compound pace delta). Model A includes tyre data (2019-2024) while Model B operates without tyre data (2018-2025), sharing a common core builder.

## Key Components
| Component | Type | Source File |
|-----------|------|-------------|
| lap_features.py | Module | src/f1_predictor/features/lap_features.py |
| build_lap_tyre_features() | Function | src/f1_predictor/features/lap_features.py |
| build_lap_notyre_features() | Function | src/f1_predictor/features/lap_features.py |
| _build_lap_features() | Function (private) | src/f1_predictor/features/lap_features.py |
| _add_race_normalization() | Function (private) | src/f1_predictor/features/lap_features.py |
| _add_rolling_pace() | Function (private) | src/f1_predictor/features/lap_features.py |
| _add_position_features() | Function (private) | src/f1_predictor/features/lap_features.py |
| _add_gap_features() | Function (private) | src/f1_predictor/features/lap_features.py |
| _add_pit_features() | Function (private) | src/f1_predictor/features/lap_features.py |
| _add_race_progress() | Function (private) | src/f1_predictor/features/lap_features.py |
| _add_tyre_features() | Function (private) | src/f1_predictor/features/lap_features.py |
| _compute_degradation_rate() | Function (private) | src/f1_predictor/features/lap_features.py |
| _compute_compound_pace_delta() | Function (private) | src/f1_predictor/features/lap_features.py |

## Relationships

### Internal
- `build_lap_tyre_features()` --calls--> `_build_lap_features()` [1.0]
- `build_lap_notyre_features()` --calls--> `_build_lap_features()` [1.0]
- `_build_lap_features()` --calls--> `_add_race_normalization()` [1.0]
- `_build_lap_features()` --calls--> `_add_rolling_pace()` [1.0]
- `_build_lap_features()` --calls--> `_add_position_features()` [1.0]
- `_build_lap_features()` --calls--> `_add_gap_features()` [1.0]
- `_build_lap_features()` --calls--> `_add_pit_features()` [1.0]
- `_build_lap_features()` --calls--> `_add_race_progress()` [1.0]
- `_build_lap_features()` --calls--> `_add_tyre_features()` [1.0]
- `_add_tyre_features()` --calls--> `_compute_degradation_rate()` [1.0]
- `_add_tyre_features()` --calls--> `_compute_compound_pace_delta()` [1.0]

### Cross-community
- Called by [Feature Build Functions](Feature_Build_Functions.md) (Model A and Model B training notebooks)
- Uses utilities from [Common Feature Utilities](Common_Feature_Utilities.md) (rolling_mean_by_group, encode_compound_onehot)
- Tests are in [Lap Feature Tests](Lap_Feature_Tests.md) and [Feature Module Concepts](Feature_Module_Concepts.md)
- Consumes raw lap data from [Data Collection Core](Data_Collection_Core.md)

## Source Files
- `src/f1_predictor/features/lap_features.py`
