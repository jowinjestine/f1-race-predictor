# Race Feature Functions

## Overview
This community captures the detailed function-level call graph for the race-level feature engineering pipeline (Model C). It shows how `build_race_features()` orchestrates sub-functions for qualifying, season form, driver-circuit history, team form, circuit classification, and weather features -- all consuming common utility helpers. It also documents key architectural patterns like the Data Leakage Prevention Pattern (shift(1)) and domain lookups such as Street Circuits Classification and Location Aliases Normalization.

## Key Components
| Component | Type | Source File |
|-----------|------|-------------|
| `Build Race Features (Model C)` | Function | `src/f1_predictor/features/race_features.py` |
| `Add Qualifying Features` | Function | `src/f1_predictor/features/race_features.py` |
| `Add Season Form Features` | Function | `src/f1_predictor/features/race_features.py` |
| `Add Driver Circuit History Features` | Function | `src/f1_predictor/features/race_features.py` |
| `Add Team Form Features` | Function | `src/f1_predictor/features/race_features.py` |
| `Add Circuit Type Features` | Function | `src/f1_predictor/features/race_features.py` |
| `Add Weather Features` | Function | `src/f1_predictor/features/race_features.py` |
| `Impute Weather Historical` | Function | `src/f1_predictor/features/race_features.py` |
| `Rolling Mean by Group` | Function | `src/f1_predictor/features/common.py` |
| `Expanding Mean by Group` | Function | `src/f1_predictor/features/common.py` |
| `Safe Divide` | Function | `src/f1_predictor/features/common.py` |
| `Data Leakage Prevention Pattern` | Pattern | `src/f1_predictor/features/common.py` |
| `Street Circuits Classification` | Domain Data | `src/f1_predictor/features/race_features.py` |
| `Location Aliases Normalization` | Domain Data | `src/f1_predictor/features/race_features.py` |

## Relationships

### Internal
- `Build Race Features (Model C)` --calls--> `Add Qualifying Features` [1.0]
- `Build Race Features (Model C)` --calls--> `Add Season Form Features` [1.0]
- `Build Race Features (Model C)` --calls--> `Add Driver Circuit History Features` [1.0]
- `Build Race Features (Model C)` --calls--> `Add Team Form Features` [1.0]
- `Build Race Features (Model C)` --calls--> `Add Circuit Type Features` [1.0]
- `Build Race Features (Model C)` --calls--> `Add Weather Features` [1.0]
- `Build Race Features (Model C)` --references--> `Location Aliases Normalization` [1.0]
- `Add Qualifying Features` --calls--> `Safe Divide` [1.0]
- `Add Season Form Features` --calls--> `Rolling Mean by Group` [1.0]
- `Add Season Form Features` --calls--> `Expanding Mean by Group` [1.0]
- `Add Driver Circuit History Features` --calls--> `Expanding Mean by Group` [1.0]
- `Add Team Form Features` --calls--> `Rolling Mean by Group` [1.0]
- `Add Weather Features` --calls--> `Impute Weather Historical` [1.0]
- `Add Circuit Type Features` --references--> `Street Circuits Classification` [1.0]
- `Rolling Mean by Group` --implements--> `Data Leakage Prevention Pattern` [1.0]
- `Expanding Mean by Group` --implements--> `Data Leakage Prevention Pattern` [1.0]

### Cross-community
- Detailed function view of [Race Feature Engineering](Race_Feature_Engineering.md) module
- Utility functions sourced from [Common Feature Utilities](Common_Feature_Utilities.md)
- `Build Race Features (Model C)` --references--> `Model C: Pre-Race Features` in [Feature Module Concepts](Feature_Module_Concepts.md) [0.85]
- `Location Aliases Normalization` --semantically_similar_to--> `Circuit Coordinates Lookup` [0.75]
- Tested by [Race Feature Tests](Race_Feature_Tests.md)

## Source Files
- `src/f1_predictor/features/race_features.py`
- `src/f1_predictor/features/common.py`
