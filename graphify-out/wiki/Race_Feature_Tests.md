# Race Feature Tests

## Overview
This community contains the comprehensive test suite for pre-race feature engineering (Model C). It validates qualifying features, season form, driver-circuit history, team form, circuit classification, weather features, and data leakage prevention -- all within a single test file using a shared `_make_races()` fixture that creates minimal race data for testing.

## Key Components
| Component | Type | Source File |
|-----------|------|-------------|
| test_race_features.py | Module | tests/test_features/test_race_features.py |
| _make_races() | Helper | tests/test_features/test_race_features.py |
| TestBuildRaceFeatures | Test Class | tests/test_features/test_race_features.py |
| TestQualifyingFeatures | Test Class | tests/test_features/test_race_features.py |
| TestSeasonFormFeatures | Test Class | tests/test_features/test_race_features.py |
| TestDriverCircuitFeatures | Test Class | tests/test_features/test_race_features.py |
| TestTeamFormFeatures | Test Class | tests/test_features/test_race_features.py |
| TestCircuitFeatures | Test Class | tests/test_features/test_race_features.py |
| TestWeatherFeatures | Test Class | tests/test_features/test_race_features.py |
| TestLeakagePrevention | Test Class | tests/test_features/test_race_features.py |

## Relationships

### Internal
- `test_race_features.py` --contains--> `TestBuildRaceFeatures` [1.0]
- `test_race_features.py` --contains--> `TestQualifyingFeatures` [1.0]
- `test_race_features.py` --contains--> `TestSeasonFormFeatures` [1.0]
- `test_race_features.py` --contains--> `TestDriverCircuitFeatures` [1.0]
- `test_race_features.py` --contains--> `TestTeamFormFeatures` [1.0]
- `test_race_features.py` --contains--> `TestCircuitFeatures` [1.0]
- `test_race_features.py` --contains--> `TestWeatherFeatures` [1.0]
- `test_race_features.py` --contains--> `TestLeakagePrevention` [1.0]
- `_make_races()` --calls--> all test methods (shared fixture) [1.0]

### Cross-community
- Tests validate features built by the [Race Feature Engineering](Race_Feature_Engineering.md) module
- TestLeakagePrevention verifies principles from [Feature Module Concepts](Feature_Module_Concepts.md) (Data Leakage Prevention Pattern)
- TestCircuitFeatures validates circuit type logic also relevant to [Data Collection Core](Data_Collection_Core.md)

## Source Files
- `tests/test_features/test_race_features.py`
