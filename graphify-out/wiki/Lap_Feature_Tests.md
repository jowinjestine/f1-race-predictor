# Lap Feature Tests

## Overview
This community contains the test suite for lap-level feature engineering, covering Models A (with tyre data) and B (without tyre data). The tests verify correct column generation, season filtering, tyre column handling, target variable assignment, and critically, data leakage prevention in rolling and expanding features.

## Key Components
| Component | Type | Source File |
|-----------|------|-------------|
| `test_lap_features.py` | Test Module | `tests/test_features/test_lap_features.py` |
| `_make_laps()` | Test Fixture | `tests/test_features/test_lap_features.py` |
| `TestBuildLapTyreFeatures` | Test Class | `tests/test_features/test_lap_features.py` |
| `TestBuildLapNotyreFeatures` | Test Class | `tests/test_features/test_lap_features.py` |
| `TestLeakagePrevention` | Test Class | `tests/test_features/test_lap_features.py` |

## Relationships

### Internal
- `_make_laps()` --calls--> all test methods in `TestBuildLapTyreFeatures` [1.0]
- `_make_laps()` --calls--> all test methods in `TestBuildLapNotyreFeatures` [1.0]
- `_make_laps()` --calls--> all test methods in `TestLeakagePrevention` [1.0]
- `TestBuildLapTyreFeatures` --method--> `.test_filters_to_2019_2024()` [1.0]
- `TestBuildLapTyreFeatures` --method--> `.test_output_has_tyre_columns()` [1.0]
- `TestBuildLapTyreFeatures` --method--> `.test_output_has_shared_features()` [1.0]
- `TestBuildLapTyreFeatures` --method--> `.test_drops_null_tyre_rows()` [1.0]
- `TestBuildLapTyreFeatures` --method--> `.test_position_is_target()` [1.0]
- `TestBuildLapTyreFeatures` --method--> `.test_rolling_3_leakage_prevention()` [1.0]
- `TestBuildLapNotyreFeatures` --method--> `.test_includes_all_seasons()` [1.0]
- `TestBuildLapNotyreFeatures` --method--> `.test_no_tyre_columns()` [1.0]
- `TestBuildLapNotyreFeatures` --method--> `.test_has_shared_features()` [1.0]
- `TestBuildLapNotyreFeatures` --method--> `.test_gap_to_leader_for_leader_is_zero()` [1.0]
- `TestBuildLapNotyreFeatures` --method--> `.test_race_progress_at_last_lap_is_one()` [1.0]
- `TestBuildLapNotyreFeatures` --method--> `.test_pit_count_increments()` [1.0]
- `TestBuildLapNotyreFeatures` --method--> `.test_position_change_from_lap1()` [1.0]
- `TestLeakagePrevention` --method--> `.test_no_future_lap_in_race_normalization()` [1.0]
- `TestLeakagePrevention` --method--> `.test_first_lap_race_normalization_is_nan()` [1.0]
- `TestLeakagePrevention` --method--> `.test_degradation_rate_nan_when_insufficient_data()` [1.0]

### Cross-community
- Tests validate the [Lap Feature Engineering](Lap_Feature_Engineering.md) module
- Leakage prevention tests relate to the [Common Feature Utilities](Common_Feature_Utilities.md) shift(1) pattern
- Feature output tested here feeds into the [Feature Build Functions](Feature_Build_Functions.md) pipeline

## Source Files
- `tests/test_features/test_lap_features.py`
