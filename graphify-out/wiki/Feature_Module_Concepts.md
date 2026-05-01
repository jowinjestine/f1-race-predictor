# Feature Module Concepts

## Overview
This community captures the high-level conceptual architecture of the feature engineering and testing modules. It maps test classes to their corresponding production modules (common utilities, lap features, race features, CV splits) and defines cross-cutting patterns like Data Leakage Prevention and Expanding Window Cross-Validation. It also includes the Dual Weather Source Strategy concept for weather feature robustness.

## Key Components
| Component | Type | Source File |
|-----------|------|-------------|
| Common Feature Utilities Module | Module Ref | tests/test_features/test_common.py |
| Lap Feature Engineering Module | Module Ref | tests/test_features/test_lap_features.py |
| Race Feature Engineering Module | Module Ref | tests/test_features/test_race_features.py |
| CV Splits Module | Module Ref | tests/test_features/test_splits.py |
| Data Leakage Prevention Pattern | Concept | tests/test_features/test_lap_features.py |
| Expanding Window Cross-Validation | Concept | tests/test_features/test_splits.py |
| TestSafeDivide (common features) | Test Class | tests/test_features/test_common.py |
| TestRollingMeanByGroup | Test Class | tests/test_features/test_common.py |
| TestExpandingMeanByGroup | Test Class | tests/test_features/test_common.py |
| TestExpandingSumByGroup | Test Class | tests/test_features/test_common.py |
| TestExpandingCountByGroup | Test Class | tests/test_features/test_common.py |
| TestRollingSumByGroup | Test Class | tests/test_features/test_common.py |
| TestEncodeCompoundOnehot | Test Class | tests/test_features/test_common.py |
| TestBuildLapTyreFeatures | Test Class | tests/test_features/test_lap_features.py |
| TestBuildLapNotyreFeatures | Test Class | tests/test_features/test_lap_features.py |
| TestLeakagePrevention (lap features) | Test Class | tests/test_features/test_lap_features.py |
| TestLeakagePrevention (race features) | Test Class | tests/test_features/test_race_features.py |
| TestLeaveOneSeasonOut | Test Class | tests/test_features/test_splits.py |
| TestExpandingWindowSplit | Test Class | tests/test_features/test_splits.py |

## Relationships

### Internal
- `Common Feature Utilities Module` --calls--> `TestSafeDivide`, `TestRollingMeanByGroup`, `TestExpandingMeanByGroup`, `TestExpandingSumByGroup`, `TestExpandingCountByGroup`, `TestRollingSumByGroup`, `TestEncodeCompoundOnehot` [1.0]
- `Lap Feature Engineering Module` --calls--> `TestBuildLapTyreFeatures`, `TestBuildLapNotyreFeatures`, `TestLeakagePrevention (lap features)` [1.0]
- `Race Feature Engineering Module` --calls--> `TestBuildRaceFeatures`, `TestQualifyingFeatures`, `TestSeasonFormFeatures`, etc. [1.0]
- `CV Splits Module` --calls--> `TestLeaveOneSeasonOut`, `TestExpandingWindowSplit` [1.0]
- `Data Leakage Prevention Pattern` --implements--> `TestLeakagePrevention (lap features)` [1.0]
- `Data Leakage Prevention Pattern` --implements--> `TestLeakagePrevention (race features)` [1.0]
- `Data Leakage Prevention Pattern` --semantically_similar_to--> `Expanding Window Cross-Validation` [0.75]

### Cross-community
- `Common Feature Utilities Module` --shares_data_with--> [Lap Feature Engineering](Lap_Feature_Engineering.md) [0.85]
- `Common Feature Utilities Module` --shares_data_with--> [Race Feature Engineering](Race_Feature_Engineering.md) [0.8]
- `TestEncodeCompoundOnehot` --semantically_similar_to--> `TestNormalizeCompound` in [Lap Collection Tests](Lap_Collection_Tests.md) [0.7]
- Detailed test implementations are in [Race Feature Tests](Race_Feature_Tests.md), [Lap Feature Tests](Lap_Feature_Tests.md), [Common Feature Tests](Common_Feature_Tests.md), and [CV Splits & Init](CV_Splits_Init.md)

## Source Files
- `tests/test_features/test_common.py`
- `tests/test_features/test_lap_features.py`
- `tests/test_features/test_race_features.py`
- `tests/test_features/test_splits.py`
