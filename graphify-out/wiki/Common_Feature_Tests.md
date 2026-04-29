# Common Feature Tests

## Overview
This community contains the test suite for shared feature engineering utility functions defined in `common.py`. The tests cover safe division, grouped rolling/expanding mean/sum/count operations (all with shift(1) leakage prevention), and tire compound one-hot encoding. These utilities underpin both the lap-level and race-level feature pipelines.

## Key Components
| Component | Type | Source File |
|-----------|------|-------------|
| `test_common.py` | Test Module | `tests/test_features/test_common.py` |
| `TestSafeDivide` | Test Class | `tests/test_features/test_common.py` |
| `TestRollingMeanByGroup` | Test Class | `tests/test_features/test_common.py` |
| `TestExpandingMeanByGroup` | Test Class | `tests/test_features/test_common.py` |
| `TestExpandingSumByGroup` | Test Class | `tests/test_features/test_common.py` |
| `TestExpandingCountByGroup` | Test Class | `tests/test_features/test_common.py` |
| `TestRollingSumByGroup` | Test Class | `tests/test_features/test_common.py` |
| `TestEncodeCompoundOnehot` | Test Class | `tests/test_features/test_common.py` |

## Relationships

### Internal
- `test_common.py` --contains--> `TestSafeDivide` [1.0]
- `test_common.py` --contains--> `TestRollingMeanByGroup` [1.0]
- `test_common.py` --contains--> `TestExpandingMeanByGroup` [1.0]
- `test_common.py` --contains--> `TestExpandingSumByGroup` [1.0]
- `test_common.py` --contains--> `TestExpandingCountByGroup` [1.0]
- `test_common.py` --contains--> `TestRollingSumByGroup` [1.0]
- `test_common.py` --contains--> `TestEncodeCompoundOnehot` [1.0]
- `TestSafeDivide` --method--> `.test_normal_division()`, `.test_zero_denominator_returns_nan()`, `.test_nan_numerator()` [1.0]
- `TestRollingMeanByGroup` --method--> `.test_basic_rolling_mean()`, `.test_shift_prevents_leakage()`, `.test_multiple_groups()` [1.0]
- `TestExpandingMeanByGroup` --method--> `.test_expanding_mean()`, `.test_groups_are_independent()` [1.0]
- `TestEncodeCompoundOnehot` --method--> `.test_all_compounds()`, `.test_unknown_compound_is_all_zeros()`, `.test_null_compound()` [1.0]

### Cross-community
- Tests validate the [Common Feature Utilities](Common_Feature_Utilities.md) module (`src/f1_predictor/features/common.py`)
- `TestEncodeCompoundOnehot` --semantically_similar_to--> `TestNormalizeCompound` in [Collection Module Structure](Collection_Module_Structure.md) [0.7]
- Rolling/expanding tests verify leakage prevention used by [Race Feature Engineering](Race_Feature_Engineering.md) and [Lap Feature Engineering](Lap_Feature_Engineering.md)

## Source Files
- `tests/test_features/test_common.py`
