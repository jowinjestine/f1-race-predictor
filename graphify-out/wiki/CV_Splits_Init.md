# CV Splits & Init

## Overview
This community defines the cross-validation splitting strategies used across all four models in the F1 predictor, along with package initialization files. It contains `LeaveOneSeasonOut` (used by Model A, 2019-2024) and `ExpandingWindowSplit` (used by Models B and C, 2018-2025), plus their corresponding test suites that validate fold counts, temporal ordering, and test-season exclusion.

## Key Components
| Component | Type | Source File |
|-----------|------|-------------|
| splits.py | Module | src/f1_predictor/features/splits.py |
| LeaveOneSeasonOut | Class | src/f1_predictor/features/splits.py |
| ExpandingWindowSplit | Class | src/f1_predictor/features/splits.py |
| test_splits.py | Test Module | tests/test_features/test_splits.py |
| TestLeaveOneSeasonOut | Test Class | tests/test_features/test_splits.py |
| TestExpandingWindowSplit | Test Class | tests/test_features/test_splits.py |
| __init__.py (f1_predictor) | Init | src/f1_predictor/__init__.py |
| __init__.py (features) | Init | src/f1_predictor/features/__init__.py |

## Relationships

### Internal
- `splits.py` --contains--> `LeaveOneSeasonOut` [1.0]
- `splits.py` --contains--> `ExpandingWindowSplit` [1.0]
- `test_splits.py` --contains--> `TestLeaveOneSeasonOut` [1.0]
- `test_splits.py` --contains--> `TestExpandingWindowSplit` [1.0]
- `TestLeaveOneSeasonOut` --uses--> `LeaveOneSeasonOut` [0.5]
- `TestExpandingWindowSplit` --uses--> `ExpandingWindowSplit` [0.5]
- `LeaveOneSeasonOut` has methods: `.__init__()`, `.split()`, `.get_n_splits()`, `.get_test_split()`
- `ExpandingWindowSplit` has methods: `.__init__()`, `.split()`, `.get_n_splits()`, `.get_test_split()`

### Cross-community
- Both splitters are consumed by [Feature Build Functions](Feature_Build_Functions.md) (Model A uses LeaveOneSeasonOut, Models B/C use ExpandingWindowSplit)
- The `__init__.py` file describes the feature engineering pipeline shared with [Lap Feature Engineering](Lap_Feature_Engineering.md) and [Race Feature Engineering](Race_Feature_Engineering.md)
- Expanding window concept connects to the Data Leakage Prevention Pattern in [Feature Module Concepts](Feature_Module_Concepts.md)

## Source Files
- `src/f1_predictor/features/splits.py`
- `src/f1_predictor/__init__.py`
- `src/f1_predictor/features/__init__.py`
- `tests/test_features/test_splits.py`
