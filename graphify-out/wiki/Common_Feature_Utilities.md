# Common Feature Utilities

## Overview
This community provides the shared utility functions used across all feature engineering modules in the F1 predictor. The core design principle is data leakage prevention via `shift(1)` on all grouped rolling and expanding window operations. It also includes safe arithmetic helpers and tire compound encoding used by both lap-level (Models A/B) and race-level (Model C) feature pipelines.

## Key Components
| Component | Type | Source File |
|-----------|------|-------------|
| `common.py` | Module | `src/f1_predictor/features/common.py` |
| `safe_divide()` | Function | `src/f1_predictor/features/common.py` |
| `rolling_mean_by_group()` | Function | `src/f1_predictor/features/common.py` |
| `expanding_mean_by_group()` | Function | `src/f1_predictor/features/common.py` |
| `expanding_sum_by_group()` | Function | `src/f1_predictor/features/common.py` |
| `expanding_count_by_group()` | Function | `src/f1_predictor/features/common.py` |
| `rolling_sum_by_group()` | Function | `src/f1_predictor/features/common.py` |
| `encode_compound_onehot()` | Function | `src/f1_predictor/features/common.py` |

## Relationships

### Internal
- `common.py` --contains--> `safe_divide()` [1.0]
- `common.py` --contains--> `rolling_mean_by_group()` [1.0]
- `common.py` --contains--> `expanding_mean_by_group()` [1.0]
- `common.py` --contains--> `expanding_sum_by_group()` [1.0]
- `common.py` --contains--> `expanding_count_by_group()` [1.0]
- `common.py` --contains--> `rolling_sum_by_group()` [1.0]
- `common.py` --contains--> `encode_compound_onehot()` [1.0]

### Cross-community
- Tested by the [Common Feature Tests](Common_Feature_Tests.md) community
- `rolling_mean_by_group()` and `expanding_mean_by_group()` are used by the [Race Feature Engineering](Race_Feature_Engineering.md) pipeline
- `rolling_mean_by_group()` is used by the [Lap Feature Engineering](Lap_Feature_Engineering.md) pipeline
- `safe_divide()` is used by the [Race Feature Functions](Race_Feature_Functions.md) community
- `encode_compound_onehot()` is used by the [Lap Feature Engineering](Lap_Feature_Engineering.md) for tyre data encoding
- Implements the data leakage prevention pattern documented in [Race Feature Functions](Race_Feature_Functions.md)

## Source Files
- `src/f1_predictor/features/common.py`
