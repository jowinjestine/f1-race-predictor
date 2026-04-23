# Feature Build Functions

## Overview
This community represents the complete model training pipeline, from feature building to model persistence. It defines all four models (A: Lap+Tyre, B: Lap No-Tyre, C: Pre-Race, D: Stacking Meta-Model), the model comparison notebook, training scripts (local and remote GCP), and the storage functions they consume. It implements a Three-Round Model Tournament Protocol and uses GCS-with-Local-Fallback patterns for data loading.

## Key Components
| Component | Type | Source File |
|-----------|------|-------------|
| Model A: Lap + Tyre (2019-2024) | Model | scripts/make_training_notebooks.py |
| Model B: Lap No-Tyre (2018-2025) | Model | scripts/make_training_notebooks.py |
| Model C: Pre-Race Features (2018-2025) | Model | scripts/make_training_notebooks.py |
| Model D: Stacking Meta-Model | Model | scripts/make_training_notebooks.py |
| Model Comparison Notebook (10 Criteria) | Notebook | scripts/make_training_notebooks.py |
| Make Training Notebooks Script | Script | scripts/make_training_notebooks.py |
| Remote GCP Training Script | Script | scripts/run_training_remote.sh |
| Fetch Training Results Script | Script | scripts/fetch_training_results.sh |
| Load from GCS or Local Fallback | Function | src/f1_predictor/data/storage.py |
| Save Training Parquet | Function | src/f1_predictor/data/storage.py |
| Save Model Pickle to GCS | Function | src/f1_predictor/data/storage.py |
| Sync Training from GCS | Function | src/f1_predictor/data/storage.py |
| Ensure Latest Data from GCS | Function | src/f1_predictor/data/storage.py |
| LeaveOneSeasonOut CV Splitter | Class | src/f1_predictor/features/splits.py |
| ExpandingWindowSplit CV Splitter | Class | src/f1_predictor/features/splits.py |
| Build Lap Tyre Features (Model A) | Function | src/f1_predictor/features/lap_features.py |
| Build Lap No-Tyre Features (Model B) | Function | src/f1_predictor/features/lap_features.py |
| Build Race Features (Model C) | Function | src/f1_predictor/features/race_features.py |
| Core Lap Feature Builder | Function | src/f1_predictor/features/lap_features.py |
| Compute Tyre Degradation Rate | Function | src/f1_predictor/features/lap_features.py |
| Compute Compound Pace Delta | Function | src/f1_predictor/features/lap_features.py |
| Three-Round Model Tournament Protocol | Pattern | scripts/make_training_notebooks.py |
| GCS-with-Local-Fallback Pattern | Pattern | src/f1_predictor/data/storage.py |
| Rolling Mean by Group | Function | src/f1_predictor/features/common.py |
| Encode Compound One-Hot | Function | src/f1_predictor/features/common.py |

## Relationships

### Internal
- `Make Training Notebooks Script` --calls--> `Model A`, `Model B`, `Model C`, `Model D`, `Model Comparison Notebook` [1.0]
- `Model A` --calls--> `Save Training Parquet`, `Save Model Pickle to GCS`, `Load from GCS or Local Fallback` [1.0]
- `Model A` --references--> `LeaveOneSeasonOut CV Splitter`, `Build Lap Tyre Features (Model A)` [1.0/0.85]
- `Model B` --references--> `ExpandingWindowSplit CV Splitter`, `Build Lap No-Tyre Features (Model B)` [1.0/0.85]
- `Model C` --references--> `ExpandingWindowSplit CV Splitter`, `Build Race Features (Model C)` [1.0/0.85]
- `Model D` --calls--> `Sync Training from GCS` [1.0]
- `Model D` --references--> `Model A`, `Model B`, `Model C` [1.0]
- `Three-Round Model Tournament Protocol` --implements--> `Model A`, `Model B`, `Model C` [1.0]
- `Core Lap Feature Builder` --calls--> `Rolling Mean by Group`, `Encode Compound One-Hot`, `Compute Tyre Degradation Rate`, `Compute Compound Pace Delta` [1.0]
- `Remote GCP Training Script` --calls--> `Make Training Notebooks Script`, `Model A`, `Model B` [1.0]
- `GCS-with-Local-Fallback Pattern` --implements--> `Load from GCS or Local Fallback`, `Ensure Latest Data from GCS` [1.0]

### Cross-community
- Storage functions are defined in [GCS Storage Layer](GCS_Storage_Layer.md)
- CV splitters are defined in [CV Splits & Init](CV_Splits_Init.md)
- Lap features are detailed in [Lap Feature Engineering](Lap_Feature_Engineering.md)
- Race features are detailed in [Race Feature Engineering](Race_Feature_Engineering.md)
- Common utilities in [Common Feature Utilities](Common_Feature_Utilities.md)
- Fetch Training Results connects to [Data Collection Core](Data_Collection_Core.md) GCS Storage Module

## Source Files
- `scripts/make_training_notebooks.py`
- `scripts/run_training_remote.sh`
- `scripts/fetch_training_results.sh`
- `src/f1_predictor/data/storage.py`
- `src/f1_predictor/features/splits.py`
- `src/f1_predictor/features/lap_features.py`
- `src/f1_predictor/features/race_features.py`
- `src/f1_predictor/features/common.py`
