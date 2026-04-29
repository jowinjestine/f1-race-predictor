# GCS Storage Layer

## Overview
This community implements the Google Cloud Storage persistence layer for the F1 predictor. It provides functions for uploading/downloading parquet files, syncing season data, saving model pickles and notebooks, and a GCS-with-local-fallback pattern used by training notebooks. The `get_client()` function serves as the central GCS client factory that most other storage functions depend on.

## Key Components
| Component | Type | Source File |
|-----------|------|-------------|
| storage.py | Module | src/f1_predictor/data/storage.py |
| get_client() | Function | src/f1_predictor/data/storage.py |
| upload_parquet() | Function | src/f1_predictor/data/storage.py |
| download_parquet() | Function | src/f1_predictor/data/storage.py |
| read_parquet_from_gcs() | Function | src/f1_predictor/data/storage.py |
| upload_season_files() | Function | src/f1_predictor/data/storage.py |
| list_remote_seasons() | Function | src/f1_predictor/data/storage.py |
| sync_to_local() | Function | src/f1_predictor/data/storage.py |
| load_from_gcs_or_local() | Function | src/f1_predictor/data/storage.py |
| list_blobs() | Function | src/f1_predictor/data/storage.py |
| upload_blob() | Function | src/f1_predictor/data/storage.py |
| download_blob() | Function | src/f1_predictor/data/storage.py |
| save_training_parquet() | Function | src/f1_predictor/data/storage.py |
| save_model_pickle() | Function | src/f1_predictor/data/storage.py |
| save_notebook() | Function | src/f1_predictor/data/storage.py |
| load_training_parquet() | Function | src/f1_predictor/data/storage.py |
| sync_training_from_gcs() | Function | src/f1_predictor/data/storage.py |
| ensure_latest() | Function | src/f1_predictor/data/storage.py |

## Relationships

### Internal
- `get_client()` --calls--> `upload_parquet()`, `download_parquet()`, `list_remote_seasons()`, `list_blobs()`, `upload_blob()`, `download_blob()` [1.0]
- `upload_season_files()` --calls--> `upload_parquet()` [1.0]
- `sync_to_local()` --calls--> `download_parquet()`, `list_remote_seasons()` [1.0]
- `load_from_gcs_or_local()` --calls--> `read_parquet_from_gcs()`, `load_training_parquet()` [1.0]
- `save_training_parquet()` --calls--> `upload_blob()` [1.0]
- `save_model_pickle()` --calls--> `upload_blob()` [1.0]
- `save_notebook()` --calls--> `upload_blob()` [1.0]
- `sync_training_from_gcs()` --calls--> `list_blobs()`, `download_blob()` [1.0]
- `ensure_latest()` --calls--> `sync_to_local()` [1.0]

### Cross-community
- `save_training_parquet()`, `save_model_pickle()`, and `load_from_gcs_or_local()` are consumed by [Feature Build Functions](Feature_Build_Functions.md) for model training pipelines
- `upload_season_files()` is called by [Data Collection Core](Data_Collection_Core.md) after collecting season data
- Storage tests are in [Storage Layer Tests](Storage_Layer_Tests.md) (TestEnsureLatest, TestSyncToLocal)

## Source Files
- `src/f1_predictor/data/storage.py`
