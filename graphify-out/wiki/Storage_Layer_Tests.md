# Storage Layer Tests

## Overview
This community contains the test suite for the GCS (Google Cloud Storage) storage module. The tests verify the `ensure_latest()` and `sync_to_local()` functions, covering scenarios like reading combined Parquet files, falling back to per-season files, handling GCS failures gracefully, downloading missing files, skipping existing local files, and returning empty results when no data is available.

## Key Components
| Component | Type | Source File |
|-----------|------|-------------|
| `test_storage.py` | Test Module | `tests/test_storage.py` |
| `TestEnsureLatest` | Test Class | `tests/test_storage.py` |
| `test_reads_combined_file()` | Test Function | `tests/test_storage.py` |
| `test_falls_back_to_season_files()` | Test Function | `tests/test_storage.py` |
| `test_handles_gcs_failure()` | Test Function | `tests/test_storage.py` |
| `test_returns_empty_when_nothing_available()` | Test Function | `tests/test_storage.py` |
| `TestSyncToLocal` | Test Class | `tests/test_storage.py` |
| `test_downloads_missing_files()` | Test Function | `tests/test_storage.py` |
| `test_skips_existing_files()` | Test Function | `tests/test_storage.py` |
| `test_returns_empty_when_no_remote()` | Test Function | `tests/test_storage.py` |

## Relationships

### Internal
- `test_storage.py` --contains--> `TestEnsureLatest` [1.0]
- `test_storage.py` --contains--> `TestSyncToLocal` [1.0]
- `test_storage.py` --contains--> `test_reads_combined_file()` [1.0]
- `test_storage.py` --contains--> `test_falls_back_to_season_files()` [1.0]
- `test_storage.py` --contains--> `test_handles_gcs_failure()` [1.0]
- `test_storage.py` --contains--> `test_returns_empty_when_nothing_available()` [1.0]
- `test_storage.py` --contains--> `test_downloads_missing_files()` [1.0]
- `test_storage.py` --contains--> `test_skips_existing_files()` [1.0]
- `test_storage.py` --contains--> `test_returns_empty_when_no_remote()` [1.0]

### Cross-community
- Tests validate the [GCS Storage Layer](GCS_Storage_Layer.md) module
- Storage functions are used by the [Season Collection Pipeline](Season_Collection_Pipeline.md) for uploads
- Storage functions are used by the [Lap Collection Functions](Lap_Collection_Functions.md) for lap data uploads
- Storage is consumed by the [Notebook Generation](Notebook_Generation.md) pipeline for data loading

## Source Files
- `tests/test_storage.py`
