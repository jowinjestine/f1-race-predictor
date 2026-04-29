# Graph Report - .  (2026-04-23)

## Corpus Check
- Corpus is ~17,694 words - fits in a single context window. You may not need a graph.

## Summary
- 536 nodes · 673 edges · 30 communities detected
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 27 edges (avg confidence: 0.71)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Race Feature Tests|Race Feature Tests]]
- [[_COMMUNITY_CV Splits & Init|CV Splits & Init]]
- [[_COMMUNITY_Lap Collection Tests|Lap Collection Tests]]
- [[_COMMUNITY_GCS Storage Layer|GCS Storage Layer]]
- [[_COMMUNITY_Data Collection Tests|Data Collection Tests]]
- [[_COMMUNITY_Data Collection Core|Data Collection Core]]
- [[_COMMUNITY_Lap Feature Engineering|Lap Feature Engineering]]
- [[_COMMUNITY_Feature Module Concepts|Feature Module Concepts]]
- [[_COMMUNITY_Jolpica API Tests|Jolpica API Tests]]
- [[_COMMUNITY_Feature Build Functions|Feature Build Functions]]
- [[_COMMUNITY_Lap Collection Functions|Lap Collection Functions]]
- [[_COMMUNITY_Lap Feature Tests|Lap Feature Tests]]
- [[_COMMUNITY_Collection Module Structure|Collection Module Structure]]
- [[_COMMUNITY_Season Collection Pipeline|Season Collection Pipeline]]
- [[_COMMUNITY_Common Feature Tests|Common Feature Tests]]
- [[_COMMUNITY_Race Feature Engineering|Race Feature Engineering]]
- [[_COMMUNITY_Jolpica API Client|Jolpica API Client]]
- [[_COMMUNITY_Common Feature Utilities|Common Feature Utilities]]
- [[_COMMUNITY_Race Feature Functions|Race Feature Functions]]
- [[_COMMUNITY_Notebook Generation|Notebook Generation]]
- [[_COMMUNITY_Storage Layer Tests|Storage Layer Tests]]
- [[_COMMUNITY_API Package Init|API Package Init]]
- [[_COMMUNITY_Data Package Init|Data Package Init]]
- [[_COMMUNITY_Explain Package Init|Explain Package Init]]
- [[_COMMUNITY_Models Package Init|Models Package Init]]
- [[_COMMUNITY_Tests Package Init|Tests Package Init]]
- [[_COMMUNITY_Feature Tests Init|Feature Tests Init]]
- [[_COMMUNITY_F1 Predictor Package|F1 Predictor Package]]
- [[_COMMUNITY_Features Init|Features Init]]
- [[_COMMUNITY_Common Utils|Common Utils]]

## God Nodes (most connected - your core abstractions)
1. `_make_races()` - 27 edges
2. `_make_laps()` - 18 edges
3. `_build_lap_features()` - 11 edges
4. `TestExpandingWindowSplit` - 11 edges
5. `LeaveOneSeasonOut` - 10 edges
6. `ExpandingWindowSplit` - 10 edges
7. `TestNormalizeCompound` - 10 edges
8. `TestLeaveOneSeasonOut` - 10 edges
9. `Jolpica API Client Module` - 10 edges
10. `Model A: Lap + Tyre (2019-2024)` - 9 edges

## Surprising Connections (you probably didn't know these)
- `Fetch Training Results Script` --shares_data_with--> `GCS Storage Module`  [INFERRED]
  scripts/fetch_training_results.sh → src/f1_predictor/data/storage.py
- `test_version()` --references--> `F1 Race Predictor Project Overview`  [INFERRED]
  tests\test_smoke.py → README.md
- `test_import_subpackages()` --references--> `Project Structure`  [INFERRED]
  tests\test_smoke.py → README.md
- `Race-Level Data Schema` --references--> `Collect All Seasons`  [INFERRED]
  docs/data_dictionary.md → src/f1_predictor/data/collect.py
- `Race-Level Data Schema` --references--> `Add Target Variables`  [INFERRED]
  docs/data_dictionary.md → src/f1_predictor/data/collect.py

## Hyperedges (group relationships)
- **Data Collection Pipeline (FastF1 + Jolpica + Weather)** — collect_all, collect_season, collect_all_laps, collect_laps_fastf1, collect_laps_jolpica, backfill_qualifying, get_openmeteo_weather, fastf1_api, jolpica_api, openmeteo_api [EXTRACTED 0.95]
- **Model A/B/C/D Training and Stacking Ensemble** — model_a, model_b, model_c, model_d, three_round_tournament, leave_one_season_out, expanding_window_split [EXTRACTED 0.95]
- **Feature Engineering Pipeline (Lap + Race Level)** — build_lap_tyre_features, build_lap_notyre_features, build_race_features, common_utils, rolling_mean_by_group, expanding_mean_by_group, data_leakage_prevention [EXTRACTED 0.90]
- **Data Leakage Prevention Testing Pattern** — concept_leakage_prevention, test_lap_features_TestLeakagePrevention, test_race_features_TestLeakagePrevention, concept_expanding_window_cv [EXTRACTED 0.90]
- **Dual-Source Weather Data Pipeline** — test_collect_TestAggregateFastf1Weather, test_collect_TestGetOpenmeteoWeather, test_race_features_TestWeatherFeatures, concept_weather_dual_source [EXTRACTED 0.90]
- **Tyre Compound Normalization and Encoding Flow** — test_collect_laps_TestNormalizeCompound, test_common_TestEncodeCompoundOnehot, test_lap_features_TestBuildLapTyreFeatures [INFERRED 0.80]

## Communities

### Community 0 - "Race Feature Tests"
Cohesion: 0.08
Nodes (13): _make_races(), Tests for pre-race feature engineering (Model C)., Create minimal race data for testing., Verify that removing a future race doesn't change past features., Removing a future race shouldn't change past weather values., TestBuildRaceFeatures, TestCircuitFeatures, TestDriverCircuitFeatures (+5 more)

### Community 1 - "CV Splits & Init"
Cohesion: 0.06
Nodes (13): Feature engineering pipeline for F1 race prediction models., ExpandingWindowSplit, LeaveOneSeasonOut, Cross-validation splitters for temporal F1 data., Leave-one-season-out CV for Model A (2019-2024).      Each fold holds out one se, Yield (train_indices, val_indices) for each fold., Return (train_indices, test_indices) for final evaluation., Expanding-window CV for Models B and C (2018-2025).      Each fold uses a growin (+5 more)

### Community 2 - "Lap Collection Tests"
Cohesion: 0.06
Nodes (8): Tests for lap-level data collection module., TestAddPitDuration, TestBuildDriverIdToCode, TestBuildDriverIdToTeam, TestBuildPitstopMap, TestFastf1LapRow, TestNormalizeCompound, TestSafeInt

### Community 3 - "GCS Storage Layer"
Cohesion: 0.09
Nodes (34): download_blob(), download_parquet(), ensure_latest(), get_client(), list_blobs(), list_remote_seasons(), load_from_gcs_or_local(), load_training_parquet() (+26 more)

### Community 4 - "Data Collection Tests"
Cohesion: 0.06
Nodes (7): Tests for data collection module., TestAddTargetVariables, TestAggregateFastf1Weather, TestBackfillQualifying, TestFirst, TestGetOpenmeteoWeather, TestTdToSeconds

### Community 5 - "Data Collection Core"
Cohesion: 0.07
Nodes (30): Add Target Variables, Backfill Qualifying from Jolpica, Build Pitstop Map, Circuit Coordinates Lookup, Collect All Seasons, Collect All Laps, Collect Laps FastF1, Collect Laps Jolpica (+22 more)

### Community 6 - "Lap Feature Engineering"
Cohesion: 0.11
Nodes (25): _add_gap_features(), _add_pit_features(), _add_position_features(), _add_race_normalization(), _add_race_progress(), _add_rolling_pace(), _add_tyre_features(), _build_lap_features() (+17 more)

### Community 7 - "Feature Module Concepts"
Cohesion: 0.08
Nodes (26): Expanding Window Cross-Validation, Data Leakage Prevention Pattern, Common Feature Utilities Module, Lap Feature Engineering Module, Race Feature Engineering Module, CV Splits Module, TestEncodeCompoundOnehot, TestExpandingCountByGroup (+18 more)

### Community 8 - "Jolpica API Tests"
Cohesion: 0.08
Nodes (9): Tests for Jolpica API client., TestGetLaps, TestGetPitstops, TestGetQualifyingResults, TestGetRaceResults, TestGetSeasonSchedule, TestParseLapTime, TestParseRaceTimeMillis (+1 more)

### Community 9 - "Feature Build Functions"
Cohesion: 0.13
Nodes (25): Core Lap Feature Builder, Build Lap No-Tyre Features (Model B), Build Lap Tyre Features (Model A), Compute Compound Pace Delta, Compute Tyre Degradation Rate, Encode Compound One-Hot, Ensure Latest Data from GCS, ExpandingWindowSplit CV Splitter (+17 more)

### Community 10 - "Lap Collection Functions"
Cohesion: 0.12
Nodes (23): add_pit_duration(), _build_driver_id_to_code(), _build_driver_id_to_team(), _build_pitstop_map(), collect_all_laps(), collect_laps_fastf1(), collect_laps_jolpica(), _fastf1_lap_row() (+15 more)

### Community 11 - "Lap Feature Tests"
Cohesion: 0.14
Nodes (7): _make_laps(), Tests for lap-level feature engineering (Models A and B)., Removing future laps shouldn't change past normalization., Create minimal lap data for testing., TestBuildLapNotyreFeatures, TestBuildLapTyreFeatures, TestLeakagePrevention

### Community 12 - "Collection Module Structure"
Cohesion: 0.09
Nodes (24): Lap-Level Data Collection Module, Data Collection Module, Dual Weather Source Strategy, Jolpica API Client Module, TestAddTargetVariables (collect tests), TestAggregateFastf1Weather, TestBackfillQualifying, TestFirst (collect tests) (+16 more)

### Community 13 - "Season Collection Pipeline"
Cohesion: 0.13
Nodes (22): add_target_variables(), _aggregate_fastf1_weather(), backfill_qualifying(), collect_all(), collect_season(), _empty_f1_weather(), ensure_cache(), _first() (+14 more)

### Community 14 - "Common Feature Tests"
Cohesion: 0.09
Nodes (8): Tests for shared feature engineering utilities., TestEncodeCompoundOnehot, TestExpandingCountByGroup, TestExpandingMeanByGroup, TestExpandingSumByGroup, TestRollingMeanByGroup, TestRollingSumByGroup, TestSafeDivide

### Community 15 - "Race Feature Engineering"
Cohesion: 0.14
Nodes (19): _add_circuit_features(), _add_driver_circuit_features(), _add_qualifying_features(), _add_season_form_features(), _add_team_form_features(), _add_weather_features(), build_race_features(), _compute_position_trend() (+11 more)

### Community 16 - "Jolpica API Client"
Cohesion: 0.16
Nodes (18): collect_season_jolpica(), _get_json(), get_laps(), get_pitstops(), get_qualifying_results(), get_race_results(), get_season_schedule(), parse_lap_time() (+10 more)

### Community 17 - "Common Feature Utilities"
Cohesion: 0.12
Nodes (15): encode_compound_onehot(), expanding_count_by_group(), expanding_mean_by_group(), expanding_sum_by_group(), Shared utilities for feature engineering., One-hot encode tire compound into 5 boolean columns., Element-wise division returning NaN when denominator is zero., Grouped rolling mean with shift(1) to prevent data leakage.      The current row (+7 more)

### Community 18 - "Race Feature Functions"
Cohesion: 0.15
Nodes (16): Add Circuit Type Features, Add Driver Circuit History Features, Add Qualifying Features, Add Season Form Features, Add Team Form Features, Add Weather Features, Build Race Features (Model C), Data Leakage Prevention Pattern (+8 more)

### Community 19 - "Notebook Generation"
Cohesion: 0.44
Nodes (11): code(), main(), make_cell(), make_model_a(), make_model_b(), make_model_c(), make_model_comparison(), make_model_d() (+3 more)

### Community 20 - "Storage Layer Tests"
Cohesion: 0.18
Nodes (3): Tests for GCS storage module., TestEnsureLatest, TestSyncToLocal

### Community 21 - "API Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 22 - "Data Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 23 - "Explain Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 24 - "Models Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 25 - "Tests Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 26 - "Feature Tests Init"
Cohesion: 1.0
Nodes (0): 

### Community 27 - "F1 Predictor Package"
Cohesion: 1.0
Nodes (1): F1 Predictor Package

### Community 28 - "Features Init"
Cohesion: 1.0
Nodes (1): Features Package Init

### Community 29 - "Common Utils"
Cohesion: 1.0
Nodes (1): Common Feature Utilities

## Knowledge Gaps
- **162 isolated node(s):** `Generate training notebooks for Models A, B, C, D.`, `Collect F1 race results and weather data for 2018-2025 seasons.  Uses FastF1 f`, `Fetch race-day weather from Open-Meteo archive API.`, `Collect all race results and weather for a single season.`, `Convert to float, returning None for NaN.` (+157 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `API Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Data Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Explain Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Models Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tests Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Feature Tests Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `F1 Predictor Package`** (1 nodes): `F1 Predictor Package`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Features Init`** (1 nodes): `Features Package Init`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Common Utils`** (1 nodes): `Common Feature Utilities`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GCS Storage Module` connect `Data Collection Core` to `Feature Build Functions`?**
  _High betweenness centrality (0.007) - this node is a cross-community bridge._
- **Why does `Build Race Features (Model C)` connect `Race Feature Functions` to `Feature Build Functions`?**
  _High betweenness centrality (0.007) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `TestExpandingWindowSplit` (e.g. with `ExpandingWindowSplit` and `LeaveOneSeasonOut`) actually correct?**
  _`TestExpandingWindowSplit` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `LeaveOneSeasonOut` (e.g. with `Feature engineering pipeline for F1 race prediction models.` and `TestLeaveOneSeasonOut`) actually correct?**
  _`LeaveOneSeasonOut` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Generate training notebooks for Models A, B, C, D.`, `Collect F1 race results and weather data for 2018-2025 seasons.  Uses FastF1 f`, `Fetch race-day weather from Open-Meteo archive API.` to the rest of the system?**
  _162 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Race Feature Tests` be split into smaller, more focused modules?**
  _Cohesion score 0.08 - nodes in this community are weakly interconnected._
- **Should `CV Splits & Init` be split into smaller, more focused modules?**
  _Cohesion score 0.06 - nodes in this community are weakly interconnected._