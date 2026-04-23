# F1 Race Position Predictor — Knowledge Graph Wiki

**536 nodes · 673 edges · 30 communities**

## Architecture Overview

This codebase predicts F1 race finishing positions using a multi-model stacking ensemble (Models A/B/C/D) trained on FastF1 and Jolpica API data from 2018–2025 seasons, with features engineered at both lap and race levels.

## Community Index

### Core Pipeline
- [Data Collection Core](Data_Collection_Core.md) — FastF1 + Jolpica + weather data collection
- [Season Collection Pipeline](Season_Collection_Pipeline.md) — Per-season data orchestration
- [Jolpica API Client](Jolpica_API_Client.md) — REST client for Jolpica F1 data API
- [Lap Collection Functions](Lap_Collection_Functions.md) — Lap-level data extraction from FastF1
- [GCS Storage Layer](GCS_Storage_Layer.md) — Google Cloud Storage persistence layer

### Feature Engineering
- [Lap Feature Engineering](Lap_Feature_Engineering.md) — Lap-level features for Models A & B
- [Race Feature Engineering](Race_Feature_Engineering.md) — Race-level features for Model C
- [Common Feature Utilities](Common_Feature_Utilities.md) — Shared rolling/expanding aggregators
- [Feature Build Functions](Feature_Build_Functions.md) — Model training pipeline & stacking ensemble
- [Race Feature Functions](Race_Feature_Functions.md) — Circuit, qualifying, weather, form features
- [CV Splits & Init](CV_Splits_Init.md) — LeaveOneSeasonOut & ExpandingWindowSplit splitters
- [Feature Module Concepts](Feature_Module_Concepts.md) — High-level module concepts & leakage prevention

### Model Training
- [Notebook Generation](Notebook_Generation.md) — Programmatic training notebook generation (Models A–D)

### Module Structure
- [Collection Module Structure](Collection_Module_Structure.md) — Module-level architecture for data collection

### Test Suites
- [Race Feature Tests](Race_Feature_Tests.md) — Tests for race-level feature engineering
- [Lap Feature Tests](Lap_Feature_Tests.md) — Tests for lap-level feature engineering
- [Common Feature Tests](Common_Feature_Tests.md) — Tests for shared feature utilities
- [Lap Collection Tests](Lap_Collection_Tests.md) — Tests for lap data collection
- [Data Collection Tests](Data_Collection_Tests.md) — Tests for race data collection
- [Jolpica API Tests](Jolpica_API_Tests.md) — Tests for Jolpica API client
- [Storage Layer Tests](Storage_Layer_Tests.md) — Tests for GCS storage module

## God Nodes (most connected)

1. `_make_races()` — 27 edges (test fixture, Race Feature Tests)
2. `_make_laps()` — 18 edges (test fixture, Lap Feature Tests)
3. `_build_lap_features()` — 11 edges (core builder, Lap Feature Engineering)
4. `TestExpandingWindowSplit` — 11 edges (CV Splits & Init)
5. `LeaveOneSeasonOut` — 10 edges (CV Splits & Init)
6. `ExpandingWindowSplit` — 10 edges (CV Splits & Init)
7. `Jolpica API Client Module` — 10 edges (Jolpica API Client)
8. `Model A: Lap + Tyre (2019-2024)` — 9 edges (Feature Build Functions)

## Hyperedges (group relationships)

- **Data Collection Pipeline** — collect_all, collect_season, collect_all_laps, weather APIs
- **Model A/B/C/D Training & Stacking** — 4 models + Three-Round Tournament + CV splitters
- **Feature Engineering Pipeline** — lap features, race features, common utils, leakage prevention
- **Data Leakage Prevention Testing** — expanding window CV + leakage test classes
- **Dual-Source Weather Pipeline** — FastF1 weather + OpenMeteo weather + weather features
- **Tyre Compound Normalization Flow** — normalize compound + encode one-hot + build lap features
