# Lap Collection Tests

## Overview
This community contains the test suite for the lap-level data collection module. It covers safe integer conversion, tire compound normalization, pitstop map construction, driver ID-to-code/team mapping, FastF1 lap row extraction, and pit duration enrichment. These tests ensure the raw lap data pipeline handles edge cases like NaN, NaT, invalid inputs, and empty datasets gracefully.

## Key Components
| Component | Type | Source File |
|-----------|------|-------------|
| test_collect_laps.py | Test Module | tests/test_collect_laps.py |
| TestSafeInt | Test Class | tests/test_collect_laps.py |
| TestNormalizeCompound | Test Class | tests/test_collect_laps.py |
| TestBuildPitstopMap | Test Class | tests/test_collect_laps.py |
| TestBuildDriverIdToCode | Test Class | tests/test_collect_laps.py |
| TestBuildDriverIdToTeam | Test Class | tests/test_collect_laps.py |
| TestFastf1LapRow | Test Class | tests/test_collect_laps.py |
| TestAddPitDuration | Test Class | tests/test_collect_laps.py |

## Relationships

### Internal
- `test_collect_laps.py` --contains--> `TestSafeInt` [1.0]
- `test_collect_laps.py` --contains--> `TestNormalizeCompound` [1.0]
- `test_collect_laps.py` --contains--> `TestBuildPitstopMap` [1.0]
- `test_collect_laps.py` --contains--> `TestBuildDriverIdToCode` [1.0]
- `test_collect_laps.py` --contains--> `TestBuildDriverIdToTeam` [1.0]
- `test_collect_laps.py` --contains--> `TestFastf1LapRow` [1.0]
- `test_collect_laps.py` --contains--> `TestAddPitDuration` [1.0]

### Cross-community
- Tests validate functions from the [Data Collection Core](Data_Collection_Core.md) lap collection pipeline (collect_laps.py)
- TestNormalizeCompound is semantically related to TestEncodeCompoundOnehot in [Feature Module Concepts](Feature_Module_Concepts.md)
- Pitstop map and driver mappings feed into [Lap Feature Engineering](Lap_Feature_Engineering.md)

## Source Files
- `tests/test_collect_laps.py`
