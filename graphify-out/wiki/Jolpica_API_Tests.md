# Jolpica API Tests

## Overview
This community contains the comprehensive test suite for the Jolpica REST API client. It validates lap time parsing, race time millisecond parsing, season schedule retrieval, race results, qualifying results, lap data retrieval, and pitstop data with pagination. The tests ensure the API client handles edge cases like empty responses, invalid formats, and HTTP failures gracefully.

## Key Components
| Component | Type | Source File |
|-----------|------|-------------|
| test_jolpica.py | Test Module | tests/test_jolpica.py |
| TestParseLapTime | Test Class | tests/test_jolpica.py |
| TestParseRaceTimeMillis | Test Class | tests/test_jolpica.py |
| TestGetSeasonSchedule | Test Class | tests/test_jolpica.py |
| TestGetRaceResults | Test Class | tests/test_jolpica.py |
| TestGetQualifyingResults | Test Class | tests/test_jolpica.py |
| TestGetLaps | Test Class | tests/test_jolpica.py |
| TestGetPitstops | Test Class | tests/test_jolpica.py |
| TestParseRaceTimeMillisEdge | Test Class | tests/test_jolpica.py |
| test_returns_races() | Function | tests/test_jolpica.py |
| test_returns_results() | Function | tests/test_jolpica.py |
| test_returns_qualifying() | Function | tests/test_jolpica.py |
| test_returns_laps() | Function | tests/test_jolpica.py |
| test_returns_pitstops() | Function | tests/test_jolpica.py |
| test_paginates_pitstops() | Function | tests/test_jolpica.py |

## Relationships

### Internal
- `test_jolpica.py` --contains--> `TestParseLapTime` [1.0]
- `test_jolpica.py` --contains--> `TestParseRaceTimeMillis` [1.0]
- `test_jolpica.py` --contains--> `TestGetSeasonSchedule` [1.0]
- `test_jolpica.py` --contains--> `TestGetRaceResults` [1.0]
- `test_jolpica.py` --contains--> `TestGetQualifyingResults` [1.0]
- `test_jolpica.py` --contains--> `TestGetLaps` [1.0]
- `test_jolpica.py` --contains--> `TestGetPitstops` [1.0]
- `test_jolpica.py` --contains--> `TestParseRaceTimeMillisEdge` [1.0]

### Cross-community
- Tests validate the [Jolpica API Client](Jolpica_API_Client.md) module
- Jolpica API is used for qualifying backfill in [Data Collection Core](Data_Collection_Core.md) and lap collection via Jolpica in [Data Collection Core](Data_Collection_Core.md)
- Pitstop data feeds into [Lap Collection Tests](Lap_Collection_Tests.md) (TestBuildPitstopMap)

## Source Files
- `tests/test_jolpica.py`
