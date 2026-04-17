"""Fetch F1 race results and qualifying data from the Jolpica API."""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.jolpi.ca/ergast/f1"
REQUEST_DELAY = 0.5  # stay under 4 req/s burst limit
MAX_RETRIES = 3


def _get_json(url: str) -> dict[str, Any] | None:
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                logger.warning("Rate limited, waiting %ds before retry", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.json()  # type: ignore[no-any-return]
        except (requests.RequestException, ValueError):
            logger.warning("Jolpica request failed: %s", url, exc_info=True)
            return None
    logger.warning("Exhausted retries for %s", url)
    return None


def get_season_schedule(year: int) -> list[dict[str, Any]]:
    """Return list of race dicts for a season."""
    data = _get_json(f"{BASE_URL}/{year}.json?limit=100")
    if data is None:
        return []
    return data.get("MRData", {}).get("RaceTable", {}).get("Races", [])  # type: ignore[no-any-return]


def get_race_results(year: int, round_num: int) -> list[dict[str, Any]]:
    """Return list of result dicts for a specific race."""
    data = _get_json(f"{BASE_URL}/{year}/{round_num}/results.json?limit=100")
    if data is None:
        return []
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return []
    return races[0].get("Results", [])  # type: ignore[no-any-return]


def get_qualifying_results(year: int, round_num: int) -> list[dict[str, Any]]:
    """Return list of qualifying result dicts for a specific race."""
    data = _get_json(f"{BASE_URL}/{year}/{round_num}/qualifying.json?limit=100")
    if data is None:
        return []
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return []
    return races[0].get("QualifyingResults", [])  # type: ignore[no-any-return]


def _parse_lap_time(time_str: str | None) -> float | None:
    """Convert 'm:ss.sss' lap time string to seconds."""
    if not time_str:
        return None
    try:
        parts = time_str.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except (ValueError, IndexError):
        return None


def _parse_race_time_millis(millis_str: str | None) -> float | None:
    """Convert milliseconds string to seconds."""
    if not millis_str:
        return None
    try:
        return float(millis_str) / 1000.0
    except ValueError:
        return None


def collect_season_jolpica(year: int) -> pd.DataFrame:
    """Collect all race results + qualifying for a season from Jolpica."""
    logger.info("Collecting season %d from Jolpica", year)
    schedule = get_season_schedule(year)
    if not schedule:
        logger.warning("No schedule found for %d", year)
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for race in schedule:
        round_num = int(race["round"])
        race_name = race.get("raceName", "")
        circuit = race.get("Circuit", {})
        location_info = circuit.get("Location", {})
        location = location_info.get("locality", "")
        country = location_info.get("country", "")
        race_date = race.get("date", "")

        logger.info("  Round %d: %s (%s)", round_num, race_name, location)

        results = get_race_results(year, round_num)
        if not results:
            logger.warning("  No results for round %d", round_num)
            continue

        quali_by_driver: dict[str, dict[str, float | None]] = {}
        quali_results = get_qualifying_results(year, round_num)
        for qr in quali_results:
            driver_code = qr.get("Driver", {}).get("code", "")
            quali_by_driver[driver_code] = {
                "q1_time_sec": _parse_lap_time(qr.get("Q1")),
                "q2_time_sec": _parse_lap_time(qr.get("Q2")),
                "q3_time_sec": _parse_lap_time(qr.get("Q3")),
            }

        for result in results:
            driver = result.get("Driver", {})
            constructor = result.get("Constructor", {})
            driver_code = driver.get("code", "")
            quali = quali_by_driver.get(driver_code, {})

            race_time_obj = result.get("Time", {})
            race_time_ms = race_time_obj.get("millis") if isinstance(race_time_obj, dict) else None

            row: dict[str, Any] = {
                "season": year,
                "round": round_num,
                "event_name": race_name,
                "location": location,
                "country": country,
                "event_date": race_date,
                "driver_number": result.get("number", ""),
                "driver_abbrev": driver_code,
                "driver_id": driver.get("driverId", ""),
                "first_name": driver.get("givenName", ""),
                "last_name": driver.get("familyName", ""),
                "team": constructor.get("name", ""),
                "team_id": constructor.get("constructorId", ""),
                "finish_position": int(result["position"]) if result.get("position") else None,
                "grid_position": int(result["grid"]) if result.get("grid") else None,
                "status": result.get("status", ""),
                "points": float(result.get("points", 0)),
                "laps_completed": int(result.get("laps", 0)),
                "is_classified": result.get("positionText", "R") != "R",
                "q1_time_sec": quali.get("q1_time_sec"),
                "q2_time_sec": quali.get("q2_time_sec"),
                "q3_time_sec": quali.get("q3_time_sec"),
                "race_time_sec": _parse_race_time_millis(race_time_ms),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    n_races = len(df["round"].unique()) if len(df) > 0 else 0
    logger.info("Jolpica season %d: %d rows from %d races", year, len(df), n_races)
    return df
