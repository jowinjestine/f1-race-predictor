"""Collect F1 race results and weather data for 2018-2025 seasons.

Uses FastF1 for 2018-2024 (race results + weather telemetry) and the
Jolpica API for qualifying times and the 2025 season.  Finished data
is persisted to Google Cloud Storage so notebooks can read from GCS
instead of re-fetching from upstream APIs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import fastf1
import pandas as pd
import requests

from f1_predictor.data.jolpica import (
    _parse_lap_time as _jolpica_parse_lap_time,
)
from f1_predictor.data.jolpica import (
    collect_season_jolpica,
    get_qualifying_results,
)

logger = logging.getLogger(__name__)

FASTF1_SEASONS = range(2018, 2025)
JOLPICA_ONLY_SEASONS = range(2025, 2026)
CACHE_DIR = Path.home() / ".fastf1_cache"
DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "f1-predictor" / "raw"

OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"

CIRCUIT_COORDS: dict[str, tuple[float, float]] = {
    "Sakhir": (26.0325, 50.5106),
    "Jeddah": (21.6319, 39.1044),
    "Melbourne": (-37.8497, 144.9680),
    "Shanghai": (31.3389, 121.2198),
    "Suzuka": (34.8431, 136.5410),
    "Miami": (25.9581, -80.2389),
    "Imola": (44.3439, 11.7167),
    "Monaco": (43.7347, 7.4206),
    "Barcelona": (41.5700, 2.2611),
    "Montreal": (45.5000, -73.5228),
    "Spielberg": (47.2197, 14.7647),
    "Silverstone": (52.0786, -1.0169),
    "Budapest": (47.5789, 19.2486),
    "Spa-Francorchamps": (50.4372, 5.9714),
    "Spa": (50.4372, 5.9714),
    "Zandvoort": (52.3888, 4.5409),
    "Monza": (45.6156, 9.2811),
    "Marina Bay": (1.2914, 103.8640),
    "Singapore": (1.2914, 103.8640),
    "Baku": (40.3725, 49.8533),
    "Austin": (30.1328, -97.6411),
    "COTA": (30.1328, -97.6411),
    "Mexico City": (19.4042, -99.0907),
    "São Paulo": (-23.7014, -46.6969),
    "Sao Paulo": (-23.7014, -46.6969),
    "Interlagos": (-23.7014, -46.6969),
    "Las Vegas": (36.1162, -115.1745),
    "Lusail": (25.4900, 51.4542),
    "Yas Island": (24.4672, 54.6031),
    "Yas Marina": (24.4672, 54.6031),
    "Abu Dhabi": (24.4672, 54.6031),
    "Le Castellet": (43.2506, 5.7917),
    "Paul Ricard": (43.2506, 5.7917),
    "Hockenheim": (49.3278, 8.5656),
    "Portimão": (37.2270, -8.6267),
    "Portimao": (37.2270, -8.6267),
    "Istanbul": (40.9517, 29.4050),
    "Losail": (25.4900, 51.4542),
    "Mugello": (43.9975, 11.3719),
    "Nürburgring": (50.3356, 6.9475),
    "Nurburgring": (50.3356, 6.9475),
    "Djeddah": (21.6319, 39.1044),
    "Doha": (25.4900, 51.4542),
}


def ensure_cache() -> None:  # pragma: no cover
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def get_openmeteo_weather(lat: float, lon: float, date: str) -> dict[str, float | None]:
    """Fetch race-day weather from Open-Meteo archive API."""
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
        "timezone": "auto",
    }
    try:
        resp = requests.get(OPENMETEO_URL, params=params, timeout=10)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        return {
            "weather_temp_max": _first(daily.get("temperature_2m_max")),
            "weather_temp_min": _first(daily.get("temperature_2m_min")),
            "weather_precip_mm": _first(daily.get("precipitation_sum")),
            "weather_wind_max_kph": _first(daily.get("windspeed_10m_max")),
        }
    except (requests.RequestException, ValueError, KeyError, IndexError):
        logger.warning("Failed to fetch weather for %s at (%s, %s)", date, lat, lon, exc_info=True)
        return {
            "weather_temp_max": None,
            "weather_temp_min": None,
            "weather_precip_mm": None,
            "weather_wind_max_kph": None,
        }


def _first(lst: list[Any] | None) -> float | None:
    if lst and len(lst) > 0:
        return float(lst[0]) if lst[0] is not None else None
    return None


def collect_season(year: int) -> pd.DataFrame:  # pragma: no cover
    """Collect all race results and weather for a single season."""
    logger.info("Collecting season %d", year)
    schedule = fastf1.get_event_schedule(year)
    races = schedule[schedule["RoundNumber"] > 0]

    season_rows: list[dict[str, Any]] = []

    for _, event in races.iterrows():
        round_num = int(event["RoundNumber"])
        event_name = str(event["EventName"])
        location = str(event["Location"])
        country = str(event["Country"])
        event_date = pd.Timestamp(event["EventDate"])

        logger.info("  Round %d: %s (%s)", round_num, event_name, location)

        try:
            session = fastf1.get_session(year, round_num, "R")
            session.load(telemetry=False, weather=True, messages=False)
        except Exception:
            logger.warning("  Failed to load round %d, skipping", round_num, exc_info=True)
            continue

        results = session.results
        if results is None or len(results) == 0:
            logger.warning("  No results for round %d, skipping", round_num)
            continue

        fastf1_weather = _aggregate_fastf1_weather(session)

        coords = CIRCUIT_COORDS.get(location)
        openmeteo = {}
        if coords:
            date_str = event_date.strftime("%Y-%m-%d")
            openmeteo = get_openmeteo_weather(coords[0], coords[1], date_str)
        else:
            logger.warning("  No coordinates for location: %s", location)
            openmeteo = {
                "weather_temp_max": None,
                "weather_temp_min": None,
                "weather_precip_mm": None,
                "weather_wind_max_kph": None,
            }

        for _, driver in results.iterrows():
            position = driver.get("Position")
            grid_position = driver.get("GridPosition")

            row: dict[str, Any] = {
                "season": year,
                "round": round_num,
                "event_name": event_name,
                "location": location,
                "country": country,
                "event_date": event_date.strftime("%Y-%m-%d"),
                "driver_number": str(driver.get("DriverNumber", "")),
                "driver_abbrev": str(driver.get("Abbreviation", "")),
                "driver_id": str(driver.get("DriverId", "")),
                "first_name": str(driver.get("FirstName", "")),
                "last_name": str(driver.get("LastName", "")),
                "team": str(driver.get("TeamName", "")),
                "team_id": str(driver.get("TeamId", "")),
                "finish_position": int(position) if pd.notna(position) else None,
                "grid_position": int(grid_position) if pd.notna(grid_position) else None,
                "status": str(driver.get("Status", "")),
                "points": float(driver.get("Points", 0)),
                "laps_completed": int(driver.get("Laps", 0)) if pd.notna(driver.get("Laps")) else 0,
                "is_classified": pd.notna(driver.get("ClassifiedPosition")),
                "q1_time_sec": _td_to_seconds(driver.get("Q1")),
                "q2_time_sec": _td_to_seconds(driver.get("Q2")),
                "q3_time_sec": _td_to_seconds(driver.get("Q3")),
                "race_time_sec": _td_to_seconds(driver.get("Time")),
                **fastf1_weather,
                **openmeteo,
            }
            season_rows.append(row)

    df = pd.DataFrame(season_rows)
    n_races = len(df["round"].unique()) if len(df) > 0 else 0
    logger.info("Season %d: %d rows from %d races", year, len(df), n_races)
    return df


def _aggregate_fastf1_weather(session: Any) -> dict[str, float | bool | None]:
    """Aggregate FastF1's per-minute weather data into race-level stats."""
    try:
        weather = session.weather_data
        if weather is None or len(weather) == 0:
            return _empty_f1_weather()
        return {
            "f1_air_temp_mean": float(weather["AirTemp"].mean()),
            "f1_track_temp_mean": float(weather["TrackTemp"].mean()),
            "f1_humidity_mean": float(weather["Humidity"].mean()),
            "f1_pressure_mean": float(weather["Pressure"].mean()),
            "f1_wind_speed_mean": float(weather["WindSpeed"].mean()),
            "f1_rainfall": bool(weather["Rainfall"].any()),
        }
    except Exception:
        logger.debug("Failed to aggregate FastF1 weather", exc_info=True)
        return _empty_f1_weather()


def _empty_f1_weather() -> dict[str, float | bool | None]:
    return {
        "f1_air_temp_mean": None,
        "f1_track_temp_mean": None,
        "f1_humidity_mean": None,
        "f1_pressure_mean": None,
        "f1_wind_speed_mean": None,
        "f1_rainfall": None,
    }


def _td_to_seconds(val: Any) -> float | None:
    """Convert a timedelta or NaT to seconds."""
    if pd.isna(val):
        return None
    if hasattr(val, "total_seconds"):
        secs = val.total_seconds()
        return float(secs) if secs > 0 else None
    return None


def backfill_qualifying(df: pd.DataFrame) -> pd.DataFrame:
    """Fill in null qualifying times from the Jolpica API."""
    if df.empty:
        return df

    has_nulls = df["q1_time_sec"].isna().all()
    if not has_nulls:
        return df

    df = df.copy()
    season = int(df["season"].iloc[0])
    rounds = sorted(df["round"].unique())
    logger.info("Backfilling qualifying for season %d (%d rounds)", season, len(rounds))

    for round_num in rounds:
        quali_results = get_qualifying_results(season, int(round_num))
        quali_map: dict[str, dict[str, float | None]] = {}
        for qr in quali_results:
            code = qr.get("Driver", {}).get("code", "")
            quali_map[code] = {
                "q1_time_sec": _jolpica_parse_lap_time(qr.get("Q1")),
                "q2_time_sec": _jolpica_parse_lap_time(qr.get("Q2")),
                "q3_time_sec": _jolpica_parse_lap_time(qr.get("Q3")),
            }

        mask = df["round"] == round_num
        for idx in df.index[mask]:
            code = str(df.at[idx, "driver_abbrev"])
            if code in quali_map:
                for col in ("q1_time_sec", "q2_time_sec", "q3_time_sec"):
                    if pd.isna(df.at[idx, col]) and quali_map[code].get(col) is not None:
                        df.at[idx, col] = quali_map[code][col]

    filled = df["q1_time_sec"].notna().sum()
    logger.info("Backfilled %d qualifying entries for season %d", filled, season)
    return df


def collect_all(  # pragma: no cover
    output_dir: Path | None = None,
    upload_gcs: bool = True,
) -> pd.DataFrame:
    """Collect all seasons and save as partitioned Parquet.

    Fetches 2018-2024 from FastF1 (with Jolpica qualifying backfill)
    and 2025 from Jolpica directly.  Optionally uploads to GCS.
    """
    ensure_cache()
    out = output_dir or DEFAULT_DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    all_seasons: list[pd.DataFrame] = []

    for year in FASTF1_SEASONS:
        parquet_path = out / f"season_{year}.parquet"
        if parquet_path.exists():
            logger.info("Season %d already collected, loading from cache", year)
            all_seasons.append(pd.read_parquet(parquet_path))
            continue

        df = collect_season(year)
        if len(df) > 0:
            df = backfill_qualifying(df)
            df.to_parquet(parquet_path, engine="pyarrow", index=False)
            logger.info("Saved %s", parquet_path)
            all_seasons.append(df)

    for year in JOLPICA_ONLY_SEASONS:
        parquet_path = out / f"season_{year}.parquet"
        if parquet_path.exists():
            logger.info("Season %d already collected, loading from cache", year)
            all_seasons.append(pd.read_parquet(parquet_path))
            continue

        df = collect_season_jolpica(year)
        if len(df) > 0:
            coords_map = CIRCUIT_COORDS
            for location in df["location"].unique():
                coords = coords_map.get(location)
                if coords:
                    mask = df["location"] == location
                    date_str = df.loc[mask, "event_date"].iloc[0]
                    weather = get_openmeteo_weather(coords[0], coords[1], date_str)
                    for col, val in weather.items():
                        df.loc[mask, col] = val

            df.to_parquet(parquet_path, engine="pyarrow", index=False)
            logger.info("Saved %s", parquet_path)
            all_seasons.append(df)

    if all_seasons:
        combined = pd.concat(all_seasons, ignore_index=True)
        combined = add_target_variables(combined)
        combined.to_parquet(out / "all_races.parquet", engine="pyarrow", index=False)
        logger.info("Combined dataset: %d rows, %d columns", len(combined), len(combined.columns))

        if upload_gcs:
            _upload_to_gcs(out)

        return combined

    return pd.DataFrame()


def _upload_to_gcs(data_dir: Path) -> None:  # pragma: no cover
    """Upload all parquet files to GCS."""
    try:
        from f1_predictor.data.storage import upload_season_files

        uris = upload_season_files(data_dir)
        logger.info("Uploaded %d files to GCS", len(uris))
    except Exception:
        logger.warning("Failed to upload to GCS", exc_info=True)


def add_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived target columns."""
    df = df.copy()
    df["is_podium"] = df["finish_position"].apply(lambda x: x is not None and x <= 3)
    df["is_points_finish"] = df["finish_position"].apply(lambda x: x is not None and x <= 10)
    df["is_dnf"] = df["status"].apply(
        lambda x: (
            x not in ("Finished", "+1 Lap", "+2 Laps", "+3 Laps", "+4 Laps", "+5 Laps")
            if x
            else True
        )
    )
    return df


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    project_data_dir = Path(__file__).resolve().parents[3] / "data" / "raw"
    df = collect_all(output_dir=project_data_dir, upload_gcs=True)
    if len(df) > 0:
        out_path = project_data_dir / "all_races.parquet"
        df.to_parquet(out_path, engine="pyarrow", index=False)
        print(f"\nDone! {len(df)} rows, {len(df.columns)} columns")
        print(f"Seasons: {sorted(df['season'].unique())}")
        print(f"Races: {df.groupby('season')['round'].nunique().to_dict()}")
