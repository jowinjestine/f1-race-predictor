"""Collect lap-by-lap F1 data for 2018-2025 seasons.

Uses FastF1 session.laps for 2018-2024 and the Jolpica API /laps + /pitstops
endpoints for 2025.  Output is one row per driver per lap, stored as Parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import fastf1
import pandas as pd

from f1_predictor.data.collect import (
    FASTF1_SEASONS,
    JOLPICA_ONLY_SEASONS,
    _td_to_seconds,
    ensure_cache,
)
from f1_predictor.data.jolpica import (
    get_laps,
    get_pitstops,
    get_race_results,
    get_season_schedule,
    parse_lap_time,
)

logger = logging.getLogger(__name__)

DEFAULT_LAPS_DIR = Path.home() / ".local" / "share" / "f1-predictor" / "raw" / "laps"


def _fastf1_lap_row(
    year: int,
    round_num: int,
    event_name: str,
    lap: Any,
) -> dict[str, Any]:
    """Convert a single FastF1 lap row to our schema."""
    pit_in = lap.get("PitInTime")
    pit_out = lap.get("PitOutTime")
    return {
        "season": year,
        "round": round_num,
        "event_name": event_name,
        "driver_abbrev": str(lap.get("Driver", "")),
        "team": str(lap.get("Team", "")),
        "lap_number": int(lap.get("LapNumber", 0)),
        "lap_time_sec": _td_to_seconds(lap.get("LapTime")),
        "sector_1_sec": _td_to_seconds(lap.get("Sector1Time")),
        "sector_2_sec": _td_to_seconds(lap.get("Sector2Time")),
        "sector_3_sec": _td_to_seconds(lap.get("Sector3Time")),
        "position": _safe_int(lap.get("Position")),
        "tire_compound": _normalize_compound(lap.get("Compound")),
        "tire_life": _safe_int(lap.get("TyreLife")),
        "stint": _safe_int(lap.get("Stint")),
        "is_pit_in_lap": pd.notna(pit_in) and hasattr(pit_in, "total_seconds"),
        "is_pit_out_lap": pd.notna(pit_out) and hasattr(pit_out, "total_seconds"),
        "track_status": str(lap.get("TrackStatus", "")),
        "is_personal_best": bool(lap.get("IsPersonalBest", False)),
    }


def _safe_int(val: Any) -> int | None:
    """Convert to int, returning None for NaN/NaT."""
    if pd.isna(val):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _normalize_compound(compound: Any) -> str | None:
    """Normalize tire compound string."""
    if pd.isna(compound) or not compound:
        return None
    s = str(compound).upper().strip()
    return s if s in ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET") else None


def collect_laps_fastf1(year: int) -> pd.DataFrame:  # pragma: no cover
    """Collect lap-by-lap data for a single season from FastF1."""
    logger.info("Collecting laps for season %d from FastF1", year)
    schedule = fastf1.get_event_schedule(year)
    races = schedule[schedule["RoundNumber"] > 0]

    rows: list[dict[str, Any]] = []

    for _, event in races.iterrows():
        round_num = int(event["RoundNumber"])
        event_name = str(event["EventName"])
        logger.info("  Round %d: %s", round_num, event_name)

        try:
            session = fastf1.get_session(year, round_num, "R")
            session.load(telemetry=False, weather=False, messages=False)
        except Exception:
            logger.warning("  Failed to load round %d, skipping", round_num, exc_info=True)
            continue

        laps = session.laps
        if laps is None or len(laps) == 0:
            logger.warning("  No lap data for round %d", round_num)
            continue

        for _, lap in laps.iterrows():
            rows.append(_fastf1_lap_row(year, round_num, event_name, lap))

    df = pd.DataFrame(rows)
    n_races = len(df["round"].unique()) if len(df) > 0 else 0
    logger.info("FastF1 laps %d: %d rows from %d races", year, len(df), n_races)
    return df


def _build_pitstop_map(
    pitstops: list[dict[str, Any]],
) -> dict[tuple[str, int], float | None]:
    """Build (driverId, lap) -> pit_duration_sec map from Jolpica pitstop data."""
    pit_map: dict[tuple[str, int], float | None] = {}
    for ps in pitstops:
        driver_id = ps.get("driverId", "")
        lap = int(ps.get("lap", 0))
        duration_str = ps.get("duration")
        duration = parse_lap_time(duration_str) if duration_str else None
        pit_map[(driver_id, lap)] = duration
    return pit_map


def _build_driver_id_to_code(
    results: list[dict[str, Any]],
) -> dict[str, str]:
    """Build driverId -> code mapping from race results."""
    mapping: dict[str, str] = {}
    for r in results:
        driver = r.get("Driver", {})
        driver_id = driver.get("driverId", "")
        code = driver.get("code", "")
        if driver_id and code:
            mapping[driver_id] = code
    return mapping


def _build_driver_id_to_team(
    results: list[dict[str, Any]],
) -> dict[str, str]:
    """Build driverId -> team name mapping from race results."""
    mapping: dict[str, str] = {}
    for r in results:
        driver = r.get("Driver", {})
        constructor = r.get("Constructor", {})
        driver_id = driver.get("driverId", "")
        team = constructor.get("name", "")
        if driver_id:
            mapping[driver_id] = team
    return mapping


def collect_laps_jolpica(year: int) -> pd.DataFrame:  # pragma: no cover
    """Collect lap-by-lap data for a season from Jolpica API."""
    logger.info("Collecting laps for season %d from Jolpica", year)
    schedule = get_season_schedule(year)
    if not schedule:
        logger.warning("No schedule found for %d", year)
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for race in schedule:
        round_num = int(race["round"])
        race_name = race.get("raceName", "")
        logger.info("  Round %d: %s", round_num, race_name)

        results = get_race_results(year, round_num)
        id_to_code = _build_driver_id_to_code(results)
        id_to_team = _build_driver_id_to_team(results)

        laps_data = get_laps(year, round_num)
        pitstops = get_pitstops(year, round_num)
        pit_map = _build_pitstop_map(pitstops)

        if not laps_data:
            logger.warning("  No lap data for round %d", round_num)
            continue

        driver_stints: dict[str, int] = {}
        driver_tire_life: dict[str, int] = {}

        for lap_obj in laps_data:
            lap_number = int(lap_obj.get("number", 0))
            for timing in lap_obj.get("Timings", []):
                driver_id = timing.get("driverId", "")
                driver_code = id_to_code.get(driver_id, driver_id)
                team = id_to_team.get(driver_id, "")

                is_pit_in = (driver_id, lap_number) in pit_map
                if lap_number == 1:
                    driver_stints[driver_id] = 1
                    driver_tire_life[driver_id] = 1
                else:
                    prev_pit = (driver_id, lap_number - 1) in pit_map
                    if prev_pit:
                        driver_stints[driver_id] = driver_stints.get(driver_id, 1) + 1
                        driver_tire_life[driver_id] = 1
                    else:
                        driver_tire_life[driver_id] = driver_tire_life.get(driver_id, 0) + 1

                pit_duration = pit_map.get((driver_id, lap_number))

                rows.append(
                    {
                        "season": year,
                        "round": round_num,
                        "event_name": race_name,
                        "driver_abbrev": driver_code,
                        "team": team,
                        "lap_number": lap_number,
                        "lap_time_sec": parse_lap_time(timing.get("time")),
                        "sector_1_sec": None,
                        "sector_2_sec": None,
                        "sector_3_sec": None,
                        "position": _safe_int(timing.get("position")),
                        "tire_compound": None,
                        "tire_life": driver_tire_life.get(driver_id),
                        "stint": driver_stints.get(driver_id, 1),
                        "is_pit_in_lap": is_pit_in,
                        "is_pit_out_lap": (
                            (driver_id, lap_number - 1) in pit_map if lap_number > 1 else False
                        ),
                        "track_status": None,
                        "is_personal_best": False,
                        "pit_duration_sec": pit_duration,
                    }
                )

    df = pd.DataFrame(rows)
    n_races = len(df["round"].unique()) if len(df) > 0 else 0
    logger.info("Jolpica laps %d: %d rows from %d races", year, len(df), n_races)
    return df


def add_pit_duration(
    df: pd.DataFrame, pit_map: dict[tuple[str, int], float | None]
) -> pd.DataFrame:
    """Add pit_duration_sec column from a precomputed pit stop map."""
    if "pit_duration_sec" in df.columns:
        return df
    df = df.copy()
    df["pit_duration_sec"] = None
    for idx in df.index:
        driver = str(df.at[idx, "driver_abbrev"])
        lap_val = df.at[idx, "lap_number"]
        lap = int(str(lap_val))
        duration = pit_map.get((driver, lap))
        if duration is not None:
            df.at[idx, "pit_duration_sec"] = duration
    return df


def collect_all_laps(  # pragma: no cover
    output_dir: Path | None = None,
    upload_gcs: bool = False,
) -> pd.DataFrame:
    """Collect lap-level data for all seasons and save as Parquet."""
    ensure_cache()
    out = output_dir or DEFAULT_LAPS_DIR
    out.mkdir(parents=True, exist_ok=True)

    all_seasons: list[pd.DataFrame] = []

    for year in FASTF1_SEASONS:
        parquet_path = out / f"laps_{year}.parquet"
        if parquet_path.exists():
            logger.info("Laps %d already collected, loading from cache", year)
            all_seasons.append(pd.read_parquet(parquet_path))
            continue

        df = collect_laps_fastf1(year)
        if len(df) > 0:
            if "pit_duration_sec" not in df.columns:
                df["pit_duration_sec"] = None
            df.to_parquet(parquet_path, engine="pyarrow", index=False)
            logger.info("Saved %s", parquet_path)
            all_seasons.append(df)

    for year in JOLPICA_ONLY_SEASONS:
        parquet_path = out / f"laps_{year}.parquet"
        if parquet_path.exists():
            logger.info("Laps %d already collected, loading from cache", year)
            all_seasons.append(pd.read_parquet(parquet_path))
            continue

        df = collect_laps_jolpica(year)
        if len(df) > 0:
            df.to_parquet(parquet_path, engine="pyarrow", index=False)
            logger.info("Saved %s", parquet_path)
            all_seasons.append(df)

    if all_seasons:
        combined = pd.concat(all_seasons, ignore_index=True)
        combined.to_parquet(out / "all_laps.parquet", engine="pyarrow", index=False)
        logger.info("Combined laps: %d rows, %d columns", len(combined), len(combined.columns))

        if upload_gcs:
            _upload_laps_to_gcs(out)

        return combined

    return pd.DataFrame()


def _upload_laps_to_gcs(data_dir: Path) -> None:  # pragma: no cover
    """Upload all lap parquet files to GCS."""
    try:
        from f1_predictor.data.storage import upload_parquet

        for path in sorted(data_dir.glob("*.parquet")):
            upload_parquet(path, blob_name=f"data/raw/laps/{path.name}")
        logger.info("Uploaded lap files to GCS")
    except Exception:
        logger.warning("Failed to upload lap data to GCS", exc_info=True)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    project_data_dir = Path(__file__).resolve().parents[3] / "data" / "raw" / "laps"
    df = collect_all_laps(output_dir=project_data_dir, upload_gcs=True)
    if len(df) > 0:
        print(f"\nDone! {len(df)} rows, {len(df.columns)} columns")
        print(f"Seasons: {sorted(df['season'].unique())}")
        print(f"Races: {df.groupby('season')['round'].nunique().to_dict()}")
