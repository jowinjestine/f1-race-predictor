"""Data endpoints — circuits, drivers, races."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from f1_predictor.api.dependencies import registry
from f1_predictor.api.schemas import CircuitInfo, DriverInfo, RaceInfo

router = APIRouter(prefix="/api/v1", tags=["data"])


@router.get("/circuits", response_model=list[CircuitInfo])
def list_circuits() -> list[CircuitInfo]:
    if not registry.ready:
        raise HTTPException(503, "Models not loaded yet")

    results = []
    for name, info in sorted(registry.circuit_defaults.items()):
        seq = [c for c in info.get("common_sequence", []) if c is not None]
        results.append(
            CircuitInfo(
                name=name,
                total_laps=info["total_laps"],
                typical_stops=info["typical_stops"],
                pit_windows=info.get("pit_windows", []),
                common_sequence=seq,
            )
        )
    return results


@router.get("/drivers/{season}", response_model=list[DriverInfo])
def list_drivers(season: int) -> list[DriverInfo]:
    if not registry.ready or registry.races_df is None:
        raise HTTPException(503, "Data not loaded yet")

    df = registry.races_df[registry.races_df["season"] == season]
    if df.empty:
        raise HTTPException(404, f"No data for season {season}")

    seen: set[str] = set()
    drivers: list[DriverInfo] = []
    for _, row in df.iterrows():
        drv = row["driver_abbrev"]
        if drv not in seen:
            seen.add(drv)
            team = row.get("team_name") if "team_name" in df.columns else None
            drivers.append(DriverInfo(driver_abbrev=drv, team=team))

    return sorted(drivers, key=lambda d: d.driver_abbrev)


@router.get("/races/{season}", response_model=list[RaceInfo])
def list_races(season: int) -> list[RaceInfo]:
    if not registry.ready or registry.races_df is None:
        raise HTTPException(503, "Data not loaded yet")

    df = registry.races_df[registry.races_df["season"] == season]
    if df.empty:
        raise HTTPException(404, f"No data for season {season}")

    seen: set[str] = set()
    races: list[RaceInfo] = []
    for _, row in df.iterrows():
        ev = row["event_name"]
        if ev not in seen:
            seen.add(ev)
            races.append(
                RaceInfo(
                    season=int(row["season"]),
                    round=int(row["round"]),
                    event_name=ev,
                )
            )

    return sorted(races, key=lambda r: r.round)
