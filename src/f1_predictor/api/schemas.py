"""Pydantic request/response models for the simulation API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DriverInput(BaseModel):
    driver: str
    grid_position: int = Field(ge=1, le=20)
    q1: float | None = None
    q2: float | None = None
    q3: float | None = None
    initial_tyre: str = "MEDIUM"


class StrategyLeg(BaseModel):
    compound: str
    pit_on_lap: int | None = None


class SimulationRequest(BaseModel):
    circuit: str
    drivers: list[DriverInput] = Field(min_length=2, max_length=20)
    strategies: dict[str, list[StrategyLeg]] | None = None
    blend_laps: int = Field(default=10, ge=0, le=30)
    monte_carlo: bool = False
    n_simulations: int = Field(default=200, ge=10, le=1000)


class LapRecordOut(BaseModel):
    lap_number: int
    driver: str
    position: int
    lap_time: float
    cum_time: float
    gap_to_leader: float
    compound: str
    tire_life: int
    stint: int


class FinalStanding(BaseModel):
    driver: str
    position: int
    total_time: float
    gap_to_leader: float
    pit_stops: int


class MonteCarloStanding(BaseModel):
    driver: str
    position: int
    position_mean: float
    position_p10: int
    position_p25: int
    position_p75: int
    position_p90: int
    position_std: float


class SimulationResponse(BaseModel):
    circuit: str
    total_laps: int
    model: str
    blend_laps: int
    lap_records: list[LapRecordOut]
    final_standings: list[FinalStanding]


class MonteCarloResponse(BaseModel):
    circuit: str
    n_simulations: int
    model: str
    standings: list[MonteCarloStanding]


class CircuitInfo(BaseModel):
    name: str
    total_laps: int
    typical_stops: int
    pit_windows: list[int]
    common_sequence: list[str]


class RaceInfo(BaseModel):
    season: int
    round: int
    event_name: str


class DriverInfo(BaseModel):
    driver_abbrev: str
    team: str | None = None
