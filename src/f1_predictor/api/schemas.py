"""Pydantic request/response models for the simulation API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DriverInput(BaseModel):
    """A driver entry for the starting grid."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "driver": "VER",
                    "grid_position": 1,
                    "q1": 71.5,
                    "q2": 70.8,
                    "q3": 70.2,
                    "initial_tyre": "MEDIUM",
                }
            ]
        }
    }

    driver: str = Field(
        description="Three-letter driver abbreviation (e.g. VER, NOR, LEC)",
    )
    grid_position: int = Field(ge=1, le=20, description="Starting grid position (1-20)")
    q1: float | None = Field(default=None, description="Q1 lap time in seconds (optional)")
    q2: float | None = Field(default=None, description="Q2 lap time in seconds (optional)")
    q3: float | None = Field(
        default=None,
        description=(
            "Q3 lap time in seconds. Best of q1/q2/q3 is used"
            " as the driver's qualifying pace baseline."
        ),
    )
    initial_tyre: str = Field(
        default="MEDIUM",
        description="Starting tyre compound: SOFT, MEDIUM, HARD, INTERMEDIATE, or WET",
    )
    dnf_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Probability of this driver retiring from the race (0.0 to 1.0)."
            " Converted to a per-lap hazard rate. Set to 0 to disable."
        ),
    )


class StrategyLeg(BaseModel):
    """One stint in a pit strategy."""

    compound: str = Field(
        description="Tyre compound for this stint (SOFT, MEDIUM, HARD, INTERMEDIATE, WET)",
    )
    pit_on_lap: int | None = Field(
        default=None,
        description="Lap number to pit at the end of this stint. Null means no pit (final stint).",
    )


class SimulationRequest(BaseModel):
    """Request body for race simulation.

    Provide a circuit name, a grid of 2-20 drivers with qualifying times,
    and optionally custom pit strategies. The simulator runs lap-by-lap using
    Model H (delta-baseline LightGBM) and optionally refines final positions
    with Model E (rich feature stacker).
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "circuit": "Monaco Grand Prix",
                    "drivers": [
                        {"driver": "VER", "grid_position": 1, "q3": 70.2},
                        {"driver": "LEC", "grid_position": 2, "q3": 70.4},
                        {"driver": "NOR", "grid_position": 3, "q3": 70.6},
                    ],
                    "blend_laps": 0,
                }
            ]
        }
    }

    circuit: str = Field(
        description=(
            "Circuit name as it appears in the F1 calendar"
            " (e.g. 'Monaco Grand Prix'). Use GET /api/v1/circuits to list options."
        ),
    )
    drivers: list[DriverInput] = Field(
        min_length=2,
        max_length=20,
        description="Starting grid: 2 to 20 drivers with qualifying data",
    )
    strategies: dict[str, list[StrategyLeg]] | None = Field(
        default=None,
        description=(
            "Per-driver pit strategies keyed by driver abbreviation."
            " Each is an ordered list of stints."
            " If omitted, circuit-default strategies are used."
        ),
    )
    blend_laps: int = Field(
        default=10,
        ge=0,
        le=30,
        description=(
            "Laps at the end to blend H trajectories toward E predictions."
            " Set to 0 for pure Model H (recommended)."
        ),
    )
    monte_carlo: bool = Field(default=False, description="Reserved for future use")
    n_simulations: int = Field(
        default=200,
        ge=10,
        le=1000,
        description="Number of Monte Carlo simulations (for /simulate/monte-carlo only)",
    )


class LapRecordOut(BaseModel):
    """Telemetry for one driver at one lap."""

    lap_number: int = Field(description="Lap number (1-indexed)")
    driver: str = Field(description="Driver abbreviation")
    position: int = Field(description="Race position at end of this lap")
    lap_time: float = Field(description="Lap time in seconds")
    cum_time: float = Field(description="Cumulative race time in seconds")
    gap_to_leader: float = Field(
        description="Time gap to the race leader in seconds",
    )
    compound: str = Field(description="Current tyre compound")
    tire_life: int = Field(
        description="Laps on the current set of tyres",
    )
    stint: int = Field(
        description="Stint number (increments after each pit stop)",
    )


class FinalStanding(BaseModel):
    """Final race result for one driver."""

    driver: str = Field(description="Driver abbreviation")
    position: int = Field(description="Final finishing position")
    total_time: float = Field(description="Total race time in seconds")
    gap_to_leader: float = Field(
        description="Time gap to the winner in seconds (0.0 for P1)",
    )
    pit_stops: int = Field(description="Total number of pit stops made")
    status: str = Field(
        default="Finished",
        description="'Finished' or 'DNF'",
    )
    laps_completed: int = Field(
        default=0,
        description="Number of laps completed (equals total_laps if finished)",
    )


class MonteCarloStanding(BaseModel):
    """Position distribution for one driver from Monte Carlo simulation."""

    driver: str = Field(description="Driver abbreviation")
    position: int = Field(
        description="Median finishing position across all simulations",
    )
    position_mean: float = Field(description="Mean finishing position")
    position_p10: int = Field(
        description="10th percentile (optimistic scenario)",
    )
    position_p25: int = Field(description="25th percentile position")
    position_p75: int = Field(description="75th percentile position")
    position_p90: int = Field(
        description="90th percentile (pessimistic scenario)",
    )
    position_std: float = Field(
        description="Std dev of positions (lower = more predictable)",
    )
    dnf_rate: float = Field(
        default=0.0,
        description="Fraction of simulations where the driver retired (DNF)",
    )


class SimulationResponse(BaseModel):
    """Full deterministic simulation result with lap-by-lap telemetry."""

    circuit: str = Field(description="Circuit name")
    total_laps: int = Field(description="Total number of race laps")
    model: str = Field(description="'H only' or 'H+E ensemble'")
    blend_laps: int = Field(
        description="Blend laps applied (0 = pure Model H)",
    )
    lap_records: list[LapRecordOut] = Field(
        description="Lap-by-lap telemetry for every driver at every lap",
    )
    final_standings: list[FinalStanding] = Field(
        description="Final standings sorted by finishing position",
    )


class MonteCarloResponse(BaseModel):
    """Aggregated Monte Carlo result with position distributions."""

    circuit: str = Field(description="Circuit name")
    n_simulations: int = Field(description="Number of simulations run")
    model: str = Field(description="Model used (e.g. 'H Monte Carlo')")
    standings: list[MonteCarloStanding] = Field(
        description="Position distributions per driver, sorted by median",
    )


class CircuitInfo(BaseModel):
    """Circuit metadata and default pit strategy."""

    name: str = Field(
        description="Circuit name (use as 'circuit' in simulation requests)",
    )
    total_laps: int = Field(description="Total race laps")
    typical_stops: int = Field(
        description="Typical number of pit stops for this circuit",
    )
    pit_windows: list[int] = Field(
        description="Common pit window laps (e.g. [18, 35])",
    )
    common_sequence: list[str] = Field(
        description="Most common tyre sequence (e.g. ['MEDIUM', 'HARD'])",
    )


class RaceInfo(BaseModel):
    """A race in the calendar."""

    season: int = Field(description="Season year")
    round: int = Field(description="Round number within the season")
    event_name: str = Field(
        description="Grand Prix name (e.g. 'Bahrain Grand Prix')",
    )


class DriverInfo(BaseModel):
    """A driver in the dataset."""

    driver_abbrev: str = Field(
        description="Three-letter abbreviation (e.g. VER, NOR)",
    )
    team: str | None = Field(default=None, description="Constructor/team name (if available)")


# --- Strategy optimization ---


class OptimizeStrategyRequest(BaseModel):
    """Request body for pit strategy optimization."""

    circuit: str = Field(description="Circuit name (same as simulation requests)")
    drivers: list[DriverInput] = Field(
        min_length=2,
        max_length=20,
        description="Starting grid with qualifying data",
    )
    target_driver: str = Field(
        description="Driver abbreviation to optimize strategy for (e.g. 'VER')",
    )
    use_monte_carlo: bool = Field(
        default=False,
        description="Run Monte Carlo on top 5 strategies for confidence intervals",
    )
    n_simulations: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of Monte Carlo simulations per strategy (if enabled)",
    )
    max_candidates: int = Field(
        default=30,
        ge=5,
        le=100,
        description="Maximum number of candidate strategies to evaluate",
    )
    pit_lap_delta: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Pit lap variation range (+/- laps from circuit default windows)",
    )


class StrategyOption(BaseModel):
    """One evaluated strategy option."""

    rank: int = Field(description="Rank (1 = best)")
    strategy: list[StrategyLeg] = Field(description="The pit strategy")
    description: str = Field(description="Human-readable strategy description")
    predicted_position: int = Field(description="Predicted finishing position")
    predicted_time: float = Field(description="Predicted total race time in seconds")
    gap_to_leader: float = Field(description="Predicted gap to race leader")
    position_mean: float | None = Field(
        default=None,
        description="Mean position from Monte Carlo (if run)",
    )
    position_std: float | None = Field(
        default=None,
        description="Position std dev from Monte Carlo (if run)",
    )


class OptimizeStrategyResponse(BaseModel):
    """Ranked strategy options for the target driver."""

    circuit: str = Field(description="Circuit name")
    target_driver: str = Field(description="Driver being optimized")
    n_candidates_tested: int = Field(description="Total candidates evaluated")
    strategies: list[StrategyOption] = Field(
        description="Strategies ranked by predicted performance",
    )
