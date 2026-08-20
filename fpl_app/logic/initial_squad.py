"""Optimisation of a new FPL squad without requiring a manager id.

The module deliberately contains no Streamlit or network code.  Callers supply a
point-in-time player table and per-gameweek forecasts; the optimiser only turns
those inputs into a valid squad decision.

Forecasts used by the current application are experimental estimates, not a
guarantee of future FPL points.  Keep that distinction visible in any UI that
uses this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Optional, Sequence

import pandas as pd
import pulp

try:  # Package import used by Vercel and tests.
    from ..domain.rules import SQUAD
except ImportError:  # Legacy Streamlit working-directory import.
    from domain.rules import SQUAD


SQUAD_QUOTA = {position.value: count for position, count in SQUAD.position_quotas.items()}
MAX_PER_TEAM = SQUAD.max_players_per_club
POSITION_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
DEFAULT_GW_WEIGHTS = (1.0, 0.85, 0.70, 0.55, 0.40)
DEFAULT_BENCH_WEIGHTS = (0.12, 0.08, 0.04, 0.02)
EXPERIMENTAL_NOTICE_DA = (
    "Prognoserne er eksperimentelle estimater og er ikke en garanti for fremtidige FPL-point."
)


class InitialSquadError(ValueError):
    """Base exception for invalid inputs or an unavailable optimum."""


class InitialSquadInfeasibleError(InitialSquadError):
    """Raised when no squad can satisfy all FPL constraints."""


@dataclass(frozen=True)
class InitialSquadResult:
    """A deterministic initial-squad recommendation.

    ``bench_ids`` is ordered as the three outfield substitution slots followed
    by the reserve goalkeeper.  ``objective_points`` includes the captain's
    extra score and the configured, discounted bench contribution.
    """

    squad_ids: tuple[int, ...]
    starting_ids: tuple[int, ...]
    captain_id: int
    vice_captain_id: int
    bench_ids: tuple[int, ...]
    formation: str
    total_cost_tenths: int
    bank_tenths: int
    projected_xi_points: float
    projected_captain_bonus: float
    projected_bench_contribution: float
    objective_points: float
    horizon: int
    gw_weights: tuple[float, ...]
    forecast_columns: tuple[str, ...]
    experimental_notice: str = EXPERIMENTAL_NOTICE_DA

    def as_dict(self) -> dict[str, object]:
        """Return a serialization-friendly representation."""

        return {
            "squad_ids": list(self.squad_ids),
            "starting_ids": list(self.starting_ids),
            "captain_id": self.captain_id,
            "vice_captain_id": self.vice_captain_id,
            "bench_ids": list(self.bench_ids),
            "formation": self.formation,
            "total_cost_tenths": self.total_cost_tenths,
            "bank_tenths": self.bank_tenths,
            "projected_xi_points": self.projected_xi_points,
            "projected_captain_bonus": self.projected_captain_bonus,
            "projected_bench_contribution": self.projected_bench_contribution,
            "objective_points": self.objective_points,
            "horizon": self.horizon,
            "gw_weights": list(self.gw_weights),
            "forecast_columns": list(self.forecast_columns),
            "experimental_notice": self.experimental_notice,
        }


def _numeric_sequence(values: Sequence[float], name: str, expected_length: int) -> tuple[float, ...]:
    if len(values) != expected_length:
        raise InitialSquadError(f"{name} must contain exactly {expected_length} values")

    converted = tuple(float(value) for value in values)
    if any(not isfinite(value) or value < 0 for value in converted):
        raise InitialSquadError(f"{name} must contain finite, non-negative values")
    if not any(value > 0 for value in converted):
        raise InitialSquadError(f"at least one value in {name} must be positive")
    return converted


def _integer_column(df: pd.DataFrame, column: str, *, minimum: int = 0) -> pd.Series:
    numeric = pd.to_numeric(df[column], errors="coerce")
    if numeric.isna().any() or (~numeric.map(isfinite)).any():
        raise InitialSquadError(f"{column} must contain finite numbers")
    if (numeric < minimum).any() or ((numeric - numeric.round()).abs() > 1e-9).any():
        raise InitialSquadError(f"{column} must contain integers >= {minimum}")
    return numeric.round().astype(int)


def _resolve_forecasts(
    df: pd.DataFrame,
    horizon: int,
    forecast_columns: Optional[Sequence[str]],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Return per-GW forecasts, deriving them from cumulative EP when needed."""

    if forecast_columns is not None:
        columns = tuple(str(column) for column in forecast_columns)
        if len(columns) != horizon:
            raise InitialSquadError("forecast_columns must have one column per gameweek")
        if len(set(columns)) != len(columns):
            raise InitialSquadError("forecast_columns cannot contain duplicates")
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise InitialSquadError(f"missing forecast columns: {', '.join(missing)}")
        per_gw = df.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        source_columns = columns
    else:
        direct_columns = tuple(f"ep_gw{gw}" for gw in range(1, horizon + 1))
        if all(column in df.columns for column in direct_columns):
            per_gw = df.loc[:, direct_columns].apply(pd.to_numeric, errors="coerce")
            source_columns = direct_columns
        else:
            first = "ep_next1" if "ep_next1" in df.columns else "ep_next_gw"
            cumulative_columns = (first,) + tuple(f"ep_next{gw}" for gw in range(2, horizon + 1))
            if not all(column in df.columns for column in cumulative_columns):
                expected = ", ".join(direct_columns)
                raise InitialSquadError(
                    f"provide forecast_columns or per-gameweek columns {expected}"
                )
            cumulative = df.loc[:, cumulative_columns].apply(pd.to_numeric, errors="coerce")
            per_gw = cumulative.copy()
            for offset in range(horizon - 1, 0, -1):
                per_gw.iloc[:, offset] = cumulative.iloc[:, offset] - cumulative.iloc[:, offset - 1]
            source_columns = cumulative_columns

    if per_gw.isna().any().any():
        raise InitialSquadError("forecast columns must contain numeric values for every player")
    if not per_gw.apply(lambda column: column.map(isfinite)).all().all():
        raise InitialSquadError("forecast columns must contain finite values")

    per_gw.columns = [f"_forecast_gw{gw}" for gw in range(1, horizon + 1)]
    return per_gw.astype(float), source_columns


def _prepare_players(
    players: pd.DataFrame,
    horizon: int,
    gw_weights: tuple[float, ...],
    forecast_columns: Optional[Sequence[str]],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if not isinstance(players, pd.DataFrame) or players.empty:
        raise InitialSquadError("players must be a non-empty pandas DataFrame")

    required = {"id", "team_id", "pos", "now_cost"}
    missing = sorted(required - set(players.columns))
    if missing:
        raise InitialSquadError(f"missing required columns: {', '.join(missing)}")

    df = players.copy(deep=True).reset_index(drop=True)
    df["id"] = _integer_column(df, "id", minimum=1)
    if df["id"].duplicated().any():
        raise InitialSquadError("player ids must be unique")
    df["team_id"] = _integer_column(df, "team_id", minimum=1)
    df["now_cost"] = _integer_column(df, "now_cost", minimum=0)

    df["pos"] = df["pos"].astype(str).str.upper().replace({"GK": "GKP"})
    unknown_positions = sorted(set(df["pos"]) - set(SQUAD_QUOTA))
    if unknown_positions:
        raise InitialSquadError(f"unknown positions: {', '.join(unknown_positions)}")

    per_gw, source_columns = _resolve_forecasts(df, horizon, forecast_columns)
    df = pd.concat([df, per_gw], axis=1)
    df["_weighted_forecast"] = sum(
        df[f"_forecast_gw{gw}"] * gw_weights[gw - 1]
        for gw in range(1, horizon + 1)
    )

    # The input order must never decide a tied optimisation result.
    df = df.sort_values("id", kind="stable").reset_index(drop=True)
    return df, source_columns


def optimize_initial_squad(
    players: pd.DataFrame,
    *,
    horizon: int = 5,
    gw_weights: Optional[Sequence[float]] = None,
    forecast_columns: Optional[Sequence[str]] = None,
    budget_tenths: int = SQUAD.initial_budget_tenths,
    bench_weights: Sequence[float] = DEFAULT_BENCH_WEIGHTS,
    solver: Optional[pulp.LpSolver] = None,
) -> InitialSquadResult:
    """Select a valid 15-player FPL squad, XI, captain, vice and bench.

    Forecast input can either be explicit per-GW columns supplied through
    ``forecast_columns``, automatically detected ``ep_gw1`` ... ``ep_gw5``
    columns, or cumulative ``ep_next_gw``/``ep_next2`` ... columns.  Cumulative
    values are converted to marginal gameweek forecasts before weighting.

    All prices and ``budget_tenths`` are in FPL's £0.1m units.  The default
    bench coefficients give reserves a small expected contribution without
    allowing them to dominate the starting XI.
    """

    if not isinstance(horizon, int) or isinstance(horizon, bool) or not 1 <= horizon <= 5:
        raise InitialSquadError("horizon must be an integer from 1 to 5")
    if not isinstance(budget_tenths, int) or isinstance(budget_tenths, bool) or budget_tenths < 0:
        raise InitialSquadError("budget_tenths must be a non-negative integer")
    weights = _numeric_sequence(
        gw_weights if gw_weights is not None else DEFAULT_GW_WEIGHTS[:horizon],
        "gw_weights",
        horizon,
    )
    reserve_weights = _numeric_sequence(bench_weights, "bench_weights", 4)
    if any(reserve_weights[index] < reserve_weights[index + 1] for index in range(3)):
        raise InitialSquadError("bench_weights must be ordered from highest to lowest")

    df, source_columns = _prepare_players(players, horizon, weights, forecast_columns)
    for position, needed in SQUAD_QUOTA.items():
        if int((df["pos"] == position).sum()) < needed:
            raise InitialSquadInfeasibleError(
                f"not enough {position} candidates to fill the squad quota"
            )

    indices = list(df.index)
    bench_slots = range(4)
    model = pulp.LpProblem("initial_fpl_squad", pulp.LpMaximize)
    squad = {i: pulp.LpVariable(f"squad_{i}", cat="Binary") for i in indices}
    start = {i: pulp.LpVariable(f"start_{i}", cat="Binary") for i in indices}
    captain = {i: pulp.LpVariable(f"captain_{i}", cat="Binary") for i in indices}
    vice = {i: pulp.LpVariable(f"vice_{i}", cat="Binary") for i in indices}
    bench = {
        (i, slot): pulp.LpVariable(f"bench_{i}_{slot + 1}", cat="Binary")
        for i in indices
        for slot in bench_slots
    }

    model += pulp.lpSum(squad.values()) == 15
    model += pulp.lpSum(start.values()) == 11
    model += pulp.lpSum(captain.values()) == 1
    model += pulp.lpSum(vice.values()) == 1
    model += pulp.lpSum(squad[i] * int(df.at[i, "now_cost"]) for i in indices) <= budget_tenths

    for i in indices:
        model += start[i] <= squad[i]
        model += captain[i] <= start[i]
        model += vice[i] <= start[i]
        model += captain[i] + vice[i] <= 1
        model += pulp.lpSum(bench[(i, slot)] for slot in bench_slots) == squad[i] - start[i]

        if df.at[i, "pos"] == "GKP":
            for slot in range(3):
                model += bench[(i, slot)] == 0
        else:
            model += bench[(i, 3)] == 0

    for slot in bench_slots:
        model += pulp.lpSum(bench[(i, slot)] for i in indices) == 1

    for position, quota in SQUAD_QUOTA.items():
        members = [i for i in indices if df.at[i, "pos"] == position]
        model += pulp.lpSum(squad[i] for i in members) == quota

    for _, members in df.groupby("team_id").groups.items():
        model += pulp.lpSum(squad[int(i)] for i in members) <= MAX_PER_TEAM

    goalkeepers = [i for i in indices if df.at[i, "pos"] == "GKP"]
    defenders = [i for i in indices if df.at[i, "pos"] == "DEF"]
    midfielders = [i for i in indices if df.at[i, "pos"] == "MID"]
    forwards = [i for i in indices if df.at[i, "pos"] == "FWD"]
    model += pulp.lpSum(start[i] for i in goalkeepers) == 1
    model += pulp.lpSum(start[i] for i in defenders) >= 3
    model += pulp.lpSum(start[i] for i in defenders) <= 5
    model += pulp.lpSum(start[i] for i in midfielders) >= 2
    model += pulp.lpSum(start[i] for i in midfielders) <= 5
    model += pulp.lpSum(start[i] for i in forwards) >= 1
    model += pulp.lpSum(start[i] for i in forwards) <= 3

    # Integer milli-point coefficients make it impossible for the deterministic
    # tie-break below to overturn a genuine 0.001-point objective difference.
    score_milli = {
        i: int(round(float(df.at[i, "_weighted_forecast"]) * 1000)) for i in indices
    }
    bench_score_milli = {
        (i, slot): int(round(float(df.at[i, "_weighted_forecast"]) * reserve_weights[slot] * 1000))
        for i in indices
        for slot in bench_slots
    }
    primary_objective = (
        pulp.lpSum(start[i] * score_milli[i] for i in indices)
        + pulp.lpSum(captain[i] * score_milli[i] for i in indices)
        + pulp.lpSum(bench[(i, slot)] * bench_score_milli[(i, slot)] for i in indices for slot in bench_slots)
    )

    # CBC is deterministic for a fixed variable order.  This bounded secondary
    # objective first favours the strongest eligible vice and then lower ids in
    # otherwise tied solutions.  Its total value remains below one tenth of a
    # milli-point, so it cannot change the primary recommendation.
    count = len(indices)
    rank_upper_bound = max(1, 2 * count * count)
    minimum_score = min(score_milli.values(), default=0)
    maximum_score = max(score_milli.values(), default=0)
    vice_priority_multiplier = rank_upper_bound + 1
    rank_tie = pulp.lpSum(
        (count - i) * (squad[i] + start[i])
        + vice[i] * (count - i) / (count + 1)
        for i in indices
    )
    vice_tie = pulp.lpSum(
        vice[i] * (score_milli[i] - minimum_score) * vice_priority_multiplier
        for i in indices
    )
    tie_raw = vice_tie + rank_tie
    tie_upper_bound = max(
        1.0,
        (maximum_score - minimum_score) * vice_priority_multiplier + rank_upper_bound + 1,
    )
    model += primary_objective + (0.09 / tie_upper_bound) * tie_raw

    selected_solver = solver or pulp.PULP_CBC_CMD(
        msg=False,
        threads=1,
        options=["randomSeed 0"],
    )
    try:
        model.solve(selected_solver)
    except pulp.PulpSolverError as exc:
        raise InitialSquadError(f"the optimisation solver could not run: {exc}") from exc

    if pulp.LpStatus.get(model.status) != "Optimal":
        status = pulp.LpStatus.get(model.status, str(model.status))
        raise InitialSquadInfeasibleError(
            f"no valid initial squad exists for the supplied candidates and budget (status: {status})"
        )

    chosen = [i for i in indices if pulp.value(squad[i]) > 0.5]
    starters = [i for i in indices if pulp.value(start[i]) > 0.5]
    captain_index = next(i for i in indices if pulp.value(captain[i]) > 0.5)
    vice_index = next(i for i in indices if pulp.value(vice[i]) > 0.5)
    bench_indices = [
        next(i for i in indices if pulp.value(bench[(i, slot)]) > 0.5)
        for slot in bench_slots
    ]

    def player_sort_key(index: int) -> tuple[int, float, int]:
        return (
            POSITION_ORDER[str(df.at[index, "pos"])],
            -float(df.at[index, "_weighted_forecast"]),
            int(df.at[index, "id"]),
        )

    chosen.sort(key=player_sort_key)
    starters.sort(key=player_sort_key)
    formation_counts = {
        position: sum(df.at[i, "pos"] == position for i in starters)
        for position in ("DEF", "MID", "FWD")
    }
    formation = f"{formation_counts['DEF']}-{formation_counts['MID']}-{formation_counts['FWD']}"

    total_cost = int(sum(int(df.at[i, "now_cost"]) for i in chosen))
    xi_points = float(sum(float(df.at[i, "_weighted_forecast"]) for i in starters))
    captain_bonus = float(df.at[captain_index, "_weighted_forecast"])
    bench_contribution = float(
        sum(
            float(df.at[index, "_weighted_forecast"]) * reserve_weights[slot]
            for slot, index in enumerate(bench_indices)
        )
    )

    return InitialSquadResult(
        squad_ids=tuple(int(df.at[i, "id"]) for i in chosen),
        starting_ids=tuple(int(df.at[i, "id"]) for i in starters),
        captain_id=int(df.at[captain_index, "id"]),
        vice_captain_id=int(df.at[vice_index, "id"]),
        bench_ids=tuple(int(df.at[i, "id"]) for i in bench_indices),
        formation=formation,
        total_cost_tenths=total_cost,
        bank_tenths=budget_tenths - total_cost,
        projected_xi_points=round(xi_points, 3),
        projected_captain_bonus=round(captain_bonus, 3),
        projected_bench_contribution=round(bench_contribution, 3),
        objective_points=round(xi_points + captain_bonus + bench_contribution, 3),
        horizon=horizon,
        gw_weights=weights,
        forecast_columns=source_columns,
    )
