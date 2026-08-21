"""Optimise one fixed FPL squad with a separate legal plan per gameweek.

The module is deliberately independent of network and UI code.  Callers supply
point-in-time forecasts and availability probabilities; the optimiser selects
one 15-player squad and then a distinct XI, captain, vice-captain and bench order
for every gameweek in the requested horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Optional, Sequence

import pandas as pd
import pulp

try:  # Package imports used by Vercel and tests.
    from ..domain.rules import SQUAD
except ImportError:  # pragma: no cover - legacy Streamlit import layout
    from domain.rules import SQUAD


SQUAD_QUOTA = {
    position.value: count for position, count in SQUAD.position_quotas.items()
}
MAX_PER_TEAM = SQUAD.max_players_per_club
POSITION_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
DEFAULT_GW_WEIGHTS = (1.0, 0.85, 0.70, 0.55, 0.40)
DEFAULT_BENCH_WEIGHTS = (0.12, 0.08, 0.04, 0.02)


class SquadPlanError(ValueError):
    """Raised for invalid input or when no optimal squad plan is available."""


class SquadPlanInfeasibleError(SquadPlanError):
    """Raised when the supplied candidates cannot form a legal squad plan."""


@dataclass(frozen=True)
class GameweekPlan:
    """One gameweek's legal plan for the shared squad.

    ``bench_ids`` contains the three ordered outfield substitution slots followed
    by the reserve goalkeeper.
    """

    gameweek: int
    starting_ids: tuple[int, ...]
    captain_id: int
    vice_captain_id: int
    bench_ids: tuple[int, ...]
    formation: str
    projected_xi_points: float
    projected_captain_bonus: float
    projected_bench_contribution: float
    objective_points: float

    def as_dict(self) -> dict[str, object]:
        return {
            "gameweek": self.gameweek,
            "starting_ids": list(self.starting_ids),
            "captain_id": self.captain_id,
            "vice_captain_id": self.vice_captain_id,
            "bench_ids": list(self.bench_ids),
            "formation": self.formation,
            "projected_xi_points": self.projected_xi_points,
            "projected_captain_bonus": self.projected_captain_bonus,
            "projected_bench_contribution": self.projected_bench_contribution,
            "objective_points": self.objective_points,
        }


@dataclass(frozen=True)
class SquadPlanResult:
    """A fixed squad and its independently optimised per-gameweek plans."""

    squad_ids: tuple[int, ...]
    gameweeks: tuple[GameweekPlan, ...]
    total_cost_tenths: int
    bank_tenths: int
    objective_points: float
    horizon: int
    gw_weights: tuple[float, ...]
    forecast_columns: tuple[str, ...]
    appearance_columns: tuple[str, ...]
    no_show_columns: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "squad_ids": list(self.squad_ids),
            "gameweeks": [plan.as_dict() for plan in self.gameweeks],
            "total_cost_tenths": self.total_cost_tenths,
            "bank_tenths": self.bank_tenths,
            "objective_points": self.objective_points,
            "horizon": self.horizon,
            "gw_weights": list(self.gw_weights),
            "forecast_columns": list(self.forecast_columns),
            "appearance_columns": list(self.appearance_columns),
            "no_show_columns": list(self.no_show_columns),
        }


def _numeric_sequence(
    values: Sequence[float],
    name: str,
    expected_length: int,
) -> tuple[float, ...]:
    if len(values) != expected_length:
        raise SquadPlanError(f"{name} must contain exactly {expected_length} values")
    converted = tuple(float(value) for value in values)
    if any(not isfinite(value) or value < 0 for value in converted):
        raise SquadPlanError(f"{name} must contain finite, non-negative values")
    if not any(value > 0 for value in converted):
        raise SquadPlanError(f"at least one value in {name} must be positive")
    return converted


def _integer_column(df: pd.DataFrame, column: str, *, minimum: int) -> pd.Series:
    numeric = pd.to_numeric(df[column], errors="coerce")
    if numeric.isna().any() or (~numeric.map(isfinite)).any():
        raise SquadPlanError(f"{column} must contain finite numbers")
    if (numeric < minimum).any() or ((numeric - numeric.round()).abs() > 1e-9).any():
        raise SquadPlanError(f"{column} must contain integers >= {minimum}")
    return numeric.round().astype(int)


def _prepare_players(
    players: pd.DataFrame,
    horizon: int,
    gw_weights: tuple[float, ...],
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if not isinstance(players, pd.DataFrame) or players.empty:
        raise SquadPlanError("players must be a non-empty pandas DataFrame")

    forecast_columns = tuple(f"ep_gw{gameweek}" for gameweek in range(1, horizon + 1))
    appearance_columns = tuple(
        f"appearance_prob_gw{gameweek}" for gameweek in range(1, horizon + 1)
    )
    no_show_columns = tuple(
        f"no_show_prob_gw{gameweek}" for gameweek in range(1, horizon + 1)
    )
    required = {
        "id",
        "team_id",
        "pos",
        "now_cost",
        *forecast_columns,
        *appearance_columns,
        *no_show_columns,
    }
    missing = sorted(required - set(players.columns))
    if missing:
        raise SquadPlanError(f"missing required columns: {', '.join(missing)}")

    df = players.copy(deep=True).reset_index(drop=True)
    df["id"] = _integer_column(df, "id", minimum=1)
    if df["id"].duplicated().any():
        raise SquadPlanError("player ids must be unique")
    df["team_id"] = _integer_column(df, "team_id", minimum=1)
    df["now_cost"] = _integer_column(df, "now_cost", minimum=0)
    df["pos"] = df["pos"].astype(str).str.upper().replace({"GK": "GKP"})
    unknown_positions = sorted(set(df["pos"]) - set(SQUAD_QUOTA))
    if unknown_positions:
        raise SquadPlanError(f"unknown positions: {', '.join(unknown_positions)}")

    for column in forecast_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or (~values.map(isfinite)).any():
            raise SquadPlanError(f"{column} must contain finite numeric values")
        df[column] = values.astype(float)

    for column in (*appearance_columns, *no_show_columns):
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or (~values.map(isfinite)).any():
            raise SquadPlanError(f"{column} must contain finite numeric values")
        if ((values < 0) | (values > 1)).any():
            raise SquadPlanError(f"{column} must contain probabilities from 0 to 1")
        df[column] = values.astype(float)

    df["_weighted_forecast"] = sum(
        df[forecast_columns[gameweek - 1]] * gw_weights[gameweek - 1]
        for gameweek in range(1, horizon + 1)
    )
    df = df.sort_values("id", kind="stable").reset_index(drop=True)
    return df, forecast_columns, appearance_columns, no_show_columns


def _solve_or_raise(
    model: pulp.LpProblem,
    solver: pulp.LpSolver,
) -> None:
    try:
        model.solve(solver)
    except pulp.PulpSolverError as exc:
        raise SquadPlanError(f"the optimisation solver could not run: {exc}") from exc
    if pulp.LpStatus.get(model.status) != "Optimal":
        status = pulp.LpStatus.get(model.status, str(model.status))
        raise SquadPlanInfeasibleError(
            "no valid squad plan exists for the supplied candidates and budget "
            f"(status: {status})"
        )


def optimize_squad_plan(
    players: pd.DataFrame,
    *,
    horizon: int = 5,
    gw_weights: Optional[Sequence[float]] = None,
    budget_tenths: int = SQUAD.initial_budget_tenths,
    bench_weights: Sequence[float] = DEFAULT_BENCH_WEIGHTS,
    solver: Optional[pulp.LpSolver] = None,
) -> SquadPlanResult:
    """Select one squad and a separate legal XI and bench for every gameweek.

    The primary objective is the decreasing-GW-weighted sum of XI points,
    captain bonus and a position-appropriate average reserve contribution.
    Forecast EP already includes appearance and official availability.  Reserve
    value is therefore adjusted only by the independent
    ``1 - no_show_probability`` signal.  Once the squad and XI are fixed, the three
    outfield reserves are ordered by that same availability-adjusted forecast;
    the reserve goalkeeper is fourth.  This removes symmetric bench-assignment
    binaries while retaining the decision-relevant bench signal.  Starting and
    captain forecasts already have availability reflected in ``ep_gwN``.
    """

    if not isinstance(horizon, int) or isinstance(horizon, bool) or not 1 <= horizon <= 5:
        raise SquadPlanError("horizon must be an integer from 1 to 5")
    if (
        not isinstance(budget_tenths, int)
        or isinstance(budget_tenths, bool)
        or budget_tenths < 0
    ):
        raise SquadPlanError("budget_tenths must be a non-negative integer")

    weights = _numeric_sequence(
        gw_weights if gw_weights is not None else DEFAULT_GW_WEIGHTS[:horizon],
        "gw_weights",
        horizon,
    )
    if any(weights[index] < weights[index + 1] for index in range(horizon - 1)):
        raise SquadPlanError("gw_weights must be ordered from highest to lowest")
    reserve_weights = _numeric_sequence(bench_weights, "bench_weights", 4)
    if any(reserve_weights[index] < reserve_weights[index + 1] for index in range(3)):
        raise SquadPlanError("bench_weights must be ordered from highest to lowest")

    df, forecast_columns, appearance_columns, no_show_columns = _prepare_players(
        players,
        horizon,
        weights,
    )
    for position, needed in SQUAD_QUOTA.items():
        if int((df["pos"] == position).sum()) < needed:
            raise SquadPlanInfeasibleError(
                f"not enough {position} candidates to fill the squad quota"
            )

    indices = list(df.index)
    gameweeks = range(horizon)
    model = pulp.LpProblem("per_gameweek_fpl_squad_plan", pulp.LpMaximize)
    squad = {index: pulp.LpVariable(f"squad_{index}", cat="Binary") for index in indices}
    start = {
        (gameweek, index): pulp.LpVariable(f"start_{gameweek + 1}_{index}", cat="Binary")
        for gameweek in gameweeks
        for index in indices
    }
    captain = {
        (gameweek, index): pulp.LpVariable(f"captain_{gameweek + 1}_{index}", cat="Binary")
        for gameweek in gameweeks
        for index in indices
    }
    model += pulp.lpSum(squad.values()) == SQUAD.squad_size
    model += (
        pulp.lpSum(squad[index] * int(df.at[index, "now_cost"]) for index in indices)
        <= budget_tenths
    )
    for position, quota in SQUAD_QUOTA.items():
        members = [index for index in indices if df.at[index, "pos"] == position]
        model += pulp.lpSum(squad[index] for index in members) == quota
    for _, members in df.groupby("team_id").groups.items():
        model += pulp.lpSum(squad[int(index)] for index in members) <= MAX_PER_TEAM

    goalkeepers = [index for index in indices if df.at[index, "pos"] == "GKP"]
    defenders = [index for index in indices if df.at[index, "pos"] == "DEF"]
    midfielders = [index for index in indices if df.at[index, "pos"] == "MID"]
    forwards = [index for index in indices if df.at[index, "pos"] == "FWD"]

    for gameweek in gameweeks:
        model += pulp.lpSum(start[(gameweek, index)] for index in indices) == SQUAD.starting_size
        model += pulp.lpSum(captain[(gameweek, index)] for index in indices) == 1
        for index in indices:
            model += start[(gameweek, index)] <= squad[index]
            model += captain[(gameweek, index)] <= start[(gameweek, index)]

        model += pulp.lpSum(start[(gameweek, index)] for index in goalkeepers) == 1
        model += pulp.lpSum(start[(gameweek, index)] for index in defenders) >= 3
        model += pulp.lpSum(start[(gameweek, index)] for index in defenders) <= 5
        model += pulp.lpSum(start[(gameweek, index)] for index in midfielders) >= 2
        model += pulp.lpSum(start[(gameweek, index)] for index in midfielders) <= 5
        model += pulp.lpSum(start[(gameweek, index)] for index in forwards) >= 1
        model += pulp.lpSum(start[(gameweek, index)] for index in forwards) <= 3

    # Integer milli-point coefficients make the primary objective exact.
    start_score: dict[tuple[int, int], int] = {}
    bench_score: dict[tuple[int, int], int] = {}
    for gameweek in gameweeks:
        ep_column = forecast_columns[gameweek]
        no_show_column = no_show_columns[gameweek]
        for index in indices:
            ep = float(df.at[index, ep_column])
            start_score[(gameweek, index)] = int(round(ep * weights[gameweek] * 1000))
            independent_availability = 1.0 - float(df.at[index, no_show_column])
            reserve_weight = (
                reserve_weights[3]
                if df.at[index, "pos"] == "GKP"
                else sum(reserve_weights[:3]) / 3.0
            )
            bench_score[(gameweek, index)] = int(
                round(
                    ep
                    * independent_availability
                    * reserve_weight
                    * weights[gameweek]
                    * 1000
                )
            )

    primary_objective = (
        pulp.lpSum(
            start[(gameweek, index)] * start_score[(gameweek, index)]
            + captain[(gameweek, index)] * start_score[(gameweek, index)]
            for gameweek in gameweeks
            for index in indices
        )
        + pulp.lpSum(
            (squad[index] - start[(gameweek, index)])
            * bench_score[(gameweek, index)]
            for gameweek in gameweeks
            for index in indices
        )
    )
    model += primary_objective

    selected_solver = solver or pulp.PULP_CBC_CMD(
        msg=False,
        threads=1,
        # CBC interprets seed 0 as time-of-day entropy.  Fixed non-zero seeds
        # keep both its LP perturbation and branch search reproducible.
        options=["randomSeed 17", "randomCbcSeed 17"],
    )
    _solve_or_raise(model, selected_solver)

    chosen = [index for index in indices if pulp.value(squad[index]) > 0.5]

    def squad_sort_key(index: int) -> tuple[int, float, int]:
        return (
            POSITION_ORDER[str(df.at[index, "pos"])],
            -float(df.at[index, "_weighted_forecast"]),
            int(df.at[index, "id"]),
        )

    chosen.sort(key=squad_sort_key)
    plans: list[GameweekPlan] = []
    weighted_objective = 0.0
    for gameweek in gameweeks:
        ep_column = forecast_columns[gameweek]
        no_show_column = no_show_columns[gameweek]
        starters = [
            index for index in indices if pulp.value(start[(gameweek, index)]) > 0.5
        ]
        captain_index = next(
            index for index in indices if pulp.value(captain[(gameweek, index)]) > 0.5
        )
        # Vice-captaincy does not affect the optimisation objective.  Selecting
        # it after the solve removes a large set of symmetric binary solutions
        # while deterministically choosing the strongest non-captain starter.
        vice_index = min(
            (index for index in starters if index != captain_index),
            key=lambda index: (
                -float(df.at[index, ep_column]),
                int(df.at[index, "id"]),
            ),
        )
        chosen_set = set(chosen)
        starter_set = set(starters)
        reserves = chosen_set - starter_set
        outfield_reserves = [
            index for index in reserves if df.at[index, "pos"] != "GKP"
        ]
        outfield_reserves.sort(
            key=lambda index: (
                -float(df.at[index, ep_column])
                * (1.0 - float(df.at[index, no_show_column])),
                int(df.at[index, "id"]),
            )
        )
        reserve_goalkeepers = [
            index for index in reserves if df.at[index, "pos"] == "GKP"
        ]
        if len(outfield_reserves) != 3 or len(reserve_goalkeepers) != 1:
            raise SquadPlanError("optimal lineup produced an invalid reserve composition")
        bench_indices = [*outfield_reserves, reserve_goalkeepers[0]]

        starters.sort(
            key=lambda index: (
                POSITION_ORDER[str(df.at[index, "pos"])],
                -float(df.at[index, ep_column]),
                int(df.at[index, "id"]),
            )
        )
        formation_counts = {
            position: sum(df.at[index, "pos"] == position for index in starters)
            for position in ("DEF", "MID", "FWD")
        }
        formation = (
            f"{formation_counts['DEF']}-{formation_counts['MID']}-"
            f"{formation_counts['FWD']}"
        )
        xi_points = float(sum(float(df.at[index, ep_column]) for index in starters))
        captain_bonus = float(df.at[captain_index, ep_column])
        bench_contribution = 0.0
        for slot, index in enumerate(bench_indices):
            independent_availability = 1.0 - float(df.at[index, no_show_column])
            reserve_weight = (
                reserve_weights[3]
                if df.at[index, "pos"] == "GKP"
                else sum(reserve_weights[:3]) / 3.0
            )
            bench_contribution += (
                float(df.at[index, ep_column])
                * independent_availability
                * reserve_weight
            )
        objective_points = xi_points + captain_bonus + bench_contribution
        weighted_objective += objective_points * weights[gameweek]
        plans.append(
            GameweekPlan(
                gameweek=gameweek + 1,
                starting_ids=tuple(int(df.at[index, "id"]) for index in starters),
                captain_id=int(df.at[captain_index, "id"]),
                vice_captain_id=int(df.at[vice_index, "id"]),
                bench_ids=tuple(int(df.at[index, "id"]) for index in bench_indices),
                formation=formation,
                projected_xi_points=round(xi_points, 3),
                projected_captain_bonus=round(captain_bonus, 3),
                projected_bench_contribution=round(bench_contribution, 3),
                objective_points=round(objective_points, 3),
            )
        )

    total_cost = int(sum(int(df.at[index, "now_cost"]) for index in chosen))
    return SquadPlanResult(
        squad_ids=tuple(int(df.at[index, "id"]) for index in chosen),
        gameweeks=tuple(plans),
        total_cost_tenths=total_cost,
        bank_tenths=budget_tenths - total_cost,
        objective_points=round(weighted_objective, 3),
        horizon=horizon,
        gw_weights=weights,
        forecast_columns=forecast_columns,
        appearance_columns=appearance_columns,
        no_show_columns=no_show_columns,
    )


__all__ = [
    "DEFAULT_BENCH_WEIGHTS",
    "DEFAULT_GW_WEIGHTS",
    "GameweekPlan",
    "SquadPlanError",
    "SquadPlanInfeasibleError",
    "SquadPlanResult",
    "optimize_squad_plan",
]
