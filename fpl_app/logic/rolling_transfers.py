"""Deterministic, rolling transfer decisions for an existing FPL squad.

The optimiser is deliberately independent of network, API and UI code.  It
compares rolling the current free transfer with making up to the currently
banked number of free transfers now (and always at least two), then evaluates
the resulting fixed squad over one to five supplied
``ep_gwN`` forecasts.  Chips and future-gameweek transfers are intentionally
outside this first bounded version; callers run it again when the next
point-in-time forecast is available.

All prices are integer tenths of a million pounds.  Selling prices for owned
players are explicit inputs and are checked against the official half-profit
rule so an accidental use of current market value cannot inflate the budget.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from math import isfinite
from time import monotonic
from typing import Optional, Sequence

import pandas as pd
import pulp

try:  # Package imports used by Vercel and tests.
    from ..domain.rules import (
        SQUAD,
        TRANSFERS,
        free_transfers_next_gameweek,
        selling_price_tenths,
        transfer_points_cost,
    )
except ImportError:  # pragma: no cover - legacy Streamlit working directory.
    from domain.rules import (  # type: ignore[no-redef]
        SQUAD,
        TRANSFERS,
        free_transfers_next_gameweek,
        selling_price_tenths,
        transfer_points_cost,
    )


DEFAULT_GW_WEIGHTS = (1.0, 0.85, 0.70, 0.55, 0.40)
DEFAULT_BENCH_WEIGHTS = (0.12, 0.08, 0.04, 0.02)
DEFAULT_SOLVER_BUDGET_SECONDS = 15.0
POSITION_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
SQUAD_QUOTA = {
    position.value: count for position, count in SQUAD.position_quotas.items()
}


class RollingTransferError(ValueError):
    """Raised for invalid inputs or an unavailable optimisation result."""


class RollingTransferInfeasibleError(RollingTransferError):
    """Raised when the supplied current squad is not a legal FPL squad."""


@dataclass(frozen=True)
class TransferMove:
    """One position-preserving player replacement."""

    out_id: int
    in_id: int
    position: str
    out_purchase_price_tenths: int
    out_current_price_tenths: int
    out_selling_price_tenths: int
    in_price_tenths: int

    def as_dict(self) -> dict[str, object]:
        return {
            "out_id": self.out_id,
            "in_id": self.in_id,
            "position": self.position,
            "out_purchase_price_tenths": self.out_purchase_price_tenths,
            "out_current_price_tenths": self.out_current_price_tenths,
            "out_selling_price_tenths": self.out_selling_price_tenths,
            "in_price_tenths": self.in_price_tenths,
        }


@dataclass(frozen=True)
class TransferGameweekPlan:
    """The best legal XI and captain for one forecast gameweek."""

    gameweek: int
    starting_ids: tuple[int, ...]
    captain_id: int
    formation: str
    projected_points: float

    def as_dict(self) -> dict[str, object]:
        return {
            "gameweek": self.gameweek,
            "starting_ids": list(self.starting_ids),
            "captain_id": self.captain_id,
            "formation": self.formation,
            "projected_points": self.projected_points,
        }


@dataclass(frozen=True)
class TransferPlan:
    """One immediately executable zero- to five-transfer plan."""

    action: str
    transfers: tuple[TransferMove, ...]
    squad_ids: tuple[int, ...]
    gameweeks: tuple[TransferGameweekPlan, ...]
    bank_before_tenths: int
    bank_after_tenths: int
    free_transfers_before: int
    free_transfers_next_gameweek: int
    hit_points: int
    projected_points: float
    banked_ft_value_points: float
    decision_value_points: float
    delta_vs_base_points: float

    @property
    def transfer_count(self) -> int:
        return len(self.transfers)

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "transfer_count": self.transfer_count,
            "transfers": [move.as_dict() for move in self.transfers],
            "squad_ids": list(self.squad_ids),
            "gameweeks": [gameweek.as_dict() for gameweek in self.gameweeks],
            "bank_before_tenths": self.bank_before_tenths,
            "bank_after_tenths": self.bank_after_tenths,
            "free_transfers_before": self.free_transfers_before,
            "free_transfers_next_gameweek": self.free_transfers_next_gameweek,
            "hit_points": self.hit_points,
            "projected_points": self.projected_points,
            "banked_ft_value_points": self.banked_ft_value_points,
            "decision_value_points": self.decision_value_points,
            "delta_vs_base_points": self.delta_vs_base_points,
        }


@dataclass(frozen=True)
class RollingTransferResult:
    """Decision result with an unadjusted base and actionable alternatives."""

    base: TransferPlan
    roll: TransferPlan
    best_action: TransferPlan
    alternatives: tuple[TransferPlan, ...]
    horizon: int
    gw_weights: tuple[float, ...]
    forecast_columns: tuple[str, ...]
    roll_ft_value_points: float

    def as_dict(self) -> dict[str, object]:
        return {
            "base": self.base.as_dict(),
            "roll": self.roll.as_dict(),
            "best_action": self.best_action.as_dict(),
            "alternatives": [plan.as_dict() for plan in self.alternatives],
            "horizon": self.horizon,
            "gw_weights": list(self.gw_weights),
            "forecast_columns": list(self.forecast_columns),
            "roll_ft_value_points": self.roll_ft_value_points,
        }


def _integer_series(
    frame: pd.DataFrame,
    column: str,
    *,
    minimum: int,
    required_mask: Optional[pd.Series] = None,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    mask = required_mask if required_mask is not None else pd.Series(True, index=frame.index)
    required = values.loc[mask]
    if required.isna().any() or (~required.map(isfinite)).any():
        raise RollingTransferError(f"{column} must contain finite numbers where required")
    if (required < minimum).any() or ((required - required.round()).abs() > 1e-9).any():
        raise RollingTransferError(
            f"{column} must contain integer tenths >= {minimum} where required"
        )
    return values.round().astype("Int64")


def _numeric_weights(values: Sequence[float], horizon: int) -> tuple[float, ...]:
    if len(values) != horizon:
        raise RollingTransferError("gw_weights must contain one value per gameweek")
    converted = tuple(float(value) for value in values)
    if any(not isfinite(value) or value < 0 for value in converted):
        raise RollingTransferError("gw_weights must be finite and non-negative")
    if not any(value > 0 for value in converted):
        raise RollingTransferError("at least one gw_weight must be positive")
    if any(converted[index] < converted[index + 1] for index in range(horizon - 1)):
        raise RollingTransferError("gw_weights must be ordered from highest to lowest")
    return converted


def _numeric_bench_weights(values: Sequence[float]) -> tuple[float, ...]:
    if len(values) != 4:
        raise RollingTransferError("bench_weights must contain exactly 4 values")
    converted = tuple(float(value) for value in values)
    if any(not isfinite(value) or value < 0 for value in converted):
        raise RollingTransferError(
            "bench_weights must contain finite, non-negative values"
        )
    if not any(value > 0 for value in converted):
        raise RollingTransferError("at least one bench_weight must be positive")
    if any(converted[index] < converted[index + 1] for index in range(3)):
        raise RollingTransferError("bench_weights must be ordered from highest to lowest")
    return converted


def _prepare_players(
    players: pd.DataFrame,
    current_squad_ids: Sequence[int],
    horizon: int,
) -> tuple[
    pd.DataFrame,
    tuple[str, ...],
    tuple[str, ...],
    frozenset[int],
]:
    if not isinstance(players, pd.DataFrame) or players.empty:
        raise RollingTransferError("players must be a non-empty pandas DataFrame")

    forecast_columns = tuple(f"ep_gw{gameweek}" for gameweek in range(1, horizon + 1))
    no_show_columns = tuple(
        f"no_show_prob_gw{gameweek}" for gameweek in range(1, horizon + 1)
    )
    required = {
        "id",
        "team_id",
        "pos",
        "now_cost",
        "purchase_price",
        "selling_price",
        *forecast_columns,
    }
    missing = sorted(required - set(players.columns))
    if missing:
        raise RollingTransferError(f"missing required columns: {', '.join(missing)}")

    try:
        squad_ids = tuple(int(player_id) for player_id in current_squad_ids)
    except (TypeError, ValueError) as exc:
        raise RollingTransferError("current_squad_ids must contain integer player ids") from exc
    if len(squad_ids) != SQUAD.squad_size or len(set(squad_ids)) != SQUAD.squad_size:
        raise RollingTransferInfeasibleError(
            f"current_squad_ids must contain exactly {SQUAD.squad_size} unique ids"
        )
    if any(player_id < 1 for player_id in squad_ids):
        raise RollingTransferError("player ids must be positive integers")
    owned_ids = frozenset(squad_ids)

    frame = players.copy(deep=True).reset_index(drop=True)
    frame["id"] = _integer_series(frame, "id", minimum=1).astype(int)
    if frame["id"].duplicated().any():
        raise RollingTransferError("player ids must be unique")
    absent = sorted(owned_ids - set(frame["id"]))
    if absent:
        raise RollingTransferError(
            "current squad ids missing from players: " + ", ".join(map(str, absent))
        )
    frame["team_id"] = _integer_series(frame, "team_id", minimum=1).astype(int)
    frame["now_cost"] = _integer_series(frame, "now_cost", minimum=0).astype(int)
    frame["pos"] = frame["pos"].astype(str).str.upper().replace({"GK": "GKP"})
    unknown_positions = sorted(set(frame["pos"]) - set(SQUAD_QUOTA))
    if unknown_positions:
        raise RollingTransferError(f"unknown positions: {', '.join(unknown_positions)}")

    owned_mask = frame["id"].isin(owned_ids)
    frame["purchase_price"] = _integer_series(
        frame,
        "purchase_price",
        minimum=0,
        required_mask=owned_mask,
    )
    frame["selling_price"] = _integer_series(
        frame,
        "selling_price",
        minimum=0,
        required_mask=owned_mask,
    )
    for index in frame.index[owned_mask]:
        purchase = int(frame.at[index, "purchase_price"])
        current = int(frame.at[index, "now_cost"])
        supplied_sale = int(frame.at[index, "selling_price"])
        expected_sale = selling_price_tenths(purchase, current)
        if supplied_sale != expected_sale:
            player_id = int(frame.at[index, "id"])
            raise RollingTransferError(
                f"selling_price for owned player {player_id} must be {expected_sale} "
                "under the FPL half-profit rule"
            )

    for column in forecast_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (~values.map(isfinite)).any():
            raise RollingTransferError(f"{column} must contain finite numeric values")
        frame[column] = values.astype(float)

    # Older callers supplied only EP.  Treat a missing independent no-show
    # signal as zero to preserve that API while using it whenever available.
    for column in no_show_columns:
        if column not in frame.columns:
            frame[column] = 0.0
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (~values.map(isfinite)).any():
            raise RollingTransferError(f"{column} must contain finite numeric values")
        if ((values < 0) | (values > 1)).any():
            raise RollingTransferError(
                f"{column} must contain probabilities from 0 to 1"
            )
        frame[column] = values.astype(float)

    current = frame.loc[owned_mask]
    position_counts = Counter(current["pos"])
    if any(position_counts.get(position, 0) != quota for position, quota in SQUAD_QUOTA.items()):
        raise RollingTransferInfeasibleError("current squad violates the FPL position quotas")
    # FPL can leave a manager temporarily above the normal club limit after a
    # real-world player transfer.  Rolling that grandfathered squad is legal;
    # any FPL transfer action must restore the ordinary limit.

    frame = frame.sort_values("id", kind="stable").reset_index(drop=True)
    return frame, forecast_columns, no_show_columns, owned_ids


def _action_sort_key(plan: TransferPlan) -> tuple[object, ...]:
    signature = tuple((move.out_id, move.in_id) for move in plan.transfers)
    return (
        -plan.decision_value_points,
        plan.hit_points,
        plan.transfer_count,
        signature,
    )


def _build_moves(
    frame: pd.DataFrame,
    outgoing_indices: Sequence[int],
    incoming_indices: Sequence[int],
) -> tuple[TransferMove, ...]:
    outgoing_by_position: dict[str, list[int]] = {position: [] for position in SQUAD_QUOTA}
    incoming_by_position: dict[str, list[int]] = {position: [] for position in SQUAD_QUOTA}
    for index in outgoing_indices:
        outgoing_by_position[str(frame.at[index, "pos"])].append(index)
    for index in incoming_indices:
        incoming_by_position[str(frame.at[index, "pos"])].append(index)

    moves: list[TransferMove] = []
    for position in sorted(SQUAD_QUOTA, key=POSITION_ORDER.__getitem__):
        outgoing = sorted(
            outgoing_by_position[position], key=lambda index: int(frame.at[index, "id"])
        )
        incoming = sorted(
            incoming_by_position[position], key=lambda index: int(frame.at[index, "id"])
        )
        if len(outgoing) != len(incoming):
            raise RollingTransferError("optimal transfers do not preserve position quotas")
        for out_index, in_index in zip(outgoing, incoming):
            moves.append(
                TransferMove(
                    out_id=int(frame.at[out_index, "id"]),
                    in_id=int(frame.at[in_index, "id"]),
                    position=position,
                    out_purchase_price_tenths=int(frame.at[out_index, "purchase_price"]),
                    out_current_price_tenths=int(frame.at[out_index, "now_cost"]),
                    out_selling_price_tenths=int(frame.at[out_index, "selling_price"]),
                    in_price_tenths=int(frame.at[in_index, "now_cost"]),
                )
            )
    return tuple(moves)


def _solve_transfer_count(
    frame: pd.DataFrame,
    forecast_columns: tuple[str, ...],
    no_show_columns: tuple[str, ...],
    owned_ids: frozenset[int],
    *,
    transfer_count: int,
    bank_tenths: int,
    free_transfers: int,
    weights: tuple[float, ...],
    bench_weights: tuple[float, ...],
    roll_ft_value_points: float,
    number_of_plans: int,
    solver: Optional[pulp.LpSolver],
    solver_deadline: float,
    excluded_incoming_id_sets: Sequence[frozenset[int]] = (),
    require_conclusive_first_solve: bool = False,
) -> list[TransferPlan]:
    indices = list(frame.index)
    owned = [index for index in indices if int(frame.at[index, "id"]) in owned_ids]
    available = [index for index in indices if int(frame.at[index, "id"]) not in owned_ids]
    gameweeks = range(len(forecast_columns))

    model = pulp.LpProblem(f"rolling_transfer_{transfer_count}", pulp.LpMaximize)
    squad = {index: pulp.LpVariable(f"squad_{index}", cat="Binary") for index in indices}
    start = {
        (gameweek, index): pulp.LpVariable(
            f"start_{gameweek + 1}_{index}", cat="Binary"
        )
        for gameweek in gameweeks
        for index in indices
    }
    captain = {
        (gameweek, index): pulp.LpVariable(
            f"captain_{gameweek + 1}_{index}", cat="Binary"
        )
        for gameweek in gameweeks
        for index in indices
    }

    model += pulp.lpSum(squad.values()) == SQUAD.squad_size
    model += pulp.lpSum(squad[index] for index in available) == transfer_count
    for position, quota in SQUAD_QUOTA.items():
        members = [index for index in indices if frame.at[index, "pos"] == position]
        model += pulp.lpSum(squad[index] for index in members) == quota
    if transfer_count > 0:
        for _, members in frame.groupby("team_id").groups.items():
            model += (
                pulp.lpSum(squad[int(index)] for index in members)
                <= SQUAD.max_players_per_club
            )

    sale_proceeds = pulp.lpSum(
        (1 - squad[index]) * int(frame.at[index, "selling_price"])
        for index in owned
    )
    purchase_cost = pulp.lpSum(
        squad[index] * int(frame.at[index, "now_cost"]) for index in available
    )
    model += purchase_cost <= bank_tenths + sale_proceeds

    available_by_id = {int(frame.at[index, "id"]): index for index in available}
    for exclusion_number, incoming_ids in enumerate(excluded_incoming_id_sets):
        excluded_indices = [
            available_by_id[player_id]
            for player_id in incoming_ids
            if player_id in available_by_id
        ]
        if len(excluded_indices) != transfer_count:
            raise RollingTransferError(
                "excluded incoming sets must identify one complete transfer plan"
            )
        model += (
            pulp.lpSum(squad[index] for index in excluded_indices)
            <= transfer_count - 1,
            f"exclude_seed_{transfer_count}_{exclusion_number}",
        )

    goalkeepers = [index for index in indices if frame.at[index, "pos"] == "GKP"]
    defenders = [index for index in indices if frame.at[index, "pos"] == "DEF"]
    midfielders = [index for index in indices if frame.at[index, "pos"] == "MID"]
    forwards = [index for index in indices if frame.at[index, "pos"] == "FWD"]
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

    start_score_milli = {
        (gameweek, index): int(
            round(
                float(frame.at[index, forecast_columns[gameweek]])
                * weights[gameweek]
                * 1000
            )
        )
        for gameweek in gameweeks
        for index in indices
    }
    bench_score_milli = {}
    average_outfield_bench_weight = sum(bench_weights[:3]) / 3.0
    for gameweek in gameweeks:
        ep_column = forecast_columns[gameweek]
        no_show_column = no_show_columns[gameweek]
        for index in indices:
            reserve_weight = (
                bench_weights[3]
                if frame.at[index, "pos"] == "GKP"
                else average_outfield_bench_weight
            )
            bench_score_milli[(gameweek, index)] = int(
                round(
                    float(frame.at[index, ep_column])
                    * (1.0 - float(frame.at[index, no_show_column]))
                    * reserve_weight
                    * weights[gameweek]
                    * 1000
                )
            )
    primary = pulp.lpSum(
        (start[(gameweek, index)] + captain[(gameweek, index)])
        * start_score_milli[(gameweek, index)]
        + (squad[index] - start[(gameweek, index)])
        * bench_score_milli[(gameweek, index)]
        for gameweek in gameweeks
        for index in indices
    )

    # A bounded integer secondary objective makes tied CBC solutions independent
    # of input row order without ever overturning a 0.001-point EP difference.
    count = len(indices)
    tie = pulp.lpSum(
        (count - index)
        * (
            squad[index]
            + pulp.lpSum(
                start[(gameweek, index)] + 2 * captain[(gameweek, index)]
                for gameweek in gameweeks
            )
        )
        for index in indices
    )
    tie_upper_bound = max(1, count * count * (1 + 3 * len(forecast_columns)))
    model += primary * (tie_upper_bound + 1) + tie

    results: list[TransferPlan] = []
    for alternative_number in range(number_of_plans):
        remaining_seconds = solver_deadline - monotonic()
        if remaining_seconds <= 0:
            if require_conclusive_first_solve and alternative_number == 0:
                raise RollingTransferError(
                    "the optimisation could not conclusively evaluate "
                    f"{transfer_count} transfers within the shared "
                    f"{DEFAULT_SOLVER_BUDGET_SECONDS:g}-second solver budget"
                )
            break
        selected_solver = solver or pulp.PULP_CBC_CMD(
            msg=False,
            threads=1,
            timeLimit=remaining_seconds,
            options=["randomSeed 17", "randomCbcSeed 17"],
        )
        original_time_limit = getattr(selected_solver, "timeLimit", None)
        if solver is not None:
            bounded_time_limit = remaining_seconds
            if (
                isinstance(original_time_limit, (int, float))
                and original_time_limit > 0
            ):
                bounded_time_limit = min(float(original_time_limit), remaining_seconds)
            selected_solver.timeLimit = bounded_time_limit
        try:
            model.solve(selected_solver)
        except pulp.PulpSolverError as exc:
            raise RollingTransferError(f"the optimisation solver could not run: {exc}") from exc
        finally:
            if solver is not None:
                selected_solver.timeLimit = original_time_limit
        status = pulp.LpStatus.get(model.status, str(model.status))
        solution_status = getattr(model, "sol_status", None)
        proven_optimal = status == "Optimal" and solution_status in {
            None,
            pulp.LpSolutionOptimal,
        }
        if not proven_optimal:
            if status == "Infeasible":
                break
            if require_conclusive_first_solve and alternative_number == 0:
                raise RollingTransferError(
                    "the optimisation could not conclusively evaluate "
                    f"{transfer_count} transfers within the shared solver budget "
                    f"(status: {status})"
                )
            break

        chosen = [index for index in indices if pulp.value(squad[index]) > 0.5]
        incoming = [index for index in available if pulp.value(squad[index]) > 0.5]
        outgoing = [index for index in owned if pulp.value(squad[index]) < 0.5]
        moves = _build_moves(frame, outgoing, incoming)
        sale_total = sum(move.out_selling_price_tenths for move in moves)
        buy_total = sum(move.in_price_tenths for move in moves)
        bank_after = bank_tenths + sale_total - buy_total
        hit_points = transfer_points_cost(transfer_count, free_transfers)
        next_free_transfers = free_transfers_next_gameweek(
            free_transfers,
            transfer_count,
        )
        # The automatic single FT available next week is common to all actions.
        # Only banked flexibility above that baseline receives decision utility.
        banked_ft_value = (
            max(0, next_free_transfers - TRANSFERS.free_transfers_per_gameweek)
            * roll_ft_value_points
        )

        gameweek_plans: list[TransferGameweekPlan] = []
        projected_points = 0.0
        for gameweek in gameweeks:
            ep_column = forecast_columns[gameweek]
            starters = [
                index
                for index in indices
                if pulp.value(start[(gameweek, index)]) > 0.5
            ]
            captain_index = next(
                index
                for index in indices
                if pulp.value(captain[(gameweek, index)]) > 0.5
            )
            starters.sort(
                key=lambda index: (
                    POSITION_ORDER[str(frame.at[index, "pos"])],
                    -float(frame.at[index, ep_column]),
                    int(frame.at[index, "id"]),
                )
            )
            position_counts = Counter(str(frame.at[index, "pos"]) for index in starters)
            formation = (
                f"{position_counts['DEF']}-{position_counts['MID']}-"
                f"{position_counts['FWD']}"
            )
            gameweek_points = sum(float(frame.at[index, ep_column]) for index in starters)
            gameweek_points += float(frame.at[captain_index, ep_column])
            starter_set = set(starters)
            for index in chosen:
                if index in starter_set:
                    continue
                reserve_weight = (
                    bench_weights[3]
                    if frame.at[index, "pos"] == "GKP"
                    else average_outfield_bench_weight
                )
                gameweek_points += (
                    float(frame.at[index, ep_column])
                    * (1.0 - float(frame.at[index, no_show_columns[gameweek]]))
                    * reserve_weight
                )
            projected_points += gameweek_points * weights[gameweek]
            gameweek_plans.append(
                TransferGameweekPlan(
                    gameweek=gameweek + 1,
                    starting_ids=tuple(int(frame.at[index, "id"]) for index in starters),
                    captain_id=int(frame.at[captain_index, "id"]),
                    formation=formation,
                    projected_points=round(gameweek_points, 3),
                )
            )

        squad_ids = tuple(sorted(int(frame.at[index, "id"]) for index in chosen))
        decision_value = projected_points - hit_points + banked_ft_value
        action = "roll" if transfer_count == 0 else f"{transfer_count}_transfer"
        results.append(
            TransferPlan(
                action=action,
                transfers=moves,
                squad_ids=squad_ids,
                gameweeks=tuple(gameweek_plans),
                bank_before_tenths=bank_tenths,
                bank_after_tenths=bank_after,
                free_transfers_before=free_transfers,
                free_transfers_next_gameweek=next_free_transfers,
                hit_points=hit_points,
                projected_points=round(projected_points, 3),
                banked_ft_value_points=round(banked_ft_value, 3),
                decision_value_points=round(decision_value, 3),
                delta_vs_base_points=0.0,
            )
        )

        if transfer_count == 0:
            break
        # Exclude this exact incoming set.  Squad size and exact transfer count
        # make the incoming set a unique transfer plan.
        model += (
            pulp.lpSum(squad[index] for index in incoming) <= transfer_count - 1,
            f"exclude_{transfer_count}_{alternative_number}",
        )

    return results


def optimize_rolling_transfers(
    players: pd.DataFrame,
    current_squad_ids: Sequence[int],
    *,
    bank_tenths: int,
    free_transfers: int,
    horizon: int = 5,
    gw_weights: Optional[Sequence[float]] = None,
    roll_ft_value_points: float = 1.0,
    plans_per_transfer_count: int = 5,
    bench_weights: Sequence[float] = DEFAULT_BENCH_WEIGHTS,
    solver: Optional[pulp.LpSolver] = None,
) -> RollingTransferResult:
    """Compare rolling with the best legal immediate transfer plans.

    ``players`` must contain the full candidate pool.  ``purchase_price`` and
    ``selling_price`` are required only for the 15 ids in
    ``current_squad_ids``; candidate rows may use missing values because their
    acquisition cost is ``now_cost``.  Forecasts are direct, point-in-time
    ``ep_gw1`` ... ``ep_gwN`` columns.

    The decision score is optimal weighted XI, captain points and discounted
    availability-adjusted bench value, less four points per transfer above the
    current free-transfer bank, plus
    ``roll_ft_value_points`` for each next-gameweek FT banked above the ordinary
    one-transfer baseline.  No chips or speculative later transfers are used.
    """

    if not isinstance(horizon, int) or isinstance(horizon, bool) or not 1 <= horizon <= 5:
        raise RollingTransferError("horizon must be an integer from 1 to 5")
    if (
        not isinstance(bank_tenths, int)
        or isinstance(bank_tenths, bool)
        or bank_tenths < 0
    ):
        raise RollingTransferError("bank_tenths must be a non-negative integer")
    if (
        not isinstance(free_transfers, int)
        or isinstance(free_transfers, bool)
        or not 1 <= free_transfers <= TRANSFERS.max_banked_free_transfers
    ):
        raise RollingTransferError(
            f"free_transfers must be an integer from 1 to {TRANSFERS.max_banked_free_transfers}"
        )
    if (
        not isinstance(plans_per_transfer_count, int)
        or isinstance(plans_per_transfer_count, bool)
        or plans_per_transfer_count < 1
    ):
        raise RollingTransferError("plans_per_transfer_count must be a positive integer")
    roll_value = float(roll_ft_value_points)
    if not isfinite(roll_value) or roll_value < 0:
        raise RollingTransferError("roll_ft_value_points must be finite and non-negative")

    weights = _numeric_weights(
        gw_weights if gw_weights is not None else DEFAULT_GW_WEIGHTS[:horizon],
        horizon,
    )
    reserve_weights = _numeric_bench_weights(bench_weights)
    frame, forecast_columns, no_show_columns, owned_ids = _prepare_players(
        players,
        current_squad_ids,
        horizon,
    )
    solver_deadline = monotonic() + DEFAULT_SOLVER_BUDGET_SECONDS

    roll_candidates = _solve_transfer_count(
        frame,
        forecast_columns,
        no_show_columns,
        owned_ids,
        transfer_count=0,
        bank_tenths=bank_tenths,
        free_transfers=free_transfers,
        weights=weights,
        bench_weights=reserve_weights,
        roll_ft_value_points=roll_value,
        number_of_plans=1,
        solver=solver,
        solver_deadline=solver_deadline,
        require_conclusive_first_solve=True,
    )
    if not roll_candidates:
        raise RollingTransferInfeasibleError("the current squad has no legal lineup")
    roll = roll_candidates[0]
    base = replace(
        roll,
        action="base",
        banked_ft_value_points=0.0,
        decision_value_points=roll.projected_points,
        delta_vs_base_points=0.0,
    )

    actionable: list[TransferPlan] = [roll]
    maximum_immediate_transfers = max(2, free_transfers)
    coverage_plans: dict[int, TransferPlan] = {}
    for transfer_count in range(1, maximum_immediate_transfers + 1):
        candidates = _solve_transfer_count(
            frame,
            forecast_columns,
            no_show_columns,
            owned_ids,
            transfer_count=transfer_count,
            bank_tenths=bank_tenths,
            free_transfers=free_transfers,
            weights=weights,
            bench_weights=reserve_weights,
            roll_ft_value_points=roll_value,
            number_of_plans=1,
            solver=solver,
            solver_deadline=solver_deadline,
            require_conclusive_first_solve=True,
        )
        if candidates:
            coverage_plans[transfer_count] = candidates[0]
            actionable.append(candidates[0])

    # Spend only the remaining budget on alternative one- and two-transfer
    # plans.  Seed exclusions ensure that rebuilding the MILP cannot return the
    # already collected coverage plan.
    extra_plans_per_count = plans_per_transfer_count - 1
    if extra_plans_per_count > 0:
        for transfer_count in (1, 2):
            if monotonic() >= solver_deadline:
                break
            coverage_plan = coverage_plans.get(transfer_count)
            if coverage_plan is None:
                continue
            excluded_incoming_ids = frozenset(
                move.in_id for move in coverage_plan.transfers
            )
            actionable.extend(
                _solve_transfer_count(
                    frame,
                    forecast_columns,
                    no_show_columns,
                    owned_ids,
                    transfer_count=transfer_count,
                    bank_tenths=bank_tenths,
                    free_transfers=free_transfers,
                    weights=weights,
                    bench_weights=reserve_weights,
                    roll_ft_value_points=roll_value,
                    number_of_plans=extra_plans_per_count,
                    solver=solver,
                    solver_deadline=solver_deadline,
                    excluded_incoming_id_sets=(excluded_incoming_ids,),
                )
            )

    actionable = [
        replace(
            plan,
            delta_vs_base_points=round(
                plan.decision_value_points - base.projected_points,
                3,
            ),
        )
        for plan in actionable
    ]
    actionable.sort(key=_action_sort_key)
    best_action = actionable[0]
    alternatives = tuple(actionable[1:])
    adjusted_roll = next(plan for plan in actionable if plan.transfer_count == 0)
    return RollingTransferResult(
        base=base,
        roll=adjusted_roll,
        best_action=best_action,
        alternatives=alternatives,
        horizon=horizon,
        gw_weights=weights,
        forecast_columns=forecast_columns,
        roll_ft_value_points=roll_value,
    )


__all__ = [
    "DEFAULT_BENCH_WEIGHTS",
    "DEFAULT_GW_WEIGHTS",
    "RollingTransferError",
    "RollingTransferInfeasibleError",
    "RollingTransferResult",
    "TransferGameweekPlan",
    "TransferMove",
    "TransferPlan",
    "optimize_rolling_transfers",
]
