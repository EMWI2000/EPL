"""Bounded four-deadline transfer roadmaps for an existing FPL squad.

The executable first deadline is restricted to explicit :class:`TransferPlan`
candidates produced by the immediate planner.  One CBC MILP then models three
additional provisional deadlines and evaluates the fourth resulting squad over
the rest of a six- to ten-gameweek forecast horizon.

The model deliberately keeps today's player prices fixed.  An initially owned
player retains the manager's confirmed purchase/selling basis until first sold;
any later purchase uses today's market price as both purchase and selling basis.
The result is optimal only inside the supplied first-step set, player pool,
four-deadline horizon and two-transfer cap at provisional deadlines.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from typing import Collection, Optional, Sequence

import pandas as pd
import pulp

from ..domain.rules import (
    SQUAD,
    TRANSFERS,
    free_transfers_next_gameweek,
    selling_price_tenths,
    transfer_points_cost,
)
from . import rolling_transfers as immediate
from .rolling_transfers import TransferGameweekPlan, TransferMove, TransferPlan


DEFAULT_STRATEGY_HORIZON = 8
MIN_STRATEGY_HORIZON = 6
MAX_STRATEGY_HORIZON = 10
MODELLED_DEADLINES = 4
MAX_PROVISIONAL_TRANSFERS = 2
MAX_FIRST_STEP_CANDIDATES = 16
DEFAULT_STRATEGY_SOLVER_BUDGET_SECONDS = 15.0
DEFAULT_STRATEGY_GW_WEIGHTS = tuple(0.85**offset for offset in range(MAX_STRATEGY_HORIZON))
FUTURE_PRICE_ASSUMPTION = "fixed_current_prices"
FIRST_STEP_SEARCH = "explicit_bounded_transfer_plans"
SEARCH_SCOPE = "four_deadlines_max_two_provisional_transfers"
ASSUMPTIONS = (
    "fixed_current_prices",
    "four_transfer_deadlines_modelled",
    "maximum_two_transfers_at_each_provisional_deadline",
    "first_action_restricted_to_supplied_explicit_plans",
    "no_transfers_after_deadline_four_assumed",
    "recalculate_at_every_real_deadline",
)


class StrategyPlannerError(ValueError):
    """Raised for invalid input or when no proven bounded roadmap is available."""


@dataclass(frozen=True)
class StrategyTransferStep:
    """One action and the gameweeks governed by its resulting squad."""

    deadline_offset: int
    provisional: bool
    transfers: tuple[TransferMove, ...]
    squad_ids: tuple[int, ...]
    gameweeks: tuple[TransferGameweekPlan, ...]
    bank_before_tenths: int
    bank_after_tenths: int
    free_transfers_before: int
    free_transfers_next_gameweek: int
    hit_points: int
    weighted_projected_points: float
    weighted_hit_cost_points: float

    @property
    def transfer_count(self) -> int:
        return len(self.transfers)

    def as_dict(self) -> dict[str, object]:
        return {
            "deadline_offset": self.deadline_offset,
            "provisional": self.provisional,
            "transfer_count": self.transfer_count,
            "transfers": [move.as_dict() for move in self.transfers],
            "squad_ids": list(self.squad_ids),
            "gameweeks": [gameweek.as_dict() for gameweek in self.gameweeks],
            "bank_before_tenths": self.bank_before_tenths,
            "bank_after_tenths": self.bank_after_tenths,
            "free_transfers_before": self.free_transfers_before,
            "free_transfers_next_gameweek": self.free_transfers_next_gameweek,
            "hit_points": self.hit_points,
            "weighted_projected_points": self.weighted_projected_points,
            "weighted_hit_cost_points": self.weighted_hit_cost_points,
        }


@dataclass(frozen=True)
class FourDeadlineRoadmap:
    """The selected executable action followed by three provisional actions."""

    first_step_candidate_index: int
    steps: tuple[
        StrategyTransferStep,
        StrategyTransferStep,
        StrategyTransferStep,
        StrategyTransferStep,
    ]
    weighted_projected_points: float
    total_hit_points: int
    weighted_hit_cost_points: float
    terminal_banked_ft_value_points: float
    decision_value_points: float

    def as_dict(self) -> dict[str, object]:
        return {
            "first_step_candidate_index": self.first_step_candidate_index,
            "steps": [step.as_dict() for step in self.steps],
            "weighted_projected_points": self.weighted_projected_points,
            "total_hit_points": self.total_hit_points,
            "weighted_hit_cost_points": self.weighted_hit_cost_points,
            "terminal_banked_ft_value_points": self.terminal_banked_ft_value_points,
            "decision_value_points": self.decision_value_points,
        }


@dataclass(frozen=True)
class BoundedStrategyResult:
    """A proven optimum for the explicitly documented bounded roadmap search."""

    best_roadmap: FourDeadlineRoadmap
    horizon: int
    gw_weights: tuple[float, ...]
    forecast_columns: tuple[str, ...]
    first_step_candidate_count: int
    modelled_deadlines: int = MODELLED_DEADLINES
    maximum_provisional_transfers: int = MAX_PROVISIONAL_TRANSFERS
    first_step_search: str = FIRST_STEP_SEARCH
    search_scope: str = SEARCH_SCOPE
    future_price_assumption: str = FUTURE_PRICE_ASSUMPTION
    assumptions: tuple[str, ...] = ASSUMPTIONS
    solver_proven_optimal_within_bounds: bool = True
    globally_optimal: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "best_roadmap": self.best_roadmap.as_dict(),
            "horizon": self.horizon,
            "gw_weights": list(self.gw_weights),
            "forecast_columns": list(self.forecast_columns),
            "first_step_candidate_count": self.first_step_candidate_count,
            "modelled_deadlines": self.modelled_deadlines,
            "maximum_provisional_transfers": self.maximum_provisional_transfers,
            "first_step_search": self.first_step_search,
            "search_scope": self.search_scope,
            "future_price_assumption": self.future_price_assumption,
            "assumptions": list(self.assumptions),
            "solver_proven_optimal_within_bounds": self.solver_proven_optimal_within_bounds,
            "globally_optimal": self.globally_optimal,
        }


@dataclass(frozen=True)
class _PreparedCandidate:
    index: int
    plan: TransferPlan
    squad_ids: frozenset[int]
    uninterrupted_initial_ids: frozenset[int]
    first_gameweek_points: float


def _strict_non_negative_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise StrategyPlannerError(f"{name} must be an integer >= {minimum}")
    return value


def _initial_ids_from_plan(plan: TransferPlan) -> frozenset[int]:
    squad_ids = tuple(int(player_id) for player_id in plan.squad_ids)
    if len(squad_ids) != SQUAD.squad_size or len(set(squad_ids)) != SQUAD.squad_size:
        raise StrategyPlannerError("each first-step squad must contain 15 unique ids")
    out_ids = tuple(move.out_id for move in plan.transfers)
    in_ids = tuple(move.in_id for move in plan.transfers)
    if len(set(out_ids)) != len(out_ids) or len(set(in_ids)) != len(in_ids):
        raise StrategyPlannerError("first-step transfers must not repeat players")
    if set(out_ids) & set(in_ids):
        raise StrategyPlannerError("a first-step player cannot be both sold and bought")
    if not set(in_ids).issubset(squad_ids) or set(out_ids) & set(squad_ids):
        raise StrategyPlannerError("first-step transfers do not reconcile with the resulting squad")
    initial = (set(squad_ids) - set(in_ids)) | set(out_ids)
    if len(initial) != SQUAD.squad_size:
        raise StrategyPlannerError("first-step transfers do not reconstruct one initial squad")
    return frozenset(initial)


def _validate_squad(
    frame_by_id: pd.DataFrame,
    squad_ids: frozenset[int],
    *,
    transfers_made: int,
    path: str,
) -> None:
    if len(squad_ids) != SQUAD.squad_size:
        raise StrategyPlannerError(f"{path} must contain 15 players")
    try:
        squad = frame_by_id.loc[list(squad_ids)]
    except KeyError as exc:
        raise StrategyPlannerError(f"{path} contains a player outside the pool") from exc
    position_counts = Counter(str(value) for value in squad["pos"])
    if any(
        position_counts.get(position, 0) != quota
        for position, quota in immediate.SQUAD_QUOTA.items()
    ):
        raise StrategyPlannerError(f"{path} violates the FPL position quotas")
    if (
        transfers_made > 0
        and int(squad["team_id"].value_counts().max()) > SQUAD.max_players_per_club
    ):
        raise StrategyPlannerError(
            f"{path} must restore the maximum players-per-club rule after a transfer"
        )


def _lineup_points(
    frame_by_id: pd.DataFrame,
    squad_ids: frozenset[int],
    gameweek: TransferGameweekPlan,
    *,
    offset: int,
    bench_weights: tuple[float, ...],
    validate_reported: bool,
) -> float:
    starters = tuple(int(player_id) for player_id in gameweek.starting_ids)
    if len(starters) != SQUAD.starting_size or len(set(starters)) != SQUAD.starting_size:
        raise StrategyPlannerError("each lineup must contain 11 unique ids")
    if not set(starters).issubset(squad_ids):
        raise StrategyPlannerError("a lineup contains a player outside its squad")
    captain_id = int(gameweek.captain_id)
    if captain_id not in starters:
        raise StrategyPlannerError("the captain must be in the starting lineup")

    starter_rows = frame_by_id.loc[list(starters)]
    counts = Counter(str(value) for value in starter_rows["pos"])
    expected_formation = f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"
    if (
        counts["GKP"] != 1
        or not 3 <= counts["DEF"] <= 5
        or not 2 <= counts["MID"] <= 5
        or not 1 <= counts["FWD"] <= 3
        or gameweek.formation != expected_formation
    ):
        raise StrategyPlannerError("a lineup is not a legal FPL formation")

    ep_column = f"ep_gw{offset}"
    no_show_column = f"no_show_prob_gw{offset}"
    points = sum(float(frame_by_id.at[player_id, ep_column]) for player_id in starters)
    points += float(frame_by_id.at[captain_id, ep_column])
    average_outfield_weight = sum(bench_weights[:3]) / 3.0
    for player_id in squad_ids - set(starters):
        reserve_weight = (
            bench_weights[3]
            if str(frame_by_id.at[player_id, "pos"]) == "GKP"
            else average_outfield_weight
        )
        points += (
            float(frame_by_id.at[player_id, ep_column])
            * (1.0 - float(frame_by_id.at[player_id, no_show_column]))
            * reserve_weight
        )
    if validate_reported and abs(float(gameweek.projected_points) - round(points, 3)) > 0.001:
        raise StrategyPlannerError("a first-step projected_points value does not match its lineup")
    return points


def _prepare_inputs(
    players: pd.DataFrame,
    first_step_candidates: Sequence[TransferPlan],
    *,
    horizon: int,
    bench_weights: tuple[float, ...],
) -> tuple[
    pd.DataFrame,
    tuple[str, ...],
    tuple[str, ...],
    frozenset[int],
    tuple[_PreparedCandidate, ...],
]:
    if not isinstance(first_step_candidates, Sequence) or isinstance(
        first_step_candidates, (str, bytes)
    ):
        raise StrategyPlannerError("first_step_candidates must be a sequence")
    candidates = tuple(first_step_candidates)
    if not candidates:
        raise StrategyPlannerError("at least one first-step candidate is required")
    if len(candidates) > MAX_FIRST_STEP_CANDIDATES:
        raise StrategyPlannerError(
            f"at most {MAX_FIRST_STEP_CANDIDATES} first-step candidates are allowed"
        )
    if any(not isinstance(plan, TransferPlan) for plan in candidates):
        raise StrategyPlannerError("first_step_candidates must contain TransferPlan values")

    initial_ids = _initial_ids_from_plan(candidates[0])
    try:
        frame, forecast_columns, no_show_columns, _ = immediate._prepare_players(
            players,
            tuple(sorted(initial_ids)),
            horizon,
        )
    except immediate.RollingTransferError as exc:
        raise StrategyPlannerError(str(exc)) from exc
    frame_by_id = frame.set_index("id", drop=False)

    shared_bank = _strict_non_negative_integer(
        candidates[0].bank_before_tenths,
        "bank_before_tenths",
    )
    shared_ft = _strict_non_negative_integer(
        candidates[0].free_transfers_before,
        "free_transfers_before",
        minimum=1,
    )
    if shared_ft > TRANSFERS.max_banked_free_transfers:
        raise StrategyPlannerError("free_transfers_before is outside the legal range")

    prepared: list[_PreparedCandidate] = []
    for candidate_index, plan in enumerate(candidates):
        path = f"first_step_candidates[{candidate_index}]"
        if _initial_ids_from_plan(plan) != initial_ids:
            raise StrategyPlannerError("all first-step candidates must share one confirmed squad")
        if plan.bank_before_tenths != shared_bank:
            raise StrategyPlannerError("all first-step candidates must share one bank")
        if plan.free_transfers_before != shared_ft:
            raise StrategyPlannerError("all first-step candidates must share one FT bank")

        transfer_count = len(plan.transfers)
        if transfer_count > max(2, shared_ft):
            raise StrategyPlannerError(f"{path} exceeds the immediate transfer-count bound")
        expected_action = "roll" if transfer_count == 0 else f"{transfer_count}_transfer"
        if plan.action != expected_action:
            raise StrategyPlannerError(f"{path}.action must be {expected_action}")

        out_ids: set[int] = set()
        in_ids: set[int] = set()
        sale_total = 0
        buy_total = 0
        for move in plan.transfers:
            if move.out_id in out_ids or move.in_id in in_ids:
                raise StrategyPlannerError(f"{path} repeats a transferred player")
            out_ids.add(move.out_id)
            in_ids.add(move.in_id)
            if move.out_id not in initial_ids or move.in_id in initial_ids:
                raise StrategyPlannerError(
                    f"{path} must replace an owned player with an unowned player"
                )
            if move.out_id not in frame_by_id.index or move.in_id not in frame_by_id.index:
                raise StrategyPlannerError(f"{path} references a player outside the pool")
            outgoing = frame_by_id.loc[move.out_id]
            incoming = frame_by_id.loc[move.in_id]
            expected = (
                str(outgoing["pos"]),
                int(outgoing["purchase_price"]),
                int(outgoing["now_cost"]),
                int(outgoing["selling_price"]),
                int(incoming["now_cost"]),
            )
            supplied = (
                move.position,
                move.out_purchase_price_tenths,
                move.out_current_price_tenths,
                move.out_selling_price_tenths,
                move.in_price_tenths,
            )
            if supplied != expected or str(incoming["pos"]) != move.position:
                raise StrategyPlannerError(f"{path} has inconsistent position or price fields")
            sale_total += move.out_selling_price_tenths
            buy_total += move.in_price_tenths

        expected_squad = (set(initial_ids) - out_ids) | in_ids
        squad_ids = frozenset(int(player_id) for player_id in plan.squad_ids)
        if squad_ids != expected_squad:
            raise StrategyPlannerError(f"{path}.squad_ids do not match its transfers")
        _validate_squad(
            frame_by_id,
            squad_ids,
            transfers_made=transfer_count,
            path=f"{path}.squad_ids",
        )
        expected_bank_after = shared_bank + sale_total - buy_total
        if expected_bank_after < 0 or plan.bank_after_tenths != expected_bank_after:
            raise StrategyPlannerError(f"{path}.bank_after_tenths does not reconcile")
        expected_ft_next = free_transfers_next_gameweek(shared_ft, transfer_count)
        if plan.free_transfers_next_gameweek != expected_ft_next:
            raise StrategyPlannerError(f"{path}.free_transfers_next_gameweek is inconsistent")
        expected_hit = transfer_points_cost(transfer_count, shared_ft)
        if plan.hit_points != expected_hit:
            raise StrategyPlannerError(f"{path}.hit_points is inconsistent")
        if not plan.gameweeks or plan.gameweeks[0].gameweek != 1:
            raise StrategyPlannerError(f"{path}.gameweeks must begin at relative gameweek 1")
        first_points = _lineup_points(
            frame_by_id,
            squad_ids,
            plan.gameweeks[0],
            offset=1,
            bench_weights=bench_weights,
            validate_reported=True,
        )
        prepared.append(
            _PreparedCandidate(
                index=candidate_index,
                plan=plan,
                squad_ids=squad_ids,
                uninterrupted_initial_ids=frozenset(initial_ids - out_ids),
                first_gameweek_points=first_points,
            )
        )

    return frame, forecast_columns, no_show_columns, initial_ids, tuple(prepared)


def _build_moves(
    frame: pd.DataFrame,
    purchase_prices: dict[int, int],
    outgoing_indices: Sequence[int],
    incoming_indices: Sequence[int],
) -> tuple[TransferMove, ...]:
    outgoing_by_position: dict[str, list[int]] = {
        position: [] for position in immediate.SQUAD_QUOTA
    }
    incoming_by_position: dict[str, list[int]] = {
        position: [] for position in immediate.SQUAD_QUOTA
    }
    for index in outgoing_indices:
        outgoing_by_position[str(frame.at[index, "pos"])].append(index)
    for index in incoming_indices:
        incoming_by_position[str(frame.at[index, "pos"])].append(index)

    moves: list[TransferMove] = []
    for position in sorted(immediate.SQUAD_QUOTA, key=immediate.POSITION_ORDER.__getitem__):
        outgoing = sorted(
            outgoing_by_position[position],
            key=lambda index: int(frame.at[index, "id"]),
        )
        incoming = sorted(
            incoming_by_position[position],
            key=lambda index: int(frame.at[index, "id"]),
        )
        if len(outgoing) != len(incoming):
            raise StrategyPlannerError("provisional transfers do not preserve position quotas")
        for out_index, in_index in zip(outgoing, incoming):
            out_id = int(frame.at[out_index, "id"])
            purchase_price = purchase_prices[out_id]
            current_price = int(frame.at[out_index, "now_cost"])
            moves.append(
                TransferMove(
                    out_id=out_id,
                    in_id=int(frame.at[in_index, "id"]),
                    position=position,
                    out_purchase_price_tenths=purchase_price,
                    out_current_price_tenths=current_price,
                    out_selling_price_tenths=selling_price_tenths(
                        purchase_price,
                        current_price,
                    ),
                    in_price_tenths=int(frame.at[in_index, "now_cost"]),
                )
            )
    return tuple(moves)


def optimize_strategy_roadmap(
    players: pd.DataFrame,
    first_step_candidates: Sequence[TransferPlan],
    *,
    horizon: int = DEFAULT_STRATEGY_HORIZON,
    gw_weights: Optional[Sequence[float]] = None,
    roll_ft_value_points: float = 1.0,
    bench_weights: Sequence[float] = immediate.DEFAULT_BENCH_WEIGHTS,
    eligible_transfer_in_ids: Optional[Collection[int]] = None,
    solver_budget_seconds: float = DEFAULT_STRATEGY_SOLVER_BUDGET_SECONDS,
    solver: Optional[pulp.LpSolver] = None,
) -> BoundedStrategyResult:
    """Optimise one bounded four-deadline roadmap across six to ten gameweeks."""

    if (
        not isinstance(horizon, int)
        or isinstance(horizon, bool)
        or not MIN_STRATEGY_HORIZON <= horizon <= MAX_STRATEGY_HORIZON
    ):
        raise StrategyPlannerError(
            f"horizon must be an integer from {MIN_STRATEGY_HORIZON} to {MAX_STRATEGY_HORIZON}"
        )
    try:
        weights = immediate._numeric_weights(
            gw_weights
            if gw_weights is not None
            else DEFAULT_STRATEGY_GW_WEIGHTS[:horizon],
            horizon,
        )
        reserve_weights = immediate._numeric_bench_weights(bench_weights)
    except immediate.RollingTransferError as exc:
        raise StrategyPlannerError(str(exc)) from exc
    roll_value = float(roll_ft_value_points)
    if not isfinite(roll_value) or roll_value < 0:
        raise StrategyPlannerError("roll_ft_value_points must be finite and non-negative")
    budget_seconds = float(solver_budget_seconds)
    if not isfinite(budget_seconds) or budget_seconds <= 0:
        raise StrategyPlannerError("solver_budget_seconds must be finite and positive")

    (
        frame,
        forecast_columns,
        no_show_columns,
        initial_ids,
        candidates,
    ) = _prepare_inputs(
        players,
        first_step_candidates,
        horizon=horizon,
        bench_weights=reserve_weights,
    )
    indices = list(frame.index)
    frame_by_id = frame.set_index("id", drop=False)
    pool_ids = frozenset(int(value) for value in frame["id"])
    if eligible_transfer_in_ids is None:
        eligible_ids = pool_ids
    else:
        if isinstance(eligible_transfer_in_ids, (str, bytes)):
            raise StrategyPlannerError(
                "eligible_transfer_in_ids must be a collection of player ids"
            )
        supplied = tuple(eligible_transfer_in_ids)
        if any(
            not isinstance(player_id, int)
            or isinstance(player_id, bool)
            or player_id < 1
            for player_id in supplied
        ):
            raise StrategyPlannerError(
                "eligible_transfer_in_ids must contain positive integer ids"
            )
        eligible_ids = frozenset(supplied)
        if not eligible_ids.issubset(pool_ids):
            raise StrategyPlannerError(
                "eligible_transfer_in_ids contains a player outside the pool"
            )

    candidate_range = range(len(candidates))
    provisional_deadlines = range(2, MODELLED_DEADLINES + 1)
    all_deadlines = range(1, MODELLED_DEADLINES + 1)
    future_gameweeks = range(2, horizon + 1)

    model = pulp.LpProblem("bounded_four_deadline_strategy", pulp.LpMaximize)
    selected = {
        candidate_index: pulp.LpVariable(f"select_first_{candidate_index}", cat="Binary")
        for candidate_index in candidate_range
    }
    model += pulp.lpSum(selected.values()) == 1
    for candidate_index, candidate in enumerate(candidates):
        if any(move.in_id not in eligible_ids for move in candidate.plan.transfers):
            model += selected[candidate_index] == 0

    squad = {
        (deadline, index): pulp.LpVariable(
            f"squad_d{deadline}_{index}",
            cat="Binary",
        )
        for deadline in all_deadlines
        for index in indices
    }
    original_basis = {
        (deadline, index): pulp.LpVariable(
            f"original_basis_d{deadline}_{index}",
            cat="Binary",
        )
        for deadline in all_deadlines
        for index in indices
    }

    for index in indices:
        player_id = int(frame.at[index, "id"])
        model += squad[(1, index)] == pulp.lpSum(
            selected[candidate_index]
            for candidate_index, candidate in enumerate(candidates)
            if player_id in candidate.squad_ids
        )
        model += original_basis[(1, index)] == pulp.lpSum(
            selected[candidate_index]
            for candidate_index, candidate in enumerate(candidates)
            if player_id in candidate.uninterrupted_initial_ids
        )

    transfer_in = {
        (deadline, index): pulp.LpVariable(
            f"transfer_in_d{deadline}_{index}",
            cat="Binary",
        )
        for deadline in provisional_deadlines
        for index in indices
    }
    transfer_out = {
        (deadline, index): pulp.LpVariable(
            f"transfer_out_d{deadline}_{index}",
            cat="Binary",
        )
        for deadline in provisional_deadlines
        for index in indices
    }
    original_sale = {
        (deadline, index): pulp.LpVariable(
            f"original_sale_d{deadline}_{index}",
            cat="Binary",
        )
        for deadline in provisional_deadlines
        for index in indices
    }

    for deadline in provisional_deadlines:
        for index in indices:
            previous_squad = squad[(deadline - 1, index)]
            incoming = transfer_in[(deadline, index)]
            outgoing = transfer_out[(deadline, index)]
            legacy = original_basis[(deadline - 1, index)]
            legacy_sale = original_sale[(deadline, index)]
            model += squad[(deadline, index)] == previous_squad + incoming - outgoing
            model += incoming <= 1 - previous_squad
            model += outgoing <= previous_squad
            model += legacy_sale <= outgoing
            model += legacy_sale <= legacy
            model += legacy_sale >= outgoing + legacy - 1
            model += original_basis[(deadline, index)] == legacy - legacy_sale
            if int(frame.at[index, "id"]) not in eligible_ids:
                model += incoming == 0

    count_choice: dict[tuple[int, int, int], pulp.LpVariable] = {}
    for deadline in provisional_deadlines:
        choices = []
        for ft_before in range(1, TRANSFERS.max_banked_free_transfers + 1):
            for transfer_count in range(MAX_PROVISIONAL_TRANSFERS + 1):
                choice = pulp.LpVariable(
                    f"state_d{deadline}_ft{ft_before}_n{transfer_count}",
                    cat="Binary",
                )
                count_choice[(deadline, ft_before, transfer_count)] = choice
                choices.append(choice)
        model += pulp.lpSum(choices) == 1

    transfer_count_expression: dict[int, pulp.LpAffineExpression] = {}
    ft_before_expression: dict[int, pulp.LpAffineExpression] = {}
    ft_next_expression: dict[int, pulp.LpAffineExpression] = {}
    for deadline in provisional_deadlines:
        transfer_count_expression[deadline] = pulp.lpSum(
            transfer_count * count_choice[(deadline, ft_before, transfer_count)]
            for ft_before in range(1, TRANSFERS.max_banked_free_transfers + 1)
            for transfer_count in range(MAX_PROVISIONAL_TRANSFERS + 1)
        )
        ft_before_expression[deadline] = pulp.lpSum(
            ft_before * count_choice[(deadline, ft_before, transfer_count)]
            for ft_before in range(1, TRANSFERS.max_banked_free_transfers + 1)
            for transfer_count in range(MAX_PROVISIONAL_TRANSFERS + 1)
        )
        ft_next_expression[deadline] = pulp.lpSum(
            free_transfers_next_gameweek(ft_before, transfer_count)
            * count_choice[(deadline, ft_before, transfer_count)]
            for ft_before in range(1, TRANSFERS.max_banked_free_transfers + 1)
            for transfer_count in range(MAX_PROVISIONAL_TRANSFERS + 1)
        )
        model += (
            pulp.lpSum(transfer_in[(deadline, index)] for index in indices)
            == transfer_count_expression[deadline]
        )
        model += (
            pulp.lpSum(transfer_out[(deadline, index)] for index in indices)
            == transfer_count_expression[deadline]
        )

    model += ft_before_expression[2] == pulp.lpSum(
        candidate.plan.free_transfers_next_gameweek * selected[candidate_index]
        for candidate_index, candidate in enumerate(candidates)
    )
    for deadline in range(3, MODELLED_DEADLINES + 1):
        model += ft_before_expression[deadline] == ft_next_expression[deadline - 1]

    for deadline in all_deadlines:
        model += pulp.lpSum(squad[(deadline, index)] for index in indices) == SQUAD.squad_size
        for position, quota in immediate.SQUAD_QUOTA.items():
            members = [
                index for index in indices if str(frame.at[index, "pos"]) == position
            ]
            model += pulp.lpSum(squad[(deadline, index)] for index in members) == quota
        if deadline >= 2:
            no_transfer = pulp.lpSum(
                count_choice[(deadline, ft_before, 0)]
                for ft_before in range(1, TRANSFERS.max_banked_free_transfers + 1)
            )
            for _, members in frame.groupby("team_id").groups.items():
                model += (
                    pulp.lpSum(squad[(deadline, int(index))] for index in members)
                    <= SQUAD.max_players_per_club
                    + (SQUAD.squad_size - SQUAD.max_players_per_club) * no_transfer
                )

    bank_after = {
        deadline: pulp.LpVariable(
            f"bank_after_d{deadline}",
            lowBound=0,
            cat="Integer",
        )
        for deadline in all_deadlines
    }
    model += bank_after[1] == pulp.lpSum(
        candidate.plan.bank_after_tenths * selected[candidate_index]
        for candidate_index, candidate in enumerate(candidates)
    )
    initial_selling_prices = {
        player_id: int(frame_by_id.at[player_id, "selling_price"])
        for player_id in initial_ids
    }
    for deadline in provisional_deadlines:
        sale_proceeds = pulp.lpSum(
            int(frame.at[index, "now_cost"]) * transfer_out[(deadline, index)]
            + (
                initial_selling_prices[int(frame.at[index, "id"])]
                - int(frame.at[index, "now_cost"])
            )
            * original_sale[(deadline, index)]
            for index in indices
            if int(frame.at[index, "id"]) in initial_ids
        ) + pulp.lpSum(
            int(frame.at[index, "now_cost"]) * transfer_out[(deadline, index)]
            for index in indices
            if int(frame.at[index, "id"]) not in initial_ids
        )
        purchase_cost = pulp.lpSum(
            int(frame.at[index, "now_cost"]) * transfer_in[(deadline, index)]
            for index in indices
        )
        model += bank_after[deadline] == bank_after[deadline - 1] + sale_proceeds - purchase_cost

    start = {
        (gameweek, index): pulp.LpVariable(
            f"start_gw{gameweek}_{index}",
            cat="Binary",
        )
        for gameweek in future_gameweeks
        for index in indices
    }
    captain = {
        (gameweek, index): pulp.LpVariable(
            f"captain_gw{gameweek}_{index}",
            cat="Binary",
        )
        for gameweek in future_gameweeks
        for index in indices
    }
    goalkeepers = [index for index in indices if str(frame.at[index, "pos"]) == "GKP"]
    defenders = [index for index in indices if str(frame.at[index, "pos"]) == "DEF"]
    midfielders = [index for index in indices if str(frame.at[index, "pos"]) == "MID"]
    forwards = [index for index in indices if str(frame.at[index, "pos"]) == "FWD"]
    for gameweek in future_gameweeks:
        governing_deadline = min(gameweek, MODELLED_DEADLINES)
        model += pulp.lpSum(start[(gameweek, index)] for index in indices) == SQUAD.starting_size
        model += pulp.lpSum(captain[(gameweek, index)] for index in indices) == 1
        for index in indices:
            model += start[(gameweek, index)] <= squad[(governing_deadline, index)]
            model += captain[(gameweek, index)] <= start[(gameweek, index)]
        model += pulp.lpSum(start[(gameweek, index)] for index in goalkeepers) == 1
        model += pulp.lpSum(start[(gameweek, index)] for index in defenders) >= 3
        model += pulp.lpSum(start[(gameweek, index)] for index in defenders) <= 5
        model += pulp.lpSum(start[(gameweek, index)] for index in midfielders) >= 2
        model += pulp.lpSum(start[(gameweek, index)] for index in midfielders) <= 5
        model += pulp.lpSum(start[(gameweek, index)] for index in forwards) >= 1
        model += pulp.lpSum(start[(gameweek, index)] for index in forwards) <= 3

    average_outfield_weight = sum(reserve_weights[:3]) / 3.0
    start_score_milli: dict[tuple[int, int], int] = {}
    bench_score_milli: dict[tuple[int, int], int] = {}
    for gameweek in future_gameweeks:
        ep_column = forecast_columns[gameweek - 1]
        no_show_column = no_show_columns[gameweek - 1]
        weight = weights[gameweek - 1]
        for index in indices:
            start_score_milli[(gameweek, index)] = int(
                round(float(frame.at[index, ep_column]) * weight * 1000)
            )
            reserve_weight = (
                reserve_weights[3]
                if str(frame.at[index, "pos"]) == "GKP"
                else average_outfield_weight
            )
            bench_score_milli[(gameweek, index)] = int(
                round(
                    float(frame.at[index, ep_column])
                    * (1.0 - float(frame.at[index, no_show_column]))
                    * reserve_weight
                    * weight
                    * 1000
                )
            )

    first_points_milli = pulp.lpSum(
        int(round(candidate.first_gameweek_points * weights[0] * 1000))
        * selected[candidate_index]
        for candidate_index, candidate in enumerate(candidates)
    )
    future_points_milli = pulp.lpSum(
        (start[(gameweek, index)] + captain[(gameweek, index)])
        * start_score_milli[(gameweek, index)]
        + (
            squad[(min(gameweek, MODELLED_DEADLINES), index)]
            - start[(gameweek, index)]
        )
        * bench_score_milli[(gameweek, index)]
        for gameweek in future_gameweeks
        for index in indices
    )
    first_hit_milli = pulp.lpSum(
        int(round(candidate.plan.hit_points * weights[0] * 1000))
        * selected[candidate_index]
        for candidate_index, candidate in enumerate(candidates)
    )
    future_hit_milli = pulp.lpSum(
        int(
            round(
                transfer_points_cost(transfer_count, ft_before)
                * weights[deadline - 1]
                * 1000
            )
        )
        * count_choice[(deadline, ft_before, transfer_count)]
        for deadline in provisional_deadlines
        for ft_before in range(1, TRANSFERS.max_banked_free_transfers + 1)
        for transfer_count in range(MAX_PROVISIONAL_TRANSFERS + 1)
    )
    unmodelled_rolls = horizon - MODELLED_DEADLINES
    terminal_ft_value_milli = pulp.lpSum(
        int(
            round(
                max(
                    0,
                    min(TRANSFERS.max_banked_free_transfers, ft_next + unmodelled_rolls)
                    - 1,
                )
                * roll_value
                * 1000
            )
        )
        * count_choice[(MODELLED_DEADLINES, ft_before, transfer_count)]
        for ft_before in range(1, TRANSFERS.max_banked_free_transfers + 1)
        for transfer_count in range(MAX_PROVISIONAL_TRANSFERS + 1)
        for ft_next in [free_transfers_next_gameweek(ft_before, transfer_count)]
    )
    primary = (
        first_points_milli
        + future_points_milli
        - first_hit_milli
        - future_hit_milli
        + terminal_ft_value_milli
    )

    player_count = len(indices)
    tie = (
        pulp.lpSum(
            (len(candidates) - candidate_index) * selected[candidate_index]
            for candidate_index in candidate_range
        )
        + pulp.lpSum(
            MAX_PROVISIONAL_TRANSFERS - transfer_count_expression[deadline]
            for deadline in provisional_deadlines
        )
        + pulp.lpSum(
            (player_count - index)
            * (
                pulp.lpSum(squad[(deadline, index)] for deadline in all_deadlines)
                + pulp.lpSum(
                    start[(gameweek, index)] + 2 * captain[(gameweek, index)]
                    for gameweek in future_gameweeks
                )
            )
            for index in indices
        )
    )
    tie_upper_bound = max(
        1,
        len(candidates)
        + MODELLED_DEADLINES * MAX_PROVISIONAL_TRANSFERS
        + player_count * player_count * (MODELLED_DEADLINES + 3 * horizon),
    )
    model += primary * (tie_upper_bound + 1) + tie

    selected_solver = solver or pulp.PULP_CBC_CMD(
        msg=False,
        threads=1,
        timeLimit=budget_seconds,
        options=["randomSeed 17", "randomCbcSeed 17"],
    )
    original_time_limit = getattr(selected_solver, "timeLimit", None)
    if solver is not None:
        bounded_time_limit = budget_seconds
        if isinstance(original_time_limit, (int, float)) and original_time_limit > 0:
            bounded_time_limit = min(float(original_time_limit), budget_seconds)
        selected_solver.timeLimit = bounded_time_limit
    try:
        model.solve(selected_solver)
    except pulp.PulpSolverError as exc:
        raise StrategyPlannerError(f"the strategy solver could not run: {exc}") from exc
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
        raise StrategyPlannerError(
            "the bounded four-deadline model could not prove an optimal result "
            f"within {budget_seconds:g} seconds (status: {status})"
        )

    selected_candidate_index = next(
        candidate_index
        for candidate_index in candidate_range
        if pulp.value(selected[candidate_index]) > 0.5
    )
    chosen_candidate = candidates[selected_candidate_index]
    purchase_prices = {
        player_id: (
            int(frame_by_id.at[player_id, "now_cost"])
            if player_id not in chosen_candidate.uninterrupted_initial_ids
            else int(frame_by_id.at[player_id, "purchase_price"])
        )
        for player_id in chosen_candidate.squad_ids
    }

    steps: list[StrategyTransferStep] = []
    first_weighted_points = round(
        chosen_candidate.first_gameweek_points * weights[0],
        3,
    )
    first_step = StrategyTransferStep(
        deadline_offset=1,
        provisional=False,
        transfers=chosen_candidate.plan.transfers,
        squad_ids=chosen_candidate.plan.squad_ids,
        gameweeks=(chosen_candidate.plan.gameweeks[0],),
        bank_before_tenths=chosen_candidate.plan.bank_before_tenths,
        bank_after_tenths=chosen_candidate.plan.bank_after_tenths,
        free_transfers_before=chosen_candidate.plan.free_transfers_before,
        free_transfers_next_gameweek=chosen_candidate.plan.free_transfers_next_gameweek,
        hit_points=chosen_candidate.plan.hit_points,
        weighted_projected_points=first_weighted_points,
        weighted_hit_cost_points=round(
            chosen_candidate.plan.hit_points * weights[0],
            3,
        ),
    )
    steps.append(first_step)

    previous_bank = first_step.bank_after_tenths
    previous_ft = first_step.free_transfers_next_gameweek
    for deadline in provisional_deadlines:
        current_squad_indices = [
            index for index in indices if pulp.value(squad[(deadline, index)]) > 0.5
        ]
        outgoing_indices = [
            index for index in indices if pulp.value(transfer_out[(deadline, index)]) > 0.5
        ]
        incoming_indices = [
            index for index in indices if pulp.value(transfer_in[(deadline, index)]) > 0.5
        ]
        moves = _build_moves(
            frame,
            purchase_prices,
            outgoing_indices,
            incoming_indices,
        )
        transfer_count = len(moves)
        bank_now = (
            previous_bank
            + sum(move.out_selling_price_tenths for move in moves)
            - sum(move.in_price_tenths for move in moves)
        )
        ft_next = free_transfers_next_gameweek(previous_ft, transfer_count)
        hit_points = transfer_points_cost(transfer_count, previous_ft)
        for move in moves:
            purchase_prices.pop(move.out_id)
            purchase_prices[move.in_id] = move.in_price_tenths

        governed_gameweeks = (
            (deadline,)
            if deadline < MODELLED_DEADLINES
            else tuple(range(MODELLED_DEADLINES, horizon + 1))
        )
        gameweek_plans: list[TransferGameweekPlan] = []
        weighted_points = 0.0
        squad_ids = frozenset(
            int(frame.at[index, "id"]) for index in current_squad_indices
        )
        for gameweek in governed_gameweeks:
            ep_column = forecast_columns[gameweek - 1]
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
                    immediate.POSITION_ORDER[str(frame.at[index, "pos"])],
                    -float(frame.at[index, ep_column]),
                    int(frame.at[index, "id"]),
                )
            )
            starter_ids = tuple(int(frame.at[index, "id"]) for index in starters)
            captain_id = int(frame.at[captain_index, "id"])
            counts = Counter(str(frame.at[index, "pos"]) for index in starters)
            formation = f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"
            provisional_gameweek = TransferGameweekPlan(
                gameweek=gameweek,
                starting_ids=starter_ids,
                captain_id=captain_id,
                formation=formation,
                projected_points=0.0,
            )
            points = _lineup_points(
                frame_by_id,
                squad_ids,
                provisional_gameweek,
                offset=gameweek,
                bench_weights=reserve_weights,
                validate_reported=False,
            )
            weighted_points += points * weights[gameweek - 1]
            gameweek_plans.append(
                TransferGameweekPlan(
                    gameweek=gameweek,
                    starting_ids=starter_ids,
                    captain_id=captain_id,
                    formation=formation,
                    projected_points=round(points, 3),
                )
            )

        step = StrategyTransferStep(
            deadline_offset=deadline,
            provisional=True,
            transfers=moves,
            squad_ids=tuple(sorted(squad_ids)),
            gameweeks=tuple(gameweek_plans),
            bank_before_tenths=previous_bank,
            bank_after_tenths=bank_now,
            free_transfers_before=previous_ft,
            free_transfers_next_gameweek=ft_next,
            hit_points=hit_points,
            weighted_projected_points=round(weighted_points, 3),
            weighted_hit_cost_points=round(
                hit_points * weights[deadline - 1],
                3,
            ),
        )
        steps.append(step)
        previous_bank = bank_now
        previous_ft = ft_next

    if len(steps) != MODELLED_DEADLINES:
        raise StrategyPlannerError("the optimal roadmap did not produce four chained steps")
    weighted_projected_points = round(
        sum(step.weighted_projected_points for step in steps),
        3,
    )
    weighted_hit_cost_points = round(
        sum(step.weighted_hit_cost_points for step in steps),
        3,
    )
    terminal_ft = min(
        TRANSFERS.max_banked_free_transfers,
        previous_ft + horizon - MODELLED_DEADLINES,
    )
    terminal_ft_value = round(max(0, terminal_ft - 1) * roll_value, 3)
    roadmap = FourDeadlineRoadmap(
        first_step_candidate_index=chosen_candidate.index,
        steps=(steps[0], steps[1], steps[2], steps[3]),
        weighted_projected_points=weighted_projected_points,
        total_hit_points=sum(step.hit_points for step in steps),
        weighted_hit_cost_points=weighted_hit_cost_points,
        terminal_banked_ft_value_points=terminal_ft_value,
        decision_value_points=round(
            weighted_projected_points
            - weighted_hit_cost_points
            + terminal_ft_value,
            3,
        ),
    )
    return BoundedStrategyResult(
        best_roadmap=roadmap,
        horizon=horizon,
        gw_weights=weights,
        forecast_columns=forecast_columns,
        first_step_candidate_count=len(candidates),
    )


__all__ = [
    "ASSUMPTIONS",
    "BoundedStrategyResult",
    "DEFAULT_STRATEGY_GW_WEIGHTS",
    "DEFAULT_STRATEGY_HORIZON",
    "DEFAULT_STRATEGY_SOLVER_BUDGET_SECONDS",
    "FIRST_STEP_SEARCH",
    "FUTURE_PRICE_ASSUMPTION",
    "FourDeadlineRoadmap",
    "MAX_FIRST_STEP_CANDIDATES",
    "MAX_PROVISIONAL_TRANSFERS",
    "MAX_STRATEGY_HORIZON",
    "MIN_STRATEGY_HORIZON",
    "MODELLED_DEADLINES",
    "SEARCH_SCOPE",
    "StrategyPlannerError",
    "StrategyTransferStep",
    "optimize_strategy_roadmap",
]
