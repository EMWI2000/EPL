"""Bounded two-deadline transfer planning for an existing FPL squad.

The first deadline is deliberately restricted to explicit ``TransferPlan``
candidates produced by the existing immediate optimiser.  One MILP selects one
of those candidates and optimises a single provisional transfer action at the
next deadline, after which that second squad is evaluated over the remaining
forecast horizon.

Future prices are not forecast.  Every player therefore keeps today's market
price throughout the model.  A player bought in the first step has that market
price as the purchase and selling basis in the provisional second step.  The
result is solver-optimal only inside the supplied first-step set, the supplied
player pool and the documented immediate-transfer cap; it is not a global FPL
optimum.
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
    transfer_points_cost,
)
from . import rolling_transfers as immediate
from .rolling_transfers import TransferGameweekPlan, TransferMove, TransferPlan


DEFAULT_SEQUENTIAL_SOLVER_BUDGET_SECONDS = 15.0
MAX_FIRST_STEP_CANDIDATES = 16
FUTURE_PRICE_ASSUMPTION = "fixed_current_prices"


class SequentialTransferError(ValueError):
    """Raised for invalid inputs or an unavailable bounded sequence result."""


@dataclass(frozen=True)
class SequentialTransferStep:
    """One deadline action and the gameweeks governed by its resulting squad."""

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
class TwoDeadlineSequencePlan:
    """The chosen executable step plus one provisional next-deadline step."""

    first_step_candidate_index: int
    steps: tuple[SequentialTransferStep, SequentialTransferStep]
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
class BoundedSequentialTransferResult:
    """A proven result for a bounded first-step set, not a global optimum."""

    best_sequence: TwoDeadlineSequencePlan
    horizon: int
    gw_weights: tuple[float, ...]
    forecast_columns: tuple[str, ...]
    first_step_candidate_count: int
    first_step_search: str = "explicit_bounded_transfer_plans"
    future_price_assumption: str = FUTURE_PRICE_ASSUMPTION
    solver_proven_optimal_within_bounds: bool = True
    globally_optimal: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "best_sequence": self.best_sequence.as_dict(),
            "horizon": self.horizon,
            "gw_weights": list(self.gw_weights),
            "forecast_columns": list(self.forecast_columns),
            "first_step_candidate_count": self.first_step_candidate_count,
            "first_step_search": self.first_step_search,
            "future_price_assumption": self.future_price_assumption,
            "solver_proven_optimal_within_bounds": (
                self.solver_proven_optimal_within_bounds
            ),
            "globally_optimal": self.globally_optimal,
        }


@dataclass(frozen=True)
class _PreparedCandidate:
    index: int
    plan: TransferPlan
    prior_ids: frozenset[int]
    purchase_prices: dict[int, int]
    selling_prices: dict[int, int]
    first_gameweek_points: float


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise SequentialTransferError(f"{name} must be an integer >= {minimum}")
    return value


def _initial_ids_from_plan(plan: TransferPlan) -> frozenset[int]:
    squad_ids = tuple(int(player_id) for player_id in plan.squad_ids)
    if len(squad_ids) != SQUAD.squad_size or len(set(squad_ids)) != SQUAD.squad_size:
        raise SequentialTransferError(
            f"each first-step squad must contain {SQUAD.squad_size} unique ids"
        )
    out_ids = tuple(move.out_id for move in plan.transfers)
    in_ids = tuple(move.in_id for move in plan.transfers)
    if len(set(out_ids)) != len(out_ids) or len(set(in_ids)) != len(in_ids):
        raise SequentialTransferError("first-step transfers must not repeat players")
    if set(out_ids) & set(in_ids):
        raise SequentialTransferError("a first-step player cannot be both sold and bought")
    if not set(in_ids).issubset(squad_ids) or set(out_ids) & set(squad_ids):
        raise SequentialTransferError(
            "first-step transfers do not reconcile with the resulting squad"
        )
    initial = (set(squad_ids) - set(in_ids)) | set(out_ids)
    if len(initial) != SQUAD.squad_size:
        raise SequentialTransferError(
            "first-step transfers do not reconstruct one 15-player initial squad"
        )
    return frozenset(initial)


def _validate_squad(
    frame_by_id: pd.DataFrame,
    squad_ids: frozenset[int],
    *,
    transfers_made: int,
    path: str,
) -> None:
    if len(squad_ids) != SQUAD.squad_size:
        raise SequentialTransferError(f"{path} must contain 15 players")
    try:
        squad = frame_by_id.loc[list(squad_ids)]
    except KeyError as exc:
        raise SequentialTransferError(f"{path} contains a player outside the pool") from exc
    counts = Counter(str(value) for value in squad["pos"])
    if any(
        counts.get(position, 0) != quota
        for position, quota in immediate.SQUAD_QUOTA.items()
    ):
        raise SequentialTransferError(f"{path} violates the FPL position quotas")
    if transfers_made > 0 and int(squad["team_id"].value_counts().max()) > SQUAD.max_players_per_club:
        raise SequentialTransferError(
            f"{path} must restore the maximum players-per-club rule after a transfer"
        )


def _gameweek_points(
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
        raise SequentialTransferError("each first-step lineup must contain 11 unique ids")
    if not set(starters).issubset(squad_ids):
        raise SequentialTransferError("a lineup contains a player outside its squad")
    if int(gameweek.captain_id) not in starters:
        raise SequentialTransferError("the captain must be in the starting lineup")

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
        raise SequentialTransferError("a first-step lineup is not a legal FPL formation")

    ep_column = f"ep_gw{offset}"
    no_show_column = f"no_show_prob_gw{offset}"
    points = sum(float(frame_by_id.at[player_id, ep_column]) for player_id in starters)
    points += float(frame_by_id.at[int(gameweek.captain_id), ep_column])
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
        raise SequentialTransferError(
            "a first-step gameweek's projected_points do not match its lineup"
        )
    return points


def _prepare_candidates(
    players: pd.DataFrame,
    first_step_candidates: Sequence[TransferPlan],
    *,
    horizon: int,
    bench_weights: tuple[float, ...],
) -> tuple[
    pd.DataFrame,
    tuple[str, ...],
    tuple[str, ...],
    tuple[_PreparedCandidate, ...],
]:
    if not isinstance(first_step_candidates, Sequence) or isinstance(
        first_step_candidates, (str, bytes)
    ):
        raise SequentialTransferError("first_step_candidates must be a sequence")
    candidates = tuple(first_step_candidates)
    if not candidates:
        raise SequentialTransferError("at least one first-step candidate is required")
    if len(candidates) > MAX_FIRST_STEP_CANDIDATES:
        raise SequentialTransferError(
            f"at most {MAX_FIRST_STEP_CANDIDATES} first-step candidates are allowed"
        )
    if any(not isinstance(plan, TransferPlan) for plan in candidates):
        raise SequentialTransferError(
            "first_step_candidates must contain existing TransferPlan values"
        )

    initial_ids = _initial_ids_from_plan(candidates[0])
    try:
        frame, forecast_columns, no_show_columns, _ = immediate._prepare_players(
            players,
            tuple(sorted(initial_ids)),
            horizon,
        )
    except immediate.RollingTransferError as exc:
        raise SequentialTransferError(str(exc)) from exc
    frame_by_id = frame.set_index("id", drop=False)

    shared_bank = _integer(candidates[0].bank_before_tenths, "bank_before_tenths")
    shared_ft = _integer(
        candidates[0].free_transfers_before,
        "free_transfers_before",
    )
    if shared_ft > TRANSFERS.max_banked_free_transfers:
        raise SequentialTransferError("free_transfers_before is outside the legal range")

    prepared: list[_PreparedCandidate] = []
    for candidate_index, plan in enumerate(candidates):
        path = f"first_step_candidates[{candidate_index}]"
        if _initial_ids_from_plan(plan) != initial_ids:
            raise SequentialTransferError(
                "all first-step candidates must start from the same confirmed squad"
            )
        if plan.bank_before_tenths != shared_bank:
            raise SequentialTransferError(
                "all first-step candidates must start with the same bank"
            )
        if plan.free_transfers_before != shared_ft:
            raise SequentialTransferError(
                "all first-step candidates must start with the same free-transfer bank"
            )

        transfer_count = len(plan.transfers)
        if transfer_count > max(2, shared_ft):
            raise SequentialTransferError(
                f"{path} exceeds the immediate optimiser's transfer-count bound"
            )
        expected_action = "roll" if transfer_count == 0 else f"{transfer_count}_transfer"
        if plan.action != expected_action:
            raise SequentialTransferError(f"{path}.action must be {expected_action}")

        out_ids: set[int] = set()
        in_ids: set[int] = set()
        sale_total = 0
        buy_total = 0
        for move in plan.transfers:
            if move.out_id in out_ids or move.in_id in in_ids:
                raise SequentialTransferError(f"{path} repeats a transferred player")
            out_ids.add(move.out_id)
            in_ids.add(move.in_id)
            if move.out_id not in initial_ids or move.in_id in initial_ids:
                raise SequentialTransferError(
                    f"{path} must replace an initially owned player with an unowned player"
                )
            if move.out_id not in frame_by_id.index or move.in_id not in frame_by_id.index:
                raise SequentialTransferError(f"{path} references a player outside the pool")
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
                raise SequentialTransferError(
                    f"{path} contains inconsistent position or price fields"
                )
            sale_total += move.out_selling_price_tenths
            buy_total += move.in_price_tenths

        expected_squad = (set(initial_ids) - out_ids) | in_ids
        prior_ids = frozenset(int(player_id) for player_id in plan.squad_ids)
        if prior_ids != expected_squad:
            raise SequentialTransferError(
                f"{path}.squad_ids do not match its transfers"
            )
        _validate_squad(
            frame_by_id,
            prior_ids,
            transfers_made=transfer_count,
            path=f"{path}.squad_ids",
        )

        expected_bank_after = shared_bank + sale_total - buy_total
        if expected_bank_after < 0 or plan.bank_after_tenths != expected_bank_after:
            raise SequentialTransferError(
                f"{path}.bank_after_tenths does not reconcile with its transfers"
            )
        expected_ft_next = free_transfers_next_gameweek(shared_ft, transfer_count)
        if plan.free_transfers_next_gameweek != expected_ft_next:
            raise SequentialTransferError(
                f"{path}.free_transfers_next_gameweek is inconsistent"
            )
        expected_hit = transfer_points_cost(transfer_count, shared_ft)
        if plan.hit_points != expected_hit:
            raise SequentialTransferError(f"{path}.hit_points is inconsistent")

        if len(plan.gameweeks) != horizon:
            raise SequentialTransferError(
                f"{path}.gameweeks must match the requested horizon"
            )
        first_gameweek_points = 0.0
        for offset, gameweek in enumerate(plan.gameweeks, start=1):
            if gameweek.gameweek != offset:
                raise SequentialTransferError(
                    f"{path}.gameweeks must use consecutive relative offsets"
                )
            points = _gameweek_points(
                frame_by_id,
                prior_ids,
                gameweek,
                offset=offset,
                bench_weights=bench_weights,
                validate_reported=True,
            )
            if offset == 1:
                first_gameweek_points = points

        first_in_ids = {move.in_id for move in plan.transfers}
        purchase_prices: dict[int, int] = {}
        selling_prices: dict[int, int] = {}
        for player_id in prior_ids:
            if player_id in first_in_ids:
                price = int(frame_by_id.at[player_id, "now_cost"])
                purchase_prices[player_id] = price
                selling_prices[player_id] = price
            else:
                purchase_prices[player_id] = int(
                    frame_by_id.at[player_id, "purchase_price"]
                )
                selling_prices[player_id] = int(
                    frame_by_id.at[player_id, "selling_price"]
                )
        prepared.append(
            _PreparedCandidate(
                index=candidate_index,
                plan=plan,
                prior_ids=prior_ids,
                purchase_prices=purchase_prices,
                selling_prices=selling_prices,
                first_gameweek_points=first_gameweek_points,
            )
        )

    return frame, forecast_columns, no_show_columns, tuple(prepared)


def _build_future_moves(
    frame: pd.DataFrame,
    candidate: _PreparedCandidate,
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
    for position in sorted(
        immediate.SQUAD_QUOTA,
        key=immediate.POSITION_ORDER.__getitem__,
    ):
        outgoing = sorted(
            outgoing_by_position[position],
            key=lambda index: int(frame.at[index, "id"]),
        )
        incoming = sorted(
            incoming_by_position[position],
            key=lambda index: int(frame.at[index, "id"]),
        )
        if len(outgoing) != len(incoming):
            raise SequentialTransferError(
                "the provisional transfers do not preserve position quotas"
            )
        for out_index, in_index in zip(outgoing, incoming):
            out_id = int(frame.at[out_index, "id"])
            moves.append(
                TransferMove(
                    out_id=out_id,
                    in_id=int(frame.at[in_index, "id"]),
                    position=position,
                    out_purchase_price_tenths=candidate.purchase_prices[out_id],
                    out_current_price_tenths=int(frame.at[out_index, "now_cost"]),
                    out_selling_price_tenths=candidate.selling_prices[out_id],
                    in_price_tenths=int(frame.at[in_index, "now_cost"]),
                )
            )
    return tuple(moves)


def optimize_two_deadline_sequence(
    players: pd.DataFrame,
    first_step_candidates: Sequence[TransferPlan],
    *,
    horizon: int,
    gw_weights: Optional[Sequence[float]] = None,
    roll_ft_value_points: float = 1.0,
    bench_weights: Sequence[float] = immediate.DEFAULT_BENCH_WEIGHTS,
    eligible_transfer_in_ids: Optional[Collection[int]] = None,
    solver_budget_seconds: float = DEFAULT_SEQUENTIAL_SOLVER_BUDGET_SECONDS,
    solver: Optional[pulp.LpSolver] = None,
) -> BoundedSequentialTransferResult:
    """Select a bounded first step and optimise one provisional second step.

    The first step can only be one of ``first_step_candidates``.  The second
    step is jointly optimised for the next deadline, and its resulting squad is
    evaluated over gameweeks 2 through ``horizon``.  All prices stay fixed at
    their supplied current values.  No chips or later transfer steps are used.
    """

    if not isinstance(horizon, int) or isinstance(horizon, bool) or not 2 <= horizon <= 5:
        raise SequentialTransferError("horizon must be an integer from 2 to 5")
    try:
        weights = immediate._numeric_weights(
            gw_weights if gw_weights is not None else immediate.DEFAULT_GW_WEIGHTS[:horizon],
            horizon,
        )
        reserve_weights = immediate._numeric_bench_weights(bench_weights)
    except immediate.RollingTransferError as exc:
        raise SequentialTransferError(str(exc)) from exc
    roll_value = float(roll_ft_value_points)
    if not isfinite(roll_value) or roll_value < 0:
        raise SequentialTransferError(
            "roll_ft_value_points must be finite and non-negative"
        )
    budget_seconds = float(solver_budget_seconds)
    if not isfinite(budget_seconds) or budget_seconds <= 0:
        raise SequentialTransferError(
            "solver_budget_seconds must be finite and positive"
        )

    frame, forecast_columns, no_show_columns, candidates = _prepare_candidates(
        players,
        first_step_candidates,
        horizon=horizon,
        bench_weights=reserve_weights,
    )
    indices = list(frame.index)
    pool_ids = frozenset(int(value) for value in frame["id"])
    if eligible_transfer_in_ids is None:
        eligible_in_ids = pool_ids
    else:
        if isinstance(eligible_transfer_in_ids, (str, bytes)):
            raise SequentialTransferError(
                "eligible_transfer_in_ids must be a collection of player ids"
            )
        supplied_ids = tuple(eligible_transfer_in_ids)
        if any(
            not isinstance(player_id, int)
            or isinstance(player_id, bool)
            or player_id < 1
            for player_id in supplied_ids
        ):
            raise SequentialTransferError(
                "eligible_transfer_in_ids must contain positive integer ids"
            )
        eligible_in_ids = frozenset(supplied_ids)
        if not eligible_in_ids.issubset(pool_ids):
            raise SequentialTransferError(
                "eligible_transfer_in_ids contains a player outside the pool"
            )
    future_offsets = range(1, horizon)
    candidate_indices = range(len(candidates))

    model = pulp.LpProblem("bounded_two_deadline_transfer_sequence", pulp.LpMaximize)
    selected = {
        candidate_index: pulp.LpVariable(
            f"select_first_{candidate_index}", cat="Binary"
        )
        for candidate_index in candidate_indices
    }
    model += pulp.lpSum(selected.values()) == 1

    squad = {index: pulp.LpVariable(f"future_squad_{index}", cat="Binary") for index in indices}
    transfer_in = {
        index: pulp.LpVariable(f"future_in_{index}", cat="Binary") for index in indices
    }
    transfer_out = {
        index: pulp.LpVariable(f"future_out_{index}", cat="Binary") for index in indices
    }
    for index in indices:
        player_id = int(frame.at[index, "id"])
        prior_owned = pulp.lpSum(
            selected[candidate_index]
            for candidate_index, candidate in enumerate(candidates)
            if player_id in candidate.prior_ids
        )
        model += squad[index] == prior_owned + transfer_in[index] - transfer_out[index]
        model += transfer_in[index] <= 1 - prior_owned
        model += transfer_out[index] <= prior_owned
        if player_id not in eligible_in_ids:
            model += transfer_in[index] == 0

    count_choice: dict[tuple[int, int], pulp.LpVariable] = {}
    for candidate_index, candidate in enumerate(candidates):
        maximum = max(2, candidate.plan.free_transfers_next_gameweek)
        choices = []
        for transfer_count in range(maximum + 1):
            choice = pulp.LpVariable(
                f"future_count_{candidate_index}_{transfer_count}", cat="Binary"
            )
            count_choice[(candidate_index, transfer_count)] = choice
            choices.append(choice)
        model += pulp.lpSum(choices) == selected[candidate_index]

    transfer_count_expression = pulp.lpSum(
        transfer_count * choice
        for (candidate_index, transfer_count), choice in count_choice.items()
    )
    model += pulp.lpSum(transfer_in.values()) == transfer_count_expression
    model += pulp.lpSum(transfer_out.values()) == transfer_count_expression

    model += pulp.lpSum(squad.values()) == SQUAD.squad_size
    for position, quota in immediate.SQUAD_QUOTA.items():
        members = [index for index in indices if str(frame.at[index, "pos"]) == position]
        model += pulp.lpSum(squad[index] for index in members) == quota
    no_second_transfer = pulp.lpSum(
        count_choice[(candidate_index, 0)] for candidate_index in candidate_indices
    )
    for _, members in frame.groupby("team_id").groups.items():
        model += (
            pulp.lpSum(squad[int(index)] for index in members)
            <= SQUAD.max_players_per_club
            + (SQUAD.squad_size - SQUAD.max_players_per_club) * no_second_transfer
        )

    sale_gate: dict[tuple[int, int], pulp.LpVariable] = {}
    sale_proceeds_terms = []
    for candidate_index, candidate in enumerate(candidates):
        for index in indices:
            player_id = int(frame.at[index, "id"])
            if player_id not in candidate.prior_ids:
                continue
            gate = pulp.LpVariable(
                f"future_sale_{candidate_index}_{index}", cat="Binary"
            )
            sale_gate[(candidate_index, index)] = gate
            model += gate <= transfer_out[index]
            model += gate <= selected[candidate_index]
            model += gate >= transfer_out[index] + selected[candidate_index] - 1
            sale_proceeds_terms.append(candidate.selling_prices[player_id] * gate)
    sale_proceeds = pulp.lpSum(sale_proceeds_terms)
    purchase_cost = pulp.lpSum(
        int(frame.at[index, "now_cost"]) * transfer_in[index] for index in indices
    )
    bank_before_expression = pulp.lpSum(
        candidate.plan.bank_after_tenths * selected[candidate_index]
        for candidate_index, candidate in enumerate(candidates)
    )
    bank_after_expression = bank_before_expression + sale_proceeds - purchase_cost
    model += bank_after_expression >= 0

    start = {
        (offset, index): pulp.LpVariable(
            f"future_start_{offset + 1}_{index}", cat="Binary"
        )
        for offset in future_offsets
        for index in indices
    }
    captain = {
        (offset, index): pulp.LpVariable(
            f"future_captain_{offset + 1}_{index}", cat="Binary"
        )
        for offset in future_offsets
        for index in indices
    }
    goalkeepers = [index for index in indices if str(frame.at[index, "pos"]) == "GKP"]
    defenders = [index for index in indices if str(frame.at[index, "pos"]) == "DEF"]
    midfielders = [index for index in indices if str(frame.at[index, "pos"]) == "MID"]
    forwards = [index for index in indices if str(frame.at[index, "pos"]) == "FWD"]
    for offset in future_offsets:
        model += pulp.lpSum(start[(offset, index)] for index in indices) == SQUAD.starting_size
        model += pulp.lpSum(captain[(offset, index)] for index in indices) == 1
        for index in indices:
            model += start[(offset, index)] <= squad[index]
            model += captain[(offset, index)] <= start[(offset, index)]
        model += pulp.lpSum(start[(offset, index)] for index in goalkeepers) == 1
        model += pulp.lpSum(start[(offset, index)] for index in defenders) >= 3
        model += pulp.lpSum(start[(offset, index)] for index in defenders) <= 5
        model += pulp.lpSum(start[(offset, index)] for index in midfielders) >= 2
        model += pulp.lpSum(start[(offset, index)] for index in midfielders) <= 5
        model += pulp.lpSum(start[(offset, index)] for index in forwards) >= 1
        model += pulp.lpSum(start[(offset, index)] for index in forwards) <= 3

    average_outfield_weight = sum(reserve_weights[:3]) / 3.0
    start_score_milli: dict[tuple[int, int], int] = {}
    bench_score_milli: dict[tuple[int, int], int] = {}
    for offset in future_offsets:
        ep_column = forecast_columns[offset]
        no_show_column = no_show_columns[offset]
        for index in indices:
            start_score_milli[(offset, index)] = int(
                round(float(frame.at[index, ep_column]) * weights[offset] * 1000)
            )
            reserve_weight = (
                reserve_weights[3]
                if str(frame.at[index, "pos"]) == "GKP"
                else average_outfield_weight
            )
            bench_score_milli[(offset, index)] = int(
                round(
                    float(frame.at[index, ep_column])
                    * (1.0 - float(frame.at[index, no_show_column]))
                    * reserve_weight
                    * weights[offset]
                    * 1000
                )
            )

    first_points_milli = pulp.lpSum(
        int(round(candidate.first_gameweek_points * weights[0] * 1000))
        * selected[candidate_index]
        for candidate_index, candidate in enumerate(candidates)
    )
    future_points_milli = pulp.lpSum(
        (start[(offset, index)] + captain[(offset, index)])
        * start_score_milli[(offset, index)]
        + (squad[index] - start[(offset, index)])
        * bench_score_milli[(offset, index)]
        for offset in future_offsets
        for index in indices
    )
    first_hit_milli = pulp.lpSum(
        candidate.plan.hit_points * 1000 * selected[candidate_index]
        for candidate_index, candidate in enumerate(candidates)
    )
    future_hit_milli = pulp.lpSum(
        int(
            round(
                transfer_points_cost(
                    transfer_count,
                    candidates[candidate_index].plan.free_transfers_next_gameweek,
                )
                * weights[1]
                * 1000
            )
        )
        * choice
        for (candidate_index, transfer_count), choice in count_choice.items()
    )
    terminal_ft_value_milli = pulp.lpSum(
        int(
            round(
                max(
                    0,
                    free_transfers_next_gameweek(
                        candidates[candidate_index].plan.free_transfers_next_gameweek,
                        transfer_count,
                    )
                    - TRANSFERS.free_transfers_per_gameweek,
                )
                * roll_value
                * 1000
            )
        )
        * choice
        for (candidate_index, transfer_count), choice in count_choice.items()
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
            for candidate_index in candidate_indices
        )
        + (TRANSFERS.max_banked_free_transfers - transfer_count_expression)
        + pulp.lpSum(
            (player_count - index)
            * (
                squad[index]
                + pulp.lpSum(
                    start[(offset, index)] + 2 * captain[(offset, index)]
                    for offset in future_offsets
                )
            )
            for index in indices
        )
    )
    tie_upper_bound = max(
        1,
        len(candidates)
        + TRANSFERS.max_banked_free_transfers
        + player_count * player_count * (1 + 3 * len(tuple(future_offsets))),
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
        raise SequentialTransferError(
            f"the sequential optimisation solver could not run: {exc}"
        ) from exc
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
        raise SequentialTransferError(
            "the bounded two-deadline model could not prove an optimal result "
            f"within {budget_seconds:g} seconds (status: {status})"
        )

    selected_candidate_index = next(
        candidate_index
        for candidate_index in candidate_indices
        if pulp.value(selected[candidate_index]) > 0.5
    )
    chosen_candidate = candidates[selected_candidate_index]
    future_squad_indices = [
        index for index in indices if pulp.value(squad[index]) > 0.5
    ]
    future_incoming_indices = [
        index for index in indices if pulp.value(transfer_in[index]) > 0.5
    ]
    future_outgoing_indices = [
        index for index in indices if pulp.value(transfer_out[index]) > 0.5
    ]
    future_moves = _build_future_moves(
        frame,
        chosen_candidate,
        future_outgoing_indices,
        future_incoming_indices,
    )
    future_transfer_count = len(future_moves)
    future_bank_before = int(chosen_candidate.plan.bank_after_tenths)
    future_bank_after = (
        future_bank_before
        + sum(move.out_selling_price_tenths for move in future_moves)
        - sum(move.in_price_tenths for move in future_moves)
    )
    future_ft_before = int(chosen_candidate.plan.free_transfers_next_gameweek)
    future_ft_next = free_transfers_next_gameweek(
        future_ft_before,
        future_transfer_count,
    )
    future_hit_points = transfer_points_cost(
        future_transfer_count,
        future_ft_before,
    )

    future_gameweeks: list[TransferGameweekPlan] = []
    future_weighted_points = 0.0
    for offset in future_offsets:
        ep_column = forecast_columns[offset]
        starters = [
            index
            for index in indices
            if pulp.value(start[(offset, index)]) > 0.5
        ]
        captain_index = next(
            index
            for index in indices
            if pulp.value(captain[(offset, index)]) > 0.5
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
        provisional = TransferGameweekPlan(
            gameweek=offset + 1,
            starting_ids=starter_ids,
            captain_id=captain_id,
            formation=formation,
            projected_points=0.0,
        )
        squad_ids = frozenset(
            int(frame.at[index, "id"]) for index in future_squad_indices
        )
        points = _gameweek_points(
            frame.set_index("id", drop=False),
            squad_ids,
            provisional,
            offset=offset + 1,
            bench_weights=reserve_weights,
            validate_reported=False,
        )
        future_weighted_points += points * weights[offset]
        future_gameweeks.append(
            TransferGameweekPlan(
                gameweek=offset + 1,
                starting_ids=starter_ids,
                captain_id=captain_id,
                formation=formation,
                projected_points=round(points, 3),
            )
        )

    first_gameweek = chosen_candidate.plan.gameweeks[0]
    first_step = SequentialTransferStep(
        deadline_offset=1,
        provisional=False,
        transfers=chosen_candidate.plan.transfers,
        squad_ids=chosen_candidate.plan.squad_ids,
        gameweeks=(first_gameweek,),
        bank_before_tenths=chosen_candidate.plan.bank_before_tenths,
        bank_after_tenths=chosen_candidate.plan.bank_after_tenths,
        free_transfers_before=chosen_candidate.plan.free_transfers_before,
        free_transfers_next_gameweek=(
            chosen_candidate.plan.free_transfers_next_gameweek
        ),
        hit_points=chosen_candidate.plan.hit_points,
        weighted_projected_points=round(
            chosen_candidate.first_gameweek_points * weights[0], 3
        ),
        weighted_hit_cost_points=float(chosen_candidate.plan.hit_points),
    )
    second_step = SequentialTransferStep(
        deadline_offset=2,
        provisional=True,
        transfers=future_moves,
        squad_ids=tuple(
            sorted(int(frame.at[index, "id"]) for index in future_squad_indices)
        ),
        gameweeks=tuple(future_gameweeks),
        bank_before_tenths=future_bank_before,
        bank_after_tenths=future_bank_after,
        free_transfers_before=future_ft_before,
        free_transfers_next_gameweek=future_ft_next,
        hit_points=future_hit_points,
        weighted_projected_points=round(future_weighted_points, 3),
        weighted_hit_cost_points=round(future_hit_points * weights[1], 3),
    )
    weighted_projected_points = (
        first_step.weighted_projected_points
        + second_step.weighted_projected_points
    )
    weighted_hit_cost_points = (
        first_step.weighted_hit_cost_points
        + second_step.weighted_hit_cost_points
    )
    terminal_ft_value = (
        max(
            0,
            future_ft_next - TRANSFERS.free_transfers_per_gameweek,
        )
        * roll_value
    )
    sequence = TwoDeadlineSequencePlan(
        first_step_candidate_index=chosen_candidate.index,
        steps=(first_step, second_step),
        weighted_projected_points=round(weighted_projected_points, 3),
        total_hit_points=first_step.hit_points + second_step.hit_points,
        weighted_hit_cost_points=round(weighted_hit_cost_points, 3),
        terminal_banked_ft_value_points=round(terminal_ft_value, 3),
        decision_value_points=round(
            weighted_projected_points
            - weighted_hit_cost_points
            + terminal_ft_value,
            3,
        ),
    )
    return BoundedSequentialTransferResult(
        best_sequence=sequence,
        horizon=horizon,
        gw_weights=weights,
        forecast_columns=forecast_columns,
        first_step_candidate_count=len(candidates),
    )


__all__ = [
    "BoundedSequentialTransferResult",
    "DEFAULT_SEQUENTIAL_SOLVER_BUDGET_SECONDS",
    "FUTURE_PRICE_ASSUMPTION",
    "MAX_FIRST_STEP_CANDIDATES",
    "SequentialTransferError",
    "SequentialTransferStep",
    "TwoDeadlineSequencePlan",
    "optimize_two_deadline_sequence",
]
