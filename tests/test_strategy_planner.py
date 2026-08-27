from __future__ import annotations

from collections import Counter
from itertools import combinations
import json
from time import monotonic

import pandas as pd
import pulp
import pytest

from fpl_app.domain.rules import free_transfers_next_gameweek, transfer_points_cost
from fpl_app.logic.rolling_transfers import (
    DEFAULT_BENCH_WEIGHTS,
    TransferGameweekPlan,
    TransferMove,
    TransferPlan,
)
from fpl_app.logic.strategy_planner import (
    ASSUMPTIONS,
    DEFAULT_STRATEGY_GW_WEIGHTS,
    FUTURE_PRICE_ASSUMPTION,
    MODELLED_DEADLINES,
    StrategyPlannerError,
    optimize_strategy_roadmap,
)


INITIAL_IDS = tuple(range(1, 16))


def _pool(horizon: int = 8) -> pd.DataFrame:
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    rows: list[dict[str, object]] = []
    for player_id, position in enumerate(positions, start=1):
        row: dict[str, object] = {
            "id": player_id,
            "team_id": ((player_id - 1) % 8) + 1,
            "pos": position,
            "now_cost": 50,
            "purchase_price": 50,
            "selling_price": 50,
        }
        for offset in range(1, horizon + 1):
            row[f"ep_gw{offset}"] = 5.0
            row[f"no_show_prob_gw{offset}"] = 0.0
        rows.append(row)

    for player_id, position, team_id in (
        (101, "DEF", 9),
        (102, "MID", 10),
        (103, "FWD", 11),
        (104, "MID", 12),
        (105, "DEF", 13),
    ):
        row = {
            "id": player_id,
            "team_id": team_id,
            "pos": position,
            "now_cost": 50,
            "purchase_price": pd.NA,
            "selling_price": pd.NA,
        }
        for offset in range(1, horizon + 1):
            row[f"ep_gw{offset}"] = 1.0
            row[f"no_show_prob_gw{offset}"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _best_lineup(
    frame: pd.DataFrame,
    squad_ids: tuple[int, ...],
    offset: int = 1,
) -> TransferGameweekPlan:
    by_id = frame.set_index("id")
    ids_by_position = {
        position: tuple(
            player_id
            for player_id in squad_ids
            if str(by_id.at[player_id, "pos"]) == position
        )
        for position in ("GKP", "DEF", "MID", "FWD")
    }
    ep_column = f"ep_gw{offset}"
    best: tuple[float, tuple[int, ...], int, str] | None = None
    for goalkeeper in ids_by_position["GKP"]:
        for defender_count in range(3, 6):
            for midfielder_count in range(2, 6):
                forward_count = 10 - defender_count - midfielder_count
                if not 1 <= forward_count <= 3:
                    continue
                for defenders in combinations(ids_by_position["DEF"], defender_count):
                    for midfielders in combinations(ids_by_position["MID"], midfielder_count):
                        for forwards in combinations(ids_by_position["FWD"], forward_count):
                            starters = tuple(
                                sorted((goalkeeper, *defenders, *midfielders, *forwards))
                            )
                            captain = min(
                                starters,
                                key=lambda player_id: (
                                    -float(by_id.at[player_id, ep_column]),
                                    player_id,
                                ),
                            )
                            points = sum(
                                float(by_id.at[player_id, ep_column])
                                for player_id in starters
                            )
                            points += float(by_id.at[captain, ep_column])
                            for player_id in set(squad_ids) - set(starters):
                                reserve_weight = (
                                    DEFAULT_BENCH_WEIGHTS[3]
                                    if str(by_id.at[player_id, "pos"]) == "GKP"
                                    else sum(DEFAULT_BENCH_WEIGHTS[:3]) / 3.0
                                )
                                points += float(by_id.at[player_id, ep_column]) * reserve_weight
                            candidate = (
                                points,
                                starters,
                                captain,
                                f"{defender_count}-{midfielder_count}-{forward_count}",
                            )
                            if best is None or (-candidate[0], candidate[1]) < (
                                -best[0],
                                best[1],
                            ):
                                best = candidate
    assert best is not None
    points, starters, captain, formation = best
    return TransferGameweekPlan(
        gameweek=offset,
        starting_ids=starters,
        captain_id=captain,
        formation=formation,
        projected_points=round(points, 3),
    )


def _first_plan(
    frame: pd.DataFrame,
    replacements: tuple[tuple[int, int], ...] = (),
    *,
    bank_tenths: int = 0,
    free_transfers: int = 1,
) -> TransferPlan:
    by_id = frame.set_index("id")
    moves: list[TransferMove] = []
    squad_ids = set(INITIAL_IDS)
    for out_id, in_id in replacements:
        position = str(by_id.at[out_id, "pos"])
        assert str(by_id.at[in_id, "pos"]) == position
        moves.append(
            TransferMove(
                out_id=out_id,
                in_id=in_id,
                position=position,
                out_purchase_price_tenths=int(by_id.at[out_id, "purchase_price"]),
                out_current_price_tenths=int(by_id.at[out_id, "now_cost"]),
                out_selling_price_tenths=int(by_id.at[out_id, "selling_price"]),
                in_price_tenths=int(by_id.at[in_id, "now_cost"]),
            )
        )
        squad_ids.remove(out_id)
        squad_ids.add(in_id)
    ordered_squad = tuple(sorted(squad_ids))
    first_gameweek = _best_lineup(frame, ordered_squad)
    transfer_count = len(moves)
    hit_points = transfer_points_cost(transfer_count, free_transfers)
    ft_next = free_transfers_next_gameweek(free_transfers, transfer_count)
    bank_after = (
        bank_tenths
        + sum(move.out_selling_price_tenths for move in moves)
        - sum(move.in_price_tenths for move in moves)
    )
    banked_value = max(0, ft_next - 1)
    return TransferPlan(
        action="roll" if not moves else f"{transfer_count}_transfer",
        transfers=tuple(moves),
        squad_ids=ordered_squad,
        gameweeks=(first_gameweek,),
        bank_before_tenths=bank_tenths,
        bank_after_tenths=bank_after,
        free_transfers_before=free_transfers,
        free_transfers_next_gameweek=ft_next,
        hit_points=hit_points,
        projected_points=first_gameweek.projected_points,
        banked_ft_value_points=float(banked_value),
        decision_value_points=round(
            first_gameweek.projected_points - hit_points + banked_value,
            3,
        ),
        delta_vs_base_points=0.0,
    )


def _assert_step_is_legal(frame: pd.DataFrame, step) -> None:
    by_id = frame.set_index("id")
    squad = by_id.loc[list(step.squad_ids)]
    assert len(squad) == 15
    assert Counter(squad["pos"]) == Counter(
        {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    )
    if step.transfer_count:
        assert int(squad["team_id"].value_counts().max()) <= 3
    for gameweek in step.gameweeks:
        starters = by_id.loc[list(gameweek.starting_ids)]
        counts = Counter(starters["pos"])
        assert len(gameweek.starting_ids) == 11
        assert set(gameweek.starting_ids).issubset(step.squad_ids)
        assert gameweek.captain_id in gameweek.starting_ids
        assert counts["GKP"] == 1
        assert 3 <= counts["DEF"] <= 5
        assert 2 <= counts["MID"] <= 5
        assert 1 <= counts["FWD"] <= 3
        assert gameweek.formation == f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"


def _assert_chained(result, frame: pd.DataFrame) -> None:
    steps = result.best_roadmap.steps
    assert len(steps) == MODELLED_DEADLINES
    assert [step.deadline_offset for step in steps] == [1, 2, 3, 4]
    assert [step.provisional for step in steps] == [False, True, True, True]
    assert [gameweek.gameweek for gameweek in steps[0].gameweeks] == [1]
    assert [gameweek.gameweek for gameweek in steps[1].gameweeks] == [2]
    assert [gameweek.gameweek for gameweek in steps[2].gameweeks] == [3]
    assert [gameweek.gameweek for gameweek in steps[3].gameweeks] == list(
        range(4, result.horizon + 1)
    )
    for index, step in enumerate(steps):
        _assert_step_is_legal(frame, step)
        if index == 0:
            continue
        previous = steps[index - 1]
        assert step.bank_before_tenths == previous.bank_after_tenths
        assert step.free_transfers_before == previous.free_transfers_next_gameweek
        expected = (set(previous.squad_ids) - {move.out_id for move in step.transfers}) | {
            move.in_id for move in step.transfers
        }
        assert set(step.squad_ids) == expected
        assert step.bank_after_tenths == (
            step.bank_before_tenths
            + sum(move.out_selling_price_tenths for move in step.transfers)
            - sum(move.in_price_tenths for move in step.transfers)
        )
        assert step.free_transfers_next_gameweek == free_transfers_next_gameweek(
            step.free_transfers_before,
            step.transfer_count,
        )
        assert step.hit_points == transfer_points_cost(
            step.transfer_count,
            step.free_transfers_before,
        )


def test_builds_a_legal_deterministic_four_deadline_roadmap() -> None:
    players = _pool()
    players.loc[players["id"] == 102, [f"ep_gw{gw}" for gw in range(2, 9)]] = 8.0
    players.loc[players["id"] == 103, [f"ep_gw{gw}" for gw in range(3, 9)]] = 7.0
    roll = _first_plan(players)

    result = optimize_strategy_roadmap(players, (roll,), horizon=8)
    shuffled = optimize_strategy_roadmap(
        players.sample(frac=1.0, random_state=17),
        (roll,),
        horizon=8,
    )

    assert result.as_dict() == shuffled.as_dict()
    assert result.best_roadmap.first_step_candidate_index == 0
    assert result.solver_proven_optimal_within_bounds is True
    assert result.globally_optimal is False
    assert result.future_price_assumption == FUTURE_PRICE_ASSUMPTION
    assert result.assumptions == ASSUMPTIONS
    assert result.gw_weights == pytest.approx(DEFAULT_STRATEGY_GW_WEIGHTS[:8])
    _assert_chained(result, players)
    assert result.best_roadmap.total_hit_points == sum(
        step.hit_points for step in result.best_roadmap.steps
    )
    assert result.best_roadmap.weighted_projected_points == pytest.approx(
        sum(step.weighted_projected_points for step in result.best_roadmap.steps),
        abs=0.002,
    )
    json.dumps(result.as_dict(), allow_nan=False)


def test_zero_free_transfers_is_legal_only_for_the_current_deadline() -> None:
    players = _pool()
    roll = _first_plan(players, free_transfers=0)

    result = optimize_strategy_roadmap(players, (roll,), horizon=8)

    first, *future = result.best_roadmap.steps
    assert first.free_transfers_before == 0
    assert first.free_transfers_next_gameweek == 1
    assert all(1 <= step.free_transfers_before <= 5 for step in future)
    assert all(1 <= step.free_transfers_next_gameweek <= 5 for step in future)


def test_original_player_keeps_confirmed_half_profit_basis_until_sold() -> None:
    players = _pool()
    players.loc[
        players["id"] == 8,
        ["purchase_price", "now_cost", "selling_price"],
    ] = [45, 50, 47]
    players.loc[players["id"] == 8, [f"ep_gw{gw}" for gw in range(2, 9)]] = 0.0
    players.loc[players["id"] == 102, [f"ep_gw{gw}" for gw in range(2, 9)]] = 25.0
    roll = _first_plan(players, bank_tenths=3)

    result = optimize_strategy_roadmap(players, (roll,), horizon=8, roll_ft_value_points=0)
    second = result.best_roadmap.steps[1]

    move = next(move for move in second.transfers if move.out_id == 8)
    assert move.in_id == 102
    assert move.out_purchase_price_tenths == 45
    assert move.out_current_price_tenths == 50
    assert move.out_selling_price_tenths == 47
    assert second.bank_before_tenths == 3
    assert second.bank_after_tenths == 0


def test_player_bought_first_uses_current_price_basis_when_sold_later() -> None:
    players = _pool()
    players.loc[
        players["id"] == 8,
        ["purchase_price", "now_cost", "selling_price"],
    ] = [45, 50, 47]
    players.loc[players["id"] == 102, "ep_gw1"] = 40.0
    players.loc[players["id"] == 102, [f"ep_gw{gw}" for gw in range(2, 9)]] = 0.0
    players.loc[players["id"] == 104, [f"ep_gw{gw}" for gw in range(2, 9)]] = 30.0
    buy_first = _first_plan(players, ((8, 102),), bank_tenths=3)

    result = optimize_strategy_roadmap(
        players,
        (buy_first,),
        horizon=8,
        roll_ft_value_points=0,
    )
    second = result.best_roadmap.steps[1]

    move = next(move for move in second.transfers if move.out_id == 102)
    assert move.in_id == 104
    assert move.out_purchase_price_tenths == 50
    assert move.out_selling_price_tenths == 50
    assert second.bank_before_tenths == 0
    assert second.bank_after_tenths == 0


def test_provisional_purchase_uses_current_price_basis_at_the_next_deadline() -> None:
    players = _pool()
    for player_id in (9, 10, 11, 12):
        players.loc[
            players["id"] == player_id,
            [f"ep_gw{gw}" for gw in range(2, 9)],
        ] = 100.0
    players.loc[players["id"] == 8, [f"ep_gw{gw}" for gw in range(2, 9)]] = 0.0
    players.loc[
        players["id"] == 102,
        [f"ep_gw{gw}" for gw in range(2, 9)],
    ] = [200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    players.loc[
        players["id"] == 104,
        [f"ep_gw{gw}" for gw in range(2, 9)],
    ] = [0.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0]

    result = optimize_strategy_roadmap(
        players,
        (_first_plan(players),),
        horizon=8,
        roll_ft_value_points=0,
    )
    second, third = result.best_roadmap.steps[1:3]

    assert [(move.out_id, move.in_id) for move in second.transfers] == [(8, 102)]
    assert [(move.out_id, move.in_id) for move in third.transfers] == [(102, 104)]
    move = third.transfers[0]
    assert move.out_purchase_price_tenths == 50
    assert move.out_current_price_tenths == 50
    assert move.out_selling_price_tenths == 50


def test_free_transfers_and_hit_cost_are_chained_across_deadlines() -> None:
    players = _pool()
    players.loc[players["id"] == 101, "ep_gw1"] = 35.0
    players.loc[
        players["id"].isin([102, 103]),
        [f"ep_gw{gw}" for gw in range(2, 9)],
    ] = 30.0
    first = _first_plan(players, ((3, 101),), free_transfers=1)

    result = optimize_strategy_roadmap(
        players,
        (first,),
        horizon=8,
        roll_ft_value_points=0,
    )
    first_step, second = result.best_roadmap.steps[:2]

    assert first_step.free_transfers_next_gameweek == 1
    assert second.free_transfers_before == 1
    assert second.transfer_count == 2
    assert {move.in_id for move in second.transfers} == {102, 103}
    assert second.hit_points == 4
    assert second.free_transfers_next_gameweek == 1
    assert second.weighted_hit_cost_points == pytest.approx(
        4 * DEFAULT_STRATEGY_GW_WEIGHTS[1],
        abs=0.001,
    )


def test_transfer_in_allowlist_applies_to_first_and_provisional_steps() -> None:
    players = _pool()
    players.loc[players["id"] == 102, [f"ep_gw{gw}" for gw in range(2, 9)]] = 100.0
    roll = _first_plan(players)
    forbidden_first = _first_plan(players, ((8, 102),))
    eligible_ids = {int(player_id) for player_id in players["id"] if int(player_id) != 102}

    result = optimize_strategy_roadmap(
        players,
        (forbidden_first, roll),
        horizon=8,
        eligible_transfer_in_ids=eligible_ids,
    )

    assert result.best_roadmap.first_step_candidate_index == 1
    assert all(
        move.in_id != 102
        for step in result.best_roadmap.steps
        for move in step.transfers
    )
    assert all(102 not in step.squad_ids for step in result.best_roadmap.steps)


def test_terminal_ft_value_includes_unmodelled_rolls_once() -> None:
    players = _pool(horizon=10)
    roll = _first_plan(players, free_transfers=1)

    result = optimize_strategy_roadmap(
        players,
        (roll,),
        horizon=10,
        roll_ft_value_points=1.25,
    )
    roadmap = result.best_roadmap
    after_four = roadmap.steps[-1].free_transfers_next_gameweek
    terminal_ft = min(5, after_four + 10 - MODELLED_DEADLINES)

    assert terminal_ft == 5
    assert roadmap.terminal_banked_ft_value_points == pytest.approx(5.0)
    assert roadmap.decision_value_points == pytest.approx(
        roadmap.weighted_projected_points
        - roadmap.weighted_hit_cost_points
        + roadmap.terminal_banked_ft_value_points,
        abs=0.001,
    )


@pytest.mark.parametrize("horizon", [5, 11, True])
def test_rejects_horizons_outside_six_to_ten(horizon) -> None:
    players = _pool()
    with pytest.raises(StrategyPlannerError, match="horizon"):
        optimize_strategy_roadmap(players, (_first_plan(players),), horizon=horizon)


def test_solver_budget_is_bounded_restored_and_proven_optimal() -> None:
    players = _pool(horizon=6)
    solver = pulp.PULP_CBC_CMD(msg=False, threads=1, timeLimit=60)
    observed_limits: list[float] = []
    actual_solve = solver.actualSolve

    def record_actual_solve(model, **kwargs):
        observed_limits.append(float(solver.timeLimit))
        return actual_solve(model, **kwargs)

    solver.actualSolve = record_actual_solve  # type: ignore[method-assign]
    started = monotonic()
    result = optimize_strategy_roadmap(
        players,
        (_first_plan(players),),
        horizon=6,
        solver_budget_seconds=5,
        solver=solver,
    )

    assert monotonic() - started < 5
    assert observed_limits == [5.0]
    assert solver.timeLimit == 60
    assert result.solver_proven_optimal_within_bounds is True


@pytest.mark.parametrize(
    ("status", "solution_status"),
    [
        (pulp.LpStatusNotSolved, pulp.LpSolutionNoSolutionFound),
        (pulp.LpStatusOptimal, pulp.LpSolutionIntegerFeasible),
    ],
)
def test_nonoptimal_or_time_limited_incumbent_fails_closed_and_restores_limit(
    status: int,
    solution_status: int,
) -> None:
    players = _pool(horizon=6)
    solver = pulp.PULP_CBC_CMD(msg=False, threads=1, timeLimit=60)

    def no_solution(model, **kwargs):
        model.status = status
        model.sol_status = solution_status
        return model.status

    solver.actualSolve = no_solution  # type: ignore[method-assign]
    with pytest.raises(StrategyPlannerError, match="could not prove an optimal"):
        optimize_strategy_roadmap(
            players,
            (_first_plan(players),),
            horizon=6,
            solver_budget_seconds=0.01,
            solver=solver,
        )
    assert solver.timeLimit == 60
