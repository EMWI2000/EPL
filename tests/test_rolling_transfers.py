from __future__ import annotations

from collections import Counter

import pandas as pd
import pulp
import pytest

import fpl_app.logic.rolling_transfers as rolling_module
from fpl_app.logic.rolling_transfers import (
    RollingTransferError,
    optimize_rolling_transfers,
)


def _pool() -> tuple[pd.DataFrame, tuple[int, ...]]:
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    rows: list[dict[str, object]] = []
    for player_id, position in enumerate(positions, start=1):
        rows.append(
            {
                "id": player_id,
                "team_id": ((player_id - 1) % 8) + 1,
                "pos": position,
                "now_cost": 50,
                "purchase_price": 50,
                "selling_price": 50,
                "ep_gw1": 5.0,
                "ep_gw2": 4.0,
            }
        )

    candidate_id = 101
    for position, teams in {
        "GKP": (9, 10),
        "DEF": (9, 10, 11),
        "MID": (9, 10, 11),
        "FWD": (9, 10, 11),
    }.items():
        for offset, team_id in enumerate(teams):
            rows.append(
                {
                    "id": candidate_id,
                    "team_id": team_id,
                    "pos": position,
                    "now_cost": 50,
                    "purchase_price": pd.NA,
                    "selling_price": pd.NA,
                    "ep_gw1": 4.0 - offset,
                    "ep_gw2": 3.0 - offset,
                }
            )
            candidate_id += 1
    return pd.DataFrame(rows), tuple(range(1, 16))


def _assert_legal(frame: pd.DataFrame, plan) -> None:
    by_id = frame.set_index("id")
    squad = by_id.loc[list(plan.squad_ids)]
    assert len(squad) == 15
    assert Counter(squad["pos"]) == Counter({"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3})
    assert squad["team_id"].value_counts().max() <= 3
    for gameweek in plan.gameweeks:
        starters = by_id.loc[list(gameweek.starting_ids)]
        counts = Counter(starters["pos"])
        assert len(starters) == 11
        assert counts["GKP"] == 1
        assert 3 <= counts["DEF"] <= 5
        assert 2 <= counts["MID"] <= 5
        assert 1 <= counts["FWD"] <= 3
        assert gameweek.captain_id in gameweek.starting_ids


def test_roll_is_best_when_transfers_do_not_improve_the_squad() -> None:
    players, squad_ids = _pool()

    result = optimize_rolling_transfers(
        players,
        squad_ids,
        bank_tenths=0,
        free_transfers=1,
        horizon=2,
        roll_ft_value_points=1.25,
    )

    assert result.base.action == "base"
    assert result.base.banked_ft_value_points == 0
    assert result.roll.action == "roll"
    assert result.roll.free_transfers_next_gameweek == 2
    assert result.roll.banked_ft_value_points == 1.25
    assert result.best_action == result.roll
    assert result.roll.delta_vs_base_points == 1.25
    _assert_legal(players, result.best_action)


def test_uses_actual_selling_price_and_returns_the_best_affordable_single() -> None:
    players, squad_ids = _pool()
    # Player 8 rose from 7.0 to 7.5 and can therefore be sold only for 7.2.
    players.loc[players["id"] == 8, ["purchase_price", "now_cost", "selling_price"]] = [
        70,
        75,
        72,
    ]
    midfielder_candidates = players[(players["id"] >= 106) & (players["id"] <= 108)].index
    affordable = midfielder_candidates[0]
    unaffordable = midfielder_candidates[1]
    players.loc[affordable, ["now_cost", "ep_gw1", "ep_gw2"]] = [73, 15.0, 13.0]
    players.loc[unaffordable, ["now_cost", "ep_gw1", "ep_gw2"]] = [74, 30.0, 30.0]

    result = optimize_rolling_transfers(
        players,
        squad_ids,
        bank_tenths=1,
        free_transfers=1,
        horizon=2,
        roll_ft_value_points=0,
    )

    assert result.best_action.transfer_count == 1
    move = result.best_action.transfers[0]
    assert (move.out_id, move.in_id) == (8, int(players.at[affordable, "id"]))
    assert move.out_selling_price_tenths == 72
    assert result.best_action.bank_after_tenths == 0
    assert all(move.in_id != int(players.at[unaffordable, "id"]) for move in result.best_action.transfers)


def test_two_transfers_charge_a_hit_only_above_the_free_transfer_bank() -> None:
    players, squad_ids = _pool()
    # Two different positions improve enough that the pair is optimal even at -4.
    defender = players.index[players["id"] == 103][0]
    midfielder = players.index[players["id"] == 106][0]
    players.loc[defender, ["ep_gw1", "ep_gw2"]] = [16.0, 14.0]
    players.loc[midfielder, ["ep_gw1", "ep_gw2"]] = [17.0, 15.0]

    with_one_ft = optimize_rolling_transfers(
        players,
        squad_ids,
        bank_tenths=0,
        free_transfers=1,
        horizon=2,
        roll_ft_value_points=0,
    )
    with_two_fts = optimize_rolling_transfers(
        players,
        squad_ids,
        bank_tenths=0,
        free_transfers=2,
        horizon=2,
        roll_ft_value_points=0,
    )

    assert with_one_ft.best_action.transfer_count == 2
    assert with_one_ft.best_action.hit_points == 4
    assert with_one_ft.best_action.free_transfers_next_gameweek == 1
    assert with_two_fts.best_action.transfer_count == 2
    assert with_two_fts.best_action.hit_points == 0
    assert with_two_fts.best_action.free_transfers_next_gameweek == 1


def test_zero_remaining_free_transfers_charges_the_next_move_as_a_hit() -> None:
    players, squad_ids = _pool()
    defender = players.index[players["id"] == 103][0]
    players.loc[defender, ["ep_gw1", "ep_gw2"]] = [20.0, 18.0]

    result = optimize_rolling_transfers(
        players,
        squad_ids,
        bank_tenths=0,
        free_transfers=0,
        horizon=2,
        roll_ft_value_points=0,
    )

    assert result.roll.free_transfers_before == 0
    assert result.roll.free_transfers_next_gameweek == 1
    assert result.best_action.transfer_count == 1
    assert result.best_action.free_transfers_before == 0
    assert result.best_action.hit_points == 4
    assert result.best_action.free_transfers_next_gameweek == 1


def test_club_limit_is_enforced_for_transfer_candidates() -> None:
    players, squad_ids = _pool()
    # Club 1 has two goalkeepers and a midfielder, so a fourth defender cannot
    # be made legal by simultaneously selling a same-position club-1 player.
    players.loc[players["id"].isin([1, 2, 8]), "team_id"] = 1
    players.loc[players["id"].isin([9]), "team_id"] = 8
    illegal_candidate = players.index[players["id"] == 103][0]
    legal_candidate = players.index[players["id"] == 104][0]
    players.loc[illegal_candidate, ["team_id", "ep_gw1", "ep_gw2"]] = [1, 50.0, 50.0]
    players.loc[legal_candidate, ["team_id", "ep_gw1", "ep_gw2"]] = [9, 15.0, 15.0]

    result = optimize_rolling_transfers(
        players,
        squad_ids,
        bank_tenths=0,
        free_transfers=1,
        horizon=2,
        roll_ft_value_points=0,
    )

    plans = (result.best_action, *result.alternatives)
    single_plans = [plan for plan in plans if plan.transfer_count == 1]
    assert single_plans
    assert all(
        move.in_id != 103
        for plan in single_plans
        for move in plan.transfers
    )
    _assert_legal(players, result.best_action)


def test_grandfathered_four_player_club_can_roll_but_transfers_restore_limit() -> None:
    players, squad_ids = _pool()
    players.loc[players["id"].isin([1, 2, 8, 9]), "team_id"] = 1

    result = optimize_rolling_transfers(
        players,
        squad_ids,
        bank_tenths=0,
        free_transfers=1,
        horizon=2,
    )

    by_id = players.set_index("id")
    rolled = by_id.loc[list(result.roll.squad_ids)]
    assert int((rolled["team_id"] == 1).sum()) == 4
    transfer_plans = [
        plan
        for plan in (result.best_action, *result.alternatives)
        if plan.transfer_count > 0
    ]
    assert transfer_plans
    for plan in transfer_plans:
        selected = by_id.loc[list(plan.squad_ids)]
        assert int(selected["team_id"].value_counts().max()) <= 3


def test_no_show_adjusted_bench_value_affects_objective_and_reported_points() -> None:
    players, squad_ids = _pool()
    players["no_show_prob_gw1"] = 0.0
    players.loc[players["id"] == 7, "no_show_prob_gw1"] = 1.0
    players.loc[players["id"].isin([3, 4, 5]), "ep_gw1"] = 10.0
    players.loc[players["id"].isin([6, 7, 103, 104, 105]), "ep_gw1"] = 1.0
    # Player 7 and this equally projected candidate are both reserve defenders;
    # only the candidate contributes expected substitution value (1.0 * 0.08).

    result = optimize_rolling_transfers(
        players,
        squad_ids,
        bank_tenths=0,
        free_transfers=1,
        horizon=1,
        roll_ft_value_points=0,
    )

    assert result.best_action.transfer_count == 1
    assert [(move.out_id, move.in_id) for move in result.best_action.transfers] == [
        (7, 103)
    ]
    assert result.best_action.projected_points == pytest.approx(
        result.roll.projected_points + 0.08
    )

    plan = result.best_action.gameweeks[0]
    by_id = players.set_index("id")
    starters = set(plan.starting_ids)
    bench = set(result.best_action.squad_ids) - starters
    expected = sum(float(by_id.at[player_id, "ep_gw1"]) for player_id in starters)
    expected += float(by_id.at[plan.captain_id, "ep_gw1"])
    expected += sum(
        float(by_id.at[player_id, "ep_gw1"])
        * (1.0 - float(by_id.at[player_id, "no_show_prob_gw1"]))
        * (0.02 if by_id.at[player_id, "pos"] == "GKP" else 0.08)
        for player_id in bench
    )
    assert plan.projected_points == pytest.approx(expected)


def test_five_free_transfers_are_evaluated_with_one_bounded_plan_each() -> None:
    players, squad_ids = _pool()
    players.loc[
        players["id"].isin([103, 104, 106, 107, 109]),
        ["ep_gw1", "ep_gw2"],
    ] = [20.0, 18.0]

    result = optimize_rolling_transfers(
        players,
        squad_ids,
        bank_tenths=0,
        free_transfers=5,
        horizon=2,
        plans_per_transfer_count=5,
    )

    assert result.best_action.transfer_count == 5
    counts = Counter(
        plan.transfer_count for plan in (result.best_action, *result.alternatives)
    )
    assert counts[3] == 1
    assert counts[4] == 1
    assert counts[5] == 1


def test_each_cbc_solve_uses_only_the_remaining_global_budget(monkeypatch) -> None:
    players, squad_ids = _pool()
    solver = pulp.PULP_CBC_CMD(msg=False, threads=1, timeLimit=60)
    observed_limits: list[float] = []
    actual_solve = solver.actualSolve

    def record_actual_solve(model, **kwargs):
        observed_limits.append(float(solver.timeLimit))
        return actual_solve(model, **kwargs)

    monkeypatch.setattr(solver, "actualSolve", record_actual_solve)
    optimize_rolling_transfers(
        players,
        squad_ids,
        bank_tenths=0,
        free_transfers=1,
        horizon=1,
        plans_per_transfer_count=1,
        solver=solver,
    )

    assert observed_limits
    assert all(0 < limit <= 15 for limit in observed_limits)
    assert observed_limits == sorted(observed_limits, reverse=True)
    assert solver.timeLimit == 60


def test_solver_covers_every_transfer_count_before_extra_alternatives(monkeypatch) -> None:
    players, squad_ids = _pool()
    solver = pulp.PULP_CBC_CMD(msg=False, threads=1, timeLimit=60)
    clock = [0.0]
    solve_order: list[int] = []
    actual_solve = solver.actualSolve

    monkeypatch.setattr(rolling_module, "monotonic", lambda: clock[0])

    def record_actual_solve(model, **kwargs):
        solve_order.append(int(model.name.rsplit("_", 1)[1]))
        result = actual_solve(model, **kwargs)
        clock[0] += 1.0
        return result

    monkeypatch.setattr(solver, "actualSolve", record_actual_solve)
    optimize_rolling_transfers(
        players,
        squad_ids,
        bank_tenths=0,
        free_transfers=5,
        horizon=1,
        plans_per_transfer_count=5,
        solver=solver,
    )

    assert solve_order[:6] == [0, 1, 2, 3, 4, 5]
    assert solve_order[6:] == [1, 1, 1, 1, 2, 2, 2, 2]


def test_coverage_fails_closed_when_budget_expires_before_next_count(monkeypatch) -> None:
    players, squad_ids = _pool()
    solver = pulp.PULP_CBC_CMD(msg=False, threads=1, timeLimit=60)
    clock = [0.0]
    actual_solve = solver.actualSolve

    monkeypatch.setattr(rolling_module, "monotonic", lambda: clock[0])

    def consume_budget_after_first_transfer_count(model, **kwargs):
        result = actual_solve(model, **kwargs)
        transfer_count = int(model.name.rsplit("_", 1)[1])
        clock[0] = 15.0 if transfer_count == 1 else 1.0
        return result

    monkeypatch.setattr(
        solver,
        "actualSolve",
        consume_budget_after_first_transfer_count,
    )

    with pytest.raises(
        RollingTransferError,
        match=r"conclusively evaluate 2 transfers.*15-second solver budget",
    ):
        optimize_rolling_transfers(
            players,
            squad_ids,
            bank_tenths=0,
            free_transfers=1,
            horizon=1,
            plans_per_transfer_count=1,
            solver=solver,
        )


def test_five_free_transfers_make_one_transfer_use_it_or_lose_it() -> None:
    players, squad_ids = _pool()
    candidate = players.index[players["id"] == 103][0]
    players.loc[candidate, ["ep_gw1", "ep_gw2"]] = [8.0, 7.0]

    result = optimize_rolling_transfers(
        players,
        squad_ids,
        bank_tenths=0,
        free_transfers=5,
        horizon=2,
        roll_ft_value_points=2.0,
    )

    assert result.roll.free_transfers_next_gameweek == 5
    assert result.roll.banked_ft_value_points == 8.0
    assert result.best_action.transfer_count == 1
    assert result.best_action.free_transfers_next_gameweek == 5
    assert result.best_action.banked_ft_value_points == 8.0


def test_results_are_deterministic_when_input_rows_are_shuffled() -> None:
    players, squad_ids = _pool()
    candidates = players.index[players["id"].isin([103, 104, 106, 107])]
    players.loc[candidates, ["ep_gw1", "ep_gw2"]] = [10.0, 9.0]

    first = optimize_rolling_transfers(
        players,
        squad_ids,
        bank_tenths=0,
        free_transfers=2,
        horizon=2,
        plans_per_transfer_count=3,
    )
    second = optimize_rolling_transfers(
        players.sample(frac=1.0, random_state=42),
        squad_ids,
        bank_tenths=0,
        free_transfers=2,
        horizon=2,
        plans_per_transfer_count=3,
    )

    assert first == second


def test_rejects_a_selling_price_that_uses_full_market_profit() -> None:
    players, squad_ids = _pool()
    players.loc[players["id"] == 8, ["purchase_price", "now_cost", "selling_price"]] = [
        70,
        75,
        75,
    ]

    with pytest.raises(RollingTransferError, match="half-profit rule"):
        optimize_rolling_transfers(
            players,
            squad_ids,
            bank_tenths=0,
            free_transfers=1,
            horizon=2,
        )
