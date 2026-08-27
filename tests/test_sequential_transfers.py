from __future__ import annotations

from collections import Counter
from itertools import combinations
import json

import pandas as pd
import pulp
import pytest

from fpl_app.domain.rules import (
    free_transfers_next_gameweek,
    transfer_points_cost,
)
from fpl_app.logic.rolling_transfers import (
    DEFAULT_BENCH_WEIGHTS,
    DEFAULT_GW_WEIGHTS,
    TransferGameweekPlan,
    TransferMove,
    TransferPlan,
)
from fpl_app.logic.sequential_transfers import (
    FUTURE_PRICE_ASSUMPTION,
    SequentialTransferError,
    optimize_two_deadline_sequence,
)


INITIAL_IDS = tuple(range(1, 16))


def _pool(horizon: int = 2) -> pd.DataFrame:
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
    offset: int,
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
                    for midfielders in combinations(
                        ids_by_position["MID"], midfielder_count
                    ):
                        for forwards in combinations(
                            ids_by_position["FWD"], forward_count
                        ):
                            starters = tuple(
                                sorted(
                                    (goalkeeper, *defenders, *midfielders, *forwards)
                                )
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
                                weight = (
                                    DEFAULT_BENCH_WEIGHTS[3]
                                    if str(by_id.at[player_id, "pos"]) == "GKP"
                                    else sum(DEFAULT_BENCH_WEIGHTS[:3]) / 3.0
                                )
                                points += float(by_id.at[player_id, ep_column]) * weight
                            formation = (
                                f"{defender_count}-{midfielder_count}-{forward_count}"
                            )
                            candidate = (points, starters, captain, formation)
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
    horizon: int = 2,
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
    gameweeks = tuple(
        _best_lineup(frame, ordered_squad, offset)
        for offset in range(1, horizon + 1)
    )
    projected_points = sum(
        gameweek.projected_points * DEFAULT_GW_WEIGHTS[offset]
        for offset, gameweek in enumerate(gameweeks)
    )
    transfer_count = len(moves)
    hit_points = transfer_points_cost(transfer_count, free_transfers)
    next_free_transfers = free_transfers_next_gameweek(
        free_transfers,
        transfer_count,
    )
    bank_after = (
        bank_tenths
        + sum(move.out_selling_price_tenths for move in moves)
        - sum(move.in_price_tenths for move in moves)
    )
    banked_value = max(0, next_free_transfers - 1)
    return TransferPlan(
        action="roll" if not moves else f"{len(moves)}_transfer",
        transfers=tuple(moves),
        squad_ids=ordered_squad,
        gameweeks=gameweeks,
        bank_before_tenths=bank_tenths,
        bank_after_tenths=bank_after,
        free_transfers_before=free_transfers,
        free_transfers_next_gameweek=next_free_transfers,
        hit_points=hit_points,
        projected_points=round(projected_points, 3),
        banked_ft_value_points=float(banked_value),
        decision_value_points=round(projected_points - hit_points + banked_value, 3),
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
        assert set(gameweek.starting_ids).issubset(step.squad_ids)
        assert gameweek.captain_id in gameweek.starting_ids
        assert counts["GKP"] == 1
        assert gameweek.formation == (
            f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"
        )
        assert 3 <= counts["DEF"] <= 5
        assert 2 <= counts["MID"] <= 5
        assert 1 <= counts["FWD"] <= 3


def test_rolls_first_then_makes_the_deferred_second_deadline_transfer() -> None:
    players = _pool()
    players.loc[players["id"] == 102, ["ep_gw1", "ep_gw2"]] = [0.0, 20.0]
    roll = _first_plan(players)
    buy_early = _first_plan(players, ((8, 102),))

    result = optimize_two_deadline_sequence(
        players,
        (roll, buy_early),
        horizon=2,
    )

    sequence = result.best_sequence
    assert sequence.first_step_candidate_index == 0
    first, second = sequence.steps
    assert first.transfer_count == 0
    assert first.free_transfers_next_gameweek == 2
    assert second.transfer_count == 1
    assert second.transfers[0].in_id == 102
    assert second.transfers[0].out_id in {8, 9, 10, 11, 12}
    assert second.free_transfers_before == 2
    assert second.free_transfers_next_gameweek == 2
    assert second.hit_points == 0
    assert 102 not in first.squad_ids
    assert 102 in second.squad_ids
    assert second.provisional is True
    _assert_step_is_legal(players, first)
    _assert_step_is_legal(players, second)


def test_zero_free_transfers_is_legal_only_at_the_first_deadline() -> None:
    players = _pool()
    roll = _first_plan(players, free_transfers=0)

    result = optimize_two_deadline_sequence(players, (roll,), horizon=2)

    first, second = result.best_sequence.steps
    assert first.free_transfers_before == 0
    assert first.free_transfers_next_gameweek == 1
    assert second.free_transfers_before == 1
    assert 1 <= second.free_transfers_next_gameweek <= 5


def test_player_bought_first_uses_fixed_current_price_when_sold_second() -> None:
    players = _pool()
    players.loc[
        players["id"] == 8,
        ["purchase_price", "now_cost", "selling_price"],
    ] = [45, 50, 47]
    players.loc[players["id"] == 102, ["ep_gw1", "ep_gw2"]] = [30.0, 0.0]
    players.loc[players["id"] == 104, ["ep_gw1", "ep_gw2"]] = [0.0, 30.0]
    first_candidate = _first_plan(
        players,
        ((8, 102),),
        bank_tenths=3,
    )

    result = optimize_two_deadline_sequence(
        players,
        (first_candidate,),
        horizon=2,
        roll_ft_value_points=0,
    )

    first, second = result.best_sequence.steps
    assert first.bank_after_tenths == 0
    assert [(move.out_id, move.in_id) for move in second.transfers] == [(102, 104)]
    move = second.transfers[0]
    assert move.out_purchase_price_tenths == 50
    assert move.out_selling_price_tenths == 50
    assert second.bank_before_tenths == 0
    assert second.bank_after_tenths == 0
    assert result.future_price_assumption == FUTURE_PRICE_ASSUMPTION


def test_second_deadline_cannot_buy_a_player_outside_the_transfer_in_allowlist() -> None:
    players = _pool()
    players.loc[players["id"] == 8, ["ep_gw1", "ep_gw2"]] = [0.0, 30.0]
    players.loc[players["id"] == 102, ["ep_gw1", "ep_gw2"]] = [30.0, 0.0]
    first_candidate = _first_plan(players, ((8, 102),))

    unrestricted = optimize_two_deadline_sequence(
        players,
        (first_candidate,),
        horizon=2,
        roll_ft_value_points=0,
    )
    assert [move.in_id for move in unrestricted.best_sequence.steps[1].transfers] == [8]

    restricted = optimize_two_deadline_sequence(
        players,
        (first_candidate,),
        horizon=2,
        roll_ft_value_points=0,
        eligible_transfer_in_ids={
            int(player_id) for player_id in players["id"] if int(player_id) != 8
        },
    )
    second = restricted.best_sequence.steps[1]
    assert all(move.in_id != 8 for move in second.transfers)
    assert 8 not in second.squad_ids


def test_second_deadline_propagates_free_transfers_and_charges_a_hit() -> None:
    players = _pool()
    players.loc[players["id"] == 101, ["ep_gw1", "ep_gw2"]] = [30.0, 5.0]
    players.loc[players["id"].isin([102, 103]), ["ep_gw1", "ep_gw2"]] = [
        0.0,
        30.0,
    ]
    first_candidate = _first_plan(players, ((3, 101),))

    result = optimize_two_deadline_sequence(
        players,
        (first_candidate,),
        horizon=2,
        roll_ft_value_points=0,
    )

    first, second = result.best_sequence.steps
    assert first.free_transfers_before == 1
    assert first.free_transfers_next_gameweek == 1
    assert second.free_transfers_before == 1
    assert second.transfer_count == 2
    assert {move.in_id for move in second.transfers} == {102, 103}
    assert second.hit_points == 4
    assert second.free_transfers_next_gameweek == 1
    assert result.best_sequence.total_hit_points == 4
    assert result.best_sequence.weighted_hit_cost_points == pytest.approx(3.4)
    _assert_step_is_legal(players, second)


def test_result_is_explicitly_bounded_json_safe_and_solver_time_limited(
    monkeypatch,
) -> None:
    players = _pool()
    roll = _first_plan(players)
    solver = pulp.PULP_CBC_CMD(msg=False, threads=1, timeLimit=60)
    observed_limits: list[float] = []
    actual_solve = solver.actualSolve

    def record_actual_solve(model, **kwargs):
        observed_limits.append(float(solver.timeLimit))
        return actual_solve(model, **kwargs)

    monkeypatch.setattr(solver, "actualSolve", record_actual_solve)
    result = optimize_two_deadline_sequence(
        players.sample(frac=1.0, random_state=7),
        (roll,),
        horizon=2,
        solver_budget_seconds=5,
        solver=solver,
    )

    assert result.first_step_search == "explicit_bounded_transfer_plans"
    assert result.first_step_candidate_count == 1
    assert result.solver_proven_optimal_within_bounds is True
    assert result.globally_optimal is False
    assert observed_limits == [5.0]
    assert solver.timeLimit == 60
    json.dumps(result.as_dict(), allow_nan=False)


def test_rejects_a_first_step_whose_bank_does_not_reconcile() -> None:
    players = _pool()
    valid = _first_plan(players, ((8, 102),))
    invalid = TransferPlan(
        **{
            **valid.__dict__,
            "bank_after_tenths": valid.bank_after_tenths + 1,
        }
    )

    with pytest.raises(SequentialTransferError, match="bank_after_tenths"):
        optimize_two_deadline_sequence(
            players,
            (invalid,),
            horizon=2,
        )
