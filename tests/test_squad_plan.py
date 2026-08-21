from __future__ import annotations

from collections import Counter

import pandas as pd

from fpl_app.logic.squad_plan import optimize_squad_plan


def _fixed_squad_two_gameweeks() -> pd.DataFrame:
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    gw1 = [6, 1, 12, 11, 10, 2, 1, 16, 9, 8, 7, 6, 14, 13, 3]
    gw2 = [7, 2, 12, 11, 10, 9, 8, 7, 6, 3, 2, 1, 18, 15, 14]
    rows = []
    for player_id, (position, ep_gw1, ep_gw2) in enumerate(
        zip(positions, gw1, gw2),
        start=1,
    ):
        rows.append(
            {
                "id": player_id,
                "team_id": ((player_id - 1) % 8) + 1,
                "pos": position,
                "now_cost": 50,
                "ep_gw1": float(ep_gw1),
                "ep_gw2": float(ep_gw2),
                "appearance_prob_gw1": 1.0,
                "appearance_prob_gw2": 1.0,
                "no_show_prob_gw1": 0.0,
                "no_show_prob_gw2": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _candidate_pool() -> pd.DataFrame:
    rows = []
    player_id = 1
    for position, count, peak in (
        ("GKP", 4, 8.0),
        ("DEF", 8, 13.0),
        ("MID", 8, 15.0),
        ("FWD", 6, 16.0),
    ):
        for offset in range(count):
            rows.append(
                {
                    "id": player_id,
                    "team_id": ((player_id - 1) % 8) + 1,
                    "pos": position,
                    "now_cost": 50,
                    "ep_gw1": peak - offset * 0.5,
                    "ep_gw2": peak * 0.8 - offset * 0.25,
                    "appearance_prob_gw1": 1.0,
                    "appearance_prob_gw2": 1.0,
                    "no_show_prob_gw1": 0.0,
                    "no_show_prob_gw2": 0.0,
                }
            )
            player_id += 1
    return pd.DataFrame(rows)


def _assert_gameweek_legality(players: pd.DataFrame, squad_ids: set[int], plan) -> None:
    by_id = players.set_index("id")
    starters = set(plan.starting_ids)
    bench = set(plan.bench_ids)
    assert len(starters) == 11
    assert len(bench) == 4
    assert starters | bench == squad_ids
    assert not starters & bench
    assert plan.captain_id in starters
    assert plan.vice_captain_id in starters
    assert plan.captain_id != plan.vice_captain_id

    starter_positions = Counter(by_id.loc[list(starters), "pos"])
    assert starter_positions["GKP"] == 1
    assert 3 <= starter_positions["DEF"] <= 5
    assert 2 <= starter_positions["MID"] <= 5
    assert 1 <= starter_positions["FWD"] <= 3
    assert plan.formation == (
        f"{starter_positions['DEF']}-{starter_positions['MID']}-"
        f"{starter_positions['FWD']}"
    )
    assert all(by_id.at[player_id, "pos"] != "GKP" for player_id in plan.bench_ids[:3])
    assert by_id.at[plan.bench_ids[3], "pos"] == "GKP"


def test_uses_distinct_legal_lineup_captain_and_formation_per_gameweek() -> None:
    players = _fixed_squad_two_gameweeks()

    result = optimize_squad_plan(players, horizon=2)

    assert len(result.squad_ids) == 15
    assert len(result.gameweeks) == 2
    squad_ids = set(result.squad_ids)
    for plan in result.gameweeks:
        _assert_gameweek_legality(players, squad_ids, plan)

    first, second = result.gameweeks
    assert first.formation == "3-5-2"
    assert second.formation == "5-2-3"
    assert first.captain_id == 8
    assert second.captain_id == 13
    assert first.starting_ids != second.starting_ids


def test_selected_squad_respects_budget_position_quotas_and_club_limit() -> None:
    players = _candidate_pool()
    expensive_id = int(players.loc[players["pos"] == "MID", "id"].iloc[-1])
    players.loc[players["id"] == expensive_id, ["now_cost", "ep_gw1", "ep_gw2"]] = [
        400,
        100.0,
        100.0,
    ]
    team_99_ids = [1, 5, 13, 21]
    players.loc[players["id"].isin(team_99_ids), "team_id"] = 99
    players.loc[players["id"].isin(team_99_ids), ["ep_gw1", "ep_gw2"]] = [30.0, 30.0]

    result = optimize_squad_plan(players, horizon=2, budget_tenths=1000)
    selected = players.set_index("id").loc[list(result.squad_ids)]

    assert len(result.squad_ids) == 15
    assert result.total_cost_tenths <= 1000
    assert result.bank_tenths == 1000 - result.total_cost_tenths
    assert expensive_id not in result.squad_ids
    assert selected["pos"].value_counts().to_dict() == {
        "MID": 5,
        "DEF": 5,
        "FWD": 3,
        "GKP": 2,
    }
    assert int(selected["team_id"].value_counts().max()) <= 3
    assert int((selected["team_id"] == 99).sum()) == 3


def test_low_appearance_player_is_penalized_in_bench_order() -> None:
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    scores = [10, 5, 14, 13, 12, 8, 3, 15, 14, 13, 12, 11, 16, 15, 14]
    rows = []
    for player_id, (position, score) in enumerate(zip(positions, scores), start=1):
        rows.append(
            {
                "id": player_id,
                "team_id": ((player_id - 1) % 8) + 1,
                "pos": position,
                "now_cost": 50,
                "ep_gw1": float(score),
                "appearance_prob_gw1": 1.0,
                "no_show_prob_gw1": 0.0,
            }
        )
    players = pd.DataFrame(rows)
    low_appearance_id = 6
    players.loc[players["id"] == low_appearance_id, "appearance_prob_gw1"] = 0.1
    players.loc[players["id"] == low_appearance_id, "no_show_prob_gw1"] = 0.9

    plan = optimize_squad_plan(players, horizon=1).gameweeks[0]

    assert low_appearance_id in plan.bench_ids[:3]
    assert 7 in plan.bench_ids[:3]
    assert plan.bench_ids.index(7) < plan.bench_ids.index(low_appearance_id)


def test_result_is_deterministic_when_input_rows_are_shuffled() -> None:
    players = _candidate_pool()

    first = optimize_squad_plan(players, horizon=2)
    second = optimize_squad_plan(
        players.sample(frac=1.0, random_state=42),
        horizon=2,
    )

    assert first == second
