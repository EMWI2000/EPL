from __future__ import annotations

from collections import Counter
from dataclasses import replace
from time import monotonic

import pandas as pd
import pytest

from fpl_app.domain.rules import Chip, free_transfers_next_gameweek, transfer_points_cost
from fpl_app.logic.chip_sequences import evaluate_chip_sequences
from fpl_app.logic.chip_strategy import evaluate_chip_strategy
from fpl_app.logic.squad_plan import SquadPlanResult
from tests.test_chip_strategy import WEIGHTS, _players, _roadmap, _usage


def _wildcard(squad_ids=tuple(range(1, 16))) -> SquadPlanResult:
    return SquadPlanResult(
        squad_ids=squad_ids, gameweeks=(), total_cost_tenths=750, bank_tenths=0,
        objective_points=0.0, horizon=8, gw_weights=WEIGHTS,
        forecast_columns=tuple(f"ep_gw{offset}" for offset in range(1, 9)),
        appearance_columns=(), no_show_columns=(),
    )


def _compare(*, target=4, usage=None, free_transfers=1, wildcard=True, players=None, roadmap=None, deadline=None):
    frame = _players() if players is None else players
    return evaluate_chip_sequences(
        frame, _roadmap() if roadmap is None else roadmap, _wildcard() if wildcard else None,
        target_event=target,
        usage={Chip(key): value for key, value in (usage or _usage()).items()},
        current_squad_ids=tuple(range(1, 16)), bank_tenths=0,
        free_transfers=free_transfers, eligible_transfer_in_ids=frozenset(frame["id"]),
        deadline=monotonic() + 2.0 if deadline is None else deadline,
    )


def test_four_paths_use_same_horizon_and_have_legal_chip_dates_and_ft_transition() -> None:
    result = _compare(free_transfers=0)
    assert result["status"] == "ready"
    paths = {path["sequence_id"]: path for path in result["sequences"]}
    assert set(paths) == {"normal", "normal-bboost", "wildcard", "wildcard-bboost"}
    assert paths["normal"]["weighted_net_points"] == paths["wildcard"]["weighted_net_points"]
    assert paths["normal"]["actions"][0]["free_transfers_next_gameweek"] == 1
    assert paths["wildcard"]["actions"][0]["free_transfers_next_gameweek"] == 0
    wc_bb = paths["wildcard-bboost"]
    assert wc_bb["actions"][0]["chip"] == "wildcard"
    bb_event = next(action["event"] for action in wc_bb["actions"] if action["chip"] == "bboost")
    assert bb_event > 4
    for path in paths.values():
        assert len(path["actions"]) == 8
        assert path["gain_vs_normal_points"] == pytest.approx(path["weighted_net_points"] - paths["normal"]["weighted_net_points"], abs=0.001)
        for offset, action in enumerate(path["actions"], start=1):
            assert len(action["squad_ids"]) == len(set(action["squad_ids"])) == 15
            assert action["event"] == 3 + offset
            assert action["bank_after_tenths"] >= 0
            assert action["hit_points"] == transfer_points_cost(len(action["transfer_out_ids"]), action["free_transfers_before"], active_chip=action["chip"])
            assert action["free_transfers_next_gameweek"] == free_transfers_next_gameweek(action["free_transfers_before"], len(action["transfer_out_ids"]), active_chip=action["chip"])
            if offset > 4:
                assert action["transfer_out_ids"] == []


def test_bench_boost_cannot_be_carried_across_half_expiry_or_played_with_wildcard() -> None:
    result = _compare(target=19)
    paths = {path["sequence_id"]: path for path in result["sequences"]}
    assert "normal-bboost" in paths
    assert "wildcard-bboost" not in paths
    assert next(action["event"] for action in paths["normal-bboost"]["actions"] if action["chip"] == "bboost") == 19
    first_half = _compare(target=18, usage=_usage(bboost=(1,)))
    assert all("bboost" not in path["sequence_id"] for path in first_half["sequences"])
    second_half = _compare(target=20, usage=_usage(bboost=(1,)))
    assert any(path["sequence_id"] == "wildcard-bboost" for path in second_half["sequences"])


def test_used_wildcard_is_not_replayed_in_the_same_half() -> None:
    result = _compare(target=4, usage=_usage(wildcard=(2,)))
    assert all(not path["sequence_id"].startswith("wildcard") for path in result["sequences"])


def test_expired_budget_returns_no_asymmetric_partial_comparison() -> None:
    result = _compare(deadline=monotonic() - 1)
    assert result["status"] == "unavailable"
    assert result["sequences"] == []
    assert result["highest_projected_sequence_id"] is None


def test_mid_calculation_timeout_discards_both_paths(monkeypatch) -> None:
    calls = 0

    def clock():
        nonlocal calls
        calls += 1
        return 0.0 if calls < 12 else 10.0

    monkeypatch.setattr("fpl_app.logic.chip_sequences.monotonic", clock)
    result = _compare(deadline=5.0)
    assert result["status"] == "unavailable"
    assert result["sequences"] == []


def test_future_transfer_uses_exact_sell_basis_and_is_restricted_to_known_roadmap_players() -> None:
    players = _players()
    # A future roadmap player and an even stronger unrelated pool player.
    addition = players.loc[players["id"] == 3].iloc[0].to_dict()
    addition.update(id=16, name="Future defender", now_cost=50, purchase_price=50, selling_price=50, team_id=8)
    outsider = {**addition, "id": 17, "name": "Not in evaluated roadmap"}
    for offset in range(1, 9):
        addition[f"ep_gw{offset}"] = 0.0 if offset == 1 else 20.0
        outsider[f"ep_gw{offset}"] = 30.0
    players = pd.concat([players, pd.DataFrame([addition, outsider])], ignore_index=True)
    roadmap = _roadmap()
    steps = list(roadmap.best_roadmap.steps)
    steps[1] = replace(steps[1], squad_ids=tuple(value for value in range(1, 16) if value != 3) + (16,))
    roadmap = replace(roadmap, best_roadmap=replace(roadmap.best_roadmap, steps=tuple(steps)))
    result = _compare(players=players, roadmap=roadmap)
    assert result["status"] == "ready"
    for path in result["sequences"]:
        assert 16 in path["actions"][1]["transfer_in_ids"]
        assert all(17 not in action["squad_ids"] for action in path["actions"])
        previous = set(range(1, 16))
        for action in path["actions"]:
            assert set(action["transfer_out_ids"]).issubset(previous)
            assert not set(action["transfer_in_ids"]) & previous
            updated = (previous - set(action["transfer_out_ids"])) | set(action["transfer_in_ids"])
            assert updated == set(action["squad_ids"])
            assert max(Counter(int(players.set_index("id").at[value, "team_id"]) for value in updated).values()) <= 3
            previous = updated


def test_wildcard_retains_appreciated_players_without_repurchase_and_uses_paired_score() -> None:
    players = _players()
    players["purchase_price"] = 40
    players["now_cost"] = 50
    # Manager reconciliation stores optional unowned selling prices in an
    # object column; pandas 3 requires an explicit numeric cast on assignment.
    players["selling_price"] = pd.Series([45] * len(players), dtype=object)
    result = evaluate_chip_strategy(
        players, _roadmap(), target_event=4, chip_usage=_usage(),
        current_squad_ids=tuple(range(1, 16)), bank_tenths=0,
        free_transfers=1, chip_solver_budget_seconds=2.0,
    )
    wildcard = next(scenario for scenario in result.scenarios if scenario.chip == "wildcard")
    assert len(wildcard.squad_ids) == 15
    assert wildcard.bank_after_tenths == 0
    assert wildcard.change_count == 0
    comparison = result.sequence_comparison
    assert comparison["status"] == "ready"
    paths = {path["sequence_id"]: path for path in comparison["sequences"]}
    assert wildcard.baseline_points == paths["normal"]["weighted_net_points"]
    assert wildcard.chip_points == paths["wildcard"]["weighted_net_points"]
    assert wildcard.estimated_gain_points == pytest.approx(wildcard.chip_points - wildcard.baseline_points, abs=0.001)
    assert result.as_dict()["sequence_comparison"] == comparison
