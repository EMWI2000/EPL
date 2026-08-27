from __future__ import annotations

import pandas as pd
import pytest

from fpl_app.logic.chip_strategy import ChipStrategyError, evaluate_chip_strategy
from fpl_app.logic.rolling_transfers import TransferGameweekPlan
from fpl_app.logic.strategy_planner import (
    BoundedStrategyResult,
    FourDeadlineRoadmap,
    StrategyTransferStep,
)


WEIGHTS = tuple(0.85**offset for offset in range(8))


def _players() -> pd.DataFrame:
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    rows = []
    for player_id, position in enumerate(positions, start=1):
        row = {
            "id": player_id,
            "name": f"Player {player_id}",
            "team": f"T{((player_id - 1) % 8) + 1}",
            "team_id": ((player_id - 1) % 8) + 1,
            "pos": position,
            "status": "a",
            "now_cost": 50,
            "purchase_price": 50,
            "selling_price": 50,
        }
        for offset in range(1, 9):
            # Player 8 is the clear captain; the three weakest outfield players
            # and reserve goalkeeper form a deterministic bench.
            ep = 9.0 if player_id == 8 else max(1.0, 7.0 - player_id * 0.2)
            row[f"ep_gw{offset}"] = ep
            row[f"no_show_prob_gw{offset}"] = 0.0
            row[f"appearance_prob_gw{offset}"] = 1.0
            row[f"confidence_gw{offset}"] = 0.8
            row[f"reliability_gw{offset}"] = "medium"
            row[f"is_blank_gw{offset}"] = False
            row[f"is_dgw_gw{offset}"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def _lineup(offset: int) -> TransferGameweekPlan:
    starters = (1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14)
    return TransferGameweekPlan(
        gameweek=offset,
        starting_ids=starters,
        captain_id=8,
        formation="3-5-2",
        projected_points=75.0,
    )


def _roadmap() -> BoundedStrategyResult:
    squad_ids = tuple(range(1, 16))
    steps = []
    for deadline in range(1, 5):
        governed = (deadline,) if deadline < 4 else tuple(range(4, 9))
        steps.append(
            StrategyTransferStep(
                deadline_offset=deadline,
                provisional=deadline > 1,
                transfers=(),
                squad_ids=squad_ids,
                gameweeks=tuple(_lineup(offset) for offset in governed),
                bank_before_tenths=0,
                bank_after_tenths=0,
                free_transfers_before=min(5, deadline),
                free_transfers_next_gameweek=min(5, deadline + 1),
                hit_points=0,
                weighted_projected_points=round(
                    sum(75.0 * WEIGHTS[offset - 1] for offset in governed),
                    3,
                ),
                weighted_hit_cost_points=0.0,
            )
        )
    projected = round(sum(step.weighted_projected_points for step in steps), 3)
    plan = FourDeadlineRoadmap(
        first_step_candidate_index=0,
        steps=(steps[0], steps[1], steps[2], steps[3]),
        weighted_projected_points=projected,
        total_hit_points=0,
        weighted_hit_cost_points=0.0,
        terminal_banked_ft_value_points=4.0,
        decision_value_points=round(projected + 4.0, 3),
    )
    return BoundedStrategyResult(
        best_roadmap=plan,
        horizon=8,
        gw_weights=WEIGHTS,
        forecast_columns=tuple(f"ep_gw{offset}" for offset in range(1, 9)),
        first_step_candidate_count=1,
    )


def _usage(**overrides: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
    result = {"wildcard": (), "freehit": (), "bboost": (), "3xc": ()}
    result.update(overrides)
    return result


def test_returns_personalized_four_chip_scenarios_and_current_call() -> None:
    result = evaluate_chip_strategy(
        _players(),
        _roadmap(),
        target_event=2,
        chip_usage=_usage(),
        current_squad_ids=tuple(range(1, 16)),
        bank_tenths=0,
        free_transfers=1,
        chip_solver_budget_seconds=2.0,
    )

    assert [entry.chip for entry in result.inventory] == [
        "wildcard",
        "freehit",
        "bboost",
        "3xc",
    ]
    assert all(entry.available_for_target for entry in result.inventory)
    scenarios = {scenario.chip: scenario for scenario in result.scenarios}
    assert scenarios["freehit"].signal == "hold"
    assert scenarios["freehit"].event is None
    assert scenarios["3xc"].event == 2
    assert scenarios["3xc"].estimated_gain_points == 9.0
    assert scenarios["3xc"].signal == "consider"
    assert scenarios["bboost"].estimated_gain_points is not None
    assert scenarios["wildcard"].change_count == 0
    assert result.recommendation.action == "consider"
    assert result.recommendation.scenario_id in {"3xc-gw2", "bboost-gw2"}
    assert result.recommendation.scenario_id == max(
        (scenarios["3xc"], scenarios["bboost"]),
        key=lambda scenario: scenario.estimated_gain_points or 0.0,
    ).scenario_id
    assert result.globally_optimal is False


def test_accepts_zero_remaining_free_transfers_for_the_current_deadline() -> None:
    result = evaluate_chip_strategy(
        _players(),
        _roadmap(),
        target_event=2,
        chip_usage=_usage(),
        current_squad_ids=tuple(range(1, 16)),
        bank_tenths=0,
        free_transfers=0,
        chip_solver_budget_seconds=2.0,
    )

    assert len(result.scenarios) == 4


def test_used_chip_is_unavailable_only_in_its_used_half() -> None:
    first_half = evaluate_chip_strategy(
        _players(),
        _roadmap(),
        target_event=2,
        chip_usage=_usage(**{"3xc": (1,)}),
        current_squad_ids=tuple(range(1, 16)),
        bank_tenths=0,
        free_transfers=1,
        chip_solver_budget_seconds=2.0,
    )
    inventory = {entry.chip: entry for entry in first_half.inventory}
    assert inventory["3xc"].available_for_target is False
    assert {scenario.chip: scenario for scenario in first_half.scenarios}["3xc"].available is False


def test_confirmed_blank_or_double_creates_only_a_free_hit_watchpoint() -> None:
    players = _players()
    players["is_blank_gw3"] = True
    result = evaluate_chip_strategy(
        players,
        _roadmap(),
        target_event=2,
        chip_usage=_usage(),
        current_squad_ids=tuple(range(1, 16)),
        bank_tenths=0,
        free_transfers=1,
        chip_solver_budget_seconds=2.0,
    )
    free_hit = {scenario.chip: scenario for scenario in result.scenarios}["freehit"]
    assert free_hit.event == 4
    assert free_hit.signal == "watch"
    assert free_hit.estimated_gain_points is not None
    assert len(free_hit.squad_ids) == 15
    assert free_hit.model_scope == "single_gameweek_counterfactual"


def test_free_hit_screen_counts_unique_clubs_not_player_rows() -> None:
    players = _players()
    players.loc[players["id"].isin([1, 2, 9, 10]), "is_blank_gw3"] = True
    players.loc[players["id"].isin([3, 4, 5]), "is_blank_gw4"] = True
    result = evaluate_chip_strategy(
        players,
        _roadmap(),
        target_event=2,
        chip_usage=_usage(),
        current_squad_ids=tuple(range(1, 16)),
        bank_tenths=0,
        free_transfers=1,
        chip_solver_budget_seconds=2.0,
    )
    free_hit = {scenario.chip: scenario for scenario in result.scenarios}["freehit"]
    assert free_hit.event == 5
    assert free_hit.signal == "watch"


@pytest.mark.parametrize(
    "usage",
    [
        {"mystery": ()},
        {"wildcard": (1,)},
        {"freehit": (5, 10)},
        {"3xc": (True,)},
        {"bboost": (1,), "3xc": (1,)},
        {"wildcard": (2,)},
    ],
)
def test_rejects_unknown_illegal_duplicate_or_non_integer_usage(usage) -> None:
    with pytest.raises(ChipStrategyError):
        evaluate_chip_strategy(
            _players(),
            _roadmap(),
            target_event=2,
            chip_usage=usage,
            current_squad_ids=tuple(range(1, 16)),
            bank_tenths=0,
            free_transfers=1,
            chip_solver_budget_seconds=1.0,
        )


def test_rejects_consecutive_free_hits_across_chip_halves() -> None:
    with pytest.raises(ChipStrategyError, match="consecutive"):
        evaluate_chip_strategy(
            _players(),
            _roadmap(),
            target_event=21,
            chip_usage=_usage(freehit=(19, 20)),
            current_squad_ids=tuple(range(1, 16)),
            bank_tenths=0,
            free_transfers=1,
            chip_solver_budget_seconds=1.0,
        )
