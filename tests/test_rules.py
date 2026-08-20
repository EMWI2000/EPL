from collections import Counter

import pytest

from fpl_app.domain.rules import (
    BONUS_POINTS,
    CHIPS,
    CHIP_RULES,
    FORMATION_BY_CODE,
    FORMATIONS,
    POSITION_SCORING,
    SCORING,
    SQUAD,
    TRANSFERS,
    Chip,
    DefensiveAction,
    Position,
    SquadPlayer,
    appearance_points,
    can_play_chip,
    chip_window,
    defensive_contribution_points,
    formation_code,
    free_transfers_next_gameweek,
    goalkeeper_save_points,
    goals_conceded_points,
    is_valid_formation,
    selling_price_tenths,
    transfer_points_cost,
    validate_squad,
)


def test_squad_constraints_match_official_game() -> None:
    assert SQUAD.squad_size == 15
    assert SQUAD.starting_size == 11
    assert SQUAD.position_quotas == {
        Position.GOALKEEPER: 2,
        Position.DEFENDER: 5,
        Position.MIDFIELDER: 5,
        Position.FORWARD: 3,
    }
    assert SQUAD.max_players_per_club == 3
    assert SQUAD.initial_budget_tenths == 1000


def test_all_eight_legal_formations_are_available() -> None:
    assert set(FORMATION_BY_CODE) == {"343", "352", "433", "442", "451", "523", "532", "541"}
    assert len(FORMATIONS) == 8
    assert is_valid_formation({"GKP": 1, "DEF": 5, "MID": 2, "FWD": 3})
    assert formation_code({"GKP": 1, "DEF": 5, "MID": 2, "FWD": 3}) == "523"
    assert not is_valid_formation({"GKP": 1, "DEF": 2, "MID": 5, "FWD": 3})


def test_2026_27_position_scoring_constants() -> None:
    assert POSITION_SCORING[Position.GOALKEEPER].goal == 10
    assert POSITION_SCORING[Position.DEFENDER].goal == 6
    assert POSITION_SCORING[Position.MIDFIELDER].goal == 5
    assert POSITION_SCORING[Position.FORWARD].goal == 4
    assert {rule.assist for rule in POSITION_SCORING.values()} == {3}
    assert SCORING.penalty_save == 5
    assert SCORING.penalty_miss == -2
    assert SCORING.yellow_card == -1
    assert SCORING.red_card == -3
    assert SCORING.own_goal == -2


def test_appearance_saves_and_goals_conceded_use_threshold_blocks() -> None:
    assert [appearance_points(minutes) for minutes in (0, 1, 59, 60, 90)] == [0, 1, 1, 2, 2]
    assert [goalkeeper_save_points(saves) for saves in (0, 2, 3, 5, 6)] == [0, 0, 1, 1, 2]
    assert goals_conceded_points("GKP", 1) == 0
    assert goals_conceded_points("DEF", 4) == -2
    assert goals_conceded_points("MID", 8) == 0
    assert goals_conceded_points("FWD", 8) == 0


def test_defenders_get_two_points_at_ten_cbit_but_recoveries_do_not_count() -> None:
    assert defensive_contribution_points("DEF", clearances=3, blocks=2, interceptions=2, tackles=3) == 2
    assert defensive_contribution_points("DEF", clearances=9, ball_recoveries=20) == 0
    assert POSITION_SCORING[Position.DEFENDER].defensive_contribution_actions == {
        DefensiveAction.CLEARANCE,
        DefensiveAction.BLOCK,
        DefensiveAction.INTERCEPTION,
        DefensiveAction.TACKLE,
    }


@pytest.mark.parametrize("position", [Position.MIDFIELDER, Position.FORWARD])
def test_midfielders_and_forwards_get_two_points_at_twelve_cbirt(position: Position) -> None:
    assert defensive_contribution_points(position, tackles=2, ball_recoveries=10) == 2
    assert defensive_contribution_points(position, tackles=2, ball_recoveries=9) == 0


def test_goalkeepers_are_not_eligible_for_defensive_contribution_points() -> None:
    assert defensive_contribution_points("GKP", clearances=99, tackles=99) == 0


def test_announced_2026_27_bps_changes_are_explicit() -> None:
    assert BONUS_POINTS.cbi_actions_per_bps == 3
    assert BONUS_POINTS.any_goalkeeper_save_bps == 2
    assert BONUS_POINTS.inside_box_save_extra_bps == 1
    assert BONUS_POINTS.big_chance_save_extra_bps == 1
    assert BONUS_POINTS.penalty_save_bps == 7
    assert BONUS_POINTS.tackled_metric_active is False


def test_transfer_hits_and_banked_free_transfers() -> None:
    assert TRANSFERS.max_banked_free_transfers == 5
    assert transfer_points_cost(3, 2) == 4
    assert transfer_points_cost(5, 2) == 12
    assert transfer_points_cost(15, 1, active_chip=Chip.WILDCARD) == 0
    assert transfer_points_cost(15, 1, active_chip=Chip.FREE_HIT) == 0
    assert free_transfers_next_gameweek(4, 0) == 5
    assert free_transfers_next_gameweek(4, 1) == 4
    assert free_transfers_next_gameweek(4, 8) == 1
    assert free_transfers_next_gameweek(4, 12, active_chip=Chip.WILDCARD) == 4
    assert free_transfers_next_gameweek(4, 12, active_chip=Chip.FREE_HIT) == 4


def test_selling_price_shares_only_half_of_profit_rounded_down() -> None:
    assert selling_price_tenths(70, 69) == 69
    assert selling_price_tenths(70, 70) == 70
    assert selling_price_tenths(70, 71) == 70
    assert selling_price_tenths(70, 72) == 71
    assert selling_price_tenths(70, 75) == 72


def test_two_complete_chip_sets_and_gameweek_windows() -> None:
    assert set(CHIP_RULES) == set(Chip)
    assert CHIPS.total_chips == 8
    assert all(len(rule.windows) == 2 for rule in CHIP_RULES.values())
    assert chip_window(Chip.BENCH_BOOST, 1).half == 1
    assert chip_window(Chip.TRIPLE_CAPTAIN, 1).half == 1
    assert chip_window(Chip.WILDCARD, 1) is None
    assert chip_window(Chip.FREE_HIT, 1) is None
    assert chip_window(Chip.WILDCARD, 19).half == 1
    assert chip_window(Chip.WILDCARD, 20).half == 2
    assert chip_window(Chip.FREE_HIT, 38).half == 2


def test_chip_usage_constraints() -> None:
    assert can_play_chip(Chip.WILDCARD, 10)
    assert not can_play_chip(Chip.WILDCARD, 10, used_halves={1})
    assert not can_play_chip(Chip.BENCH_BOOST, 10, another_chip_selected=True)
    assert not can_play_chip(Chip.FREE_HIT, 20, previous_gameweek_chip=Chip.FREE_HIT)
    assert can_play_chip(Chip.FREE_HIT, 20, previous_gameweek_chip=Chip.WILDCARD)


def _valid_squad() -> list[SquadPlayer]:
    positions = (
        [Position.GOALKEEPER] * 2
        + [Position.DEFENDER] * 5
        + [Position.MIDFIELDER] * 5
        + [Position.FORWARD] * 3
    )
    return [
        SquadPlayer(player_id=index, team_id=((index - 1) // 3) + 1, position=position, price_tenths=60)
        for index, position in enumerate(positions, start=1)
    ]


def test_squad_validator_accepts_a_legal_initial_squad() -> None:
    players = _valid_squad()
    assert Counter(player.position for player in players) == Counter(SQUAD.position_quotas)
    assert validate_squad(players) == ()


def test_squad_validator_reports_quota_club_budget_and_duplicates() -> None:
    players = _valid_squad()
    players[1] = SquadPlayer(
        player_id=players[0].player_id,
        team_id=2,
        position=Position.DEFENDER,
        price_tenths=200,
    )
    errors = validate_squad(players)
    assert "Squad cannot contain duplicate players" in errors
    assert "Squad must contain exactly 2 GKP players" in errors
    assert "Squad must contain exactly 5 DEF players" in errors
    assert "Squad can contain at most 3 players per club" in errors
    assert "Squad cannot cost more than 1000 tenths" in errors


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: appearance_points(-1), "Minutes cannot be negative"),
        (lambda: goalkeeper_save_points(-1), "Saves cannot be negative"),
        (lambda: goals_conceded_points("DEF", -1), "Goals conceded cannot be negative"),
        (lambda: transfer_points_cost(-1, 1), "Transfer counts cannot be negative"),
        (lambda: selling_price_tenths(-1, 50), "Prices cannot be negative"),
    ],
)
def test_invalid_counts_are_rejected(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
