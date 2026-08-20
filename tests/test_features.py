import pandas as pd

from logic.features import (
    expected_points_for_player,
    gameweek_window,
    get_dgw_events,
    next_n_fixtures_for_team,
)


def _fixtures(rows):
    return pd.DataFrame(
        rows,
        columns=["event", "home_team", "away_team", "home_fdr", "away_fdr", "kickoff_time"],
    )


def _player(team_id=1, status="a"):
    return pd.Series(
        {
            "team_id": team_id,
            "singular_name_short": "MID",
            "status": status,
            "points_per_game": 5.0,
            "form": 5.0,
            "minutes": 900,
            "expected_goals": 2.0,
            "expected_assists": 2.0,
            "expected_goals_conceded": 10.0,
            "bps": 100,
            "ict_index": 100,
        }
    )


def test_gameweek_window_is_global_and_consecutive():
    fixtures = _fixtures(
        [
            (1, 1, 2, 2, 3, "2026-08-15T14:00:00Z"),
            (2, 3, 4, 2, 3, "2026-08-22T14:00:00Z"),
            (3, 1, 3, 2, 3, "2026-08-29T14:00:00Z"),
            (4, 1, 4, 2, 3, "2026-09-12T14:00:00Z"),
        ]
    )

    assert gameweek_window(fixtures, n=3) == [1, 2, 3]
    assert [f["event"] for f in next_n_fixtures_for_team(fixtures, team_id=1, n=3)] == [1, 3]


def test_expected_points_preserve_blank_gameweek_as_zero():
    fixtures = _fixtures(
        [
            (1, 1, 2, 2, 3, "2026-08-15T14:00:00Z"),
            (2, 3, 4, 2, 3, "2026-08-22T14:00:00Z"),
            (3, 1, 3, 2, 3, "2026-08-29T14:00:00Z"),
            (4, 1, 4, 2, 3, "2026-09-12T14:00:00Z"),
        ]
    )

    forecast = expected_points_for_player(_player(), fixtures, n=3)

    assert [row["event"] for row in forecast["per_gw"]] == [1, 2, 3]
    assert forecast["per_gw"][1] == {
        "event": 2,
        "ep": 0.0,
        "fixtures_count": 0,
        "is_dgw": False,
        "is_blank": True,
    }
    assert forecast["forecast_method"] == "heuristic_baseline"
    assert forecast["is_experimental"] is True


def test_double_gameweek_keeps_both_fixtures_inside_window():
    fixtures = _fixtures(
        [
            (5, 1, 2, 2, 3, "2026-09-19T14:00:00Z"),
            (5, 3, 1, 3, 2, "2026-09-22T18:45:00Z"),
            (6, 2, 3, 3, 3, "2026-09-26T14:00:00Z"),
        ]
    )

    forecast = expected_points_for_player(_player(), fixtures, n=2)

    assert get_dgw_events(fixtures, team_id=1, n_events=2) == [5]
    assert forecast["per_gw"][0]["fixtures_count"] == 2
    assert forecast["per_gw"][0]["is_dgw"] is True
    assert forecast["per_gw"][1]["is_blank"] is True


def test_unavailable_player_keeps_fixture_shape_but_scores_zero():
    fixtures = _fixtures([(8, 1, 2, 2, 3, "2026-10-17T14:00:00Z")])

    forecast = expected_points_for_player(_player(status="i"), fixtures, n=1)

    assert forecast["per_gw"][0]["fixtures_count"] == 1
    assert forecast["per_gw"][0]["is_blank"] is False
    assert forecast["per_gw"][0]["ep"] == 0.0
