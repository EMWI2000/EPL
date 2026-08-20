import pandas as pd

from logic.captain import captain_score, get_captain_recommendations, vice_captain_recommendation


def test_captain_score_does_not_double_count_features():
    row = pd.Series(
        {
            "ep_next_gw": 7.25,
            "is_home": True,
            "penalties_order": 1,
            "form": 10.0,
            "next_fdr": 1,
            "is_dgw": True,
        }
    )

    assert captain_score(row) == 7.25
    details = captain_score(row, include_details=True)
    assert details["final_score"] == 7.25
    assert details["method"] == "projected_points"


def test_recommendations_rank_by_projected_points():
    players = pd.DataFrame(
        [
            {"id": 1, "name": "A", "ep_next_gw": 6.0, "is_home": True},
            {"id": 2, "name": "B", "ep_next_gw": 7.0, "is_home": False},
        ]
    )

    result = get_captain_recommendations(players, top_n=2)

    assert result["id"].tolist() == [2, 1]


def test_vice_captain_uses_best_remaining_projection():
    players = pd.DataFrame(
        [
            {"id": 1, "team_id": 10, "ep_next_gw": 8.0},
            {"id": 2, "team_id": 10, "ep_next_gw": 7.5},
            {"id": 3, "team_id": 11, "ep_next_gw": 6.0},
        ]
    )

    vice = vice_captain_recommendation(players, captain_id=1)

    assert vice is not None
    assert int(vice["id"]) == 2
