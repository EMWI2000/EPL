import pandas as pd

from logic.optimizer import FORMATIONS, find_best_formation


def test_optimizer_exposes_all_legal_formations_including_523():
    assert set(FORMATIONS) == {"343", "352", "433", "442", "451", "523", "532", "541"}


def test_best_formation_can_choose_523():
    rows = []
    player_id = 1
    for position, scores in {
        "GKP": [5, 1],
        "DEF": [10, 10, 10, 10, 10],
        "MID": [8, 8, 1, 1, 1],
        "FWD": [9, 9, 9],
    }.items():
        for score in scores:
            rows.append(
                {
                    "id": player_id,
                    "team_id": ((player_id - 1) % 8) + 1,
                    "pos": position,
                    "ep_next_gw": score,
                }
            )
            player_id += 1

    formation, starting_indices, expected_points = find_best_formation(pd.DataFrame(rows))

    assert formation == "523"
    assert len(starting_indices) == 11
    assert expected_points == 98
