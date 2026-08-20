import pandas as pd

from services import data_layer


def _ep(*args, **kwargs):
    return {"per_gw": [], "total_next_n": 0.0, "has_dgw": False}


def test_build_my_team_uses_fpl_selling_price(monkeypatch):
    monkeypatch.setattr(data_layer, "expected_points_for_player", _ep)
    players = pd.DataFrame(
        [
            {
                "id": 10,
                "web_name": "Owned",
                "team_id": 1,
                "short_name": "ONE",
                "singular_name_short": "MID",
                "status": "a",
                "now_cost": 80,
                "form": 0,
            }
        ]
    )
    picks = {
        "picks": [
            {"element": 10, "purchase_price": 70, "selling_price": 75, "position": 1}
        ]
    }

    result = data_layer.build_my_team_df(
        players,
        pd.DataFrame(),
        picks,
        horizon=3,
        odds_ctx_by_fixture=None,
        teams_df=pd.DataFrame(),
    )

    assert result.loc[0, "now_cost"] == 80
    assert result.loc[0, "purchase_price"] == 70
    assert result.loc[0, "sell_price"] == 75
    assert bool(result.loc[0, "sell_price_estimated"]) is False


def test_build_my_team_labels_current_price_fallback(monkeypatch):
    monkeypatch.setattr(data_layer, "expected_points_for_player", _ep)
    players = pd.DataFrame(
        [
            {
                "id": 10,
                "web_name": "Owned",
                "team_id": 1,
                "short_name": "ONE",
                "singular_name_short": "MID",
                "status": "a",
                "now_cost": 80,
                "form": 0,
            }
        ]
    )

    result = data_layer.build_my_team_df(
        players,
        pd.DataFrame(),
        {"picks": [{"element": 10, "position": 1}]},
        horizon=3,
        odds_ctx_by_fixture=None,
        teams_df=pd.DataFrame(),
    )

    assert result.loc[0, "sell_price"] == 80
    assert bool(result.loc[0, "sell_price_estimated"]) is True
