# ml/predict.py
"""
Runtime prediction modul til FPL EP-model.
Bruger den prætræende LightGBM model til at forudsige spillers forventede point.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "ep_model.joblib"
FEATURES_PATH = MODEL_DIR / "feature_columns.joblib"
META_PATH = MODEL_DIR / "model_meta.json"

_model_cache: Dict[str, Any] = {}


def model_available() -> bool:
    """Tjek om modellen er tilgængelig."""
    return MODEL_PATH.exists() and FEATURES_PATH.exists()


def load_model():
    """Load model og features (cached)."""
    if "model" not in _model_cache:
        import joblib
        _model_cache["model"] = joblib.load(MODEL_PATH)
        _model_cache["features"] = joblib.load(FEATURES_PATH)
    return _model_cache["model"], _model_cache["features"]


def load_meta() -> Dict[str, Any]:
    """Load model metadata."""
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {}


def prepare_features_for_player(
    player_row: pd.Series,
    fixtures_data: pd.DataFrame,
    fixture_info: Dict[str, Any],
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Konverterer en spillers FPL API-data til feature-vektor som modellen forventer.

    Args:
        player_row: Series med spillerdata fra FPL API (elements_df)
        fixtures_data: Fixtures DataFrame
        fixture_info: Dict med {event, fdr, is_home, opponent_team_id}
        feature_cols: Liste af feature-navne modellen forventer
    """
    minutes = float(player_row.get("minutes", 0) or 0)
    games_played = max(minutes / 90.0, 0.5)

    # Base stats
    features = {
        "pos_code": _pos_to_code(str(player_row.get("singular_name_short", "MID"))),
        "is_home": int(fixture_info.get("is_home", False)),
        "price": float(player_row.get("now_cost", 50)) / 10.0,
        "selected_pct": float(player_row.get("selected_by_percent", 0) or 0),
        "net_transfers": float(player_row.get("transfers_in_event", 0) or 0)
                        - float(player_row.get("transfers_out_event", 0) or 0),
        "cum_minutes": minutes,
        "opponent_team_id": int(fixture_info.get("opponent_team_id", 0)),
    }

    # Per-90 stats
    goals = float(player_row.get("goals_scored", 0) or 0)
    assists = float(player_row.get("assists", 0) or 0)
    cs = float(player_row.get("clean_sheets", 0) or 0)
    features["goals_per_90"] = goals / games_played
    features["assists_per_90"] = assists / games_played
    features["cs_rate"] = cs / games_played

    # xG per 90
    xG = float(player_row.get("expected_goals", 0) or 0)
    xA = float(player_row.get("expected_assists", 0) or 0)
    features["xG_per_90"] = xG / games_played
    features["xA_per_90"] = xA / games_played

    # Rolling features - vi bruger "form" og andre API-felter som proxy
    form = float(player_row.get("form", 0) or 0)
    ppg = float(player_row.get("points_per_game", 0) or 0)
    ict = float(player_row.get("ict_index", 0) or 0)
    bps = float(player_row.get("bps", 0) or 0)
    bonus = float(player_row.get("bonus", 0) or 0)

    # form ≈ gennemsnit af de sidste par GWs point
    for window in [3, 5]:
        w = f"_{window}gw"
        features[f"roll_pts{w}"] = form  # FPL "form" er allerede rolling avg
        features[f"roll_min{w}"] = min(minutes / max(games_played, 1), 90)
        features[f"roll_bps{w}"] = bps / games_played
        features[f"roll_ict{w}"] = ict / games_played
        features[f"roll_bonus{w}"] = bonus / games_played
        features[f"roll_xG{w}"] = xG / games_played
        features[f"roll_xA{w}"] = xA / games_played

    # Lav DataFrame med kun de kolonner modellen forventer
    row_data = {col: features.get(col, 0.0) for col in feature_cols}
    return pd.DataFrame([row_data])


def _pos_to_code(pos: str) -> int:
    """Konverterer position til numerisk kode."""
    return {"GKP": 0, "GK": 0, "DEF": 1, "MID": 2, "FWD": 3}.get(pos, 2)


def predict_single_player_multi_gw(
    player_row: pd.Series,
    fixtures_df: pd.DataFrame,
    n: int = 5,
    teams_table: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Forudsiger EP for en spiller over de næste n runder med ML-model.
    Returnerer samme format som expected_points_for_player() i features.py.
    """
    if not model_available():
        return {"per_gw": [], "total_next_n": 0.0, "has_dgw": False, "dgw_events": []}

    model, feature_cols = load_model()

    # Tjek tilgængelighed
    status = str(player_row.get("status", "a"))
    if status in ("i", "s"):
        return {"per_gw": [], "total_next_n": 0.0, "has_dgw": False, "dgw_events": []}

    avail_mult = 1.0
    if status == "d":
        avail_mult = 0.5
    elif status == "u":
        avail_mult = 0.75

    team_id = int(player_row.get("team_id", player_row.get("team", 0)))

    # Hent kommende fixtures
    from logic.features import next_n_fixtures_for_team
    upcoming = next_n_fixtures_for_team(fixtures_df, team_id, n=n)

    if not upcoming:
        return {"per_gw": [], "total_next_n": 0.0, "has_dgw": False, "dgw_events": []}

    # Gruppér per event
    events_fixtures: Dict[int, List[Dict]] = {}
    for fix in upcoming:
        ev = fix["event"]
        if ev not in events_fixtures:
            events_fixtures[ev] = []
        events_fixtures[ev].append(fix)

    per_gw = []
    dgw_events = []

    for event in sorted(events_fixtures.keys()):
        event_fixtures = events_fixtures[event]
        fixtures_count = len(event_fixtures)
        if fixtures_count >= 2:
            dgw_events.append(event)

        event_ep = 0.0
        for fix in event_fixtures:
            X = prepare_features_for_player(player_row, fixtures_df, fix, feature_cols)
            pred = model.predict(X)[0]
            # Clamp til realistisk range og apply availability
            pred = max(float(pred), 0.0) * avail_mult
            event_ep += pred

        per_gw.append({
            "event": int(event),
            "ep": round(event_ep, 2),
            "fixtures_count": fixtures_count,
            "is_dgw": fixtures_count >= 2,
        })

    total = float(sum(x["ep"] for x in per_gw))

    return {
        "per_gw": per_gw,
        "total_next_n": total,
        "has_dgw": len(dgw_events) > 0,
        "dgw_events": dgw_events,
    }
