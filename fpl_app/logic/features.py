# logic/features.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from services.odds import attack_def_factors

# FDR fallback-faktorer (bruges hvis ingen odds)
FDR_FACTOR = {1: 1.30, 2: 1.15, 3: 1.00, 4: 0.88, 5: 0.75}

# Synonym-ordbog til holdnavne (FPL -> Odds)
TEAM_SYNONYMS: Dict[str, str] = {
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Spurs": "Tottenham Hotspur",
    "Tottenham": "Tottenham Hotspur",
    "Wolves": "Wolverhampton Wanderers",
    "Wolverhampton": "Wolverhampton Wanderers",
    "West Ham": "West Ham United",
    "Brighton": "Brighton & Hove Albion",
    "Sheffield Utd": "Sheffield United",
    "Nott'm Forest": "Nottingham Forest",
    "Leeds": "Leeds United",
    "Newcastle": "Newcastle United",
    "Leicester": "Leicester City"
}

# Point-værdier for forskellige handlinger per position
POINTS_CONFIG = {
    "GKP": {"goal": 6, "assist": 3, "cs": 4, "goals_conceded_penalty": -0.5},
    "DEF": {"goal": 6, "assist": 3, "cs": 4, "goals_conceded_penalty": -0.5},
    "MID": {"goal": 5, "assist": 3, "cs": 1, "goals_conceded_penalty": 0},
    "FWD": {"goal": 4, "assist": 3, "cs": 0, "goals_conceded_penalty": 0},
}


def elements_df(bootstrap: Dict[str, Any]) -> pd.DataFrame:
    """Konverterer bootstrap elements til DataFrame med alle relevante kolonner."""
    el = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])[["id", "name", "short_name"]].rename(columns={"id": "team_id"})
    types = pd.DataFrame(bootstrap["element_types"])[["id", "singular_name_short"]].rename(
        columns={"id": "element_type"})
    df = el.merge(teams, left_on="team", right_on="team_id", how="left") \
        .merge(types, on="element_type", how="left")

    # Konverter numeriske kolonner
    numeric_cols = [
        "form", "points_per_game", "now_cost", "minutes", "goals_scored", "assists",
        "clean_sheets", "goals_conceded", "bonus", "bps", "ict_index", "influence",
        "creativity", "threat", "expected_goals", "expected_assists",
        "expected_goal_involvements", "expected_goals_conceded",
        "transfers_in_event", "transfers_out_event", "selected_by_percent",
        "penalties_order", "corners_and_indirect_freekicks_order", "direct_freekicks_order"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Normaliser strings
    for c in ["name", "web_name", "short_name", "singular_name_short"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df


def fixtures_df(fixts: List[Dict[str, Any]]) -> pd.DataFrame:
    """Konverterer liste af fixtures til DataFrame med DGW-flag."""
    rows = []
    for f in fixts:
        ev = f.get("event")
        if ev is None:
            ev_num = 0
        else:
            ev_num = int(ev)
        rows.append({
            "event": ev_num,
            "home_team": int(f["team_h"]),
            "away_team": int(f["team_a"]),
            "home_fdr": int(f["team_h_difficulty"]),
            "away_fdr": int(f["team_a_difficulty"]),
            "unscheduled": bool(ev is None),
            "kickoff_time": f.get("kickoff_time", ""),
        })
    return pd.DataFrame(rows)


def count_fixtures_in_gw(fixtures: pd.DataFrame, team_id: int, event: int) -> int:
    """Tæller kampe for et hold i en given GW (1 = normal, 2+ = DGW)."""
    if event == 0:
        return 0
    mask = ((fixtures["home_team"] == team_id) | (fixtures["away_team"] == team_id)) & (fixtures["event"] == event)
    return len(fixtures[mask])


def get_dgw_events(fixtures: pd.DataFrame, team_id: int, n_events: int = 5) -> List[int]:
    """Returnerer liste af events hvor holdet har DGW (2+ kampe)."""
    mask = (fixtures["home_team"] == team_id) | (fixtures["away_team"] == team_id)
    team_fixt = fixtures.loc[mask].copy()
    events = sorted([e for e in team_fixt["event"].unique() if e != 0])[:n_events]
    return [e for e in events if count_fixtures_in_gw(fixtures, team_id, e) >= 2]


def next_n_fixtures_for_team(fixtures: pd.DataFrame, team_id: int, n: int = 5) -> List[Dict[str, Any]]:
    """
    Henter alle kampe i de næste N runder for et givet hold.
    Returnerer liste af dicts med event, fdr, is_home, opponent_team_id.
    Håndterer DGW (flere kampe i samme runde).
    """
    mask = (fixtures["home_team"] == team_id) | (fixtures["away_team"] == team_id)
    team_fixt = fixtures.loc[mask].copy()
    team_fixt = team_fixt.sort_values("event")

    # Få de næste N unikke runder (ekskluder event=0)
    upcoming_events = [e for e in team_fixt["event"].unique() if e != 0][:n]

    out = []
    for e in upcoming_events:
        games = team_fixt[team_fixt["event"] == e]
        for _, r in games.iterrows():
            is_home = int(r["home_team"]) == int(team_id)
            opponent_id = int(r["away_team"]) if is_home else int(r["home_team"])
            fdr = int(r["home_fdr"]) if is_home else int(r["away_fdr"])
            out.append({
                "event": int(r["event"]),
                "fdr": fdr,
                "is_home": is_home,
                "opponent_team_id": opponent_id,
            })
    return out


def next_n_fdrs_for_team(fixtures: pd.DataFrame, team_id: int, n: int = 5) -> List[Tuple[int, int, bool]]:
    """Legacy funktion - returnerer (event, fdr, is_home) tuples."""
    fixt_list = next_n_fixtures_for_team(fixtures, team_id, n)
    return [(f["event"], f["fdr"], f["is_home"]) for f in fixt_list]


def fixture_run_quality(fdrs: List[int], weights: Optional[List[float]] = None) -> float:
    """
    Beregner kvaliteten af en fixture-run med vægtning (tidlige kampe vigtigere).
    Højere = bedre fixtures.
    """
    if not fdrs:
        return 1.0
    if weights is None:
        weights = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5][:len(fdrs)]

    score = sum(FDR_FACTOR.get(fdr, 1.0) * w for fdr, w in zip(fdrs, weights))
    return score / sum(weights[:len(fdrs)])


def availability_multiplier(status: str) -> float:
    """Returnerer multiplier baseret på spillerstatus."""
    if status in ("i", "s"):
        return 0.0  # injured/suspended
    if status == "d":
        return 0.5  # doubtful
    if status == "u":
        return 0.75  # unlikely
    return 1.0


def position_cs_bonus(pos_short: str) -> float:
    """Returnerer CS bonus for position."""
    return POINTS_CONFIG.get(pos_short, {}).get("cs", 0)


def normalize_team_name(name: str) -> str:
    """Normaliserer holdnavn til odds-kontekst."""
    n = str(name).strip()
    return TEAM_SYNONYMS.get(n, n)


def price_change_indicator(player_row: pd.Series) -> float:
    """
    Estimerer sandsynlighed for prisstigning baseret på net transfers.
    Returnerer multiplier (>1 = bør købes nu, <1 = undgå).
    """
    transfers_in = float(player_row.get("transfers_in_event", 0) or 0)
    transfers_out = float(player_row.get("transfers_out_event", 0) or 0)
    net = transfers_in - transfers_out

    if net > 100000:
        return 1.08
    elif net > 50000:
        return 1.05
    elif net > 20000:
        return 1.02
    elif net < -50000:
        return 0.95
    elif net < -20000:
        return 0.98
    return 1.0


def calculate_base_ep(player_row: pd.Series, pos: str) -> float:
    """
    Beregner base expected points baseret på xG, xA, ICT og historisk form.
    Meget mere avanceret end den simple PPG+form model.
    """
    # Hent stats fra player_row
    ppg = float(player_row.get("points_per_game", 0) or 0)
    form = float(player_row.get("form", 0) or 0)
    minutes = float(player_row.get("minutes", 0) or 0)

    # Expected stats (kumuleret over sæsonen)
    xG = float(player_row.get("expected_goals", 0) or 0)
    xA = float(player_row.get("expected_assists", 0) or 0)
    xGI = float(player_row.get("expected_goal_involvements", 0) or 0)
    xGC = float(player_row.get("expected_goals_conceded", 0) or 0)

    # ICT index komponenter
    ict = float(player_row.get("ict_index", 0) or 0)
    threat = float(player_row.get("threat", 0) or 0)
    creativity = float(player_row.get("creativity", 0) or 0)
    influence = float(player_row.get("influence", 0) or 0)

    # BPS (bonus point system)
    bps = float(player_row.get("bps", 0) or 0)

    # Beregn antal kampe spillet
    games_played = minutes / 90.0 if minutes > 0 else 0

    # Minimum spilletid for at være relevant
    if games_played < 1.5:
        # For nye/skadede spillere, brug form hvis tilgængelig
        if form > 0:
            return form * 0.8
        return ppg * 0.5

    # Per-game stats
    xG_per_game = xG / games_played if games_played > 0 else 0
    xA_per_game = xA / games_played if games_played > 0 else 0
    xGC_per_game = xGC / games_played if games_played > 0 else 0
    bps_per_game = bps / games_played if games_played > 0 else 0
    ict_per_game = ict / games_played if games_played > 0 else 0

    # Point config for position
    cfg = POINTS_CONFIG.get(pos, POINTS_CONFIG["MID"])

    # Base beregning afhængig af position
    if pos == "FWD":
        # Forwards: xG og xA er kritiske
        base = (
                cfg["goal"] * xG_per_game +  # Forventede mål
                cfg["assist"] * xA_per_game +  # Forventede assists
                0.015 * ict_per_game +  # ICT bonus
                0.008 * bps_per_game +  # BPS bonus
                2.0  # Base for at spille
        )
    elif pos == "MID":
        # Midfielders: Balance mellem offense og clean sheets
        cs_contribution = max(0, 1 - xGC_per_game * 0.5) * cfg["cs"] * 0.25
        base = (
                cfg["goal"] * xG_per_game +
                cfg["assist"] * xA_per_game +
                cs_contribution +
                0.012 * ict_per_game +
                0.008 * bps_per_game +
                2.0
        )
    elif pos == "DEF":
        # Defenders: CS og assists vigtigere
        cs_prob = max(0, 0.35 - xGC_per_game * 0.15)  # Estimeret CS sandsynlighed
        base = (
                cfg["goal"] * xG_per_game +
                cfg["assist"] * xA_per_game +
                cfg["cs"] * cs_prob +  # Clean sheet forventning
                cfg["goals_conceded_penalty"] * max(0, xGC_per_game - 2) +
                0.008 * bps_per_game +
                2.0
        )
    else:  # GKP
        # Goalkeepers: CS og saves
        cs_prob = max(0, 0.30 - xGC_per_game * 0.12)
        saves_bonus = min(0.5, bps_per_game * 0.005)  # BPS som proxy for saves
        base = (
                cfg["cs"] * cs_prob +
                saves_bonus +
                cfg["goals_conceded_penalty"] * max(0, xGC_per_game - 2) +
                2.0
        )

    # Blend med historisk form (50% model, 50% faktisk performance)
    if form > 0:
        base = 0.5 * base + 0.5 * form
    elif ppg > 0:
        base = 0.6 * base + 0.4 * ppg

    return max(base, 0.5)


def expected_points_for_player(
        player_row: pd.Series,
        fixtures: pd.DataFrame,
        n: int = 5,
        odds_ctx_by_fixture: Optional[Dict[Tuple[int, int, int], Dict[str, float]]] = None,
        teams_table: Optional[pd.DataFrame] = None,
        use_ml: bool = True,
) -> Dict[str, Any]:
    """
    Avanceret EP-beregning for en spiller over de næste n runder.

    Features:
    - xG/xA-baseret base beregning
    - DGW-detektion (dobbelt point for dobbelt-runder)
    - Odds-justering (hvis tilgængelig)
    - Position-specifik beregning
    - Form og ICT index integration

    Returns:
        Dict med 'per_gw' (liste af {event, ep, fixtures_count}) og 'total_next_n'
    """
    # Forsøg ML-model først
    if use_ml:
        try:
            from ml.predict import model_available, predict_single_player_multi_gw
            if model_available():
                return predict_single_player_multi_gw(player_row, fixtures, n, teams_table)
        except Exception:
            pass  # Fald tilbage til heuristik

    status = str(player_row.get("status", "a"))
    avail = availability_multiplier(status)

    if avail == 0.0:
        return {"per_gw": [], "total_next_n": 0.0, "has_dgw": False, "dgw_events": []}

    pos = str(player_row.get("singular_name_short", "MID"))
    team_id = int(player_row.get("team_id", player_row.get("team", 0)))

    # Beregn base EP med avanceret model
    base = calculate_base_ep(player_row, pos)

    # Hent kommende kampe
    upcoming = next_n_fixtures_for_team(fixtures, team_id, n=n)

    if not upcoming:
        return {"per_gw": [], "total_next_n": 0.0, "has_dgw": False, "dgw_events": []}

    # Gruppér kampe per event (for DGW håndtering)
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
            fdr = fix["fdr"]
            is_home = fix["is_home"]
            opponent_id = fix["opponent_team_id"]

            # FDR multiplier
            fdr_mult = FDR_FACTOR.get(int(fdr), 1.0)

            # Hjemmebane boost
            home_mult = 1.08 if is_home else 1.00

            # Start med base * adjustments
            ep = base * fdr_mult * home_mult
            cs_expected = 0.0

            # Odds-justering hvis tilgængelig
            if odds_ctx_by_fixture is not None:
                # Find kamp i fixtures
                if is_home:
                    h, a = team_id, opponent_id
                else:
                    h, a = opponent_id, team_id

                oc = odds_ctx_by_fixture.get((event, h, a))
                if oc:
                    pH = oc.get("pH", 0.0)
                    pD = oc.get("pD", 0.0)
                    pA = oc.get("pA", 0.0)
                    pOver25 = oc.get("pOver25", 0.0)

                    attack, cs_prob, def_factor = attack_def_factors(pH, pD, pA, pOver25, is_home=is_home)

                    if pos in ("MID", "FWD"):
                        ep = base * home_mult * max(0.7, attack)
                    elif pos in ("GKP", "DEF"):
                        ep = base * home_mult * max(0.7, def_factor)
                        cs_expected = position_cs_bonus(pos) * cs_prob

            ep = (ep + cs_expected) * avail
            event_ep += max(float(ep), 0.0)

        per_gw.append({
            "event": int(event),
            "ep": round(event_ep, 2),
            "fixtures_count": fixtures_count,
            "is_dgw": fixtures_count >= 2
        })

    total = float(sum(x["ep"] for x in per_gw))

    return {
        "per_gw": per_gw,
        "total_next_n": total,
        "has_dgw": len(dgw_events) > 0,
        "dgw_events": dgw_events
    }


def get_player_detailed_stats(player_row: pd.Series, fixtures: pd.DataFrame, n: int = 5) -> Dict[str, Any]:
    """
    Returnerer detaljerede stats for en spiller til visning i UI.
    """
    team_id = int(player_row.get("team_id", player_row.get("team", 0)))
    pos = str(player_row.get("singular_name_short", "MID"))

    # Kommende fixtures
    upcoming = next_n_fixtures_for_team(fixtures, team_id, n=n)
    fdrs = [f["fdr"] for f in upcoming]

    # DGW info
    dgw_events = get_dgw_events(fixtures, team_id, n)

    # Transfer trend
    transfers_in = float(player_row.get("transfers_in_event", 0) or 0)
    transfers_out = float(player_row.get("transfers_out_event", 0) or 0)
    net_transfers = transfers_in - transfers_out

    # Set piece info
    penalty_order = int(player_row.get("penalties_order", 99) or 99)
    corner_order = int(player_row.get("corners_and_indirect_freekicks_order", 99) or 99)
    fk_order = int(player_row.get("direct_freekicks_order", 99) or 99)

    return {
        "fixture_run_quality": round(fixture_run_quality(fdrs), 2),
        "fdrs": fdrs,
        "has_dgw": len(dgw_events) > 0,
        "dgw_events": dgw_events,
        "net_transfers": int(net_transfers),
        "is_rising": net_transfers > 50000,
        "is_falling": net_transfers < -50000,
        "is_penalty_taker": penalty_order == 1,
        "is_set_piece_taker": corner_order <= 2 or fk_order == 1,
        "xG": float(player_row.get("expected_goals", 0) or 0),
        "xA": float(player_row.get("expected_assists", 0) or 0),
        "ict_index": float(player_row.get("ict_index", 0) or 0),
        "form": float(player_row.get("form", 0) or 0),
        "selected_by_percent": float(player_row.get("selected_by_percent", 0) or 0),
    }
