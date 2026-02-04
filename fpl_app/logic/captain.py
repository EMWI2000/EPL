# logic/captain.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Any, Optional



def captain_score(row: pd.Series, include_details: bool = False) -> float | Dict[str, Any]:
    """
    Beregner en avanceret score for kaptajnvalg baseret på:
    - Forventede point (næste kamp)
    - Hjemmebane
    - Penalty-tager status
    - Set piece tager status
    - Form
    - Modstander-styrke (FDR)

    Args:
        row: Series med spillerdata
        include_details: Hvis True, returnerer dict med score og breakdown

    Returns:
        Float score eller dict med detaljer
    """
    base_ep = float(row.get("ep_next_gw", 0.0))
    score = base_ep

    breakdown = {
        "base_ep": base_ep,
        "boosts": []
    }

    # 1) Hjemmebaneboost (8%)
    if row.get("is_home", False):
        score *= 1.08
        breakdown["boosts"].append(("Hjemmebane", "+8%"))

    # 2) Penalty-taker boost (15% - STOR faktor)
    penalty_order = row.get("penalties_order", 99)
    if penalty_order is not None:
        try:
            if int(penalty_order) == 1:
                score *= 1.15
                breakdown["boosts"].append(("Penalty-tager #1", "+15%"))
            elif int(penalty_order) == 2:
                score *= 1.05
                breakdown["boosts"].append(("Penalty-tager #2", "+5%"))
        except (ValueError, TypeError):
            pass

    # 3) Corner/indirect free kick boost
    corner_order = row.get("corners_and_indirect_freekicks_order", 99)
    if corner_order is not None:
        try:
            if int(corner_order) == 1:
                score *= 1.06
                breakdown["boosts"].append(("Corner-tager #1", "+6%"))
            elif int(corner_order) == 2:
                score *= 1.03
                breakdown["boosts"].append(("Corner-tager #2", "+3%"))
        except (ValueError, TypeError):
            pass

    # 4) Direct free kick boost
    fk_order = row.get("direct_freekicks_order", 99)
    if fk_order is not None:
        try:
            if int(fk_order) == 1:
                score *= 1.04
                breakdown["boosts"].append(("Frisparks-tager", "+4%"))
        except (ValueError, TypeError):
            pass

    # 5) Form boost (spillere i god form)
    form = float(row.get("form", 0) or 0)
    if form >= 8.0:
        score *= 1.12
        breakdown["boosts"].append((f"Exceptionel form ({form:.1f})", "+12%"))
    elif form >= 6.5:
        score *= 1.08
        breakdown["boosts"].append((f"God form ({form:.1f})", "+8%"))
    elif form >= 5.0:
        score *= 1.04
        breakdown["boosts"].append((f"Okay form ({form:.1f})", "+4%"))
    elif form < 3.0 and form > 0:
        score *= 0.92
        breakdown["boosts"].append((f"Dårlig form ({form:.1f})", "-8%"))

    # 6) Modstander-styrke (FDR) - hvis tilgængelig
    opp_fdr = row.get("opponent_fdr", row.get("next_fdr", None))
    if opp_fdr is not None:
        try:
            fdr = int(opp_fdr)
            if fdr == 1:
                score *= 1.15
                breakdown["boosts"].append(("Nem modstander (FDR 1)", "+15%"))
            elif fdr == 2:
                score *= 1.08
                breakdown["boosts"].append(("Let modstander (FDR 2)", "+8%"))
            elif fdr == 4:
                score *= 0.95
                breakdown["boosts"].append(("Svær modstander (FDR 4)", "-5%"))
            elif fdr == 5:
                score *= 0.88
                breakdown["boosts"].append(("Meget svær modstander (FDR 5)", "-12%"))
        except (ValueError, TypeError):
            pass

    # 7) DGW boost (dobbelt gameweek)
    if row.get("is_dgw", False) or row.get("fixtures_count", 1) >= 2:
        score *= 1.25
        breakdown["boosts"].append(("Dobbelt Gameweek", "+25%"))

    # 8) ICT Index boost (høj threat/creativity)
    threat = float(row.get("threat", row.get("threat_norm", 0)) or 0)
    if threat > 100:  # Høj threat over sæsonen
        score *= 1.05
        breakdown["boosts"].append(("Høj threat-index", "+5%"))

    # 9) Setpiece boost (legacy support)
    if row.get("setpiece_boost", 0):
        score *= 1.05
        breakdown["boosts"].append(("Set piece boost", "+5%"))

    breakdown["final_score"] = round(score, 2)

    if include_details:
        return breakdown
    return score


def get_captain_recommendations(
        players_df: pd.DataFrame,
        top_n: int = 5,
        min_ep: float = 3.0
) -> pd.DataFrame:
    """
    Returnerer top kaptajn-kandidater med detaljeret scoring.

    Args:
        players_df: DataFrame med spillere (skal have ep_next_gw)
        top_n: Antal kandidater at returnere
        min_ep: Minimum EP for at være kandidat

    Returns:
        DataFrame med top kandidater og scoring detaljer
    """
    df = players_df.copy()

    # Filtrér spillere med for lav EP
    df = df[df.get("ep_next_gw", 0) >= min_ep]

    if df.empty:
        return pd.DataFrame()

    # Beregn captain score med detaljer
    scores = []
    for idx, row in df.iterrows():
        details = captain_score(row, include_details=True)
        scores.append({
            "idx": idx,
            "cap_score": details["final_score"],
            "boosts": ", ".join([f"{b[0]}" for b in details["boosts"]]) if details["boosts"] else "Ingen"
        })

    scores_df = pd.DataFrame(scores).set_index("idx")
    df = df.join(scores_df)

    # Sortér og returner top N
    df = df.sort_values("cap_score", ascending=False).head(top_n)

    return df


def vice_captain_recommendation(
        players_df: pd.DataFrame,
        captain_id: int
) -> Optional[pd.Series]:
    """
    Anbefaler vice-kaptajn (bedste efter kaptajn, gerne fra andet hold).

    Args:
        players_df: DataFrame med spillere
        captain_id: ID på valgt kaptajn

    Returns:
        Series med anbefalet vice-kaptajn eller None
    """
    df = players_df.copy()

    # Fjern kaptajnen
    df = df[df["id"] != captain_id]

    if df.empty:
        return None

    # Beregn scores
    df["cap_score"] = df.apply(captain_score, axis=1)

    # Få kaptajnens hold
    captain_row = players_df[players_df["id"] == captain_id]
    if not captain_row.empty:
        captain_team = captain_row.iloc[0].get("team_id", None)

        # Foretruk vice fra andet hold (reducerer risiko)
        other_team = df[df["team_id"] != captain_team]
        if not other_team.empty:
            return other_team.sort_values("cap_score", ascending=False).iloc[0]

    # Ellers returner bare bedste
    return df.sort_values("cap_score", ascending=False).iloc[0]
