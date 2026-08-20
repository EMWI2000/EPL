# logic/captain.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Any, Optional



def captain_score(row: pd.Series, include_details: bool = False) -> float | Dict[str, Any]:
    """Rangér en kaptajn efter den allerede beregnede næste-GW-prognose.

    Hjemmebane, form, dødbolde, modstander og DGW må indgå i selve
    pointprognosen. At gange dem på igen her giver systematisk dobbeltregning.
    """
    base_ep = float(row.get("ep_next_gw", 0.0))
    breakdown = {
        "base_ep": base_ep,
        "boosts": [],
        "final_score": round(base_ep, 2),
        "method": "projected_points",
    }

    if include_details:
        return breakdown
    return base_ep


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
    if "ep_next_gw" not in players_df.columns:
        raise ValueError("players_df skal indeholde kolonnen 'ep_next_gw'")

    df = players_df.copy()

    # Filtrér spillere med for lav EP
    df = df[df["ep_next_gw"] >= min_ep]

    if df.empty:
        return pd.DataFrame()

    # Beregn captain score med detaljer
    scores = []
    for idx, row in df.iterrows():
        details = captain_score(row, include_details=True)
        scores.append({
            "idx": idx,
            "cap_score": details["final_score"],
            "boosts": "Indregnet i pointprognosen"
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

    # Vicekaptajnen vælges også på forventede point. Holddiversifikation må kun
    # anvendes som en eksplicit risikopreference, ikke som skjult standardregel.
    return df.sort_values("cap_score", ascending=False).iloc[0]
