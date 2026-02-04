# services/data_layer.py
"""
Centraliseret data-lag for FPL-appen.
Eliminerer duplikering af picks-hentning, odds-building og hold-building.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import streamlit as st
import pandas as pd

from services.fpl_api import bootstrap_static, fixtures, entry_picks, manager_summary
from logic.features import elements_df, fixtures_df, expected_points_for_player, get_player_detailed_stats
from services.odds import epl_odds, build_odds_context, _norm_team_name, find_odds_for_pair


# ---------------------------------------------------------------------------
# Cached data-hentning
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_bootstrap() -> Dict[str, Any]:
    return bootstrap_static()


@st.cache_data(ttl=300)
def get_fixtures_df(future_only: bool = True) -> pd.DataFrame:
    return fixtures_df(fixtures(future_only=future_only))


@st.cache_data(ttl=300)
def get_elements_df(_bs_hash: str, bs: Dict[str, Any]) -> pd.DataFrame:
    return elements_df(bs)


@st.cache_data(ttl=300)
def get_teams_df(_bs_hash: str, bs: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(bs["teams"])[["id", "name", "short_name"]].rename(columns={"id": "team_id"})


def _bs_hash(bs: Dict[str, Any]) -> str:
    """Simpel hash til cache-nøgle for bootstrap data."""
    try:
        events = bs.get("events", [])
        if events:
            last = events[-1]
            return f"{last.get('id', 0)}_{last.get('finished', False)}"
    except Exception:
        pass
    return "default"


def load_base_data() -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Henter og returnerer (bootstrap, events, els, fixt, teams_df).
    Brug dette i stedet for at kalde bootstrap_static() + elements_df() + ... manuelt.
    """
    bs = get_bootstrap()
    h = _bs_hash(bs)
    events = pd.DataFrame(bs["events"])
    els = get_elements_df(h, bs)
    fixt = get_fixtures_df(future_only=True)
    teams_df = get_teams_df(h, bs)
    return bs, events, els, fixt, teams_df


# ---------------------------------------------------------------------------
# Picks med fallback
# ---------------------------------------------------------------------------

def get_picks_any(
    entry_id: int,
    preferred_gw: int,
    events_df: pd.DataFrame,
) -> Tuple[Optional[Dict[str, Any]], Optional[int], List[int]]:
    """
    Hent picks for en manager med fallback til andre GWs.
    Returnerer (picks_dict, brugt_gw, forsøgte_gws).
    """
    cands: List[int] = []
    tried: List[int] = []

    if preferred_gw:
        cands.append(int(preferred_gw))

    try:
        cands += events_df.loc[events_df["is_current"] == True, "id"].tolist()
    except Exception:
        pass

    try:
        nxt = events_df.loc[events_df["is_next"] == True, "id"].tolist()
        cands += [x for x in nxt if x not in cands]
    except Exception:
        pass

    if preferred_gw and preferred_gw > 1:
        cands.append(int(preferred_gw - 1))

    try:
        if "finished" in events_df.columns and (events_df["finished"] == True).any():
            last_fin = int(events_df.loc[events_df["finished"] == True, "id"].max())
            if last_fin not in cands:
                cands.append(last_fin)
    except Exception:
        pass

    ordered = list(dict.fromkeys(gw for gw in cands if gw is not None))

    for gw in ordered:
        try:
            p = entry_picks(int(entry_id), int(gw))
            if p and p.get("picks"):
                return p, gw, tried
            tried.append(gw)
        except Exception:
            tried.append(gw)

    return None, None, tried


# ---------------------------------------------------------------------------
# Odds-kontekst
# ---------------------------------------------------------------------------

def build_odds_context_for_fixtures(
    odds_key: str,
    fixt: pd.DataFrame,
    teams_df: pd.DataFrame,
    use_odds: bool,
) -> Tuple[Optional[Dict], str]:
    """
    Bygger odds-kontekst for fixtures.
    Returnerer (odds_ctx_by_fixture, odds_status_tekst).
    """
    if not use_odds:
        return None, "Ikke aktiveret"

    try:
        odds_payload = epl_odds(odds_key)
        if not odds_payload:
            return None, "Ingen data"

        odds_table = build_odds_context(odds_payload)
        team_name_norm = {
            int(r.team_id): _norm_team_name(r.name)
            for r in teams_df.itertuples(index=False)
        }

        odds_ctx: Dict = {}
        matched = 0
        for r in fixt.itertuples(index=False):
            Hn = team_name_norm.get(int(r.home_team), "")
            An = team_name_norm.get(int(r.away_team), "")
            oc, _ = find_odds_for_pair(odds_table, Hn, An)
            if oc:
                matched += 1
                odds_ctx[(int(r.event), int(r.home_team), int(r.away_team))] = oc

        return odds_ctx, f"Aktiv ({matched} kampe)"
    except Exception:
        return None, "Fejl ved hentning"


# ---------------------------------------------------------------------------
# Byg hold-DataFrame med EP
# ---------------------------------------------------------------------------

def build_my_team_df(
    els: pd.DataFrame,
    fixt: pd.DataFrame,
    picks: Dict[str, Any],
    horizon: int,
    odds_ctx_by_fixture: Optional[Dict],
    teams_df: pd.DataFrame,
    include_details: bool = False,
) -> pd.DataFrame:
    """
    Bygger DataFrame med dit holds 15 spillere inkl. EP-beregninger.
    """
    ep_col = f"ep_next{horizon}"
    my_el_ids = [p["element"] for p in picks.get("picks", [])]
    if not my_el_ids:
        return pd.DataFrame()

    my15 = els[els["id"].isin(my_el_ids)].copy()
    my15["sell_price"] = my15["now_cost"]
    my15["pos"] = my15["singular_name_short"]

    rows = []
    for _, pl in my15.iterrows():
        ep = expected_points_for_player(
            pl, fixt, n=horizon,
            odds_ctx_by_fixture=odds_ctx_by_fixture,
            teams_table=teams_df,
        )
        per_gw = ep["per_gw"]
        ep1 = per_gw[0]["ep"] if per_gw else 0.0

        is_home = False
        next_fdr = 3
        if per_gw:
            ev = per_gw[0]["event"]
            m = fixt[
                (fixt["event"] == ev)
                & ((fixt["home_team"] == pl["team_id"]) | (fixt["away_team"] == pl["team_id"]))
            ]
            if len(m):
                is_home = bool(int(m["home_team"].iloc[0]) == int(pl["team_id"]))
                next_fdr = int(m["home_fdr"].iloc[0]) if is_home else int(m["away_fdr"].iloc[0])

        row = {
            "id": int(pl["id"]),
            "name": str(pl["web_name"]),
            "team_id": int(pl["team_id"]),
            "team": str(pl["short_name"]),
            "pos": str(pl["singular_name_short"]),
            "now_cost": float(pl["now_cost"]),
            "sell_price": float(pl.get("sell_price", pl["now_cost"])),
            "status": str(pl["status"]),
            "form": float(pl.get("form", 0) or 0),
            "ep_next_gw": round(float(ep1), 2),
            ep_col: round(float(ep["total_next_n"]), 2),
            "is_home": is_home,
            "next_fdr": next_fdr,
            "has_dgw": ep.get("has_dgw", False),
        }

        if include_details:
            stats = get_player_detailed_stats(pl, fixt, n=horizon)
            row.update({
                "penalties_order": float(pl.get("penalties_order", 99) or 99),
                "corners_and_indirect_freekicks_order": float(
                    pl.get("corners_and_indirect_freekicks_order", 99) or 99
                ),
                "direct_freekicks_order": float(pl.get("direct_freekicks_order", 99) or 99),
                "is_penalty_taker": stats.get("is_penalty_taker", False),
                "is_set_piece_taker": stats.get("is_set_piece_taker", False),
                "selected_by_percent": stats.get("selected_by_percent", 0),
            })

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Byg kandidat-DataFrame med EP
# ---------------------------------------------------------------------------

def build_candidates_df(
    els: pd.DataFrame,
    my_ids: List[int],
    fixt: pd.DataFrame,
    horizon: int,
    odds_ctx_by_fixture: Optional[Dict],
    teams_df: pd.DataFrame,
    max_cost_extra: float = 10,
) -> pd.DataFrame:
    """
    Bygger DataFrame med EP for alle spillere der IKKE er i dit hold.
    """
    ep_col = f"ep_next{horizon}"

    # Begræns til realistiske priser
    max_cost = float(els.loc[els["id"].isin(my_ids), "now_cost"].max()) + max_cost_extra if my_ids else 200

    cand_src = els.loc[
        (~els["id"].isin(my_ids)) & (els["now_cost"] <= max_cost)
    ].copy()

    rows = []
    for _, cpl in cand_src.iterrows():
        ep = expected_points_for_player(
            cpl, fixt, n=horizon,
            odds_ctx_by_fixture=odds_ctx_by_fixture,
            teams_table=teams_df,
        )
        per_gw = ep["per_gw"]
        ep1 = per_gw[0]["ep"] if per_gw else 0.0

        is_home = False
        if per_gw:
            ev = per_gw[0]["event"]
            m = fixt[
                (fixt["event"] == ev)
                & ((fixt["home_team"] == cpl["team_id"]) | (fixt["away_team"] == cpl["team_id"]))
            ]
            if len(m):
                is_home = bool(int(m["home_team"].iloc[0]) == int(cpl["team_id"]))

        rows.append({
            "id": int(cpl["id"]),
            "name": str(cpl["web_name"]),
            "team_id": int(cpl["team_id"]),
            "team": str(cpl["short_name"]),
            "singular_name_short": str(cpl["singular_name_short"]),
            "pos": str(cpl["singular_name_short"]),
            "now_cost": float(cpl["now_cost"]),
            "status": str(cpl["status"]),
            "form": float(cpl.get("form", 0) or 0),
            "ep_next_gw": round(float(ep1), 2),
            ep_col: round(float(ep["total_next_n"]), 2),
            "is_home": is_home,
            "has_dgw": ep.get("has_dgw", False),
            "selected_by_percent": float(cpl.get("selected_by_percent", 0) or 0),
        })

    return pd.DataFrame(rows).sort_values(ep_col, ascending=False) if rows else pd.DataFrame()
