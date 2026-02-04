# pages/5_Chip_Strategi.py
"""Chip-strategi: Anbefaling af hvornår man skal bruge WC, BB, TC, FH."""
import streamlit as st
import pandas as pd

from services.fpl_api import manager_summary, entry_history
from services.data_layer import (
    load_base_data, get_picks_any, build_odds_context_for_fixtures,
    build_my_team_df,
)
from logic.features import expected_points_for_player, next_n_fixtures_for_team, count_fixtures_in_gw
from logic.optimizer import select_starting_xi, suggest_wildcard_team
from utils.helpers import safe_df
from utils.session import manager_id_input, get_manager_id
from utils.ui import inject_css

st.set_page_config(page_title="Chip-strategi", layout="wide")
inject_css()
st.title("🃏 Chip-strategi")
st.caption("Automatisk anbefaling af hvornår du bør bruge dine chips baseret på fixtures og holdanalyse.")

bs, events, els, fixt, teams_df = load_base_data()

with st.sidebar:
    st.header("⚙️ Indstillinger")
    entry_id = manager_id_input()
    odds_key = st.secrets.get("THE_ODDS_API_KEY", "")

entry_id = get_manager_id()
if not entry_id:
    st.info("👆 Indtast dit manager-ID i sidepanelet.")
    st.stop()

try:
    mgr = manager_summary(int(entry_id))
    history = entry_history(int(entry_id))
except Exception as e:
    st.error(f"❌ Fejl: {e}")
    st.stop()

# Find brugte chips
used_chips = set()
for chip in history.get("chips", []):
    chip_name = chip.get("name", "").lower()
    used_chips.add(chip_name)

ALL_CHIPS = {
    "wildcard": {"name": "Wildcard", "icon": "🔄", "desc": "Skift hele dit hold gratis"},
    "bboost": {"name": "Bench Boost", "icon": "📈", "desc": "Bænkspillere scorer point"},
    "3xc": {"name": "Triple Captain", "icon": "👑", "desc": "Kaptajnen scorer 3x point"},
    "freehit": {"name": "Free Hit", "icon": "⚡", "desc": "Optimalt hold i én runde"},
}

# Vis chip-status
st.markdown("### 📋 Dine chips")
cols = st.columns(4)
available_chips = []
for i, (key, info) in enumerate(ALL_CHIPS.items()):
    with cols[i]:
        is_used = key in used_chips
        if is_used:
            st.error(f"{info['icon']} **{info['name']}**\n\nBrugt ✓")
        else:
            st.success(f"{info['icon']} **{info['name']}**\n\nTilgængelig")
            available_chips.append(key)

if not available_chips:
    st.info("Du har brugt alle dine chips denne sæson.")
    st.stop()

st.markdown("---")

# Hent picks
preferred_gw = int(events.loc[events["is_next"] == True, "id"].iloc[0]) if (events["is_next"] == True).any() else 1
picks, used_gw, _ = get_picks_any(int(entry_id), preferred_gw, events)
if not picks:
    st.error("❌ Fandt ingen picks.")
    st.stop()

# Byg hold
my = build_my_team_df(els, fixt, picks, horizon=5, odds_ctx_by_fixture=None, teams_df=teams_df)
if my.empty:
    st.error("❌ Ingen spillere fundet.")
    st.stop()

team_value_m = float(my["now_cost"].sum()) / 10.0
bank_tenths = int(mgr.get("last_deadline_bank", 0) or 0)

# --- Analysér hver tilgængelig chip ---
st.markdown("### 🎯 Chip-anbefalinger")

# Find kommende GWs
upcoming_gws = sorted([int(e) for e in fixt["event"].unique() if e > 0])[:8]
my_team_ids = set(int(t) for t in my["team_id"].unique())


def score_bench_boost(gw: int) -> float:
    """Score BB: sum af bænkspillere EP i given GW."""
    try:
        xi_idx = select_starting_xi(my, formation="343")
        bench = my[~my.index.isin(xi_idx)]
        if bench.empty:
            return 0.0
        # Beregn EP kun for den GW
        total = 0.0
        for _, pl in bench.iterrows():
            n_fixtures = count_fixtures_in_gw(fixt, int(pl["team_id"]), gw)
            ep_per_fixture = float(pl.get("ep_next_gw", 0))
            total += ep_per_fixture * max(n_fixtures, 1)
        return total
    except Exception:
        return 0.0


def score_triple_captain(gw: int) -> tuple[float, str]:
    """Score TC: bedste kaptajns EP i given GW. Returnerer (score, spillernavn)."""
    best_ep = 0.0
    best_name = ""
    for _, pl in my.iterrows():
        n_fixtures = count_fixtures_in_gw(fixt, int(pl["team_id"]), gw)
        ep = float(pl.get("ep_next_gw", 0)) * max(n_fixtures, 1)
        if ep > best_ep:
            best_ep = ep
            best_name = str(pl["name"])
    # TC giver 2x ekstra (3x total - 2x normal = +1x)
    return best_ep, best_name


def count_team_fixtures(gw: int) -> dict:
    """Tæl kampe per hold i en GW."""
    counts = {}
    for team_id in my["team_id"].unique():
        counts[int(team_id)] = count_fixtures_in_gw(fixt, int(team_id), gw)
    return counts


def find_blank_gws() -> list[int]:
    """Find GWs hvor mange af dine spillere ikke har kamp."""
    blanks = []
    for gw in upcoming_gws:
        no_fixture = sum(1 for tid in my["team_id"] if count_fixtures_in_gw(fixt, int(tid), gw) == 0)
        if no_fixture >= 3:
            blanks.append(gw)
    return blanks


def find_dgw_gws() -> list[int]:
    """Find GWs med DGW-hold."""
    dgws = []
    for gw in upcoming_gws:
        dgw_teams = sum(1 for tid in teams_df["team_id"] if count_fixtures_in_gw(fixt, int(tid), gw) >= 2)
        if dgw_teams >= 2:
            dgws.append(gw)
    return dgws


dgw_gws = find_dgw_gws()
blank_gws = find_blank_gws()

# GW-by-GW analyse
gw_data = []
for gw in upcoming_gws:
    team_fixtures = count_team_fixtures(gw)
    my_dgw_count = sum(1 for c in team_fixtures.values() if c >= 2)
    my_blank_count = sum(1 for c in team_fixtures.values() if c == 0)
    bb_score = score_bench_boost(gw)
    tc_score, tc_player = score_triple_captain(gw)

    gw_data.append({
        "GW": gw,
        "DGW-spillere": my_dgw_count,
        "Blanke": my_blank_count,
        "BB Score": round(bb_score, 1),
        "TC Score": round(tc_score, 1),
        "TC Spiller": tc_player,
        "DGW": "🎯" if gw in dgw_gws else "",
        "BGW": "⚠️" if gw in blank_gws else "",
    })

gw_df = pd.DataFrame(gw_data)

# Anbefalinger per chip
for chip_key in available_chips:
    info = ALL_CHIPS[chip_key]

    with st.expander(f"{info['icon']} **{info['name']}** – {info['desc']}", expanded=True):

        if chip_key == "bboost":
            if dgw_gws:
                best_gw = max(dgw_gws, key=lambda g: score_bench_boost(g))
                bb_score = score_bench_boost(best_gw)
                st.success(f"**Anbefalet GW: {best_gw}** (DGW) – Bænk-EP: ~{bb_score:.1f} ekstra point")
                st.write("Bench Boost er bedst i Double Gameweeks hvor dine bænkspillere også har 2 kampe.")
            else:
                st.info("Ingen DGW fundet i de næste runder. Vent med BB til en DGW dukker op.")

        elif chip_key == "3xc":
            best_gw = max(upcoming_gws[:5], key=lambda g: score_triple_captain(g)[0])
            tc_score, tc_player = score_triple_captain(best_gw)
            is_dgw = best_gw in dgw_gws
            if is_dgw:
                st.success(f"**Anbefalet GW: {best_gw}** (DGW) – TC på **{tc_player}** (~{tc_score:.1f} EP)")
            else:
                st.info(f"Bedste kandidat: GW{best_gw} – TC på **{tc_player}** (~{tc_score:.1f} EP)")
                st.write("TC er mest værdifuld i en DGW. Overvej at vente.")

        elif chip_key == "freehit":
            if blank_gws:
                st.success(f"**Anbefalet GW: {blank_gws[0]}** (BGW) – {sum(1 for tid in my['team_id'] if count_fixtures_in_gw(fixt, int(tid), blank_gws[0]) == 0)} af dine spillere mangler kamp")
                st.write("Free Hit er perfekt i Blank Gameweeks hvor mange af dine spillere ikke spiller.")
            else:
                st.info("Ingen BGW fundet i de næste runder. Free Hit er bedst i blanke runder.")

        elif chip_key == "wildcard":
            # Estimér gevinst ved WC
            ep5_current = float(my["ep_next5"].sum()) if "ep_next5" in my.columns else float(my.get("ep_next_gw", pd.Series([0])).sum() * 5)
            st.write(f"Dit holds nuværende EP (5 GW): **{ep5_current:.1f}**")
            if dgw_gws:
                st.success(f"Overvej WC før **GW{min(dgw_gws)}** for at tilpasse holdet til DGW.")
            else:
                st.info("Brug WC når du har 4+ ønskede transfers, eller før en favorabel fixture-swing.")

st.markdown("---")

# GW-by-GW tabel
st.markdown("### 📅 GW-by-GW Oversigt")
st.dataframe(safe_df(gw_df), use_container_width=True)

# DGW/BGW info
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 🎯 Double Gameweeks")
    if dgw_gws:
        for gw in dgw_gws:
            dgw_teams = [
                teams_df.loc[teams_df["team_id"] == tid, "short_name"].iloc[0]
                for tid in teams_df["team_id"]
                if count_fixtures_in_gw(fixt, int(tid), gw) >= 2
                and tid in teams_df["team_id"].values
            ]
            st.write(f"**GW{gw}:** {', '.join(dgw_teams[:10])}")
    else:
        st.info("Ingen DGW i de næste runder.")

with col2:
    st.markdown("#### ⚠️ Blank Gameweeks")
    if blank_gws:
        for gw in blank_gws:
            blank_players = [
                str(pl["name"]) for _, pl in my.iterrows()
                if count_fixtures_in_gw(fixt, int(pl["team_id"]), gw) == 0
            ]
            if blank_players:
                st.write(f"**GW{gw}:** {', '.join(blank_players)} mangler kamp")
    else:
        st.info("Ingen BGW i de næste runder.")

st.markdown("---")
st.caption("🃏 Chip-strategi baseret på fixture-analyse og EP-beregninger")
