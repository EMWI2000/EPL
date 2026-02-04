# pages/2_Opsummering.py
import streamlit as st
import pandas as pd

from services.fpl_api import manager_summary
from services.data_layer import (
    load_base_data, get_picks_any, build_odds_context_for_fixtures,
    build_my_team_df, build_candidates_df,
)
from logic.captain import captain_score
from logic.optimizer import best_n_transfers_with_quotas, best_two_transfers, find_best_formation
from utils.helpers import safe_df
from utils.session import manager_id_input, get_manager_id
from utils.ui import inject_css

st.set_page_config(page_title="Opsummering", layout="wide")
inject_css()
st.title("📊 Opsummering: Anbefalinger & Forklaringer")
st.caption("Komplet overblik over dit hold med kaptajn- og transfer-anbefalinger.")

bs, events, els, fixt, teams_df = load_base_data()

with st.sidebar:
    st.header("⚙️ Indstillinger")
    entry_id = manager_id_input()

    default_gw = int(events.loc[events["is_next"] == True, "id"].iloc[0]) if (events["is_next"] == True).any() else 1
    target_gw = st.number_input("Gameweek (mål/GW)", 1, 38, value=default_gw)
    horizon = st.slider("Horisont (antal runder)", 1, 5, 5)
    use_odds = st.toggle("Brug odds i beregninger", value=False)
    odds_key = st.secrets.get("THE_ODDS_API_KEY", "")

entry_id = get_manager_id()
if not entry_id:
    st.info("👆 Indtast dit manager-ID i sidepanelet.")
    st.stop()

try:
    mgr = manager_summary(int(entry_id))
except Exception as e:
    st.error(f"❌ Kunne ikke hente manager-data. Fejl: {e}")
    st.stop()

ep_col = f"ep_next{horizon}"

# Odds
odds_ctx_by_fixture, odds_status = build_odds_context_for_fixtures(odds_key, fixt, teams_df, use_odds)
if use_odds and "Fejl" in odds_status:
    st.warning("⚠️ Kunne ikke hente/parse odds – fortsætter uden.")

# Picks
picks, used_gw, _ = get_picks_any(int(entry_id), int(target_gw), events)
if not picks:
    st.error("❌ Fandt ingen picks.")
    st.stop()

# Byg hold med detaljer (inkl. set piece info til kaptajn-scoring)
my = build_my_team_df(els, fixt, picks, horizon, odds_ctx_by_fixture, teams_df, include_details=True)
if my.empty:
    st.error("❌ Fandt ingen spillere.")
    st.stop()

# Kaptajn
my["cap_score"] = my.apply(captain_score, axis=1)
cap = my.sort_values("cap_score", ascending=False).head(5)

# Kandidater
cand = build_candidates_df(els, my["id"].tolist(), fixt, horizon, odds_ctx_by_fixture, teams_df)

team_value_m = float(my["now_cost"].sum()) / 10.0
bank_tenths = int(mgr.get("last_deadline_bank", 0) or 0)
bank_m = bank_tenths / 10.0

# Transfer-forslag
singles = best_n_transfers_with_quotas(my, cand, bank_tenths, team_value_m, horizon=horizon, top_n=5)
doubles = best_two_transfers(my, cand, bank_tenths, team_value_m, horizon=horizon, top_n=3)

# Manager header
st.markdown(f"### 👤 {mgr.get('name', 'Dit hold')}")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Overall Rank", f"{mgr.get('summary_overall_rank', 'N/A'):,}")
with col2:
    st.metric("Total Point", mgr.get('summary_overall_points', 'N/A'))
with col3:
    st.metric("Holdværdi", f"£{team_value_m:.1f}m")
with col4:
    st.metric("Bank", f"£{bank_m:.1f}m")

st.markdown("---")

# KAPTAJN SEKTION
st.markdown("### 👑 Kaptajn (næste kamp)")
if not cap.empty:
    top = cap.iloc[0]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.success(f"**Anbefalet kaptajn:** {top['name']} ({top['team']}, {top['pos']})")
        st.write(
            f"📊 EP næste GW: **{top['ep_next_gw']:.2f}** | Form: **{top['form']:.1f}** | FDR: **{top['next_fdr']}**")

        details = captain_score(top, include_details=True)
        if details.get("boosts"):
            boost_str = " • ".join([f"{b[0]}" for b in details["boosts"]])
            st.caption(f"✨ Boosts: {boost_str}")

    with col2:
        if top["is_home"]:
            st.info("🏠 Hjemmekamp")
        if top["has_dgw"]:
            st.success("🎯 Dobbelt Gameweek!")

    st.markdown("**Top 5 kandidater:**")
    cap_display = cap[["name", "team", "pos", "ep_next_gw", "form", "cap_score", "next_fdr"]].copy()
    cap_display.columns = ["Navn", "Hold", "Pos", "EP", "Form", "Score", "FDR"]
    st.dataframe(safe_df(cap_display), use_container_width=True)
else:
    st.info("Ingen data til kaptajnanbefaling.")

st.markdown("---")

# TRANSFER SEKTION
st.markdown(f"### 🔄 Bedste Transfers (næste {horizon} GW)")

st.markdown("#### 📝 Top 5 Enkelt-Transfers")
if singles:
    for i, t in enumerate(singles, 1):
        out = t["out"]
        inp = t["in"]
        delta = t["delta"]
        out_price = float(out["now_cost"]) / 10.0
        in_price = float(inp["now_cost"]) / 10.0

        msg = (
            f"**{i}. {out['name']} ({out['team']}) → {inp['name']} ({inp['team']})** | "
            f"ΔEP: {'+' if delta > 0 else ''}{delta:.2f} | Pris: £{out_price:.1f}m → £{in_price:.1f}m"
        )
        if i == 1:
            st.success(msg)
        else:
            st.info(msg)
else:
    st.info("Ingen positive enkelt-transfers fundet under gældende begrænsninger.")

st.markdown("---")

st.markdown("#### 📝 Top 3 Dobbelt-Transfers (inkl. -4 hit)")
if doubles:
    for i, t in enumerate(doubles, 1):
        outs = t["out"]
        ins = t["in"]
        delta = t["delta"]
        delta_before_hit = t.get("delta_before_hit", delta + 4)

        out_names = " + ".join([o["name"] for o in outs])
        in_names = " + ".join([o["name"] for o in ins])

        if delta > 0:
            st.success(
                f"**{i}. {out_names} → {in_names}** | "
                f"ΔEP netto: **+{delta:.2f}** (før hit: +{delta_before_hit:.2f})"
            )
        else:
            st.warning(
                f"**{i}. {out_names} → {in_names}** | "
                f"ΔEP netto: {delta:.2f} (før hit: +{delta_before_hit:.2f}) - Ikke værd med hit"
            )

        with st.expander("Se detaljer"):
            for j, (out, inp) in enumerate(zip(outs, ins)):
                st.write(
                    f"**Transfer {j + 1}:** {out['name']} ({out['team']}, £{float(out['now_cost']) / 10:.1f}m) → "
                    f"{inp['name']} ({inp['team']}, £{float(inp['now_cost']) / 10:.1f}m)"
                )
            st.write(f"💰 Ny bank: £{t['new_bank']:.1f}m")
else:
    st.info("Ingen positive dobbelt-transfers fundet.")

st.markdown("---")

# DIT HOLD
st.markdown(f"### 🧱 Dit hold – forventede point (næste {horizon} GW)")
df_show = my.copy()
df_show["Pris (£m)"] = df_show["now_cost"] / 10.0
df_show["🏠"] = df_show["is_home"].apply(lambda x: "✅" if x else "")
df_show["DGW"] = df_show["has_dgw"].apply(lambda x: "🎯" if x else "")

display_cols = ["name", "team", "pos", "Pris (£m)", "status", "form", "ep_next_gw", ep_col, "🏠", "DGW"]
st.dataframe(
    safe_df(df_show[display_cols].sort_values(ep_col, ascending=False)),
    use_container_width=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(f"Total EP ({horizon} GW)", f"{my[ep_col].sum():.1f}")
with col2:
    st.metric("Gns. EP per spiller", f"{my[ep_col].mean():.2f}")
with col3:
    dgw_count = len(my[my["has_dgw"] == True])
    st.metric("Spillere med DGW", dgw_count)

st.markdown("---")
st.caption(f"📊 Data fra GW{used_gw} | Odds: {odds_status} | Horisont: {horizon} GW")
