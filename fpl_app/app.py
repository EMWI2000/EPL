# app.py
import streamlit as st
import pandas as pd

from services.fpl_api import manager_summary
from services.data_layer import (
    load_base_data, get_picks_any, build_odds_context_for_fixtures,
    build_my_team_df, build_candidates_df,
)
from logic.captain import captain_score
from logic.optimizer import select_starting_xi, best_one_transfer_with_quotas
from services.chat import suggest_from_ai, get_provider_name, get_provider
from utils.helpers import safe_df, safe_event_id, fmt_prompt_table
from utils.session import init_manager_id
from utils.ui import inject_css

st.set_page_config(page_title="FPL HoldPlanner DK", layout="wide")
inject_css()
st.title("⚽ FPL HoldPlanner DK – Mit Hold")
st.caption("Personlig analyse på dansk: kaptajn, transfers og forventede point. Vælg horisont (1–5 GW) og valgfri odds-justering.")

# --- Hent basisdata ---
bs, events, els, fixt, teams_df = load_base_data()

with st.sidebar:
    st.header("Indstillinger")
    _ = init_manager_id()
    entry_id = st.text_input("Dit FPL manager-ID", key="entry_id", placeholder="fx 1499152")

    try:
        st.query_params["entry_id"] = entry_id
    except Exception:
        pass

    default_gw = safe_event_id(events, "is_next", safe_event_id(events, "is_current", 1))
    target_gw = st.number_input("Gameweek (mål/GW)", min_value=1, max_value=38, value=int(default_gw))
    horizon = st.slider("Horisont (antal runder)", min_value=1, max_value=5, value=5)
    use_odds = st.toggle("Brug odds i beregninger", value=False)
    odds_key = st.secrets.get("THE_ODDS_API_KEY", "")

ai_provider = get_provider()

entry_id = st.session_state.get("entry_id", "") or ""
if not entry_id:
    st.info("Indtast dit manager-ID i sidepanelet for at fortsætte.")
    st.stop()

# Managerinfo
try:
    mgr = manager_summary(int(entry_id))
except Exception as e:
    st.error(f"Kunne ikke hente manager-data for ID {entry_id}. Tjek ID. Fejl: {e}")
    st.stop()

fav_team_id = mgr.get("favourite_team")
fav_team = teams_df.loc[teams_df["team_id"] == fav_team_id, "name"].iloc[0] \
    if isinstance(fav_team_id, int) and fav_team_id in teams_df["team_id"].values else "—"

st.subheader("🔎 Manager")
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    st.write(f"**Holdnavn:** {mgr.get('name', '?')}")
with c2:
    st.write(f"**Favorithold:** {fav_team}")
with c3:
    tv_tenths = mgr.get("last_deadline_value", None)
    bank_tenths = mgr.get("last_deadline_bank", 0)
    if tv_tenths is not None:
        st.write(f"**Værdi:** {tv_tenths / 10:.1f} mio. (bank: {bank_tenths / 10:.1f} mio.)")
    else:
        st.write("**Værdi:** —")

# Hent picks
picks, used_gw, tried_gws = get_picks_any(int(entry_id), int(target_gw), events)
if not picks:
    st.error("Kunne ikke hente dine picks for: " + ", ".join(str(x) for x in tried_gws))
    st.stop()
if used_gw != int(target_gw):
    st.info(f"Viser dine picks for **GW{used_gw}** (ikke GW{target_gw}).")

# Odds-kontekst
odds_ctx_by_fixture, odds_status = build_odds_context_for_fixtures(odds_key, fixt, teams_df, use_odds)
if use_odds and "Fejl" in odds_status:
    st.warning("Kunne ikke hente/parse odds – fortsætter uden odds.")

ep_col = f"ep_next{horizon}"

# Byg mit hold
my15_view = build_my_team_df(els, fixt, picks, horizon, odds_ctx_by_fixture, teams_df)
if my15_view.empty:
    st.error("Fandt ingen picks i runden – prøv en anden GW.")
    st.stop()

# Kaptajn
my15_view["setpiece_boost"] = 0
my15_view["threat_norm"] = 0.5
my15_view["cap_score"] = my15_view.apply(captain_score, axis=1)
cap_rec = my15_view.sort_values("cap_score", ascending=False).head(3)

# Kandidater
cand_df = build_candidates_df(els, my15_view["id"].tolist(), fixt, horizon, odds_ctx_by_fixture, teams_df)

team_value_m = float(my15_view["now_cost"].sum()) / 10.0
bank_tenths = int(mgr.get("last_deadline_bank", 0) or 0)

best = best_one_transfer_with_quotas(
    my15_view, cand_df, bank_tenths=bank_tenths, team_value_m=team_value_m, horizon=horizon
)

# Faner
tab1, tab2, tab3, tab4 = st.tabs(["Mit hold (EP)", "Kaptajn", "Start-XI", "Transfers"])

with tab1:
    st.markdown(f"### 🧱 Mit hold – forventede point (næste {horizon} GW)  \n_Dine picks er fra GW{used_gw}._")
    df_show = my15_view.copy()
    df_show["Pris (mio)"] = df_show["now_cost"] / 10.0
    st.dataframe(
        safe_df(
            df_show[["name", "team", "pos", "Pris (mio)", "status", "ep_next_gw", ep_col]]
            .sort_values(ep_col, ascending=False)
        ),
        use_container_width=True,
    )
    st.caption("EP = Forventede point. Odds-justeret hvis valgt i sidepanelet.")

with tab2:
    st.markdown("### 🧭 Kaptajn-anbefaling (kun næste GW)")
    st.write(cap_rec[["name", "team", "pos", "ep_next_gw", "cap_score"]])
    st.caption("Valgt baseret på forv. point for næste kamp + små boosts.")

with tab3:
    st.markdown("### 🧩 Startopstilling (optimering for næste GW)")
    colA, colB = st.columns(2)
    with colA:
        xi343_idx = select_starting_xi(my15_view, formation="343")
        st.write("**Anbefalet 3-4-3:**")
        st.write(my15_view.loc[xi343_idx, ["name", "team", "pos", "ep_next_gw"]])
    with colB:
        xi352_idx = select_starting_xi(my15_view, formation="352")
        st.write("**Anbefalet 3-5-2:**")
        st.write(my15_view.loc[xi352_idx, ["name", "team", "pos", "ep_next_gw"]])

with tab4:
    st.markdown(f"### 🔁 1-transfer forslag (optimeret for næste {horizon} GW)")
    if best:
        st.success(
            f"**Ud:** {best['out']['name']} ({best['out']['team']}, {best['out']['pos']})  ➜  "
            f"**Ind:** {best['in']['name']} ({best['in']['team']}, {best['in']['pos']})  |  "
            f"ΔEP ({horizon} GW): {best['delta']:.2f}"
        )
        colU, colI = st.columns(2)
        with colU:
            st.markdown("**↙️ Ud (detaljer)**")
            o = best["out"]
            out_price = float(o["now_cost"]) / 10.0
            st.table(pd.DataFrame([{
                "Navn": o["name"], "Hold": o["team"], "Pos": o["pos"],
                "Pris (mio)": f"{out_price:.1f}", "Status": o["status"],
                "EP næste GW": o.get("ep_next_gw"), f"EP næste {horizon}": o.get(ep_col),
            }]))
        with colI:
            st.markdown("**↗️ Ind (detaljer)**")
            i = best["in"]
            in_price = float(i["now_cost"]) / 10.0
            st.table(pd.DataFrame([{
                "Navn": i["name"], "Hold": i["team"], "Pos": i["pos"],
                "Pris (mio)": f"{in_price:.1f}", "Status": i["status"],
                "EP næste GW": i.get("ep_next_gw"), f"EP næste {horizon}": i.get(ep_col),
                "Hjemme næste": "Ja" if bool(i.get("is_home", False)) else "Nej",
            }]))
    else:
        st.info("Ingen positiv forbedring fundet under budget/kvoter – prøv anden horisont.")

st.markdown("---")
st.subheader("🤖 Chat-anbefaling (inkl. spillere uden for din trup)")

_my_cols = ["name", "team", "pos", "now_cost", "status", "ep_next_gw", ep_col]
my_for_prompt = my15_view.sort_values(ep_col, ascending=False).copy()

_cand_cols = ["name", "team", "pos", "now_cost", "status", "ep_next_gw", ep_col, "is_home"]
cand_for_prompt = cand_df.sort_values(ep_col, ascending=False).copy()

bank_mio = (bank_tenths or 0) / 10.0
used_gw_txt = f"GW{used_gw}"
pos_counts = my15_view["pos"].value_counts().to_dict()
team_counts = my15_view["team"].value_counts().head(5).to_dict()

best_hint = ""
if best:
    best_hint = (
        f"Foreløbig bedste 1-transfer: Ud {best['out']['name']} -> Ind {best['in']['name']} | "
        f"ΔEP({horizon}GW) {best['delta']:.2f}\n"
    )

prompt = f"""
KONTEKST:
- FPL, mit hold og kandidater nedenfor. Bank: {bank_mio:.1f} mio. Kvoter (2/5/5/3) og max 3 pr klub.
- Horisont: næste {horizon} GW. Picks vist for {used_gw_txt}.
- Positionsfordeling: {pos_counts}. Top klubfordeling: {team_counts}.
{best_hint}DATA – MIT HOLD (øverst = højest EP({horizon}GW)):
{fmt_prompt_table(my_for_prompt, _my_cols, n=15)}

DATA – KANDIDATER (ikke i truppen):
{fmt_prompt_table(cand_for_prompt, _cand_cols, n=15)}

OPGAVE (dansk, punktvis):
1) Anbefal ét bytte (Ud -> Ind) der respekterer budget/kvoter, med pris, EP næste GW og EP({horizon}GW) + ΔEP({horizon}GW) og 1-linjers begrundelse.
2) Giv 2–3 alternativer i forskellige prislag.
3) Kaptajn: 1 hovedvalg + 1–2 alternativer (for næste GW).
4) Formation (3-4-3 eller 3-5-2) for næste GW.
"""

if ai_provider:
    st.caption(f"Aktiv AI-provider: {get_provider_name()}")
    with st.spinner("Henter AI-anbefaling..."):
        try:
            ai_reply = suggest_from_ai("", prompt)
            st.markdown(ai_reply)
        except Exception as e:
            st.error(f"AI-fejl: {e}")
else:
    st.info("Tilføj ANTHROPIC_API_KEY eller OPENAI_API_KEY i secrets.toml for AI-anbefalinger.")
