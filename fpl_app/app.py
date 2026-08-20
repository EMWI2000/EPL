# app.py
import streamlit as st
import pandas as pd

from services.fpl_api import manager_summary
from services.data_layer import (
    load_base_data, get_picks_any, build_odds_context_for_fixtures,
    build_my_team_df, build_candidates_df,
)
from logic.captain import captain_score
from logic.optimizer import find_best_formation, best_one_transfer_with_quotas
from utils.helpers import safe_df, safe_event_id
from utils.config import get_secret
from utils.session import init_manager_id
from utils.ui import inject_css

st.set_page_config(page_title="FPL HoldPlanner DK", layout="wide")
inject_css()
st.title("⚽ FPL HoldPlanner DK – Mit Hold")
st.caption("Eksperimentel beslutningsstøtte til kaptajn, start-XI og transfer-screening.")
st.warning("Pointprognosen er endnu ikke backtestet og må ikke opfattes som et sikkert facit.")

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
    odds_key = get_secret("THE_ODDS_API_KEY", "") or ""

entry_id = st.session_state.get("entry_id", "") or ""
if not entry_id:
    st.info("Du har ikke et hold endnu. Byg først et lovligt, optimeret udkast uden manager-ID.")
    st.page_link("pages/0_Byg_starttrup.py", label="Byg dit første hold", icon="🧩")
    st.caption("Har du allerede oprettet holdet i FPL, kan du indtaste manager-ID i sidepanelet.")
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

# Kaptajn: alle kendte faktorer skal allerede være indeholdt i EP-prognosen.
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
    st.caption("EP = eksperimentelt estimat. Odds-justeret hvis valgt i sidepanelet.")

with tab2:
    st.markdown("### 🧭 Kaptajn-anbefaling (kun næste GW)")
    st.write(cap_rec[["name", "team", "pos", "ep_next_gw", "cap_score"]])
    st.caption("Rangeret direkte efter næste-GW-prognosen; fixture, form og DGW lægges ikke oveni igen.")

with tab3:
    st.markdown("### 🧩 Startopstilling (optimering for næste GW)")
    formation, xi_idx, xi_ep = find_best_formation(my15_view)
    st.write(f"**Bedste lovlige formation: {formation[0]}-{formation[1]}-{formation[2]}** · EP {xi_ep:.2f}")
    st.write(my15_view.loc[xi_idx, ["name", "team", "pos", "ep_next_gw"]])

with tab4:
    st.markdown(f"### 🔁 1-transfer screening (næste {horizon} GW)")
    st.warning(
        "Dette er en bruttoscreening. Den værdisætter endnu ikke en gemt free transfer, fremtidige "
        "hits, start-XI, bænk eller chips og er derfor ikke en endelig transferanbefaling."
    )
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
st.subheader("🤖 AI-forklaring")
st.info(
    "Automatiske AI-råd er sat på pause, indtil prognoser og transferlogik er backtestet. "
    "En senere version må forklare de beregnede scenarier, men ikke opfinde nye tal eller skjule usikkerhed."
)
