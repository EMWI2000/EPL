# pages/4_Data_Inspector.py
import streamlit as st
import pandas as pd
import json

from services.fpl_api import bootstrap_static, fixtures, entry_picks, manager_summary, entry_history, element_summary
from logic.features import elements_df, fixtures_df
from utils.session import init_manager_id, manager_id_input, get_manager_id
from utils.ui import inject_css

st.set_page_config(page_title="Data Inspector", layout="wide")
inject_css()
st.title("🔍 Data Inspector")
st.caption("Udforsk rå data fra FPL API – til debugging og analyse.")

with st.sidebar:
    st.header("⚙️ Indstillinger")

    # Brug delt session management
    entry_id = manager_id_input()

# Hent basis data
bs = bootstrap_static()
events = pd.DataFrame(bs["events"])
teams_df = pd.DataFrame(bs["teams"])
els = elements_df(bs)
fixt = fixtures_df(fixtures(future_only=True))

# Faner
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Spillere", "📅 Kampe", "🏆 Hold", "👤 Manager", "📈 Events"])

with tab1:
    st.subheader("Alle spillere")

    # Filtre
    col1, col2, col3 = st.columns(3)
    with col1:
        pos_filter = st.multiselect("Position", ["GKP", "DEF", "MID", "FWD"], default=["GKP", "DEF", "MID", "FWD"])
    with col2:
        team_filter = st.multiselect("Hold", sorted(els["short_name"].unique().tolist()))
    with col3:
        search = st.text_input("Søg spiller", "")

    # Filtrer
    df = els.copy()
    df = df[df["singular_name_short"].isin(pos_filter)]
    if team_filter:
        df = df[df["short_name"].isin(team_filter)]
    if search:
        df = df[df["web_name"].str.lower().str.contains(search.lower())]

    # Vælg kolonner
    display_cols = ["web_name", "short_name", "singular_name_short", "now_cost", "status",
                    "form", "points_per_game", "total_points", "minutes", "goals_scored",
                    "assists", "clean_sheets", "expected_goals", "expected_assists",
                    "ict_index", "influence", "creativity", "threat", "bps",
                    "penalties_order", "corners_and_indirect_freekicks_order", "selected_by_percent"]
    available_cols = [c for c in display_cols if c in df.columns]

    st.dataframe(
        df[available_cols].sort_values("total_points", ascending=False).head(100),
        use_container_width=True,
        height=500
    )

    st.caption(f"Viser {min(100, len(df))} af {len(df)} spillere")

    # Spiller detaljer
    st.markdown("---")
    st.subheader("Spiller detaljer")
    player_id = st.number_input("Spiller ID", min_value=1, max_value=1000, value=1)
    if st.button("Hent spiller detaljer"):
        try:
            player_data = element_summary(int(player_id))
            st.json(player_data)
        except Exception as e:
            st.error(f"Fejl: {e}")

with tab2:
    st.subheader("Kommende kampe")

    # Tilføj holdnavne
    fixt_display = fixt.copy()
    team_map = dict(zip(teams_df["id"], teams_df["short_name"]))
    fixt_display["home"] = fixt_display["home_team"].map(team_map)
    fixt_display["away"] = fixt_display["away_team"].map(team_map)

    # Filter på GW
    gw_filter = st.selectbox("Gameweek", sorted(fixt_display["event"].unique()))
    fixt_gw = fixt_display[fixt_display["event"] == gw_filter]

    st.dataframe(
        fixt_gw[["event", "home", "away", "home_fdr", "away_fdr", "kickoff_time"]],
        use_container_width=True
    )

    # DGW check
    st.markdown("---")
    st.subheader("Double Gameweek Check")
    for team_id, team_name in team_map.items():
        mask = ((fixt_display["home_team"] == team_id) | (fixt_display["away_team"] == team_id)) & (
                    fixt_display["event"] == gw_filter)
        games = len(fixt_display[mask])
        if games >= 2:
            st.success(f"🎯 **{team_name}** har {games} kampe i GW{gw_filter} (DGW)")

with tab3:
    st.subheader("Hold data")
    st.dataframe(teams_df, use_container_width=True)

    # Hold stats
    st.markdown("---")
    st.subheader("Spillere per hold")
    team_stats = els.groupby("short_name").agg({
        "id": "count",
        "now_cost": "mean",
        "total_points": "sum",
        "form": "mean"
    }).round(2)
    team_stats.columns = ["Antal spillere", "Gns. pris", "Total point", "Gns. form"]
    st.dataframe(team_stats.sort_values("Total point", ascending=False), use_container_width=True)

with tab4:
    st.subheader("Manager data")

    entry_id = get_manager_id()

    if not entry_id:
        st.info("👆 Indtast dit manager-ID i sidepanelet.")
    else:
        try:
            mgr = manager_summary(int(entry_id))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Holdnavn", mgr.get("name", "?"))
                st.metric("Overall Rank", f"{mgr.get('summary_overall_rank', 'N/A'):,}")
                st.metric("Total Point", mgr.get("summary_overall_points", "N/A"))
            with col2:
                st.metric("GW Rank", mgr.get("summary_event_rank", "N/A"))
                tv = mgr.get("last_deadline_value", 0)
                bank = mgr.get("last_deadline_bank", 0)
                st.metric("Holdværdi", f"£{tv / 10:.1f}m" if tv else "N/A")
                st.metric("Bank", f"£{bank / 10:.1f}m" if bank else "N/A")

            # Rå data
            with st.expander("Se rå manager data"):
                st.json(mgr)

            # Historik
            st.markdown("---")
            st.subheader("Sæson historik")
            try:
                history = entry_history(int(entry_id))
                current = pd.DataFrame(history.get("current", []))
                if not current.empty:
                    st.line_chart(current.set_index("event")["overall_rank"])
                    st.dataframe(current, use_container_width=True)

                # Chips
                chips = history.get("chips", [])
                if chips:
                    st.markdown("**Brugte chips:**")
                    for chip in chips:
                        st.write(f"- {chip.get('name', '?')} (GW{chip.get('event', '?')})")
            except Exception as e:
                st.warning(f"Kunne ikke hente historik: {e}")

            # Picks
            st.markdown("---")
            st.subheader("Nuværende hold")
            gw_pick = st.number_input("Hent picks for GW", 1, 38, value=1)
            if st.button("Hent picks"):
                try:
                    picks = entry_picks(int(entry_id), int(gw_pick))

                    picks_list = picks.get("picks", [])
                    picks_df = pd.DataFrame(picks_list)

                    # Tilføj spiller navne
                    if not picks_df.empty:
                        player_map = dict(zip(els["id"], els["web_name"]))
                        picks_df["name"] = picks_df["element"].map(player_map)
                        st.dataframe(picks_df, use_container_width=True)

                    with st.expander("Se rå picks data"):
                        st.json(picks)
                except Exception as e:
                    st.error(f"Fejl: {e}")

        except Exception as e:
            st.error(f"Fejl ved hentning af manager data: {e}")

with tab5:
    st.subheader("Gameweek Events")
    st.dataframe(events, use_container_width=True, height=500)

    # Find current/next GW
    current_gw = events.loc[events["is_current"] == True, "id"].tolist()
    next_gw = events.loc[events["is_next"] == True, "id"].tolist()

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Current GW:** {current_gw[0] if current_gw else 'N/A'}")
    with col2:
        st.info(f"**Next GW:** {next_gw[0] if next_gw else 'N/A'}")

st.markdown("---")
st.caption("🔍 Data Inspector – til debugging og udforskning af FPL API data")
