# pages/6_Historik.py
"""Historisk performance: Rank, point, transfers og chips over sæsonen."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from services.fpl_api import manager_summary, entry_history, entry_transfers
from services.data_layer import load_base_data
from logic.features import elements_df
from utils.session import manager_id_input, get_manager_id
from utils.ui import inject_css

st.set_page_config(page_title="Historik", layout="wide")
inject_css()
st.title("📈 Sæson-historik")
st.caption("Se din performance over sæsonen: rank, point, transfers og chips.")

bs, events, els, fixt, teams_df = load_base_data()

with st.sidebar:
    st.header("⚙️ Indstillinger")
    entry_id = manager_id_input()

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

current = pd.DataFrame(history.get("current", []))
chips = history.get("chips", [])

if current.empty:
    st.info("Ingen historisk data fundet for denne sæson.")
    st.stop()

# Konverter numeriske
for c in ["points", "total_points", "overall_rank", "rank", "bank", "value",
          "event_transfers", "event_transfers_cost", "points_on_bench"]:
    if c in current.columns:
        current[c] = pd.to_numeric(current[c], errors="coerce").fillna(0)

# --- Nøgletal ---
st.markdown(f"### 👤 {mgr.get('name', 'Dit hold')}")

best_gw = current.loc[current["points"].idxmax()] if not current.empty else None
worst_gw = current.loc[current["points"].idxmin()] if not current.empty else None
total_transfers = int(current["event_transfers"].sum()) if "event_transfers" in current.columns else 0
total_hits = int(current["event_transfers_cost"].sum()) if "event_transfers_cost" in current.columns else 0
total_bench_pts = int(current["points_on_bench"].sum()) if "points_on_bench" in current.columns else 0

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Point", f"{mgr.get('summary_overall_points', 0):,}")
with col2:
    st.metric("Overall Rank", f"{mgr.get('summary_overall_rank', 'N/A'):,}")
with col3:
    if best_gw is not None:
        st.metric("Bedste GW", f"{int(best_gw['points'])} pt (GW{int(best_gw['event'])})")
with col4:
    st.metric("Total Transfers", total_transfers)
with col5:
    st.metric("Point tabt (hits)", f"-{total_hits}")

st.markdown("---")

# --- Grafer ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🏆 Overall Rank over sæsonen")
    if "overall_rank" in current.columns:
        fig_rank = go.Figure()
        fig_rank.add_trace(go.Scatter(
            x=current["event"],
            y=current["overall_rank"],
            mode="lines+markers",
            line=dict(color="#37003c", width=3),
            marker=dict(size=6, color="#00ff87"),
            name="Overall Rank",
            hovertemplate="GW%{x}<br>Rank: %{y:,}<extra></extra>",
        ))
        fig_rank.update_layout(
            yaxis=dict(autorange="reversed", title="Rank (lavere = bedre)"),
            xaxis=dict(title="Gameweek", dtick=1),
            height=400,
            margin=dict(l=20, r=20, t=20, b=40),
            hovermode="x unified",
        )
        st.plotly_chart(fig_rank, use_container_width=True)

with col_right:
    st.subheader("📊 Point per Gameweek")
    if "points" in current.columns:
        avg_pts = current["points"].mean()
        fig_pts = go.Figure()
        fig_pts.add_trace(go.Bar(
            x=current["event"],
            y=current["points"],
            marker_color=["#00ff87" if p >= avg_pts else "#e90052" for p in current["points"]],
            name="Point",
            hovertemplate="GW%{x}<br>%{y} point<extra></extra>",
        ))
        fig_pts.add_hline(
            y=avg_pts, line_dash="dash", line_color="#37003c",
            annotation_text=f"Gns: {avg_pts:.1f}",
        )
        fig_pts.update_layout(
            xaxis=dict(title="Gameweek", dtick=1),
            yaxis=dict(title="Point"),
            height=400,
            margin=dict(l=20, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_pts, use_container_width=True)

st.markdown("---")

# --- Holdværdi ---
col_left2, col_right2 = st.columns(2)

with col_left2:
    st.subheader("💰 Holdværdi over sæsonen")
    if "value" in current.columns:
        fig_value = go.Figure()
        fig_value.add_trace(go.Scatter(
            x=current["event"],
            y=current["value"] / 10.0,
            mode="lines+markers",
            line=dict(color="#04f5ff", width=2),
            marker=dict(size=5),
            name="Holdværdi",
            hovertemplate="GW%{x}<br>£%{y:.1f}m<extra></extra>",
        ))
        if "bank" in current.columns:
            fig_value.add_trace(go.Scatter(
                x=current["event"],
                y=current["bank"] / 10.0,
                mode="lines",
                line=dict(color="#e90052", width=1, dash="dot"),
                name="Bank",
                hovertemplate="GW%{x}<br>Bank: £%{y:.1f}m<extra></extra>",
            ))
        fig_value.update_layout(
            xaxis=dict(title="Gameweek", dtick=1),
            yaxis=dict(title="Værdi (£m)"),
            height=350,
            margin=dict(l=20, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_value, use_container_width=True)

with col_right2:
    st.subheader("💺 Point på bænken")
    if "points_on_bench" in current.columns:
        fig_bench = go.Figure()
        fig_bench.add_trace(go.Bar(
            x=current["event"],
            y=current["points_on_bench"],
            marker_color="#ff6b35",
            name="Bænk-point",
            hovertemplate="GW%{x}<br>%{y} bænk-point<extra></extra>",
        ))
        fig_bench.update_layout(
            xaxis=dict(title="Gameweek", dtick=1),
            yaxis=dict(title="Point"),
            height=350,
            margin=dict(l=20, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_bench, use_container_width=True)
        st.caption(f"Total bænk-point: **{total_bench_pts}** over sæsonen")

st.markdown("---")

# --- Chips brugt ---
st.subheader("🃏 Chips brugt")
if chips:
    chip_cols = st.columns(len(chips))
    for i, chip in enumerate(chips):
        with chip_cols[i]:
            chip_name = chip.get("name", "?")
            chip_gw = chip.get("event", "?")
            # Find point i den GW
            gw_row = current[current["event"] == chip_gw]
            pts = int(gw_row["points"].iloc[0]) if not gw_row.empty else "?"
            st.info(f"**{chip_name}**\nGW{chip_gw} – {pts} point")
else:
    st.info("Ingen chips brugt endnu denne sæson.")

st.markdown("---")

# --- Transfer-log ---
st.subheader("🔄 Transfer-historik")
try:
    transfers = entry_transfers(int(entry_id))
    if transfers:
        # Byg tabel med spillernavne
        player_map = dict(zip(els["id"], els["web_name"]))
        team_map = dict(zip(teams_df["team_id"], teams_df["short_name"]))

        t_rows = []
        for t in transfers:
            t_rows.append({
                "GW": t.get("event", "?"),
                "Ud": player_map.get(t.get("element_out"), f"ID {t.get('element_out')}"),
                "Ind": player_map.get(t.get("element_in"), f"ID {t.get('element_in')}"),
                "Pris ud": f"£{t.get('element_out_cost', 0) / 10:.1f}m",
                "Pris ind": f"£{t.get('element_in_cost', 0) / 10:.1f}m",
                "Tidspunkt": str(t.get("time", ""))[:16],
            })
        t_df = pd.DataFrame(t_rows)
        st.dataframe(t_df, use_container_width=True, height=400)
        st.caption(f"Total: {len(transfers)} transfers denne sæson")
    else:
        st.info("Ingen transfers endnu.")
except Exception as e:
    st.warning(f"Kunne ikke hente transfers: {e}")

st.markdown("---")

# --- GW-by-GW detaljer ---
st.subheader("📋 GW-by-GW Detaljer")
gw_detail = current.copy()
if "value" in gw_detail.columns:
    gw_detail["Værdi (£m)"] = (gw_detail["value"] / 10.0).round(1)
if "bank" in gw_detail.columns:
    gw_detail["Bank (£m)"] = (gw_detail["bank"] / 10.0).round(1)

display_cols = ["event", "points", "total_points", "overall_rank", "rank",
                "event_transfers", "event_transfers_cost", "points_on_bench"]
display_cols = [c for c in display_cols if c in gw_detail.columns]
if "Værdi (£m)" in gw_detail.columns:
    display_cols.append("Værdi (£m)")
if "Bank (£m)" in gw_detail.columns:
    display_cols.append("Bank (£m)")

rename_map = {
    "event": "GW", "points": "Point", "total_points": "Total",
    "overall_rank": "Overall Rank", "rank": "GW Rank",
    "event_transfers": "Transfers", "event_transfers_cost": "Hit-cost",
    "points_on_bench": "Bænk-point",
}
gw_show = gw_detail[display_cols].rename(columns=rename_map)
st.dataframe(gw_show, use_container_width=True, height=500)

st.caption("📈 Historisk data fra FPL API")
