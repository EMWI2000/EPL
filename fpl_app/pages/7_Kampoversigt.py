# pages/7_Kampoversigt.py
"""Fixture Planner: Visuel fixture-kalender med farvekodede FDR og DGW-markering."""
import streamlit as st
import pandas as pd

from services.data_layer import load_base_data
from logic.features import count_fixtures_in_gw
from utils.session import init_manager_id
from utils.ui import inject_css, fdr_color, fdr_text_color

st.set_page_config(page_title="Kampoversigt", layout="wide")
inject_css()
st.title("📅 Kampoversigt – Fixture Planner")
st.caption("Farvekoderet oversigt over alle holds kampe. Grøn = let, rød = svær. 🎯 = Double Gameweek.")

bs, events, els, fixt, teams_df = load_base_data()

with st.sidebar:
    st.header("⚙️ Indstillinger")
    init_manager_id()

# Find tilgængelige GWs
all_gws = sorted([int(e) for e in fixt["event"].unique() if e > 0])
if not all_gws:
    st.info("Ingen kommende kampe fundet.")
    st.stop()

# GW-range selector
default_start = all_gws[0]
default_end = min(all_gws[0] + 5, all_gws[-1])

col1, col2 = st.columns(2)
with col1:
    gw_start = st.selectbox("Fra GW", all_gws, index=0)
with col2:
    end_options = [g for g in all_gws if g >= gw_start]
    default_idx = min(5, len(end_options) - 1)
    gw_end = st.selectbox("Til GW", end_options, index=default_idx)

selected_gws = [g for g in all_gws if gw_start <= g <= gw_end]

if not selected_gws:
    st.info("Vælg et gyldigt GW-interval.")
    st.stop()

# Byg team → short_name map
team_map = dict(zip(teams_df["team_id"], teams_df["short_name"]))
team_name_map = dict(zip(teams_df["team_id"], teams_df["name"]))

# Byg fixture grid
# For hvert hold og hver GW, find modstander(e), FDR og H/A
all_teams = sorted(teams_df["team_id"].tolist())


def get_fixtures_for_team_gw(team_id: int, gw: int):
    """Returnerer liste af (opponent_short, fdr, is_home) for et hold i en GW."""
    mask = (fixt["event"] == gw) & ((fixt["home_team"] == team_id) | (fixt["away_team"] == team_id))
    matches = fixt[mask]
    results = []
    for _, m in matches.iterrows():
        is_home = int(m["home_team"]) == team_id
        opponent_id = int(m["away_team"]) if is_home else int(m["home_team"])
        fdr = int(m["home_fdr"]) if is_home else int(m["away_fdr"])
        opponent_short = team_map.get(opponent_id, "?")
        results.append((opponent_short, fdr, is_home))
    return results


# Beregn total FDR-score per hold for sortering
team_fdr_scores = {}
for team_id in all_teams:
    total_fdr = 0
    total_fixtures = 0
    for gw in selected_gws:
        fixtures_in_gw = get_fixtures_for_team_gw(team_id, gw)
        for _, fdr, _ in fixtures_in_gw:
            total_fdr += fdr
            total_fixtures += 1
    avg_fdr = total_fdr / total_fixtures if total_fixtures > 0 else 3.0
    team_fdr_scores[team_id] = avg_fdr

# Sortér hold efter nemmeste fixtures
sort_option = st.radio(
    "Sortér efter",
    ["Nemmeste fixtures", "Alfabetisk", "Flest kampe"],
    horizontal=True,
)

if sort_option == "Nemmeste fixtures":
    sorted_teams = sorted(all_teams, key=lambda t: team_fdr_scores.get(t, 3))
elif sort_option == "Flest kampe":
    team_fixture_counts = {}
    for t in all_teams:
        count = sum(len(get_fixtures_for_team_gw(t, gw)) for gw in selected_gws)
        team_fixture_counts[t] = count
    sorted_teams = sorted(all_teams, key=lambda t: team_fixture_counts.get(t, 0), reverse=True)
else:
    sorted_teams = sorted(all_teams, key=lambda t: team_map.get(t, ""))

st.markdown("---")

# Byg HTML tabel
html = '<table style="width:100%;border-collapse:collapse;font-family:sans-serif;">'

# Header
html += '<tr style="background:#37003c;color:white;">'
html += '<th style="padding:8px 12px;text-align:left;position:sticky;left:0;background:#37003c;z-index:1;">Hold</th>'
for gw in selected_gws:
    html += f'<th style="padding:8px 6px;text-align:center;min-width:70px;">GW{gw}</th>'
html += '<th style="padding:8px 6px;text-align:center;">Gns. FDR</th>'
html += '</tr>'

# Rækker
for i, team_id in enumerate(sorted_teams):
    short_name = team_map.get(team_id, "?")
    bg = "#f8f9fa" if i % 2 == 0 else "#ffffff"

    html += f'<tr style="background:{bg};">'
    html += (
        f'<td style="padding:6px 12px;font-weight:700;white-space:nowrap;'
        f'position:sticky;left:0;background:{bg};z-index:1;">{short_name}</td>'
    )

    for gw in selected_gws:
        fixtures_list = get_fixtures_for_team_gw(team_id, gw)
        if not fixtures_list:
            # Blank GW
            html += '<td style="background:#f5f5f5;text-align:center;padding:6px 4px;color:#ccc;">-</td>'
        else:
            is_dgw = len(fixtures_list) >= 2
            cell_parts = []
            avg_fdr = sum(f[1] for f in fixtures_list) / len(fixtures_list)
            cell_bg = fdr_color(round(avg_fdr))
            cell_text = fdr_text_color(round(avg_fdr))

            for opp, fdr, is_home in fixtures_list:
                venue = "H" if is_home else "A"
                cell_parts.append(f"{opp} ({venue})")

            cell_content = "<br>".join(cell_parts)
            dgw_border = "border:2px solid gold;" if is_dgw else ""
            dgw_icon = " 🎯" if is_dgw else ""

            html += (
                f'<td style="background:{cell_bg};color:{cell_text};text-align:center;'
                f'padding:4px 3px;font-size:0.78em;font-weight:600;{dgw_border}">'
                f'{cell_content}{dgw_icon}</td>'
            )

    # Gennemsnit FDR
    avg = team_fdr_scores.get(team_id, 3.0)
    avg_bg = fdr_color(round(avg))
    avg_text = fdr_text_color(round(avg))
    html += (
        f'<td style="background:{avg_bg};color:{avg_text};text-align:center;'
        f'padding:6px 4px;font-weight:700;">{avg:.1f}</td>'
    )

    html += '</tr>'

html += '</table>'

# Render
st.markdown(html, unsafe_allow_html=True)

# Legende
st.markdown("---")
st.markdown("### 🎨 Legende")
legend_cols = st.columns(6)
fdr_labels = {1: "Meget let", 2: "Let", 3: "Middel", 4: "Svær", 5: "Meget svær"}
for i, (fdr, label) in enumerate(fdr_labels.items()):
    with legend_cols[i]:
        color = fdr_color(fdr)
        text = fdr_text_color(fdr)
        st.markdown(
            f'<div style="background:{color};color:{text};padding:8px;border-radius:6px;'
            f'text-align:center;font-weight:600;">FDR {fdr}<br>{label}</div>',
            unsafe_allow_html=True,
        )
with legend_cols[5]:
    st.markdown(
        '<div style="border:2px solid gold;padding:8px;border-radius:6px;'
        'text-align:center;font-weight:600;">🎯 DGW<br>Dobbelt GW</div>',
        unsafe_allow_html=True,
    )

# Best teams to target
st.markdown("---")
st.subheader("🎯 Bedste hold at targete (nemmeste fixtures)")
best_teams = sorted(all_teams, key=lambda t: team_fdr_scores.get(t, 3))[:6]
best_cols = st.columns(6)
for i, tid in enumerate(best_teams):
    with best_cols[i]:
        name = team_map.get(tid, "?")
        avg = team_fdr_scores.get(tid, 3)
        color = fdr_color(round(avg))
        st.markdown(
            f'<div style="background:{color};color:white;padding:12px;border-radius:8px;'
            f'text-align:center;font-weight:700;">{name}<br>FDR {avg:.2f}</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")
st.caption(f"📅 Fixture data fra FPL API | GW{gw_start}–GW{gw_end}")
