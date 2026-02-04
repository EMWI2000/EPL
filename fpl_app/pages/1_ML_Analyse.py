# pages/1_ML_Analyse.py
import streamlit as st
import pandas as pd
import altair as alt
import warnings

from services.data_layer import load_base_data, build_odds_context_for_fixtures
from logic.features import expected_points_for_player, get_player_detailed_stats, price_change_indicator
from utils.session import init_manager_id
from utils.ui import inject_css


# Helper: sikrer pandas DataFrame til Altair
def _as_pandas(df):
    if hasattr(df, "to_native"):
        try:
            df = df.to_native()
        except Exception:
            pass
    if hasattr(df, "to_pandas"):
        try:
            return df.to_pandas()
        except Exception:
            pass
    try:
        if isinstance(df, pd.DataFrame):
            return df
        return pd.DataFrame(df)
    except Exception:
        return df


warnings.filterwarnings(
    "ignore",
    message=r"You passed a `<class 'narwhals\.stable\.v1\.DataFrame'>` to `is_pandas_dataframe`",
    category=UserWarning,
    module="altair.utils.data",
)

st.set_page_config(page_title="ML-analyse", layout="wide")
inject_css()
st.title("🧠 ML-inspireret analyse (1–5 GW)")
st.caption("Avanceret prediktion af forventede point baseret på xG, xA, ICT, fixtures og odds.")

bs, events, els, fixt, teams_df = load_base_data()

with st.sidebar:
    st.header("⚙️ Indstillinger")
    init_manager_id()

    default_gw = int(events.loc[events["is_next"] == True, "id"].iloc[0]) if (events["is_next"] == True).any() else 1

    horizon = st.slider("Horisont (antal runder)", 1, 5, 5)
    use_odds = st.toggle("Brug odds", value=True)
    odds_key = st.secrets.get("THE_ODDS_API_KEY", "")

    st.markdown("---")
    st.subheader("Filtre")
    positions = st.multiselect("Positioner", ["GKP", "DEF", "MID", "FWD"], default=["GKP", "DEF", "MID", "FWD"])
    min_price, max_price = st.slider("Pris (£m)", 3.5, 15.0, (3.5, 15.0), 0.5)
    min_form = st.slider("Minimum form", 0.0, 10.0, 0.0, 0.5)
    only_available = st.toggle("Kun tilgængelige spillere", value=True)
    show_dgw_only = st.toggle("Vis kun DGW-spillere", value=False)

# Odds
odds_ctx_by_fixture, odds_status = build_odds_context_for_fixtures(odds_key, fixt, teams_df, use_odds)
if use_odds and "Fejl" in odds_status:
    st.warning("⚠️ Kunne ikke hente/parse odds – fortsætter uden odds.")

ep_col = f"EP næste {horizon}"

# Beregn EP for ALLE spillere
with st.spinner("Beregner forventede point for alle spillere..."):
    rows = []
    for _, pl in els.iterrows():
        ep = expected_points_for_player(pl, fixt, n=horizon, odds_ctx_by_fixture=odds_ctx_by_fixture,
                                        teams_table=teams_df)
        total = ep["total_next_n"]
        per_gw = ep["per_gw"]
        ep_next_gw = per_gw[0]["ep"] if per_gw else 0.0

        stats = get_player_detailed_stats(pl, fixt, n=horizon)
        price_mult = price_change_indicator(pl)

        rows.append({
            "id": int(pl["id"]),
            "Navn": str(pl["web_name"]),
            "Hold": str(pl["short_name"]),
            "Pos": str(pl["singular_name_short"]),
            "Pris": float(pl["now_cost"]) / 10.0,
            "Status": str(pl["status"]),
            "Form": float(pl.get("form", 0) or 0),
            "EP nxt GW": round(float(ep_next_gw), 2),
            ep_col: round(float(total), 2),
            "xG": round(stats.get("xG", 0), 2),
            "xA": round(stats.get("xA", 0), 2),
            "ICT": round(stats.get("ict_index", 0), 1),
            "FDRs": stats.get("fdrs", []),
            "Fixture Quality": stats.get("fixture_run_quality", 1.0),
            "DGW": "🎯" if stats.get("has_dgw", False) else "",
            "has_dgw": stats.get("has_dgw", False),
            "⚽ Penalty": "✅" if stats.get("is_penalty_taker", False) else "",
            "🎯 Set piece": "✅" if stats.get("is_set_piece_taker", False) else "",
            "Net Transfers": stats.get("net_transfers", 0),
            "Trend": "🔥" if stats.get("is_rising", False) else ("📉" if stats.get("is_falling", False) else ""),
            "Ownership": round(stats.get("selected_by_percent", 0), 1),
        })

pred_df = pd.DataFrame(rows)
pred_df["ROI"] = (pred_df[ep_col] / pred_df["Pris"]).replace([float("inf")], 0.0).round(2)

# Filtrering
f = pred_df[pred_df["Pos"].isin(positions)]
f = f[(f["Pris"] >= min_price) & (f["Pris"] <= max_price)]
f = f[f["Form"] >= min_form]

if only_available:
    f = f[~f["Status"].isin(["i", "s", "u"])]

if show_dgw_only:
    f = f[f["has_dgw"] == True]

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Spillere analyseret", len(pred_df))
with col2:
    st.metric("Efter filtre", len(f))
with col3:
    dgw_count = len(f[f["has_dgw"] == True])
    st.metric("Med DGW", dgw_count)
with col4:
    st.metric("Odds status", odds_status)

# Toplister
st.markdown("---")
colA, colB = st.columns(2)

display_cols = ["Navn", "Hold", "Pos", "Pris", "Form", ep_col, "ROI", "DGW", "⚽ Penalty", "Trend"]

with colA:
    st.subheader("🏆 Top 15 – Forventede Point")
    top_pts = f.nlargest(15, ep_col)
    st.dataframe(
        top_pts[display_cols].reset_index(drop=True),
        use_container_width=True,
        column_config={
            "Pris": st.column_config.NumberColumn("Pris (£m)", format="%.1f"),
            "Form": st.column_config.NumberColumn("Form", format="%.1f"),
            ep_col: st.column_config.NumberColumn(ep_col, format="%.2f"),
            "ROI": st.column_config.NumberColumn("ROI", format="%.2f"),
        },
    )

with colB:
    st.subheader("💰 Top 15 – ROI (Value)")
    top_roi = f.nlargest(15, "ROI")
    st.dataframe(
        top_roi[display_cols].reset_index(drop=True),
        use_container_width=True,
        column_config={
            "Pris": st.column_config.NumberColumn("Pris (£m)", format="%.1f"),
            "Form": st.column_config.NumberColumn("Form", format="%.1f"),
            ep_col: st.column_config.NumberColumn(ep_col, format="%.2f"),
            "ROI": st.column_config.NumberColumn("ROI", format="%.2f"),
        },
    )

# DGW
st.markdown("---")
st.subheader("🎯 Double Gameweek Spillere")
dgw_players = f[f["has_dgw"] == True].nlargest(15, ep_col)
if not dgw_players.empty:
    st.dataframe(
        dgw_players[["Navn", "Hold", "Pos", "Pris", "Form", ep_col, "ROI", "⚽ Penalty"]].reset_index(drop=True),
        use_container_width=True,
    )
else:
    st.info(f"Ingen spillere med DGW i de næste {horizon} runder (eller filtreret væk).")

# Budget picks
st.markdown("---")
st.subheader("🔍 Budget Picks (under £6.0m)")
budget_picks = f[f["Pris"] < 6.0].nlargest(10, ep_col)
if not budget_picks.empty:
    st.dataframe(
        budget_picks[["Navn", "Hold", "Pos", "Pris", "Form", ep_col, "ROI", "Ownership"]].reset_index(drop=True),
        use_container_width=True,
    )

# Premium picks
st.subheader("👑 Premium Picks (over £10.0m)")
premium_picks = f[f["Pris"] >= 10.0].nlargest(10, ep_col)
if not premium_picks.empty:
    st.dataframe(
        premium_picks[["Navn", "Hold", "Pos", "Pris", "Form", ep_col, "ROI", "⚽ Penalty", "Ownership"]].reset_index(drop=True),
        use_container_width=True,
    )

# Visualiseringer
st.markdown("---")
st.subheader("📊 Visualiseringer")

c1, c2 = st.columns(2)

with c1:
    st.caption("Pris vs Forventede Point (størrelse = ROI)")
    chart_data = _as_pandas(f.head(100))
    chart = alt.Chart(chart_data).mark_circle(opacity=0.7).encode(
        x=alt.X("Pris:Q", title="Pris (£m)"),
        y=alt.Y(f"{ep_col}:Q", title=f"EP ({horizon} GW)"),
        size=alt.Size("ROI:Q", legend=None, scale=alt.Scale(range=[50, 500])),
        color=alt.Color("Pos:N", legend=alt.Legend(title="Position")),
        tooltip=["Navn", "Hold", "Pos", "Pris", ep_col, "ROI", "Form"],
    ).interactive().properties(height=400)
    st.altair_chart(chart, use_container_width=True)

with c2:
    st.caption("Top 20 spillere efter EP")
    top20 = f.nlargest(20, ep_col)
    bar = alt.Chart(_as_pandas(top20)).mark_bar().encode(
        x=alt.X(f"{ep_col}:Q", title=f"EP ({horizon} GW)"),
        y=alt.Y("Navn:N", sort="-x", title=""),
        color=alt.Color("Pos:N"),
        tooltip=["Navn", "Hold", "Pos", "Pris", ep_col, "ROI"],
    ).properties(height=400)
    st.altair_chart(bar, use_container_width=True)

# Position breakdown
st.markdown("---")
st.subheader("📈 Gennemsnit per Position")
pos_stats = f.groupby("Pos").agg({ep_col: "mean", "ROI": "mean", "Pris": "mean", "Form": "mean"}).round(2)
st.dataframe(pos_stats, use_container_width=True)

# Trending
st.markdown("---")
st.subheader("🔥 Trending Spillere (høj net transfers)")
trending = f[f["Net Transfers"] > 20000].nlargest(10, "Net Transfers")
if not trending.empty:
    trending_display = trending[["Navn", "Hold", "Pos", "Pris", ep_col, "Net Transfers", "Ownership"]].copy()
    trending_display["Net Transfers"] = trending_display["Net Transfers"].apply(lambda x: f"+{x:,}")
    st.dataframe(trending_display.reset_index(drop=True), use_container_width=True)
else:
    st.info("Ingen spillere med markant positive net transfers i de filtrerede data.")

# Fuld tabel
st.markdown("---")
st.subheader("📋 Komplet Tabel")
st.caption("Filtrér og sortér som du vil")

all_display_cols = [
    "Navn", "Hold", "Pos", "Pris", "Status", "Form", "EP nxt GW", ep_col, "ROI",
    "xG", "xA", "ICT", "DGW", "⚽ Penalty", "🎯 Set piece", "Trend", "Ownership",
]
st.dataframe(
    f[all_display_cols].sort_values(ep_col, ascending=False).reset_index(drop=True),
    use_container_width=True,
    height=600,
)

st.markdown("---")
st.caption(f"📊 Data fra FPL API | Odds: {odds_status} | Horisont: {horizon} GW")
