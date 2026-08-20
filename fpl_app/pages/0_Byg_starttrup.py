"""Build a legal initial FPL squad without requiring a manager id."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from logic.features import expected_points_for_player, gameweek_window
from logic.initial_squad import (
    DEFAULT_GW_WEIGHTS,
    InitialSquadError,
    optimize_initial_squad,
)
from logic.projections import overlay_solio_projections
from services.data_layer import load_base_data
from services.solio import SolioClient, SolioError
from utils.ui import inject_css


st.set_page_config(page_title="Byg starttrup", layout="wide")
inject_css()
st.title("🧩 Byg et nyt starthold")
st.caption("Et lovligt 15-mandshold, start-XI, kaptajn, vicekaptajn og bænk — uden manager-ID.")
st.warning(
    "Prognosen er eksperimentel. Næste GW kan suppleres med sikkert matchede Solio-projektioner; "
    "øvrige værdier er en intern heuristisk baseline. Brug resultatet som et gennemsigtigt "
    "første udkast, ikke som en garanti eller et færdigvalideret facit."
)

bs, events, elements, fixtures, teams = load_base_data()

with st.sidebar:
    st.header("Optimering")
    horizon = st.slider("Horisont", min_value=1, max_value=5, value=5, help="Antal kommende gameweeks")
    include_doubtful = st.toggle("Medtag tvivlsomme spillere", value=True)
    use_solio = st.toggle(
        "Brug Solio til næste GW",
        value=True,
        help="Overlay'er kun de spillere, som kan matches entydigt til Solios offentlige feed.",
    )
    st.caption("Budget: £100,0m · 2 GKP · 5 DEF · 5 MID · 3 FWD · maks. 3 pr. klub")

window = gameweek_window(fixtures, n=horizon)
if not window:
    st.error("Der er ingen planlagte kommende gameweeks i FPL-data endnu.")
    st.stop()


@st.cache_data(ttl=14_400, show_spinner=False)
def load_solio_projection():
    result = SolioClient().fetch_latest()
    return dict(result.payload), result.gameweek, result.generated_at.isoformat()


@st.cache_data(ttl=900, show_spinner=False)
def build_forecast_pool(
    player_table: pd.DataFrame,
    fixture_table: pd.DataFrame,
    team_table: pd.DataFrame,
    forecast_horizon: int,
) -> pd.DataFrame:
    rows = []
    for _, player in player_table.iterrows():
        forecast = expected_points_for_player(
            player,
            fixture_table,
            n=forecast_horizon,
            teams_table=team_table,
            use_ml=False,
        )
        row = {
            "id": int(player["id"]),
            "name": str(player["web_name"]),
            "team_id": int(player["team_id"]),
            "team": str(player["short_name"]),
            "pos": str(player["singular_name_short"]),
            "now_cost": int(player["now_cost"]),
            "status": str(player.get("status", "a")),
        }
        for offset, gameweek in enumerate(forecast["per_gw"], start=1):
            row[f"ep_gw{offset}"] = float(gameweek["ep"])
        rows.append(row)
    return pd.DataFrame(rows)


with st.spinner("Beregner spillerprognoser og løser holdoptimeringen …"):
    pool = build_forecast_pool(elements, fixtures, teams, horizon)
    solio_metadata = None
    if use_solio:
        try:
            solio_payload, solio_gameweek, solio_generated_at = load_solio_projection()
            if solio_gameweek != window[0]:
                st.warning(
                    f"Solio-feedet er for GW{solio_gameweek}, mens analysevinduet starter i "
                    f"GW{window[0]}. Feedet er derfor ikke lagt ind."
                )
            else:
                overlay = overlay_solio_projections(elements, solio_payload)
                projected = overlay.players.set_index("id")[overlay.output_column]
                pool["intern_ep_gw1"] = pool["ep_gw1"]
                pool["solio_ep_gw1"] = pool["id"].map(projected)
                pool["gw1_source"] = "Intern baseline"
                matched = pool["solio_ep_gw1"].notna()
                pool.loc[matched, "ep_gw1"] = pool.loc[matched, "solio_ep_gw1"].astype(float)
                pool.loc[matched, "gw1_source"] = "Solio Analytics"
                solio_metadata = {
                    "gameweek": solio_gameweek,
                    "generated_at": solio_generated_at,
                    "matched": overlay.diagnostics.matched_projection_count,
                    "usable": overlay.diagnostics.usable_projection_count,
                    "ambiguous": len(overlay.diagnostics.ambiguous),
                    "unmatched": len(overlay.diagnostics.unmatched),
                }
        except SolioError as exc:
            st.warning(f"Solio kunne ikke hentes eller valideres; bruger intern baseline. ({exc})")

    allowed_statuses = {"a", "d"} if include_doubtful else {"a"}
    eligible = pool[pool["status"].isin(allowed_statuses)].copy()
    forecast_columns = [f"ep_gw{offset}" for offset in range(1, horizon + 1)]
    try:
        result = optimize_initial_squad(
            eligible,
            horizon=horizon,
            forecast_columns=forecast_columns,
            gw_weights=DEFAULT_GW_WEIGHTS[:horizon],
        )
    except InitialSquadError as exc:
        st.error(f"Kunne ikke bygge et lovligt hold: {exc}")
        st.stop()

if solio_metadata:
    st.info(
        f"GW{solio_metadata['gameweek']}: {solio_metadata['matched']} af "
        f"{solio_metadata['usable']} unikke Solio-projektioner blev matchet entydigt. "
        "Umatchede spillere og senere gameweeks bruger den interne baseline. "
        "Kilde: [Solio Analytics](https://fpl.solioanalytics.com/)."
    )
    st.caption(
        f"Solio genereret {solio_metadata['generated_at']} · "
        f"{solio_metadata['unmatched']} uden match · {solio_metadata['ambiguous']} tvetydige."
    )

by_id = eligible.set_index("id")
squad = by_id.loc[list(result.squad_ids)].copy()
weights = result.gw_weights
squad["Vægtet EP"] = sum(
    squad[f"ep_gw{offset}"] * weights[offset - 1]
    for offset in range(1, horizon + 1)
)
squad["Pris"] = squad["now_cost"] / 10.0
squad["Rolle"] = "Start-XI"
squad.loc[list(result.bench_ids), "Rolle"] = "Bænk"
squad.loc[result.captain_id, "Rolle"] += " · C"
squad.loc[result.vice_captain_id, "Rolle"] += " · VC"

bench_order = {player_id: order for order, player_id in enumerate(result.bench_ids, start=1)}
squad["Bænk nr."] = [bench_order.get(int(player_id), "") for player_id in squad.index]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pris", f"£{result.total_cost_tenths / 10:.1f}m")
c2.metric("I banken", f"£{result.bank_tenths / 10:.1f}m")
c3.metric("Formation", result.formation)
c4.metric("Analyse", f"GW{window[0]}–GW{window[-1]}")

st.subheader("Anbefalet start-XI")
starters = squad.loc[list(result.starting_ids)].copy()
position_order = pd.Categorical(starters["pos"], ["GKP", "DEF", "MID", "FWD"], ordered=True)
starters = starters.assign(_position_order=position_order).sort_values(
    ["_position_order", "Vægtet EP"], ascending=[True, False]
)
st.dataframe(
    starters[["name", "team", "pos", "Pris", "Rolle", "Vægtet EP"]].rename(
        columns={"name": "Spiller", "team": "Hold", "pos": "Pos"}
    ),
    width="stretch",
    hide_index=True,
)

st.subheader("Bænk")
bench = squad.loc[list(result.bench_ids)].copy()
st.dataframe(
    bench[["Bænk nr.", "name", "team", "pos", "Pris", "Vægtet EP"]].rename(
        columns={"name": "Spiller", "team": "Hold", "pos": "Pos"}
    ),
    width="stretch",
    hide_index=True,
)

with st.expander("Se gameweek-prognoser og metode"):
    gw_labels = {
        f"ep_gw{offset}": f"GW{event} EP"
        for offset, event in enumerate(window, start=1)
    }
    detail_columns = ["name", "team", "pos", "Pris", *forecast_columns, "Vægtet EP"]
    if "gw1_source" in squad.columns:
        detail_columns.insert(4, "gw1_source")
    st.dataframe(
        squad[detail_columns]
        .rename(columns={"name": "Spiller", "team": "Hold", "pos": "Pos", **gw_labels})
        .sort_values("Vægtet EP", ascending=False),
        width="stretch",
    )
    st.write(
        "Målfunktionen vægter de tidligste gameweeks højest, giver en lille diskonteret "
        "værdi til bænken og medregner kaptajnens ekstra point. Alle FPL-formationer er tilladt."
    )
    if solio_metadata:
        st.caption(
            "Solio-data er et delvist offentligt leaderboard-feed, ikke en komplet spillerfil. "
            "Overlay'et kræver samme hold, position og pris samt et entydigt navnematch."
        )

download = result.as_dict()
download["projection_sources"] = {
    "internal": "experimental_heuristic_baseline",
    "solio": solio_metadata,
}
download["players"] = {
    str(player_id): {
        "name": str(by_id.at[player_id, "name"]),
        "team": str(by_id.at[player_id, "team"]),
        "position": str(by_id.at[player_id, "pos"]),
    }
    for player_id in result.squad_ids
}
st.download_button(
    "Download holdforslag (JSON)",
    data=json.dumps(download, ensure_ascii=False, indent=2),
    file_name=f"fpl-starttrup-gw{window[0]}.json",
    mime="application/json",
)
st.caption("Overfør altid valgene manuelt i FPL, og genberegn efter skader, pressemøder og prisændringer.")
