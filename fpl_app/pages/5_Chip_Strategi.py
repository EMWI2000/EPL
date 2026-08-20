"""2026/27 chip rules and the validation status of chip optimisation."""

import streamlit as st

from domain.rules import CHIPS
from utils.ui import inject_css


st.set_page_config(page_title="Chip-strategi", layout="wide")
inject_css()
st.title("🃏 Chip-strategi 2026/27")
st.warning(
    "Automatiske chipanbefalinger er sat på pause. Den gamle beregning genbrugte næste-GW-point "
    "i senere gameweeks og kunne derfor udpege en forkert uge."
)

st.subheader("Regler, som motoren skal respektere")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Første chipsæt**")
    st.write("Bench Boost og Triple Captain: GW1–19")
    st.write("Wildcard og Free Hit: GW2–19")
with c2:
    st.markdown("**Andet chipsæt**")
    st.write("Alle fire chips: GW20–38")
    st.write("Ubrugte chips fra første halvdel overføres ikke")

st.info(
    f"Der er {CHIPS.chip_sets} chipsæt og højst {CHIPS.max_chips_per_gameweek} chip pr. gameweek. "
    "Free Hit kan ikke spilles i både GW19 og GW20."
)

st.subheader("Sådan bliver den statistiske chipmodel")
st.markdown(
    """
1. Simulér hver kommende gameweek med dens egne spiller- og spilletidsfordelinger.
2. Sammenlign chipscenariet med den bedste plan uden chip — forskellen er chipværdien.
3. Medregn fremtidige transfers, hits, blanks/doubles og værdien af at gemme chippen.
4. Vis usikkerhedsinterval og følsomhed over for udsatte kampe og startchancer.
5. Anbefal kun en chip, når gevinsten er robust på tværs af scenarier.
"""
)

st.caption(
    "Regelsættet er centraliseret og enhedstestet. Næste afhængighed er deadline-snapshots og "
    "walk-forward-backtests af pointprognoserne."
)
