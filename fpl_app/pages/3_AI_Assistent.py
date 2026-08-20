"""Explain why generative recommendations are paused during model validation."""

import streamlit as st

from utils.ui import inject_css


st.set_page_config(page_title="AI-forklaring", layout="wide")
inject_css()
st.title("🤖 AI-forklaring")
st.warning("Automatiske AI-beslutninger er midlertidigt sat på pause.")

st.write(
    "En sprogmodel kan formulere et overbevisende FPL-råd uden at opdage, at en prognose er "
    "forældet, at en blank gameweek mangler, eller at en transfer bryder budgettet. Derfor bliver "
    "AI først genaktiveret som forklaringslag, når den underliggende beslutning er beregnet og testet."
)

st.subheader("Krav før genåbning")
st.markdown(
    """
- point-in-time snapshots før hver deadline
- walk-forward-backtests uden datalækage
- lovlige og reproducerbare transfer-/chipscenarier
- kilde, hentetid og usikkerhedsinterval på alle vigtige tal
- AI må forklare beregningen, men må ikke ændre den eller opfinde spillerdata
"""
)

st.page_link("pages/0_Byg_starttrup.py", label="Byg et første hold med den kontrollerede optimering", icon="🧩")
st.page_link("pages/4_Data_Inspector.py", label="Undersøg de rå FPL-data", icon="🔎")
