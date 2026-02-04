# pages/3_AI_Assistent.py
import streamlit as st
import pandas as pd

from services.fpl_api import manager_summary
from services.data_layer import (
    load_base_data, get_picks_any, build_odds_context_for_fixtures,
    build_my_team_df, build_candidates_df,
)
from services.chat import chat_with_ai, build_comprehensive_context, create_fpl_chat_messages, get_provider_name
from logic.captain import captain_score
from logic.optimizer import best_n_transfers_with_quotas, best_two_transfers
from utils.session import manager_id_input, get_manager_id
from utils.ui import inject_css

st.set_page_config(page_title="AI-Assistent", layout="wide")
inject_css()
st.title("🤖 AI-Assistent – Chat med dine FPL-analyser")
st.caption("Få personlig rådgivning baseret på dit holds data, fixtures og statistik.")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

bs, events, els, fixt, teams_df = load_base_data()

with st.sidebar:
    st.header("⚙️ Indstillinger")
    entry_id = manager_id_input()
    horizon = st.slider("Horisont (antal runder)", 1, 5, 5)
    use_odds = st.toggle("Brug odds i beregninger", value=False)

    st.markdown("---")
    odds_key = st.secrets.get("THE_ODDS_API_KEY", "")
    openai_key = st.secrets.get("OPENAI_API_KEY", "")
    anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "")

    provider_name = get_provider_name()
    if provider_name == "Ingen AI konfigureret":
        st.error("⚠️ Ingen AI API-nøgle fundet")
    else:
        st.success(f"🤖 AI: **{provider_name}**")

    st.markdown("---")
    if st.button("🗑️ Nulstil chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()

    with st.expander("ℹ️ Om AI-assistenten"):
        st.markdown(f"""
        AI-assistenten bruger **{provider_name}** og har adgang til:
        - Dit komplette hold med EP-beregninger
        - Kaptajn-anbefalinger med scoring
        - Transfer-forslag (enkelt og dobbelt)
        - Kandidater til køb
        - Fixture-analyse
        """)

entry_id = get_manager_id()
if not entry_id:
    st.info("👆 Indtast dit manager-ID i sidepanelet for at fortsætte.")
    st.stop()

if not openai_key and not anthropic_key:
    st.error("❌ Ingen AI API-nøgle fundet i secrets.toml - AI-assistenten kan ikke fungere.")
    st.stop()

try:
    mgr = manager_summary(int(entry_id))
except Exception as e:
    st.error(f"❌ Kunne ikke hente manager-data for ID {entry_id}. Fejl: {e}")
    st.stop()

ep_col = f"ep_next{horizon}"

# Picks
preferred_gw = int(events.loc[events["is_next"] == True, "id"].iloc[0]) if (events["is_next"] == True).any() else 1
picks, used_gw, _ = get_picks_any(int(entry_id), preferred_gw, events)
if not picks:
    st.error("❌ Fandt ingen picks at arbejde med.")
    st.stop()
if used_gw != preferred_gw:
    st.info(f"📅 Bruger dine picks fra **GW{used_gw}** (da GW{preferred_gw} ikke var tilgængelig).")

# Odds
odds_ctx_by_fixture, odds_status = build_odds_context_for_fixtures(odds_key, fixt, teams_df, use_odds)
if use_odds and "Fejl" in odds_status:
    st.warning("⚠️ Kunne ikke hente odds – fortsætter uden.")

# Hold
my_df = build_my_team_df(els, fixt, picks, horizon, odds_ctx_by_fixture, teams_df, include_details=True)
if my_df.empty:
    st.error("❌ Ingen spillere fundet.")
    st.stop()

my_df["cap_score"] = my_df.apply(captain_score, axis=1)
cap_top = my_df.sort_values("cap_score", ascending=False).head(5)

# Kandidater
cand_df = build_candidates_df(els, my_df["id"].tolist(), fixt, horizon, odds_ctx_by_fixture, teams_df, max_cost_extra=15)

team_value_m = float(my_df["now_cost"].sum()) / 10.0
bank_tenths = int(mgr.get("last_deadline_bank", 0) or 0)
bank_m = bank_tenths / 10.0

single_transfers = best_n_transfers_with_quotas(my_df, cand_df, bank_tenths, team_value_m, horizon, top_n=5)
double_transfers = best_two_transfers(my_df, cand_df, bank_tenths, team_value_m, horizon, top_n=3)

# Byg AI-kontekst
context = build_comprehensive_context(
    manager_info=mgr,
    my_team=my_df,
    captain_candidates=cap_top,
    transfer_suggestions=single_transfers,
    double_transfer_suggestions=double_transfers,
    top_candidates=cand_df.head(25),
    horizon=horizon,
    bank_m=bank_m,
    team_value_m=team_value_m,
    used_gw=used_gw,
    odds_status=odds_status,
    free_transfers=1,
)

with st.expander("📋 Se data sendt til AI", expanded=False):
    st.text(context)

st.markdown("---")

# Quick suggestions
if not st.session_state.chat_messages:
    st.markdown("### 💬 Hvad vil du vide?")
    st.caption("Klik på et forslag eller skriv dit eget spørgsmål:")

    suggestions = [
        "Hvem skal være kaptajn denne uge?",
        "Hvilken transfer giver mest værdi?",
        "Hvordan ser mine fixtures ud?",
        "Er der gode budget-spillere jeg bør overveje?",
        "Skal jeg tage en -4 hit for en dobbelt-transfer?",
        "Hvilke spillere i mit hold performer dårligt?",
    ]

    cols = st.columns(3)
    for i, sugg in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(sugg, key=f"sugg_{i}", use_container_width=True):
                st.session_state.chat_messages.append({"role": "user", "content": sugg})
                st.rerun()

# Chat historik
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Stil et spørgsmål om dit FPL-hold..."):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    messages = create_fpl_chat_messages(
        context=context,
        user_message=prompt,
        chat_history=st.session_state.chat_messages,
        horizon=horizon,
    )

    with st.chat_message("assistant"):
        with st.spinner("🤔 AI'en analyserer dine data..."):
            try:
                response = chat_with_ai(
                    openai_key=openai_key,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1500,
                )
                st.markdown(response)
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"❌ Fejl ved AI-kald: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Dit holds EP", f"{my_df[ep_col].sum():.1f}")
with col2:
    st.metric("Anbefalet C", cap_top.iloc[0]["name"] if not cap_top.empty else "N/A")
with col3:
    st.metric("Bedste transfer ΔEP", f"+{single_transfers[0]['delta']:.1f}" if single_transfers else "0")
with col4:
    st.metric("Bank", f"£{bank_m:.1f}m")
