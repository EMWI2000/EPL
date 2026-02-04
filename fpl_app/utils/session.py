# utils/session.py
"""
Delt session-håndtering for manager ID på tværs af alle sider.
Importeres i alle pages for konsistent manager ID håndtering.
"""
from __future__ import annotations
import streamlit as st
from typing import Optional


def init_manager_id() -> str:
    """
    Initialiserer og returnerer manager-ID i st.session_state.entry_id.
    Tjekker i rækkefølgen:
    1) Allerede sat i session_state
    2) URL query params (?entry_id=123456)
    3) DEFAULT_MANAGER_ID i .streamlit/secrets.toml
    4) Fallback: tom streng
    """
    # 1) Fra session_state (allerede sat)
    if "entry_id" in st.session_state and st.session_state.entry_id:
        return str(st.session_state.entry_id)

    # 2) Fra URL query params
    entry_from_qp = ""
    try:
        qp = st.query_params
        if "entry_id" in qp:
            entry_from_qp = str(qp["entry_id"])
    except Exception:
        try:
            qp = st.experimental_get_query_params()
            if "entry_id" in qp and qp["entry_id"]:
                entry_from_qp = str(qp["entry_id"][0])
        except Exception:
            pass

    if entry_from_qp:
        st.session_state.entry_id = entry_from_qp
        return entry_from_qp

    # 3) Fra secrets (DEFAULT_MANAGER_ID)
    default_id = str(st.secrets.get("DEFAULT_MANAGER_ID", "") or "").strip()
    if default_id:
        st.session_state.entry_id = default_id
        return default_id

    # 4) Tom
    st.session_state.entry_id = ""
    return ""


def get_manager_id() -> str:
    """Returnerer nuværende manager ID fra session state."""
    return str(st.session_state.get("entry_id", "") or "")


def set_manager_id(entry_id: str) -> None:
    """Sætter manager ID i session state og URL params."""
    st.session_state.entry_id = entry_id
    try:
        st.query_params["entry_id"] = entry_id
    except Exception:
        try:
            st.experimental_set_query_params(entry_id=entry_id)
        except Exception:
            pass


def manager_id_input(label: str = "Dit FPL manager-ID", placeholder: str = "fx 1499152") -> str:
    """
    Viser input-felt til manager ID med automatisk session binding.
    Returnerer det aktuelle manager ID.
    """
    # Initialiser først
    init_manager_id()

    # Vis input felt bundet til session state
    entry_id = st.text_input(
        label,
        key="entry_id",
        placeholder=placeholder,
    )

    # Opdater URL params
    if entry_id:
        try:
            st.query_params["entry_id"] = entry_id
        except Exception:
            try:
                st.experimental_set_query_params(entry_id=entry_id)
            except Exception:
                pass

    return entry_id


def require_manager_id() -> Optional[str]:
    """
    Tjekker om manager ID er sat. Hvis ikke, viser info og stopper.
    Returnerer manager ID hvis sat, ellers None (og stopper execution).
    """
    entry_id = get_manager_id()
    if not entry_id:
        st.info("Indtast dit manager-ID i sidepanelet for at fortsætte.")
        st.stop()
        return None
    return entry_id
