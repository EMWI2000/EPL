"""Configuration helpers that also work when no secrets file exists."""

from __future__ import annotations

import os
from typing import Optional

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


def get_secret(name: str, default: Optional[str] = "") -> Optional[str]:
    """Read a Streamlit secret, then an environment variable, then a default.

    ``st.secrets.get`` raises when no secrets file exists at all. Optional
    configuration must not prevent the public, no-login part of the app from
    starting.
    """
    environment_value = os.getenv(name)
    try:
        value = st.secrets.get(name, environment_value if environment_value is not None else default)
    except (FileNotFoundError, StreamlitSecretNotFoundError):
        value = environment_value if environment_value is not None else default
    return value
