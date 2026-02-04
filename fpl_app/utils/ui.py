# utils/ui.py
"""Delte UI-komponenter og styling."""
from __future__ import annotations
from pathlib import Path
import streamlit as st


def inject_css():
    """Injecter custom CSS på siden."""
    css_path = Path(__file__).parent.parent / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def fdr_color(fdr: int) -> str:
    """Returnerer farve for FDR-værdi."""
    colors = {
        1: "#257d5a",  # mørkegrøn
        2: "#01fc7a",  # lysegrøn
        3: "#e7e7e7",  # grå
        4: "#ff1751",  # rød
        5: "#80072d",  # mørkerød
    }
    return colors.get(fdr, "#e7e7e7")


def fdr_text_color(fdr: int) -> str:
    """Returnerer tekst-farve for FDR-værdi."""
    return "white" if fdr in (1, 5) else "#1a1a2e"


def fdr_badge_html(opponent: str, fdr: int, is_home: bool) -> str:
    """Returnerer HTML badge for en fixture."""
    bg = fdr_color(fdr)
    text = fdr_text_color(fdr)
    venue = "H" if is_home else "A"
    return (
        f'<span style="background:{bg};color:{text};padding:3px 8px;'
        f'border-radius:4px;font-size:0.85em;font-weight:600;'
        f'display:inline-block;min-width:60px;text-align:center;">'
        f'{opponent} ({venue})</span>'
    )


def fdr_cell_html(opponent: str, fdr: int, is_home: bool, is_dgw: bool = False) -> str:
    """Returnerer HTML celle for fixture planner."""
    bg = fdr_color(fdr)
    text = fdr_text_color(fdr)
    venue = "H" if is_home else "A"
    border = "border: 2px solid gold;" if is_dgw else ""
    return (
        f'<td style="background:{bg};color:{text};text-align:center;'
        f'padding:6px 4px;font-size:0.8em;font-weight:600;{border}">'
        f'{opponent}<br><small>({venue})</small></td>'
    )


def empty_cell_html() -> str:
    """Tom celle til fixture planner (BGW)."""
    return '<td style="background:#f5f5f5;text-align:center;padding:6px 4px;color:#ccc;">-</td>'
