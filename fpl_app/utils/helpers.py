# utils/helpers.py
"""Delte hjælpefunktioner til visning og formatering."""
from __future__ import annotations
import pandas as pd


def safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Gør DataFrame sikker for visning i Streamlit (undgår React #185)."""
    if df is None or getattr(df, "empty", False):
        return pd.DataFrame()
    out = df.copy()
    for c in out.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(out[c]) or pd.api.types.is_timedelta64_dtype(out[c]):
                out[c] = out[c].astype(str)
            else:
                out[c] = out[c].apply(
                    lambda x: x if isinstance(x, (int, float, str, bool, type(None))) else str(x)
                )
        except Exception:
            out[c] = out[c].astype(str)
    return out.reset_index(drop=True)


def safe_event_id(df: pd.DataFrame, col: str, default: int = 1) -> int:
    """Hent event-id fra events DataFrame baseret på boolesk kolonne."""
    try:
        mask = df[col] == True
        if mask.any():
            return int(df.loc[mask, "id"].iloc[0])
    except Exception:
        pass
    return default


def fmt_prompt_table(df: pd.DataFrame, cols: list[str], n: int = 15) -> str:
    """Formaterer DataFrame til tekst til AI-prompt."""
    if df.empty:
        return "(ingen data)"
    d = df.loc[:, [c for c in cols if c in df.columns]].head(n).copy()
    if "now_cost" in d.columns and "pris_mio" not in d.columns:
        d["pris_mio"] = (pd.to_numeric(d["now_cost"], errors="coerce") / 10.0).round(1)
        base_cols = [c for c in cols if c in d.columns and c != "now_cost"]
        d = d[base_cols + ["pris_mio"]]
    return d.to_string(index=False)
