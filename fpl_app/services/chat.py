# services/chat.py
"""
AI-integration med support for både OpenAI og Anthropic (Claude).
Vælger automatisk baseret på tilgængelige API-nøgler.
"""
from __future__ import annotations
import os
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Provider abstraktion
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3,
             max_tokens: int = 1500) -> str: ...


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model

    @property
    def name(self) -> str:
        return f"OpenAI ({self.model})"

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3,
             max_tokens: int = 1500) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post(self.API_URL, json=data, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model

    @property
    def name(self) -> str:
        return f"Claude ({self.model.split('-')[1] if '-' in self.model else self.model})"

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3,
             max_tokens: int = 1500) -> str:
        # Anthropic bruger separat system parameter
        system_parts = []
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                user_messages.append({"role": msg["role"], "content": msg["content"]})

        # Sikr at der er mindst én user message
        if not user_messages:
            user_messages.append({"role": "user", "content": "Hej"})

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": user_messages,
        }
        if system_parts:
            data["system"] = "\n\n".join(system_parts)

        r = requests.post(self.API_URL, json=data, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()["content"][0]["text"]


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def get_provider() -> Optional[LLMProvider]:
    """
    Returnerer den bedste tilgængelige provider.
    Prioritet: Anthropic > OpenAI.
    """
    anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "")
    openai_key = st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")

    if anthropic_key:
        return AnthropicProvider(anthropic_key)
    if openai_key:
        return OpenAIProvider(openai_key)
    return None


def get_provider_name() -> str:
    """Returnerer navnet på den aktive provider."""
    provider = get_provider()
    return provider.name if provider else "Ingen AI konfigureret"


# ---------------------------------------------------------------------------
# Backward-kompatible funktioner
# ---------------------------------------------------------------------------

def suggest_from_ai(openai_key: str, prompt: str) -> str:
    """Simpel one-shot AI forespørgsel (backward-kompatibel)."""
    provider = get_provider()
    if not provider:
        return "Ingen AI API-nøgle sat i .streamlit/secrets.toml."

    messages = [
        {"role": "system", "content": "Du er en dansk FPL-assistent. Svar kort, præcist og handlingsorienteret."},
        {"role": "user", "content": prompt},
    ]
    try:
        return provider.chat(messages, temperature=0.2)
    except Exception as e:
        return f"[AI-fejl] {e}"


def chat_with_ai(
    openai_key: str,
    messages: List[Dict[str, str]],
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1500,
) -> str:
    """Multi-turn chat (backward-kompatibel)."""
    provider = get_provider()
    if not provider:
        return "Ingen AI API-nøgle sat i .streamlit/secrets.toml."

    try:
        return provider.chat(messages, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        return f"[AI-fejl] {e}"


# ---------------------------------------------------------------------------
# Kontekst-building (uændret fra original)
# ---------------------------------------------------------------------------

def build_comprehensive_context(
    manager_info: Dict[str, Any],
    my_team: pd.DataFrame,
    captain_candidates: pd.DataFrame,
    transfer_suggestions: List[Dict[str, Any]],
    double_transfer_suggestions: List[Dict[str, Any]],
    top_candidates: pd.DataFrame,
    horizon: int,
    bank_m: float,
    team_value_m: float,
    used_gw: int,
    odds_status: str = "Ikke aktiveret",
    free_transfers: int = 1,
) -> str:
    """Bygger en omfattende kontekst-streng til AI med alle relevante data."""

    def df_to_text(df: pd.DataFrame, cols: List[str], n: int = 15) -> str:
        if df.empty:
            return "(ingen data)"
        available_cols = [c for c in cols if c in df.columns]
        d = df[available_cols].head(n).copy()
        if "now_cost" in d.columns:
            d["pris_m"] = (pd.to_numeric(d["now_cost"], errors="coerce") / 10.0).round(1)
            d = d.drop(columns=["now_cost"], errors="ignore")
        return d.to_string(index=False)

    def format_transfer(t: Dict, include_delta: bool = True) -> str:
        out = t["out"]
        inp = t["in"]
        out_name = out.get("name", out.get("web_name", "?"))
        in_name = inp.get("name", inp.get("web_name", "?"))
        out_price = float(out.get("now_cost", 0)) / 10.0
        in_price = float(inp.get("now_cost", 0)) / 10.0
        s = f"  Ud: {out_name} ({out.get('team', '?')}, {out.get('pos', '?')}, £{out_price:.1f}m)"
        s += f" -> Ind: {in_name} ({inp.get('team', '?')}, {inp.get('pos', '?')}, £{in_price:.1f}m)"
        if include_delta:
            s += f" | dEP: +{t['delta']:.2f}"
        return s

    def format_double_transfer(t: Dict) -> str:
        lines = []
        for i, (out, inp) in enumerate(zip(t["out"], t["in"])):
            out_name = out.get("name", out.get("web_name", "?"))
            in_name = inp.get("name", inp.get("web_name", "?"))
            lines.append(f"    {i + 1}. {out_name} -> {in_name}")
        hit_info = f" (inkl. -4 hit)" if t.get("hit", 0) > 0 else " (gratis)"
        return "\n".join(lines) + f"\n    dEP: +{t['delta']:.2f}{hit_info}"

    ep_col = f"ep_next{horizon}"

    manager_name = f"{manager_info.get('player_first_name', '')} {manager_info.get('player_last_name', '')}".strip()
    team_name = manager_info.get('name', 'Ukendt hold')

    pos_counts = my_team["pos"].value_counts().to_dict() if "pos" in my_team.columns else {}
    team_counts = my_team["team"].value_counts().head(5).to_dict() if "team" in my_team.columns else {}

    context = f"""
FPL ANALYSE KONTEKST

MANAGER: {manager_name} | Hold: {team_name}
Overall rank: {manager_info.get('summary_overall_rank', 'N/A'):,}
Total point: {manager_info.get('summary_overall_points', 'N/A')}

OKONOMI: Holdvaerdi £{team_value_m:.1f}m | Bank £{bank_m:.1f}m | Gratis transfers: {free_transfers}
Horisont: Naeste {horizon} GWs | Data fra GW{used_gw} | Odds: {odds_status}

DIT HOLD ({len(my_team)} spillere):
Positioner: {pos_counts} | Klubber (top 5): {team_counts}

Spillere (sorteret efter EP):
{df_to_text(my_team.sort_values(ep_col, ascending=False), ['name', 'team', 'pos', 'status', 'ep_next_gw', ep_col, 'form', 'now_cost'], 15)}

KAPTAJN-KANDIDATER:
{df_to_text(captain_candidates, ['name', 'team', 'pos', 'ep_next_gw', 'cap_score', 'form'], 5)}
Anbefalet: {captain_candidates.iloc[0]['name'] if not captain_candidates.empty else 'N/A'}

TRANSFER-FORSLAG (TOP 5):
"""
    if transfer_suggestions:
        for i, t in enumerate(transfer_suggestions[:5], 1):
            context += f"{i}. {format_transfer(t)}\n"
    else:
        context += "Ingen positive forbedringer.\n"

    if double_transfer_suggestions:
        context += "\nDOBBELT-TRANSFERS (TOP 3):\n"
        for i, t in enumerate(double_transfer_suggestions[:3], 1):
            context += f"Forslag {i}:\n{format_double_transfer(t)}\n\n"

    context += f"""
TOP KANDIDATER (ikke i dit hold):
{df_to_text(top_candidates.head(20), ['name', 'team', 'pos', ep_col, 'ep_next_gw', 'form', 'now_cost', 'selected_by_percent'], 20)}
"""
    return context


def get_system_prompt(horizon: int = 5) -> str:
    """Returnerer system prompt til FPL AI assistenten."""
    return f"""Du er en ekspert FPL (Fantasy Premier League) assistent der hjaelper danske managere med at optimere deres hold. Du har dyb viden om FPL-strategi, spillerform, fixtures og statistik.

DINE KERNEKOMPETENCER:
1. Transfer-raadgivning baseret paa EP (Expected Points) og fixture-analyse
2. Kaptajnvalg med fokus paa form, fixtures og set pieces
3. Holdstrategi og chip-timing (Wildcard, Bench Boost, Triple Captain, Free Hit)
4. Budget-optimering og value picks
5. Differentials til at klatre i rank

REGLER FOR DINE SVAR:
- Svar ALTID paa dansk
- Vaer konkret og handlingsorienteret - giv specifikke navne og tal
- Brug data fra konteksten aktivt - referer til EP, form, pris osv.
- Forklar HVORFOR du anbefaler noget (fx "pga. favorable fixtures" eller "form paa 8.2")
- Hold dig til facts fra de data du faar - opfind ikke spillerstatistik
- Naar du naevner transfers, inkluder altid: spiller ud, spiller ind, pris, EP-forskel
- Ved kaptajnvalg, overvej: hjemmebane, modstander FDR, form, set pieces

FORMATTERING:
- Korte afsnit, gerne punktform til lister
- Prioriter de vigtigste raad foerst

ANALYSEHORISONT: Du optimerer for de naeste {horizon} gameweeks.

Naar brugeren stiller spoergsmaal, brug ALTID den medfølgende kontekst med holddata, transfers og kandidater til at give praecise, datadrevne svar."""


def create_fpl_chat_messages(
    context: str,
    user_message: str,
    chat_history: List[Dict[str, str]],
    horizon: int = 5,
) -> List[Dict[str, str]]:
    """Opretter den fulde message-liste til AI API."""
    messages = [
        {"role": "system", "content": get_system_prompt(horizon)},
        {"role": "system", "content": f"Her er den aktuelle data for brugerens hold:\n\n{context}"},
    ]

    for msg in chat_history[-10:]:
        if msg["role"] in ["user", "assistant"]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    if not chat_history or chat_history[-1].get("content") != user_message:
        messages.append({"role": "user", "content": user_message})

    return messages


def get_quick_suggestions(context: str, openai_key: str) -> List[str]:
    """Genererer hurtige forslag baseret på konteksten."""
    prompt = f"""Baseret paa denne FPL-kontekst, generer 4 relevante spoergsmaal som brugeren kunne stille.
Spoergsmaalene skal vaere korte (max 8 ord) og handlingsorienterede.
Returner kun spoergsmaalene, et per linje, uden nummerering.

{context[:2000]}"""

    try:
        response = chat_with_ai(openai_key, [
            {"role": "system", "content": "Du genererer korte FPL-spoergsmaal paa dansk."},
            {"role": "user", "content": prompt},
        ], temperature=0.7, max_tokens=200)

        suggestions = [s.strip() for s in response.split("\n") if s.strip()]
        return suggestions[:4]
    except Exception:
        return [
            "Hvem skal vaere kaptajn?",
            "Hvilken transfer skal jeg lave?",
            "Hvordan ser mine fixtures ud?",
            "Er der gode budget-picks?",
        ]
