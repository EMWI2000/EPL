# services/fpl_api.py
from __future__ import annotations
import requests
from functools import lru_cache
from typing import Dict, Any, List

BASE = "https://fantasy.premierleague.com/api"

def _get(url: str) -> Any:
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            raise requests.HTTPError(f"404 Not Found for URL: {url}")
        r.raise_for_status()
        return r.json()
    except requests.HTTPError:
        raise
    except Exception:
        raise

@lru_cache(maxsize=1)
def bootstrap_static() -> Dict[str, Any]:
    return _get(f"{BASE}/bootstrap-static/")

@lru_cache(maxsize=8)
def fixtures(future_only: bool = True) -> List[Dict[str, Any]]:
    url = f"{BASE}/fixtures/"
    if future_only:
        url += "?future=1"
    return _get(url)

def entry_picks(entry_id: int, event_id: int) -> Dict[str, Any]:
    return _get(f"{BASE}/entry/{entry_id}/event/{event_id}/picks/")

def manager_summary(entry_id: int) -> Dict[str, Any]:
    return _get(f"{BASE}/entry/{entry_id}/")

@lru_cache(maxsize=4096)
def element_summary(player_id: int) -> Dict[str, Any]:
    return _get(f"{BASE}/element-summary/{player_id}/")

def entry_history(entry_id: int) -> Dict[str, Any]:
    """Henter managers historik (chips, transfers, ranks per GW)."""
    return _get(f"{BASE}/entry/{entry_id}/history/")

def entry_transfers(entry_id: int) -> List[Dict[str, Any]]:
    """Henter managers transfers for sæsonen."""
    return _get(f"{BASE}/entry/{entry_id}/transfers/")
