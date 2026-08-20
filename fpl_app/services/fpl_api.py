# services/fpl_api.py
from __future__ import annotations
import requests
from typing import Dict, Any, List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://fantasy.premierleague.com/api"
DEFAULT_CONNECT_TIMEOUT_SECONDS = 3.05
DEFAULT_READ_TIMEOUT_SECONDS = 6
DEFAULT_RETRY_ATTEMPTS = 1
DEFAULT_TIMEOUT = (DEFAULT_CONNECT_TIMEOUT_SECONDS, DEFAULT_READ_TIMEOUT_SECONDS)


def _build_session() -> requests.Session:
    retry = Retry(
        total=DEFAULT_RETRY_ATTEMPTS,
        connect=DEFAULT_RETRY_ATTEMPTS,
        read=DEFAULT_RETRY_ATTEMPTS,
        status=DEFAULT_RETRY_ATTEMPTS,
        backoff_factor=0.25,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        # A large upstream Retry-After must not outlive Vercel's function budget.
        respect_retry_after_header=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "EPL-FPL-Decision-Support/0.2",
        }
    )
    session.mount("https://", adapter)
    return session


_SESSION = _build_session()

def _get(url: str) -> Any:
    response = _SESSION.get(url, timeout=DEFAULT_TIMEOUT)
    if response.status_code == 404:
        raise requests.HTTPError(f"404 Not Found for URL: {url}", response=response)
    response.raise_for_status()
    return response.json()

def bootstrap_static() -> Dict[str, Any]:
    return _get(f"{BASE}/bootstrap-static/")

def fixtures(future_only: bool = True) -> List[Dict[str, Any]]:
    url = f"{BASE}/fixtures/"
    if future_only:
        url += "?future=1"
    return _get(url)

def entry_picks(entry_id: int, event_id: int) -> Dict[str, Any]:
    return _get(f"{BASE}/entry/{entry_id}/event/{event_id}/picks/")

def manager_summary(entry_id: int) -> Dict[str, Any]:
    return _get(f"{BASE}/entry/{entry_id}/")

def element_summary(player_id: int) -> Dict[str, Any]:
    return _get(f"{BASE}/element-summary/{player_id}/")

def entry_history(entry_id: int) -> Dict[str, Any]:
    """Henter managers historik (chips, transfers, ranks per GW)."""
    return _get(f"{BASE}/entry/{entry_id}/history/")

def entry_transfers(entry_id: int) -> List[Dict[str, Any]]:
    """Henter managers transfers for sæsonen."""
    return _get(f"{BASE}/entry/{entry_id}/transfers/")
