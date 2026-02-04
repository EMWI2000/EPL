# services/odds.py
from __future__ import annotations
import re
import requests
from typing import Dict, Any, List, Optional, Tuple

ODDS_BASE = "https://api.the-odds-api.com/v4"

# ----------------------------- HTTP utils ----------------------------- #
def _get(url: str, params: Dict[str, Any]) -> Optional[Any]:
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# --------------------------- Odds helpers ----------------------------- #
def implied_prob(decimal_odds: float | None) -> float:
    try:
        return 1.0 / float(decimal_odds) if decimal_odds and float(decimal_odds) > 0 else 0.0
    except Exception:
        return 0.0

def epl_odds(
    api_key: str,
    regions: str = "uk",
    markets: Tuple[str, ...] = ("h2h", "totals"),
) -> Optional[List[Dict[str, Any]]]:
    """
    Henter odds for EPL. Vi beder om 1X2 ('h2h') + 'totals' (over/under) hvis muligt.
    Returnerer liste over events med bookmakers/markets.
    """
    if not api_key:
        return None
    url = f"{ODDS_BASE}/sports/soccer_epl/odds"
    params = {
        "regions": regions,
        "oddsFormat": "decimal",
        "markets": ",".join(markets),
        "apiKey": api_key,
    }
    return _get(url, params)

# --------------------- Navne-normalisering (kritisk) ------------------ #
def _clean_text(name: str) -> str:
    s = (name or "").lower()
    s = s.replace("&", "and").replace(".", " ").replace("-", " ").replace("'", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)        # fjern øvrig tegnsætning
    s = re.sub(r"\b(the|fc|afc)\b", " ", s)   # fjern fyldord
    # fix for "nott m forest" → "nottm forest"
    s = re.sub(r"\bnott\s+m\b", "nottm", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _norm_team_name(name: str) -> str:
    """
    Normaliserer holdnavne mellem FPL og TheOddsAPI.
    - Lowercase, fjern tegnsætning/"the/fc/afc", "&"→"and"
    - Ensret aliaser (fx "spurs" -> "tottenham hotspur")
    """
    s = _clean_text(name)
    alias = {
        # Manchester/Spurs/Wolves/Newcastle/West Ham
        "man utd": "manchester united",
        "man united": "manchester united",
        "man u": "manchester united",
        "manchester u": "manchester united",
        "man utd fc": "manchester united",
        "manchester utd": "manchester united",
        "manchester utd fc": "manchester united",
        "man city": "manchester city",
        "city": "manchester city",
        "tottenham": "tottenham hotspur",
        "spurs": "tottenham hotspur",
        "wolves": "wolverhampton wanderers",
        "wolverhampton": "wolverhampton wanderers",
        "west ham": "west ham united",
        "west ham utd": "west ham united",
        "west ham united fc": "west ham united",
        "newcastle": "newcastle united",
        "newcastle utd": "newcastle united",

        # Brighton/Forest/Palace/Bournemouth/Sheffield/Leeds/Leicester/Ipswich/Sunderland
        "brighton": "brighton and hove albion",
        "brighton and hove": "brighton and hove albion",
        "brighton hove albion": "brighton and hove albion",
        "brighton & hove albion": "brighton and hove albion",
        "nottm forest": "nottingham forest",
        "nott m forest": "nottingham forest",
        "nott forest": "nottingham forest",
        "notts forest": "nottingham forest",
        "n forest": "nottingham forest",
        "nottingham": "nottingham forest",
        "forest": "nottingham forest",
        "crystal": "crystal palace",
        "palace": "crystal palace",
        "afc bournemouth": "bournemouth",
        "bournemouth": "bournemouth",
        "sheff utd": "sheffield united",
        "sheffield utd": "sheffield united",
        "sheff united": "sheffield united",
        "sheff united fc": "sheffield united",
        "sheffield united fc": "sheffield united",
        "leeds": "leeds united",
        "leeds utd": "leeds united",
        "leicester": "leicester city",
        "ipswich": "ipswich town",
        "ipswich town fc": "ipswich town",
        "sunderland afc": "sunderland",
        "sunderland": "sunderland",

        # Kanoniske navne (bevar konsistens)
        "manchester city": "manchester city",
        "manchester united": "manchester united",
        "arsenal": "arsenal",
        "liverpool": "liverpool",
        "chelsea": "chelsea",
        "everton": "everton",
        "aston villa": "aston villa",
        "brentford": "brentford",
        "brighton and hove albion": "brighton and hove albion",
        "fulham": "fulham",
        "ipswich town": "ipswich town",
        "leicester city": "leicester city",
        "nottingham forest": "nottingham forest",
        "crystal palace": "crystal palace",
        "southampton": "southampton",
        "west ham united": "west ham united",
        "newcastle united": "newcastle united",
        "tottenham hotspur": "tottenham hotspur",
        "wolverhampton wanderers": "wolverhampton wanderers",

        # Typisk set i feeds (cup/sæsonskift)
        "sheffield united": "sheffield united",
        "burnley": "burnley",
        "luton": "luton town",
        "luton town": "luton town",
        "west brom": "west bromwich albion",
        "west bromwich albion": "west bromwich albion",

        # "fc"-varianter (selv om _clean_text normalt fjerner dem)
        "everton fc": "everton",
        "chelsea fc": "chelsea",
        "liverpool fc": "liverpool",
        "arsenal fc": "arsenal",
        "aston villa fc": "aston villa",
        "brentford fc": "brentford",
        "fulham fc": "fulham",
    }
    return alias.get(s, s)

# -------------------- Parsing af bookmakers/markets ------------------- #
def _find_markets(bookmakers: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    out = []
    for bm in bookmakers or []:
        for m in bm.get("markets", []) or []:
            if m.get("key") == key:
                out.append(m)
    return out

def _safe_mean(vals: List[float]) -> float:
    vals = [v for v in vals if isinstance(v, (int, float)) and v >= 0]
    return sum(vals) / len(vals) if vals else 0.0

def _odds_triplet_from_h2h_market(
    market: Dict[str, Any], home_norm: str, away_norm: str
) -> Tuple[float, float, float]:
    home_prob = draw_prob = away_prob = 0.0
    for o in market.get("outcomes") or []:
        nm_raw = o.get("name") or ""
        nm_clean = _clean_text(nm_raw)
        if nm_clean in ("home", "home team"):
            nm = home_norm
        elif nm_clean in ("away", "away team"):
            nm = away_norm
        elif nm_clean in ("draw", "tie"):
            nm = "draw"
        else:
            nm = _norm_team_name(nm_raw)
        price = o.get("price")
        if nm == home_norm:
            home_prob = implied_prob(price)
        elif nm == away_norm:
            away_prob = implied_prob(price)
        elif nm == "draw":
            draw_prob = implied_prob(price)
    total = home_prob + draw_prob + away_prob
    if total > 0:
        home_prob /= total; draw_prob /= total; away_prob /= total
    return home_prob, draw_prob, away_prob

def _triplet_from_bookmakers_h2h(
    bookmakers: List[Dict[str, Any]], home_norm: str, away_norm: str
) -> Tuple[float, float, float]:
    triples: List[Tuple[float, float, float]] = []
    for m in _find_markets(bookmakers, "h2h"):
        ph, pd, pa = _odds_triplet_from_h2h_market(m, home_norm, away_norm)
        if ph > 0 or pd > 0 or pa > 0:
            triples.append((ph, pd, pa))
    if not triples:
        return 0.0, 0.0, 0.0
    pH = _safe_mean([t[0] for t in triples])
    pD = _safe_mean([t[1] for t in triples])
    pA = _safe_mean([t[2] for t in triples])
    tot = pH + pD + pA
    if tot > 0:
        pH, pD, pA = pH/tot, pD/tot, pA/tot
    return pH, pD, pA

def _over25_prob_from_totals_market(market: Dict[str, Any]) -> float:
    best = 0.0
    for o in market.get("outcomes") or []:
        nm = (o.get("name") or "").lower()
        pt = o.get("point")
        if "over" in nm and pt is not None:
            try:
                if abs(float(pt) - 2.5) <= 0.5:
                    best = max(best, implied_prob(o.get("price")))
            except Exception:
                continue
    return best

def _over25_from_bookmakers_totals(bookmakers: List[Dict[str, Any]]) -> float:
    vals = []
    for m in _find_markets(bookmakers, "totals"):
        p = _over25_prob_from_totals_market(m)
        if p > 0:
            vals.append(p)
    return _safe_mean(vals)

# ------------------- Byg samlet odds-kontekst (public) ---------------- #
def build_odds_context(
    odds_payload: Optional[List[Dict[str, Any]]],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Returnerer mapping {(home_norm, away_norm): {"pH","pD","pA","pOver25"}}.
    """
    ctx: Dict[Tuple[str, str], Dict[str, float]] = {}
    if not odds_payload:
        return ctx
    for ev in odds_payload:
        H = _norm_team_name(ev.get("home_team") or "")
        A = _norm_team_name(ev.get("away_team") or "")
        bms = ev.get("bookmakers") or []
        pH, pD, pA = _triplet_from_bookmakers_h2h(bms, H, A)
        pOver25 = _over25_from_bookmakers_totals(bms)
        ctx[(H, A)] = {"pH": pH, "pD": pD, "pA": pA, "pOver25": pOver25}
    return ctx

# --------- Robust odds-lookup (matcher også omvendt H/A) -------------- #
def find_odds_for_pair(
    odds_ctx_by_pair: Dict[Tuple[str, str], Dict[str, float]],
    home_norm: str,
    away_norm: str
) -> Tuple[Optional[Dict[str, float]], bool]:
    """
    Prøv (H,A). Hvis ikke fundet, prøv (A,H) og swap pH<->pA.
    Returnerer (odds_dict | None, flipped: bool).
    """
    d = odds_ctx_by_pair.get((home_norm, away_norm))
    if d:
        return d, False
    d_rev = odds_ctx_by_pair.get((away_norm, home_norm))
    if d_rev:
        swapped = {
            "pH": d_rev.get("pA", 0.0),
            "pD": d_rev.get("pD", 0.0),
            "pA": d_rev.get("pH", 0.0),
            "pOver25": d_rev.get("pOver25", 0.0),
        }
        return swapped, True
    return None, False

# -------------------- Afledte faktorer (som før) --------------------- #
def attack_def_factors(
    pH: float, pD: float, pA: float, pOver25: float, is_home: bool
) -> Tuple[float, float, float]:
    """
    Afleder simple faktorer fra odds:
    - attack_factor ~ >1 ved favorit og høj Over 2.5
    - cs_prob_est   ~ baseret på draw + Under
    - def_factor    ~ >1 når CS mere sandsynlig
    """
    win_prob = pH if is_home else pA
    attack = 1.0 + 0.6 * (win_prob - 1/3) + 0.5 * (pOver25 - 0.5)
    attack = min(max(attack, 0.7), 1.4)
    cs_prob = 0.25 + 0.35*(pD) + 0.40*max(0.0, 1.0 - pOver25)
    cs_prob = min(max(cs_prob, 0.05), 0.75)
    def_factor = 0.9 + 0.4*(cs_prob - 0.3)
    def_factor = min(max(def_factor, 0.7), 1.3)
    return attack, cs_prob, def_factor
