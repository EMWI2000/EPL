"""Pure, point-in-time forecast primitives for the experimental v2 model.

The module deliberately has no network or application-layer dependencies.  It
accepts the normalized FPL player and fixture ``DataFrame`` objects already
available to :mod:`api.compute` and returns one auditable row per player and
gameweek.

The model is intentionally conservative:

* cumulative per-90 rates are continuously shrunk towards dynamic
  position/price priors (there is no minimum-minutes cliff);
* expected minutes are estimated separately from scoring rates and availability
  is applied exactly once there;
* appearance, attacking, clean-sheet, bonus, and goals-conceded points remain
  separate components; and
* malformed/non-finite optional statistics fall back to priors instead of
  leaking ``NaN`` or infinity into the optimizer.

It is still an experimental heuristic.  The prior strengths and calibration
must be evaluated on deadline-safe snapshots before claiming an improvement
over independent baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
import math
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import pandas as pd


FORECAST_VERSION = "forecast_v2"
DEFAULT_RATE_PRIOR_MINUTES = 900.0
# Playing role is observed much sooner than scoring ability.  One pseudo-match
# lets actual starts/minutes lead after two or three fixtures; scoring-rate
# shrinkage remains the separately documented 900-minute prior.
DEFAULT_MINUTES_PRIOR_MATCHES = 1.0

# These are deliberately the same documented fallback factors as v1.  In v2
# they affect only the relevant attack/defence components, never appearance.
FDR_FACTOR: Mapping[int, float] = MappingProxyType(
    {1: 1.30, 2: 1.15, 3: 1.00, 4: 0.88, 5: 0.75}
)
HOME_FACTOR = 1.08

POSITIONS = ("GKP", "DEF", "MID", "FWD")
PRICE_BANDS = ("low", "mid", "high")
POSITION_BY_ELEMENT_TYPE = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

GOAL_POINTS = {"GKP": 10.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}
ASSIST_POINTS = {position: 3.0 for position in POSITIONS}
CLEAN_SHEET_POINTS = {"GKP": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0}
DEFENSIVE_CONTRIBUTION_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}

# Used only when the supplied pool has no usable exposure (normally GW1).
# Once observations exist, priors are rebuilt from the entire point-in-time
# player pool, by position and within-position price band.
_FALLBACK_RATES: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        "GKP": MappingProxyType(
            {
                "xg_per90": 0.002,
                "xa_per90": 0.010,
                "bonus_per90": 0.12,
                "clean_sheets_per90": 0.28,
                "xgc_per90": 1.35,
                "defensive_actions_per90": 0.0,
                "saves_per90": 3.0,
            }
        ),
        "DEF": MappingProxyType(
            {
                "xg_per90": 0.050,
                "xa_per90": 0.075,
                "bonus_per90": 0.13,
                "clean_sheets_per90": 0.28,
                "xgc_per90": 1.35,
                "defensive_actions_per90": 9.0,
                "saves_per90": 0.0,
            }
        ),
        "MID": MappingProxyType(
            {
                "xg_per90": 0.210,
                "xa_per90": 0.190,
                "bonus_per90": 0.17,
                "clean_sheets_per90": 0.27,
                "xgc_per90": 1.35,
                "defensive_actions_per90": 7.0,
                "saves_per90": 0.0,
            }
        ),
        "FWD": MappingProxyType(
            {
                "xg_per90": 0.360,
                "xa_per90": 0.145,
                "bonus_per90": 0.20,
                "clean_sheets_per90": 0.0,
                "xgc_per90": 0.0,
                "defensive_actions_per90": 5.0,
                "saves_per90": 0.0,
            }
        ),
    }
)

_FALLBACK_ROLE_MINUTES: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        "GKP": MappingProxyType({"low": 24.0, "mid": 45.0, "high": 68.0}),
        "DEF": MappingProxyType({"low": 30.0, "mid": 48.0, "high": 66.0}),
        "MID": MappingProxyType({"low": 25.0, "mid": 46.0, "high": 67.0}),
        "FWD": MappingProxyType({"low": 25.0, "mid": 48.0, "high": 68.0}),
    }
)


class Reliability(str, Enum):
    """Coarse, user-facing label for the numeric forecast confidence."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ForecastConfidence:
    """Reliability metadata derived from player and prior exposure."""

    score: float
    reliability: Reliability
    sample_minutes: float
    sample_matches: float
    rate_prior_weight: float
    prior_support_minutes: float


@dataclass(frozen=True)
class RatePrior:
    """Dynamic position/price prior for rates and playing role."""

    position: str
    price_band: str
    xg_per90: float
    xa_per90: float
    bonus_per90: float
    clean_sheets_per90: float
    xgc_per90: float
    defensive_actions_per90: float
    saves_per90: float
    minutes_per_match: float
    start_probability: float
    unused_minutes_per_match: float
    unused_start_probability: float
    support_minutes: float
    support_player_matches: float


@dataclass(frozen=True)
class ForecastPriors:
    """All priors needed to forecast a player pool without external I/O."""

    by_position_price: Mapping[tuple[str, str], RatePrior]
    by_position: Mapping[str, RatePrior]
    price_cutoffs: Mapping[str, tuple[float, float]]
    team_matches_played: Mapping[int, float]
    rate_prior_minutes: float = DEFAULT_RATE_PRIOR_MINUTES
    minutes_prior_matches: float = DEFAULT_MINUTES_PRIOR_MATCHES
    start_event: int | None = None

    def price_band(self, position: str, price: Any) -> str:
        """Return the deterministic within-position price band for a player."""

        low_cutoff, high_cutoff = self.price_cutoffs.get(position, (0.0, 0.0))
        numeric_price = _finite_or_none(price)
        if numeric_price is None:
            return "mid"
        if numeric_price <= low_cutoff:
            return "low"
        if numeric_price >= high_cutoff:
            return "high"
        return "mid"

    def for_player(self, player: pd.Series | Mapping[str, Any]) -> RatePrior:
        position = _position(_get(player, "singular_name_short", "pos", "position", "element_type"))
        band = self.price_band(position, _get(player, "now_cost", "price", default=None))
        return self.by_position_price.get(
            (position, band),
            self.by_position[position],
        )


@dataclass(frozen=True)
class ExpectedMinutes:
    """Unconditional minutes and appearance probabilities for one fixture."""

    expected_minutes: float
    baseline_minutes: float
    appearance_probability: float
    sixty_probability: float
    availability_probability: float
    confidence: ForecastConfidence


@dataclass(frozen=True)
class ShrunkRates:
    """Posterior player rates after empirical-Bayes shrinkage."""

    xg_per90: float
    xa_per90: float
    bonus_per90: float
    clean_sheets_per90: float
    xgc_per90: float
    defensive_actions_per90: float
    saves_per90: float


@dataclass(frozen=True)
class ForecastComponents:
    """Expected FPL points, kept decomposed for auditability."""

    appearance: float = 0.0
    goals: float = 0.0
    assists: float = 0.0
    clean_sheet: float = 0.0
    bonus: float = 0.0
    defensive_contribution: float = 0.0
    saves: float = 0.0
    goals_conceded: float = 0.0

    @property
    def total(self) -> float:
        return float(
            self.appearance
            + self.goals
            + self.assists
            + self.clean_sheet
            + self.bonus
            + self.defensive_contribution
            + self.saves
            + self.goals_conceded
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "appearance": float(self.appearance),
            "goals": float(self.goals),
            "assists": float(self.assists),
            "clean_sheet": float(self.clean_sheet),
            "bonus": float(self.bonus),
            "defensive_contribution": float(self.defensive_contribution),
            "saves": float(self.saves),
            "goals_conceded": float(self.goals_conceded),
            "total": self.total,
        }

    def __add__(self, other: "ForecastComponents") -> "ForecastComponents":
        if not isinstance(other, ForecastComponents):
            return NotImplemented
        return ForecastComponents(
            appearance=self.appearance + other.appearance,
            goals=self.goals + other.goals,
            assists=self.assists + other.assists,
            clean_sheet=self.clean_sheet + other.clean_sheet,
            bonus=self.bonus + other.bonus,
            defensive_contribution=(
                self.defensive_contribution + other.defensive_contribution
            ),
            saves=self.saves + other.saves,
            goals_conceded=self.goals_conceded + other.goals_conceded,
        )


@dataclass(frozen=True)
class GameweekForecast:
    """One player's aggregate forecast for one FPL gameweek."""

    player_id: int
    event: int
    gw_offset: int
    expected_minutes: float
    appearance_probability: float
    sixty_probability: float
    confidence: ForecastConfidence
    components: ForecastComponents
    fixtures_count: int
    is_blank: bool
    is_dgw: bool

    @property
    def ep(self) -> float:
        return self.components.total

    def as_row(self) -> dict[str, Any]:
        component_values = self.components.as_dict()
        return {
            # ``id`` makes the result immediately mergeable with the current
            # optimizer pool; ``player_id`` keeps the long-form schema explicit.
            "id": int(self.player_id),
            "player_id": int(self.player_id),
            "event": int(self.event),
            "gw_offset": int(self.gw_offset),
            "ep": _finite(self.ep),
            "expected_minutes": _finite(self.expected_minutes),
            "appearance_probability": _probability(self.appearance_probability),
            "sixty_probability": _probability(self.sixty_probability),
            "confidence": _probability(self.confidence.score),
            "reliability": self.confidence.reliability.value,
            "fixtures_count": int(self.fixtures_count),
            "is_blank": bool(self.is_blank),
            "is_dgw": bool(self.is_dgw),
            "forecast_version": FORECAST_VERSION,
            "components": component_values,
            **{
                f"component_{name}": _finite(value)
                for name, value in component_values.items()
            },
        }


@dataclass(frozen=True)
class PlayerForecast:
    """Complete v2 forecast for one player across the requested horizon."""

    player_id: int
    position: str
    price_band: str
    prior: RatePrior
    expected_minutes: ExpectedMinutes
    rates: ShrunkRates
    per_gw: tuple[GameweekForecast, ...]
    forecast_version: str = FORECAST_VERSION

    @property
    def total_next_n(self) -> float:
        return float(sum(row.ep for row in self.per_gw))

    def as_rows(self) -> list[dict[str, Any]]:
        return [row.as_row() for row in self.per_gw]


OUTPUT_COLUMNS = (
    "id",
    "player_id",
    "event",
    "gw_offset",
    "ep",
    "expected_minutes",
    "appearance_probability",
    "sixty_probability",
    "confidence",
    "reliability",
    "fixtures_count",
    "is_blank",
    "is_dgw",
    "forecast_version",
    "components",
    "component_appearance",
    "component_goals",
    "component_assists",
    "component_clean_sheet",
    "component_bonus",
    "component_defensive_contribution",
    "component_saves",
    "component_goals_conceded",
    "component_total",
)


def _get(row: pd.Series | Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        try:
            value = row.get(key, None)
        except AttributeError:
            value = None
        if value is not None:
            return value
    return default


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _finite(value: Any, default: float = 0.0) -> float:
    number = _finite_or_none(value)
    return float(default if number is None else number)


def _non_negative(value: Any, default: float = 0.0) -> float:
    return max(0.0, _finite(value, default))


def _clamp(value: Any, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, _finite(value, minimum)))


def _probability(value: Any) -> float:
    return _clamp(value, 0.0, 1.0)


def _position(value: Any) -> str:
    numeric = _finite_or_none(value)
    if numeric is not None and int(numeric) in POSITION_BY_ELEMENT_TYPE:
        return POSITION_BY_ELEMENT_TYPE[int(numeric)]
    normalized = str(value or "MID").strip().upper()
    aliases = {
        "GK": "GKP",
        "GOALKEEPER": "GKP",
        "D": "DEF",
        "DEFENDER": "DEF",
        "M": "MID",
        "MIDFIELDER": "MID",
        "F": "FWD",
        "FORWARD": "FWD",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in POSITIONS else "MID"


def _team_id(player: pd.Series | Mapping[str, Any]) -> int:
    value = _finite_or_none(_get(player, "team_id", "team", default=0))
    return max(0, int(value or 0))


def _player_id(player: pd.Series | Mapping[str, Any], default: int = 0) -> int:
    value = _finite_or_none(_get(player, "id", "player_id", default=default))
    return int(value) if value is not None else int(default)


def _fixture_columns(fixtures: pd.DataFrame) -> tuple[str, str, str, str, str]:
    """Resolve normalized or raw official fixture column names."""

    if {"home_team", "away_team"}.issubset(fixtures.columns):
        return (
            "event",
            "home_team",
            "away_team",
            "home_fdr" if "home_fdr" in fixtures.columns else "",
            "away_fdr" if "away_fdr" in fixtures.columns else "",
        )
    if {"team_h", "team_a"}.issubset(fixtures.columns):
        return (
            "event",
            "team_h",
            "team_a",
            "team_h_difficulty" if "team_h_difficulty" in fixtures.columns else "",
            "team_a_difficulty" if "team_a_difficulty" in fixtures.columns else "",
        )
    return "event", "home_team", "away_team", "home_fdr", "away_fdr"


def _window(fixtures: pd.DataFrame, horizon: int, start_event: int | None) -> tuple[int, ...]:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 38:
        raise ValueError("horizon must be an integer from 1 to 38")
    first = int(start_event) if start_event is not None else None
    if first is None and "event" in fixtures.columns:
        events = pd.to_numeric(fixtures["event"], errors="coerce")
        events = events[(events > 0) & events.map(math.isfinite)]
        if not events.empty:
            first = int(events.min())
    if first is None:
        # Empty/malformed fixture data is still representable as a blank window.
        first = 1
    if first <= 0:
        raise ValueError("start_event must be a positive gameweek")
    return tuple(range(first, first + horizon))


def _completed_team_matches(fixtures: pd.DataFrame | None) -> dict[int, float]:
    if fixtures is None or fixtures.empty or "finished" not in fixtures.columns:
        return {}
    _, home_col, away_col, _, _ = _fixture_columns(fixtures)
    if home_col not in fixtures.columns or away_col not in fixtures.columns:
        return {}
    counts: dict[int, float] = {}
    for _, fixture in fixtures.iterrows():
        if not bool(fixture.get("finished", False)):
            continue
        for column in (home_col, away_col):
            team = _finite_or_none(fixture.get(column))
            if team is not None and team > 0:
                team_id = int(team)
                counts[team_id] = counts.get(team_id, 0.0) + 1.0
    return counts


def _inferred_team_matches(players: pd.DataFrame, start_event: int | None) -> dict[int, float]:
    """Infer the shared season exposure when only future fixtures are supplied.

    Vercel deliberately fetches future fixtures only.  At GW1 the bootstrap can
    still contain the previous season's aggregate player totals, so using each
    player's starts as the match denominator would make a two-start cameo look
    like a 90-minute regular.  The largest starts/minutes exposure in the pool
    is a conservative proxy for completed team matches and is shared across
    clubs, as Premier League teams play the same schedule apart from temporary
    blanks and doubles.
    """

    if players.empty:
        return {}
    starts = pd.to_numeric(players.get("starts"), errors="coerce").fillna(0.0)
    minutes = pd.to_numeric(players.get("minutes"), errors="coerce").fillna(0.0)
    starts = starts.where(starts.map(math.isfinite), 0.0).clip(lower=0.0)
    minutes = minutes.where(minutes.map(math.isfinite), 0.0).clip(lower=0.0)
    schedule_floor = max(0.0, float((start_event or 1) - 1))
    inferred = max(
        schedule_floor,
        float(starts.max()) if not starts.empty else 0.0,
        math.ceil(float(minutes.max()) / 90.0) if not minutes.empty else 0.0,
    )
    if inferred <= 0.0:
        return {}
    team_values = players.get("team_id", players.get("team"))
    if team_values is None:
        return {}
    team_ids = pd.to_numeric(team_values, errors="coerce").dropna().astype(int)
    return {team_id: float(inferred) for team_id in set(team_ids) if team_id > 0}


def _sample_matches(
    player: pd.Series | Mapping[str, Any],
    *,
    start_event: int | None,
    team_matches_played: Mapping[int, float],
) -> float:
    explicit = _finite_or_none(
        _get(
            player,
            "team_matches_played",
            "sample_matches",
            "matches_played",
            default=None,
        )
    )
    if explicit is not None:
        sample = max(0.0, explicit)
    else:
        sample = max(0.0, team_matches_played.get(_team_id(player), 0.0))
        if sample <= 0.0 and start_event is not None:
            # This is exact in an ordinary schedule and a conservative fallback
            # when the caller only has the future-fixtures endpoint.
            sample = float(max(0, int(start_event) - 1))

    starts = _non_negative(_get(player, "starts", default=0.0))
    # Bad upstream samples must not imply more starts than team fixtures.
    return max(sample, starts)


def _iso_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except (ValueError, OverflowError):
        return None


def _suspension_end(
    player: pd.Series | Mapping[str, Any], reference_date: date | None
) -> date | None:
    """Read only FPL's explicit suspension wording, never infer injury recovery.

    FPL normally omits the year.  Resolve it from the news timestamp (or the
    first forecast fixture), rejecting dates outside a 120-day window rather
    than extrapolating a season-long absence from ambiguous free text.
    """

    news = _get(player, "news", default="")
    if not isinstance(news, str):
        return None
    match = re.fullmatch(r"Suspended until (\d{1,2}) ([A-Za-z]{3})(?: (\d{4}))?", news.strip())
    if match is None:
        return None
    anchor = _iso_date(_get(player, "news_added", default=None)) or reference_date
    if anchor is None:
        return None
    month_names = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
    try:
        month = month_names.index(match[2].lower()) + 1
        years = [int(match[3])] if match[3] else [anchor.year - 1, anchor.year, anchor.year + 1]
        candidates = [date(year, month, int(match[1])) for year in years]
    except (ValueError, OverflowError):
        return None
    eligible = [value for value in candidates if -7 <= (value - anchor).days <= 120]
    if reference_date is not None:
        eligible = [value for value in eligible if -7 <= (value - reference_date).days <= 120]
    return min(eligible, key=lambda value: abs((value - anchor).days)) if eligible else None


def _availability(
    player: pd.Series | Mapping[str, Any],
    *,
    fixture_date: date | None = None,
    reference_date: date | None = None,
) -> float:
    status = str(_get(player, "status", default="a") or "a").strip().lower()
    if status == "s" and fixture_date is not None:
        suspension_end = _suspension_end(player, reference_date)
        if suspension_end is not None and fixture_date >= suspension_end:
            return 1.0
    # Hard-unavailable states fail closed even if an inconsistent chance field
    # says otherwise.
    if status in {"i", "s", "u", "n"}:
        return 0.0

    chance = _finite_or_none(
        _get(
            player,
            "chance_of_playing_next_round",
            "chance_of_playing_this_round",
            default=None,
        )
    )
    if chance is not None:
        return _probability(chance / 100.0)
    if status == "d":
        return 0.5
    return 1.0


def _price_cutoffs(players: pd.DataFrame, position: str) -> tuple[float, float]:
    if players.empty:
        return (0.0, 0.0)
    positions = players.apply(
        lambda row: _position(
            _get(row, "singular_name_short", "pos", "position", "element_type")
        ),
        axis=1,
    )
    price_column = "now_cost" if "now_cost" in players.columns else "price"
    if price_column not in players.columns:
        return (0.0, 0.0)
    prices = pd.to_numeric(players[price_column], errors="coerce")
    finite_prices = prices.map(
        lambda value: math.isfinite(value) if pd.notna(value) else False
    )
    values = prices[(positions == position) & finite_prices]
    if values.empty:
        return (0.0, 0.0)
    low = float(values.quantile(1.0 / 3.0))
    high = float(values.quantile(2.0 / 3.0))
    return (low, high)


def _metric_total(player: pd.Series | Mapping[str, Any], *keys: str) -> float | None:
    value = _finite_or_none(_get(player, *keys, default=None))
    return None if value is None else max(0.0, value)


def _aggregate_rate(
    rows: Iterable[pd.Series],
    keys: tuple[str, ...],
    *,
    fallback: float,
    prior_support_minutes: float = DEFAULT_RATE_PRIOR_MINUTES,
    maximum: float,
) -> tuple[float, float]:
    total = 0.0
    exposure = 0.0
    for row in rows:
        minutes = _non_negative(_get(row, "minutes", default=0.0))
        metric = _metric_total(row, *keys)
        if minutes <= 0.0 or metric is None:
            continue
        total += metric
        exposure += minutes
    observed = (90.0 * total / exposure) if exposure > 0.0 else fallback
    # Even the group prior is shrunk to a conservative positional fallback;
    # this prevents a single early-season return from becoming everybody's
    # prior while still making the prior fully data-driven as exposure grows.
    rate = (
        observed * exposure + fallback * prior_support_minutes
    ) / (exposure + prior_support_minutes)
    return (_clamp(rate, 0.0, maximum), exposure)


def _aggregate_role(
    rows: Iterable[pd.Series],
    *,
    fallback_minutes: float,
    start_event: int | None,
    team_matches_played: Mapping[int, float],
    include_unused: bool = False,
) -> tuple[float, float, float]:
    weighted_minutes = 0.0
    weighted_starts = 0.0
    support_matches = 0.0
    for row in rows:
        sample = _sample_matches(
            row,
            start_event=start_event,
            team_matches_played=team_matches_played,
        )
        if sample <= 0.0:
            continue
        minutes = min(_non_negative(_get(row, "minutes", default=0.0)), 90.0 * sample)
        starts = min(_non_negative(_get(row, "starts", default=0.0)), sample)
        # An unused squad member is evidence about their own role, not evidence
        # that a player starting every match should lose most of their minutes.
        # Keep the all-player prior separately for genuinely unused players.
        if not include_unused and minutes <= 0.0 and starts <= 0.0:
            continue
        weighted_minutes += minutes
        weighted_starts += starts
        support_matches += sample

    observed_minutes = weighted_minutes / support_matches if support_matches else fallback_minutes
    observed_start = weighted_starts / support_matches if support_matches else fallback_minutes / 90.0
    role_support = 12.0
    minutes_prior = (
        observed_minutes * support_matches + fallback_minutes * role_support
    ) / (support_matches + role_support)
    start_prior = (
        observed_start * support_matches
        + _probability(fallback_minutes / 90.0) * role_support
    ) / (support_matches + role_support)
    return (
        _clamp(minutes_prior, 0.0, 90.0),
        _probability(start_prior),
        support_matches,
    )


def _make_prior(
    rows: list[pd.Series],
    *,
    position: str,
    price_band: str,
    fallback_rates: Mapping[str, float],
    fallback_minutes: float,
    start_event: int | None,
    team_matches_played: Mapping[int, float],
) -> RatePrior:
    xg, xg_support = _aggregate_rate(
        rows,
        ("expected_goals", "xg"),
        fallback=fallback_rates["xg_per90"],
        maximum=3.0,
    )
    xa, xa_support = _aggregate_rate(
        rows,
        ("expected_assists", "xa"),
        fallback=fallback_rates["xa_per90"],
        maximum=3.0,
    )
    bonus, bonus_support = _aggregate_rate(
        rows,
        ("bonus",),
        fallback=fallback_rates["bonus_per90"],
        maximum=3.0,
    )
    clean_sheets, cs_support = _aggregate_rate(
        rows,
        ("clean_sheets",),
        fallback=fallback_rates["clean_sheets_per90"],
        maximum=1.0,
    )
    xgc, xgc_support = _aggregate_rate(
        rows,
        ("expected_goals_conceded", "xgc"),
        fallback=fallback_rates["xgc_per90"],
        maximum=6.0,
    )
    defensive_actions, defensive_support = _aggregate_rate(
        rows,
        ("defensive_contribution",),
        fallback=fallback_rates["defensive_actions_per90"],
        maximum=40.0,
    )
    saves, saves_support = _aggregate_rate(
        rows,
        ("saves",),
        fallback=fallback_rates["saves_per90"],
        maximum=15.0,
    )
    minutes_per_match, start_probability, support_matches = _aggregate_role(
        rows,
        fallback_minutes=fallback_minutes,
        start_event=start_event,
        team_matches_played=team_matches_played,
    )
    unused_minutes, unused_start, _ = _aggregate_role(
        rows,
        fallback_minutes=fallback_minutes,
        start_event=start_event,
        team_matches_played=team_matches_played,
        include_unused=True,
    )
    usable_supports = [
        xg_support,
        xa_support,
        bonus_support,
        cs_support,
        xgc_support,
        defensive_support,
        saves_support,
    ]
    return RatePrior(
        position=position,
        price_band=price_band,
        xg_per90=xg,
        xa_per90=xa,
        bonus_per90=bonus,
        clean_sheets_per90=clean_sheets,
        xgc_per90=xgc,
        defensive_actions_per90=defensive_actions,
        saves_per90=saves,
        minutes_per_match=minutes_per_match,
        start_probability=start_probability,
        unused_minutes_per_match=unused_minutes,
        unused_start_probability=unused_start,
        support_minutes=max(usable_supports, default=0.0),
        support_player_matches=support_matches,
    )


def build_forecast_priors(
    players: pd.DataFrame,
    *,
    fixtures: pd.DataFrame | None = None,
    start_event: int | None = None,
    rate_prior_minutes: float = DEFAULT_RATE_PRIOR_MINUTES,
) -> ForecastPriors:
    """Build dynamic, point-in-time priors from the complete supplied pool.

    Price bands are within-position tertiles.  Sparse band-level priors inherit
    strength from their dynamic position prior, while the first-gameweek edge
    case falls back to explicit constants.  No future result data or network I/O
    is accessed.
    """

    if not isinstance(players, pd.DataFrame):
        raise TypeError("players must be a pandas DataFrame")
    if fixtures is not None and not isinstance(fixtures, pd.DataFrame):
        raise TypeError("fixtures must be a pandas DataFrame or None")
    prior_minutes = _finite(rate_prior_minutes, DEFAULT_RATE_PRIOR_MINUTES)
    if prior_minutes <= 0.0:
        raise ValueError("rate_prior_minutes must be finite and positive")
    if start_event is not None and int(start_event) <= 0:
        raise ValueError("start_event must be a positive gameweek")

    completed_matches = _completed_team_matches(fixtures)
    if not completed_matches:
        completed_matches = _inferred_team_matches(players, start_event)
    cutoffs = {position: _price_cutoffs(players, position) for position in POSITIONS}

    prepared: list[tuple[pd.Series, str, str]] = []
    for _, row in players.iterrows():
        position = _position(
            _get(row, "singular_name_short", "pos", "position", "element_type")
        )
        low_cutoff, high_cutoff = cutoffs[position]
        price = _finite_or_none(_get(row, "now_cost", "price", default=None))
        if price is None:
            band = "mid"
        elif price <= low_cutoff:
            band = "low"
        elif price >= high_cutoff:
            band = "high"
        else:
            band = "mid"
        prepared.append((row, position, band))

    by_position: dict[str, RatePrior] = {}
    for position in POSITIONS:
        position_rows = [row for row, row_position, _ in prepared if row_position == position]
        # The position prior uses the middle role fallback only; price-specific
        # role priors are introduced in the next level.
        by_position[position] = _make_prior(
            position_rows,
            position=position,
            price_band="all",
            fallback_rates=_FALLBACK_RATES[position],
            fallback_minutes=_FALLBACK_ROLE_MINUTES[position]["mid"],
            start_event=start_event,
            team_matches_played=completed_matches,
        )

    by_band: dict[tuple[str, str], RatePrior] = {}
    for position in POSITIONS:
        position_prior = by_position[position]
        inherited_rates = {
            "xg_per90": position_prior.xg_per90,
            "xa_per90": position_prior.xa_per90,
            "bonus_per90": position_prior.bonus_per90,
            "clean_sheets_per90": position_prior.clean_sheets_per90,
            "xgc_per90": position_prior.xgc_per90,
            "defensive_actions_per90": position_prior.defensive_actions_per90,
            "saves_per90": position_prior.saves_per90,
        }
        for band in PRICE_BANDS:
            rows = [
                row
                for row, row_position, row_band in prepared
                if row_position == position and row_band == band
            ]
            by_band[(position, band)] = _make_prior(
                rows,
                position=position,
                price_band=band,
                fallback_rates=inherited_rates,
                fallback_minutes=_FALLBACK_ROLE_MINUTES[position][band],
                start_event=start_event,
                team_matches_played=completed_matches,
            )

    return ForecastPriors(
        by_position_price=MappingProxyType(by_band),
        by_position=MappingProxyType(by_position),
        price_cutoffs=MappingProxyType(cutoffs),
        team_matches_played=MappingProxyType(completed_matches),
        rate_prior_minutes=prior_minutes,
        minutes_prior_matches=DEFAULT_MINUTES_PRIOR_MATCHES,
        start_event=int(start_event) if start_event is not None else None,
    )


def _confidence(
    *,
    sample_minutes: float,
    sample_matches: float,
    prior: RatePrior,
    rate_prior_minutes: float,
) -> ForecastConfidence:
    rate_fraction = sample_minutes / (sample_minutes + rate_prior_minutes)
    match_fraction = sample_matches / (sample_matches + DEFAULT_MINUTES_PRIOR_MATCHES)
    player_evidence = math.sqrt(max(0.0, rate_fraction * match_fraction))
    prior_evidence = prior.support_minutes / (prior.support_minutes + rate_prior_minutes)
    score = _probability(0.05 + 0.70 * player_evidence + 0.20 * prior_evidence)
    if score < 0.40:
        reliability = Reliability.LOW
    elif score < 0.70:
        reliability = Reliability.MEDIUM
    else:
        reliability = Reliability.HIGH
    return ForecastConfidence(
        score=score,
        reliability=reliability,
        sample_minutes=sample_minutes,
        sample_matches=sample_matches,
        rate_prior_weight=rate_prior_minutes / (sample_minutes + rate_prior_minutes),
        prior_support_minutes=prior.support_minutes,
    )


def expected_minutes_for_player(
    player: pd.Series | Mapping[str, Any],
    priors: ForecastPriors,
    *,
    fixture_date: date | None = None,
    reference_date: date | None = None,
) -> ExpectedMinutes:
    """Estimate unconditional minutes and threshold probabilities per fixture."""

    prior = priors.for_player(player)
    minutes = _non_negative(_get(player, "minutes", default=0.0))
    starts = _non_negative(_get(player, "starts", default=0.0))
    sample_matches = _sample_matches(
        player,
        start_event=priors.start_event,
        team_matches_played=priors.team_matches_played,
    )
    # Blend towards the active-role prior as the player's own starts/exposure
    # support it.  A one-minute debut must not suddenly inherit a starter prior,
    # while a player starting every match must not inherit unused reserves.
    role_evidence = _probability(max(starts, minutes / 90.0) / sample_matches) if sample_matches > 0 else 0.0
    prior_minutes = prior.unused_minutes_per_match + role_evidence * (
        prior.minutes_per_match - prior.unused_minutes_per_match
    )
    prior_start = prior.unused_start_probability + role_evidence * (
        prior.start_probability - prior.unused_start_probability
    )
    if sample_matches > 0.0:
        capped_minutes = min(minutes, 90.0 * sample_matches)
        capped_starts = min(starts, sample_matches)
        observed_minutes = capped_minutes / sample_matches
        observed_start_minutes = 90.0 * capped_starts / sample_matches
        # Actual minutes carry most of the role information.  Starts add a
        # modest, continuous signal that distinguishes regular starters from
        # players accumulating the same minutes through cameos.
        observed_role = 0.80 * observed_minutes + 0.20 * observed_start_minutes
        baseline = (
            observed_role * sample_matches
            + prior_minutes * priors.minutes_prior_matches
        ) / (sample_matches + priors.minutes_prior_matches)
        start_probability = (
            capped_starts
            + prior_start * priors.minutes_prior_matches
        ) / (sample_matches + priors.minutes_prior_matches)
    else:
        baseline = prior_minutes
        start_probability = prior_start

    baseline = _clamp(baseline, 0.0, 90.0)
    start_probability = _probability(start_probability)
    # With aggregate bootstrap data, substitute appearances are not explicit.
    # A monotone minutes-to-appearance mapping supplies the missing information
    # while never allowing P(start) to exceed P(appearance).
    appearance = _probability(max(start_probability, baseline / 30.0))
    # Approximate the distribution as short appearances (20 minutes) and long
    # appearances (75 minutes), then solve E[M] for P(60+).  This is preferable
    # to treating E[M] >= 60 as a deterministic threshold.
    sixty = _probability((baseline - 20.0 * appearance) / 55.0)
    sixty = min(sixty, appearance)

    availability = _availability(player, fixture_date=fixture_date, reference_date=reference_date)
    projection = ExpectedMinutes(
        expected_minutes=_clamp(baseline * availability, 0.0, 90.0),
        baseline_minutes=baseline,
        appearance_probability=_probability(appearance * availability),
        sixty_probability=_probability(sixty * availability),
        availability_probability=availability,
        confidence=_confidence(
            sample_minutes=minutes,
            sample_matches=sample_matches,
            prior=prior,
            rate_prior_minutes=priors.rate_prior_minutes,
        ),
    )
    return projection


def _posterior_rate(
    total: float | None,
    sample_minutes: float,
    prior_rate: float,
    prior_minutes: float,
    maximum: float,
) -> float:
    if total is None or sample_minutes <= 0.0:
        return _clamp(prior_rate, 0.0, maximum)
    observed_rate = 90.0 * max(0.0, total) / sample_minutes
    posterior = (
        observed_rate * sample_minutes + prior_rate * prior_minutes
    ) / (sample_minutes + prior_minutes)
    return _clamp(posterior, 0.0, maximum)


def shrunk_rates_for_player(
    player: pd.Series | Mapping[str, Any],
    priors: ForecastPriors,
) -> ShrunkRates:
    """Return continuous empirical-Bayes per-90 rates for one player."""

    prior = priors.for_player(player)
    minutes = _non_negative(_get(player, "minutes", default=0.0))
    strength = priors.rate_prior_minutes
    return ShrunkRates(
        xg_per90=_posterior_rate(
            _metric_total(player, "expected_goals", "xg"),
            minutes,
            prior.xg_per90,
            strength,
            3.0,
        ),
        xa_per90=_posterior_rate(
            _metric_total(player, "expected_assists", "xa"),
            minutes,
            prior.xa_per90,
            strength,
            3.0,
        ),
        bonus_per90=_posterior_rate(
            _metric_total(player, "bonus"),
            minutes,
            prior.bonus_per90,
            strength,
            3.0,
        ),
        clean_sheets_per90=_posterior_rate(
            _metric_total(player, "clean_sheets"),
            minutes,
            prior.clean_sheets_per90,
            strength,
            1.0,
        ),
        xgc_per90=_posterior_rate(
            _metric_total(player, "expected_goals_conceded", "xgc"),
            minutes,
            prior.xgc_per90,
            strength,
            6.0,
        ),
        defensive_actions_per90=_posterior_rate(
            _metric_total(player, "defensive_contribution"),
            minutes,
            prior.defensive_actions_per90,
            strength,
            40.0,
        ),
        saves_per90=_posterior_rate(
            _metric_total(player, "saves"),
            minutes,
            prior.saves_per90,
            strength,
            15.0,
        ),
    )


def _fixtures_for_player(
    fixtures: pd.DataFrame,
    *,
    team_id: int,
    event: int,
    fixture_lookup: Mapping[tuple[int, int], tuple[tuple[int, bool, date | None], ...]] | None = None,
) -> list[tuple[int, bool, date | None]]:
    if fixture_lookup is not None:
        return list(fixture_lookup.get((int(team_id), int(event)), ()))
    if fixtures.empty or "event" not in fixtures.columns or team_id <= 0:
        return []
    _, home_col, away_col, home_fdr_col, away_fdr_col = _fixture_columns(fixtures)
    required = {home_col, away_col}
    if not required.issubset(fixtures.columns):
        return []

    result: list[tuple[int, bool, date | None]] = []
    for _, fixture in fixtures.iterrows():
        fixture_event = _finite_or_none(fixture.get("event"))
        home = _finite_or_none(fixture.get(home_col))
        away = _finite_or_none(fixture.get(away_col))
        if fixture_event is None or int(fixture_event) != event:
            continue
        if home is not None and int(home) == team_id:
            fdr = _finite_or_none(fixture.get(home_fdr_col))
            result.append((int(fdr) if fdr is not None else 3, True, _iso_date(fixture.get("kickoff_time"))))
        elif away is not None and int(away) == team_id:
            fdr = _finite_or_none(fixture.get(away_fdr_col))
            result.append((int(fdr) if fdr is not None else 3, False, _iso_date(fixture.get("kickoff_time"))))
    return result


def _build_fixture_lookup(
    fixtures: pd.DataFrame,
) -> Mapping[tuple[int, int], tuple[tuple[int, bool, date | None], ...]]:
    """Index fixtures once for pool forecasts instead of scanning per player/GW."""

    if fixtures.empty or "event" not in fixtures.columns:
        return MappingProxyType({})
    _, home_col, away_col, home_fdr_col, away_fdr_col = _fixture_columns(fixtures)
    if not {home_col, away_col}.issubset(fixtures.columns):
        return MappingProxyType({})

    mutable: dict[tuple[int, int], list[tuple[int, bool, date | None]]] = {}
    for _, fixture in fixtures.iterrows():
        raw_event = _finite_or_none(fixture.get("event"))
        raw_home = _finite_or_none(fixture.get(home_col))
        raw_away = _finite_or_none(fixture.get(away_col))
        if raw_event is None or int(raw_event) <= 0:
            continue
        event = int(raw_event)
        if raw_home is not None and int(raw_home) > 0:
            raw_fdr = _finite_or_none(fixture.get(home_fdr_col))
            mutable.setdefault((int(raw_home), event), []).append(
                (int(raw_fdr) if raw_fdr is not None else 3, True, _iso_date(fixture.get("kickoff_time")))
            )
        if raw_away is not None and int(raw_away) > 0:
            raw_fdr = _finite_or_none(fixture.get(away_fdr_col))
            mutable.setdefault((int(raw_away), event), []).append(
                (int(raw_fdr) if raw_fdr is not None else 3, False, _iso_date(fixture.get("kickoff_time")))
            )
    return MappingProxyType(
        {key: tuple(values) for key, values in mutable.items()}
    )


def _poisson_tail(mean: float, threshold: int) -> float:
    """Return P(X >= threshold) for a finite Poisson mean."""

    if threshold <= 0:
        return 1.0
    rate = _clamp(mean, 0.0, 60.0)
    probability = math.exp(-rate)
    cumulative = probability
    for value in range(1, threshold):
        probability *= rate / value
        cumulative += probability
    return _probability(1.0 - cumulative)


def _expected_goalkeeper_save_points(mean_saves: float) -> float:
    """Expected FPL save points under the one-point-per-three-saves rule."""

    rate = _clamp(mean_saves, 0.0, 15.0)
    if rate <= 0.0:
        return 0.0
    upper = max(3, int(math.ceil(rate + 8.0 * math.sqrt(rate) + 9.0)))
    return sum(_poisson_tail(rate, threshold) for threshold in range(3, upper + 1, 3))


def _expected_conceded_goal_blocks(mean_goals: float) -> float:
    """E[floor(X / 2)] for Poisson X, the official one-point-per-two rule.

    floor(X/2) = (X - 1{X is odd})/2 and P(X odd)=(1-exp(-2*mean))/2.
    ``expm1`` avoids cancellation for small rates.  This is conditional on
    playing and does not require a 60-minute appearance.
    """

    rate = _clamp(mean_goals, 0.0, 12.0)
    return max(0.0, rate / 2.0 + math.expm1(-2.0 * rate) / 4.0)


def _fixture_components(
    *,
    position: str,
    minutes: ExpectedMinutes,
    rates: ShrunkRates,
    fdr: int,
    is_home: bool,
) -> ForecastComponents:
    favorable = FDR_FACTOR.get(int(fdr), 1.0) * (HOME_FACTOR if is_home else 1.0)
    favorable = _clamp(favorable, 0.50, 1.60)
    minute_share = minutes.expected_minutes / 90.0
    appearance = minutes.appearance_probability + minutes.sixty_probability
    goals = minute_share * rates.xg_per90 * GOAL_POINTS[position] * favorable
    assists = minute_share * rates.xa_per90 * ASSIST_POINTS[position] * favorable
    clean_sheet_probability = _probability(rates.clean_sheets_per90 * favorable)
    clean_sheet = (
        minutes.sixty_probability
        * CLEAN_SHEET_POINTS[position]
        * clean_sheet_probability
    )
    # Bonus correlates weakly with the same match environment but is deliberately
    # much less fixture-sensitive than attacking output.
    bonus = minute_share * rates.bonus_per90 * (0.90 + 0.10 * favorable)
    appearance_probability = minutes.appearance_probability
    conditional_minute_share = (
        _clamp(
            minutes.expected_minutes / (90.0 * appearance_probability),
            0.0,
            1.0,
        )
        if appearance_probability > 0.0
        else 0.0
    )
    defensive_contribution = 0.0
    threshold = DEFENSIVE_CONTRIBUTION_THRESHOLD.get(position)
    if threshold is not None:
        expected_actions = rates.defensive_actions_per90 * conditional_minute_share
        defensive_contribution = (
            appearance_probability * 2.0 * _poisson_tail(expected_actions, threshold)
        )
    saves = 0.0
    if position == "GKP":
        expected_saves = rates.saves_per90 * conditional_minute_share
        saves = appearance_probability * _expected_goalkeeper_save_points(expected_saves)
    goals_conceded = 0.0
    if position in {"GKP", "DEF"}:
        expected_conceded = rates.xgc_per90 * conditional_minute_share / favorable
        goals_conceded = -appearance_probability * _expected_conceded_goal_blocks(expected_conceded)

    values = (
        appearance,
        goals,
        assists,
        clean_sheet,
        bonus,
        defensive_contribution,
        saves,
        goals_conceded,
    )
    if not all(math.isfinite(value) for value in values):
        # This should be unreachable because all inputs are sanitized.  Keeping
        # the guard here makes the optimizer boundary fail safe if the module is
        # later extended with a less controlled component.
        return ForecastComponents()
    return ForecastComponents(
        appearance=appearance,
        goals=goals,
        assists=assists,
        clean_sheet=clean_sheet,
        bonus=bonus,
        defensive_contribution=defensive_contribution,
        saves=saves,
        goals_conceded=goals_conceded,
    )


def forecast_player_v2(
    player: pd.Series | Mapping[str, Any],
    fixtures: pd.DataFrame,
    priors: ForecastPriors,
    *,
    horizon: int = 5,
    start_event: int | None = None,
    _fixture_lookup: Mapping[
        tuple[int, int], tuple[tuple[int, bool, date | None], ...]
    ] | None = None,
) -> PlayerForecast:
    """Forecast one player across consecutive gameweeks."""

    if not isinstance(fixtures, pd.DataFrame):
        raise TypeError("fixtures must be a pandas DataFrame")
    effective_start = start_event if start_event is not None else priors.start_event
    window = _window(fixtures, horizon, effective_start)
    player_id = _player_id(player)
    team_id = _team_id(player)
    position = _position(
        _get(player, "singular_name_short", "pos", "position", "element_type")
    )
    prior = priors.for_player(player)
    minutes = expected_minutes_for_player(player, priors)
    rates = shrunk_rates_for_player(player, priors)
    fixture_lookup = (
        _fixture_lookup
        if _fixture_lookup is not None
        else _build_fixture_lookup(fixtures)
    )
    fixture_dates = [
        fixture_date
        for (fixture_team, fixture_event), values in fixture_lookup.items()
        if fixture_team == team_id and fixture_event in window
        for _, _, fixture_date in values
        if fixture_date is not None
    ]
    reference_date = min(fixture_dates) if fixture_dates else None
    has_suspension = str(_get(player, "status", default="a") or "a").strip().lower() == "s"

    per_gw: list[GameweekForecast] = []
    for offset, event in enumerate(window, start=1):
        event_fixtures = _fixtures_for_player(
            fixtures,
            team_id=team_id,
            event=event,
            fixture_lookup=fixture_lookup,
        )
        components = ForecastComponents()
        no_appearance_probability = 1.0
        no_sixty_probability = 1.0
        event_expected_minutes = 0.0
        for fdr, is_home, fixture_date in event_fixtures:
            fixture_minutes = expected_minutes_for_player(
                player, priors, fixture_date=fixture_date, reference_date=reference_date
            ) if has_suspension else minutes
            components = components + _fixture_components(
                position=position,
                minutes=fixture_minutes,
                rates=rates,
                fdr=fdr,
                is_home=is_home,
            )
            event_expected_minutes += fixture_minutes.expected_minutes
            no_appearance_probability *= 1.0 - fixture_minutes.appearance_probability
            no_sixty_probability *= 1.0 - fixture_minutes.sixty_probability

        fixture_count = len(event_fixtures)
        per_gw.append(
            GameweekForecast(
                player_id=player_id,
                event=event,
                gw_offset=offset,
                expected_minutes=event_expected_minutes,
                appearance_probability=(
                    1.0 - no_appearance_probability if fixture_count else 0.0
                ),
                sixty_probability=(1.0 - no_sixty_probability if fixture_count else 0.0),
                confidence=minutes.confidence,
                components=components,
                fixtures_count=fixture_count,
                is_blank=fixture_count == 0,
                is_dgw=fixture_count >= 2,
            )
        )

    return PlayerForecast(
        player_id=player_id,
        position=position,
        price_band=prior.price_band,
        prior=prior,
        expected_minutes=minutes,
        rates=rates,
        per_gw=tuple(per_gw),
    )


def forecast_players_v2(
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    horizon: int = 5,
    start_event: int | None = None,
    priors: ForecastPriors | None = None,
) -> pd.DataFrame:
    """Return one complete, optimizer-safe row per player and gameweek.

    This is the intended :mod:`api.compute` integration point.  Existing player
    metadata can be merged on ``id`` and each gameweek can be pivoted into the
    current ``ep_gwN`` columns.  The long form preserves decomposition and
    confidence for API/UI serialization.
    """

    if not isinstance(players, pd.DataFrame):
        raise TypeError("players must be a pandas DataFrame")
    if not isinstance(fixtures, pd.DataFrame):
        raise TypeError("fixtures must be a pandas DataFrame")
    built_priors = priors or build_forecast_priors(
        players,
        fixtures=fixtures,
        start_event=start_event,
    )
    fixture_lookup = _build_fixture_lookup(fixtures)
    rows: list[dict[str, Any]] = []
    for fallback_id, (_, player) in enumerate(players.iterrows(), start=1):
        if _finite_or_none(_get(player, "id", "player_id", default=None)) is None:
            player = player.copy()
            player["id"] = fallback_id
        rows.extend(
            forecast_player_v2(
                player,
                fixtures,
                built_priors,
                horizon=horizon,
                start_event=start_event,
                _fixture_lookup=fixture_lookup,
            ).as_rows()
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


__all__ = [
    "DEFAULT_RATE_PRIOR_MINUTES",
    "FORECAST_VERSION",
    "ExpectedMinutes",
    "ForecastComponents",
    "ForecastConfidence",
    "ForecastPriors",
    "GameweekForecast",
    "PlayerForecast",
    "RatePrior",
    "Reliability",
    "ShrunkRates",
    "build_forecast_priors",
    "expected_minutes_for_player",
    "forecast_player_v2",
    "forecast_players_v2",
    "shrunk_rates_for_player",
]
