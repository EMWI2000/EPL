"""Internal initial-squad function exposed as ``POST /api/compute``.

The HTTP layer has no Streamlit dependency.  Official FPL and optional Solio
data are fetched inside the serverless function, then passed to the existing
pure forecasting, projection-matching, and squad-optimisation modules.
"""

from __future__ import annotations

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
import hmac
import json
import logging
import os
from typing import Any, Mapping

import pandas as pd


from fpl_app.logic.features import (
    elements_df,
    expected_points_for_player,
    fixtures_df,
    gameweek_window,
)
from fpl_app.logic.forecast_v2 import forecast_players_v2
from fpl_app.logic.squad_plan import (
    DEFAULT_GW_WEIGHTS,
    SquadPlanError,
    optimize_squad_plan,
)
from fpl_app.logic.projections import overlay_solio_projections
from fpl_app.services.fpl_api import bootstrap_static, fixtures as fetch_fixtures
from fpl_app.services.solio import SolioClient, SolioError


LOGGER = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 16_384
INTERNAL_TOKEN_ENV = "INTERNAL_API_TOKEN"
INTERNAL_TOKEN_HEADER = "X-Internal-Token"
ALLOWED_REQUEST_FIELDS = frozenset(
    {"horizon", "include_doubtful", "use_solio", "forecast_version"}
)
DEFAULT_OPTIONS = {
    "horizon": 5,
    "include_doubtful": True,
    "use_solio": True,
    "forecast_version": "v2",
}
SOLIO_TIMEOUT_SECONDS = 6.0
FORECAST_VERSIONS = frozenset({"v2", "legacy"})
OPTIMIZER_CANDIDATE_LIMITS = {"GKP": 6, "DEF": 15, "MID": 15, "FWD": 9}
EXPERIMENTAL_NOTICE_DA = (
    "Prognosen er en eksperimentel beslutningsstøtte, ikke en facitliste. "
    "V2 dæmper små stikprøver, estimerer spilletid og viser usikkerhed, men "
    "skal fortsat dokumenteres i deadline-sikre walk-forward-backtests."
)


class RequestValidationError(ValueError):
    """Raised when a client option violates the public request contract."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = dict(details or {})


class RecommendationUnavailableError(RuntimeError):
    """Raised when valid upstream data cannot produce a recommendation."""


def is_internal_request_authorized(
    provided_token: str | None,
    *,
    configured_token: str | None = None,
) -> bool:
    """Validate the private BFF credential using constant-time comparison.

    Production callers omit ``configured_token`` so the expected value is read
    from ``INTERNAL_API_TOKEN``.  The optional argument only makes the primitive
    directly testable without mutating process-wide environment state.

    Missing or empty values always fail closed.  Comparing UTF-8 bytes avoids
    ``compare_digest`` rejecting non-ASCII strings before it can return a safe
    negative result.
    """

    expected = (
        os.environ.get(INTERNAL_TOKEN_ENV)
        if configured_token is None
        else configured_token
    )
    if not isinstance(expected, str) or not expected:
        return False
    if not isinstance(provided_token, str) or not provided_token:
        return False
    return hmac.compare_digest(provided_token.encode("utf-8"), expected.encode("utf-8"))


def validate_request_payload(payload: Any) -> dict[str, Any]:
    """Validate and default the public recommendation options."""

    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise RequestValidationError("Request body must be a JSON object.")

    unknown = sorted(set(payload) - ALLOWED_REQUEST_FIELDS)
    if unknown:
        raise RequestValidationError(
            "Request body contains unsupported fields.",
            details={"unsupported_fields": unknown},
        )

    options = dict(DEFAULT_OPTIONS)
    options.update(payload)

    horizon = options["horizon"]
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 5:
        raise RequestValidationError(
            "horizon must be an integer from 1 to 5.",
            details={"field": "horizon"},
        )

    for field in ("include_doubtful", "use_solio"):
        if not isinstance(options[field], bool):
            raise RequestValidationError(
                f"{field} must be a boolean.",
                details={"field": field},
            )

    forecast_version = options["forecast_version"]
    if not isinstance(forecast_version, str) or forecast_version not in FORECAST_VERSIONS:
        raise RequestValidationError(
            "forecast_version must be either 'v2' or 'legacy'.",
            details={"field": "forecast_version"},
        )

    return options


def _load_official_data() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch and normalize the two official FPL inputs used by the model."""

    bootstrap = bootstrap_static()
    players = elements_df(bootstrap)
    fixture_table = fixtures_df(fetch_fixtures(future_only=True))
    teams = pd.DataFrame(bootstrap["teams"])[["id", "name", "short_name"]].rename(
        columns={"id": "team_id"}
    )
    return bootstrap, players, fixture_table, teams


def next_open_gameweek(
    bootstrap: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> int | None:
    """Return the first gameweek whose official deadline has not passed.

    The fixtures endpoint can still include unplayed matches from the current
    gameweek after its transfer deadline.  Deadline data, not fixture status,
    therefore defines whether a squad can still be changed for that event.
    """

    reference_time = now or datetime.now(timezone.utc)
    if reference_time.tzinfo is None or reference_time.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    reference_time = reference_time.astimezone(timezone.utc)

    candidates: list[tuple[bool, datetime, int]] = []
    events = bootstrap.get("events", [])
    if not isinstance(events, list):
        return None

    for event in events:
        if not isinstance(event, Mapping):
            continue
        event_id = event.get("id")
        deadline_value = event.get("deadline_time")
        if isinstance(event_id, bool) or not isinstance(event_id, int) or event_id <= 0:
            continue
        if not isinstance(deadline_value, str) or not deadline_value.strip():
            continue
        candidate = deadline_value.strip()
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            deadline = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if deadline.tzinfo is None or deadline.utcoffset() is None:
            continue
        deadline = deadline.astimezone(timezone.utc)
        if deadline <= reference_time:
            continue
        candidates.append((bool(event.get("is_next")), deadline, event_id))

    if not candidates:
        return None
    return min(candidates, key=lambda item: (not item[0], item[1], item[2]))[2]


def _build_forecast_pool(
    players: pd.DataFrame,
    fixture_table: pd.DataFrame,
    teams: pd.DataFrame,
    horizon: int,
    start_event: int,
    forecast_version: str = "v2",
) -> pd.DataFrame:
    """Build the optimizer input without accessing UI or client state.

    V2 estimates minutes separately and continuously shrinks sparse player rates
    toward point-in-time position/price priors.  The legacy path remains
    available as an explicit comparison baseline; it is never selected through
    an implicit exception fallback.
    """

    if forecast_version == "legacy":
        return _build_legacy_forecast_pool(
            players,
            fixture_table,
            teams,
            horizon,
            start_event,
        )

    forecasts = forecast_players_v2(
        players,
        fixture_table,
        horizon=horizon,
        start_event=start_event,
    )
    by_player_offset = forecasts.set_index(["id", "gw_offset"], drop=False)

    rows: list[dict[str, Any]] = []
    for _, player in players.iterrows():
        player_id = int(player["id"])
        row: dict[str, Any] = {
            "id": player_id,
            "name": str(player["web_name"]),
            "team_id": int(player["team_id"]),
            "team": str(player["short_name"]),
            "pos": str(player["singular_name_short"]),
            "now_cost": int(player["now_cost"]),
            "status": str(player.get("status", "a")),
        }
        for offset in range(1, horizon + 1):
            gameweek = by_player_offset.loc[(player_id, offset)]
            row[f"ep_gw{offset}"] = float(gameweek["ep"])
            row[f"source_gw{offset}"] = "internal_v2"
            row[f"expected_minutes_gw{offset}"] = float(gameweek["expected_minutes"])
            row[f"appearance_prob_gw{offset}"] = float(
                gameweek["appearance_probability"]
            )
            # ``appearance_prob`` already includes official availability.  The
            # independent no-show input is reserved for a future calibrated
            # late-absence model and must not double-discount the same signal.
            row[f"no_show_prob_gw{offset}"] = 0.0
            row[f"sixty_prob_gw{offset}"] = float(gameweek["sixty_probability"])
            row[f"confidence_gw{offset}"] = float(gameweek["confidence"])
            row[f"reliability_gw{offset}"] = str(gameweek["reliability"])
            row[f"fixtures_count_gw{offset}"] = int(gameweek["fixtures_count"])
            row[f"is_blank_gw{offset}"] = bool(gameweek["is_blank"])
            row[f"is_dgw_gw{offset}"] = bool(gameweek["is_dgw"])
            row[f"components_gw{offset}"] = dict(gameweek["components"])
        rows.append(row)
    return pd.DataFrame(rows)


def _build_legacy_forecast_pool(
    players: pd.DataFrame,
    fixture_table: pd.DataFrame,
    teams: pd.DataFrame,
    horizon: int,
    start_event: int,
) -> pd.DataFrame:
    """Build the documented v1 heuristic as an explicit comparison baseline."""

    rows: list[dict[str, Any]] = []
    for _, player in players.iterrows():
        forecast = expected_points_for_player(
            player,
            fixture_table,
            n=horizon,
            teams_table=teams,
            use_ml=False,
            start_event=start_event,
        )
        row: dict[str, Any] = {
            "id": int(player["id"]),
            "name": str(player["web_name"]),
            "team_id": int(player["team_id"]),
            "team": str(player["short_name"]),
            "pos": str(player["singular_name_short"]),
            "now_cost": int(player["now_cost"]),
            "status": str(player.get("status", "a")),
        }
        for offset, gameweek in enumerate(forecast["per_gw"], start=1):
            ep = float(gameweek["ep"])
            fixture_count = int(gameweek.get("fixtures_count", 0))
            status = row["status"]
            appearance = 0.5 if status == "d" else 1.0
            if fixture_count == 0:
                appearance = 0.0
            row[f"ep_gw{offset}"] = ep
            row[f"source_gw{offset}"] = "internal_legacy"
            row[f"expected_minutes_gw{offset}"] = 90.0 * appearance * fixture_count
            row[f"appearance_prob_gw{offset}"] = appearance
            row[f"no_show_prob_gw{offset}"] = 0.0
            row[f"sixty_prob_gw{offset}"] = appearance
            row[f"confidence_gw{offset}"] = 0.0
            row[f"reliability_gw{offset}"] = "low"
            row[f"fixtures_count_gw{offset}"] = fixture_count
            row[f"is_blank_gw{offset}"] = fixture_count == 0
            row[f"is_dgw_gw{offset}"] = fixture_count >= 2
            row[f"components_gw{offset}"] = {"total": ep}
        rows.append(row)
    return pd.DataFrame(rows)


def _ensure_optimizer_contract(pool: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Fill audit/probability columns for legacy test fixtures and adapters."""

    normalized = pool.copy()
    for offset in range(1, horizon + 1):
        ep_column = f"ep_gw{offset}"
        if ep_column not in normalized.columns:
            raise RecommendationUnavailableError(f"Forecast is missing {ep_column}.")
        ep = pd.to_numeric(normalized[ep_column], errors="coerce")
        default_has_fixture = ep.fillna(0.0).ne(0.0)
        defaults: dict[str, Any] = {
            f"source_gw{offset}": "internal_legacy",
            f"expected_minutes_gw{offset}": default_has_fixture.astype(float) * 90.0,
            f"appearance_prob_gw{offset}": default_has_fixture.astype(float),
            f"no_show_prob_gw{offset}": 0.0,
            f"sixty_prob_gw{offset}": default_has_fixture.astype(float),
            f"confidence_gw{offset}": 0.0,
            f"reliability_gw{offset}": "low",
            f"fixtures_count_gw{offset}": default_has_fixture.astype(int),
            f"is_blank_gw{offset}": ~default_has_fixture,
            f"is_dgw_gw{offset}": False,
        }
        for column, default in defaults.items():
            if column not in normalized.columns:
                normalized[column] = default
        components_column = f"components_gw{offset}"
        if components_column not in normalized.columns:
            normalized[components_column] = [
                {"total": float(value)} for value in ep.fillna(0.0)
            ]
    return normalized


def _shortlist_optimizer_pool(
    eligible: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Build a deterministic, auditable serverless candidate set.

    The joint multi-gameweek problem grows with players and gameweeks.  The full
    public FPL pool is therefore screened before the MILP: every position keeps
    its squad quota of cheapest enablers, then alternates between horizon score,
    value and each individual GW.  This preserves budget routes and specialists
    without relying on input row order.  The solve is exact for its documented
    squad/XI/captain objective within the returned shortlist; reserve order is
    assigned deterministically afterwards.
    """

    weights = DEFAULT_GW_WEIGHTS[:horizon]
    ranked = eligible.copy()
    ranked["_weighted_ep"] = sum(
        pd.to_numeric(ranked[f"ep_gw{offset}"], errors="coerce").fillna(0.0)
        * weights[offset - 1]
        for offset in range(1, horizon + 1)
    )
    ranked["_value_ep"] = ranked["_weighted_ep"] / pd.to_numeric(
        ranked["now_cost"], errors="coerce"
    ).clip(lower=1.0)

    quota = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    chosen_ids: list[int] = []
    for position, limit in OPTIMIZER_CANDIDATE_LIMITS.items():
        candidates = ranked[ranked["pos"] == position].copy()
        if len(candidates) <= limit:
            chosen_ids.extend(int(value) for value in candidates["id"])
            continue

        selected: list[int] = []
        selected_set: set[int] = set()

        def add_from(frame: pd.DataFrame, count: int | None = None) -> None:
            added = 0
            for value in frame["id"]:
                player_id = int(value)
                if player_id in selected_set:
                    continue
                selected.append(player_id)
                selected_set.add(player_id)
                added += 1
                if len(selected) >= limit or (count is not None and added >= count):
                    break

        cheapest = candidates.sort_values(
            ["now_cost", "_weighted_ep", "id"],
            ascending=[True, False, True],
            kind="stable",
        )
        add_from(cheapest, count=quota[position])

        rankings = [
            candidates.sort_values(
                ["_weighted_ep", "now_cost", "id"],
                ascending=[False, True, True],
                kind="stable",
            ),
            candidates.sort_values(
                ["_value_ep", "_weighted_ep", "id"],
                ascending=[False, False, True],
                kind="stable",
            ),
            *[
                candidates.sort_values(
                    [f"ep_gw{offset}", "now_cost", "id"],
                    ascending=[False, True, True],
                    kind="stable",
                )
                for offset in range(1, horizon + 1)
            ],
        ]
        pointers = [0 for _ in rankings]
        while len(selected) < limit:
            made_progress = False
            for index, frame in enumerate(rankings):
                while pointers[index] < len(frame):
                    player_id = int(frame.iloc[pointers[index]]["id"])
                    pointers[index] += 1
                    if player_id in selected_set:
                        continue
                    selected.append(player_id)
                    selected_set.add(player_id)
                    made_progress = True
                    break
                if len(selected) >= limit:
                    break
            if not made_progress:
                break
        chosen_ids.extend(selected)

    shortlisted = ranked[ranked["id"].isin(chosen_ids)].copy()
    return shortlisted.drop(columns=["_weighted_ep", "_value_ep"]).sort_values(
        "id", kind="stable"
    )


def _empty_solio_metadata(requested: bool) -> dict[str, Any]:
    return {
        "requested": requested,
        "applied": False,
        "gameweek": None,
        "generated_at": None,
        "deadline_at": None,
        "matched": 0,
        "usable": 0,
        "unmatched": 0,
        "ambiguous": 0,
        "official_player_coverage": 0.0,
        "warning": None,
    }


def _apply_solio_overlay(
    pool: pd.DataFrame,
    official_players: pd.DataFrame,
    first_gameweek: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Overlay safe Solio GW projections and return audit metadata.

    Solio failure is intentionally non-fatal: the internal forecast remains a
    complete fallback and the response tells the frontend why no overlay was
    applied.
    """

    metadata = _empty_solio_metadata(requested=True)
    try:
        source = SolioClient(timeout=SOLIO_TIMEOUT_SECONDS).fetch_latest()
        metadata.update(
            {
                "gameweek": source.gameweek,
                "generated_at": source.generated_at.isoformat(),
                "deadline_at": source.deadline_at.isoformat(),
            }
        )
        if source.gameweek != first_gameweek:
            metadata["warning"] = (
                f"Solio projection is for GW{source.gameweek}; the analysis starts in "
                f"GW{first_gameweek}. Internal projections were used instead."
            )
            return pool, metadata

        overlay = overlay_solio_projections(official_players, source.payload)
        projected = overlay.players.set_index("id")[overlay.output_column]
        updated = pool.copy()
        solio_values = updated["id"].map(projected)
        matched = solio_values.notna()
        updated.loc[matched, "ep_gw1"] = solio_values.loc[matched].astype(float)
        updated.loc[matched, "source_gw1"] = "solio_points_internal_minutes"

        diagnostics = overlay.diagnostics
        metadata.update(
            {
                "applied": bool(matched.any()),
                "matched": int(diagnostics.matched_projection_count),
                "usable": int(diagnostics.usable_projection_count),
                "unmatched": len(diagnostics.unmatched),
                "ambiguous": len(diagnostics.ambiguous),
                "official_player_coverage": round(
                    float(diagnostics.official_player_coverage), 6
                ),
            }
        )
        if matched.any():
            metadata["warning"] = (
                "Solio supplies GW1 point projections. Expected minutes are not "
                "published for Solio rows; appearance probability and confidence "
                "remain the internal v2 estimates."
            )
        return updated, metadata
    except SolioError as exc:
        metadata["warning"] = f"Solio was unavailable or invalid; internal projections were used. ({exc})"
        return pool, metadata


def _round(value: Any, digits: int = 3) -> float:
    return round(float(value), digits)


def _serialize_recommendation(
    *,
    result: Any,
    eligible: pd.DataFrame,
    window: list[int],
    options: Mapping[str, Any],
    official_player_count: int,
    optimizer_candidate_count: int,
    solio_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    by_id = eligible.set_index("id", drop=False)
    weights = tuple(float(weight) for weight in result.gw_weights)
    first_plan = result.gameweeks[0]
    bench_order = {
        int(player_id): order
        for order, player_id in enumerate(first_plan.bench_ids, start=1)
    }

    def player_payload(player_id: int, role: str) -> dict[str, Any]:
        row = by_id.loc[int(player_id)]
        projections: list[dict[str, Any]] = []
        for offset, gameweek in enumerate(window, start=1):
            ep = _round(row[f"ep_gw{offset}"])
            source = str(row.get(f"source_gw{offset}", "internal_legacy"))
            raw_components = row.get(f"components_gw{offset}", {"total": ep})
            components = (
                {
                    str(name): _round(value)
                    for name, value in raw_components.items()
                }
                if isinstance(raw_components, Mapping)
                else {"total": ep}
            )
            if source.startswith("solio"):
                components = {
                    "internal_model": _round(components.get("total", ep)),
                    "solio_projection": ep,
                }
            reliability = str(row.get(f"reliability_gw{offset}", "low"))
            if reliability not in {"low", "medium", "high"}:
                reliability = "low"
            projections.append(
                {
                    "gameweek": int(gameweek),
                    "ep": ep,
                    "source": source,
                    "expected_minutes": (
                        None
                        if source.startswith("solio")
                        else _round(
                            row.get(f"expected_minutes_gw{offset}", 0.0), 1
                        )
                    ),
                    "appearance_probability": _round(
                        row.get(f"appearance_prob_gw{offset}", 0.0), 4
                    ),
                    "sixty_probability": _round(
                        row.get(f"sixty_prob_gw{offset}", 0.0), 4
                    ),
                    "confidence": _round(
                        row.get(f"confidence_gw{offset}", 0.0), 4
                    ),
                    "reliability": reliability,
                    "fixtures_count": int(
                        row.get(f"fixtures_count_gw{offset}", 0)
                    ),
                    "is_blank": bool(row.get(f"is_blank_gw{offset}", False)),
                    "is_dgw": bool(row.get(f"is_dgw_gw{offset}", False)),
                    "components": components,
                }
            )
        weighted_ep = sum(
            float(row[f"ep_gw{offset}"]) * weights[offset - 1]
            for offset in range(1, len(window) + 1)
        )
        price_tenths = int(row["now_cost"])
        return {
            "id": int(player_id),
            "name": str(row["name"]),
            "team_id": int(row["team_id"]),
            "team": str(row["team"]),
            "position": str(row["pos"]),
            "price": _round(price_tenths / 10.0, 1),
            "price_tenths": price_tenths,
            "status": str(row["status"]),
            "role": role,
            "is_captain": int(player_id) == int(first_plan.captain_id),
            "is_vice_captain": int(player_id) == int(first_plan.vice_captain_id),
            "bench_order": bench_order.get(int(player_id)),
            "weighted_ep": _round(weighted_ep),
            "projections": projections,
        }

    starters = [
        player_payload(player_id, "starter") for player_id in first_plan.starting_ids
    ]
    bench = [player_payload(player_id, "bench") for player_id in first_plan.bench_ids]
    by_player_id = {player["id"]: player for player in (*starters, *bench)}
    squad = [by_player_id[int(player_id)] for player_id in result.squad_ids]
    gameweeks = [
        {
            "gameweek": int(window[index]),
            "formation": str(plan.formation),
            "starting_ids": [int(player_id) for player_id in plan.starting_ids],
            "bench_ids": [int(player_id) for player_id in plan.bench_ids],
            "captain_id": int(plan.captain_id),
            "vice_captain_id": int(plan.vice_captain_id),
            "projected_xi_points": _round(plan.projected_xi_points),
            "projected_captain_bonus": _round(plan.projected_captain_bonus),
            "projected_bench_contribution": _round(
                plan.projected_bench_contribution
            ),
            "objective_points": _round(plan.objective_points),
        }
        for index, plan in enumerate(result.gameweeks)
    ]
    weighted_xi = sum(
        plan.projected_xi_points * weights[index]
        for index, plan in enumerate(result.gameweeks)
    )
    weighted_captain = sum(
        plan.projected_captain_bonus * weights[index]
        for index, plan in enumerate(result.gameweeks)
    )
    weighted_bench = sum(
        plan.projected_bench_contribution * weights[index]
        for index, plan in enumerate(result.gameweeks)
    )
    forecast_version = str(options["forecast_version"])
    validation = {
        "status": "unvalidated" if forecast_version == "v2" else "baseline",
        "message": (
            "V2's priorer og minutmodel er testet teknisk, men endnu ikke "
            "kalibreret på en komplet række deadline-snapshots."
            if forecast_version == "v2"
            else "Legacy-modellen er kun bevaret som sammenligningsbaseline."
        ),
    }

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "gameweek_window": [int(gameweek) for gameweek in window],
            "horizon": int(options["horizon"]),
            "forecast_version": forecast_version,
            "include_doubtful": bool(options["include_doubtful"]),
            "use_solio_requested": bool(options["use_solio"]),
            "validation": validation,
            "data_sources": {
                "fpl": {
                    "player_count": int(official_player_count),
                    "eligible_count": int(len(eligible)),
                    "optimizer_candidate_count": int(optimizer_candidate_count),
                    "shortlist_method": "position_price_value_and_per_gw_v1",
                },
                "solio": dict(solio_metadata),
            },
        },
        "summary": {
            "total_cost": _round(result.total_cost_tenths / 10.0, 1),
            "total_cost_tenths": int(result.total_cost_tenths),
            "bank": _round(result.bank_tenths / 10.0, 1),
            "bank_tenths": int(result.bank_tenths),
            "formation": str(first_plan.formation),
            "objective_points": _round(result.objective_points),
            "projected_xi_points": _round(weighted_xi),
            "projected_captain_bonus": _round(weighted_captain),
            "projected_bench_contribution": _round(weighted_bench),
        },
        "team": {
            "squad": squad,
            "starters": starters,
            "bench": bench,
            "captain_id": int(first_plan.captain_id),
            "vice_captain_id": int(first_plan.vice_captain_id),
            "gameweeks": gameweeks,
        },
        "experimental_notice": EXPERIMENTAL_NOTICE_DA,
    }


def generate_recommendation(payload: Any = None) -> dict[str, Any]:
    """Generate a JSON-serializable initial-squad recommendation."""

    options = validate_request_payload(payload)
    bootstrap, official_players, fixture_table, teams = _load_official_data()
    start_event = next_open_gameweek(bootstrap)
    if start_event is None:
        raise RecommendationUnavailableError("No open future gameweek deadline is available.")
    window = gameweek_window(
        fixture_table,
        n=options["horizon"],
        start_event=start_event,
    )
    if not window:
        raise RecommendationUnavailableError("No scheduled upcoming gameweeks are available.")

    pool = _build_forecast_pool(
        official_players,
        fixture_table,
        teams,
        options["horizon"],
        start_event,
        options["forecast_version"],
    )
    pool = _ensure_optimizer_contract(pool, options["horizon"])
    solio_metadata = _empty_solio_metadata(requested=options["use_solio"])
    if options["use_solio"]:
        pool, solio_metadata = _apply_solio_overlay(
            pool,
            official_players,
            first_gameweek=window[0],
        )

    statuses = {"a", "d"} if options["include_doubtful"] else {"a"}
    eligible = pool[pool["status"].isin(statuses)].copy()
    optimizer_pool = _shortlist_optimizer_pool(eligible, options["horizon"])
    try:
        result = optimize_squad_plan(
            optimizer_pool,
            horizon=options["horizon"],
            gw_weights=DEFAULT_GW_WEIGHTS[: options["horizon"]],
        )
    except SquadPlanError as exc:
        raise RecommendationUnavailableError(str(exc)) from exc

    return _serialize_recommendation(
        result=result,
        eligible=eligible,
        window=window,
        options=options,
        official_player_count=len(official_players),
        optimizer_candidate_count=len(optimizer_pool),
        solio_metadata=solio_metadata,
    )


def _error_payload(
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = dict(details)
    return {"error": error}


class handler(BaseHTTPRequestHandler):
    """Vercel Python function handler."""

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _read_payload(self) -> Any:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise RequestValidationError("Content-Length must be an integer.") from exc
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise RequestValidationError(
                f"Request body must not exceed {MAX_REQUEST_BYTES} bytes."
            )
        if length == 0:
            return {}
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise RequestValidationError("Content-Type must be application/json.")
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RequestValidationError("Request body must contain valid UTF-8 JSON.") from exc

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if not is_internal_request_authorized(self.headers.get(INTERNAL_TOKEN_HEADER)):
            if not os.environ.get(INTERNAL_TOKEN_ENV):
                LOGGER.error(
                    "Internal compute API rejected a request because its token is not configured"
                )
            self._send_json(
                401,
                _error_payload(
                    "unauthorized",
                    "A valid internal service credential is required.",
                ),
            )
            return
        try:
            payload = self._read_payload()
            response = generate_recommendation(payload)
        except RequestValidationError as exc:
            self._send_json(
                422,
                _error_payload("invalid_request", str(exc), details=exc.details),
            )
            return
        except RecommendationUnavailableError as exc:
            self._send_json(
                503,
                _error_payload("recommendation_unavailable", str(exc)),
            )
            return
        except Exception:
            LOGGER.exception("Recommendation request failed")
            self._send_json(
                502,
                _error_payload(
                    "upstream_error",
                    "Official FPL data could not be retrieved or processed.",
                ),
            )
            return
        self._send_json(200, response)

    def do_OPTIONS(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        self.send_response(204)
        self.send_header("Allow", "POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        self._send_json(
            405,
            _error_payload("method_not_allowed", "Use POST for this endpoint."),
        )
