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
from fpl_app.logic.initial_squad import (
    DEFAULT_GW_WEIGHTS,
    InitialSquadError,
    optimize_initial_squad,
)
from fpl_app.logic.projections import overlay_solio_projections
from fpl_app.services.fpl_api import bootstrap_static, fixtures as fetch_fixtures
from fpl_app.services.solio import SolioClient, SolioError


LOGGER = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 16_384
INTERNAL_TOKEN_ENV = "INTERNAL_API_TOKEN"
INTERNAL_TOKEN_HEADER = "X-Internal-Token"
ALLOWED_REQUEST_FIELDS = frozenset({"horizon", "include_doubtful", "use_solio"})
DEFAULT_OPTIONS = {
    "horizon": 5,
    "include_doubtful": True,
    "use_solio": True,
}
SOLIO_TIMEOUT_SECONDS = 6.0


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
) -> pd.DataFrame:
    """Build the optimizer input without accessing UI or client state."""

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
            row[f"ep_gw{offset}"] = float(gameweek["ep"])
            row[f"source_gw{offset}"] = "internal_heuristic"
        rows.append(row)
    return pd.DataFrame(rows)


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
        updated.loc[matched, "source_gw1"] = "solio"

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
    solio_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    by_id = eligible.set_index("id", drop=False)
    weights = tuple(float(weight) for weight in result.gw_weights)
    bench_order = {
        int(player_id): order for order, player_id in enumerate(result.bench_ids, start=1)
    }

    def player_payload(player_id: int, role: str) -> dict[str, Any]:
        row = by_id.loc[int(player_id)]
        projections = [
            {
                "gameweek": int(gameweek),
                "ep": _round(row[f"ep_gw{offset}"]),
                "source": str(row.get(f"source_gw{offset}", "internal_heuristic")),
            }
            for offset, gameweek in enumerate(window, start=1)
        ]
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
            "is_captain": int(player_id) == int(result.captain_id),
            "is_vice_captain": int(player_id) == int(result.vice_captain_id),
            "bench_order": bench_order.get(int(player_id)),
            "weighted_ep": _round(weighted_ep),
            "projections": projections,
        }

    starters = [player_payload(player_id, "starter") for player_id in result.starting_ids]
    bench = [player_payload(player_id, "bench") for player_id in result.bench_ids]
    by_player_id = {player["id"]: player for player in (*starters, *bench)}
    squad = [by_player_id[int(player_id)] for player_id in result.squad_ids]

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "gameweek_window": [int(gameweek) for gameweek in window],
            "horizon": int(options["horizon"]),
            "include_doubtful": bool(options["include_doubtful"]),
            "use_solio_requested": bool(options["use_solio"]),
            "data_sources": {
                "fpl": {
                    "player_count": int(official_player_count),
                    "eligible_count": int(len(eligible)),
                },
                "solio": dict(solio_metadata),
            },
        },
        "summary": {
            "total_cost": _round(result.total_cost_tenths / 10.0, 1),
            "total_cost_tenths": int(result.total_cost_tenths),
            "bank": _round(result.bank_tenths / 10.0, 1),
            "bank_tenths": int(result.bank_tenths),
            "formation": str(result.formation),
            "objective_points": _round(result.objective_points),
            "projected_xi_points": _round(result.projected_xi_points),
            "projected_captain_bonus": _round(result.projected_captain_bonus),
            "projected_bench_contribution": _round(result.projected_bench_contribution),
        },
        "team": {
            "squad": squad,
            "starters": starters,
            "bench": bench,
            "captain_id": int(result.captain_id),
            "vice_captain_id": int(result.vice_captain_id),
        },
        "experimental_notice": str(result.experimental_notice),
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
    )
    solio_metadata = _empty_solio_metadata(requested=options["use_solio"])
    if options["use_solio"]:
        pool, solio_metadata = _apply_solio_overlay(
            pool,
            official_players,
            first_gameweek=window[0],
        )

    statuses = {"a", "d"} if options["include_doubtful"] else {"a"}
    eligible = pool[pool["status"].isin(statuses)].copy()
    forecast_columns = [f"ep_gw{offset}" for offset in range(1, options["horizon"] + 1)]
    try:
        result = optimize_initial_squad(
            eligible,
            horizon=options["horizon"],
            forecast_columns=forecast_columns,
            gw_weights=DEFAULT_GW_WEIGHTS[: options["horizon"]],
        )
    except InitialSquadError as exc:
        raise RecommendationUnavailableError(str(exc)) from exc

    return _serialize_recommendation(
        result=result,
        eligible=eligible,
        window=window,
        options=options,
        official_player_count=len(official_players),
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
