"""Internal public-manager sync exposed as ``POST /api/manager_state``.

The endpoint is called only by the authenticated Next.js BFF.  It reads the
latest state that FPL has made public, enriches it with the current official
player catalogue, and returns a manual-state draft for the next deadline.  It
does not accept FPL credentials, persist snapshots, or claim that public
last-deadline data is the manager's current draft.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
import hmac
import json
import logging
import os
from typing import Any, Mapping

import requests

from fpl_app.domain.rules import selling_price_tenths
from fpl_app.services.fpl_api import bootstrap_static, manager_summary
from fpl_app.services.fpl_price_signals import (
    PRICE_FEED_SCHEMA_VERSION,
    OfficialPlayerPriceSignal,
    OfficialPricePayloadError,
    parse_bootstrap_price_feed,
)
from fpl_app.services.personal_fpl_state import (
    PUBLIC_SOURCE,
    PersonalFplStateError,
    PublicFplNotFound,
    PublicLastDeadlineState,
    PublicManagerSummary,
    PublicManagerStateClient,
    chip_statuses_for_event,
    parse_public_manager_summary,
    serialize_deadline_snapshot,
)


LOGGER = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 16_384
INTERNAL_TOKEN_ENV = "INTERNAL_API_TOKEN"
INTERNAL_TOKEN_HEADER = "X-Internal-Token"
MIN_INTERNAL_TOKEN_BYTES = 32
RESPONSE_SCHEMA_VERSION = "fpl-manager-state-response-v2"
ALLOWED_REQUEST_FIELDS = frozenset({"manager_id"})


class RequestValidationError(ValueError):
    """Raised when the private request body violates the endpoint contract."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = dict(details or {})


class ManagerNotFoundError(LookupError):
    """Raised when no public manager state exists for the supplied id."""


class ManagerStateUnavailableError(RuntimeError):
    """Raised when valid data has no upcoming deadline to plan for."""


class UpstreamPayloadError(RuntimeError):
    """Raised when an official response cannot satisfy the safe output contract."""


def is_internal_request_authorized(
    provided_token: str | None,
    *,
    configured_token: str | None = None,
) -> bool:
    """Validate the BFF credential in constant time and fail closed."""

    expected = (
        os.environ.get(INTERNAL_TOKEN_ENV)
        if configured_token is None
        else configured_token
    )
    if (
        not isinstance(expected, str)
        or len(expected.encode("utf-8")) < MIN_INTERNAL_TOKEN_BYTES
    ):
        return False
    if not isinstance(provided_token, str) or not provided_token:
        return False
    return hmac.compare_digest(provided_token.encode("utf-8"), expected.encode("utf-8"))


def validate_request_payload(payload: Any) -> int:
    """Return the one allowed request value: a positive integer manager id."""

    if not isinstance(payload, Mapping):
        raise RequestValidationError("Request body must be a JSON object.")

    unknown = sorted(set(payload) - ALLOWED_REQUEST_FIELDS)
    if unknown:
        raise RequestValidationError(
            "Request body contains unsupported fields.",
            details={"unsupported_fields": unknown},
        )
    if "manager_id" not in payload:
        raise RequestValidationError(
            "manager_id is required.",
            details={"field": "manager_id"},
        )

    manager_id = payload["manager_id"]
    if isinstance(manager_id, bool) or not isinstance(manager_id, int) or manager_id <= 0:
        raise RequestValidationError(
            "manager_id must be a positive integer.",
            details={"field": "manager_id"},
        )
    return manager_id


def _utc_now(value: datetime | None = None) -> datetime:
    observed = value or datetime.now(timezone.utc)
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return observed.astimezone(timezone.utc)


def _parse_utc_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise UpstreamPayloadError(f"{field} must be an ISO timestamp")
    text = value.strip()
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise UpstreamPayloadError(f"{field} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise UpstreamPayloadError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _target_upcoming_event(
    bootstrap: Mapping[str, Any],
    *,
    now: datetime,
) -> dict[str, Any]:
    raw_events = bootstrap.get("events")
    if not isinstance(raw_events, list):
        raise UpstreamPayloadError("bootstrap.events must be a JSON array")

    candidates: list[tuple[datetime, int, str]] = []
    invalid_deadline_seen = False
    for index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, Mapping):
            invalid_deadline_seen = True
            continue
        event_id = raw_event.get("id")
        if isinstance(event_id, bool) or not isinstance(event_id, int) or not 1 <= event_id <= 38:
            invalid_deadline_seen = True
            continue
        try:
            deadline = _parse_utc_timestamp(
                raw_event.get("deadline_time"),
                f"bootstrap.events[{index}].deadline_time",
            )
        except UpstreamPayloadError:
            invalid_deadline_seen = True
            continue
        if deadline <= now:
            continue
        raw_name = raw_event.get("name")
        name = (
            raw_name.strip()
            if isinstance(raw_name, str) and raw_name.strip()
            else f"Gameweek {event_id}"
        )
        candidates.append((deadline, event_id, name))

    if not candidates:
        if invalid_deadline_seen:
            raise UpstreamPayloadError("official event deadlines could not be parsed")
        raise ManagerStateUnavailableError("No open future FPL deadline is available.")

    deadline, event_id, name = min(candidates, key=lambda row: (row[0], row[1]))
    return {
        "event": event_id,
        "name": name,
        "deadline_time": _iso_utc(deadline),
    }


def _strict_integer(
    value: Any,
    field: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        upper = f" and <= {maximum}" if maximum is not None else ""
        raise UpstreamPayloadError(
            f"{field} must be an integer >= {minimum}{upper}"
        )
    return value


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UpstreamPayloadError(f"{field} must be a non-empty string")
    return value.strip()


def _sanitized_manager_summary(summary: PublicManagerSummary) -> dict[str, Any]:
    if summary.team_name is None:
        raise UpstreamPayloadError("manager summary.name must be a non-empty string")
    return summary.to_server_dict()


def _catalogue_indexes(
    bootstrap: Mapping[str, Any],
) -> tuple[dict[int, Mapping[str, Any]], dict[int, dict[str, Any]], dict[int, str]]:
    raw_players = bootstrap.get("elements")
    raw_teams = bootstrap.get("teams")
    raw_positions = bootstrap.get("element_types")
    if not isinstance(raw_players, list):
        raise UpstreamPayloadError("bootstrap.elements must be a JSON array")
    if not isinstance(raw_teams, list):
        raise UpstreamPayloadError("bootstrap.teams must be a JSON array")
    if not isinstance(raw_positions, list):
        raise UpstreamPayloadError("bootstrap.element_types must be a JSON array")

    players: dict[int, Mapping[str, Any]] = {}
    for index, row in enumerate(raw_players):
        if not isinstance(row, Mapping):
            raise UpstreamPayloadError(f"bootstrap.elements[{index}] must be an object")
        element_id = _strict_integer(row.get("id"), f"bootstrap.elements[{index}].id", minimum=1)
        if element_id in players:
            raise UpstreamPayloadError(f"bootstrap contains duplicate player id {element_id}")
        players[element_id] = row

    teams: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(raw_teams):
        if not isinstance(row, Mapping):
            raise UpstreamPayloadError(f"bootstrap.teams[{index}] must be an object")
        team_id = _strict_integer(row.get("id"), f"bootstrap.teams[{index}].id", minimum=1)
        if team_id in teams:
            raise UpstreamPayloadError(f"bootstrap contains duplicate team id {team_id}")
        teams[team_id] = {
            "id": team_id,
            "name": _required_text(row.get("name"), f"bootstrap.teams[{index}].name"),
            "short_name": _required_text(
                row.get("short_name"), f"bootstrap.teams[{index}].short_name"
            ),
        }

    positions: dict[int, str] = {}
    for index, row in enumerate(raw_positions):
        if not isinstance(row, Mapping):
            raise UpstreamPayloadError(f"bootstrap.element_types[{index}] must be an object")
        position_id = _strict_integer(
            row.get("id"), f"bootstrap.element_types[{index}].id", minimum=1
        )
        if position_id in positions:
            raise UpstreamPayloadError(
                f"bootstrap contains duplicate element type id {position_id}"
            )
        positions[position_id] = _required_text(
            row.get("singular_name_short"),
            f"bootstrap.element_types[{index}].singular_name_short",
        )
    return players, teams, positions


def _core_player_prices(
    state: PublicLastDeadlineState,
    players: Mapping[int, Mapping[str, Any]],
) -> dict[int, dict[str, int]]:
    """Read only documented core price fields for the manager's 15 players."""

    result: dict[int, dict[str, int]] = {}
    for element_id in state.squad_ids:
        player = players.get(element_id)
        if player is None:
            raise UpstreamPayloadError(
                f"official catalogue is missing player id {element_id}"
            )
        now_cost = _strict_integer(
            player.get("now_cost"),
            f"player {element_id}.now_cost",
            minimum=1,
            maximum=500,
        )
        cost_change_start = _strict_integer(
            player.get("cost_change_start"),
            f"player {element_id}.cost_change_start",
            minimum=-500,
            maximum=500,
        )
        season_start_price = now_cost - cost_change_start
        if not 1 <= season_start_price <= 500:
            raise UpstreamPayloadError(
                f"season-start price is invalid for player id {element_id}"
            )
        result[element_id] = {
            "now_cost_tenths": now_cost,
            "cost_change_start_tenths": cost_change_start,
            "season_start_price_tenths": season_start_price,
        }
    return result


def _replay_public_acquisitions(
    state: PublicLastDeadlineState,
) -> dict[int, Any]:
    """Replay observable transfers and retain active public acquisitions.

    A player with no movement is treated as an initial-squad player.  A
    currently owned player whose latest observable movement is out is an
    inconsistent public history and must fail closed rather than silently use
    an older acquisition price.
    """

    ordered = sorted(
        enumerate(state.public_transfers),
        key=lambda item: (
            item[1].event,
            item[1].confirmed_at or "",
            item[0],
        ),
    )
    acquisitions: dict[int, Any] = {}
    latest_movement: dict[int, tuple[str, Any]] = {}
    for _, transfer in ordered:
        if transfer.element_in == transfer.element_out:
            raise UpstreamPayloadError(
                "a public transfer cannot move the same player both in and out"
            )
        acquisitions.pop(transfer.element_out, None)
        latest_movement[transfer.element_out] = ("out", transfer)
        acquisitions[transfer.element_in] = transfer
        latest_movement[transfer.element_in] = ("in", transfer)

    for element_id in state.squad_ids:
        movement = latest_movement.get(element_id)
        if movement is not None and movement[0] == "out":
            raise UpstreamPayloadError(
                f"latest public transfer movement for owned player {element_id} is out"
            )
        if movement is not None and element_id not in acquisitions:
            raise UpstreamPayloadError(
                f"public acquisition replay is inconsistent for player {element_id}"
            )
    return acquisitions


def _estimated_player_prices(
    state: PublicLastDeadlineState,
    core_prices: Mapping[int, Mapping[str, int]],
) -> dict[int, dict[str, Any]]:
    acquisitions = _replay_public_acquisitions(state)
    estimates: dict[int, dict[str, Any]] = {}
    for element_id in state.squad_ids:
        core_price = core_prices.get(element_id)
        if core_price is None:
            raise UpstreamPayloadError(
                f"official core prices are missing player id {element_id}"
            )
        transfer = acquisitions.get(element_id)
        if transfer is None:
            purchase_price = core_price["season_start_price_tenths"]
            basis: dict[str, Any] = {"kind": "season_start_price"}
        else:
            purchase_price = transfer.element_in_cost_tenths
            basis = {
                "kind": "latest_public_transfer_in",
                "event": transfer.event,
                "confirmed_at": transfer.confirmed_at,
            }
        estimates[element_id] = {
            "element_id": element_id,
            "purchase_price_tenths": purchase_price,
            "selling_price_tenths": selling_price_tenths(
                purchase_price,
                core_price["now_cost_tenths"],
            ),
            "purchase_price_basis": basis,
            "confirmation_required": True,
        }
    return estimates


def _enriched_picks(
    state: PublicLastDeadlineState,
    *,
    players: Mapping[int, Mapping[str, Any]],
    teams: Mapping[int, Mapping[str, Any]],
    positions: Mapping[int, str],
    price_signals: Mapping[int, OfficialPlayerPriceSignal],
    core_prices: Mapping[int, Mapping[str, int]],
    price_estimates: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for pick in state.picks:
        player = players.get(pick.element_id)
        signal = price_signals.get(pick.element_id)
        core_price = core_prices.get(pick.element_id)
        estimate = price_estimates.get(pick.element_id)
        if player is None or core_price is None or estimate is None:
            raise UpstreamPayloadError(
                f"official catalogue is missing player id {pick.element_id}"
            )
        team_id = _strict_integer(
            player.get("team"), f"player {pick.element_id}.team", minimum=1
        )
        element_type = _strict_integer(
            player.get("element_type"),
            f"player {pick.element_id}.element_type",
            minimum=1,
        )
        if element_type != pick.element_type:
            raise UpstreamPayloadError(
                f"player {pick.element_id} has inconsistent position data"
            )
        team = teams.get(team_id)
        position = positions.get(element_type)
        if team is None or position is None:
            raise UpstreamPayloadError(
                f"official catalogue references unknown metadata for player {pick.element_id}"
            )
        result.append(
            {
                "element_id": pick.element_id,
                "name": _required_text(
                    player.get("web_name"), f"player {pick.element_id}.web_name"
                ),
                "club": team["short_name"],
                "club_name": team["name"],
                "club_id": team_id,
                "position": position,
                "lineup_position": pick.position,
                "multiplier": pick.multiplier,
                "is_captain": pick.is_captain,
                "is_vice_captain": pick.is_vice_captain,
                "current_price_tenths": core_price["now_cost_tenths"],
                "estimated_purchase_price_tenths": estimate[
                    "purchase_price_tenths"
                ],
                "estimated_selling_price_tenths": estimate[
                    "selling_price_tenths"
                ],
                "purchase_price_basis": dict(estimate["purchase_price_basis"]),
                "official_price_signal": (
                    signal.to_server_dict() if signal is not None else None
                ),
            }
        )
    return result


def _manual_state_template(
    state: PublicLastDeadlineState,
    *,
    target_event: int,
    price_estimates: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    player_prices = [
        {
            "element_id": element_id,
            "purchase_price_tenths": price_estimates[element_id][
                "purchase_price_tenths"
            ],
            "selling_price_tenths": price_estimates[element_id][
                "selling_price_tenths"
            ],
        }
        for element_id in state.squad_ids
    ]
    return {
        # ``state`` deliberately matches parse_manual_current_state.  The
        # surrounding confirmation metadata is UI-only and must not be sent to
        # the optimizer as part of the state object.
        "state": {
            "current_squad_ids": list(state.squad_ids),
            "bank_tenths": state.bank_tenths,
            "free_transfers": 1,
            "player_prices": player_prices,
            "chips": chip_statuses_for_event(state.chip_usage, target_event),
            "chip_usage": [row.to_server_dict() for row in state.chip_usage],
            "no_active_chip_confirmed": False,
            "effective_event": target_event,
        },
        "confirmation_required": True,
        "fields_requiring_confirmation": [
            "current_squad_ids",
            "bank_tenths",
            "free_transfers",
            "player_prices",
            "chips",
            "chip_usage",
            "no_active_chip_confirmed",
        ],
        "free_transfers_default_reason": "current_free_transfers_are_not_public",
    }


def _warning_payloads(state: PublicLastDeadlineState) -> list[dict[str, str]]:
    warnings = [
        {
            "code": "last_deadline_state_requires_confirmation",
            "message": (
                "The public squad is locked at the last deadline. Confirm the current "
                "squad, bank, free transfers, prices and chips before planning."
            ),
        }
    ]
    if isinstance(state.active_chip, str) and state.active_chip.casefold() == "freehit":
        warnings.append(
            {
                "code": "free_hit_squad_is_temporary",
                "message": (
                    "The latest public squad used Free Hit and is temporary. Confirm the "
                    "reverted permanent squad and its player prices before planning."
                ),
            }
        )
    return warnings


def _official_price_signal_overlay(
    bootstrap: Mapping[str, Any],
) -> tuple[dict[int, OfficialPlayerPriceSignal], dict[str, Any]]:
    """Parse experimental official signals without making core prices depend on them."""

    metadata: dict[str, Any] = {
        "schema_version": PRICE_FEED_SCHEMA_VERSION,
        "available": False,
        "player_count": 0,
        "price_change_deadlines": [],
        "warning": None,
    }
    try:
        feed = parse_bootstrap_price_feed(bootstrap)
    except OfficialPricePayloadError as exc:
        LOGGER.warning("Official FPL price signals were ignored: %s", exc)
        metadata["warning"] = (
            "Official price-change signals are unavailable; current and "
            "season-start prices still use documented bootstrap fields."
        )
        return {}, metadata

    signals = {player.element_id: player for player in feed.players}
    metadata.update(
        {
            "schema_version": feed.schema_version,
            "available": True,
            "player_count": len(signals),
            "price_change_deadlines": list(feed.price_change_deadlines),
        }
    )
    return signals, metadata


def generate_manager_state(
    payload: Any,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fetch and serialize a safe, non-persistent manager-state draft."""

    manager_id = validate_request_payload(payload)
    observed_at = _utc_now(now)

    try:
        raw_manager = manager_summary(manager_id)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            raise ManagerNotFoundError("No public FPL manager state was found.") from exc
        raise
    validated_summary = parse_public_manager_summary(
        raw_manager,
        expected_entry_id=manager_id,
    )

    # The public state reads history, picks and transfers while bootstrap is an
    # independent official request.  Two bounded workers overlap only those
    # independent network paths and are torn down before serialization.
    client = PublicManagerStateClient()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="manager-sync") as pool:
        state_future = pool.submit(
            client.fetch_last_deadline_state,
            manager_id,
            validated_summary=validated_summary,
        )
        bootstrap_future = pool.submit(bootstrap_static)
        try:
            public_state = state_future.result()
        except PublicFplNotFound as exc:
            raise ManagerNotFoundError("No public FPL manager state was found.") from exc
        bootstrap = bootstrap_future.result()

    if not isinstance(bootstrap, Mapping):
        raise UpstreamPayloadError("bootstrap response must be a JSON object")
    manager = _sanitized_manager_summary(validated_summary)
    target = _target_upcoming_event(bootstrap, now=observed_at)
    price_signals, price_signal_metadata = _official_price_signal_overlay(bootstrap)
    players, teams, positions = _catalogue_indexes(bootstrap)
    core_prices = _core_player_prices(public_state, players)
    price_estimates = _estimated_player_prices(public_state, core_prices)
    picks = _enriched_picks(
        public_state,
        players=players,
        teams=teams,
        positions=positions,
        price_signals=price_signals,
        core_prices=core_prices,
        price_estimates=price_estimates,
    )
    snapshot = serialize_deadline_snapshot(
        public_state,
        observed_at=observed_at,
        source=PUBLIC_SOURCE,
    )

    response = {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "generated_at": _iso_utc(observed_at),
        "manager": manager,
        "target": target,
        "last_deadline_state": {
            "schema_version": public_state.schema_version,
            "state_kind": "public_last_deadline",
            "event": public_state.event,
            "bank_tenths": public_state.bank_tenths,
            "squad_value_tenths": public_state.squad_value_tenths,
            "total_transfers_at_deadline": public_state.total_transfers_at_deadline,
            "event_transfers": public_state.event_transfers,
            "event_transfer_cost": public_state.event_transfer_cost,
            "active_chip": public_state.active_chip,
            "chip_usage": [row.to_server_dict() for row in public_state.chip_usage],
            "limitations": list(public_state.limitations),
            "picks": picks,
        },
        "price_signals": price_signal_metadata,
        "manual_state_template": _manual_state_template(
            public_state,
            target_event=target["event"],
            price_estimates=price_estimates,
        ),
        # The checksum is created in memory.  The full internal snapshot is not
        # returned because its audit DTO contains official source URLs.
        "snapshot": {
            "schema_version": snapshot["schema_version"],
            "observed_at": snapshot["observed_at"],
            "source": snapshot["source"],
            "checksum_sha256": snapshot["checksum_sha256"],
            "persisted": False,
        },
        "warnings": _warning_payloads(public_state),
    }
    # This is also a guard against accidental NaN/unsupported values before the
    # handler writes any bytes to the browser.
    json.dumps(response, ensure_ascii=False, allow_nan=False)
    return response


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
            raise RequestValidationError(
                "Request body must contain valid UTF-8 JSON."
            ) from exc

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if not is_internal_request_authorized(self.headers.get(INTERNAL_TOKEN_HEADER)):
            if not os.environ.get(INTERNAL_TOKEN_ENV):
                LOGGER.error(
                    "Internal manager-state API rejected a request because its "
                    "token is not configured"
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
            response = generate_manager_state(self._read_payload())
        except RequestValidationError as exc:
            self._send_json(
                422,
                _error_payload("invalid_request", str(exc), details=exc.details),
            )
            return
        except ManagerNotFoundError:
            self._send_json(
                404,
                _error_payload(
                    "manager_not_found",
                    "No public FPL manager state was found for that manager_id.",
                ),
            )
            return
        except ManagerStateUnavailableError:
            self._send_json(
                503,
                _error_payload(
                    "manager_state_unavailable",
                    "Public manager state is not available for an upcoming deadline.",
                ),
            )
            return
        except (
            PersonalFplStateError,
            OfficialPricePayloadError,
            UpstreamPayloadError,
            requests.RequestException,
        ):
            LOGGER.exception("Official FPL manager-state request failed")
            self._send_json(
                502,
                _error_payload(
                    "upstream_error",
                    "Official FPL data could not be retrieved or processed.",
                ),
            )
            return
        except Exception:
            LOGGER.exception("Manager-state request failed")
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
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        self._send_json(
            405,
            _error_payload("method_not_allowed", "Use POST for this endpoint."),
        )


__all__ = [
    "INTERNAL_TOKEN_ENV",
    "INTERNAL_TOKEN_HEADER",
    "MIN_INTERNAL_TOKEN_BYTES",
    "ManagerNotFoundError",
    "ManagerStateUnavailableError",
    "RequestValidationError",
    "UpstreamPayloadError",
    "generate_manager_state",
    "handler",
    "is_internal_request_authorized",
    "validate_request_payload",
]
