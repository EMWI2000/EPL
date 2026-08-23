"""Server-only contracts for a manager's weekly FPL planning state.

The public FPL endpoints expose only the most recently locked manager state.
They deliberately do not reveal the next deadline's squad, current transfers,
free-transfer balance, or player-level selling prices.  This module therefore
keeps two inputs separate:

* :class:`PublicLastDeadlineState` is reconstructed conservatively from public
  entry endpoints and never claims to be the manager's current draft.
* :class:`ManualCurrentState` is a strict, credential-free correction supplied
  by the manager for the upcoming deadline.

No password, cookie, bearer token, FPL session, or persistence provider belongs
in these DTOs.  Network access is read-only and injectable for tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import re
from typing import Any, Callable, Mapping, Sequence

import requests

from fpl_app.domain.rules import Chip, can_play_chip, chip_window


FPL_API_BASE = "https://fantasy.premierleague.com/api"
STATE_SCHEMA_VERSION = "fpl-personal-state-v2"
SNAPSHOT_SCHEMA_VERSION = "fpl-deadline-state-snapshot-v2"
PUBLIC_SOURCE = "fpl_public_last_deadline"
MANUAL_SOURCE = "manual_manager_input"

_MANUAL_REQUIRED_FIELDS = frozenset(
    {
        "current_squad_ids",
        "bank_tenths",
        "free_transfers",
        "player_prices",
        "chips",
        "chip_usage",
        "no_active_chip_confirmed",
        "effective_event",
    }
)
_MANUAL_OPTIONAL_FIELDS = frozenset()
_FORBIDDEN_CREDENTIAL_KEY = re.compile(
    r"(?:password|passwd|cookie|token|authorization|bearer|session|csrf|email)",
    re.IGNORECASE,
)
_CHIP_NAMES = frozenset(chip.value for chip in Chip)
_CHIP_STATUSES = frozenset({"available", "used", "unavailable"})


class PersonalFplStateError(ValueError):
    """Base error for invalid or unavailable personal FPL state."""


class PublicFplPayloadError(PersonalFplStateError):
    """Raised when an official public endpoint violates the expected contract."""


class PublicFplNotFound(PersonalFplStateError):
    """Raised when a public entry resource is not available yet."""


class ManualStateValidationError(PersonalFplStateError):
    """Raised when credential-free manual state is incomplete or unsafe."""


class DeadlineSnapshotError(PersonalFplStateError):
    """Raised when a deadline snapshot is malformed or fails its checksum."""


def _strict_int(
    value: Any,
    field: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PersonalFplStateError(f"{field} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        upper = f" and <= {maximum}" if maximum is not None else ""
        raise PersonalFplStateError(f"{field} must be >= {minimum}{upper}")
    return value


def _optional_str(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise PublicFplPayloadError(f"{field} must be null or a non-empty string")
    return value.strip()


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PersonalFplStateError(f"{field} must be a JSON object")
    return value


def _require_sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise PersonalFplStateError(f"{field} must be a JSON array")
    return value


def _aware_utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise DeadlineSnapshotError(f"{field} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DeadlineSnapshotError(f"state is not strict JSON: {exc}") from exc


@dataclass(frozen=True)
class PublicPick:
    element_id: int
    position: int
    multiplier: int
    is_captain: bool
    is_vice_captain: bool
    element_type: int

    def to_server_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "position": self.position,
            "multiplier": self.multiplier,
            "is_captain": self.is_captain,
            "is_vice_captain": self.is_vice_captain,
            "element_type": self.element_type,
        }


@dataclass(frozen=True)
class PublicTransfer:
    event: int
    element_in: int
    element_out: int
    element_in_cost_tenths: int
    element_out_cost_tenths: int
    confirmed_at: str | None

    def to_server_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "element_in": self.element_in,
            "element_out": self.element_out,
            "element_in_cost_tenths": self.element_in_cost_tenths,
            "element_out_cost_tenths": self.element_out_cost_tenths,
            "confirmed_at": self.confirmed_at,
        }


@dataclass(frozen=True)
class ChipUsage:
    """One official chip activation, identified only by chip code and event."""

    name: str
    event: int

    def to_server_dict(self) -> dict[str, Any]:
        return {"name": self.name, "event": self.event}


@dataclass(frozen=True)
class PublicManagerSummary:
    """Allowlisted manager-summary fields shared by server-side consumers."""

    entry_id: int
    started_event: int
    current_event: int
    last_deadline_total_transfers: int
    team_name: str | None
    overall_points: int | None
    overall_rank: int | None

    def to_server_dict(self) -> dict[str, Any]:
        return {
            "id": self.entry_id,
            "team_name": self.team_name,
            "overall_points": self.overall_points,
            "overall_rank": self.overall_rank,
        }


@dataclass(frozen=True)
class PublicLastDeadlineState:
    """The latest publicly locked state, not the manager's current draft."""

    entry_id: int
    event: int
    picks: tuple[PublicPick, ...]
    bank_tenths: int
    squad_value_tenths: int
    total_transfers_at_deadline: int
    event_transfers: int
    event_transfer_cost: int
    active_chip: str | None
    chip_usage: tuple[ChipUsage, ...]
    public_transfers: tuple[PublicTransfer, ...]
    source_urls: tuple[str, ...]
    limitations: tuple[str, ...] = (
        "state_is_locked_at_last_public_deadline",
        "current_free_transfers_not_public",
        "current_confirmed_transfers_not_public",
        "purchase_and_selling_prices_not_public",
        "next_deadline_chip_selection_not_public",
    )
    schema_version: str = STATE_SCHEMA_VERSION

    @property
    def squad_ids(self) -> tuple[int, ...]:
        return tuple(pick.element_id for pick in self.picks)

    def to_server_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state_kind": "public_last_deadline",
            "entry_id": self.entry_id,
            "event": self.event,
            "squad_ids": list(self.squad_ids),
            "picks": [pick.to_server_dict() for pick in self.picks],
            "bank_tenths": self.bank_tenths,
            "squad_value_tenths": self.squad_value_tenths,
            "total_transfers_at_deadline": self.total_transfers_at_deadline,
            "event_transfers": self.event_transfers,
            "event_transfer_cost": self.event_transfer_cost,
            "active_chip": self.active_chip,
            "chip_usage": [row.to_server_dict() for row in self.chip_usage],
            "public_transfers": [row.to_server_dict() for row in self.public_transfers],
            "source_urls": list(self.source_urls),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class PlayerPriceState:
    element_id: int
    purchase_price_tenths: int
    selling_price_tenths: int

    def to_server_dict(self) -> dict[str, int]:
        return {
            "element_id": self.element_id,
            "purchase_price_tenths": self.purchase_price_tenths,
            "selling_price_tenths": self.selling_price_tenths,
        }


@dataclass(frozen=True)
class ManualCurrentState:
    """Credential-free manager input for the upcoming deadline."""

    current_squad_ids: tuple[int, ...]
    bank_tenths: int
    free_transfers: int
    player_prices: tuple[PlayerPriceState, ...]
    no_active_chip_confirmed: bool
    chips: tuple[tuple[str, str], ...]
    chip_usage: tuple[ChipUsage, ...]
    effective_event: int
    schema_version: str = STATE_SCHEMA_VERSION

    def to_server_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state_kind": "manual_current",
            "current_squad_ids": list(self.current_squad_ids),
            "bank_tenths": self.bank_tenths,
            "free_transfers": self.free_transfers,
            "player_prices": [row.to_server_dict() for row in self.player_prices],
            "no_active_chip_confirmed": self.no_active_chip_confirmed,
            "chips": {name: status for name, status in self.chips},
            "chip_usage": [row.to_server_dict() for row in self.chip_usage],
            "effective_event": self.effective_event,
        }


def _scan_for_credentials(value: Any, path: str = "manual_state") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if _FORBIDDEN_CREDENTIAL_KEY.search(key_text):
                raise ManualStateValidationError(
                    f"{path}.{key_text} is forbidden; never submit FPL credentials"
                )
            _scan_for_credentials(child, f"{path}.{key_text}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _scan_for_credentials(child, f"{path}[{index}]")


def _validated_chip_usage(
    rows: Sequence[ChipUsage],
    *,
    field: str,
    error_type: type[PersonalFplStateError],
) -> tuple[ChipUsage, ...]:
    """Validate season windows and cross-row rules shared by public/manual state."""

    seen_chip_halves: set[tuple[str, int]] = set()
    seen_events: set[int] = set()
    free_hit_events: set[int] = set()
    for row in rows:
        try:
            chip = Chip(row.name)
        except ValueError as exc:
            raise error_type(
                f"{field} contains unknown chip name: {row.name!r}"
            ) from exc
        window = chip_window(chip, row.event)
        if window is None:
            raise error_type(
                f"{field} contains {row.name} in event {row.event}, outside its legal window"
            )
        chip_half = (row.name, window.half)
        if chip_half in seen_chip_halves:
            raise error_type(
                f"{field} contains duplicate {row.name} usage in half {window.half}"
            )
        if row.event in seen_events:
            raise error_type(
                f"{field} contains more than one chip in event {row.event}"
            )
        seen_chip_halves.add(chip_half)
        seen_events.add(row.event)
        if chip is Chip.FREE_HIT:
            free_hit_events.add(row.event)

    if {19, 20}.issubset(free_hit_events):
        raise error_type(f"{field} cannot use Free Hit in consecutive events 19 and 20")
    return tuple(sorted(rows, key=lambda row: (row.event, row.name)))


def _parse_manual_chip_usage(payload: Any) -> tuple[ChipUsage, ...]:
    if isinstance(payload, (str, bytes, bytearray)) or not isinstance(
        payload, Sequence
    ):
        raise ManualStateValidationError("chip_usage must be a JSON array")
    rows: list[ChipUsage] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise ManualStateValidationError(
                f"chip_usage[{index}] must be a JSON object"
            )
        if set(raw) != {"name", "event"}:
            raise ManualStateValidationError(
                f"chip_usage[{index}] must contain exactly: event, name"
            )
        name = raw["name"]
        if not isinstance(name, str) or name not in _CHIP_NAMES:
            raise ManualStateValidationError(
                f"chip_usage[{index}].name must be one of: "
                + ", ".join(sorted(_CHIP_NAMES))
            )
        try:
            event = _strict_int(
                raw["event"], f"chip_usage[{index}].event", minimum=1, maximum=38
            )
        except PersonalFplStateError as exc:
            raise ManualStateValidationError(str(exc)) from exc
        rows.append(ChipUsage(name=name, event=event))
    return _validated_chip_usage(
        rows,
        field="chip_usage",
        error_type=ManualStateValidationError,
    )


def _parse_public_chip_usage(payload: Any) -> tuple[ChipUsage, ...]:
    if isinstance(payload, (str, bytes, bytearray)) or not isinstance(
        payload, Sequence
    ):
        raise PublicFplPayloadError("entry history.chips must be a JSON array")
    rows: list[ChipUsage] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise PublicFplPayloadError(
                f"entry history.chips[{index}] must be a JSON object"
            )
        if set(raw) != {"name", "event", "time"}:
            raise PublicFplPayloadError(
                f"entry history.chips[{index}] must contain exactly: event, name, time"
            )
        name = raw["name"]
        if not isinstance(name, str) or name not in _CHIP_NAMES:
            raise PublicFplPayloadError(
                f"entry history.chips[{index}].name must be one of: "
                + ", ".join(sorted(_CHIP_NAMES))
            )
        try:
            event = _strict_int(
                raw["event"],
                f"entry history.chips[{index}].event",
                minimum=1,
                maximum=38,
            )
            played_at = _optional_str(
                raw["time"], f"entry history.chips[{index}].time"
            )
        except PersonalFplStateError as exc:
            raise PublicFplPayloadError(str(exc)) from exc
        if played_at is None:
            raise PublicFplPayloadError(
                f"entry history.chips[{index}].time must be a non-empty string"
            )
        candidate = played_at[:-1] + "+00:00" if played_at.endswith("Z") else played_at
        try:
            parsed_time = datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise PublicFplPayloadError(
                f"entry history.chips[{index}].time must be an ISO UTC timestamp"
            ) from exc
        if (
            parsed_time.tzinfo is None
            or parsed_time.utcoffset() is None
            or parsed_time.utcoffset().total_seconds() != 0
        ):
            raise PublicFplPayloadError(
                f"entry history.chips[{index}].time must be an ISO UTC timestamp"
            )
        rows.append(ChipUsage(name=name, event=event))
    return _validated_chip_usage(
        rows,
        field="entry history.chips",
        error_type=PublicFplPayloadError,
    )


def chip_statuses_for_event(
    chip_usage: Sequence[ChipUsage],
    effective_event: int,
) -> dict[str, str]:
    """Return all four statuses for the chip set covering ``effective_event``."""

    event = _strict_int(
        effective_event, "effective_event", minimum=1, maximum=38
    )
    current_half = 1 if event <= 19 else 2
    usage_by_event = {row.event: row.name for row in chip_usage}
    statuses: dict[str, str] = {}
    for name in sorted(_CHIP_NAMES):
        used_halves = {
            window.half
            for row in chip_usage
            if row.name == name
            for window in (chip_window(name, row.event),)
            if window is not None
        }
        if current_half in used_halves:
            statuses[name] = "used"
            continue
        statuses[name] = (
            "available"
            if can_play_chip(
                name,
                event,
                used_halves=used_halves,
                previous_gameweek_chip=usage_by_event.get(event - 1),
            )
            else "unavailable"
        )
    return statuses


def parse_manual_current_state(
    payload: Any,
    *,
    known_player_ids: set[int] | frozenset[int] | None = None,
) -> ManualCurrentState:
    """Validate an exact manual state without accepting authentication material."""

    try:
        data = _require_mapping(payload, "manual_state")
        _scan_for_credentials(data)
        fields = set(data)
        missing = sorted(_MANUAL_REQUIRED_FIELDS - fields)
        unknown = sorted(fields - _MANUAL_REQUIRED_FIELDS - _MANUAL_OPTIONAL_FIELDS)
        if missing:
            raise ManualStateValidationError(
                f"manual_state is missing required fields: {', '.join(missing)}"
            )
        if unknown:
            raise ManualStateValidationError(
                f"manual_state contains unsupported fields: {', '.join(unknown)}"
            )

        raw_squad = _require_sequence(data["current_squad_ids"], "current_squad_ids")
        squad_ids = tuple(
            _strict_int(value, f"current_squad_ids[{index}]", minimum=1)
            for index, value in enumerate(raw_squad)
        )
        if len(squad_ids) != 15 or len(set(squad_ids)) != 15:
            raise ManualStateValidationError(
                "current_squad_ids must contain exactly 15 unique player ids"
            )
        if known_player_ids is not None:
            unknown_ids = sorted(set(squad_ids) - set(known_player_ids))
            if unknown_ids:
                raise ManualStateValidationError(
                    "current_squad_ids contains unknown player ids: "
                    + ", ".join(str(value) for value in unknown_ids)
                )

        bank = _strict_int(data["bank_tenths"], "bank_tenths", maximum=1000)
        free_transfers = _strict_int(
            data["free_transfers"], "free_transfers", minimum=1, maximum=5
        )
        no_active_chip_confirmed = data["no_active_chip_confirmed"]
        if not isinstance(no_active_chip_confirmed, bool):
            raise ManualStateValidationError(
                "no_active_chip_confirmed must be a boolean"
            )

        raw_prices = _require_sequence(data["player_prices"], "player_prices")
        price_rows: list[PlayerPriceState] = []
        for index, raw in enumerate(raw_prices):
            row = _require_mapping(raw, f"player_prices[{index}]")
            expected = {
                "element_id",
                "purchase_price_tenths",
                "selling_price_tenths",
            }
            if set(row) != expected:
                raise ManualStateValidationError(
                    f"player_prices[{index}] must contain exactly: "
                    + ", ".join(sorted(expected))
                )
            price_rows.append(
                PlayerPriceState(
                    element_id=_strict_int(
                        row["element_id"], f"player_prices[{index}].element_id", minimum=1
                    ),
                    purchase_price_tenths=_strict_int(
                        row["purchase_price_tenths"],
                        f"player_prices[{index}].purchase_price_tenths",
                        minimum=1,
                        maximum=500,
                    ),
                    selling_price_tenths=_strict_int(
                        row["selling_price_tenths"],
                        f"player_prices[{index}].selling_price_tenths",
                        minimum=1,
                        maximum=500,
                    ),
                )
            )
        price_ids = [row.element_id for row in price_rows]
        if len(price_ids) != 15 or len(set(price_ids)) != 15 or set(price_ids) != set(squad_ids):
            raise ManualStateValidationError(
                "player_prices must contain exactly one row for every current squad player"
            )
        price_rows.sort(key=lambda row: row.element_id)

        event = _strict_int(
            data["effective_event"], "effective_event", minimum=1, maximum=38
        )
        chip_usage = _parse_manual_chip_usage(data["chip_usage"])
        future_usage = [row for row in chip_usage if row.event >= event]
        if future_usage:
            raise ManualStateValidationError(
                "chip_usage events must be earlier than effective_event"
            )

        raw_chips = data["chips"]
        chips_mapping = _require_mapping(raw_chips, "chips")
        missing_chips = sorted(_CHIP_NAMES - set(chips_mapping))
        unknown_chips = sorted(set(chips_mapping) - _CHIP_NAMES)
        if missing_chips or unknown_chips:
            details: list[str] = []
            if missing_chips:
                details.append("missing " + ", ".join(missing_chips))
            if unknown_chips:
                details.append("unknown " + ", ".join(unknown_chips))
            raise ManualStateValidationError(
                "chips must contain exactly all four chip names (" + "; ".join(details) + ")"
            )
        chips: list[tuple[str, str]] = []
        for name in sorted(chips_mapping):
            status = chips_mapping[name]
            if not isinstance(status, str) or status not in _CHIP_STATUSES:
                raise ManualStateValidationError(
                    f"chips.{name} must be one of: {', '.join(sorted(_CHIP_STATUSES))}"
                )
            chips.append((name, status))
        expected_chips = chip_statuses_for_event(chip_usage, event)
        if dict(chips) != expected_chips:
            raise ManualStateValidationError(
                "chips statuses are inconsistent with chip_usage and effective_event"
            )
        return ManualCurrentState(
            current_squad_ids=squad_ids,
            bank_tenths=bank,
            free_transfers=free_transfers,
            player_prices=tuple(price_rows),
            no_active_chip_confirmed=no_active_chip_confirmed,
            chips=tuple(chips),
            chip_usage=chip_usage,
            effective_event=event,
        )
    except ManualStateValidationError:
        raise
    except PersonalFplStateError as exc:
        raise ManualStateValidationError(str(exc)) from exc


JsonTransport = Callable[[str], Any]


def _requests_json(url: str) -> Any:
    try:
        response = requests.get(
            url,
            headers={"Accept": "application/json", "User-Agent": "EPL-FPL-State/1.0"},
            timeout=(3.05, 8),
        )
    except requests.RequestException as exc:
        raise PersonalFplStateError(f"public FPL request failed: {exc}") from exc
    if response.status_code == 404:
        raise PublicFplNotFound(f"public FPL resource is not available: {url}")
    try:
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise PersonalFplStateError(f"public FPL response is invalid: {url}") from exc


def _entry_url(entry_id: int, suffix: str = "") -> str:
    return f"{FPL_API_BASE}/entry/{entry_id}/{suffix}"


def parse_public_manager_summary(
    payload: Any,
    *,
    expected_entry_id: int,
) -> PublicManagerSummary:
    """Validate and allowlist a public manager-summary payload once."""

    expected = _strict_int(expected_entry_id, "expected_entry_id", minimum=1)
    summary = _require_mapping(payload, "entry summary")
    summary_id = _strict_int(summary.get("id"), "entry summary.id", minimum=1)
    if summary_id != expected:
        raise PublicFplPayloadError("entry summary id does not match requested entry")

    team_name = _optional_str(summary.get("name"), "entry summary.name")
    raw_points = summary.get("summary_overall_points")
    overall_points = (
        None
        if raw_points is None
        else _strict_int(raw_points, "entry summary.summary_overall_points")
    )
    raw_rank = summary.get("summary_overall_rank")
    overall_rank = (
        None
        if raw_rank is None
        else _strict_int(
            raw_rank,
            "entry summary.summary_overall_rank",
            minimum=1,
        )
    )
    return PublicManagerSummary(
        entry_id=summary_id,
        started_event=_strict_int(
            summary.get("started_event"),
            "entry summary.started_event",
            minimum=1,
            maximum=38,
        ),
        current_event=_strict_int(
            summary.get("current_event"),
            "entry summary.current_event",
            minimum=1,
            maximum=38,
        ),
        last_deadline_total_transfers=_strict_int(
            summary.get("last_deadline_total_transfers"),
            "entry summary.last_deadline_total_transfers",
        ),
        team_name=team_name,
        overall_points=overall_points,
        overall_rank=overall_rank,
    )


class PublicManagerStateClient:
    """Read-only client for public manager endpoints; it never accepts auth."""

    def __init__(self, transport: JsonTransport | None = None) -> None:
        self._transport = transport or _requests_json

    def _get(self, url: str) -> Any:
        return self._transport(url)

    def fetch_last_deadline_state(
        self,
        entry_id: int,
        *,
        validated_summary: PublicManagerSummary | None = None,
    ) -> PublicLastDeadlineState:
        entry = _strict_int(entry_id, "entry_id", minimum=1)
        summary_url = _entry_url(entry)
        history_url = _entry_url(entry, "history/")
        transfers_url = _entry_url(entry, "transfers/")
        if validated_summary is None:
            summary = parse_public_manager_summary(
                self._get(summary_url),
                expected_entry_id=entry,
            )
        else:
            if not isinstance(validated_summary, PublicManagerSummary):
                raise PublicFplPayloadError(
                    "validated_summary must be a PublicManagerSummary"
                )
            if validated_summary.entry_id != entry:
                raise PublicFplPayloadError(
                    "entry summary id does not match requested entry"
                )
            summary = validated_summary
        history = _require_mapping(self._get(history_url), "entry history")
        chip_usage = _parse_public_chip_usage(history.get("chips"))

        raw_current = _require_sequence(history.get("current"), "entry history.current")
        history_by_event: dict[int, Mapping[str, Any]] = {}
        for index, raw in enumerate(raw_current):
            row = _require_mapping(raw, f"entry history.current[{index}]")
            event = _strict_int(row.get("event"), f"history[{index}].event", minimum=1, maximum=38)
            if event in history_by_event:
                raise PublicFplPayloadError("entry history contains duplicate events")
            history_by_event[event] = row
        if not history_by_event:
            raise PublicFplNotFound("no public event history is available for this entry")

        public_event = max(history_by_event)
        if not summary.started_event <= public_event <= summary.current_event:
            raise PublicFplPayloadError(
                "latest entry history event is outside the manager summary range"
            )
        if any(
            row.event < summary.started_event or row.event > public_event
            for row in chip_usage
        ):
            raise PublicFplPayloadError(
                "entry history.chips contains usage outside the public manager event range"
            )
        picks_url = _entry_url(entry, f"event/{public_event}/picks/")
        try:
            picks_payload = _require_mapping(
                self._get(picks_url),
                f"entry picks GW{public_event}",
            )
        except PublicFplNotFound as exc:
            raise PublicFplNotFound(
                "latest public event history has no matching picks"
            ) from exc
        transfer_payload = self._get(transfers_url)

        raw_picks = _require_sequence(picks_payload.get("picks"), "entry picks.picks")
        picks: list[PublicPick] = []
        for index, raw in enumerate(raw_picks):
            row = _require_mapping(raw, f"entry picks.picks[{index}]")
            if not isinstance(row.get("is_captain"), bool) or not isinstance(
                row.get("is_vice_captain"), bool
            ):
                raise PublicFplPayloadError("pick captain flags must be booleans")
            picks.append(
                PublicPick(
                    element_id=_strict_int(row.get("element"), f"pick[{index}].element", minimum=1),
                    position=_strict_int(row.get("position"), f"pick[{index}].position", minimum=1, maximum=15),
                    multiplier=_strict_int(row.get("multiplier"), f"pick[{index}].multiplier", maximum=3),
                    is_captain=row["is_captain"],
                    is_vice_captain=row["is_vice_captain"],
                    element_type=_strict_int(row.get("element_type"), f"pick[{index}].element_type", minimum=1, maximum=4),
                )
            )
        if len(picks) != 15 or len({pick.element_id for pick in picks}) != 15:
            raise PublicFplPayloadError("public picks must contain 15 unique players")
        if sorted(pick.position for pick in picks) != list(range(1, 16)):
            raise PublicFplPayloadError("public pick positions must be exactly 1 through 15")
        if sum(pick.is_captain for pick in picks) != 1 or sum(
            pick.is_vice_captain for pick in picks
        ) != 1:
            raise PublicFplPayloadError("public picks require one captain and one vice-captain")

        embedded_history = picks_payload.get("entry_history")
        event_history = (
            _require_mapping(embedded_history, "entry picks.entry_history")
            if embedded_history is not None
            else history_by_event.get(public_event)
        )
        if event_history is None:
            raise PublicFplPayloadError("locked picks have no matching event history")
        if _strict_int(event_history.get("event"), "entry_history.event", minimum=1, maximum=38) != public_event:
            raise PublicFplPayloadError("picks and entry history refer to different events")

        raw_transfers = _require_sequence(transfer_payload, "entry transfers")
        transfers: list[PublicTransfer] = []
        for index, raw in enumerate(raw_transfers):
            row = _require_mapping(raw, f"entry transfers[{index}]")
            event = _strict_int(row.get("event"), f"transfer[{index}].event", minimum=1, maximum=38)
            if event > public_event:
                # Fail closed: an allegedly public future transfer must not leak
                # into a state explicitly labelled as last-deadline.
                continue
            transfers.append(
                PublicTransfer(
                    event=event,
                    element_in=_strict_int(row.get("element_in"), f"transfer[{index}].element_in", minimum=1),
                    element_out=_strict_int(row.get("element_out"), f"transfer[{index}].element_out", minimum=1),
                    element_in_cost_tenths=_strict_int(row.get("element_in_cost"), f"transfer[{index}].element_in_cost", minimum=1),
                    element_out_cost_tenths=_strict_int(row.get("element_out_cost"), f"transfer[{index}].element_out_cost", minimum=1),
                    confirmed_at=_optional_str(row.get("time"), f"transfer[{index}].time"),
                )
            )
        transfers.sort(key=lambda row: (row.event, row.confirmed_at or "", row.element_in))

        active_chip = _optional_str(picks_payload.get("active_chip"), "active_chip")
        if active_chip is not None and active_chip not in _CHIP_NAMES:
            raise PublicFplPayloadError(
                "active_chip must be null or a known official chip name"
            )
        chips_at_public_event = [
            row.name for row in chip_usage if row.event == public_event
        ]
        if chips_at_public_event != ([active_chip] if active_chip is not None else []):
            raise PublicFplPayloadError(
                "active_chip is inconsistent with entry history.chips"
            )
        return PublicLastDeadlineState(
            entry_id=entry,
            event=public_event,
            picks=tuple(sorted(picks, key=lambda row: row.position)),
            bank_tenths=_strict_int(
                event_history.get("bank"),
                "entry_history.bank",
                maximum=1000,
            ),
            squad_value_tenths=_strict_int(event_history.get("value"), "entry_history.value"),
            total_transfers_at_deadline=_strict_int(
                summary.last_deadline_total_transfers,
                "validated summary.last_deadline_total_transfers",
            ),
            event_transfers=_strict_int(event_history.get("event_transfers"), "entry_history.event_transfers"),
            event_transfer_cost=_strict_int(event_history.get("event_transfers_cost"), "entry_history.event_transfers_cost"),
            active_chip=active_chip,
            chip_usage=chip_usage,
            public_transfers=tuple(transfers),
            source_urls=(summary_url, history_url, picks_url, transfers_url),
        )


def serialize_deadline_snapshot(
    state: PublicLastDeadlineState | ManualCurrentState,
    *,
    observed_at: datetime,
    source: str,
) -> dict[str, Any]:
    """Return an integrity-tagged snapshot envelope without writing it anywhere."""

    if source not in {PUBLIC_SOURCE, MANUAL_SOURCE}:
        raise DeadlineSnapshotError("source is not an approved personal-state source")
    expected_source = PUBLIC_SOURCE if isinstance(state, PublicLastDeadlineState) else MANUAL_SOURCE
    if source != expected_source:
        raise DeadlineSnapshotError("source does not match the state kind")
    observed = _aware_utc(observed_at, "observed_at")
    unsigned = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "observed_at": _iso_utc(observed),
        "source": source,
        "state": state.to_server_dict(),
    }
    checksum = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    return {**unsigned, "checksum_sha256": checksum}


def verify_deadline_snapshot(snapshot: Any) -> dict[str, Any]:
    """Validate an in-memory snapshot and return a detached strict-JSON copy."""

    data = _require_mapping(snapshot, "snapshot")
    expected_fields = {
        "schema_version",
        "observed_at",
        "source",
        "state",
        "checksum_sha256",
    }
    if set(data) != expected_fields:
        raise DeadlineSnapshotError("snapshot fields do not match the v2 schema")
    if data.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise DeadlineSnapshotError("unsupported snapshot schema_version")
    if data.get("source") not in {PUBLIC_SOURCE, MANUAL_SOURCE}:
        raise DeadlineSnapshotError("snapshot source is not approved")
    observed_at = data.get("observed_at")
    if not isinstance(observed_at, str) or not observed_at.strip():
        raise DeadlineSnapshotError("snapshot observed_at must be an ISO timestamp")
    candidate = observed_at[:-1] + "+00:00" if observed_at.endswith("Z") else observed_at
    try:
        parsed_observed_at = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise DeadlineSnapshotError("snapshot observed_at must be an ISO timestamp") from exc
    if parsed_observed_at.tzinfo is None or parsed_observed_at.utcoffset() is None:
        raise DeadlineSnapshotError("snapshot observed_at must include a timezone")
    state = _require_mapping(data.get("state"), "snapshot.state")
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise DeadlineSnapshotError("snapshot state has an unsupported schema_version")
    expected_kind = (
        "public_last_deadline" if data["source"] == PUBLIC_SOURCE else "manual_current"
    )
    if state.get("state_kind") != expected_kind:
        raise DeadlineSnapshotError("snapshot source does not match state_kind")
    if expected_kind == "manual_current":
        try:
            _scan_for_credentials(state, "snapshot.state")
        except ManualStateValidationError as exc:
            raise DeadlineSnapshotError(str(exc)) from exc
    checksum = data.get("checksum_sha256")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise DeadlineSnapshotError("snapshot checksum_sha256 is invalid")
    unsigned = {key: data[key] for key in expected_fields if key != "checksum_sha256"}
    actual = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if not hmac.compare_digest(actual, checksum):
        raise DeadlineSnapshotError("snapshot checksum does not match its content")
    return json.loads(_canonical_json(data))


__all__ = [
    "ChipUsage",
    "DeadlineSnapshotError",
    "MANUAL_SOURCE",
    "ManualCurrentState",
    "ManualStateValidationError",
    "PersonalFplStateError",
    "PlayerPriceState",
    "PUBLIC_SOURCE",
    "PublicFplNotFound",
    "PublicFplPayloadError",
    "PublicLastDeadlineState",
    "PublicManagerSummary",
    "PublicManagerStateClient",
    "PublicPick",
    "PublicTransfer",
    "SNAPSHOT_SCHEMA_VERSION",
    "STATE_SCHEMA_VERSION",
    "chip_statuses_for_event",
    "parse_manual_current_state",
    "parse_public_manager_summary",
    "serialize_deadline_snapshot",
    "verify_deadline_snapshot",
]
