"""Validated adapter for Solio Analytics' public FPL projection feed."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, TYPE_CHECKING
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:  # Supports both package imports and Streamlit's fpl_app working directory.
    from ..domain.sources import SOLIO_SOURCE_ID, Provenance, get_source
except ImportError:  # pragma: no cover - exercised by the deployed import layout
    from domain.sources import SOLIO_SOURCE_ID, Provenance, get_source

if TYPE_CHECKING:
    try:
        from .snapshots import JsonSnapshotStore, SnapshotRef
    except ImportError:  # pragma: no cover
        from services.snapshots import JsonSnapshotStore, SnapshotRef


SOLIO_SCHEMA_VERSION = "2026-08-20"

_SECTIONS: Tuple[str, ...] = (
    "topProjected",
    "topCaptains",
    "topDifferentials",
    "topGoals",
    "topAssists",
    "bestCleanSheets",
    "topBonus",
    "topDefCon",
    "bestAttackingFixtures",
    "topTransfersIn",
    "topTransfersOut",
)


class SolioError(RuntimeError):
    """Base error for the Solio adapter."""


class SolioHTTPError(SolioError):
    """Raised when the upstream endpoint cannot be retrieved safely."""


class SolioValidationError(SolioError, ValueError):
    """Raised when upstream JSON violates the expected contract."""


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    body: bytes
    headers: Mapping[str, str]


@dataclass(frozen=True)
class SolioFetchResult:
    payload: Mapping[str, Any]
    observed_at: datetime
    generated_at: datetime
    deadline_at: datetime
    gameweek: int
    source_url: str

    @property
    def provenance(self) -> Provenance:
        return Provenance(
            source_id=SOLIO_SOURCE_ID,
            source_url=self.source_url,
            observed_at=self.observed_at,
            effective_at=self.generated_at,
            schema_version=SOLIO_SCHEMA_VERSION,
        )


Transport = Callable[[str, float, Mapping[str, str]], HttpResponse]
Clock = Callable[[], datetime]


def _parse_iso_datetime(value: Any, path: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise SolioValidationError(f"{path} must be a non-empty ISO-8601 string")
    candidate = value.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise SolioValidationError(f"{path} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SolioValidationError(f"{path} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require_non_empty_string(value: Any, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SolioValidationError(f"{path} must be a non-empty string")


def _validate_player_row(row: Any, path: str) -> None:
    if not isinstance(row, Mapping):
        raise SolioValidationError(f"{path} must be an object")
    for field in ("name", "team", "position"):
        _require_non_empty_string(row.get(field), f"{path}.{field}")
    if row["position"] not in {"GKP", "DEF", "MID", "FWD"}:
        raise SolioValidationError(f"{path}.position contains an unknown FPL position")
    for field in ("price", "prPoints"):
        if not _is_number(row.get(field)):
            raise SolioValidationError(f"{path}.{field} must be a finite number")
    if row["price"] < 0:
        raise SolioValidationError(f"{path}.price cannot be negative")
    opponents = row.get("opponents")
    if not isinstance(opponents, list):
        raise SolioValidationError(f"{path}.opponents must be an array")
    for index, opponent in enumerate(opponents):
        opponent_path = f"{path}.opponents[{index}]"
        if not isinstance(opponent, Mapping):
            raise SolioValidationError(f"{opponent_path} must be an object")
        _require_non_empty_string(opponent.get("opponent"), f"{opponent_path}.opponent")
        if not isinstance(opponent.get("isHome"), bool):
            raise SolioValidationError(f"{opponent_path}.isHome must be boolean")


def validate_solio_payload(
    payload: Any,
    *,
    require_projection_rows: bool = True,
) -> Mapping[str, Any]:
    """Validate the stable public-feed contract while allowing additive fields.

    The adapter intentionally does not transform or discard upstream fields.  A
    validated mapping is returned so the exact semantic payload can be snapshotted.
    """

    if not isinstance(payload, Mapping):
        raise SolioValidationError("payload must be a JSON object")

    generated_at = _parse_iso_datetime(payload.get("generatedAt"), "generatedAt")
    deadline_at = _parse_iso_datetime(payload.get("deadlineIso"), "deadlineIso")
    del generated_at, deadline_at  # Validation side effects are the purpose here.

    gameweek = payload.get("gameweek")
    if not isinstance(gameweek, int) or isinstance(gameweek, bool) or not 1 <= gameweek <= 38:
        raise SolioValidationError("gameweek must be an integer between 1 and 38")
    _require_non_empty_string(payload.get("source"), "source")

    for section in _SECTIONS:
        value = payload.get(section)
        if not isinstance(value, list):
            raise SolioValidationError(f"{section} must be an array")
        for index, row in enumerate(value):
            if not isinstance(row, Mapping):
                raise SolioValidationError(f"{section}[{index}] must be an object")

    projections = payload["topProjected"]
    if require_projection_rows and not projections:
        raise SolioValidationError("topProjected cannot be empty")
    for index, row in enumerate(projections):
        _validate_player_row(row, f"topProjected[{index}]")

    return payload


def _default_transport(
    url: str,
    timeout: float,
    headers: Mapping[str, str],
) -> HttpResponse:
    request = Request(url, headers=dict(headers), method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return HttpResponse(
                status_code=int(response.status),
                body=response.read(),
                headers=dict(response.headers.items()),
            )
    except HTTPError as exc:
        return HttpResponse(
            status_code=int(exc.code),
            body=exc.read(),
            headers=dict(exc.headers.items()) if exc.headers else {},
        )
    except (URLError, TimeoutError, OSError) as exc:
        raise SolioHTTPError(f"Could not retrieve Solio feed: {exc}") from exc


class SolioClient:
    """Small injectable client with explicit timeouts and schema validation."""

    def __init__(
        self,
        *,
        endpoint_url: Optional[str] = None,
        timeout: float = 20.0,
        transport: Transport = _default_transport,
        clock: Clock = lambda: datetime.now(timezone.utc),
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.endpoint_url = endpoint_url or get_source(SOLIO_SOURCE_ID).data_url
        self.timeout = float(timeout)
        self.transport = transport
        self.clock = clock
        self.headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "EMWI2000-EPL/phase-1 (+https://github.com/EMWI2000/EPL)",
        }

    def fetch_latest(self) -> SolioFetchResult:
        response = self.transport(self.endpoint_url, self.timeout, self.headers)
        if not 200 <= response.status_code < 300:
            raise SolioHTTPError(
                f"Solio returned HTTP {response.status_code} for {self.endpoint_url}"
            )
        # Observation time is when the complete response became available, not
        # when the request started. This is the safe cutoff used by backtests.
        observed_at = self.clock()
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise SolioError("Client clock must return a timezone-aware datetime")
        observed_at = observed_at.astimezone(timezone.utc)
        try:
            decoded = response.body.decode("utf-8-sig")
            payload = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SolioValidationError("Solio response is not valid UTF-8 JSON") from exc

        validated = validate_solio_payload(payload)
        return SolioFetchResult(
            payload=validated,
            observed_at=observed_at,
            generated_at=_parse_iso_datetime(validated["generatedAt"], "generatedAt"),
            deadline_at=_parse_iso_datetime(validated["deadlineIso"], "deadlineIso"),
            gameweek=int(validated["gameweek"]),
            source_url=self.endpoint_url,
        )

    def fetch_and_store(self, store: "JsonSnapshotStore") -> "SnapshotRef":
        result = self.fetch_latest()
        return store.write(result.payload, result.provenance)
