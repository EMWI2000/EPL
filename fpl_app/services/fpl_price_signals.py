"""Strict parser for the official 2026/27 FPL price-change fields.

The fields are served inside ``bootstrap-static`` but are not a stable public
API contract.  This adapter allowlists the values used by the application,
normalises numeric strings to finite numbers, and preserves the provider's raw
integer ``likelihood`` code without guessing at undocumented semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Mapping, Sequence


PRICE_FEED_SCHEMA_VERSION = "fpl-official-price-signals-v1"


class OfficialPricePayloadError(ValueError):
    """Raised when official price data is unsafe or structurally invalid."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OfficialPricePayloadError(f"{field} must be a JSON object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise OfficialPricePayloadError(f"{field} must be a JSON array")
    return value


def _integer(
    value: Any,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OfficialPricePayloadError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise OfficialPricePayloadError(f"{field} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise OfficialPricePayloadError(f"{field} must be <= {maximum}")
    return value


def _number(
    value: Any,
    field: str,
    *,
    minimum: float,
    maximum: float,
    optional: bool = False,
) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool):
        raise OfficialPricePayloadError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OfficialPricePayloadError(f"{field} must be numeric") from exc
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise OfficialPricePayloadError(
            f"{field} must be finite and between {minimum} and {maximum}"
        )
    return result


def _timestamp(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise OfficialPricePayloadError(f"{field} must be null or an ISO timestamp")
    text = value.strip()
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise OfficialPricePayloadError(f"{field} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise OfficialPricePayloadError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class PriceProjection:
    offset_days: int
    projected_percent: float
    likelihood_code: int

    def to_server_dict(self) -> dict[str, Any]:
        return {
            "offset_days": self.offset_days,
            "projected_percent": self.projected_percent,
            # Deliberately not converted to labels: provider semantics are opaque.
            "likelihood_code": self.likelihood_code,
        }


@dataclass(frozen=True)
class OfficialPlayerPriceSignal:
    element_id: int
    now_cost_tenths: int
    cost_change_start_tenths: int
    cost_change_event_tenths: int
    selected_by_percent: float
    transfers_in_event: int
    transfers_out_event: int
    price_change_percent: float | None
    price_change_hourly_rate: int | None
    projections: tuple[PriceProjection, ...]
    locked_until: str | None
    calibrating: bool

    def to_server_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "now_cost_tenths": self.now_cost_tenths,
            "cost_change_start_tenths": self.cost_change_start_tenths,
            "cost_change_event_tenths": self.cost_change_event_tenths,
            "selected_by_percent": self.selected_by_percent,
            "transfers_in_event": self.transfers_in_event,
            "transfers_out_event": self.transfers_out_event,
            "price_change_percent": self.price_change_percent,
            "price_change_hourly_rate": self.price_change_hourly_rate,
            "projections": [row.to_server_dict() for row in self.projections],
            "locked_until": self.locked_until,
            "calibrating": self.calibrating,
        }


@dataclass(frozen=True)
class OfficialPriceFeed:
    price_change_deadlines: tuple[str, ...]
    players: tuple[OfficialPlayerPriceSignal, ...]
    schema_version: str = PRICE_FEED_SCHEMA_VERSION

    def to_server_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "price_change_deadlines": list(self.price_change_deadlines),
            "players": [player.to_server_dict() for player in self.players],
        }


def _parse_projection(value: Any, path: str) -> PriceProjection:
    row = _mapping(value, path)
    return PriceProjection(
        offset_days=_integer(row.get("offset"), f"{path}.offset", minimum=0, maximum=30),
        projected_percent=float(
            _number(
                row.get("projected_percent"),
                f"{path}.projected_percent",
                minimum=-1000.0,
                maximum=1000.0,
            )
        ),
        likelihood_code=_integer(
            row.get("likelihood"), f"{path}.likelihood", minimum=-10, maximum=10
        ),
    )


def parse_bootstrap_price_feed(payload: Any) -> OfficialPriceFeed:
    """Parse allowlisted price signals from an official bootstrap payload."""

    root = _mapping(payload, "bootstrap")
    raw_elements = _sequence(root.get("elements"), "bootstrap.elements")
    settings = _mapping(root.get("game_config"), "bootstrap.game_config")
    settings = _mapping(settings.get("settings"), "bootstrap.game_config.settings")
    raw_deadlines = _sequence(
        settings.get("price_change_deadlines", []),
        "bootstrap.game_config.settings.price_change_deadlines",
    )
    deadlines = tuple(
        _timestamp(value, f"price_change_deadlines[{index}]")
        for index, value in enumerate(raw_deadlines)
    )
    if any(value is None for value in deadlines):
        raise OfficialPricePayloadError("price_change_deadlines cannot contain null")
    if len(set(deadlines)) != len(deadlines) or tuple(sorted(deadlines)) != deadlines:
        raise OfficialPricePayloadError(
            "price_change_deadlines must be unique and ordered chronologically"
        )

    players: list[OfficialPlayerPriceSignal] = []
    seen_ids: set[int] = set()
    for index, value in enumerate(raw_elements):
        path = f"bootstrap.elements[{index}]"
        row = _mapping(value, path)
        element_id = _integer(row.get("id"), f"{path}.id", minimum=1)
        if element_id in seen_ids:
            raise OfficialPricePayloadError(f"duplicate element id: {element_id}")
        seen_ids.add(element_id)
        raw_projections = _sequence(
            row.get("price_change_projections", []),
            f"{path}.price_change_projections",
        )
        projections = tuple(
            _parse_projection(item, f"{path}.price_change_projections[{projection_index}]")
            for projection_index, item in enumerate(raw_projections)
        )
        offsets = [projection.offset_days for projection in projections]
        if len(set(offsets)) != len(offsets) or offsets != sorted(offsets):
            raise OfficialPricePayloadError(
                f"{path}.price_change_projections offsets must be unique and ordered"
            )
        hourly = row.get("price_change_hourly_rate")
        hourly_rate = (
            None
            if hourly is None
            else _integer(
                hourly,
                f"{path}.price_change_hourly_rate",
                minimum=-100_000_000,
                maximum=100_000_000,
            )
        )
        calibrating = row.get("price_change_calibrating")
        if not isinstance(calibrating, bool):
            raise OfficialPricePayloadError(
                f"{path}.price_change_calibrating must be a boolean"
            )
        players.append(
            OfficialPlayerPriceSignal(
                element_id=element_id,
                now_cost_tenths=_integer(
                    row.get("now_cost"), f"{path}.now_cost", minimum=1, maximum=500
                ),
                cost_change_start_tenths=_integer(
                    row.get("cost_change_start"),
                    f"{path}.cost_change_start",
                    minimum=-500,
                    maximum=500,
                ),
                cost_change_event_tenths=_integer(
                    row.get("cost_change_event"),
                    f"{path}.cost_change_event",
                    minimum=-30,
                    maximum=30,
                ),
                selected_by_percent=float(
                    _number(
                        row.get("selected_by_percent"),
                        f"{path}.selected_by_percent",
                        minimum=0.0,
                        maximum=100.0,
                    )
                ),
                transfers_in_event=_integer(
                    row.get("transfers_in_event"),
                    f"{path}.transfers_in_event",
                    minimum=0,
                ),
                transfers_out_event=_integer(
                    row.get("transfers_out_event"),
                    f"{path}.transfers_out_event",
                    minimum=0,
                ),
                price_change_percent=_number(
                    row.get("price_change_percent"),
                    f"{path}.price_change_percent",
                    minimum=-1000.0,
                    maximum=1000.0,
                    optional=True,
                ),
                price_change_hourly_rate=hourly_rate,
                projections=projections,
                locked_until=_timestamp(
                    row.get("price_change_locked_until"),
                    f"{path}.price_change_locked_until",
                ),
                calibrating=calibrating,
            )
        )
    players.sort(key=lambda player: player.element_id)
    return OfficialPriceFeed(
        price_change_deadlines=tuple(value for value in deadlines if value is not None),
        players=tuple(players),
    )


__all__ = [
    "OfficialPlayerPriceSignal",
    "OfficialPriceFeed",
    "OfficialPricePayloadError",
    "PRICE_FEED_SCHEMA_VERSION",
    "PriceProjection",
    "parse_bootstrap_price_feed",
]
