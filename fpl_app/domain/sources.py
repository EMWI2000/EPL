"""Data-source registry and point-in-time provenance primitives.

The forecasting pipeline must be able to answer two questions for every input:

1. Where did the value come from?
2. What was known at the time of the FPL deadline?

This module deliberately contains no network or storage code.  It provides an
immutable registry plus metadata that adapters and snapshot stores can share.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Tuple
from urllib.parse import urlparse


FPL_OFFICIAL_SOURCE_ID = "fpl_official"
SOLIO_SOURCE_ID = "solio"
THE_ODDS_API_SOURCE_ID = "the_odds_api"

_SOURCE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


def _require_http_url(value: str, field_name: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be an absolute HTTP(S) URL")


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")


def as_utc(value: datetime) -> datetime:
    """Return an aware datetime normalised to UTC."""

    _require_aware_datetime(value, "datetime")
    return value.astimezone(timezone.utc)


def datetime_to_iso(value: datetime) -> str:
    """Serialise an aware timestamp with an explicit UTC marker."""

    return as_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class SourceDefinition:
    """Stable metadata describing an upstream data source."""

    source_id: str
    name: str
    homepage_url: str
    data_url: str
    access_mode: str
    attribution: str
    refresh_cadence: str
    terms_url: Optional[str] = None
    usage_notes: str = ""

    def __post_init__(self) -> None:
        if not _SOURCE_ID_PATTERN.fullmatch(self.source_id):
            raise ValueError(
                "source_id must use lowercase letters, digits and underscores "
                "and cannot contain path separators"
            )
        for field_name in (
            "name",
            "access_mode",
            "attribution",
            "refresh_cadence",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} cannot be empty")
        _require_http_url(self.homepage_url, "homepage_url")
        _require_http_url(self.data_url, "data_url")
        if self.terms_url:
            _require_http_url(self.terms_url, "terms_url")

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "source_id": self.source_id,
            "name": self.name,
            "homepage_url": self.homepage_url,
            "data_url": self.data_url,
            "access_mode": self.access_mode,
            "attribution": self.attribution,
            "refresh_cadence": self.refresh_cadence,
            "terms_url": self.terms_url,
            "usage_notes": self.usage_notes,
        }


class SourceRegistry:
    """Read-only registry that rejects duplicate source identifiers."""

    def __init__(self, sources: Iterable[SourceDefinition]) -> None:
        by_id: Dict[str, SourceDefinition] = {}
        for source in sources:
            if source.source_id in by_id:
                raise ValueError(f"Duplicate source_id: {source.source_id}")
            by_id[source.source_id] = source
        self._sources: Mapping[str, SourceDefinition] = MappingProxyType(by_id)

    def get(self, source_id: str) -> SourceDefinition:
        try:
            return self._sources[source_id]
        except KeyError as exc:
            known = ", ".join(sorted(self._sources))
            raise KeyError(f"Unknown source_id {source_id!r}. Known sources: {known}") from exc

    def all(self) -> Tuple[SourceDefinition, ...]:
        return tuple(self._sources[key] for key in sorted(self._sources))

    def as_mapping(self) -> Mapping[str, SourceDefinition]:
        return self._sources


SOURCE_REGISTRY = SourceRegistry(
    (
        SourceDefinition(
            source_id=FPL_OFFICIAL_SOURCE_ID,
            name="Fantasy Premier League API",
            homepage_url="https://fantasy.premierleague.com/",
            data_url="https://fantasy.premierleague.com/api/",
            access_mode="Public, undocumented JSON endpoints",
            attribution="Premier League / Fantasy Premier League",
            refresh_cadence="Variable during the season",
            terms_url="https://www.premierleague.com/terms-and-conditions",
            usage_notes=(
                "Cache conservatively. Treat the API as an unsupported upstream "
                "and review Premier League terms before redistributing data."
            ),
        ),
        SourceDefinition(
            source_id=SOLIO_SOURCE_ID,
            name="Solio FPL public projections",
            homepage_url="https://fpl.solioanalytics.com/",
            data_url="https://fpl.solioanalytics.com/api/data/latest.json",
            access_mode="Public JSON endpoint; no authentication",
            attribution="Solio Analytics",
            refresh_cadence="Approximately every four hours",
            terms_url=None,
            usage_notes=(
                "Attribute Solio Analytics. Keep raw snapshots for model comparison; "
                "do not assume the endpoint has an availability SLA."
            ),
        ),
        SourceDefinition(
            source_id=THE_ODDS_API_SOURCE_ID,
            name="The Odds API",
            homepage_url="https://the-odds-api.com/",
            data_url="https://api.the-odds-api.com/v4/",
            access_mode="API key and usage-credit limits",
            attribution="The Odds API",
            refresh_cadence="Plan dependent",
            terms_url="https://the-odds-api.com/terms-and-conditions.html",
            usage_notes=(
                "Persist request timestamps and market parameters so historical "
                "backtests only use prices observed before each deadline. Do not "
                "redistribute the feed as a standalone data product."
            ),
        ),
    )
)


def get_source(source_id: str) -> SourceDefinition:
    return SOURCE_REGISTRY.get(source_id)


def list_sources() -> Tuple[SourceDefinition, ...]:
    return SOURCE_REGISTRY.all()


@dataclass(frozen=True)
class Provenance:
    """The time and origin of a single untransformed payload."""

    source_id: str
    source_url: str
    observed_at: datetime
    schema_version: str
    effective_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        # Requiring a registry entry prevents path traversal and silent source drift.
        get_source(self.source_id)
        _require_http_url(self.source_url, "source_url")
        _require_aware_datetime(self.observed_at, "observed_at")
        if self.effective_at is not None:
            _require_aware_datetime(self.effective_at, "effective_at")
        if not str(self.schema_version).strip():
            raise ValueError("schema_version cannot be empty")

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "source_id": self.source_id,
            "source_url": self.source_url,
            "observed_at": datetime_to_iso(self.observed_at),
            "effective_at": (
                datetime_to_iso(self.effective_at)
                if self.effective_at is not None
                else None
            ),
            "schema_version": self.schema_version,
        }
