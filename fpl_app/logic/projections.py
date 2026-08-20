"""Pure projection extraction and matching utilities.

Solio's public endpoint is a ranked summary feed rather than a player table: the
same player can occur in several sections, while not every FPL player is present.
This module turns those repeated rows into a unique projection set and overlays
only high-confidence matches on the official FPL player frame.

No network, filesystem or Streamlit state is accessed here, which keeps deadline
backtests deterministic and makes matching behaviour straightforward to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import math
from numbers import Real
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import unicodedata

import pandas as pd


_PLAYER_PROJECTION_COLUMNS = (
    "projection_key",
    "name",
    "normalized_name",
    "team",
    "position",
    "price",
    "prPoints",
    "sections",
    "occurrences",
    "is_conflicting",
    "projection_values",
)

_MATCH_COLUMNS = (
    "fpl_player_id",
    "fpl_name",
    "solio_name",
    "team",
    "position",
    "price",
    "prPoints",
    "match_method",
    "match_score",
    "sections",
)

_TRANSLITERATION = str.maketrans(
    {
        "ø": "o",
        "Ø": "O",
        "ł": "l",
        "Ł": "L",
        "đ": "d",
        "Đ": "D",
        "ð": "d",
        "Ð": "D",
        "þ": "th",
        "Þ": "Th",
        "æ": "ae",
        "Æ": "Ae",
        "œ": "oe",
        "Œ": "Oe",
    }
)


def normalize_player_name(value: Any) -> str:
    """Return an accent- and punctuation-insensitive player name."""

    if value is None:
        return ""
    text = str(value).translate(_TRANSLITERATION).casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


_TEAM_CODES = {
    "ars": ("arsenal",),
    "avl": ("aston villa", "villa"),
    "bha": ("brighton", "brighton and hove albion"),
    "bou": ("bournemouth", "afc bournemouth"),
    "bre": ("brentford",),
    "bur": ("burnley",),
    "che": ("chelsea",),
    "cov": ("coventry", "coventry city"),
    "cry": ("crystal palace", "palace"),
    "eve": ("everton",),
    "ful": ("fulham",),
    "hul": ("hull", "hull city"),
    "ips": ("ipswich", "ipswich town"),
    "lee": ("leeds", "leeds united"),
    "lei": ("leicester", "leicester city"),
    "liv": ("liverpool",),
    "mci": ("man city", "manchester city"),
    "mun": ("man utd", "man united", "manchester united"),
    "new": ("newcastle", "newcastle united"),
    "nfo": ("nott m forest", "nottm forest", "nottingham forest", "forest"),
    "sou": ("southampton",),
    "sun": ("sunderland",),
    "tot": ("spurs", "tottenham", "tottenham hotspur"),
    "whu": ("west ham", "west ham united"),
    "wol": ("wolves", "wolverhampton", "wolverhampton wanderers"),
}

_TEAM_ALIAS_TO_CODE = {
    normalize_player_name(alias): code.upper()
    for code, aliases in _TEAM_CODES.items()
    for alias in (code, *aliases)
}

_POSITION_ALIASES = {
    "gk": "GKP",
    "gkp": "GKP",
    "goalkeeper": "GKP",
    "def": "DEF",
    "defender": "DEF",
    "mid": "MID",
    "midfielder": "MID",
    "fwd": "FWD",
    "forward": "FWD",
    "striker": "FWD",
}


def _normalize_team(value: Any) -> str:
    normalized = normalize_player_name(value)
    if not normalized:
        return ""
    return _TEAM_ALIAS_TO_CODE.get(normalized, normalized.replace(" ", "").upper())


def _normalize_position(value: Any) -> str:
    return _POSITION_ALIASES.get(normalize_player_name(value), "")


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


@dataclass(frozen=True)
class ProjectionIssue:
    kind: str
    reason: str
    solio_name: Optional[str] = None
    team: Optional[str] = None
    position: Optional[str] = None
    price: Optional[float] = None
    section: Optional[str] = None
    row_index: Optional[int] = None
    candidate_ids: Tuple[Any, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "reason": self.reason,
            "solio_name": self.solio_name,
            "team": self.team,
            "position": self.position,
            "price": self.price,
            "section": self.section,
            "row_index": self.row_index,
            "candidate_ids": list(self.candidate_ids),
        }


@dataclass(frozen=True)
class SolioExtractionResult:
    projections: pd.DataFrame
    source_rows_seen: int
    invalid_rows: Tuple[ProjectionIssue, ...]
    conflicts: Tuple[ProjectionIssue, ...]


@dataclass(frozen=True)
class ProjectionOverlayDiagnostics:
    gameweek: Optional[int]
    official_player_count: int
    source_rows_seen: int
    unique_projection_count: int
    usable_projection_count: int
    matched_projection_count: int
    projection_coverage: float
    official_player_coverage: float
    match_methods: Tuple[Tuple[str, int], ...]
    invalid_rows: Tuple[ProjectionIssue, ...]
    conflicts: Tuple[ProjectionIssue, ...]
    unmatched: Tuple[ProjectionIssue, ...]
    ambiguous: Tuple[ProjectionIssue, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gameweek": self.gameweek,
            "official_player_count": self.official_player_count,
            "source_rows_seen": self.source_rows_seen,
            "unique_projection_count": self.unique_projection_count,
            "usable_projection_count": self.usable_projection_count,
            "matched_projection_count": self.matched_projection_count,
            "projection_coverage": self.projection_coverage,
            "official_player_coverage": self.official_player_coverage,
            "match_methods": dict(self.match_methods),
            "invalid_rows": [issue.to_dict() for issue in self.invalid_rows],
            "conflicts": [issue.to_dict() for issue in self.conflicts],
            "unmatched": [issue.to_dict() for issue in self.unmatched],
            "ambiguous": [issue.to_dict() for issue in self.ambiguous],
        }


@dataclass(frozen=True)
class ProjectionOverlayResult:
    players: pd.DataFrame
    projections: pd.DataFrame
    matches: pd.DataFrame
    diagnostics: ProjectionOverlayDiagnostics
    output_column: str


@dataclass
class _ProjectionAccumulator:
    key: str
    name: str
    normalized_name: str
    team: str
    position: str
    price: float
    values: List[float]
    sections: set[str]
    occurrences: int = 1


@dataclass(frozen=True)
class _OfficialCandidate:
    row_position: int
    player_id: Any
    display_name: str
    team: str
    position: str
    price: float
    aliases: frozenset[str]


@dataclass(frozen=True)
class _ProposedMatch:
    projection_row: Mapping[str, Any]
    candidate: _OfficialCandidate
    method: str
    score: float


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def extract_solio_player_projections(payload: Mapping[str, Any]) -> SolioExtractionResult:
    """Extract every unique player row containing ``prPoints`` from list sections.

    Duplicate rows across leaderboards are consolidated by normalized
    name/team/position/price. If the same identity carries different projections,
    it remains visible with ``is_conflicting=True`` but is not usable for overlay.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")

    accumulators: Dict[str, _ProjectionAccumulator] = {}
    invalid: List[ProjectionIssue] = []
    source_rows_seen = 0

    for section, section_rows in payload.items():
        if not isinstance(section_rows, list):
            continue
        for row_index, row in enumerate(section_rows):
            if not isinstance(row, Mapping) or "prPoints" not in row:
                continue
            source_rows_seen += 1
            raw_name = row.get("name")
            normalized_name = normalize_player_name(raw_name)
            team = _normalize_team(row.get("team"))
            position = _normalize_position(row.get("position"))
            price = _finite_number(row.get("price"))
            points = _finite_number(row.get("prPoints"))

            missing = []
            if not normalized_name:
                missing.append("name")
            if not team:
                missing.append("team")
            if not position:
                missing.append("position")
            if price is None or price < 0:
                missing.append("price")
            if points is None:
                missing.append("prPoints")
            if missing:
                invalid.append(
                    ProjectionIssue(
                        kind="invalid_source_row",
                        reason=f"invalid or missing fields: {', '.join(missing)}",
                        solio_name=str(raw_name) if raw_name is not None else None,
                        team=str(row.get("team")) if row.get("team") is not None else None,
                        position=(
                            str(row.get("position"))
                            if row.get("position") is not None
                            else None
                        ),
                        price=price,
                        section=str(section),
                        row_index=row_index,
                    )
                )
                continue

            assert price is not None and points is not None
            key = f"{team}|{position}|{price:.3f}|{normalized_name}"
            existing = accumulators.get(key)
            if existing is None:
                accumulators[key] = _ProjectionAccumulator(
                    key=key,
                    name=str(raw_name),
                    normalized_name=normalized_name,
                    team=team,
                    position=position,
                    price=price,
                    values=[points],
                    sections={str(section)},
                )
            else:
                existing.values.append(points)
                existing.sections.add(str(section))
                existing.occurrences += 1

    rows: List[Dict[str, Any]] = []
    conflicts: List[ProjectionIssue] = []
    for accumulator in accumulators.values():
        unique_values = tuple(sorted(set(accumulator.values)))
        is_conflicting = len(unique_values) > 1
        if is_conflicting:
            conflicts.append(
                ProjectionIssue(
                    kind="projection_conflict",
                    reason=f"different prPoints values: {unique_values}",
                    solio_name=accumulator.name,
                    team=accumulator.team,
                    position=accumulator.position,
                    price=accumulator.price,
                )
            )
        rows.append(
            {
                "projection_key": accumulator.key,
                "name": accumulator.name,
                "normalized_name": accumulator.normalized_name,
                "team": accumulator.team,
                "position": accumulator.position,
                "price": accumulator.price,
                "prPoints": float("nan") if is_conflicting else unique_values[0],
                "sections": tuple(sorted(accumulator.sections)),
                "occurrences": accumulator.occurrences,
                "is_conflicting": is_conflicting,
                "projection_values": unique_values,
            }
        )

    projections = (
        pd.DataFrame(rows, columns=_PLAYER_PROJECTION_COLUMNS)
        if rows
        else _empty_frame(_PLAYER_PROJECTION_COLUMNS)
    )
    if not projections.empty:
        projections = projections.sort_values(
            ["team", "position", "price", "normalized_name"],
            kind="stable",
        ).reset_index(drop=True)

    return SolioExtractionResult(
        projections=projections,
        source_rows_seen=source_rows_seen,
        invalid_rows=tuple(invalid),
        conflicts=tuple(conflicts),
    )


def _first_existing(frame: pd.DataFrame, candidates: Iterable[str], purpose: str) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(
        f"Official FPL player DataFrame needs a {purpose} column; tried "
        f"{', '.join(candidates)}"
    )


def _name_aliases(row: Mapping[str, Any]) -> frozenset[str]:
    aliases: set[str] = set()
    for column in ("web_name", "player_name", "full_name", "second_name"):
        normalized = normalize_player_name(row.get(column))
        if normalized:
            aliases.add(normalized)

    first_name = normalize_player_name(row.get("first_name"))
    second_name = normalize_player_name(row.get("second_name"))
    web_name = normalize_player_name(row.get("web_name"))
    if first_name and second_name:
        aliases.add(f"{first_name} {second_name}")
        for given_name in first_name.split():
            aliases.add(f"{given_name[0]} {second_name}")
    if first_name and web_name:
        for given_name in first_name.split():
            aliases.add(f"{given_name[0]} {web_name}")

    # Punctuation is inconsistent across feeds ("N.Williams" vs "N Williams").
    aliases.update(alias.replace(" ", "") for alias in tuple(aliases))
    return frozenset(alias for alias in aliases if alias)


def _display_name(row: Mapping[str, Any], fallback: Any) -> str:
    for column in ("web_name", "player_name", "full_name"):
        value = row.get(column)
        if value is not None and str(value).strip():
            return str(value)
    first = str(row.get("first_name") or "").strip()
    second = str(row.get("second_name") or "").strip()
    full = " ".join(part for part in (first, second) if part)
    return full or str(fallback)


def _official_candidates(
    players: pd.DataFrame,
    *,
    id_column: str,
    team_column: str,
    position_column: str,
    price_column: str,
) -> Tuple[_OfficialCandidate, ...]:
    if id_column not in players.columns:
        raise ValueError(f"Official FPL player DataFrame is missing {id_column!r}")
    if players[id_column].isna().any() or players[id_column].duplicated().any():
        raise ValueError(f"{id_column!r} must contain unique, non-null player identifiers")

    candidates: List[_OfficialCandidate] = []
    for row_position, (_, series) in enumerate(players.iterrows()):
        row = series.to_dict()
        team = _normalize_team(row.get(team_column))
        position = _normalize_position(row.get(position_column))
        price = _finite_number(row.get(price_column))
        aliases = _name_aliases(row)
        if not team or not position or price is None or price < 0 or not aliases:
            raise ValueError(
                f"Official player {row.get(id_column)!r} has invalid team, position, "
                "price or name fields"
            )
        candidates.append(
            _OfficialCandidate(
                row_position=row_position,
                player_id=row[id_column],
                display_name=_display_name(row, row[id_column]),
                team=team,
                position=position,
                price=price,
                aliases=aliases,
            )
        )
    return tuple(candidates)


def _source_name_aliases(normalized_name: str) -> frozenset[str]:
    return frozenset({normalized_name, normalized_name.replace(" ", "")})


def _name_similarity(source_aliases: Iterable[str], candidate_aliases: Iterable[str]) -> float:
    return max(
        (
            SequenceMatcher(None, source, candidate).ratio()
            for source in source_aliases
            for candidate in candidate_aliases
        ),
        default=0.0,
    )


def overlay_solio_projections(
    official_players: pd.DataFrame,
    payload: Mapping[str, Any],
    *,
    output_column: Optional[str] = None,
    id_column: str = "id",
    team_column: Optional[str] = None,
    position_column: Optional[str] = None,
    price_column: Optional[str] = None,
    price_tolerance: float = 0.01,
    unique_candidate_name_threshold: float = 0.72,
    multi_candidate_name_threshold: float = 0.86,
    ambiguity_margin: float = 0.08,
) -> ProjectionOverlayResult:
    """Overlay current-GW Solio points on an official FPL player DataFrame.

    Team, position and price must agree before names are considered. Exact aliases
    (including first-initial + surname forms) are preferred. Fuzzy matches need a
    minimum score and, where several structural candidates exist, a clear margin.
    Ambiguous or colliding proposals are deliberately left unmatched.
    """

    if not isinstance(official_players, pd.DataFrame):
        raise TypeError("official_players must be a pandas DataFrame")
    if price_tolerance < 0:
        raise ValueError("price_tolerance cannot be negative")
    if not 0 <= unique_candidate_name_threshold <= 1:
        raise ValueError("unique_candidate_name_threshold must be between 0 and 1")
    if not 0 <= multi_candidate_name_threshold <= 1:
        raise ValueError("multi_candidate_name_threshold must be between 0 and 1")
    if not 0 <= ambiguity_margin <= 1:
        raise ValueError("ambiguity_margin must be between 0 and 1")

    team_column = team_column or _first_existing(
        official_players,
        ("short_name", "team_short_name", "team_code", "club", "team_name"),
        "team code/name",
    )
    position_column = position_column or _first_existing(
        official_players,
        ("singular_name_short", "position", "pos"),
        "position",
    )
    price_column = price_column or _first_existing(
        official_players,
        ("now_cost", "price_tenths", "price"),
        "price in FPL tenths",
    )

    extraction = extract_solio_player_projections(payload)
    candidates = _official_candidates(
        official_players,
        id_column=id_column,
        team_column=team_column,
        position_column=position_column,
        price_column=price_column,
    )

    unmatched: List[ProjectionIssue] = []
    ambiguous: List[ProjectionIssue] = []
    proposals: List[_ProposedMatch] = []

    usable = extraction.projections.loc[
        extraction.projections["is_conflicting"] == False  # noqa: E712
    ]
    for _, projection in usable.iterrows():
        structural = [
            candidate
            for candidate in candidates
            if candidate.team == projection["team"]
            and candidate.position == projection["position"]
            and abs(candidate.price - float(projection["price"])) <= price_tolerance
        ]
        issue_fields = {
            "solio_name": str(projection["name"]),
            "team": str(projection["team"]),
            "position": str(projection["position"]),
            "price": float(projection["price"]),
        }
        if not structural:
            unmatched.append(
                ProjectionIssue(
                    kind="unmatched_projection",
                    reason="no official player has the same team, position and price",
                    **issue_fields,
                )
            )
            continue

        source_aliases = _source_name_aliases(str(projection["normalized_name"]))
        exact = [
            candidate
            for candidate in structural
            if source_aliases.intersection(candidate.aliases)
        ]
        if len(exact) == 1:
            proposals.append(
                _ProposedMatch(projection.to_dict(), exact[0], "exact_alias", 1.0)
            )
            continue
        if len(exact) > 1:
            ambiguous.append(
                ProjectionIssue(
                    kind="ambiguous_projection",
                    reason="multiple structural candidates share an exact name alias",
                    candidate_ids=tuple(candidate.player_id for candidate in exact),
                    **issue_fields,
                )
            )
            continue

        ranked = sorted(
            (
                (
                    _name_similarity(source_aliases, candidate.aliases),
                    candidate,
                )
                for candidate in structural
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        best_score, best_candidate = ranked[0]
        threshold = (
            unique_candidate_name_threshold
            if len(ranked) == 1
            else multi_candidate_name_threshold
        )
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        if best_score < threshold:
            unmatched.append(
                ProjectionIssue(
                    kind="unmatched_projection",
                    reason=(
                        f"best name score {best_score:.3f} is below threshold "
                        f"{threshold:.3f}"
                    ),
                    candidate_ids=tuple(candidate.player_id for _, candidate in ranked),
                    **issue_fields,
                )
            )
            continue
        if len(ranked) > 1 and best_score - second_score < ambiguity_margin:
            ambiguous.append(
                ProjectionIssue(
                    kind="ambiguous_projection",
                    reason=(
                        f"best name score margin {best_score - second_score:.3f} is "
                        f"below {ambiguity_margin:.3f}"
                    ),
                    candidate_ids=tuple(candidate.player_id for _, candidate in ranked),
                    **issue_fields,
                )
            )
            continue
        proposals.append(
            _ProposedMatch(projection.to_dict(), best_candidate, "fuzzy_name", best_score)
        )

    # A target claimed by two distinct source identities is not safe to overlay.
    proposals_by_target: Dict[Any, List[_ProposedMatch]] = {}
    for proposal in proposals:
        proposals_by_target.setdefault(proposal.candidate.player_id, []).append(proposal)

    accepted: List[_ProposedMatch] = []
    for player_id, target_proposals in proposals_by_target.items():
        if len(target_proposals) == 1:
            accepted.append(target_proposals[0])
            continue
        for proposal in target_proposals:
            ambiguous.append(
                ProjectionIssue(
                    kind="target_collision",
                    reason="multiple distinct Solio identities resolve to one FPL player",
                    solio_name=str(proposal.projection_row["name"]),
                    team=str(proposal.projection_row["team"]),
                    position=str(proposal.projection_row["position"]),
                    price=float(proposal.projection_row["price"]),
                    candidate_ids=(player_id,),
                )
            )

    gameweek_raw = payload.get("gameweek") if isinstance(payload, Mapping) else None
    gameweek = (
        int(gameweek_raw)
        if isinstance(gameweek_raw, int) and not isinstance(gameweek_raw, bool)
        else None
    )
    output_column = output_column or (
        f"solio_ep_gw{gameweek}" if gameweek is not None else "solio_ep_current"
    )

    players = official_players.copy(deep=True)
    projection_values: List[Any] = [pd.NA] * len(players)
    match_methods: List[Any] = [pd.NA] * len(players)
    match_scores: List[Any] = [pd.NA] * len(players)
    source_names: List[Any] = [pd.NA] * len(players)
    source_sections: List[Any] = [None] * len(players)
    match_rows: List[Dict[str, Any]] = []
    for match in accepted:
        position = match.candidate.row_position
        projection_values[position] = float(match.projection_row["prPoints"])
        match_methods[position] = match.method
        match_scores[position] = float(match.score)
        source_names[position] = str(match.projection_row["name"])
        source_sections[position] = tuple(match.projection_row["sections"])
        match_rows.append(
            {
                "fpl_player_id": match.candidate.player_id,
                "fpl_name": match.candidate.display_name,
                "solio_name": str(match.projection_row["name"]),
                "team": str(match.projection_row["team"]),
                "position": str(match.projection_row["position"]),
                "price": float(match.projection_row["price"]),
                "prPoints": float(match.projection_row["prPoints"]),
                "match_method": match.method,
                "match_score": float(match.score),
                "sections": tuple(match.projection_row["sections"]),
            }
        )

    players[output_column] = pd.array(projection_values, dtype="Float64")
    players["solio_match_method"] = pd.array(match_methods, dtype="string")
    players["solio_match_score"] = pd.array(match_scores, dtype="Float64")
    players["solio_projection_name"] = pd.array(source_names, dtype="string")
    players["solio_projection_sections"] = source_sections
    matches = (
        pd.DataFrame(match_rows, columns=_MATCH_COLUMNS)
        if match_rows
        else _empty_frame(_MATCH_COLUMNS)
    )

    method_counts: Dict[str, int] = {}
    for match in accepted:
        method_counts[match.method] = method_counts.get(match.method, 0) + 1
    usable_count = len(usable)
    matched_count = len(accepted)
    official_count = len(official_players)
    diagnostics = ProjectionOverlayDiagnostics(
        gameweek=gameweek,
        official_player_count=official_count,
        source_rows_seen=extraction.source_rows_seen,
        unique_projection_count=len(extraction.projections),
        usable_projection_count=usable_count,
        matched_projection_count=matched_count,
        projection_coverage=(matched_count / usable_count if usable_count else 0.0),
        official_player_coverage=(matched_count / official_count if official_count else 0.0),
        match_methods=tuple(sorted(method_counts.items())),
        invalid_rows=extraction.invalid_rows,
        conflicts=extraction.conflicts,
        unmatched=tuple(unmatched),
        ambiguous=tuple(ambiguous),
    )
    return ProjectionOverlayResult(
        players=players,
        projections=extraction.projections,
        matches=matches,
        diagnostics=diagnostics,
        output_column=output_column,
    )
