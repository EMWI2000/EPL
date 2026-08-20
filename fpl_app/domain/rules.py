"""Authoritative, framework-independent FPL rules for the 2026/27 season.

All money is represented in tenths of a million pounds, matching the official
FPL API (``1000`` means £100.0m).  Keeping these rules here avoids scattering
season-specific numbers across feature engineering, scoring and optimizers.

Primary sources, checked 2026-08-20:

* https://www.premierleague.com/en/news/2174909/fpl-basics-explained-scoring-points
* https://www.premierleague.com/en/news/4679873/all-you-need-to-know-about-changes-to-fpl-for-202627
* https://www.premierleague.com/en/news/4679879/whats-happening-with-fpl-chips-in-202627
* https://www.premierleague.com/en/news/4679946/whats-new-in-202627-fantasy-changes-to-bonus-points-system
* https://fantasy.premierleague.com/api/bootstrap-static/
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence


RULESET_SEASON = "2026/27"
RULESET_VERIFIED_ON = date(2026, 8, 20)
FIRST_GAMEWEEK = 1
LAST_GAMEWEEK = 38


class Position(str, Enum):
    """Position codes used by the official FPL API."""

    GOALKEEPER = "GKP"
    DEFENDER = "DEF"
    MIDFIELDER = "MID"
    FORWARD = "FWD"


class Chip(str, Enum):
    """Chip codes used by the official FPL API."""

    WILDCARD = "wildcard"
    FREE_HIT = "freehit"
    BENCH_BOOST = "bboost"
    TRIPLE_CAPTAIN = "3xc"


class DefensiveAction(str, Enum):
    CLEARANCE = "clearance"
    BLOCK = "block"
    INTERCEPTION = "interception"
    TACKLE = "tackle"
    BALL_RECOVERY = "ball_recovery"


@dataclass(frozen=True)
class Formation:
    """One legal starting-XI formation, including its goalkeeper."""

    code: str
    positions: Mapping[Position, int]

    def __post_init__(self) -> None:
        if sum(self.positions.values()) != 11:
            raise ValueError(f"Formation {self.code} must contain 11 players")


def _formation(code: str, defenders: int, midfielders: int, forwards: int) -> Formation:
    return Formation(
        code=code,
        positions=MappingProxyType(
            {
                Position.GOALKEEPER: 1,
                Position.DEFENDER: defenders,
                Position.MIDFIELDER: midfielders,
                Position.FORWARD: forwards,
            }
        ),
    )


# The eight legal FPL formations.  In particular, 5-2-3 is legal and must not
# be omitted by an optimizer.
FORMATIONS: tuple[Formation, ...] = (
    _formation("343", 3, 4, 3),
    _formation("352", 3, 5, 2),
    _formation("433", 4, 3, 3),
    _formation("442", 4, 4, 2),
    _formation("451", 4, 5, 1),
    _formation("523", 5, 2, 3),
    _formation("532", 5, 3, 2),
    _formation("541", 5, 4, 1),
)
FORMATION_BY_CODE: Mapping[str, Formation] = MappingProxyType(
    {formation.code: formation for formation in FORMATIONS}
)

SQUAD_POSITION_QUOTAS: Mapping[Position, int] = MappingProxyType(
    {
        Position.GOALKEEPER: 2,
        Position.DEFENDER: 5,
        Position.MIDFIELDER: 5,
        Position.FORWARD: 3,
    }
)
STARTING_POSITION_MINIMUMS: Mapping[Position, int] = MappingProxyType(
    {
        Position.GOALKEEPER: 1,
        Position.DEFENDER: 3,
        Position.MIDFIELDER: 2,
        Position.FORWARD: 1,
    }
)
STARTING_POSITION_MAXIMUMS: Mapping[Position, int] = MappingProxyType(
    {
        Position.GOALKEEPER: 1,
        Position.DEFENDER: 5,
        Position.MIDFIELDER: 5,
        Position.FORWARD: 3,
    }
)


@dataclass(frozen=True)
class SquadRules:
    squad_size: int = 15
    starting_size: int = 11
    bench_size: int = 4
    max_players_per_club: int = 3
    initial_budget_tenths: int = 1000
    position_quotas: Mapping[Position, int] = field(
        default_factory=lambda: SQUAD_POSITION_QUOTAS
    )
    starting_position_minimums: Mapping[Position, int] = field(
        default_factory=lambda: STARTING_POSITION_MINIMUMS
    )
    starting_position_maximums: Mapping[Position, int] = field(
        default_factory=lambda: STARTING_POSITION_MAXIMUMS
    )
    formations: tuple[Formation, ...] = FORMATIONS


SQUAD = SquadRules()


@dataclass(frozen=True)
class PositionScoring:
    goal: int
    assist: int
    clean_sheet: int
    goals_conceded_block_size: int | None
    goals_conceded_block_points: int
    defensive_contribution_threshold: int | None
    defensive_contribution_points: int
    defensive_contribution_actions: frozenset[DefensiveAction]


_CBIT = frozenset(
    {
        DefensiveAction.CLEARANCE,
        DefensiveAction.BLOCK,
        DefensiveAction.INTERCEPTION,
        DefensiveAction.TACKLE,
    }
)
_CBIRT = _CBIT | {DefensiveAction.BALL_RECOVERY}

POSITION_SCORING: Mapping[Position, PositionScoring] = MappingProxyType(
    {
        Position.GOALKEEPER: PositionScoring(
            goal=10,
            assist=3,
            clean_sheet=4,
            goals_conceded_block_size=2,
            goals_conceded_block_points=-1,
            defensive_contribution_threshold=None,
            defensive_contribution_points=0,
            defensive_contribution_actions=frozenset(),
        ),
        Position.DEFENDER: PositionScoring(
            goal=6,
            assist=3,
            clean_sheet=4,
            goals_conceded_block_size=2,
            goals_conceded_block_points=-1,
            defensive_contribution_threshold=10,
            defensive_contribution_points=2,
            defensive_contribution_actions=_CBIT,
        ),
        Position.MIDFIELDER: PositionScoring(
            goal=5,
            assist=3,
            clean_sheet=1,
            goals_conceded_block_size=None,
            goals_conceded_block_points=0,
            defensive_contribution_threshold=12,
            defensive_contribution_points=2,
            defensive_contribution_actions=_CBIRT,
        ),
        Position.FORWARD: PositionScoring(
            goal=4,
            assist=3,
            clean_sheet=0,
            goals_conceded_block_size=None,
            goals_conceded_block_points=0,
            defensive_contribution_threshold=12,
            defensive_contribution_points=2,
            defensive_contribution_actions=_CBIRT,
        ),
    }
)


@dataclass(frozen=True)
class ScoringRules:
    appearance_under_60: int = 1
    appearance_60_or_more: int = 2
    long_appearance_minutes: int = 60
    saves_per_point: int = 3
    save_points: int = 1
    penalty_save: int = 5
    penalty_miss: int = -2
    yellow_card: int = -1
    red_card: int = -3
    own_goal: int = -2
    bonus_awards: tuple[int, int, int] = (3, 2, 1)
    captain_multiplier: int = 2
    triple_captain_multiplier: int = 3
    positions: Mapping[Position, PositionScoring] = field(
        default_factory=lambda: POSITION_SCORING
    )


SCORING = ScoringRules()


@dataclass(frozen=True)
class BonusPointRules:
    """The explicitly announced 2026/27 BPS values, not a full BPS model."""

    match_bonus_awards: tuple[int, int, int] = (3, 2, 1)
    cbi_actions_per_bps: int = 3
    any_goalkeeper_save_bps: int = 2
    inside_box_save_extra_bps: int = 1
    big_chance_save_extra_bps: int = 1
    penalty_save_bps: int = 7
    tackled_metric_active: bool = False


BONUS_POINTS = BonusPointRules()


@dataclass(frozen=True)
class TransferRules:
    free_transfers_per_gameweek: int = 1
    max_banked_free_transfers: int = 5
    additional_transfer_cost_points: int = 4
    transfer_cap_per_gameweek: int = 20
    sale_profit_share_numerator: int = 1
    sale_profit_share_denominator: int = 2
    bank_preserved_by_wildcard: bool = True
    bank_preserved_by_free_hit: bool = True


TRANSFERS = TransferRules()


@dataclass(frozen=True)
class ChipWindow:
    half: int
    first_gameweek: int
    last_gameweek: int

    def contains(self, gameweek: int) -> bool:
        return self.first_gameweek <= gameweek <= self.last_gameweek


FIRST_HALF_TEAM_CHIP_WINDOW = ChipWindow(half=1, first_gameweek=1, last_gameweek=19)
FIRST_HALF_TRANSFER_CHIP_WINDOW = ChipWindow(half=1, first_gameweek=2, last_gameweek=19)
SECOND_HALF_CHIP_WINDOW = ChipWindow(half=2, first_gameweek=20, last_gameweek=38)


@dataclass(frozen=True)
class ChipRule:
    chip: Chip
    windows: tuple[ChipWindow, ChipWindow]
    permanent_transfers: bool = False
    temporary_squad_for_one_gameweek: bool = False
    includes_bench_points: bool = False
    captain_multiplier: int = 2


CHIP_RULES: Mapping[Chip, ChipRule] = MappingProxyType(
    {
        Chip.WILDCARD: ChipRule(
            chip=Chip.WILDCARD,
            windows=(FIRST_HALF_TRANSFER_CHIP_WINDOW, SECOND_HALF_CHIP_WINDOW),
            permanent_transfers=True,
        ),
        Chip.FREE_HIT: ChipRule(
            chip=Chip.FREE_HIT,
            windows=(FIRST_HALF_TRANSFER_CHIP_WINDOW, SECOND_HALF_CHIP_WINDOW),
            temporary_squad_for_one_gameweek=True,
        ),
        Chip.BENCH_BOOST: ChipRule(
            chip=Chip.BENCH_BOOST,
            windows=(FIRST_HALF_TEAM_CHIP_WINDOW, SECOND_HALF_CHIP_WINDOW),
            includes_bench_points=True,
        ),
        Chip.TRIPLE_CAPTAIN: ChipRule(
            chip=Chip.TRIPLE_CAPTAIN,
            windows=(FIRST_HALF_TEAM_CHIP_WINDOW, SECOND_HALF_CHIP_WINDOW),
            captain_multiplier=3,
        ),
    }
)


@dataclass(frozen=True)
class ChipSetRules:
    chips_per_half: int = 4
    chip_sets: int = 2
    total_chips: int = 8
    first_set_last_gameweek: int = 19
    second_set_first_gameweek: int = 20
    max_chips_per_gameweek: int = 1
    consecutive_free_hits_across_halves_allowed: bool = False
    rules: Mapping[Chip, ChipRule] = field(default_factory=lambda: CHIP_RULES)


CHIPS = ChipSetRules()


@dataclass(frozen=True)
class SquadPlayer:
    """Minimal input needed to validate an initial FPL squad."""

    player_id: int
    team_id: int
    position: Position
    price_tenths: int


def _position(value: Position | str) -> Position:
    try:
        return value if isinstance(value, Position) else Position(value)
    except ValueError as exc:
        raise ValueError(f"Unknown FPL position: {value!r}") from exc


def is_valid_formation(position_counts: Mapping[Position | str, int]) -> bool:
    """Return whether ``position_counts`` describes a legal starting XI."""

    normalized = Counter({_position(position): count for position, count in position_counts.items()})
    return any(
        all(normalized.get(position, 0) == count for position, count in formation.positions.items())
        and sum(normalized.values()) == SQUAD.starting_size
        for formation in FORMATIONS
    )


def formation_code(position_counts: Mapping[Position | str, int]) -> str | None:
    """Return the standard three-digit code for a legal formation."""

    normalized = Counter({_position(position): count for position, count in position_counts.items()})
    for formation in FORMATIONS:
        if normalized == Counter(formation.positions):
            return formation.code
    return None


def validate_squad(players: Sequence[SquadPlayer]) -> tuple[str, ...]:
    """Return all violations of the initial-squad constraints."""

    errors: list[str] = []
    if len(players) != SQUAD.squad_size:
        errors.append(f"Squad must contain exactly {SQUAD.squad_size} players")

    player_ids = [player.player_id for player in players]
    if len(set(player_ids)) != len(player_ids):
        errors.append("Squad cannot contain duplicate players")

    position_counts = Counter(_position(player.position) for player in players)
    for position, required in SQUAD.position_quotas.items():
        if position_counts.get(position, 0) != required:
            errors.append(f"Squad must contain exactly {required} {position.value} players")

    club_counts = Counter(player.team_id for player in players)
    if any(count > SQUAD.max_players_per_club for count in club_counts.values()):
        errors.append(f"Squad can contain at most {SQUAD.max_players_per_club} players per club")

    if any(player.price_tenths < 0 for player in players):
        errors.append("Player prices cannot be negative")
    if sum(player.price_tenths for player in players) > SQUAD.initial_budget_tenths:
        errors.append(f"Squad cannot cost more than {SQUAD.initial_budget_tenths} tenths")

    return tuple(errors)


def appearance_points(minutes: int) -> int:
    if minutes < 0:
        raise ValueError("Minutes cannot be negative")
    if minutes == 0:
        return 0
    if minutes < SCORING.long_appearance_minutes:
        return SCORING.appearance_under_60
    return SCORING.appearance_60_or_more


def goalkeeper_save_points(saves: int) -> int:
    if saves < 0:
        raise ValueError("Saves cannot be negative")
    return (saves // SCORING.saves_per_point) * SCORING.save_points


def goals_conceded_points(position: Position | str, goals_conceded: int) -> int:
    if goals_conceded < 0:
        raise ValueError("Goals conceded cannot be negative")
    rule = POSITION_SCORING[_position(position)]
    if rule.goals_conceded_block_size is None:
        return 0
    return (
        goals_conceded // rule.goals_conceded_block_size
    ) * rule.goals_conceded_block_points


def defensive_contribution_points(
    position: Position | str,
    *,
    clearances: int = 0,
    blocks: int = 0,
    interceptions: int = 0,
    tackles: int = 0,
    ball_recoveries: int = 0,
) -> int:
    """Calculate the once-per-match defensive-contribution award.

    Defenders use CBIT and need 10 actions.  Midfielders and forwards use
    CBIRT (including ball recoveries) and need 12.  Goalkeepers are ineligible.
    """

    counts = {
        DefensiveAction.CLEARANCE: clearances,
        DefensiveAction.BLOCK: blocks,
        DefensiveAction.INTERCEPTION: interceptions,
        DefensiveAction.TACKLE: tackles,
        DefensiveAction.BALL_RECOVERY: ball_recoveries,
    }
    if any(value < 0 for value in counts.values()):
        raise ValueError("Defensive-action counts cannot be negative")

    rule = POSITION_SCORING[_position(position)]
    if rule.defensive_contribution_threshold is None:
        return 0
    total = sum(counts[action] for action in rule.defensive_contribution_actions)
    if total >= rule.defensive_contribution_threshold:
        return rule.defensive_contribution_points
    return 0


def transfer_points_cost(
    transfers_made: int,
    free_transfers: int,
    *,
    active_chip: Chip | str | None = None,
) -> int:
    """Return the non-negative points cost of transfers in one Gameweek."""

    if transfers_made < 0 or free_transfers < 0:
        raise ValueError("Transfer counts cannot be negative")
    chip = Chip(active_chip) if active_chip is not None else None
    if chip in {Chip.WILDCARD, Chip.FREE_HIT}:
        return 0
    paid_transfers = max(0, transfers_made - free_transfers)
    return paid_transfers * TRANSFERS.additional_transfer_cost_points


def free_transfers_next_gameweek(
    current_bank: int,
    transfers_made: int,
    *,
    active_chip: Chip | str | None = None,
) -> int:
    """Calculate the next bank, including preservation by WC and Free Hit."""

    if not 0 <= current_bank <= TRANSFERS.max_banked_free_transfers:
        raise ValueError("Current free-transfer bank is outside the legal range")
    if transfers_made < 0:
        raise ValueError("Transfers made cannot be negative")
    chip = Chip(active_chip) if active_chip is not None else None
    if chip in {Chip.WILDCARD, Chip.FREE_HIT}:
        return current_bank
    remaining = max(0, current_bank - transfers_made)
    return min(
        TRANSFERS.max_banked_free_transfers,
        remaining + TRANSFERS.free_transfers_per_gameweek,
    )


def selling_price_tenths(purchase_price_tenths: int, current_price_tenths: int) -> int:
    """Calculate FPL selling price, taking half of any rise, rounded down."""

    if purchase_price_tenths < 0 or current_price_tenths < 0:
        raise ValueError("Prices cannot be negative")
    if current_price_tenths <= purchase_price_tenths:
        return current_price_tenths
    profit = current_price_tenths - purchase_price_tenths
    realised_profit = (
        profit * TRANSFERS.sale_profit_share_numerator
    ) // TRANSFERS.sale_profit_share_denominator
    return purchase_price_tenths + realised_profit


def chip_window(chip: Chip | str, gameweek: int) -> ChipWindow | None:
    """Return the chip-set window available in ``gameweek``, if any."""

    if not FIRST_GAMEWEEK <= gameweek <= LAST_GAMEWEEK:
        return None
    rule = CHIP_RULES[Chip(chip)]
    return next((window for window in rule.windows if window.contains(gameweek)), None)


def can_play_chip(
    chip: Chip | str,
    gameweek: int,
    *,
    used_halves: Iterable[int] = (),
    another_chip_selected: bool = False,
    previous_gameweek_chip: Chip | str | None = None,
) -> bool:
    """Validate the season-level chip constraints known before a deadline."""

    selected_chip = Chip(chip)
    window = chip_window(selected_chip, gameweek)
    if window is None or window.half in set(used_halves) or another_chip_selected:
        return False
    previous_chip = Chip(previous_gameweek_chip) if previous_gameweek_chip is not None else None
    if (
        selected_chip is Chip.FREE_HIT
        and gameweek == CHIPS.second_set_first_gameweek
        and previous_chip is Chip.FREE_HIT
        and not CHIPS.consecutive_free_hits_across_halves_allowed
    ):
        return False
    return True
