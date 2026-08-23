"""Bounded chip counterfactuals for one proven transfer roadmap.

Chip values are always measured against the supplied no-chip roadmap.  The
module does not turn a chip into an executable action: it screens Triple
Captain and Bench Boost from the roadmap lineups, solves one bounded Wildcard
rebuild, and only solves a Free Hit counterfactual when the official fixture
data already contains a blank or double gameweek.

Future prices are held at today's values.  Every result is therefore a
recalculate-at-deadline decision aid, not a season-wide chip optimum.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import monotonic
from typing import Collection, Mapping, Sequence

import pandas as pd

from ..domain.rules import (
    TRANSFERS,
    Chip,
    can_play_chip,
    chip_window,
)
from .squad_plan import (
    DEFAULT_BENCH_WEIGHTS,
    SquadPlanError,
    SquadPlanResult,
    optimize_squad_plan,
)
from .strategy_planner import BoundedStrategyResult


CHIP_ORDER = (Chip.WILDCARD, Chip.FREE_HIT, Chip.BENCH_BOOST, Chip.TRIPLE_CAPTAIN)
WILDCARD_CONSIDER_GAIN_POINTS = 15.0
WILDCARD_MINIMUM_CHANGES = 4
BENCH_BOOST_CONSIDER_GAIN_POINTS = 12.0
TRIPLE_CAPTAIN_CONSIDER_GAIN_POINTS = 8.0
MODEL_SCOPE = "bounded_chip_counterfactuals"


class ChipStrategyError(ValueError):
    """Raised when chip inventory or roadmap inputs are contradictory."""


@dataclass(frozen=True)
class ChipInventoryEntry:
    chip: str
    used_events: tuple[int, ...]
    available_for_target: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "chip": self.chip,
            "used_events": list(self.used_events),
            "available_for_target": self.available_for_target,
        }


@dataclass(frozen=True)
class ChipScenario:
    scenario_id: str
    chip: str
    event: int | None
    signal: str
    available: bool
    estimated_gain_points: float | None
    baseline_points: float | None
    chip_points: float | None
    confidence: str
    model_scope: str
    reason: str
    squad_ids: tuple[int, ...] = ()
    change_count: int | None = None
    bank_after_tenths: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "chip": self.chip,
            "event": self.event,
            "signal": self.signal,
            "available": self.available,
            "estimated_gain_points": self.estimated_gain_points,
            "baseline_points": self.baseline_points,
            "chip_points": self.chip_points,
            "confidence": self.confidence,
            "model_scope": self.model_scope,
            "reason": self.reason,
            "squad_ids": list(self.squad_ids),
            "change_count": self.change_count,
            "bank_after_tenths": self.bank_after_tenths,
        }


@dataclass(frozen=True)
class ChipRecommendation:
    action: str
    scenario_id: str | None
    chip: str | None
    event: int | None
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "scenario_id": self.scenario_id,
            "chip": self.chip,
            "event": self.event,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ChipStrategyResult:
    horizon: int
    target_event: int
    inventory: tuple[ChipInventoryEntry, ...]
    scenarios: tuple[ChipScenario, ...]
    recommendation: ChipRecommendation
    model_scope: str = MODEL_SCOPE
    globally_optimal: bool = False
    recalculate_each_deadline: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "horizon": self.horizon,
            "target_event": self.target_event,
            "inventory": [entry.as_dict() for entry in self.inventory],
            "scenarios": [scenario.as_dict() for scenario in self.scenarios],
            "recommendation": self.recommendation.as_dict(),
            "model_scope": self.model_scope,
            "globally_optimal": self.globally_optimal,
            "recalculate_each_deadline": self.recalculate_each_deadline,
        }


def _normalized_usage(
    chip_usage: Mapping[str, Sequence[int]],
    *,
    before_event: int,
) -> dict[Chip, tuple[int, ...]]:
    if not isinstance(chip_usage, Mapping):
        raise ChipStrategyError("chip_usage must be a mapping")
    unknown = sorted(set(chip_usage) - {chip.value for chip in Chip})
    if unknown:
        raise ChipStrategyError("chip_usage contains unknown chips")

    normalized: dict[Chip, tuple[int, ...]] = {}
    seen_events: set[int] = set()
    free_hit_events: set[int] = set()
    for chip in CHIP_ORDER:
        raw_events = chip_usage.get(chip.value, ())
        if isinstance(raw_events, (str, bytes)) or not isinstance(raw_events, Sequence):
            raise ChipStrategyError(f"chip_usage.{chip.value} must be a sequence")
        events: list[int] = []
        used_halves: set[int] = set()
        for raw_event in raw_events:
            if not isinstance(raw_event, int) or isinstance(raw_event, bool):
                raise ChipStrategyError("chip usage events must be integers")
            if raw_event >= before_event:
                raise ChipStrategyError("chip usage events must be earlier than target_event")
            window = chip_window(chip, raw_event)
            if window is None:
                raise ChipStrategyError(
                    f"{chip.value} cannot be used in GW{raw_event}"
                )
            if window.half in used_halves:
                raise ChipStrategyError(
                    f"{chip.value} is used more than once in one half"
                )
            used_halves.add(window.half)
            if raw_event in seen_events:
                raise ChipStrategyError("only one chip can be used in one gameweek")
            seen_events.add(raw_event)
            if chip is Chip.FREE_HIT:
                free_hit_events.add(raw_event)
            events.append(raw_event)
        normalized[chip] = tuple(sorted(events))
    if {19, 20}.issubset(free_hit_events):
        raise ChipStrategyError("Free Hit cannot be used in consecutive GW19 and GW20")
    return normalized


def _roadmap_gameweeks(
    roadmap: BoundedStrategyResult,
) -> dict[int, tuple[tuple[int, ...], object]]:
    values: dict[int, tuple[tuple[int, ...], object]] = {}
    for step in roadmap.best_roadmap.steps:
        for gameweek in step.gameweeks:
            offset = int(gameweek.gameweek)
            if offset in values:
                raise ChipStrategyError("roadmap gameweeks must be unique")
            values[offset] = (tuple(int(value) for value in step.squad_ids), gameweek)
    if set(values) != set(range(1, roadmap.horizon + 1)):
        raise ChipStrategyError("roadmap must cover every strategy gameweek")
    return values


def _scenario_confidence(
    frame_by_id: pd.DataFrame,
    player_ids: Sequence[int],
    offset: int,
) -> str:
    confidence_column = f"confidence_gw{offset}"
    reliability_column = f"reliability_gw{offset}"
    if confidence_column not in frame_by_id or reliability_column not in frame_by_id:
        return "low"
    confidences = [float(frame_by_id.at[player_id, confidence_column]) for player_id in player_ids]
    reliabilities = [str(frame_by_id.at[player_id, reliability_column]) for player_id in player_ids]
    if (
        confidences
        and min(confidences) >= 0.55
        and all(value in {"medium", "high"} for value in reliabilities)
    ):
        return "medium"
    return "low"


def _available_for_event(
    chip: Chip,
    event: int,
    usage: Mapping[Chip, tuple[int, ...]],
) -> bool:
    window = chip_window(chip, event)
    if window is None:
        return False
    used_halves = {
        used_window.half
        for used_event in usage[chip]
        for used_window in [chip_window(chip, used_event)]
        if used_window is not None
    }
    previous_chip = None
    if chip is Chip.FREE_HIT and event > 1 and event - 1 in usage[chip]:
        previous_chip = Chip.FREE_HIT
    return can_play_chip(
        chip,
        event,
        used_halves=used_halves,
        previous_gameweek_chip=previous_chip,
    )


def _lineup_scenarios(
    players: pd.DataFrame,
    roadmap: BoundedStrategyResult,
    *,
    target_event: int,
    usage: Mapping[Chip, tuple[int, ...]],
) -> tuple[ChipScenario, ChipScenario]:
    by_id = players.set_index("id", drop=False)
    gameweeks = _roadmap_gameweeks(roadmap)
    bench_weights = DEFAULT_BENCH_WEIGHTS
    average_outfield_weight = sum(bench_weights[:3]) / 3.0

    tc_candidates: list[tuple[float, int, float, int, str]] = []
    bb_candidates: list[tuple[float, int, float, tuple[int, ...], str]] = []
    for offset in range(1, roadmap.horizon + 1):
        event = target_event + offset - 1
        squad_ids, gameweek = gameweeks[offset]
        starters = tuple(int(value) for value in gameweek.starting_ids)
        captain_id = int(gameweek.captain_id)
        baseline = float(gameweek.projected_points)
        ep_column = f"ep_gw{offset}"

        if _available_for_event(Chip.TRIPLE_CAPTAIN, event, usage):
            gain = float(by_id.at[captain_id, ep_column])
            confidence = _scenario_confidence(by_id, (captain_id,), offset)
            tc_candidates.append((gain, event, baseline, captain_id, confidence))

        if _available_for_event(Chip.BENCH_BOOST, event, usage):
            bench_ids = tuple(sorted(set(squad_ids) - set(starters)))
            normal_reserve_value = 0.0
            full_bench_value = 0.0
            for player_id in bench_ids:
                ep = float(by_id.at[player_id, ep_column])
                full_bench_value += ep
                no_show = float(by_id.at[player_id, f"no_show_prob_gw{offset}"])
                reserve_weight = (
                    bench_weights[3]
                    if str(by_id.at[player_id, "pos"]) == "GKP"
                    else average_outfield_weight
                )
                normal_reserve_value += ep * (1.0 - no_show) * reserve_weight
            gain = max(0.0, full_bench_value - normal_reserve_value)
            confidence = _scenario_confidence(by_id, bench_ids, offset)
            bb_candidates.append((gain, event, baseline, bench_ids, confidence))

    def unavailable(chip: Chip, scope: str) -> ChipScenario:
        return ChipScenario(
            scenario_id=f"{chip.value}-unavailable",
            chip=chip.value,
            event=None,
            signal="hold",
            available=False,
            estimated_gain_points=None,
            baseline_points=None,
            chip_points=None,
            confidence="low",
            model_scope=scope,
            reason="Chippen er ikke tilgængelig i det beregnede vindue.",
        )

    if tc_candidates:
        gain, event, baseline, _captain_id, confidence = max(
            tc_candidates,
            key=lambda value: (value[0], -value[1]),
        )
        signal = (
            "consider"
            if gain >= TRIPLE_CAPTAIN_CONSIDER_GAIN_POINTS and confidence == "medium"
            else "watch"
        )
        tc = ChipScenario(
            scenario_id=f"3xc-gw{event}",
            chip=Chip.TRIPLE_CAPTAIN.value,
            event=event,
            signal=signal,
            available=True,
            estimated_gain_points=round(gain, 3),
            baseline_points=round(baseline, 3),
            chip_points=round(baseline + gain, 3),
            confidence=confidence,
            model_scope="captain_marginal",
            reason=(
                "Bedste anfører-margin i den aktuelle prognosehorisont; "
                "startstatus og kampinformation skal genbekræftes ved deadline."
            ),
        )
    else:
        tc = unavailable(Chip.TRIPLE_CAPTAIN, "captain_marginal")

    if bb_candidates:
        gain, event, baseline, _bench_ids, confidence = max(
            bb_candidates,
            key=lambda value: (value[0], -value[1]),
        )
        signal = (
            "consider"
            if gain >= BENCH_BOOST_CONSIDER_GAIN_POINTS and confidence == "medium"
            else "watch"
        )
        bb = ChipScenario(
            scenario_id=f"bboost-gw{event}",
            chip=Chip.BENCH_BOOST.value,
            event=event,
            signal=signal,
            available=True,
            estimated_gain_points=round(gain, 3),
            baseline_points=round(baseline, 3),
            chip_points=round(baseline + gain, 3),
            confidence=confidence,
            model_scope="bench_marginal",
            reason=(
                "Bedste bænk-margin i roadmapet efter fradrag for den normale "
                "forventede indskiftningsværdi."
            ),
        )
    else:
        bb = unavailable(Chip.BENCH_BOOST, "bench_marginal")
    return bb, tc


def _wildcard_scenario(
    players: pd.DataFrame,
    roadmap: BoundedStrategyResult,
    *,
    target_event: int,
    usage: Mapping[Chip, tuple[int, ...]],
    current_squad_ids: Sequence[int],
    bank_tenths: int,
    free_transfers: int,
    eligible_transfer_in_ids: frozenset[int],
    solver_budget_seconds: float,
) -> tuple[ChipScenario, SquadPlanResult | None]:
    available = _available_for_event(Chip.WILDCARD, target_event, usage)
    if not available:
        return (
            ChipScenario(
                scenario_id="wildcard-unavailable",
                chip=Chip.WILDCARD.value,
                event=None,
                signal="hold",
                available=False,
                estimated_gain_points=None,
                baseline_points=None,
                chip_points=None,
                confidence="low",
                model_scope="multiweek_rebuild",
                reason="Wildcard er ikke tilgængeligt til næste deadline.",
            ),
            None,
        )

    by_id = players.set_index("id", drop=False)
    try:
        current_budget = bank_tenths + sum(
            int(by_id.at[int(player_id), "selling_price"])
            for player_id in current_squad_ids
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ChipStrategyError("current selling prices are required for Wildcard") from exc

    try:
        wildcard = optimize_squad_plan(
            players[players["id"].isin(eligible_transfer_in_ids)].copy(),
            horizon=roadmap.horizon,
            gw_weights=roadmap.gw_weights,
            budget_tenths=current_budget,
            solver_time_limit_seconds=solver_budget_seconds,
        )
    except SquadPlanError:
        return (
            ChipScenario(
                scenario_id=f"wildcard-gw{target_event}",
                chip=Chip.WILDCARD.value,
                event=target_event,
                signal="hold",
                available=True,
                estimated_gain_points=None,
                baseline_points=None,
                chip_points=None,
                confidence="low",
                model_scope="multiweek_rebuild",
                reason="Wildcard-modellen kunne ikke bevise en løsning inden for tidsbudgettet.",
            ),
            None,
        )

    baseline_decision = roadmap.best_roadmap.decision_value_points
    wildcard_terminal_ft = min(
        TRANSFERS.max_banked_free_transfers,
        free_transfers + max(0, roadmap.horizon - 1),
    )
    wildcard_decision = wildcard.objective_points + max(0, wildcard_terminal_ft - 1)
    gain = wildcard_decision - baseline_decision
    changes = len(set(current_squad_ids) - set(wildcard.squad_ids))
    signal = (
        "consider"
        if gain >= WILDCARD_CONSIDER_GAIN_POINTS
        and changes >= WILDCARD_MINIMUM_CHANGES
        else "hold"
    )
    return (
        ChipScenario(
            scenario_id=f"wildcard-gw{target_event}",
            chip=Chip.WILDCARD.value,
            event=target_event,
            signal=signal,
            available=True,
            estimated_gain_points=round(gain, 3),
            baseline_points=round(baseline_decision, 3),
            chip_points=round(wildcard_decision, 3),
            confidence="low",
            model_scope="multiweek_rebuild",
            reason=(
                "Én fuld 15-mands genopbygning er sammenlignet med roadmapet over "
                "samme horisont. Senere post-Wildcard-transfers og prisændringer er ikke modelleret."
            ),
            squad_ids=tuple(int(value) for value in wildcard.squad_ids),
            change_count=changes,
            bank_after_tenths=int(wildcard.bank_tenths),
        ),
        wildcard,
    )


def _free_hit_scenario(
    players: pd.DataFrame,
    roadmap: BoundedStrategyResult,
    *,
    target_event: int,
    usage: Mapping[Chip, tuple[int, ...]],
    current_squad_ids: Sequence[int],
    bank_tenths: int,
    eligible_transfer_in_ids: frozenset[int],
    solver_budget_seconds: float,
) -> ChipScenario:
    available_events = [
        event
        for event in range(target_event, target_event + roadmap.horizon)
        if _available_for_event(Chip.FREE_HIT, event, usage)
    ]
    if not available_events:
        return ChipScenario(
            scenario_id="freehit-unavailable",
            chip=Chip.FREE_HIT.value,
            event=None,
            signal="hold",
            available=False,
            estimated_gain_points=None,
            baseline_points=None,
            chip_points=None,
            confidence="low",
            model_scope="confirmed_blank_double_screen",
            reason="Free Hit er ikke tilgængeligt i det beregnede vindue.",
        )

    triggered: list[tuple[int, int]] = []
    for offset, event in enumerate(
        range(target_event, target_event + roadmap.horizon),
        start=1,
    ):
        if event not in available_events:
            continue
        blank_column = f"is_blank_gw{offset}"
        double_column = f"is_dgw_gw{offset}"
        team_signals = players.groupby("team_id")[[blank_column, double_column]].max()
        trigger_count = int(team_signals.any(axis=1).sum())
        if trigger_count > 0:
            triggered.append((trigger_count, event))

    if not triggered:
        return ChipScenario(
            scenario_id="freehit-no-confirmed-trigger",
            chip=Chip.FREE_HIT.value,
            event=None,
            signal="hold",
            available=True,
            estimated_gain_points=None,
            baseline_points=None,
            chip_points=None,
            confidence="low",
            model_scope="confirmed_blank_double_screen",
            reason=(
                "Der er ingen officielt placeret blank eller double gameweek i "
                "prognosevinduet, så Free Hit bør gemmes."
            ),
        )

    _trigger_count, event = max(triggered, key=lambda value: (value[0], -value[1]))
    if solver_budget_seconds < 0.5:
        return ChipScenario(
            scenario_id=f"freehit-gw{event}-screen",
            chip=Chip.FREE_HIT.value,
            event=event,
            signal="watch",
            available=True,
            estimated_gain_points=None,
            baseline_points=None,
            chip_points=None,
            confidence="low",
            model_scope="confirmed_blank_double_screen",
            reason=(
                "En officiel blank/double er registreret, men Free Hit-modellen "
                "havde ikke et sikkert tidsbudget og skal genberegnes."
            ),
        )

    offset = event - target_event + 1
    by_id = players.set_index("id", drop=False)
    current_budget = bank_tenths + sum(
        int(by_id.at[int(player_id), "selling_price"])
        for player_id in current_squad_ids
    )
    one_week = players[players["id"].isin(eligible_transfer_in_ids)].copy()
    for base in ("ep", "appearance_prob", "no_show_prob"):
        one_week[f"{base}_gw1"] = one_week[f"{base}_gw{offset}"]
    try:
        free_hit = optimize_squad_plan(
            one_week,
            horizon=1,
            gw_weights=(1.0,),
            budget_tenths=current_budget,
            solver_time_limit_seconds=solver_budget_seconds,
        )
    except SquadPlanError:
        return ChipScenario(
            scenario_id=f"freehit-gw{event}-screen",
            chip=Chip.FREE_HIT.value,
            event=event,
            signal="watch",
            available=True,
            estimated_gain_points=None,
            baseline_points=None,
            chip_points=None,
            confidence="low",
            model_scope="confirmed_blank_double_screen",
            reason="Free Hit-modellen kunne ikke bevise en løsning inden for tidsbudgettet.",
        )

    roadmap_gameweeks = _roadmap_gameweeks(roadmap)
    roadmap_squad, baseline_gameweek = roadmap_gameweeks[offset]
    baseline = float(baseline_gameweek.projected_points)
    chip_points = float(free_hit.objective_points)
    gain = chip_points - baseline
    # This solve values the temporary one-week team only.  It does not yet
    # re-optimise the permanent transfer roadmap that resumes afterwards, so
    # it must remain a watchpoint rather than an executable chip call.
    signal = "watch"
    return ChipScenario(
        scenario_id=f"freehit-gw{event}",
        chip=Chip.FREE_HIT.value,
        event=event,
        signal=signal,
        available=True,
        estimated_gain_points=round(gain, 3),
        baseline_points=round(baseline, 3),
        chip_points=round(chip_points, 3),
        confidence="low",
        model_scope="single_gameweek_counterfactual",
        reason=(
            "Én midlertidig 15-mands trup er løst mod roadmapets runde. Bank, "
            "permanent trup og bankede transfers gendannes efter Free Hit; den "
            "efterfølgende permanente roadmap er ikke genoptimeret."
        ),
        squad_ids=tuple(int(value) for value in free_hit.squad_ids),
        change_count=len(set(roadmap_squad) - set(free_hit.squad_ids)),
    )


def evaluate_chip_strategy(
    players: pd.DataFrame,
    roadmap: BoundedStrategyResult,
    *,
    target_event: int,
    chip_usage: Mapping[str, Sequence[int]],
    current_squad_ids: Sequence[int],
    bank_tenths: int,
    free_transfers: int,
    eligible_transfer_in_ids: Collection[int] | None = None,
    chip_solver_budget_seconds: float = 4.0,
) -> ChipStrategyResult:
    """Return four personalized chip scenarios without activating a chip."""

    if not isinstance(players, pd.DataFrame) or players.empty:
        raise ChipStrategyError("players must be a non-empty DataFrame")
    if not isinstance(roadmap, BoundedStrategyResult):
        raise ChipStrategyError("roadmap must be a BoundedStrategyResult")
    if not isinstance(target_event, int) or isinstance(target_event, bool):
        raise ChipStrategyError("target_event must be an integer")
    if target_event < 1 or target_event + roadmap.horizon - 1 > 38:
        raise ChipStrategyError("strategy window must stay inside GW1-GW38")
    if len(current_squad_ids) != 15 or len(set(current_squad_ids)) != 15:
        raise ChipStrategyError("current_squad_ids must contain 15 unique ids")
    if not isinstance(bank_tenths, int) or isinstance(bank_tenths, bool) or bank_tenths < 0:
        raise ChipStrategyError("bank_tenths must be a non-negative integer")
    if (
        not isinstance(free_transfers, int)
        or isinstance(free_transfers, bool)
        or not 1 <= free_transfers <= TRANSFERS.max_banked_free_transfers
    ):
        raise ChipStrategyError("free_transfers is outside the legal range")
    solver_budget = float(chip_solver_budget_seconds)
    if not isfinite(solver_budget) or solver_budget <= 0:
        raise ChipStrategyError("chip solver budget must be finite and positive")

    known_ids = frozenset(int(value) for value in players["id"])
    eligible_ids = (
        known_ids
        if eligible_transfer_in_ids is None
        else frozenset(int(value) for value in eligible_transfer_in_ids)
    )
    if not eligible_ids or not eligible_ids.issubset(known_ids):
        raise ChipStrategyError("eligible_transfer_in_ids must be a non-empty pool subset")

    usage = _normalized_usage(chip_usage, before_event=target_event)
    inventory = tuple(
        ChipInventoryEntry(
            chip=chip.value,
            used_events=usage[chip],
            available_for_target=_available_for_event(chip, target_event, usage),
        )
        for chip in CHIP_ORDER
    )
    bench_boost, triple_captain = _lineup_scenarios(
        players,
        roadmap,
        target_event=target_event,
        usage=usage,
    )
    solver_deadline = monotonic() + solver_budget
    free_hit_triggered = any(
        bool(players[f"is_blank_gw{offset}"].any())
        or bool(players[f"is_dgw_gw{offset}"].any())
        for offset in range(1, roadmap.horizon + 1)
    )
    free_hit_reserve = 1.0 if free_hit_triggered else 0.0
    wildcard_budget = max(
        0.5,
        min(solver_budget, solver_deadline - monotonic() - free_hit_reserve),
    )
    wildcard, _wildcard_plan = _wildcard_scenario(
        players,
        roadmap,
        target_event=target_event,
        usage=usage,
        current_squad_ids=current_squad_ids,
        bank_tenths=bank_tenths,
        free_transfers=free_transfers,
        eligible_transfer_in_ids=eligible_ids,
        solver_budget_seconds=wildcard_budget,
    )
    free_hit_budget = max(0.0, solver_deadline - monotonic())
    free_hit = _free_hit_scenario(
        players,
        roadmap,
        target_event=target_event,
        usage=usage,
        current_squad_ids=current_squad_ids,
        bank_tenths=bank_tenths,
        eligible_transfer_in_ids=eligible_ids,
        solver_budget_seconds=free_hit_budget,
    )
    scenarios = (wildcard, free_hit, bench_boost, triple_captain)

    current_considerations = [
        scenario
        for scenario in scenarios
        if scenario.signal == "consider" and scenario.event == target_event
    ]
    if current_considerations:
        selected = max(
            current_considerations,
            key=lambda scenario: (
                scenario.estimated_gain_points
                if scenario.estimated_gain_points is not None
                else float("-inf")
            ),
        )
        recommendation = ChipRecommendation(
            action="consider",
            scenario_id=selected.scenario_id,
            chip=selected.chip,
            event=selected.event,
            reason=(
                "Scenariet er stærkt nok til manuel overvejelse, men chippen er "
                "ikke aktiveret og skal genberegnes efter de seneste holdnyheder."
            ),
        )
    else:
        next_watch = min(
            (
                scenario
                for scenario in scenarios
                if (
                    scenario.event is not None
                    and scenario.event > target_event
                    and scenario.available
                    and scenario.signal in {"watch", "consider"}
                )
            ),
            key=lambda scenario: scenario.event or 99,
            default=None,
        )
        recommendation = ChipRecommendation(
            action="hold",
            scenario_id=None,
            chip=None,
            event=None,
            reason=(
                "Gem chips nu. "
                + (
                    f"Næste beregnede holdepunkt er GW{next_watch.event}."
                    if next_watch is not None
                    else "Der er endnu intet robust chipvindue i horisonten."
                )
            ),
        )

    return ChipStrategyResult(
        horizon=roadmap.horizon,
        target_event=target_event,
        inventory=inventory,
        scenarios=scenarios,
        recommendation=recommendation,
    )


__all__ = [
    "BENCH_BOOST_CONSIDER_GAIN_POINTS",
    "CHIP_ORDER",
    "ChipInventoryEntry",
    "ChipRecommendation",
    "ChipScenario",
    "ChipStrategyError",
    "ChipStrategyResult",
    "FREE_HIT_CONSIDER_GAIN_POINTS",
    "MODEL_SCOPE",
    "TRIPLE_CAPTAIN_CONSIDER_GAIN_POINTS",
    "WILDCARD_CONSIDER_GAIN_POINTS",
    "evaluate_chip_strategy",
]
