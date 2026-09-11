"""Paired, bounded chip sequences with the same continuation policy.

This is a transparent counterfactual, not another season optimizer.  One
already-solved Wildcard squad and the normal first action receive the same
deterministic continuation search: up to two greedy transfers on deadlines
two to four, screened to 24 position-preserving candidates per transfer.
No extra solver processes or network calls are used.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
from time import monotonic
from typing import Collection, Mapping, Sequence

import pandas as pd

from ..domain.rules import (
    FORMATIONS,
    Chip,
    can_play_chip,
    chip_window,
    free_transfers_next_gameweek,
    selling_price_tenths,
    transfer_points_cost,
)
from .rolling_transfers import DEFAULT_BENCH_WEIGHTS
from .squad_plan import SquadPlanResult
from .strategy_planner import BoundedStrategyResult


MODEL_SCOPE = "paired_bounded_chip_sequences"
CONTINUATION_SHORTLIST = 24
MAX_CONTINUATION_TRANSFERS = 2
ASSUMPTIONS = (
    "same_horizon_weights_and_continuation_policy",
    "fixed_current_prices_with_actual_selling_basis",
    "normal_first_action_from_transfer_roadmap",
    "one_wildcard_squad_not_jointly_optimised_for_bench_boost",
    "at_most_two_greedy_transfers_at_deadlines_two_to_four",
    "continuation_candidates_from_existing_roadmap_and_wildcard_squads",
    "top_24_individual_gain_candidates_per_transfer",
    "one_weighted_point_hurdle_for_each_provisional_transfer",
    "no_transfers_after_deadline_four",
    "one_bench_boost_in_the_current_chip_half_only",
    "future_chip_option_value_not_priced",
    "recalculate_at_every_deadline",
)


class SequenceBudgetExceeded(RuntimeError):
    """The paired comparison must be dropped if either side runs out of time."""


@dataclass(frozen=True)
class _Action:
    event: int
    chip: str | None
    transfer_out_ids: tuple[int, ...]
    transfer_in_ids: tuple[int, ...]
    squad_ids: tuple[int, ...]
    bank_after_tenths: int
    free_transfers_before: int
    free_transfers_next_gameweek: int
    hit_points: int

    def as_dict(self) -> dict[str, object]:
        return {
            "event": self.event,
            "chip": self.chip,
            "transfer_out_ids": list(self.transfer_out_ids),
            "transfer_in_ids": list(self.transfer_in_ids),
            "squad_ids": list(self.squad_ids),
            "bank_after_tenths": self.bank_after_tenths,
            "free_transfers_before": self.free_transfers_before,
            "free_transfers_next_gameweek": self.free_transfers_next_gameweek,
            "hit_points": self.hit_points,
        }


@dataclass(frozen=True)
class _Path:
    sequence_id: str
    label: str
    weighted_net_points: float
    actions: tuple[_Action, ...]

    def as_dict(self, normal_points: float) -> dict[str, object]:
        return {
            "sequence_id": self.sequence_id,
            "label": self.label,
            "weighted_net_points": round(self.weighted_net_points, 3),
            "gain_vs_normal_points": round(self.weighted_net_points - normal_points, 3),
            "total_hit_points": sum(action.hit_points for action in self.actions),
            "actions": [action.as_dict() for action in self.actions],
        }


def _chip_available(chip: Chip, event: int, usage: Mapping[Chip, Sequence[int]]) -> bool:
    used_halves = {
        window.half
        for used_event in usage.get(chip, ())
        for window in [chip_window(chip, used_event)]
        if window is not None
    }
    return can_play_chip(chip, event, used_halves=used_halves)


def evaluate_chip_sequences(
    players: pd.DataFrame,
    roadmap: BoundedStrategyResult,
    wildcard_plan: SquadPlanResult | None,
    *,
    target_event: int,
    usage: Mapping[Chip, Sequence[int]],
    current_squad_ids: Sequence[int],
    bank_tenths: int,
    free_transfers: int,
    eligible_transfer_in_ids: Collection[int],
    deadline: float,
) -> dict[str, object]:
    """Compare matched no-chip, BB, WC and WC-then-BB paths if time permits.

The Wildcard is used only at the current deadline.  Bench Boost candidates
cannot cross the current chip-set expiry.  Scores exclude terminal FT bonuses
and never treat unused chips as worthless: that unpriced option is explicit.
    """

    original = roadmap.best_roadmap
    result: dict[str, object] = {
        "status": "unavailable",
        "horizon": roadmap.horizon,
        "model_scope": MODEL_SCOPE,
        "globally_optimal": False,
        "recalculate_each_deadline": True,
        "original_roadmap_weighted_net_points": round(
            original.weighted_projected_points - original.weighted_hit_cost_points, 3
        ),
        "highest_projected_sequence_id": None,
        "sequences": [],
        "assumptions": list(ASSUMPTIONS),
        "reason": "Den parrede chipanalyse mangler et sikkert tidsbudget.",
    }
    if monotonic() >= deadline:
        return result

    weights = roadmap.gw_weights
    horizon = roadmap.horizon
    rows = {int(row["id"]): row for row in players.to_dict("records")}
    initial_ids = frozenset(int(value) for value in current_squad_ids)
    reference_ids = {
        int(value) for step in roadmap.best_roadmap.steps for value in step.squad_ids
    }
    if wildcard_plan is not None:
        reference_ids.update(int(value) for value in wildcard_plan.squad_ids)
    eligible = frozenset(int(value) for value in eligible_transfer_in_ids) & reference_ids
    positions = {player_id: str(row["pos"]) for player_id, row in rows.items()}
    teams = {player_id: int(row["team_id"]) for player_id, row in rows.items()}
    costs = {player_id: int(row["now_cost"]) for player_id, row in rows.items()}
    ep = {
        offset: {player_id: float(row[f"ep_gw{offset}"]) for player_id, row in rows.items()}
        for offset in range(1, horizon + 1)
    }
    reserve_weights = DEFAULT_BENCH_WEIGHTS
    residual = {
        offset: {
            player_id: ep[offset][player_id]
            * (1.0 - float(row[f"no_show_prob_gw{offset}"]))
            * (reserve_weights[3] if positions[player_id] == "GKP" else sum(reserve_weights[:3]) / 3)
            for player_id, row in rows.items()
        }
        for offset in range(1, horizon + 1)
    }
    formations = [
        {position.value: count for position, count in formation.positions.items()}
        for formation in FORMATIONS
    ]

    def check_budget() -> None:
        if monotonic() >= deadline:
            raise SequenceBudgetExceeded

    @lru_cache(maxsize=12000)
    def lineup(squad: tuple[int, ...], offset: int) -> tuple[float, tuple[int, ...]]:
        """Exact XI/captain score under the roadmap's linear reserve model."""
        check_budget()
        groups = {
            position: sorted(
                (player_id for player_id in squad if positions[player_id] == position),
                key=lambda player_id: (-(ep[offset][player_id] - residual[offset][player_id]), player_id),
            )
            for position in ("GKP", "DEF", "MID", "FWD")
        }
        base = sum(residual[offset][player_id] for player_id in squad)
        best_score = float("-inf")
        best_starters: tuple[int, ...] = ()
        adjusted = {value: ep[offset][value] - residual[offset][value] for value in squad}
        for counts in formations:
            starters = tuple(value for position, count in counts.items() for value in groups[position][:count])
            selected = frozenset(starters)
            base_score = base + sum(adjusted[value] for value in starters)
            for captain in squad:
                score = base_score + ep[offset][captain]
                captain_starters = starters
                if captain not in selected:
                    displaced = groups[positions[captain]][counts[positions[captain]] - 1]
                    score += adjusted[captain] - adjusted[displaced]
                    captain_starters = tuple(value for value in starters if value != displaced) + (captain,)
                ids = tuple(sorted(captain_starters))
                if score > best_score + 1e-9 or (abs(score - best_score) < 1e-9 and ids < best_starters):
                    best_score, best_starters = score, ids
        return best_score, best_starters

    def remaining_score(squad: frozenset[int], offset: int) -> float:
        return sum(
            weights[later - 1] * lineup(tuple(sorted(squad)), later)[0]
            for later in range(offset, horizon + 1)
        )

    def continue_path(
        *,
        sequence_id: str,
        label: str,
        first_squad: frozenset[int],
        first_bank: int,
        first_chip: Chip | None,
        first_out: tuple[int, ...],
        first_in: tuple[int, ...],
    ) -> _Path:
        squad = first_squad
        bank = first_bank
        purchases = {
            player_id: int(rows[player_id]["purchase_price"]) if player_id in initial_ids and player_id not in first_in else costs[player_id]
            for player_id in squad
        }
        ft = free_transfers
        first_hit = transfer_points_cost(len(first_out), ft, active_chip=first_chip)
        next_ft = free_transfers_next_gameweek(ft, len(first_out), active_chip=first_chip)
        actions = [
            _Action(target_event, first_chip.value if first_chip else None, first_out, first_in,
                    tuple(sorted(squad)), bank, ft, next_ft, first_hit)
        ]
        total = weights[0] * (lineup(tuple(sorted(squad)), 1)[0] - first_hit)
        ft = next_ft
        for offset in range(2, horizon + 1):
            check_budget()
            outs: list[int] = []
            ins: list[int] = []
            # The same greedy search is applied to both paths.  Candidate
            # screening is deliberately independent of chip/manager identity.
            if offset <= 4:
                for _ in range(MAX_CONTINUATION_TRANSFERS):
                    current_score = remaining_score(squad, offset)
                    candidate_moves = []
                    for outgoing in sorted(squad - set(ins)):
                        sale = selling_price_tenths(purchases[outgoing], costs[outgoing])
                        for incoming in sorted(eligible - squad - set(outs)):
                            if positions[incoming] != positions[outgoing] or costs[incoming] > bank + sale:
                                continue
                            revised = (squad - {outgoing}) | {incoming}
                            if max(Counter(teams[value] for value in revised).values()) > 3:
                                continue
                            screen_gain = sum(
                                weights[later - 1] * (ep[later][incoming] - ep[later][outgoing])
                                for later in range(offset, horizon + 1)
                            )
                            candidate_moves.append((screen_gain, outgoing, incoming, revised, bank + sale - costs[incoming]))
                    candidate_moves.sort(key=lambda move: (-move[0], move[1], move[2]))
                    best = None
                    best_gain = 0.0
                    incremental_hit = transfer_points_cost(len(outs) + 1, ft) - transfer_points_cost(len(outs), ft)
                    for _, outgoing, incoming, revised, next_bank in candidate_moves[:CONTINUATION_SHORTLIST]:
                        gain = remaining_score(frozenset(revised), offset) - current_score - weights[offset - 1] * incremental_hit
                        # A one-point hurdle protects optionality; this is a
                        # disclosed heuristic, not a calibrated FT valuation.
                        if gain > 1.0 and gain > best_gain + 1e-9:
                            best_gain = gain
                            best = (outgoing, incoming, frozenset(revised), next_bank)
                    if best is None:
                        break
                    outgoing, incoming, squad, bank = best
                    purchases.pop(outgoing)
                    purchases[incoming] = costs[incoming]
                    outs.append(outgoing)
                    ins.append(incoming)
            hit = transfer_points_cost(len(outs), ft)
            next_ft = free_transfers_next_gameweek(ft, len(outs))
            actions.append(_Action(target_event + offset - 1, None, tuple(outs), tuple(ins),
                                   tuple(sorted(squad)), bank, ft, next_ft, hit))
            total += weights[offset - 1] * (lineup(tuple(sorted(squad)), offset)[0] - hit)
            ft = next_ft
        return _Path(sequence_id, label, total, tuple(actions))

    def add_bench_boost(path: _Path) -> _Path | None:
        current_window = chip_window(Chip.BENCH_BOOST, target_event)
        candidates: list[tuple[float, int]] = []
        for offset, action in enumerate(path.actions, start=1):
            window = chip_window(Chip.BENCH_BOOST, action.event)
            if (action.chip is not None or window is None or current_window is None
                    or window.half != current_window.half
                    or not _chip_available(Chip.BENCH_BOOST, action.event, usage)):
                continue
            _, starters = lineup(action.squad_ids, offset)
            bench = set(action.squad_ids) - set(starters)
            gain = sum(ep[offset][value] - residual[offset][value] for value in bench)
            candidates.append((weights[offset - 1] * gain, offset))
        if not candidates:
            return None
        gain, offset = max(candidates, key=lambda candidate: (candidate[0], -candidate[1]))
        # Do not propose using a chip to reduce the modelled score.
        if gain <= 0:
            return None
        actions = list(path.actions)
        actions[offset - 1] = replace(actions[offset - 1], chip=Chip.BENCH_BOOST.value)
        return _Path(f"{path.sequence_id}-bboost", f"{path.label} + Bench Boost GW{target_event + offset - 1}",
                     path.weighted_net_points + gain, tuple(actions))

    try:
        first = original.steps[0]
        normal = continue_path(
            sequence_id="normal", label="Normale transfers",
            first_squad=frozenset(first.squad_ids), first_bank=first.bank_after_tenths,
            first_chip=None, first_out=tuple(move.out_id for move in first.transfers),
            first_in=tuple(move.in_id for move in first.transfers),
        )
        paths = [normal]
        normal_bb = add_bench_boost(normal)
        if normal_bb is not None:
            paths.append(normal_bb)
        if wildcard_plan is not None and _chip_available(Chip.WILDCARD, target_event, usage):
            wc_ids = frozenset(int(value) for value in wildcard_plan.squad_ids)
            sold = initial_ids - wc_ids
            bought = wc_ids - initial_ids
            wc_bank = bank_tenths + sum(int(rows[value]["selling_price"]) for value in sold) - sum(costs[value] for value in bought)
            if wc_bank < 0:
                raise ValueError("Wildcard sequence exceeds actual selling-value budget")
            wildcard = continue_path(
                sequence_id="wildcard", label=f"Wildcard GW{target_event}, derefter transfers",
                first_squad=wc_ids, first_bank=wc_bank, first_chip=Chip.WILDCARD,
                first_out=tuple(sorted(sold, key=lambda value: (positions[value], value))),
                first_in=tuple(sorted(bought, key=lambda value: (positions[value], value))),
            )
            paths.append(wildcard)
            wildcard_bb = add_bench_boost(wildcard)
            if wildcard_bb is not None:
                paths.append(wildcard_bb)
        check_budget()
    except SequenceBudgetExceeded:
        # Never display only the side that happened to finish first.
        return result

    highest = max(paths, key=lambda path: path.weighted_net_points)
    has_wildcard = any(path.sequence_id == "wildcard" for path in paths)
    result.update({
        "status": "ready",
        "highest_projected_sequence_id": highest.sequence_id,
        "sequences": [path.as_dict(normal.weighted_net_points) for path in paths],
        "reason": (
            (
                "Samme afgrænsede transferpolitik er kørt på begge starttrupper. "
                if has_wildcard
                else "Kun normale transfers og eventuel Bench Boost kunne sammenlignes; "
                     "der foreligger ingen tilgængelig, løst Wildcard-trup. "
            )
            +
            "Den højeste prognose er ikke automatisk den bedste chipbeslutning: "
            "værdien af at gemme en chip til senere er ikke prissat. "
            "Wildcardets 15 spillere er ikke særskilt optimeret til Bench Boost."
        ),
    })
    return result
