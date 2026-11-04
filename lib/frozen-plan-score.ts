import type { DecisionHistoryEntry, DecisionPlan } from "./decision-history.ts";
import type { ActualDecisionResult, OfficialPlayerScore } from "./decision-results.ts";
import type { PlayerPosition } from "./planner-contract.ts";

export interface FrozenPlanScore {
  available: boolean;
  gross_points: number | null;
  planned_hit_points: number | null;
  net_points: number | null;
  scoring_ids: number[];
  effective_captain_id: number | null;
  automatic_subs: { out_id: number; in_id: number }[];
  reason: string | null;
}
function unavailable(reason: string): FrozenPlanScore {
  return { available: false, gross_points: null, planned_hit_points: null, net_points: null,
    scoring_ids: [], effective_captain_id: null, automatic_subs: [], reason };
}
function legalFormation(ids: number[], scores: Map<number, OfficialPlayerScore>): boolean {
  const counts: Record<PlayerPosition, number> = { GKP: 0, DEF: 0, MID: 0, FWD: 0 };
  for (const id of ids) counts[scores.get(id)!.position]++;
  return ids.length === 11 && counts.GKP === 1 && counts.DEF >= 3 && counts.DEF <= 5
    && counts.MID >= 2 && counts.MID <= 5 && counts.FWD >= 1 && counts.FWD <= 3;
}

/**
 * Score one exactly preserved target-GW choice, without hindsight substitutions.
 * FPL's ordered bench and C/VC rules apply to whole-GW minutes, including DGWs:
 * https://fantasy.premierleague.com/help/faqs
 * This is not a multiweek transfer policy or a causal estimate of app impact.
 */
export function replayFrozenDecisionPlan(
  entry: DecisionHistoryEntry, plan: DecisionPlan | null, result: ActualDecisionResult,
): FrozenPlanScore {
  if (!result.finalized || !result.player_scores) return unavailable("FPL har endnu ikke færdigbehandlet alle spillerpoint.");
  if (entry.origin !== "recorded_v2" || entry.manager_id !== result.manager_id || entry.season !== result.season
    || entry.target_event !== result.target_event || Date.parse(entry.saved_at) >= Date.parse(result.target_deadline)) {
    return unavailable("Snapshot kan ikke knyttes til denne runde før deadline.");
  }
  const lineup = plan?.gameweeks.find(g => g.gameweek === entry.target_event);
  if (!plan || !lineup || !lineup.bench_ids || lineup.vice_captain_id === null || !entry.context) {
    return unavailable("Den fulde opstilling, viceanfører eller bænkrækkefølge blev ikke gemt.");
  }
  const chip = entry.context.planned_chip;
  if (!entry.context.no_active_chip_confirmed && chip === null) return unavailable("Det planlagte chipvalg er ukendt.");
  if (chip === "wildcard" || chip === "freehit") return unavailable("Gennemspilningen understøtter ikke et Wildcard- eller Free Hit-forløb.");
  const scores = new Map(result.player_scores.map(p => [p.id, p]));
  if (plan.squad_ids.length !== 15 || new Set(plan.squad_ids).size !== 15
    || plan.squad_ids.some(id => !scores.has(id))) return unavailable("Der mangler officielle point for mindst én spiller i den gemte trup.");
  if (plan.players.some(p => scores.get(p.id)?.position !== p.position)) return unavailable("Spillerpositioner kan ikke afstemmes med snapshot.");
  const allLineupIds = [...lineup.starting_ids, ...lineup.bench_ids];
  if (allLineupIds.length !== 15 || new Set(allLineupIds).size !== 15
    || allLineupIds.some(id => !plan.squad_ids.includes(id))
    || !lineup.starting_ids.includes(lineup.captain_id) || !lineup.starting_ids.includes(lineup.vice_captain_id)
    || lineup.captain_id === lineup.vice_captain_id || !legalFormation(lineup.starting_ids, scores)) {
    return unavailable("Den gemte opstilling opfylder ikke formationsreglerne.");
  }
  // A disciplinary event with no recorded minute has ambiguous appearance
  // semantics. Leave it unscored rather than inventing an autosub outcome.
  if (plan.squad_ids.some(id => scores.get(id)!.minutes === 0 && scores.get(id)!.points !== 0)) {
    return unavailable("FPL har registreret point uden spilletid; denne særtilfælde skal kontrolleres i det officielle resultat.");
  }
  const played = (id: number) => scores.get(id)!.minutes > 0;
  let starters = [...lineup.starting_ids];
  const substitutions: { out_id: number; in_id: number }[] = [];
  if (chip !== "bboost") {
    const startingKeeper = starters.find(id => scores.get(id)!.position === "GKP")!;
    const reserveKeepers = lineup.bench_ids.filter(id => scores.get(id)!.position === "GKP");
    if (reserveKeepers.length !== 1) return unavailable("Der skal være én reservekeeper i den gemte trup.");
    if (!played(startingKeeper) && played(reserveKeepers[0])) {
      starters = starters.map(id => id === startingKeeper ? reserveKeepers[0] : id);
      substitutions.push({ out_id: startingKeeper, in_id: reserveKeepers[0] });
    }
    const outfieldBench = lineup.bench_ids.filter(id => scores.get(id)!.position !== "GKP");
    const missing = starters.filter(id => !played(id) && scores.get(id)!.position !== "GKP");
    let best = { priority: -1, ids: starters, subs: [] as { out_id: number; in_id: number }[] };
    // At most three reserves and ten absent starters. Priority depends only on
    // the recorded bench order, never on players' realized points.
    function visit(index: number, current: number[], subs: { out_id: number; in_id: number }[], priority: number) {
      if (index === outfieldBench.length) {
        if (priority > best.priority) best = { priority, ids: current, subs };
        return;
      }
      const incoming = outfieldBench[index];
      if (played(incoming)) for (const outgoing of missing) {
        if (!current.includes(outgoing)) continue;
        const next = current.map(id => id === outgoing ? incoming : id);
        if (legalFormation(next, scores)) visit(index + 1, next, [...subs, { out_id: outgoing, in_id: incoming }], priority + 2 ** (2 - index));
      }
      visit(index + 1, current, subs, priority);
    }
    visit(0, starters, [], 0);
    starters = best.ids;
    substitutions.push(...best.subs);
  }
  const scoringIds = chip === "bboost" ? plan.squad_ids : starters;
  const captain = played(lineup.captain_id) ? lineup.captain_id : played(lineup.vice_captain_id) ? lineup.vice_captain_id : null;
  const base = scoringIds.reduce((sum, id) => sum + scores.get(id)!.points, 0);
  const captainBonus = captain === null ? 0 : scores.get(captain)!.points * (chip === "3xc" ? 2 : 1);
  const gross = base + captainBonus;
  return { available: true, gross_points: gross, planned_hit_points: plan.action.hit_points,
    net_points: gross - plan.action.hit_points, scoring_ids: scoringIds, effective_captain_id: captain,
    automatic_subs: substitutions, reason: null };
}
