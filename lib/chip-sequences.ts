import type { PlannerChipInventoryEntry, PlannerPlayerReference, PlannerStrategyPlan } from "./planner-contract.ts";

export type ChipSequenceId = "normal" | "normal-bboost" | "wildcard" | "wildcard-bboost";
export interface ChipSequenceAction {
  event: number;
  chip: "wildcard" | "bboost" | null;
  transfer_out_ids: number[];
  transfer_in_ids: number[];
  squad_ids: number[];
  bank_after_tenths: number;
  free_transfers_before: number;
  free_transfers_next_gameweek: number;
  hit_points: number;
}
export interface ChipSequence {
  sequence_id: ChipSequenceId;
  label: string;
  weighted_net_points: number;
  gain_vs_normal_points: number;
  total_hit_points: number;
  actions: ChipSequenceAction[];
}
export interface ChipSequenceComparison {
  status: "ready" | "unavailable";
  horizon: number;
  model_scope: "paired_bounded_chip_sequences";
  globally_optimal: false;
  recalculate_each_deadline: true;
  original_roadmap_weighted_net_points: number;
  highest_projected_sequence_id: ChipSequenceId | null;
  sequences: ChipSequence[];
  assumptions: string[];
  reason: string;
}
export interface ChipSequenceContext {
  strategy: PlannerStrategyPlan;
  players: ReadonlyMap<number, PlannerPlayerReference>;
  squad: ReadonlyMap<number, PlannerPlayerReference>;
  selling: ReadonlyMap<number, number>;
  bank: number;
  inventory: PlannerChipInventoryEntry[];
}

const IDS: ChipSequenceId[] = ["normal", "normal-bboost", "wildcard", "wildcard-bboost"];
export const CHIP_SEQUENCE_ASSUMPTIONS = [
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
];

function fail(message: string): never { throw new TypeError(`chip_sequences: ${message}`); }
function exact(value: unknown, keys: string[]): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) fail("Expected an object");
  const row = value as Record<string, unknown>;
  if (Object.keys(row).length !== keys.length || keys.some((key) => !Object.hasOwn(row, key))) fail("Unexpected fields");
  return row;
}
function numeric(value: unknown, min: number, max: number, integer = false): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max || (integer && !Number.isSafeInteger(value))) fail("Invalid number");
  return value;
}
function text(value: unknown, maximum: number): string {
  if (typeof value !== "string" || !value.trim() || value.length > maximum || /[\u0000-\u001f\u007f]/.test(value)) fail("Invalid text");
  return value;
}
function near(actual: number, expected: number): void {
  if (Math.abs(actual - expected) > 0.003) fail("Scores do not reconcile");
}
function ids(value: unknown, maximum: number): number[] {
  if (!Array.isArray(value) || value.length > maximum) fail("Invalid player list");
  const result = value.map((id) => numeric(id, 1, 100_000, true));
  if (new Set(result).size !== result.length) fail("Duplicate player IDs");
  return result;
}
function sameSet(actual: number[], expected: Iterable<number>): void {
  const other = [...expected].sort((a, b) => a - b);
  const sorted = [...actual].sort((a, b) => a - b);
  if (sorted.length !== other.length || sorted.some((value, index) => value !== other[index])) fail("Player transitions do not reconcile");
}
function validSquad(squad: number[], players: ChipSequenceContext["players"]): void {
  if (squad.length !== 15) fail("Squad must contain fifteen players");
  const positions = new Map<string, number>();
  const teams = new Map<string, number>();
  for (const id of squad) {
    const player = players.get(id);
    if (!player) fail("Unknown squad player");
    positions.set(player.position, (positions.get(player.position) ?? 0) + 1);
    teams.set(player.team, (teams.get(player.team) ?? 0) + 1);
  }
  for (const [position, count] of Object.entries({ GKP: 2, DEF: 5, MID: 5, FWD: 3 })) {
    if (positions.get(position) !== count) fail("Illegal squad position quota");
  }
  if ([...teams.values()].some((count) => count > 3)) fail("More than three players from one club");
}

export function parseChipSequences(value: unknown, context: ChipSequenceContext): ChipSequenceComparison {
  const row = exact(value, ["status", "horizon", "model_scope", "globally_optimal", "recalculate_each_deadline", "original_roadmap_weighted_net_points", "highest_projected_sequence_id", "sequences", "assumptions", "reason"]);
  if (row.status !== "ready" && row.status !== "unavailable") fail("Unsupported status");
  if (row.model_scope !== "paired_bounded_chip_sequences" || row.globally_optimal !== false || row.recalculate_each_deadline !== true) fail("Unsupported model scope");
  const horizon = numeric(row.horizon, 6, 10, true);
  if (horizon !== context.strategy.horizon || context.strategy.gameweek_window.length !== horizon || context.strategy.gw_weights.length !== horizon) fail("Horizon does not match the roadmap");
  const originalPoints = numeric(row.original_roadmap_weighted_net_points, -1_000, 10_000);
  near(originalPoints, context.strategy.weighted_projected_points - context.strategy.weighted_hit_cost_points);
  if (!Array.isArray(row.assumptions) || row.assumptions.length !== CHIP_SEQUENCE_ASSUMPTIONS.length
    || row.assumptions.some((assumption, index) => assumption !== CHIP_SEQUENCE_ASSUMPTIONS[index])) fail("Unsupported assumptions");
  if (!Array.isArray(row.sequences) || row.sequences.length > 4) fail("Invalid sequence list");
  const first = context.strategy.steps[0];
  const target = context.strategy.gameweek_window[0];
  const currentHalf = target <= 19 ? 1 : 2;
  const available = (chip: "wildcard" | "bboost", event: number) => {
    const inventory = context.inventory.find((item) => item.chip === chip);
    return !!inventory?.available_for_target && (event <= 19 ? 1 : 2) === currentHalf
      && !inventory.used_events.some((used) => (used <= 19 ? 1 : 2) === currentHalf);
  };
  const sequences = row.sequences.map((value): ChipSequence => {
    const sequence = exact(value, ["sequence_id", "label", "weighted_net_points", "gain_vs_normal_points", "total_hit_points", "actions"]);
    if (!IDS.includes(sequence.sequence_id as ChipSequenceId)) fail("Unknown sequence");
    const id = sequence.sequence_id as ChipSequenceId;
    if (!Array.isArray(sequence.actions) || sequence.actions.length !== horizon) fail("Incomplete sequence");
    let previous = new Set(context.squad.keys());
    validSquad([...previous], context.players);
    let bank = numeric(context.bank, 0, 10_000, true);
    let ft = first.free_transfers_before;
    const selling = new Map(context.selling);
    let wildcards = 0;
    let boosts = 0;
    const actions = sequence.actions.map((value, index): ChipSequenceAction => {
      const action = exact(value, ["event", "chip", "transfer_out_ids", "transfer_in_ids", "squad_ids", "bank_after_tenths", "free_transfers_before", "free_transfers_next_gameweek", "hit_points"]);
      const event = numeric(action.event, 1, 38, true);
      if (event !== context.strategy.gameweek_window[index]) fail("Wrong sequence event");
      if (action.chip !== null && action.chip !== "wildcard" && action.chip !== "bboost") fail("Unsupported chip");
      const chip = action.chip;
      if (chip !== null && !available(chip, event)) fail("Unavailable chip or expired chip half");
      if (chip === "wildcard") { wildcards++; if (index !== 0 || !id.startsWith("wildcard")) fail("Wildcard must be the first action"); }
      if (chip === "bboost") { boosts++; if (!id.endsWith("-bboost")) fail("Unexpected Bench Boost"); }
      const outgoing = ids(action.transfer_out_ids, index === 0 ? 15 : index < 4 ? 2 : 0);
      const incoming = ids(action.transfer_in_ids, outgoing.length);
      if (outgoing.length !== incoming.length || outgoing.some((value) => incoming.includes(value))) fail("Invalid transfers");
      const revised = new Set(previous);
      for (let move = 0; move < outgoing.length; move++) {
        const out = outgoing[move], next = incoming[move];
        const outgoingPlayer = context.players.get(out), incomingPlayer = context.players.get(next);
        if (!previous.has(out) || previous.has(next) || !outgoingPlayer || !incomingPlayer
          || outgoingPlayer.position !== incomingPlayer.position) fail("Transfer does not match the owned squad and position");
        const sale = selling.get(out);
        if (sale === undefined) fail("Missing selling price");
        bank += numeric(sale, 0, 500, true) - incomingPlayer.price_tenths;
        revised.delete(out); revised.add(next);
        selling.delete(out); selling.set(next, incomingPlayer.price_tenths);
      }
      const squad = ids(action.squad_ids, 15);
      sameSet(squad, revised); validSquad(squad, context.players);
      const reportedBank = numeric(action.bank_after_tenths, 0, 10_000, true);
      if (reportedBank !== bank) fail("Transfer bank does not reconcile with actual selling values");
      const ftBefore = numeric(action.free_transfers_before, 0, 5, true);
      const ftAfter = numeric(action.free_transfers_next_gameweek, 0, 5, true);
      const hit = numeric(action.hit_points, 0, 80, true);
      if (ftBefore !== ft || ftAfter !== (chip === "wildcard" ? ft : Math.min(5, Math.max(0, ft - outgoing.length) + 1))
        || hit !== (chip === "wildcard" ? 0 : 4 * Math.max(0, outgoing.length - ft))) fail("Invalid free transfers or hit cost");
      if (index === 0 && id.startsWith("normal")) {
        sameSet(squad, first.squad_ids);
        sameSet(outgoing, first.transfers.map((move) => move.out_id));
        sameSet(incoming, first.transfers.map((move) => move.in_id));
        if (reportedBank !== first.bank_after_tenths || ftAfter !== first.free_transfers_next_gameweek || hit !== first.hit_points) fail("Normal first action differs from the roadmap");
      }
      previous = revised; ft = ftAfter;
      return { event, chip, transfer_out_ids: outgoing, transfer_in_ids: incoming, squad_ids: squad,
        bank_after_tenths: reportedBank, free_transfers_before: ftBefore, free_transfers_next_gameweek: ftAfter, hit_points: hit };
    });
    if (wildcards !== (id.startsWith("wildcard") ? 1 : 0) || boosts !== (id.endsWith("-bboost") ? 1 : 0)) fail("Chip sequence does not match its identity");
    const totalHits = numeric(sequence.total_hit_points, 0, 200, true);
    if (totalHits !== actions.reduce((sum, action) => sum + action.hit_points, 0)) fail("Total hit cost does not reconcile");
    return { sequence_id: id, label: text(sequence.label, 160), weighted_net_points: numeric(sequence.weighted_net_points, -1_000, 10_000),
      gain_vs_normal_points: numeric(sequence.gain_vs_normal_points, -10_000, 10_000), total_hit_points: totalHits, actions };
  });
  if (new Set(sequences.map((sequence) => sequence.sequence_id)).size !== sequences.length) fail("Duplicate sequences");
  let highest: ChipSequenceId | null = null;
  if (row.status === "unavailable") {
    if (sequences.length !== 0 || row.highest_projected_sequence_id !== null) fail("Unavailable comparison contains results");
  } else {
    const normal = sequences.find((sequence) => sequence.sequence_id === "normal");
    if (!normal || !IDS.includes(row.highest_projected_sequence_id as ChipSequenceId)) fail("Missing normal baseline or highest sequence");
    highest = row.highest_projected_sequence_id as ChipSequenceId;
    const winner = sequences.find((sequence) => sequence.sequence_id === highest);
    if (!winner || sequences.some((sequence) => sequence.weighted_net_points > winner.weighted_net_points + 0.003)) fail("Incorrect highest projected sequence");
    for (const sequence of sequences) {
      near(sequence.gain_vs_normal_points, sequence.weighted_net_points - normal.weighted_net_points);
      if (!sequence.sequence_id.endsWith("-bboost")) continue;
      const base = sequences.find((candidate) => candidate.sequence_id === sequence.sequence_id.replace("-bboost", ""));
      if (!base || sequence.weighted_net_points < base.weighted_net_points) fail("Bench Boost reduced the modelled score");
      for (let i = 0; i < horizon; i++) {
        const a = sequence.actions[i], b = base.actions[i];
        if (a.chip === "bboost" && b.chip !== null) fail("Two chips on one deadline");
        if (JSON.stringify({ ...a, chip: b.chip }) !== JSON.stringify(b)) fail("Bench Boost changed the underlying transfer path");
      }
    }
  }
  return { status: row.status, horizon, model_scope: "paired_bounded_chip_sequences", globally_optimal: false,
    recalculate_each_deadline: true, original_roadmap_weighted_net_points: originalPoints, highest_projected_sequence_id: highest,
    sequences, assumptions: [...CHIP_SEQUENCE_ASSUMPTIONS], reason: text(row.reason, 800) };
}
