import type { AiReviewResponse } from "./ai-review-contract.ts";
import type { ChipName, ManualManagerState, PlannerAction, PlannerPayload, PlayerPosition } from "./planner-contract.ts";
import { parseDecisionHistoryEntry as parseLegacyEntry, type DecisionHistoryEntry as LegacyEntry } from "./decision-history-legacy.ts";

export type { DecisionHistoryAction, DecisionHistoryTransfer, StorageLike } from "./decision-history-legacy.ts";
import type { DecisionHistoryAction, StorageLike } from "./decision-history-legacy.ts";

export const DECISION_HISTORY_SCHEMA_VERSION = "fpl-decision-history-v2" as const;
export const DECISION_HISTORY_STORAGE_KEY = "fpl-decision-history-v2";
export const LEGACY_DECISION_HISTORY_STORAGE_KEY = "fpl-decision-history-v1";
export const MAX_DECISION_HISTORY_ENTRIES = 200;
export const MAX_DECISION_HISTORY_BYTES = 2 * 1024 * 1024;
export type DecisionSelectionKind = "best" | "alternative" | "wait";

export interface DecisionLineup {
  gameweek: number;
  starting_ids: number[];
  captain_id: number;
  vice_captain_id: number | null;
  /** Ordered outfield substitutes and reserve goalkeeper, in the supplied order. */
  bench_ids: number[] | null;
  formation: string;
  projected_points: number;
}
export interface DecisionPlayer {
  id: number; name: string; position: PlayerPosition; team: string;
  price_tenths: number; purchase_price_tenths: number | null; selling_price_tenths: number | null;
}
export interface DecisionPlan {
  action: DecisionHistoryAction;
  squad_ids: number[];
  players: DecisionPlayer[];
  gameweeks: DecisionLineup[];
}
export type DecisionLineupDetails = Pick<DecisionLineup, "starting_ids" | "captain_id" | "vice_captain_id" | "bench_ids">;
export interface DecisionHistoryEntry extends LegacyEntry {
  snapshot_id: string;
  origin: "recorded_v2" | "migrated_v1";
  manager_id: number | null;
  /** Derived from the deadline, and checked against the official event season later. */
  season: string;
  state_fingerprint: string | null;
  selection_source: "model" | "ai" | "user" | "unknown";
  plans: { best: DecisionPlan | null; roll: DecisionPlan | null; alternatives: DecisionPlan[] } | null;
  context: {
    squad: DecisionPlayer[];
    squad_value_tenths: number;
    budget_tenths: number;
    chips: ManualManagerState["chips"] | null;
    chip_usage: ManualManagerState["chip_usage"] | null;
    no_active_chip_confirmed: boolean;
    planned_chip: ChipName | null;
  } | null;
}
export interface DecisionHistory {
  schema_version: typeof DECISION_HISTORY_SCHEMA_VERSION;
  entries: DecisionHistoryEntry[];
  /** A read error is displayed and must never be converted into an overwrite. */
  storage_error?: string;
}
export interface DecisionHistoryBuildInput {
  saved_at?: string; target_deadline: string; state_observed_at: string; recommendation_generated_at: string;
  forecast: LegacyEntry["forecast"];
  planner: PlannerPayload; ai_review: AiReviewResponse | null;
  manager_state?: ManualManagerState;
  selection?: LegacyEntry["selection"];
  planned_chip?: ChipName | null;
  lineup_details?: {
    best?: DecisionLineupDetails;
    alternatives?: Array<DecisionLineupDetails | undefined>;
  };
}
type Obj = Record<string, unknown>;
const legacyKeys = ["saved_at", "target_event", "target_deadline", "source_event", "state_observed_at",
  "recommendation_generated_at", "forecast", "confirmed", "selection", "action", "lineup", "ai"];
const extensionKeys = ["snapshot_id", "origin", "manager_id", "season", "state_fingerprint", "selection_source", "plans", "context"];
function obj(v: unknown, where: string): Obj {
  if (!v || typeof v !== "object" || Array.isArray(v)) throw new TypeError(`${where} must be an object`);
  return v as Obj;
}
function exact(v: unknown, keys: string[], where: string): Obj {
  const r = obj(v, where);
  if (Object.keys(r).length !== keys.length || keys.some(k => !(k in r))) throw new TypeError(`${where} has unexpected or missing fields`);
  return r;
}
function integer(v: unknown, min: number, max: number, where: string): number {
  if (!Number.isSafeInteger(v) || (v as number) < min || (v as number) > max) throw new TypeError(`Invalid ${where}`);
  return v as number;
}
function finite(v: unknown, where: string): number {
  if (typeof v !== "number" || !Number.isFinite(v) || Math.abs(v) > 10000) throw new TypeError(`Invalid ${where}`);
  return v;
}
function text(v: unknown, max: number): string {
  if (typeof v !== "string" || !v.trim() || v.length > max || /[\u0000-\u001f\u007f]/.test(v)) throw new TypeError("Invalid text");
  return v;
}
function ids(v: unknown, count: number, where: string): number[] {
  if (!Array.isArray(v) || v.length !== count) throw new TypeError(`Invalid ${where}`);
  const values = v.map(x => integer(x, 1, Number.MAX_SAFE_INTEGER, where));
  if (new Set(values).size !== count) throw new TypeError(`Duplicate ${where}`);
  return values;
}
export function decisionSeason(deadline: string): string {
  const date = new Date(deadline);
  if (!Number.isFinite(date.getTime())) throw new TypeError("Invalid deadline");
  const start = date.getUTCFullYear() - (date.getUTCMonth() < 6 ? 1 : 0);
  return `${start}/${String(start + 1).slice(-2)}`;
}
function player(v: unknown): DecisionPlayer {
  const p = exact(v, ["id", "name", "position", "team", "price_tenths", "purchase_price_tenths", "selling_price_tenths"], "player");
  if (!["GKP", "DEF", "MID", "FWD"].includes(String(p.position))) throw new TypeError("Invalid position");
  return { id: integer(p.id, 1, Number.MAX_SAFE_INTEGER, "player id"), name: text(p.name, 80),
    position: p.position as PlayerPosition, team: text(p.team, 30), price_tenths: integer(p.price_tenths, 1, 1000, "price"),
    purchase_price_tenths: p.purchase_price_tenths === null ? null : integer(p.purchase_price_tenths, 1, 1000, "purchase price"),
    selling_price_tenths: p.selling_price_tenths === null ? null : integer(p.selling_price_tenths, 1, 1000, "selling price") };
}
function parsePlan(value: unknown, target: number, base: LegacyEntry): DecisionPlan {
  const p = exact(value, ["action", "squad_ids", "players", "gameweeks"], "plan");
  const squad = ids(p.squad_ids, 15, "squad");
  if (!Array.isArray(p.players) || p.players.length !== 15) throw new TypeError("Invalid plan players");
  const players = p.players.map(player);
  if (new Set(players.map(p => p.id)).size !== 15 || players.some(p => !squad.includes(p.id))) throw new TypeError("Plan players do not match squad");
  if (!Array.isArray(p.gameweeks) || !p.gameweeks.length || p.gameweeks.length > 8) throw new TypeError("Invalid plan gameweeks");
  const gameweeks = p.gameweeks.map((value, i) => {
    const g = exact(value, ["gameweek", "starting_ids", "captain_id", "vice_captain_id", "bench_ids", "formation", "projected_points"], "lineup");
    const starters = ids(g.starting_ids, 11, "starting XI");
    if (starters.some(id => !squad.includes(id))) throw new TypeError("Starting XI outside squad");
    const captain = integer(g.captain_id, 1, Number.MAX_SAFE_INTEGER, "captain");
    const vice = g.vice_captain_id === null ? null : integer(g.vice_captain_id, 1, Number.MAX_SAFE_INTEGER, "vice captain");
    if (!starters.includes(captain) || (vice !== null && (!starters.includes(vice) || vice === captain))) throw new TypeError("Invalid captain selection");
    const bench = g.bench_ids === null ? null : ids(g.bench_ids, 4, "bench");
    if (bench && (bench.some(id => !squad.includes(id) || starters.includes(id)))) throw new TypeError("Bench does not complete squad");
    return { gameweek: integer(g.gameweek, target + i, target + i, "gameweek"), starting_ids: starters,
      captain_id: captain, vice_captain_id: vice, bench_ids: bench,
      formation: text(g.formation, 10), projected_points: finite(g.projected_points, "projected points") };
  });
  // Reuse the original bounded, allowlisted action validator.
  const action = parseLegacyEntry({ ...base, selection: { kind: "best", alternative_index: null },
    action: p.action, lineup: { captain_id: gameweeks[0].captain_id, captain_name: players.find(p => p.id === gameweeks[0].captain_id)!.name }, ai: null }, "plan", false).action!;
  return { action, squad_ids: squad, players, gameweeks };
}
function legacyId(entry: LegacyEntry): string {
  // Stable migration identifier; no invented historical timestamp or manager identity.
  const source = JSON.stringify(entry);
  let hash = 2166136261;
  for (let i = 0; i < source.length; i++) hash = Math.imul(hash ^ source.charCodeAt(i), 16777619);
  return `legacy-${entry.target_event}-${Date.parse(entry.saved_at)}-${(hash >>> 0).toString(16)}`;
}
export function parseDecisionHistoryEntry(value: unknown, path = "entry"): DecisionHistoryEntry {
  const r = obj(value, path);
  if (!("snapshot_id" in r)) {
    const legacy = parseLegacyEntry(r, path);
    return { ...legacy, snapshot_id: legacyId(legacy), origin: "migrated_v1", manager_id: null,
      season: decisionSeason(legacy.target_deadline), state_fingerprint: null, selection_source: "unknown", plans: null, context: null };
  }
  exact(r, [...legacyKeys, ...extensionKeys], path);
  const base = parseLegacyEntry(Object.fromEntries(legacyKeys.map(k => [k, r[k]])), path, false);
  if (r.origin !== "recorded_v2" && r.origin !== "migrated_v1") throw new TypeError("Invalid origin");
  if (!["model", "ai", "user", "unknown"].includes(String(r.selection_source))) throw new TypeError("Invalid selection source");
  const season = text(r.season, 7);
  if (season !== decisionSeason(base.target_deadline)) throw new TypeError("Season does not match deadline");
  let plans: DecisionHistoryEntry["plans"] = null;
  if (r.plans !== null) {
    const p = exact(r.plans, ["best", "roll", "alternatives"], "plans");
    if (!Array.isArray(p.alternatives) || p.alternatives.length > 5) throw new TypeError("Invalid alternatives");
    plans = { best: p.best === null ? null : parsePlan(p.best, base.target_event, base),
      roll: p.roll === null ? null : parsePlan(p.roll, base.target_event, base),
      alternatives: p.alternatives.map(v => parsePlan(v, base.target_event, base)) };
    if (plans.roll && plans.roll.action.kind !== "roll") throw new TypeError("Invalid roll plan");
  }
  let context: DecisionHistoryEntry["context"] = null;
  if (r.context !== null) {
    const c = exact(r.context, ["squad", "squad_value_tenths", "budget_tenths", "chips", "chip_usage", "no_active_chip_confirmed", "planned_chip"], "context");
    if (!Array.isArray(c.squad) || c.squad.length !== 15) throw new TypeError("Invalid original squad");
    const squad = c.squad.map(player);
    if (new Set(squad.map(p => p.id)).size !== 15) throw new TypeError("Duplicate original players");
    const chipNames = ["wildcard", "freehit", "bboost", "3xc"] as const;
    let chips = null;
    if (c.chips !== null) {
      const parsed = exact(c.chips, [...chipNames], "chips");
      if (Object.values(parsed).some(v => !["available", "used", "unavailable"].includes(String(v)))) throw new TypeError("Invalid chip inventory");
      chips = parsed as ManualManagerState["chips"];
    }
    let usage = null;
    if (c.chip_usage !== null) {
      if (!Array.isArray(c.chip_usage) || c.chip_usage.length > 8) throw new TypeError("Invalid chip history");
      usage = c.chip_usage.map(v => {
        const chip = exact(v, ["name", "event"], "chip usage");
        if (!chipNames.includes(chip.name as ChipName)) throw new TypeError("Invalid chip name");
        return { name: chip.name as ChipName, event: integer(chip.event, 1, 38, "chip event") };
      });
    }
    if (typeof c.no_active_chip_confirmed !== "boolean" || (c.planned_chip !== null && !chipNames.includes(c.planned_chip as ChipName))) throw new TypeError("Invalid planned chip");
    context = { squad, squad_value_tenths: integer(c.squad_value_tenths, 0, 5000, "squad value"),
      budget_tenths: integer(c.budget_tenths, 0, 6000, "budget"), chips, chip_usage: usage,
      no_active_chip_confirmed: c.no_active_chip_confirmed, planned_chip: c.planned_chip as ChipName | null };
  }
  if (r.origin === "recorded_v2") {
    if (Date.parse(base.saved_at) >= Date.parse(base.target_deadline)) throw new TypeError("Beslutningen skal gemmes før deadline.");
    if (Date.parse(base.recommendation_generated_at) > Date.parse(base.saved_at)
      || Date.parse(base.state_observed_at) > Date.parse(base.saved_at)) throw new TypeError("Snapshot predates its evidence");
    if (!plans?.best || !context) throw new TypeError("New snapshots require candidate plans and context");
  } else if (plans !== null || context !== null || r.manager_id !== null) throw new TypeError("Legacy details must remain unavailable");
  const parsed = { ...base, snapshot_id: text(r.snapshot_id, 100), origin: r.origin,
    manager_id: r.manager_id === null ? null : integer(r.manager_id, 1, Number.MAX_SAFE_INTEGER, "manager id"),
    season, state_fingerprint: r.state_fingerprint === null ? null : text(r.state_fingerprint, 64),
    selection_source: r.selection_source as DecisionHistoryEntry["selection_source"], plans, context } as DecisionHistoryEntry;
  const selected = selectedDecisionPlan(parsed);
  if (plans && base.selection.kind !== "wait" && (!selected || JSON.stringify(selected.action) !== JSON.stringify(base.action))) {
    throw new TypeError("Selected action does not match its candidate");
  }
  if (selected && (base.lineup?.captain_id !== selected.gameweeks[0].captain_id
    || base.lineup.captain_name !== selected.players.find(p => p.id === base.lineup!.captain_id)?.name)) {
    throw new TypeError("Saved captain does not match selected plan");
  }
  if (context && plans) {
    const value = context.squad.reduce((sum, p) => sum + (p.selling_price_tenths ?? p.price_tenths), 0);
    if (value !== context.squad_value_tenths || value + base.confirmed.bank_tenths !== context.budget_tenths) {
      throw new TypeError("Saved budget does not match confirmed squad");
    }
    for (const candidate of [plans.best, plans.roll, ...plans.alternatives]) {
      if (!candidate) continue;
      const expected = new Set(context.squad.map(p => p.id));
      for (const transfer of candidate.action.transfers) {
        if (!expected.delete(transfer.out_id) || expected.has(transfer.in_id)) throw new TypeError("Invalid candidate transfers");
        expected.add(transfer.in_id);
      }
      if (expected.size !== 15 || candidate.squad_ids.some(id => !expected.has(id))) throw new TypeError("Candidate squad does not match transfers");
    }
  }
  return parsed;
}
export function selectedDecisionPlan(entry: DecisionHistoryEntry): DecisionPlan | null {
  if (!entry.plans || entry.selection.kind === "wait") return null;
  return entry.selection.kind === "best" ? entry.plans.best : entry.plans.alternatives[entry.selection.alternative_index!] ?? null;
}
export function emptyDecisionHistory(): DecisionHistory { return { schema_version: DECISION_HISTORY_SCHEMA_VERSION, entries: [] }; }
export function parseDecisionHistory(value: unknown): DecisionHistory {
  const r = exact(value, ["schema_version", "entries"], "history");
  if (!["fpl-decision-history-v1", DECISION_HISTORY_SCHEMA_VERSION].includes(String(r.schema_version))) throw new TypeError("Unsupported history schema");
  if (!Array.isArray(r.entries) || r.entries.length > MAX_DECISION_HISTORY_ENTRIES) throw new TypeError("Journalen er fuld. Eksportér den, før du sletter noget.");
  const entries = r.entries.map((entry, index) => parseDecisionHistoryEntry(entry, `history.entries[${index}]`));
  if (new Set(entries.map(e => e.snapshot_id)).size !== entries.length) throw new TypeError("Duplicate snapshot identifiers");
  return { schema_version: DECISION_HISTORY_SCHEMA_VERSION, entries };
}
function compactAction(action: PlannerAction): DecisionHistoryAction {
  return { kind: action.kind, transfers: action.transfers.map(t => ({ out_id: t.out_id, out_name: t.out.name,
    in_id: t.in_id, in_name: t.in.name, position: t.position })), hit_points: action.hit_points,
    bank_after_tenths: action.bank_after_tenths, free_transfers_next_gameweek: action.free_transfers_next_gameweek,
    projected_points: action.projected_points, net_points_vs_roll: action.net_points_vs_roll,
    decision_value_vs_roll: action.decision_value_vs_roll };
}
export function buildDecisionHistoryEntry(input: DecisionHistoryBuildInput): DecisionHistoryEntry {
  const saved = new Date(input.saved_at ?? Date.now()).toISOString();
  if (Date.parse(saved) >= Date.parse(input.target_deadline)) throw new TypeError("Beslutningen skal gemmes før deadline.");
  const review = input.ai_review?.review ?? null;
  if (input.ai_review && (input.ai_review.target_event !== input.planner.target_event
    || Date.parse(input.ai_review.recommendation_generated_at) !== Date.parse(input.recommendation_generated_at))) throw new TypeError("AI review does not match recommendation");
  const selection = input.selection ?? {
    kind: review?.verdict === "wait_for_information" ? "wait" : review?.verdict === "prefer_alternative" ? "alternative" : "best",
    alternative_index: review?.verdict === "prefer_alternative" ? review.alternative_index : null,
  };
  const original = input.planner.confirmed_state.squad.map(p => player({ id: p.id, name: p.name,
    position: p.position, team: p.team, price_tenths: p.price_tenths,
    purchase_price_tenths: p.purchase_price_tenths, selling_price_tenths: p.selling_price_tenths }));
  const makePlan = (action: PlannerAction, details?: DecisionLineupDetails): DecisionPlan => {
    const players = new Map(original.map(p => [p.id, p]));
    for (const t of action.transfers) { players.delete(t.out_id); players.set(t.in_id, { id: t.in_id,
      name: t.in.name, position: t.in.position, team: t.in.team, price_tenths: t.in_price_tenths,
      purchase_price_tenths: t.in_price_tenths, selling_price_tenths: t.in_price_tenths }); }
    return { action: compactAction(action), squad_ids: action.squad_ids, players: action.squad_ids.map(id => players.get(id)!),
      gameweeks: action.gameweeks.map((g, index) => {
        const extra = g as typeof g & { vice_captain_id?: number; bench_ids?: number[] };
        const d = index === 0 ? details : undefined;
        return { gameweek: g.gameweek, starting_ids: d?.starting_ids ?? g.starting_ids,
          captain_id: d?.captain_id ?? g.captain_id, vice_captain_id: d?.vice_captain_id ?? extra.vice_captain_id ?? null,
          bench_ids: d?.bench_ids ?? extra.bench_ids ?? null, formation: g.formation, projected_points: g.projected_points };
      }) };
  };
  const best = makePlan(input.planner.best_action, input.lineup_details?.best);
  const alternatives = input.planner.alternatives.map((a, i) => makePlan(a, input.lineup_details?.alternatives?.[i]));
  const plans = { best, roll: [best, ...alternatives].find(p => p.action.kind === "roll") ?? null, alternatives };
  const selected = selection.kind === "wait" ? null : selection.kind === "best" ? best : alternatives[selection.alternative_index!];
  if (selection.kind !== "wait" && !selected) throw new TypeError("Selected alternative is unavailable");
  const first = selected?.gameweeks[0];
  const squadValue = original.reduce((sum, p) => sum + (p.selling_price_tenths ?? p.price_tenths), 0);
  return parseDecisionHistoryEntry({
    snapshot_id: crypto.randomUUID(), origin: "recorded_v2", saved_at: saved,
    manager_id: input.planner.manager_id, season: decisionSeason(input.target_deadline), state_fingerprint: input.planner.state_fingerprint,
    target_event: input.planner.target_event, target_deadline: input.target_deadline, source_event: input.planner.source_event,
    state_observed_at: input.state_observed_at, recommendation_generated_at: input.recommendation_generated_at, forecast: input.forecast,
    confirmed: { bank_tenths: input.planner.confirmed_state.bank_tenths, free_transfers: input.planner.confirmed_state.free_transfers },
    selection, selection_source: input.selection ? "user" : review ? "ai" : "model",
    action: selected?.action ?? null, lineup: first ? { captain_id: first.captain_id, captain_name: selected!.players.find(p => p.id === first.captain_id)!.name } : null,
    ai: review ? { verdict: review.verdict, execution_timing: review.execution_timing, headline: review.headline, confidence: review.confidence } : null,
    plans, context: { squad: original, squad_value_tenths: squadValue, budget_tenths: squadValue + input.planner.confirmed_state.bank_tenths,
      chips: input.manager_state?.chips ?? null, chip_usage: input.manager_state?.chip_usage ?? null,
      no_active_chip_confirmed: input.planner.confirmed_state.no_active_chip_confirmed, planned_chip: input.planned_chip ?? null },
  });
}
export function upsertDecisionHistory(history: DecisionHistory, entry: DecisionHistoryEntry): DecisionHistory {
  if (history.storage_error) throw new TypeError(history.storage_error);
  const clean = parseDecisionHistory(history);
  const next = parseDecisionHistoryEntry(entry);
  return mergeEntries(clean.entries, [next]);
}
function mergeEntries(existing: DecisionHistoryEntry[], additions: DecisionHistoryEntry[]): DecisionHistory {
  const entries = new Map(existing.map(entry => [entry.snapshot_id, entry]));
  for (const next of additions) {
    const previous = entries.get(next.snapshot_id);
    if (previous && JSON.stringify(previous) !== JSON.stringify(next)) throw new TypeError("Et gemt snapshot kan ikke ændres.");
    entries.set(next.snapshot_id, next);
  }
  if (entries.size > MAX_DECISION_HISTORY_ENTRIES) throw new TypeError("Journalen er fuld. Eksportér den, før du sletter noget.");
  return { schema_version: DECISION_HISTORY_SCHEMA_VERSION,
    entries: [...entries.values()].sort((a, b) => Date.parse(b.saved_at) - Date.parse(a.saved_at)) };
}
export function readDecisionHistory(storage: StorageLike): DecisionHistory {
  try {
    let merged = emptyDecisionHistory();
    // The v1 original is never overwritten by migration. This also isolates
    // v2 snapshots from writes made by a still-open older version of the app.
    for (const key of [DECISION_HISTORY_STORAGE_KEY, LEGACY_DECISION_HISTORY_STORAGE_KEY]) {
      const raw = storage.getItem(key);
      if (raw === null) continue;
      if (new TextEncoder().encode(raw).byteLength > MAX_DECISION_HISTORY_BYTES) throw new TypeError("Journalen overskrider størrelsesgrænsen.");
      merged = mergeEntries(merged.entries, parseDecisionHistory(JSON.parse(raw)).entries);
    }
    return merged;
  } catch {
    return { ...emptyDecisionHistory(), storage_error: "Den eksisterende journal kunne ikke læses. Data bliver bevaret. Eksportér browserens oprindelige journal, før du forsøger at nulstille den." };
  }
}
export function writeDecisionHistory(storage: StorageLike, history: DecisionHistory): boolean {
  try {
    if (history.storage_error) return false;
    let merged = readDecisionHistory(storage);
    if (merged.storage_error) return false;
    merged = mergeEntries(merged.entries, parseDecisionHistory(history).entries);
    const raw = JSON.stringify(merged);
    if (new TextEncoder().encode(raw).byteLength > MAX_DECISION_HISTORY_BYTES) return false;
    storage.setItem(DECISION_HISTORY_STORAGE_KEY, raw);
    return true;
  } catch { return false; }
}
export async function saveDecisionHistoryEntry(storage: StorageLike, entry: DecisionHistoryEntry, now: () => number = Date.now): Promise<DecisionHistory> {
  const save = () => {
    if (now() >= Date.parse(entry.target_deadline)) throw new Error("Deadline er passeret. Snapshot blev ikke gemt.");
    const updated = upsertDecisionHistory(readDecisionHistory(storage), entry);
    if (!writeDecisionHistory(storage, updated)) throw new Error("Journalen kunne ikke gemmes. Den eksisterende journal er bevaret og kan eksporteres.");
    return readDecisionHistory(storage);
  };
  // A shared origin lock serializes concurrent saves from multiple tabs.
  if (typeof window !== "undefined") {
    if (navigator.locks) return navigator.locks.request("fpl-decision-history", save);
    throw new Error("Browseren understøtter ikke sikker samtidig gemning. Eksportér journalen, og brug en opdateret browser.");
  }
  return save();
}
export function clearDecisionHistory(storage: StorageLike): boolean {
  try {
    storage.removeItem(DECISION_HISTORY_STORAGE_KEY);
    storage.removeItem(LEGACY_DECISION_HISTORY_STORAGE_KEY);
    return true;
  } catch { return false; }
}
