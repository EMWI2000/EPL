import type { AiReviewResponse } from "./ai-review-contract.ts";
import type { PlannerAction, PlannerPayload, PlayerPosition } from "./planner-contract.ts";

export const DECISION_HISTORY_SCHEMA_VERSION = "fpl-decision-history-v1" as const;
export const DECISION_HISTORY_STORAGE_KEY = "fpl-decision-history-v1";
export const MAX_DECISION_HISTORY_ENTRIES = 38;
export const MAX_DECISION_HISTORY_BYTES = 64 * 1024;

export type DecisionSelectionKind = "best" | "alternative" | "wait";

export interface DecisionHistoryTransfer {
  out_id: number;
  out_name: string;
  in_id: number;
  in_name: string;
  position: PlayerPosition;
}

export interface DecisionHistoryAction {
  kind: PlannerAction["kind"];
  transfers: DecisionHistoryTransfer[];
  hit_points: number;
  bank_after_tenths: number;
  free_transfers_next_gameweek: number;
  projected_points: number;
  net_points_vs_roll: number;
  decision_value_vs_roll: number;
}

export interface DecisionHistoryEntry {
  saved_at: string;
  target_event: number;
  target_deadline: string;
  source_event: number;
  state_observed_at: string;
  recommendation_generated_at: string;
  forecast: {
    version: "v2" | "legacy";
    horizon: number;
    validation_status: string;
  };
  confirmed: {
    bank_tenths: number;
    free_transfers: number;
  };
  selection: {
    kind: DecisionSelectionKind;
    alternative_index: number | null;
  };
  action: DecisionHistoryAction | null;
  lineup: {
    captain_id: number;
    captain_name: string;
  } | null;
  ai: {
    verdict: AiReviewResponse["review"]["verdict"];
    execution_timing: AiReviewResponse["review"]["execution_timing"];
    headline: string;
    confidence: AiReviewResponse["review"]["confidence"];
  } | null;
}

export interface DecisionHistory {
  schema_version: typeof DECISION_HISTORY_SCHEMA_VERSION;
  entries: DecisionHistoryEntry[];
}

export interface DecisionHistoryBuildInput {
  saved_at?: string;
  target_deadline: string;
  state_observed_at: string;
  recommendation_generated_at: string;
  forecast: {
    version: "v2" | "legacy";
    horizon: number;
    validation_status: string;
  };
  planner: PlannerPayload;
  ai_review: AiReviewResponse | null;
}

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

type UnknownRecord = Record<string, unknown>;

function record(value: unknown, path: string): UnknownRecord {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${path} must be an object`);
  }
  return value as UnknownRecord;
}

function exactRecord(value: unknown, path: string, keys: readonly string[]): UnknownRecord {
  const result = record(value, path);
  const actual = Object.keys(result).sort();
  const expected = [...keys].sort();
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) {
    throw new TypeError(`${path} has unexpected or missing fields`);
  }
  return result;
}

function text(value: unknown, path: string, maximum: number): string {
  if (typeof value !== "string") throw new TypeError(`${path} must be text`);
  const normalized = value.replace(/[\u0000-\u001f\u007f]/g, " ").replace(/\s+/g, " ").trim();
  if (!normalized || normalized.length > maximum) {
    throw new TypeError(`${path} must contain 1 to ${maximum} characters`);
  }
  return normalized;
}

function integer(value: unknown, path: string, minimum: number, maximum: number): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum || (value as number) > maximum) {
    throw new TypeError(`${path} must be an integer from ${minimum} to ${maximum}`);
  }
  return value as number;
}

function finite(value: unknown, path: string, minimum = -10_000, maximum = 10_000): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum || value > maximum) {
    throw new TypeError(`${path} must be a finite number`);
  }
  return value;
}

function isoTimestamp(value: unknown, path: string): string {
  const raw = text(value, path, 40);
  const timestamp = new Date(raw);
  if (!Number.isFinite(timestamp.getTime()) || !/(?:Z|[+-]\d\d:\d\d)$/i.test(raw)) {
    throw new TypeError(`${path} must be an ISO timestamp with a timezone`);
  }
  return timestamp.toISOString();
}

function oneOf<T extends string>(value: unknown, path: string, allowed: readonly T[]): T {
  if (typeof value !== "string" || !allowed.includes(value as T)) {
    throw new TypeError(`${path} has an unsupported value`);
  }
  return value as T;
}

function parseTransfer(value: unknown, path: string): DecisionHistoryTransfer {
  const row = exactRecord(value, path, ["out_id", "out_name", "in_id", "in_name", "position"]);
  const outId = integer(row.out_id, `${path}.out_id`, 1, Number.MAX_SAFE_INTEGER);
  const inId = integer(row.in_id, `${path}.in_id`, 1, Number.MAX_SAFE_INTEGER);
  if (outId === inId) throw new TypeError(`${path} must contain different players`);
  return {
    out_id: outId,
    out_name: text(row.out_name, `${path}.out_name`, 80),
    in_id: inId,
    in_name: text(row.in_name, `${path}.in_name`, 80),
    position: oneOf(row.position, `${path}.position`, ["GKP", "DEF", "MID", "FWD"] as const),
  };
}

function parseAction(value: unknown, path: string): DecisionHistoryAction {
  const row = exactRecord(value, path, [
    "kind", "transfers", "hit_points", "bank_after_tenths", "free_transfers_next_gameweek",
    "projected_points", "net_points_vs_roll", "decision_value_vs_roll",
  ]);
  if (!Array.isArray(row.transfers) || row.transfers.length > 5) {
    throw new TypeError(`${path}.transfers must contain at most five transfers`);
  }
  const transfers = row.transfers.map((transfer, index) => parseTransfer(transfer, `${path}.transfers[${index}]`));
  const kind = oneOf(row.kind, `${path}.kind`, ["roll", "transfer", "hit"] as const);
  if ((kind === "roll") !== (transfers.length === 0)) {
    throw new TypeError(`${path}.kind does not match its transfers`);
  }
  return {
    kind,
    transfers,
    hit_points: integer(row.hit_points, `${path}.hit_points`, 0, 20),
    bank_after_tenths: integer(row.bank_after_tenths, `${path}.bank_after_tenths`, 0, 1_000),
    free_transfers_next_gameweek: integer(row.free_transfers_next_gameweek, `${path}.free_transfers_next_gameweek`, 1, 5),
    projected_points: finite(row.projected_points, `${path}.projected_points`, 0),
    net_points_vs_roll: finite(row.net_points_vs_roll, `${path}.net_points_vs_roll`),
    decision_value_vs_roll: finite(row.decision_value_vs_roll, `${path}.decision_value_vs_roll`),
  };
}

export function parseDecisionHistoryEntry(value: unknown, path = "entry"): DecisionHistoryEntry {
  const row = exactRecord(value, path, [
    "saved_at", "target_event", "target_deadline", "source_event", "state_observed_at",
    "recommendation_generated_at", "forecast", "confirmed", "selection", "action", "lineup", "ai",
  ]);
  const targetEvent = integer(row.target_event, `${path}.target_event`, 1, 38);
  const sourceEvent = integer(row.source_event, `${path}.source_event`, 1, 38);
  if (sourceEvent >= targetEvent) throw new TypeError(`${path}.source_event must precede target_event`);

  const forecast = exactRecord(row.forecast, `${path}.forecast`, ["version", "horizon", "validation_status"]);
  const confirmed = exactRecord(row.confirmed, `${path}.confirmed`, ["bank_tenths", "free_transfers"]);
  const selection = exactRecord(row.selection, `${path}.selection`, ["kind", "alternative_index"]);
  const selectionKind = oneOf(selection.kind, `${path}.selection.kind`, ["best", "alternative", "wait"] as const);
  const alternativeIndex = selection.alternative_index === null
    ? null
    : integer(selection.alternative_index, `${path}.selection.alternative_index`, 0, 3);
  if ((selectionKind === "alternative") !== (alternativeIndex !== null)) {
    throw new TypeError(`${path}.selection.alternative_index does not match selection kind`);
  }
  const action = row.action === null ? null : parseAction(row.action, `${path}.action`);
  if ((selectionKind === "wait") !== (action === null)) {
    throw new TypeError(`${path}.action does not match selection kind`);
  }

  let lineup: DecisionHistoryEntry["lineup"] = null;
  if (row.lineup !== null) {
    const value = exactRecord(row.lineup, `${path}.lineup`, ["captain_id", "captain_name"]);
    lineup = {
      captain_id: integer(value.captain_id, `${path}.lineup.captain_id`, 1, Number.MAX_SAFE_INTEGER),
      captain_name: text(value.captain_name, `${path}.lineup.captain_name`, 80),
    };
  }
  if ((selectionKind === "wait") !== (lineup === null)) {
    throw new TypeError(`${path}.lineup does not match selection kind`);
  }

  let ai: DecisionHistoryEntry["ai"] = null;
  if (row.ai !== null) {
    const value = exactRecord(row.ai, `${path}.ai`, ["verdict", "execution_timing", "headline", "confidence"]);
    ai = {
      verdict: oneOf(value.verdict, `${path}.ai.verdict`, ["confirm_best_action", "wait_for_information", "prefer_alternative"] as const),
      execution_timing: oneOf(value.execution_timing, `${path}.ai.execution_timing`, ["act_now", "wait_for_team_news", "monitor_price_window"] as const),
      headline: text(value.headline, `${path}.ai.headline`, 160),
      confidence: oneOf(value.confidence, `${path}.ai.confidence`, ["low", "medium", "high"] as const),
    };
    const expectedVerdict = selectionKind === "wait"
      ? "wait_for_information"
      : selectionKind === "alternative"
        ? "prefer_alternative"
        : "confirm_best_action";
    if (ai.verdict !== expectedVerdict) throw new TypeError(`${path}.ai.verdict does not match selection kind`);
  }
  if (selectionKind !== "best" && ai === null) {
    throw new TypeError(`${path}.ai is required for an alternative or wait selection`);
  }

  return {
    saved_at: isoTimestamp(row.saved_at, `${path}.saved_at`),
    target_event: targetEvent,
    target_deadline: isoTimestamp(row.target_deadline, `${path}.target_deadline`),
    source_event: sourceEvent,
    state_observed_at: isoTimestamp(row.state_observed_at, `${path}.state_observed_at`),
    recommendation_generated_at: isoTimestamp(row.recommendation_generated_at, `${path}.recommendation_generated_at`),
    forecast: {
      version: oneOf(forecast.version, `${path}.forecast.version`, ["v2", "legacy"] as const),
      horizon: integer(forecast.horizon, `${path}.forecast.horizon`, 1, 5),
      validation_status: text(forecast.validation_status, `${path}.forecast.validation_status`, 40),
    },
    confirmed: {
      bank_tenths: integer(confirmed.bank_tenths, `${path}.confirmed.bank_tenths`, 0, 1_000),
      free_transfers: integer(confirmed.free_transfers, `${path}.confirmed.free_transfers`, 1, 5),
    },
    selection: { kind: selectionKind, alternative_index: alternativeIndex },
    action,
    lineup,
    ai,
  };
}

export function emptyDecisionHistory(): DecisionHistory {
  return { schema_version: DECISION_HISTORY_SCHEMA_VERSION, entries: [] };
}

export function parseDecisionHistory(value: unknown): DecisionHistory {
  const root = exactRecord(value, "history", ["schema_version", "entries"]);
  if (root.schema_version !== DECISION_HISTORY_SCHEMA_VERSION) {
    throw new TypeError("history.schema_version is unsupported");
  }
  if (!Array.isArray(root.entries) || root.entries.length > MAX_DECISION_HISTORY_ENTRIES) {
    throw new TypeError(`history.entries must contain at most ${MAX_DECISION_HISTORY_ENTRIES} entries`);
  }
  const entries = root.entries.map((entry, index) => parseDecisionHistoryEntry(entry, `history.entries[${index}]`));
  const keys = entries.map((entry) => `${entry.target_event}:${entry.target_deadline}`);
  if (new Set(keys).size !== keys.length) throw new TypeError("history.entries contains duplicate deadlines");
  return { schema_version: DECISION_HISTORY_SCHEMA_VERSION, entries };
}

function compactAction(action: PlannerAction): DecisionHistoryAction {
  return {
    kind: action.kind,
    transfers: action.transfers.map((transfer) => ({
      out_id: transfer.out_id,
      out_name: transfer.out.name,
      in_id: transfer.in_id,
      in_name: transfer.in.name,
      position: transfer.position,
    })),
    hit_points: action.hit_points,
    bank_after_tenths: action.bank_after_tenths,
    free_transfers_next_gameweek: action.free_transfers_next_gameweek,
    projected_points: action.projected_points,
    net_points_vs_roll: action.net_points_vs_roll,
    decision_value_vs_roll: action.decision_value_vs_roll,
  };
}

export function buildDecisionHistoryEntry(input: DecisionHistoryBuildInput): DecisionHistoryEntry {
  const review = input.ai_review?.review ?? null;
  const selectionKind: DecisionSelectionKind = review?.verdict === "wait_for_information"
    ? "wait"
    : review?.verdict === "prefer_alternative"
      ? "alternative"
      : "best";
  const alternativeIndex = selectionKind === "alternative" ? review?.alternative_index ?? null : null;
  const action = selectionKind === "wait"
    ? null
    : selectionKind === "alternative" && alternativeIndex !== null
      ? input.planner.alternatives[alternativeIndex] ?? null
      : input.planner.best_action;
  if (selectionKind !== "wait" && action === null) {
    throw new TypeError("The selected planner action is unavailable");
  }
  if (input.ai_review && (
    new Date(input.ai_review.recommendation_generated_at).getTime()
      !== new Date(input.recommendation_generated_at).getTime()
    || input.ai_review.target_event !== input.planner.target_event
  )) {
    throw new TypeError("The AI review does not match the recommendation being saved");
  }
  let lineup: DecisionHistoryEntry["lineup"] = null;
  if (action !== null) {
    const firstGameweek = action.gameweeks[0];
    if (!firstGameweek) throw new TypeError("The selected planner action has no lineup");
    const players = new Map(
      input.planner.confirmed_state.squad.map((player) => [player.id, player.name]),
    );
    for (const transfer of action.transfers) {
      players.delete(transfer.out_id);
      players.set(transfer.in_id, transfer.in.name);
    }
    const captainName = players.get(firstGameweek.captain_id);
    if (!captainName) throw new TypeError("The selected planner action has an unknown captain");
    lineup = { captain_id: firstGameweek.captain_id, captain_name: captainName };
  }

  return parseDecisionHistoryEntry({
    saved_at: input.saved_at ?? new Date().toISOString(),
    target_event: input.planner.target_event,
    target_deadline: input.target_deadline,
    source_event: input.planner.source_event,
    state_observed_at: input.state_observed_at,
    recommendation_generated_at: input.recommendation_generated_at,
    forecast: {
      version: input.forecast.version,
      horizon: input.forecast.horizon,
      validation_status: input.forecast.validation_status,
    },
    confirmed: {
      bank_tenths: input.planner.confirmed_state.bank_tenths,
      free_transfers: input.planner.confirmed_state.free_transfers,
    },
    selection: { kind: selectionKind, alternative_index: alternativeIndex },
    action: action === null ? null : compactAction(action),
    lineup,
    ai: review === null ? null : {
      verdict: review.verdict,
      execution_timing: review.execution_timing,
      headline: review.headline,
      confidence: review.confidence,
    },
  });
}

export function upsertDecisionHistory(history: DecisionHistory, entry: DecisionHistoryEntry): DecisionHistory {
  const cleanHistory = parseDecisionHistory(history);
  const cleanEntry = parseDecisionHistoryEntry(entry);
  const key = `${cleanEntry.target_event}:${cleanEntry.target_deadline}`;
  const entries = cleanHistory.entries
    .filter((current) => `${current.target_event}:${current.target_deadline}` !== key)
    .concat(cleanEntry)
    .sort((left, right) => Date.parse(right.target_deadline) - Date.parse(left.target_deadline))
    .slice(0, MAX_DECISION_HISTORY_ENTRIES);
  return { schema_version: DECISION_HISTORY_SCHEMA_VERSION, entries };
}

export function readDecisionHistory(storage: StorageLike): DecisionHistory {
  try {
    const raw = storage.getItem(DECISION_HISTORY_STORAGE_KEY);
    if (raw === null || new TextEncoder().encode(raw).byteLength > MAX_DECISION_HISTORY_BYTES) {
      return emptyDecisionHistory();
    }
    return parseDecisionHistory(JSON.parse(raw));
  } catch {
    return emptyDecisionHistory();
  }
}

export function writeDecisionHistory(storage: StorageLike, history: DecisionHistory): boolean {
  try {
    const clean = parseDecisionHistory(history);
    const serialized = JSON.stringify(clean);
    if (new TextEncoder().encode(serialized).byteLength > MAX_DECISION_HISTORY_BYTES) return false;
    storage.setItem(DECISION_HISTORY_STORAGE_KEY, serialized);
    return true;
  } catch {
    return false;
  }
}

export function clearDecisionHistory(storage: StorageLike): boolean {
  try {
    storage.removeItem(DECISION_HISTORY_STORAGE_KEY);
    return true;
  } catch {
    return false;
  }
}
