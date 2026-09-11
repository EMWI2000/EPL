import { parseChipSequences, type ChipSequenceComparison } from "./chip-sequences.ts";

export const MANAGER_SYNC_SCHEMA_VERSION = "fpl-manager-state-response-v3" as const;
export const PERSONAL_STATE_SCHEMA_VERSION = "fpl-personal-state-v2" as const;
export const SNAPSHOT_SCHEMA_VERSION = "fpl-deadline-state-snapshot-v2" as const;
export const PRICE_SIGNAL_SCHEMA_VERSION = "fpl-official-price-signals-v1" as const;

export type ForecastVersion = "v2" | "legacy";
export type PlayerPosition = "GKP" | "DEF" | "MID" | "FWD";
export type PlayerStatus = "a" | "d" | "i" | "s" | "u" | "n";
export type ChipName = "wildcard" | "freehit" | "bboost" | "3xc";
export type ChipStatus = "available" | "used" | "unavailable";

export interface ChipUsage {
  name: ChipName;
  event: number;
}

export interface PriceProjection {
  offset_days: number;
  projected_percent: number;
  likelihood_code: number;
}

export interface OfficialPriceSignal {
  element_id: number;
  now_cost_tenths: number;
  cost_change_start_tenths: number;
  cost_change_event_tenths: number;
  selected_by_percent: number;
  transfers_in_event: number;
  transfers_out_event: number;
  price_change_percent: number | null;
  price_change_hourly_rate: number | null;
  projections: PriceProjection[];
  locked_until: string | null;
  calibrating: boolean;
}

export type PurchasePriceBasis =
  | { kind: "season_start_price" }
  | {
      kind: "latest_public_transfer_in";
      event: number;
      confirmed_at: string | null;
    };

export interface ManagerSyncPick {
  element_id: number;
  name: string;
  club: string;
  club_name: string;
  club_id: number;
  position: PlayerPosition;
  lineup_position: number;
  multiplier: number;
  is_captain: boolean;
  is_vice_captain: boolean;
  current_price_tenths: number;
  estimated_purchase_price_tenths: number;
  estimated_selling_price_tenths: number;
  purchase_price_basis: PurchasePriceBasis;
  official_price_signal: OfficialPriceSignal | null;
}

export interface ManagerSyncPlayerCatalogEntry {
  id: number;
  display_name: string;
  position: PlayerPosition;
  team_id: number;
  team_name: string;
  current_price_tenths: number;
}

export type PublicStateLimitation =
  | "state_is_locked_at_last_public_deadline"
  | "current_free_transfers_not_public"
  | "current_confirmed_transfers_not_public"
  | "purchase_and_selling_prices_not_public"
  | "next_deadline_chip_selection_not_public";

export interface LastDeadlineState {
  schema_version: typeof PERSONAL_STATE_SCHEMA_VERSION;
  state_kind: "public_last_deadline";
  event: number;
  bank_tenths: number;
  squad_value_tenths: number;
  total_transfers_at_deadline: number;
  event_transfers: number;
  event_transfer_cost: number;
  active_chip: ChipName | null;
  chip_usage: ChipUsage[];
  limitations: PublicStateLimitation[];
  picks: ManagerSyncPick[];
}

export interface PlayerPriceState {
  element_id: number;
  purchase_price_tenths: number;
  selling_price_tenths: number;
}

export type ChipState = Record<ChipName, ChipStatus>;

export interface ManualManagerState {
  current_squad_ids: number[];
  bank_tenths: number;
  free_transfers: number;
  player_prices: PlayerPriceState[];
  chips: ChipState;
  chip_usage: ChipUsage[];
  no_active_chip_confirmed: boolean;
  effective_event: number;
}

export type ConfirmationField =
  | "current_squad_ids"
  | "bank_tenths"
  | "free_transfers"
  | "player_prices"
  | "chips"
  | "chip_usage"
  | "no_active_chip_confirmed";

export interface ManualStateTemplate {
  state: ManualManagerState & { free_transfers: 1; effective_event: number };
  confirmation_required: true;
  fields_requiring_confirmation: ConfirmationField[];
  free_transfers_default_reason: "current_free_transfers_are_not_public";
}

export type ManagerSyncWarningCode =
  | "last_deadline_state_requires_confirmation"
  | "free_hit_squad_is_temporary";

export interface ManagerSyncResponse {
  schema_version: typeof MANAGER_SYNC_SCHEMA_VERSION;
  generated_at: string;
  manager: {
    id: number;
    team_name: string;
    overall_points: number | null;
    overall_rank: number | null;
  };
  target: { event: number; name: string; deadline_time: string };
  last_deadline_state: LastDeadlineState;
  player_catalog: ManagerSyncPlayerCatalogEntry[];
  price_signals: {
    schema_version: typeof PRICE_SIGNAL_SCHEMA_VERSION;
    available: boolean;
    player_count: number;
    price_change_deadlines: string[];
    warning: string | null;
  };
  manual_state_template: ManualStateTemplate;
  snapshot: {
    schema_version: typeof SNAPSHOT_SCHEMA_VERSION;
    observed_at: string;
    source: "fpl_public_last_deadline";
    checksum_sha256: string;
    persisted: false;
  };
  warnings: Array<{ code: ManagerSyncWarningCode; message: string }>;
}

export interface PlannerPlayerReference {
  id: number;
  name: string;
  team: string;
  position: PlayerPosition;
  price_tenths: number;
  status: PlayerStatus;
  price_signal: OfficialPriceSignal | null;
}

export interface PlannerOwnedPlayerReference extends PlannerPlayerReference {
  purchase_price_tenths: number;
  selling_price_tenths: number;
}

export interface EnrichedTransfer {
  out_id: number;
  in_id: number;
  position: PlayerPosition;
  out_purchase_price_tenths: number;
  out_current_price_tenths: number;
  out_selling_price_tenths: number;
  in_price_tenths: number;
  out: PlannerPlayerReference;
  in: PlannerPlayerReference;
}

export interface PlannerActionGameweek {
  gameweek: number;
  starting_ids: number[];
  captain_id: number;
  formation: string;
  projected_points: number;
}

export type PlannerActionKind = "roll" | "transfer" | "hit";

export interface PlannerAction {
  kind: PlannerActionKind;
  transfer_count: number;
  transfers: EnrichedTransfer[];
  squad_ids: number[];
  gameweeks: PlannerActionGameweek[];
  bank_before_tenths: number;
  bank_after_tenths: number;
  free_transfers_before: number;
  free_transfers_next_gameweek: number;
  hit_points: number;
  projected_points: number;
  banked_ft_value_points: number;
  decision_value_points: number;
  net_points_vs_roll: number;
  decision_value_vs_roll: number;
  explanation: string;
}

export interface PlannerSequentialStep {
  deadline_offset: 1 | 2;
  provisional: boolean;
  target_event: number;
  kind: PlannerActionKind;
  transfer_count: number;
  transfers: EnrichedTransfer[];
  squad_ids: number[];
  gameweeks: PlannerActionGameweek[];
  bank_before_tenths: number;
  bank_after_tenths: number;
  free_transfers_before: number;
  free_transfers_next_gameweek: number;
  hit_points: number;
  weighted_projected_points: number;
  weighted_hit_cost_points: number;
}

export interface PlannerSequentialPlan {
  horizon: number;
  gw_weights: number[];
  first_step_candidate_count: number;
  first_step_search: "explicit_bounded_transfer_plans";
  future_price_assumption: "fixed_current_prices";
  solver_proven_optimal_within_bounds: true;
  globally_optimal: false;
  best_sequence: {
    first_action: {
      source: "best_action" | "alternative";
      alternative_index: number | null;
    };
    steps: [PlannerSequentialStep, PlannerSequentialStep];
    weighted_projected_points: number;
    total_hit_points: number;
    weighted_hit_cost_points: number;
    terminal_banked_ft_value_points: number;
    decision_value_points: number;
  };
}

export interface PlannerStrategyStep {
  deadline_offset: 1 | 2 | 3 | 4;
  provisional: boolean;
  target_event: number;
  kind: PlannerActionKind;
  transfer_count: number;
  transfers: EnrichedTransfer[];
  squad_ids: number[];
  gameweeks: PlannerActionGameweek[];
  bank_before_tenths: number;
  bank_after_tenths: number;
  free_transfers_before: number;
  free_transfers_next_gameweek: number;
  hit_points: number;
  weighted_projected_points: number;
  weighted_hit_cost_points: number;
}

export interface PlannerStrategyPlan {
  horizon: number;
  gameweek_window: number[];
  gw_weights: number[];
  modelled_deadlines: 4;
  maximum_provisional_transfers: 2;
  first_step_candidate_count: number;
  first_step_search: "explicit_bounded_transfer_plans";
  search_scope: "four_deadlines_max_two_provisional_transfers";
  future_price_assumption: "fixed_current_prices";
  assumptions: string[];
  solver_proven_optimal_within_bounds: true;
  globally_optimal: false;
  recalculate_each_deadline: true;
  first_action: {
    source: "best_action" | "alternative";
    alternative_index: number | null;
  };
  steps: [
    PlannerStrategyStep,
    PlannerStrategyStep,
    PlannerStrategyStep,
    PlannerStrategyStep,
  ];
  weighted_projected_points: number;
  total_hit_points: number;
  weighted_hit_cost_points: number;
  terminal_banked_ft_value_points: number;
  decision_value_points: number;
}

export interface PlannerChipInventoryEntry {
  chip: ChipName;
  used_events: number[];
  available_for_target: boolean;
}

export type PlannerChipSignal = "hold" | "watch" | "consider";
export type PlannerChipConfidence = "low" | "medium";
export type PlannerChipScenarioScope =
  | "multiweek_rebuild"
  | "confirmed_blank_double_screen"
  | "single_gameweek_counterfactual"
  | "bench_marginal"
  | "captain_marginal";

export interface PlannerChipScenario {
  scenario_id: string;
  chip: ChipName;
  event: number | null;
  signal: PlannerChipSignal;
  available: boolean;
  estimated_gain_points: number | null;
  baseline_points: number | null;
  chip_points: number | null;
  confidence: PlannerChipConfidence;
  model_scope: PlannerChipScenarioScope;
  reason: string;
  squad: PlannerPlayerReference[];
  change_count: number | null;
  bank_after_tenths: number | null;
}

export interface PlannerChipStrategy {
  sequence_comparison?: ChipSequenceComparison | null;
  horizon: number;
  target_event: number;
  inventory: PlannerChipInventoryEntry[];
  scenarios: PlannerChipScenario[];
  recommendation: {
    action: "hold" | "consider";
    scenario_id: string | null;
    chip: ChipName | null;
    event: number | null;
    reason: string;
  };
  model_scope: "bounded_chip_counterfactuals";
  globally_optimal: false;
  recalculate_each_deadline: true;
}

export interface PlannerPayload {
  manager_id: number;
  state_fingerprint: string | null;
  source_event: number;
  target_event: number;
  confirmed_state: {
    bank_tenths: number;
    free_transfers: number;
    no_active_chip_confirmed: true;
    squad: PlannerOwnedPlayerReference[];
  };
  best_action: PlannerAction;
  alternatives: PlannerAction[];
  sequential: PlannerSequentialPlan | null;
  strategy: PlannerStrategyPlan | null;
  chip_strategy: PlannerChipStrategy | null;
  method: {
    candidate_count: number;
    plans_per_transfer_count: 5;
    higher_transfer_count_plans: 1;
    maximum_immediate_transfers: number;
    roll_ft_value_points: number;
    chips_modelled: boolean;
    bounded_roadmap_modelled: boolean;
    next_deadline_transfer_modelled: boolean;
    future_transfers_modelled: false;
  };
}

export interface PlannerRequestInput {
  manager_id: number;
  source_event: number;
  state_fingerprint: string;
  manager_state: ManualManagerState;
  horizon: number;
  include_doubtful: boolean;
  forecast_version: ForecastVersion;
}

export interface PlannerComputeRequest extends PlannerRequestInput {
  use_solio: false;
}

export class PlannerContractError extends TypeError {
  readonly path: string;

  constructor(path: string, message: string) {
    super(`${path}: ${message}`);
    this.name = "PlannerContractError";
    this.path = path;
  }
}

type UnknownRecord = Record<string, unknown>;

const POSITIONS = ["GKP", "DEF", "MID", "FWD"] as const;
const STATUSES = ["a", "d", "i", "s", "u", "n"] as const;
const CHIP_NAMES = ["wildcard", "freehit", "bboost", "3xc"] as const;
const CHIP_STATUSES = ["available", "used", "unavailable"] as const;
const STRATEGY_ASSUMPTIONS = [
  "fixed_current_prices",
  "four_transfer_deadlines_modelled",
  "maximum_two_transfers_at_each_provisional_deadline",
  "first_action_restricted_to_supplied_explicit_plans",
  "no_transfers_after_deadline_four_assumed",
  "recalculate_at_every_real_deadline",
] as const;
const LIMITATIONS: readonly PublicStateLimitation[] = [
  "state_is_locked_at_last_public_deadline",
  "current_free_transfers_not_public",
  "current_confirmed_transfers_not_public",
  "purchase_and_selling_prices_not_public",
  "next_deadline_chip_selection_not_public",
];
const CONFIRMATION_FIELDS: readonly ConfirmationField[] = [
  "current_squad_ids",
  "bank_tenths",
  "free_transfers",
  "player_prices",
  "chips",
  "chip_usage",
  "no_active_chip_confirmed",
];
const MAX_BANK_TENTHS = 1_000;
const ISO_UTC = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d{1,6})?Z$/;
const LOWER_HEX_64 = /^[0-9a-f]{64}$/;
const HEX_64 = /^[0-9a-fA-F]{64}$/;

function fail(path: string, message: string): never {
  throw new PlannerContractError(path, message);
}

function record(value: unknown, path: string): UnknownRecord {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    fail(path, "must be a JSON object");
  }
  return value as UnknownRecord;
}

function exactRecord(value: unknown, path: string, expected: readonly string[]): UnknownRecord {
  const result = record(value, path);
  const expectedSet = new Set(expected);
  const missing = expected.filter((key) => !Object.prototype.hasOwnProperty.call(result, key));
  const unknown = Object.keys(result).filter((key) => !expectedSet.has(key));
  if (missing.length || unknown.length) {
    const details = [
      missing.length ? `missing ${missing.join(", ")}` : "",
      unknown.length ? `unsupported ${unknown.join(", ")}` : "",
    ].filter(Boolean);
    fail(path, `fields do not match the contract (${details.join("; ")})`);
  }
  return result;
}

function array(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) fail(path, "must be a JSON array");
  return value;
}

function integer(value: unknown, path: string, minimum: number, maximum = Number.MAX_SAFE_INTEGER): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum || (value as number) > maximum) {
    fail(path, `must be an integer from ${minimum} to ${maximum}`);
  }
  return value as number;
}

function finiteNumber(value: unknown, path: string, minimum = -Number.MAX_VALUE, maximum = Number.MAX_VALUE): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum || value > maximum) {
    fail(path, "must be a finite number in the supported range");
  }
  return value;
}

function boolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") fail(path, "must be a boolean");
  return value;
}

function text(value: unknown, path: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    fail(path, "must be a non-empty string");
  }
  return value;
}

function oneOf<T extends string>(value: unknown, path: string, choices: readonly T[]): T {
  if (typeof value !== "string" || !choices.includes(value as T)) {
    fail(path, `must be one of: ${choices.join(", ")}`);
  }
  return value as T;
}

function literal<T extends string | number | boolean | null>(value: unknown, path: string, expected: T): T {
  if (value !== expected) fail(path, `must be ${JSON.stringify(expected)}`);
  return expected;
}

function nullable<T>(value: unknown, parser: (candidate: unknown) => T): T | null {
  return value === null ? null : parser(value);
}

function isoUtc(value: unknown, path: string): string {
  const result = text(value, path);
  const match = ISO_UTC.exec(result);
  if (!match) fail(path, "must be a UTC ISO-8601 timestamp ending in Z");
  const [, year, month, day, hour, minute, second] = match;
  const parsed = new Date(result);
  if (
    !Number.isFinite(parsed.getTime()) ||
    parsed.getUTCFullYear() !== Number(year) ||
    parsed.getUTCMonth() + 1 !== Number(month) ||
    parsed.getUTCDate() !== Number(day) ||
    parsed.getUTCHours() !== Number(hour) ||
    parsed.getUTCMinutes() !== Number(minute) ||
    parsed.getUTCSeconds() !== Number(second)
  ) {
    fail(path, "must be a real UTC timestamp");
  }
  return result;
}

function exactStringArray(value: unknown, path: string, expected: readonly string[]): void {
  const values = array(value, path);
  if (values.length !== expected.length || values.some((item, index) => item !== expected[index])) {
    fail(path, `must equal [${expected.join(", ")}]`);
  }
}

function uniqueIntegers(value: unknown, path: string, count: number, minimum = 1): number[] {
  const values = array(value, path);
  if (values.length !== count) fail(path, `must contain exactly ${count} values`);
  const parsed = values.map((item, index) => integer(item, `${path}[${index}]`, minimum));
  if (new Set(parsed).size !== parsed.length) fail(path, "must contain unique values");
  return parsed;
}

function sameValues(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value) => right.includes(value));
}

function sameJson(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) return true;
  if (Array.isArray(left) || Array.isArray(right)) {
    return Array.isArray(left) && Array.isArray(right) && left.length === right.length
      && left.every((value, index) => sameJson(value, right[index]));
  }
  if (left === null || right === null || typeof left !== "object" || typeof right !== "object") {
    return false;
  }
  const leftRecord = left as UnknownRecord;
  const rightRecord = right as UnknownRecord;
  const leftKeys = Object.keys(leftRecord).sort();
  const rightKeys = Object.keys(rightRecord).sort();
  return leftKeys.length === rightKeys.length
    && leftKeys.every((key, index) => key === rightKeys[index] && sameJson(leftRecord[key], rightRecord[key]));
}

function validateProjection(value: unknown, path: string): void {
  const row = exactRecord(value, path, ["offset_days", "projected_percent", "likelihood_code"]);
  integer(row.offset_days, `${path}.offset_days`, 0, 30);
  finiteNumber(row.projected_percent, `${path}.projected_percent`, -1000, 1000);
  integer(row.likelihood_code, `${path}.likelihood_code`, -10, 10);
}

function validateOfficialPriceSignal(value: unknown, path: string, expectedElementId?: number): void {
  const row = exactRecord(value, path, [
    "element_id", "now_cost_tenths", "cost_change_start_tenths", "cost_change_event_tenths",
    "selected_by_percent", "transfers_in_event", "transfers_out_event", "price_change_percent",
    "price_change_hourly_rate", "projections", "locked_until", "calibrating",
  ]);
  const elementId = integer(row.element_id, `${path}.element_id`, 1);
  if (expectedElementId !== undefined && elementId !== expectedElementId) {
    fail(`${path}.element_id`, "must match its player");
  }
  integer(row.now_cost_tenths, `${path}.now_cost_tenths`, 1, 500);
  integer(row.cost_change_start_tenths, `${path}.cost_change_start_tenths`, -500, 500);
  integer(row.cost_change_event_tenths, `${path}.cost_change_event_tenths`, -30, 30);
  finiteNumber(row.selected_by_percent, `${path}.selected_by_percent`, 0, 100);
  integer(row.transfers_in_event, `${path}.transfers_in_event`, 0);
  integer(row.transfers_out_event, `${path}.transfers_out_event`, 0);
  nullable(row.price_change_percent, (item) => finiteNumber(item, `${path}.price_change_percent`, -1000, 1000));
  nullable(row.price_change_hourly_rate, (item) => integer(item, `${path}.price_change_hourly_rate`, -100_000_000, 100_000_000));
  const projections = array(row.projections, `${path}.projections`);
  projections.forEach((projection, index) => validateProjection(projection, `${path}.projections[${index}]`));
  const offsets = projections.map((projection) => (projection as UnknownRecord).offset_days as number);
  if (new Set(offsets).size !== offsets.length || offsets.some((offset, index) => index > 0 && offset <= offsets[index - 1])) {
    fail(`${path}.projections`, "offsets must be unique and increasing");
  }
  nullable(row.locked_until, (item) => isoUtc(item, `${path}.locked_until`));
  boolean(row.calibrating, `${path}.calibrating`);
}

function validatePurchasePriceBasis(value: unknown, path: string, sourceEvent: number): void {
  const candidate = record(value, path);
  if (candidate.kind === "season_start_price") {
    exactRecord(candidate, path, ["kind"]);
    return;
  }
  const row = exactRecord(candidate, path, ["kind", "event", "confirmed_at"]);
  literal(row.kind, `${path}.kind`, "latest_public_transfer_in");
  const event = integer(row.event, `${path}.event`, 1, 38);
  if (event > sourceEvent) fail(`${path}.event`, "cannot be later than the public state event");
  nullable(row.confirmed_at, (item) => isoUtc(item, `${path}.confirmed_at`));
}

function sellingPrice(purchase: number, current: number): number {
  return current <= purchase ? current : purchase + Math.floor((current - purchase) / 2);
}

function validateSyncPick(value: unknown, path: string, sourceEvent: number): ManagerSyncPick {
  const row = exactRecord(value, path, [
    "element_id", "name", "club", "club_name", "club_id", "position", "lineup_position",
    "multiplier", "is_captain", "is_vice_captain", "current_price_tenths",
    "estimated_purchase_price_tenths", "estimated_selling_price_tenths",
    "purchase_price_basis", "official_price_signal",
  ]);
  const elementId = integer(row.element_id, `${path}.element_id`, 1);
  text(row.name, `${path}.name`);
  text(row.club, `${path}.club`);
  text(row.club_name, `${path}.club_name`);
  integer(row.club_id, `${path}.club_id`, 1);
  oneOf(row.position, `${path}.position`, POSITIONS);
  integer(row.lineup_position, `${path}.lineup_position`, 1, 15);
  integer(row.multiplier, `${path}.multiplier`, 0, 3);
  boolean(row.is_captain, `${path}.is_captain`);
  boolean(row.is_vice_captain, `${path}.is_vice_captain`);
  const current = integer(row.current_price_tenths, `${path}.current_price_tenths`, 1, 500);
  const purchase = integer(row.estimated_purchase_price_tenths, `${path}.estimated_purchase_price_tenths`, 1, 500);
  const selling = integer(row.estimated_selling_price_tenths, `${path}.estimated_selling_price_tenths`, 1, 500);
  validatePurchasePriceBasis(row.purchase_price_basis, `${path}.purchase_price_basis`, sourceEvent);
  const signal = row.official_price_signal;
  if (signal !== null) {
    validateOfficialPriceSignal(signal, `${path}.official_price_signal`, elementId);
    if ((signal as OfficialPriceSignal).now_cost_tenths !== current) fail(`${path}.official_price_signal.now_cost_tenths`, "must match current_price_tenths");
  }
  if (selling !== sellingPrice(purchase, current)) fail(`${path}.estimated_selling_price_tenths`, "does not follow the FPL half-profit rule");
  if ((row.purchase_price_basis as PurchasePriceBasis).kind === "season_start_price" && signal !== null && purchase !== current - (signal as OfficialPriceSignal).cost_change_start_tenths) {
    fail(`${path}.estimated_purchase_price_tenths`, "does not match the season-start price signal");
  }
  return row as unknown as ManagerSyncPick;
}

function validateSyncPlayerCatalogEntry(
  value: unknown,
  path: string,
): ManagerSyncPlayerCatalogEntry {
  const row = exactRecord(value, path, [
    "id", "display_name", "position", "team_id", "team_name", "current_price_tenths",
  ]);
  integer(row.id, `${path}.id`, 1);
  text(row.display_name, `${path}.display_name`);
  oneOf(row.position, `${path}.position`, POSITIONS);
  integer(row.team_id, `${path}.team_id`, 1);
  text(row.team_name, `${path}.team_name`);
  integer(row.current_price_tenths, `${path}.current_price_tenths`, 1, 500);
  return row as unknown as ManagerSyncPlayerCatalogEntry;
}

function validatePositionQuotas(players: readonly { position: PlayerPosition }[], path: string): void {
  const expected: Record<PlayerPosition, number> = { GKP: 2, DEF: 5, MID: 5, FWD: 3 };
  for (const position of POSITIONS) {
    if (players.filter((player) => player.position === position).length !== expected[position]) {
      fail(path, `must contain ${expected[position]} ${position} players`);
    }
  }
}

function validatePlayerPrices(value: unknown, path: string, squadIds: readonly number[]): PlayerPriceState[] {
  const rows = array(value, path);
  if (rows.length !== 15) fail(path, "must contain exactly 15 rows");
  const parsed = rows.map((value, index) => {
    const rowPath = `${path}[${index}]`;
    const row = exactRecord(value, rowPath, ["element_id", "purchase_price_tenths", "selling_price_tenths"]);
    integer(row.element_id, `${rowPath}.element_id`, 1);
    integer(row.purchase_price_tenths, `${rowPath}.purchase_price_tenths`, 1, 500);
    integer(row.selling_price_tenths, `${rowPath}.selling_price_tenths`, 1, 500);
    return row as unknown as PlayerPriceState;
  });
  const priceIds = parsed.map((row) => row.element_id);
  if (new Set(priceIds).size !== 15 || !sameValues(priceIds, squadIds)) {
    fail(path, "must contain one row for every squad player");
  }
  return parsed;
}

function chipWindowHalf(chip: ChipName, event: number): 1 | 2 | null {
  if (event >= 20 && event <= 38) return 2;
  if (event < 1 || event > 19) return null;
  if (chip === "wildcard" || chip === "freehit") return event >= 2 ? 1 : null;
  return 1;
}

function validateChipUsage(
  value: unknown,
  path: string,
  options: { beforeEvent?: number; throughEvent?: number; requireSorted?: boolean } = {},
): ChipUsage[] {
  const rows = array(value, path);
  if (rows.length > 8) fail(path, "must contain at most two legal uses of each chip");
  const parsed = rows.map((value, index) => {
    const rowPath = `${path}[${index}]`;
    const row = exactRecord(value, rowPath, ["name", "event"]);
    const name = oneOf(row.name, `${rowPath}.name`, CHIP_NAMES);
    const event = integer(row.event, `${rowPath}.event`, 1, 38);
    const half = chipWindowHalf(name, event);
    if (half === null) fail(rowPath, `${name} cannot be used in GW${event}`);
    if (options.beforeEvent !== undefined && event >= options.beforeEvent) {
      fail(`${rowPath}.event`, `must be earlier than event ${options.beforeEvent}`);
    }
    if (options.throughEvent !== undefined && event > options.throughEvent) {
      fail(`${rowPath}.event`, `cannot be later than event ${options.throughEvent}`);
    }
    return { name, event, half };
  });
  const sorted = [...parsed].sort((left, right) =>
    left.event - right.event || left.name.localeCompare(right.name),
  );
  if (
    options.requireSorted !== false &&
    parsed.some((row, index) => row.name !== sorted[index].name || row.event !== sorted[index].event)
  ) {
    fail(path, "must be sorted by event and chip name");
  }
  const events = new Set<number>();
  const chipHalves = new Set<string>();
  for (const row of parsed) {
    if (events.has(row.event)) fail(path, `cannot contain more than one chip in GW${row.event}`);
    const key = `${row.name}:${row.half}`;
    if (chipHalves.has(key)) fail(path, `cannot use ${row.name} twice in half ${row.half}`);
    events.add(row.event);
    chipHalves.add(key);
  }
  const freeHitEvents = new Set(parsed.filter((row) => row.name === "freehit").map((row) => row.event));
  if (freeHitEvents.has(19) && freeHitEvents.has(20)) {
    fail(path, "cannot use Free Hit in consecutive gameweeks 19 and 20");
  }
  return parsed.map(({ name, event }) => ({ name, event }));
}

function expectedChipState(chipUsage: readonly ChipUsage[], effectiveEvent: number): ChipState {
  const currentHalf = effectiveEvent <= 19 ? 1 : 2;
  const previousChip = chipUsage.find((row) => row.event === effectiveEvent - 1)?.name ?? null;
  return Object.fromEntries(CHIP_NAMES.map((name) => {
    const usedInHalf = chipUsage.some(
      (row) => row.name === name && chipWindowHalf(row.name, row.event) === currentHalf,
    );
    if (usedInHalf) return [name, "used"];
    const legalNow = chipWindowHalf(name, effectiveEvent) !== null && !(
      name === "freehit" && previousChip === "freehit" && effectiveEvent === 20
    );
    return [name, legalNow ? "available" : "unavailable"];
  })) as ChipState;
}

function validateChips(
  value: unknown,
  path: string,
  chipUsage: readonly ChipUsage[],
  effectiveEvent: number,
): ChipState {
  const chips = exactRecord(value, path, CHIP_NAMES);
  const expected = expectedChipState(chipUsage, effectiveEvent);
  for (const name of CHIP_NAMES) {
    const status = oneOf(chips[name], `${path}.${name}`, CHIP_STATUSES);
    if (status !== expected[name]) {
      fail(`${path}.${name}`, "is inconsistent with chip_usage and effective_event");
    }
  }
  return chips as unknown as ChipState;
}

export function parseManualManagerState(value: unknown): ManualManagerState {
  const path = "manager_state";
  const row = exactRecord(value, path, [
    "current_squad_ids", "bank_tenths", "free_transfers", "player_prices", "chips", "chip_usage",
    "no_active_chip_confirmed", "effective_event",
  ]);
  const squadIds = uniqueIntegers(row.current_squad_ids, `${path}.current_squad_ids`, 15);
  integer(row.bank_tenths, `${path}.bank_tenths`, 0, MAX_BANK_TENTHS);
  integer(row.free_transfers, `${path}.free_transfers`, 0, 5);
  validatePlayerPrices(row.player_prices, `${path}.player_prices`, squadIds);
  const effectiveEvent = integer(row.effective_event, `${path}.effective_event`, 1, 38);
  const chipUsage = validateChipUsage(row.chip_usage, `${path}.chip_usage`, {
    beforeEvent: effectiveEvent,
  });
  validateChips(row.chips, `${path}.chips`, chipUsage, effectiveEvent);
  boolean(row.no_active_chip_confirmed, `${path}.no_active_chip_confirmed`);
  return row as unknown as ManualManagerState;
}

export function parseManagerSyncResponse(value: unknown): ManagerSyncResponse {
  const root = exactRecord(value, "sync", [
    "schema_version", "generated_at", "manager", "target", "last_deadline_state",
    "price_signals", "player_catalog", "manual_state_template", "snapshot", "warnings",
  ]);
  literal(root.schema_version, "sync.schema_version", MANAGER_SYNC_SCHEMA_VERSION);
  const generatedAt = isoUtc(root.generated_at, "sync.generated_at");

  const manager = exactRecord(root.manager, "sync.manager", ["id", "team_name", "overall_points", "overall_rank"]);
  integer(manager.id, "sync.manager.id", 1);
  text(manager.team_name, "sync.manager.team_name");
  nullable(manager.overall_points, (item) => integer(item, "sync.manager.overall_points", 0));
  nullable(manager.overall_rank, (item) => integer(item, "sync.manager.overall_rank", 1));

  const target = exactRecord(root.target, "sync.target", ["event", "name", "deadline_time"]);
  const targetEvent = integer(target.event, "sync.target.event", 1, 38);
  text(target.name, "sync.target.name");
  const deadline = isoUtc(target.deadline_time, "sync.target.deadline_time");
  if (Date.parse(deadline) <= Date.parse(generatedAt)) fail("sync.target.deadline_time", "must be after generated_at");

  const state = exactRecord(root.last_deadline_state, "sync.last_deadline_state", [
    "schema_version", "state_kind", "event", "bank_tenths", "squad_value_tenths",
    "total_transfers_at_deadline", "event_transfers", "event_transfer_cost", "active_chip",
    "chip_usage", "limitations", "picks",
  ]);
  literal(state.schema_version, "sync.last_deadline_state.schema_version", PERSONAL_STATE_SCHEMA_VERSION);
  literal(state.state_kind, "sync.last_deadline_state.state_kind", "public_last_deadline");
  const sourceEvent = integer(state.event, "sync.last_deadline_state.event", 1, 38);
  if (sourceEvent >= targetEvent) fail("sync.last_deadline_state.event", "must be earlier than target.event");
  const bankTenths = integer(state.bank_tenths, "sync.last_deadline_state.bank_tenths", 0, MAX_BANK_TENTHS);
  integer(state.squad_value_tenths, "sync.last_deadline_state.squad_value_tenths", 1, 5000);
  integer(state.total_transfers_at_deadline, "sync.last_deadline_state.total_transfers_at_deadline", 0);
  integer(state.event_transfers, "sync.last_deadline_state.event_transfers", 0);
  integer(state.event_transfer_cost, "sync.last_deadline_state.event_transfer_cost", 0);
  const activeChip = nullable(state.active_chip, (item) => oneOf(item, "sync.last_deadline_state.active_chip", CHIP_NAMES));
  const chipUsage = validateChipUsage(
    state.chip_usage,
    "sync.last_deadline_state.chip_usage",
    { throughEvent: sourceEvent },
  );
  const chipsAtSourceEvent = chipUsage.filter((row) => row.event === sourceEvent);
  if (
    chipsAtSourceEvent.length !== (activeChip === null ? 0 : 1) ||
    (activeChip !== null && chipsAtSourceEvent[0]?.name !== activeChip)
  ) {
    fail("sync.last_deadline_state.active_chip", "must match chip_usage at the source event");
  }
  exactStringArray(state.limitations, "sync.last_deadline_state.limitations", LIMITATIONS);
  const pickValues = array(state.picks, "sync.last_deadline_state.picks");
  if (pickValues.length !== 15) fail("sync.last_deadline_state.picks", "must contain exactly 15 picks");
  const picks = pickValues.map((pick, index) => validateSyncPick(pick, `sync.last_deadline_state.picks[${index}]`, sourceEvent));
  if (new Set(picks.map((pick) => pick.element_id)).size !== 15) fail("sync.last_deadline_state.picks", "player ids must be unique");
  const positions = picks.map((pick) => pick.lineup_position).sort((a, b) => a - b);
  if (positions.some((position, index) => position !== index + 1)) fail("sync.last_deadline_state.picks", "lineup positions must be exactly 1 through 15");
  if (picks.filter((pick) => pick.is_captain).length !== 1 || picks.filter((pick) => pick.is_vice_captain).length !== 1) {
    fail("sync.last_deadline_state.picks", "must contain one captain and one vice-captain");
  }
  validatePositionQuotas(picks, "sync.last_deadline_state.picks");

  const playerCatalogValues = array(root.player_catalog, "sync.player_catalog");
  if (playerCatalogValues.length < 15 || playerCatalogValues.length > 1_000) {
    fail("sync.player_catalog", "must contain between 15 and 1000 players");
  }
  const playerCatalog = playerCatalogValues.map((player, index) =>
    validateSyncPlayerCatalogEntry(player, `sync.player_catalog[${index}]`)
  );
  const catalogIds = playerCatalog.map((player) => player.id);
  if (new Set(catalogIds).size !== catalogIds.length) {
    fail("sync.player_catalog", "player ids must be unique");
  }
  if (catalogIds.some((id, index) => index > 0 && id <= catalogIds[index - 1])) {
    fail("sync.player_catalog", "must be sorted by ascending player id");
  }
  const catalogById = new Map(playerCatalog.map((player) => [player.id, player]));
  const pickIdSet = new Set(picks.map((pick) => pick.element_id));
  for (const pick of picks) {
    const catalogPlayer = catalogById.get(pick.element_id);
    if (
      catalogPlayer === undefined ||
      catalogPlayer.display_name !== pick.name ||
      catalogPlayer.position !== pick.position ||
      catalogPlayer.team_id !== pick.club_id ||
      catalogPlayer.team_name !== pick.club_name ||
      catalogPlayer.current_price_tenths !== pick.current_price_tenths
    ) {
      fail("sync.player_catalog", `must match public pick ${pick.element_id}`);
    }
  }
  for (const position of POSITIONS) {
    if (!playerCatalog.some((player) =>
      player.position === position && !pickIdSet.has(player.id)
    )) {
      fail(
        "sync.player_catalog",
        `must contain at least one non-owned ${position} candidate`,
      );
    }
  }

  const priceSignals = exactRecord(root.price_signals, "sync.price_signals", [
    "schema_version", "available", "player_count", "price_change_deadlines", "warning",
  ]);
  literal(priceSignals.schema_version, "sync.price_signals.schema_version", PRICE_SIGNAL_SCHEMA_VERSION);
  const priceSignalsAvailable = boolean(priceSignals.available, "sync.price_signals.available");
  const priceSignalPlayerCount = integer(priceSignals.player_count, "sync.price_signals.player_count", 0);
  nullable(priceSignals.warning, (item) => text(item, "sync.price_signals.warning"));
  const deadlines = array(priceSignals.price_change_deadlines, "sync.price_signals.price_change_deadlines").map((item, index) => isoUtc(item, `sync.price_signals.price_change_deadlines[${index}]`));
  if (new Set(deadlines).size !== deadlines.length || deadlines.some((item, index) => index > 0 && Date.parse(item) <= Date.parse(deadlines[index - 1]))) {
    fail("sync.price_signals.price_change_deadlines", "must be unique and increasing");
  }
  if (!priceSignalsAvailable && priceSignals.warning === null) fail("sync.price_signals.warning", "must explain unavailable optional price signals");
  if (priceSignalsAvailable && priceSignals.warning !== null) fail("sync.price_signals.warning", "must be null when optional price signals are available");
  if (priceSignalsAvailable !== (priceSignalPlayerCount > 0)) fail("sync.price_signals.player_count", "must match price_signals.available");
  if (picks.some((pick) => (pick.official_price_signal !== null) !== priceSignalsAvailable)) fail("sync.last_deadline_state.picks", "price-signal availability must match price_signals.available");

  const template = exactRecord(root.manual_state_template, "sync.manual_state_template", [
    "state", "confirmation_required", "fields_requiring_confirmation", "free_transfers_default_reason",
  ]);
  const manualState = parseManualManagerState(template.state);
  const pickIds = picks.map((pick) => pick.element_id);
  if (manualState.current_squad_ids.some((id, index) => id !== pickIds[index])) fail("sync.manual_state_template.state.current_squad_ids", "must match public picks in lineup order");
  if (manualState.bank_tenths !== bankTenths) fail("sync.manual_state_template.state.bank_tenths", "must match last_deadline_state.bank_tenths");
  if (manualState.free_transfers !== 1) fail("sync.manual_state_template.state.free_transfers", "must use the public-data default of 1");
  if (manualState.effective_event !== targetEvent) fail("sync.manual_state_template.state.effective_event", "must match target.event");
  if (!sameJson(manualState.chip_usage, chipUsage)) {
    fail("sync.manual_state_template.state.chip_usage", "must match last_deadline_state.chip_usage");
  }
  if (manualState.no_active_chip_confirmed) fail("sync.manual_state_template.state.no_active_chip_confirmed", "must be false until confirmed manually");
  const byPickId = new Map(picks.map((pick) => [pick.element_id, pick]));
  for (const price of manualState.player_prices) {
    const pick = byPickId.get(price.element_id)!;
    if (price.purchase_price_tenths !== pick.estimated_purchase_price_tenths || price.selling_price_tenths !== pick.estimated_selling_price_tenths) {
      fail("sync.manual_state_template.state.player_prices", "must match the public price estimates");
    }
  }
  literal(template.confirmation_required, "sync.manual_state_template.confirmation_required", true);
  exactStringArray(template.fields_requiring_confirmation, "sync.manual_state_template.fields_requiring_confirmation", CONFIRMATION_FIELDS);
  literal(template.free_transfers_default_reason, "sync.manual_state_template.free_transfers_default_reason", "current_free_transfers_are_not_public");

  const snapshot = exactRecord(root.snapshot, "sync.snapshot", ["schema_version", "observed_at", "source", "checksum_sha256", "persisted"]);
  literal(snapshot.schema_version, "sync.snapshot.schema_version", SNAPSHOT_SCHEMA_VERSION);
  const observedAt = isoUtc(snapshot.observed_at, "sync.snapshot.observed_at");
  if (Date.parse(observedAt) !== Date.parse(generatedAt)) fail("sync.snapshot.observed_at", "must identify the generated observation");
  literal(snapshot.source, "sync.snapshot.source", "fpl_public_last_deadline");
  if (typeof snapshot.checksum_sha256 !== "string" || !LOWER_HEX_64.test(snapshot.checksum_sha256)) fail("sync.snapshot.checksum_sha256", "must be a lowercase SHA-256 checksum");
  literal(snapshot.persisted, "sync.snapshot.persisted", false);

  const warnings = array(root.warnings, "sync.warnings");
  if (warnings.length < 1 || warnings.length > 2) fail("sync.warnings", "must contain one or two known warnings");
  const warningCodes = warnings.map((warning, index) => {
    const path = `sync.warnings[${index}]`;
    const row = exactRecord(warning, path, ["code", "message"]);
    const code = oneOf(row.code, `${path}.code`, ["last_deadline_state_requires_confirmation", "free_hit_squad_is_temporary"] as const);
    text(row.message, `${path}.message`);
    return code;
  });
  if (new Set(warningCodes).size !== warningCodes.length || warningCodes[0] !== "last_deadline_state_requires_confirmation") fail("sync.warnings", "must begin with the required confirmation warning without duplicates");
  const expectsFreeHitWarning = activeChip === "freehit";
  if (warningCodes.includes("free_hit_squad_is_temporary") !== expectsFreeHitWarning) fail("sync.warnings", "free-hit warning does not match active_chip");

  return root as unknown as ManagerSyncResponse;
}

export function isManagerSyncResponse(value: unknown): value is ManagerSyncResponse {
  try {
    parseManagerSyncResponse(value);
    return true;
  } catch {
    return false;
  }
}

export function parseManagerId(value: string): number {
  if (typeof value !== "string") fail("manager_id", "must be text");
  const normalized = value.trim();
  if (!/^[1-9]\d*$/.test(normalized)) fail("manager_id", "must be a positive whole number");
  const parsed = Number(normalized);
  if (!Number.isSafeInteger(parsed)) fail("manager_id", "is too large");
  return parsed;
}

export function parseBankTenths(value: string): number {
  if (typeof value !== "string") fail("bank_tenths", "must be text");
  const match = /^(?:£\s*)?(\d+)(?:[.,](\d))?\s*(?:m)?$/i.exec(value.trim());
  if (!match) fail("bank_tenths", "use whole £m or one decimal, for example 1,2 or £0.1m");
  const whole = Number(match[1]);
  const result = whole * 10 + Number(match[2] ?? "0");
  if (!Number.isSafeInteger(result) || result > MAX_BANK_TENTHS) fail("bank_tenths", "must be between £0.0m and £100.0m in £0.1m steps");
  return result;
}

export function parseFreeTransfers(value: string): number {
  if (typeof value !== "string") fail("free_transfers", "must be text");
  const normalized = value.trim();
  if (!/^[0-5]$/.test(normalized)) fail("free_transfers", "must be a whole number from 0 to 5");
  return Number(normalized);
}

function cloneManualState(state: ManualManagerState): ManualManagerState {
  return {
    current_squad_ids: [...state.current_squad_ids],
    bank_tenths: state.bank_tenths,
    free_transfers: state.free_transfers,
    player_prices: state.player_prices.map((price) => ({ ...price })),
    chips: { ...state.chips },
    chip_usage: state.chip_usage.map((usage) => ({ ...usage })),
    no_active_chip_confirmed: state.no_active_chip_confirmed,
    effective_event: state.effective_event,
  };
}

export function serializePlannerRequest(input: PlannerRequestInput): PlannerComputeRequest {
  const source = record(input, "request");
  const managerId = integer(source.manager_id, "request.manager_id", 1);
  const sourceEvent = integer(source.source_event, "request.source_event", 1, 38);
  if (typeof source.state_fingerprint !== "string" || !LOWER_HEX_64.test(source.state_fingerprint)) fail("request.state_fingerprint", "must be a lowercase SHA-256 checksum");
  const managerState = parseManualManagerState(source.manager_state);
  if (!managerState.no_active_chip_confirmed) fail("request.manager_state.no_active_chip_confirmed", "must be confirmed before planning");
  if (managerState.effective_event !== null && sourceEvent >= managerState.effective_event) fail("request.source_event", "must be earlier than manager_state.effective_event");
  const horizon = integer(source.horizon, "request.horizon", 1, 5);
  const includeDoubtful = boolean(source.include_doubtful, "request.include_doubtful");
  const forecastVersion = oneOf(source.forecast_version, "request.forecast_version", ["v2", "legacy"] as const);
  return {
    manager_id: managerId,
    source_event: sourceEvent,
    state_fingerprint: source.state_fingerprint,
    manager_state: cloneManualState(managerState),
    horizon,
    include_doubtful: includeDoubtful,
    use_solio: false,
    forecast_version: forecastVersion,
  };
}

export const buildPlannerRequest = serializePlannerRequest;

export function stringifyPlannerRequest(input: PlannerRequestInput): string {
  return JSON.stringify(serializePlannerRequest(input));
}

function validatePlayerReference(value: unknown, path: string): PlannerPlayerReference {
  const row = exactRecord(value, path, ["id", "name", "team", "position", "price_tenths", "status", "price_signal"]);
  const id = integer(row.id, `${path}.id`, 1);
  text(row.name, `${path}.name`);
  text(row.team, `${path}.team`);
  oneOf(row.position, `${path}.position`, POSITIONS);
  const price = integer(row.price_tenths, `${path}.price_tenths`, 1, 500);
  oneOf(row.status, `${path}.status`, STATUSES);
  if (row.price_signal !== null) {
    validateOfficialPriceSignal(row.price_signal, `${path}.price_signal`, id);
    if ((row.price_signal as OfficialPriceSignal).now_cost_tenths !== price) fail(`${path}.price_signal.now_cost_tenths`, "must match price_tenths");
  }
  return row as unknown as PlannerPlayerReference;
}

function validateOwnedPlayerReference(value: unknown, path: string): PlannerOwnedPlayerReference {
  const row = exactRecord(value, path, [
    "id", "name", "team", "position", "price_tenths", "status", "price_signal",
    "purchase_price_tenths", "selling_price_tenths",
  ]);
  const reference = validatePlayerReference({
    id: row.id,
    name: row.name,
    team: row.team,
    position: row.position,
    price_tenths: row.price_tenths,
    status: row.status,
    price_signal: row.price_signal,
  }, path);
  const purchase = integer(row.purchase_price_tenths, `${path}.purchase_price_tenths`, 1, 500);
  const selling = integer(row.selling_price_tenths, `${path}.selling_price_tenths`, 1, 500);
  if (selling !== sellingPrice(purchase, reference.price_tenths)) {
    fail(`${path}.selling_price_tenths`, "does not follow the FPL half-profit rule");
  }
  return { ...reference, purchase_price_tenths: purchase, selling_price_tenths: selling };
}

function validateFormation(gameweek: UnknownRecord, path: string, squad: Map<number, PlannerPlayerReference>): void {
  const starters = uniqueIntegers(gameweek.starting_ids, `${path}.starting_ids`, 11);
  for (const id of starters) if (!squad.has(id)) fail(`${path}.starting_ids`, `contains player ${id} outside action squad`);
  const captain = integer(gameweek.captain_id, `${path}.captain_id`, 1);
  if (!starters.includes(captain)) fail(`${path}.captain_id`, "must be a starter");
  const formation = text(gameweek.formation, `${path}.formation`);
  const counts = { GKP: 0, DEF: 0, MID: 0, FWD: 0 } satisfies Record<PlayerPosition, number>;
  for (const id of starters) counts[squad.get(id)!.position] += 1;
  if (counts.GKP !== 1 || formation !== `${counts.DEF}-${counts.MID}-${counts.FWD}`) fail(`${path}.formation`, "must match the starting-player positions");
  if (counts.DEF < 3 || counts.DEF > 5 || counts.MID < 2 || counts.MID > 5 || counts.FWD < 1 || counts.FWD > 3) fail(`${path}.starting_ids`, "does not form a legal FPL XI");
  integer(gameweek.gameweek, `${path}.gameweek`, 1, 38);
  finiteNumber(gameweek.projected_points, `${path}.projected_points`, 0);
}

function validateAction(
  value: unknown,
  path: string,
  confirmed: Map<number, PlannerPlayerReference>,
  confirmedPrices: ReadonlyMap<number, { purchase: number; selling: number }>,
  bank: number,
  freeTransfers: number,
  targetEvent: number,
): PlannerAction {
  const row = exactRecord(value, path, [
    "kind", "transfer_count", "transfers", "squad_ids", "gameweeks", "bank_before_tenths",
    "bank_after_tenths", "free_transfers_before", "free_transfers_next_gameweek", "hit_points",
    "projected_points", "banked_ft_value_points", "decision_value_points", "net_points_vs_roll",
    "decision_value_vs_roll", "explanation",
  ]);
  const kind = oneOf(row.kind, `${path}.kind`, ["roll", "transfer", "hit"] as const);
  const transferCount = integer(row.transfer_count, `${path}.transfer_count`, 0, 5);
  const transferValues = array(row.transfers, `${path}.transfers`);
  if (transferValues.length !== transferCount) fail(`${path}.transfers`, "length must match transfer_count");
  const actionSquad = new Map(confirmed);
  const outIds = new Set<number>();
  const inIds = new Set<number>();
  let bankDelta = 0;
  const transfers = transferValues.map((value, index) => {
    const transferPath = `${path}.transfers[${index}]`;
    const transfer = exactRecord(value, transferPath, [
      "out_id", "in_id", "position", "out_purchase_price_tenths", "out_current_price_tenths",
      "out_selling_price_tenths", "in_price_tenths", "out", "in",
    ]);
    const outId = integer(transfer.out_id, `${transferPath}.out_id`, 1);
    const inId = integer(transfer.in_id, `${transferPath}.in_id`, 1);
    if (outId === inId || outIds.has(outId) || inIds.has(inId)) fail(transferPath, "must contain distinct, non-repeated players");
    if (!confirmed.has(outId) || confirmed.has(inId)) fail(transferPath, "must replace an owned player with an unowned player");
    outIds.add(outId);
    inIds.add(inId);
    const position = oneOf(transfer.position, `${transferPath}.position`, POSITIONS);
    const purchase = integer(transfer.out_purchase_price_tenths, `${transferPath}.out_purchase_price_tenths`, 1, 500);
    const current = integer(transfer.out_current_price_tenths, `${transferPath}.out_current_price_tenths`, 1, 500);
    const selling = integer(transfer.out_selling_price_tenths, `${transferPath}.out_selling_price_tenths`, 1, 500);
    const inPrice = integer(transfer.in_price_tenths, `${transferPath}.in_price_tenths`, 1, 500);
    const out = validatePlayerReference(transfer.out, `${transferPath}.out`);
    const incoming = validatePlayerReference(transfer.in, `${transferPath}.in`);
    if (out.id !== outId || incoming.id !== inId || out.position !== position || incoming.position !== position) fail(transferPath, "enriched players must match the transfer ids and position");
    if (out.price_tenths !== current || incoming.price_tenths !== inPrice || selling !== sellingPrice(purchase, current)) fail(transferPath, "price fields are inconsistent");
    if (!sameJson(out, confirmed.get(outId))) fail(`${transferPath}.out`, "must match the confirmed player reference");
    const confirmedPrice = confirmedPrices.get(outId);
    if (
      confirmedPrice === undefined ||
      purchase !== confirmedPrice.purchase ||
      selling !== confirmedPrice.selling
    ) {
      fail(transferPath, "must use the confirmed purchase and selling prices");
    }
    actionSquad.delete(outId);
    actionSquad.set(inId, incoming);
    bankDelta += selling - inPrice;
    return transfer as unknown as EnrichedTransfer;
  });
  const squadIds = uniqueIntegers(row.squad_ids, `${path}.squad_ids`, 15);
  if (!sameValues(squadIds, [...actionSquad.keys()])) fail(`${path}.squad_ids`, "must equal the confirmed squad after transfers");
  validatePositionQuotas([...actionSquad.values()], `${path}.squad_ids`);
  const bankBefore = integer(row.bank_before_tenths, `${path}.bank_before_tenths`, 0, MAX_BANK_TENTHS);
  const bankAfter = integer(row.bank_after_tenths, `${path}.bank_after_tenths`, 0, MAX_BANK_TENTHS);
  if (bankBefore !== bank || bankAfter !== bankBefore + bankDelta) fail(`${path}.bank_after_tenths`, "does not reconcile with confirmed bank and transfer prices");
  const ftBefore = integer(row.free_transfers_before, `${path}.free_transfers_before`, 0, 5);
  const ftNext = integer(row.free_transfers_next_gameweek, `${path}.free_transfers_next_gameweek`, 1, 5);
  if (ftBefore !== freeTransfers || ftNext !== Math.min(5, Math.max(0, ftBefore - transferCount) + 1)) fail(`${path}.free_transfers_next_gameweek`, "does not follow the rolling free-transfer rule");
  const hitPoints = integer(row.hit_points, `${path}.hit_points`, 0, 16);
  if (hitPoints !== Math.max(0, transferCount - ftBefore) * 4) fail(`${path}.hit_points`, "does not match transfer_count and free transfers");
  const expectedKind: PlannerActionKind = transferCount === 0 ? "roll" : hitPoints > 0 ? "hit" : "transfer";
  if (kind !== expectedKind) fail(`${path}.kind`, `must be ${expectedKind} for this action`);
  if (kind === "roll" && (transfers.length !== 0 || bankDelta !== 0)) fail(path, "roll must not include transfers");
  const gameweeks = array(row.gameweeks, `${path}.gameweeks`);
  if (gameweeks.length < 1 || gameweeks.length > 5) fail(`${path}.gameweeks`, "must contain one to five gameweeks");
  gameweeks.forEach((gameweek, index) => {
    const gameweekPath = `${path}.gameweeks[${index}]`;
    const gw = exactRecord(gameweek, gameweekPath, ["gameweek", "starting_ids", "captain_id", "formation", "projected_points"]);
    validateFormation(gw, gameweekPath, actionSquad);
    const event = gw.gameweek as number;
    if (index === 0 && event !== targetEvent) fail(`${gameweekPath}.gameweek`, `must begin at target event ${targetEvent}`);
    if (index > 0 && event <= ((gameweeks[index - 1] as UnknownRecord).gameweek as number)) {
      fail(`${gameweekPath}.gameweek`, "must be later than the preceding gameweek");
    }
  });
  finiteNumber(row.projected_points, `${path}.projected_points`, 0);
  finiteNumber(row.banked_ft_value_points, `${path}.banked_ft_value_points`, 0);
  finiteNumber(row.decision_value_points, `${path}.decision_value_points`);
  finiteNumber(row.net_points_vs_roll, `${path}.net_points_vs_roll`);
  finiteNumber(row.decision_value_vs_roll, `${path}.decision_value_vs_roll`);
  text(row.explanation, `${path}.explanation`);
  return row as unknown as PlannerAction;
}

function resultingSquad(
  initial: ReadonlyMap<number, PlannerPlayerReference>,
  transfers: readonly EnrichedTransfer[],
): Map<number, PlannerPlayerReference> {
  const result = new Map(initial);
  for (const transfer of transfers) {
    result.delete(transfer.out_id);
    result.set(transfer.in_id, transfer.in);
  }
  return result;
}

function closeEnough(left: number, right: number): boolean {
  return Math.abs(left - right) <= 0.005;
}

function reconcileNumber(actual: number, expected: number, path: string): void {
  if (!closeEnough(actual, expected)) {
    fail(path, `does not reconcile with the sequential plan (${actual} != ${expected})`);
  }
}

type ValidatedSequentialStep = {
  step: PlannerSequentialStep;
  squad: Map<number, PlannerPlayerReference>;
};

function validateSequentialStep(
  value: unknown,
  path: string,
  options: {
    deadlineOffset: number;
    provisional: boolean;
    targetEvent: number;
    gameweekWindow: readonly number[];
    gameweekWeights: readonly number[];
    priorSquad: ReadonlyMap<number, PlannerPlayerReference>;
    bankBefore: number;
    freeTransfersBefore: number;
    fixedPurchasePrices?: ReadonlyMap<number, number>;
    maximumTransfers?: number;
  },
): ValidatedSequentialStep {
  const row = exactRecord(value, path, [
    "deadline_offset", "provisional", "target_event", "kind", "transfer_count",
    "transfers", "squad_ids", "gameweeks", "bank_before_tenths", "bank_after_tenths",
    "free_transfers_before", "free_transfers_next_gameweek", "hit_points",
    "weighted_projected_points", "weighted_hit_cost_points",
  ]);
  literal(row.deadline_offset, `${path}.deadline_offset`, options.deadlineOffset);
  literal(row.provisional, `${path}.provisional`, options.provisional);
  literal(row.target_event, `${path}.target_event`, options.targetEvent);

  const kind = oneOf(row.kind, `${path}.kind`, ["roll", "transfer", "hit"] as const);
  const transferCount = integer(row.transfer_count, `${path}.transfer_count`, 0, 5);
  if (transferCount > (options.maximumTransfers ?? Math.max(2, options.freeTransfersBefore))) {
    fail(`${path}.transfer_count`, "exceeds the bounded immediate-transfer search");
  }
  const transferValues = array(row.transfers, `${path}.transfers`);
  if (transferValues.length !== transferCount) {
    fail(`${path}.transfers`, "length must match transfer_count");
  }

  const stepSquad = new Map(options.priorSquad);
  const outIds = new Set<number>();
  const inIds = new Set<number>();
  let bankDelta = 0;
  const transfers = transferValues.map((candidate, index) => {
    const transferPath = `${path}.transfers[${index}]`;
    const transfer = exactRecord(candidate, transferPath, [
      "out_id", "in_id", "position", "out_purchase_price_tenths", "out_current_price_tenths",
      "out_selling_price_tenths", "in_price_tenths", "out", "in",
    ]);
    const outId = integer(transfer.out_id, `${transferPath}.out_id`, 1);
    const inId = integer(transfer.in_id, `${transferPath}.in_id`, 1);
    if (
      outId === inId || outIds.has(outId) || inIds.has(inId) ||
      !options.priorSquad.has(outId) || options.priorSquad.has(inId)
    ) {
      fail(transferPath, "must replace one owned player with one distinct unowned player");
    }
    outIds.add(outId);
    inIds.add(inId);
    const position = oneOf(transfer.position, `${transferPath}.position`, POSITIONS);
    const purchase = integer(
      transfer.out_purchase_price_tenths,
      `${transferPath}.out_purchase_price_tenths`,
      1,
      500,
    );
    const current = integer(
      transfer.out_current_price_tenths,
      `${transferPath}.out_current_price_tenths`,
      1,
      500,
    );
    const selling = integer(
      transfer.out_selling_price_tenths,
      `${transferPath}.out_selling_price_tenths`,
      1,
      500,
    );
    const inPrice = integer(transfer.in_price_tenths, `${transferPath}.in_price_tenths`, 1, 500);
    const outgoing = validatePlayerReference(transfer.out, `${transferPath}.out`);
    const incoming = validatePlayerReference(transfer.in, `${transferPath}.in`);
    if (
      outgoing.id !== outId || incoming.id !== inId || outgoing.position !== position ||
      incoming.position !== position || outgoing.price_tenths !== current ||
      incoming.price_tenths !== inPrice
    ) {
      fail(transferPath, "enriched players must match the transfer ids, position and fixed prices");
    }
    if (!sameJson(outgoing, options.priorSquad.get(outId))) {
      fail(`${transferPath}.out`, "must match the player in the preceding chained squad");
    }
    if (selling !== sellingPrice(purchase, current)) {
      fail(`${transferPath}.out_selling_price_tenths`, "does not follow the FPL half-profit rule");
    }
    const fixedPurchasePrice = options.fixedPurchasePrices?.get(outId);
    if (fixedPurchasePrice !== undefined && purchase !== fixedPurchasePrice) {
      fail(`${transferPath}.out_purchase_price_tenths`, "must keep the first-step purchase price");
    }
    stepSquad.delete(outId);
    stepSquad.set(inId, incoming);
    bankDelta += selling - inPrice;
    return transfer as unknown as EnrichedTransfer;
  });

  const squadIds = uniqueIntegers(row.squad_ids, `${path}.squad_ids`, 15);
  if (!sameValues(squadIds, [...stepSquad.keys()])) {
    fail(`${path}.squad_ids`, "must equal the preceding squad after this step's transfers");
  }
  validatePositionQuotas([...stepSquad.values()], `${path}.squad_ids`);

  const bankBefore = integer(row.bank_before_tenths, `${path}.bank_before_tenths`, 0, MAX_BANK_TENTHS);
  const bankAfter = integer(row.bank_after_tenths, `${path}.bank_after_tenths`, 0, MAX_BANK_TENTHS);
  if (bankBefore !== options.bankBefore || bankAfter !== bankBefore + bankDelta) {
    fail(`${path}.bank_after_tenths`, "does not reconcile sequentially with bank and transfer prices");
  }
  const freeTransfersBefore = integer(
    row.free_transfers_before,
    `${path}.free_transfers_before`,
    options.provisional ? 1 : 0,
    5,
  );
  const freeTransfersNext = integer(
    row.free_transfers_next_gameweek,
    `${path}.free_transfers_next_gameweek`,
    1,
    5,
  );
  const expectedFreeTransfersNext = Math.min(
    5,
    Math.max(0, freeTransfersBefore - transferCount) + 1,
  );
  if (
    freeTransfersBefore !== options.freeTransfersBefore ||
    freeTransfersNext !== expectedFreeTransfersNext
  ) {
    fail(`${path}.free_transfers_next_gameweek`, "does not follow the sequential free-transfer rule");
  }
  const hitPoints = integer(row.hit_points, `${path}.hit_points`, 0, 16);
  if (hitPoints !== Math.max(0, transferCount - freeTransfersBefore) * 4) {
    fail(`${path}.hit_points`, "does not match transfer_count and sequential free transfers");
  }
  const expectedKind: PlannerActionKind = transferCount === 0
    ? "roll"
    : hitPoints > 0
      ? "hit"
      : "transfer";
  if (kind !== expectedKind) fail(`${path}.kind`, `must be ${expectedKind} for this step`);

  const gameweeks = array(row.gameweeks, `${path}.gameweeks`);
  if (gameweeks.length !== options.gameweekWindow.length) {
    fail(`${path}.gameweeks`, "must cover exactly its part of the planner horizon");
  }
  gameweeks.forEach((gameweek, index) => {
    const gameweekPath = `${path}.gameweeks[${index}]`;
    const parsed = exactRecord(gameweek, gameweekPath, [
      "gameweek", "starting_ids", "captain_id", "formation", "projected_points",
    ]);
    validateFormation(parsed, gameweekPath, stepSquad);
    literal(parsed.gameweek, `${gameweekPath}.gameweek`, options.gameweekWindow[index]);
  });

  const weightedProjectedPoints = finiteNumber(
    row.weighted_projected_points,
    `${path}.weighted_projected_points`,
    0,
  );
  const expectedWeightedProjectedPoints = gameweeks.reduce<number>(
    (total, gameweek, index) => total
      + (gameweek as PlannerActionGameweek).projected_points * options.gameweekWeights[index],
    0,
  );
  reconcileNumber(
    weightedProjectedPoints,
    expectedWeightedProjectedPoints,
    `${path}.weighted_projected_points`,
  );
  const weightedHitCostPoints = finiteNumber(
    row.weighted_hit_cost_points,
    `${path}.weighted_hit_cost_points`,
    0,
  );
  reconcileNumber(
    weightedHitCostPoints,
    hitPoints * options.gameweekWeights[0],
    `${path}.weighted_hit_cost_points`,
  );

  return {
    step: {
      ...row,
      transfers,
    } as unknown as PlannerSequentialStep,
    squad: stepSquad,
  };
}

function validateSequentialPlan(
  value: unknown,
  options: {
    horizon: number;
    targetEvent: number;
    confirmedSquad: ReadonlyMap<number, PlannerPlayerReference>;
    confirmedPurchasePrices: ReadonlyMap<number, number>;
    bank: number;
    freeTransfers: number;
    bestAction: PlannerAction;
    alternatives: readonly PlannerAction[];
    rollFtValuePoints: number;
  },
): PlannerSequentialPlan {
  const root = exactRecord(value, "planner.sequential", [
    "horizon", "gw_weights", "first_step_candidate_count", "first_step_search",
    "future_price_assumption", "solver_proven_optimal_within_bounds", "globally_optimal",
    "best_sequence",
  ]);
  const horizon = integer(root.horizon, "planner.sequential.horizon", 2, 5);
  if (horizon !== options.horizon) {
    fail("planner.sequential.horizon", "must match the immediate planner horizon");
  }
  const weights = array(root.gw_weights, "planner.sequential.gw_weights").map((weight, index) =>
    finiteNumber(weight, `planner.sequential.gw_weights[${index}]`, 0.000_001, 1),
  );
  if (weights.length !== horizon) {
    fail("planner.sequential.gw_weights", "length must match horizon");
  }
  if (!closeEnough(weights[0], 1)) {
    fail("planner.sequential.gw_weights[0]", "must weight the current deadline as 1");
  }
  if (weights.some((weight, index) => index > 0 && weight > weights[index - 1])) {
    fail("planner.sequential.gw_weights", "must be non-increasing across the horizon");
  }
  const candidateCount = integer(
    root.first_step_candidate_count,
    "planner.sequential.first_step_candidate_count",
    1,
    16,
  );
  if (candidateCount !== options.alternatives.length + 1) {
    fail(
      "planner.sequential.first_step_candidate_count",
      "must equal the visible best action plus alternatives",
    );
  }
  literal(
    root.first_step_search,
    "planner.sequential.first_step_search",
    "explicit_bounded_transfer_plans",
  );
  literal(
    root.future_price_assumption,
    "planner.sequential.future_price_assumption",
    "fixed_current_prices",
  );
  literal(
    root.solver_proven_optimal_within_bounds,
    "planner.sequential.solver_proven_optimal_within_bounds",
    true,
  );
  literal(root.globally_optimal, "planner.sequential.globally_optimal", false);

  const sequence = exactRecord(root.best_sequence, "planner.sequential.best_sequence", [
    "first_action", "steps", "weighted_projected_points", "total_hit_points",
    "weighted_hit_cost_points", "terminal_banked_ft_value_points", "decision_value_points",
  ]);
  const firstAction = exactRecord(
    sequence.first_action,
    "planner.sequential.best_sequence.first_action",
    ["source", "alternative_index"],
  );
  const source = oneOf(
    firstAction.source,
    "planner.sequential.best_sequence.first_action.source",
    ["best_action", "alternative"] as const,
  );
  let alternativeIndex: number | null = null;
  let selectedAction = options.bestAction;
  if (source === "best_action") {
    literal(
      firstAction.alternative_index,
      "planner.sequential.best_sequence.first_action.alternative_index",
      null,
    );
  } else {
    alternativeIndex = integer(
      firstAction.alternative_index,
      "planner.sequential.best_sequence.first_action.alternative_index",
      0,
      options.alternatives.length - 1,
    );
    selectedAction = options.alternatives[alternativeIndex];
    if (candidateCount < alternativeIndex + 2) {
      fail(
        "planner.sequential.first_step_candidate_count",
        "cannot be smaller than the selected existing action set",
      );
    }
  }

  const stepValues = array(sequence.steps, "planner.sequential.best_sequence.steps");
  if (stepValues.length !== 2) {
    fail("planner.sequential.best_sequence.steps", "must contain exactly two deadline steps");
  }
  const gameweekWindow = options.bestAction.gameweeks.map((gameweek) => gameweek.gameweek);
  const fixedPurchasePrices = new Map(options.confirmedPurchasePrices);
  for (const transfer of selectedAction.transfers) {
    fixedPurchasePrices.set(transfer.in_id, transfer.in_price_tenths);
  }
  const first = validateSequentialStep(
    stepValues[0],
    "planner.sequential.best_sequence.steps[0]",
    {
      deadlineOffset: 1,
      provisional: false,
      targetEvent: options.targetEvent,
      gameweekWindow: gameweekWindow.slice(0, 1),
      gameweekWeights: weights.slice(0, 1),
      priorSquad: options.confirmedSquad,
      bankBefore: options.bank,
      freeTransfersBefore: options.freeTransfers,
    },
  );
  const firstStepMatchesSelectedAction = (
    first.step.kind === selectedAction.kind &&
    first.step.transfer_count === selectedAction.transfer_count &&
    sameJson(first.step.transfers, selectedAction.transfers) &&
    sameJson(first.step.squad_ids, selectedAction.squad_ids) &&
    first.step.bank_before_tenths === selectedAction.bank_before_tenths &&
    first.step.bank_after_tenths === selectedAction.bank_after_tenths &&
    first.step.free_transfers_before === selectedAction.free_transfers_before &&
    first.step.free_transfers_next_gameweek === selectedAction.free_transfers_next_gameweek &&
    first.step.hit_points === selectedAction.hit_points &&
    sameJson(first.step.gameweeks[0], selectedAction.gameweeks[0])
  );
  if (!firstStepMatchesSelectedAction) {
    fail(
      "planner.sequential.best_sequence.steps[0]",
      "must exactly match the selected existing action at the first deadline",
    );
  }

  const secondTargetEvent = gameweekWindow[1];
  const second = validateSequentialStep(
    stepValues[1],
    "planner.sequential.best_sequence.steps[1]",
    {
      deadlineOffset: 2,
      provisional: true,
      targetEvent: secondTargetEvent,
      gameweekWindow: gameweekWindow.slice(1),
      gameweekWeights: weights.slice(1),
      priorSquad: first.squad,
      bankBefore: first.step.bank_after_tenths,
      freeTransfersBefore: first.step.free_transfers_next_gameweek,
      fixedPurchasePrices,
    },
  );

  const weightedProjectedPoints = finiteNumber(
    sequence.weighted_projected_points,
    "planner.sequential.best_sequence.weighted_projected_points",
    0,
  );
  reconcileNumber(
    weightedProjectedPoints,
    first.step.weighted_projected_points + second.step.weighted_projected_points,
    "planner.sequential.best_sequence.weighted_projected_points",
  );
  const totalHitPoints = integer(
    sequence.total_hit_points,
    "planner.sequential.best_sequence.total_hit_points",
    0,
    32,
  );
  if (totalHitPoints !== first.step.hit_points + second.step.hit_points) {
    fail("planner.sequential.best_sequence.total_hit_points", "must equal both steps' hit points");
  }
  const weightedHitCostPoints = finiteNumber(
    sequence.weighted_hit_cost_points,
    "planner.sequential.best_sequence.weighted_hit_cost_points",
    0,
  );
  reconcileNumber(
    weightedHitCostPoints,
    first.step.weighted_hit_cost_points + second.step.weighted_hit_cost_points,
    "planner.sequential.best_sequence.weighted_hit_cost_points",
  );
  const terminalBankedFtValuePoints = finiteNumber(
    sequence.terminal_banked_ft_value_points,
    "planner.sequential.best_sequence.terminal_banked_ft_value_points",
    0,
  );
  reconcileNumber(
    terminalBankedFtValuePoints,
    Math.max(0, second.step.free_transfers_next_gameweek - 1) * options.rollFtValuePoints,
    "planner.sequential.best_sequence.terminal_banked_ft_value_points",
  );
  const decisionValuePoints = finiteNumber(
    sequence.decision_value_points,
    "planner.sequential.best_sequence.decision_value_points",
  );
  reconcileNumber(
    decisionValuePoints,
    weightedProjectedPoints - weightedHitCostPoints + terminalBankedFtValuePoints,
    "planner.sequential.best_sequence.decision_value_points",
  );

  return {
    ...root,
    gw_weights: weights,
    best_sequence: {
      ...sequence,
      first_action: { source, alternative_index: alternativeIndex },
      steps: [first.step, second.step],
    },
  } as unknown as PlannerSequentialPlan;
}

function validateStrategyPlan(
  value: unknown,
  options: {
    targetEvent: number;
    confirmedSquad: ReadonlyMap<number, PlannerPlayerReference>;
    confirmedPurchasePrices: ReadonlyMap<number, number>;
    bank: number;
    freeTransfers: number;
    bestAction: PlannerAction;
    alternatives: readonly PlannerAction[];
    rollFtValuePoints: number;
  },
): PlannerStrategyPlan {
  const root = exactRecord(value, "planner.strategy", [
    "horizon", "gameweek_window", "gw_weights", "modelled_deadlines",
    "maximum_provisional_transfers", "first_step_candidate_count", "first_step_search",
    "search_scope", "future_price_assumption", "assumptions",
    "solver_proven_optimal_within_bounds", "globally_optimal", "recalculate_each_deadline",
    "first_action", "steps", "weighted_projected_points", "total_hit_points",
    "weighted_hit_cost_points", "terminal_banked_ft_value_points", "decision_value_points",
  ]);
  const horizon = integer(root.horizon, "planner.strategy.horizon", 6, 10);
  const window = array(root.gameweek_window, "planner.strategy.gameweek_window").map(
    (event, index) => integer(event, `planner.strategy.gameweek_window[${index}]`, 1, 38),
  );
  if (window.length !== horizon) {
    fail("planner.strategy.gameweek_window", "length must match the strategy horizon");
  }
  if (window[0] !== options.targetEvent) {
    fail("planner.strategy.gameweek_window[0]", "must begin at planner.target_event");
  }
  if (window.some((event, index) => index > 0 && event !== window[index - 1] + 1)) {
    fail("planner.strategy.gameweek_window", "must be a contiguous gameweek window");
  }
  const weights = array(root.gw_weights, "planner.strategy.gw_weights").map((weight, index) =>
    finiteNumber(weight, `planner.strategy.gw_weights[${index}]`, 0.000_001, 1),
  );
  if (weights.length !== horizon) fail("planner.strategy.gw_weights", "length must match horizon");
  if (!closeEnough(weights[0], 1)) fail("planner.strategy.gw_weights[0]", "must equal 1");
  if (weights.some((weight, index) => index > 0 && weight > weights[index - 1])) {
    fail("planner.strategy.gw_weights", "must be non-increasing");
  }
  literal(root.modelled_deadlines, "planner.strategy.modelled_deadlines", 4);
  literal(root.maximum_provisional_transfers, "planner.strategy.maximum_provisional_transfers", 2);
  const candidateCount = integer(
    root.first_step_candidate_count,
    "planner.strategy.first_step_candidate_count",
    1,
    16,
  );
  if (candidateCount !== options.alternatives.length + 1) {
    fail(
      "planner.strategy.first_step_candidate_count",
      "must equal the visible best action plus alternatives",
    );
  }
  literal(
    root.first_step_search,
    "planner.strategy.first_step_search",
    "explicit_bounded_transfer_plans",
  );
  literal(
    root.search_scope,
    "planner.strategy.search_scope",
    "four_deadlines_max_two_provisional_transfers",
  );
  literal(
    root.future_price_assumption,
    "planner.strategy.future_price_assumption",
    "fixed_current_prices",
  );
  exactStringArray(root.assumptions, "planner.strategy.assumptions", STRATEGY_ASSUMPTIONS);
  literal(
    root.solver_proven_optimal_within_bounds,
    "planner.strategy.solver_proven_optimal_within_bounds",
    true,
  );
  literal(root.globally_optimal, "planner.strategy.globally_optimal", false);
  literal(root.recalculate_each_deadline, "planner.strategy.recalculate_each_deadline", true);

  const firstActionRow = exactRecord(root.first_action, "planner.strategy.first_action", [
    "source", "alternative_index",
  ]);
  const source = oneOf(
    firstActionRow.source,
    "planner.strategy.first_action.source",
    ["best_action", "alternative"] as const,
  );
  let alternativeIndex: number | null = null;
  let selectedAction = options.bestAction;
  if (source === "best_action") {
    literal(firstActionRow.alternative_index, "planner.strategy.first_action.alternative_index", null);
  } else {
    alternativeIndex = integer(
      firstActionRow.alternative_index,
      "planner.strategy.first_action.alternative_index",
      0,
      options.alternatives.length - 1,
    );
    selectedAction = options.alternatives[alternativeIndex];
  }

  const stepValues = array(root.steps, "planner.strategy.steps");
  if (stepValues.length !== 4) fail("planner.strategy.steps", "must contain exactly four steps");
  const stepWindows = [
    window.slice(0, 1),
    window.slice(1, 2),
    window.slice(2, 3),
    window.slice(3),
  ];
  const stepWeights = [
    weights.slice(0, 1),
    weights.slice(1, 2),
    weights.slice(2, 3),
    weights.slice(3),
  ];
  let priorSquad = new Map(options.confirmedSquad);
  let bankBefore = options.bank;
  let freeTransfersBefore = options.freeTransfers;
  const fixedPurchasePrices = new Map(options.confirmedPurchasePrices);
  const steps: PlannerStrategyStep[] = [];
  for (let index = 0; index < stepValues.length; index += 1) {
    const path = `planner.strategy.steps[${index}]`;
    const validated = validateSequentialStep(stepValues[index], path, {
      deadlineOffset: index + 1,
      provisional: index > 0,
      targetEvent: window[index],
      gameweekWindow: stepWindows[index],
      gameweekWeights: stepWeights[index],
      priorSquad,
      bankBefore,
      freeTransfersBefore,
      fixedPurchasePrices,
      maximumTransfers: index === 0 ? Math.max(2, options.freeTransfers) : 2,
    });
    const step = validated.step as unknown as PlannerStrategyStep;
    if (index === 0) {
      const matchesSelectedAction = (
        step.kind === selectedAction.kind &&
        step.transfer_count === selectedAction.transfer_count &&
        sameJson(step.transfers, selectedAction.transfers) &&
        sameJson(step.squad_ids, selectedAction.squad_ids) &&
        step.bank_before_tenths === selectedAction.bank_before_tenths &&
        step.bank_after_tenths === selectedAction.bank_after_tenths &&
        step.free_transfers_before === selectedAction.free_transfers_before &&
        step.free_transfers_next_gameweek === selectedAction.free_transfers_next_gameweek &&
        step.hit_points === selectedAction.hit_points &&
        sameJson(step.gameweeks[0], selectedAction.gameweeks[0])
      );
      if (!matchesSelectedAction) {
        fail(path, "must exactly match the referenced visible action at the first deadline");
      }
    }
    for (const transfer of step.transfers) {
      fixedPurchasePrices.delete(transfer.out_id);
      fixedPurchasePrices.set(transfer.in_id, transfer.in_price_tenths);
    }
    steps.push(step);
    priorSquad = validated.squad;
    bankBefore = step.bank_after_tenths;
    freeTransfersBefore = step.free_transfers_next_gameweek;
  }

  const weightedProjectedPoints = finiteNumber(
    root.weighted_projected_points,
    "planner.strategy.weighted_projected_points",
    0,
  );
  reconcileNumber(
    weightedProjectedPoints,
    steps.reduce((total, step) => total + step.weighted_projected_points, 0),
    "planner.strategy.weighted_projected_points",
  );
  const totalHitPoints = integer(root.total_hit_points, "planner.strategy.total_hit_points", 0, 64);
  if (totalHitPoints !== steps.reduce((total, step) => total + step.hit_points, 0)) {
    fail("planner.strategy.total_hit_points", "must equal all four steps' hit points");
  }
  const weightedHitCostPoints = finiteNumber(
    root.weighted_hit_cost_points,
    "planner.strategy.weighted_hit_cost_points",
    0,
  );
  reconcileNumber(
    weightedHitCostPoints,
    steps.reduce((total, step) => total + step.weighted_hit_cost_points, 0),
    "planner.strategy.weighted_hit_cost_points",
  );
  const terminalFtValue = finiteNumber(
    root.terminal_banked_ft_value_points,
    "planner.strategy.terminal_banked_ft_value_points",
    0,
  );
  const terminalFreeTransfers = Math.min(5, freeTransfersBefore + horizon - 4);
  reconcileNumber(
    terminalFtValue,
    Math.max(0, terminalFreeTransfers - 1) * options.rollFtValuePoints,
    "planner.strategy.terminal_banked_ft_value_points",
  );
  const decisionValue = finiteNumber(
    root.decision_value_points,
    "planner.strategy.decision_value_points",
  );
  reconcileNumber(
    decisionValue,
    weightedProjectedPoints - weightedHitCostPoints + terminalFtValue,
    "planner.strategy.decision_value_points",
  );

  return {
    ...root,
    horizon,
    gameweek_window: window,
    gw_weights: weights,
    first_action: { source, alternative_index: alternativeIndex },
    steps: steps as PlannerStrategyPlan["steps"],
  } as unknown as PlannerStrategyPlan;
}

function chipAvailableAt(
  chip: ChipName,
  event: number,
  usedEvents: readonly number[],
): boolean {
  const half = chipWindowHalf(chip, event);
  if (half === null) return false;
  if (usedEvents.some((usedEvent) => chipWindowHalf(chip, usedEvent) === half)) return false;
  return !(chip === "freehit" && event === 20 && usedEvents.includes(19));
}

function validateChipStrategy(
  value: unknown,
  options: {
    strategy: PlannerStrategyPlan;
    confirmedSquad: ReadonlyMap<number, PlannerPlayerReference>;
    confirmedSellingPrices: ReadonlyMap<number, number>;
    confirmedBank: number;
  },
): PlannerChipStrategy {
  const root = exactRecord(value, "planner.chip_strategy", [
    "horizon", "target_event", "inventory", "scenarios", "recommendation",
    "model_scope", "globally_optimal", "recalculate_each_deadline",
    ...(typeof value === "object" && value !== null && "sequence_comparison" in value ? ["sequence_comparison"] : []),
  ]);
  literal(root.horizon, "planner.chip_strategy.horizon", options.strategy.horizon);
  literal(root.target_event, "planner.chip_strategy.target_event", options.strategy.gameweek_window[0]);
  literal(root.model_scope, "planner.chip_strategy.model_scope", "bounded_chip_counterfactuals");
  literal(root.globally_optimal, "planner.chip_strategy.globally_optimal", false);
  literal(root.recalculate_each_deadline, "planner.chip_strategy.recalculate_each_deadline", true);

  const inventoryValues = array(root.inventory, "planner.chip_strategy.inventory");
  if (inventoryValues.length !== 4) {
    fail("planner.chip_strategy.inventory", "must contain all four chips exactly once");
  }
  const usageRows: ChipUsage[] = [];
  const inventory = inventoryValues.map((value, index) => {
    const path = `planner.chip_strategy.inventory[${index}]`;
    const row = exactRecord(value, path, ["chip", "used_events", "available_for_target"]);
    const chip = oneOf(row.chip, `${path}.chip`, CHIP_NAMES);
    const usedEvents = array(row.used_events, `${path}.used_events`).map((event, eventIndex) =>
      integer(event, `${path}.used_events[${eventIndex}]`, 1, 38),
    );
    if (
      usedEvents.length > 2 ||
      new Set(usedEvents).size !== usedEvents.length ||
      usedEvents.some((event, eventIndex) => eventIndex > 0 && event <= usedEvents[eventIndex - 1])
    ) {
      fail(`${path}.used_events`, "must contain up to two unique increasing events");
    }
    usageRows.push(...usedEvents.map((event) => ({ name: chip, event })));
    const availableForTarget = boolean(row.available_for_target, `${path}.available_for_target`);
    if (availableForTarget !== chipAvailableAt(chip, options.strategy.gameweek_window[0], usedEvents)) {
      fail(`${path}.available_for_target`, "is inconsistent with used_events and target_event");
    }
    return { chip, used_events: usedEvents, available_for_target: availableForTarget };
  });
  if (new Set(inventory.map((entry) => entry.chip)).size !== 4) {
    fail("planner.chip_strategy.inventory", "must contain all four chips exactly once");
  }
  validateChipUsage(usageRows, "planner.chip_strategy.inventory.used_events", {
    beforeEvent: options.strategy.gameweek_window[0],
    requireSorted: false,
  });
  const inventoryByChip = new Map(inventory.map((entry) => [entry.chip, entry]));
  const roadmapGameweeks = new Map(
    options.strategy.steps.flatMap((step) => step.gameweeks).map((gameweek) => [
      gameweek.gameweek,
      gameweek,
    ]),
  );
  const roadmapSquads = new Map(
    options.strategy.steps.flatMap((step) => step.gameweeks.map((gameweek) => [
      gameweek.gameweek,
      step.squad_ids,
    ] as const)),
  );

  const scenarioValues = array(root.scenarios, "planner.chip_strategy.scenarios");
  if (scenarioValues.length !== 4) {
    fail("planner.chip_strategy.scenarios", "must contain one scenario for each chip");
  }
  const scopes: Record<ChipName, PlannerChipScenarioScope> = {
    wildcard: "multiweek_rebuild",
    freehit: "confirmed_blank_double_screen",
    bboost: "bench_marginal",
    "3xc": "captain_marginal",
  };
  const scenarios = scenarioValues.map((value, index) => {
    const path = `planner.chip_strategy.scenarios[${index}]`;
    const row = exactRecord(value, path, [
      "scenario_id", "chip", "event", "signal", "available", "estimated_gain_points",
      "baseline_points", "chip_points", "confidence", "model_scope", "reason", "squad",
      "change_count", "bank_after_tenths",
    ]);
    const chip = oneOf(row.chip, `${path}.chip`, CHIP_NAMES);
    const scenarioId = text(row.scenario_id, `${path}.scenario_id`);
    const event = row.event === null
      ? null
      : integer(row.event, `${path}.event`, options.strategy.gameweek_window[0], options.strategy.gameweek_window.at(-1));
    const signal = oneOf(row.signal, `${path}.signal`, ["hold", "watch", "consider"] as const);
    const available = boolean(row.available, `${path}.available`);
    const confidence = oneOf(row.confidence, `${path}.confidence`, ["low", "medium"] as const);
    const modelScope = oneOf(
      row.model_scope,
      `${path}.model_scope`,
      chip === "freehit"
        ? ["confirmed_blank_double_screen", "single_gameweek_counterfactual"] as const
        : [scopes[chip]],
    );
    text(row.reason, `${path}.reason`);
    const inventoryEntry = inventoryByChip.get(chip)!;
    const expectedAvailability = chip === "wildcard"
      ? inventoryEntry.available_for_target
      : options.strategy.gameweek_window.some((candidateEvent) =>
          chipAvailableAt(chip, candidateEvent, inventoryEntry.used_events),
        );
    if (available !== expectedAvailability) {
      fail(`${path}.available`, "is inconsistent with inventory and the strategy window");
    }
    if (event !== null && !chipAvailableAt(chip, event, inventoryEntry.used_events)) {
      fail(`${path}.event`, "is not a legal available event for this chip inventory");
    }
    if (event !== null && !available) fail(`${path}.available`, "must be true when an event is selected");
    if (!available && (event !== null || signal !== "hold" || confidence !== "low")) {
      fail(path, "an unavailable scenario must be a low-confidence hold without an event");
    }
    if (event === null && signal !== "hold") fail(`${path}.signal`, "must be hold without an event");

    const metricValues = [row.estimated_gain_points, row.baseline_points, row.chip_points];
    const metricsAreNull = metricValues.every((metric) => metric === null);
    const metricsArePresent = metricValues.every((metric) => typeof metric === "number");
    if (!metricsAreNull && !metricsArePresent) {
      fail(path, "gain, baseline and chip points must either all be null or all be numbers");
    }
    let estimatedGain: number | null = null;
    let baselinePoints: number | null = null;
    let chipPoints: number | null = null;
    if (metricsArePresent) {
      estimatedGain = finiteNumber(row.estimated_gain_points, `${path}.estimated_gain_points`);
      baselinePoints = finiteNumber(row.baseline_points, `${path}.baseline_points`, 0);
      chipPoints = finiteNumber(row.chip_points, `${path}.chip_points`, 0);
      reconcileNumber(chipPoints, baselinePoints + estimatedGain, `${path}.chip_points`);
    }
    if (signal === "consider" && (event === null || !available || estimatedGain === null)) {
      fail(`${path}.signal`, "consider requires an available event and a quantified gain");
    }

    const expectedScenarioId = event === null
      ? chip === "freehit" && available
        ? "freehit-no-confirmed-trigger"
        : `${chip}-unavailable`
      : chip === "freehit"
        ? modelScope === "single_gameweek_counterfactual"
          ? `freehit-gw${event}`
          : `freehit-gw${event}-screen`
        : `${chip}-gw${event}`;
    if (scenarioId !== expectedScenarioId) {
      fail(`${path}.scenario_id`, `must be ${expectedScenarioId}`);
    }

    const squadValues = array(row.squad, `${path}.squad`);
    let squad: PlannerPlayerReference[] = [];
    let changeCount: number | null = null;
    let bankAfter: number | null = null;
    if (chip === "wildcard") {
      if (squadValues.length !== 0 && squadValues.length !== 15) {
        fail(`${path}.squad`, "must contain either zero or exactly 15 players");
      }
      squad = squadValues.map((player, playerIndex) =>
        validatePlayerReference(player, `${path}.squad[${playerIndex}]`),
      );
      if (squad.length === 15) {
        if (new Set(squad.map((player) => player.id)).size !== 15) {
          fail(`${path}.squad`, "player ids must be unique");
        }
        validatePositionQuotas(squad, `${path}.squad`);
        const clubCounts = new Map<string, number>();
        for (const player of squad) clubCounts.set(player.team, (clubCounts.get(player.team) ?? 0) + 1);
        if ([...clubCounts.values()].some((count) => count > 3)) {
          fail(`${path}.squad`, "cannot contain more than three players from one club");
        }
        changeCount = integer(row.change_count, `${path}.change_count`, 0, 15);
        const expectedChanges = [...options.confirmedSquad.keys()].filter(
          (id) => !squad.some((player) => player.id === id),
        ).length;
        if (changeCount !== expectedChanges) {
          fail(`${path}.change_count`, "must match the confirmed players removed by the Wildcard");
        }
        bankAfter = integer(row.bank_after_tenths, `${path}.bank_after_tenths`, 0, MAX_BANK_TENTHS);
        const currentBudget = options.confirmedBank + [...options.confirmedSellingPrices.values()]
          .reduce((total, price) => total + price, 0);
        const wildcardCost = squad.reduce((total, player) => total + (options.confirmedSellingPrices.get(player.id) ?? player.price_tenths), 0);
        if (bankAfter !== currentBudget - wildcardCost) {
          fail(`${path}.bank_after_tenths`, "does not reconcile with the confirmed Wildcard budget");
        }
      } else if (row.change_count !== null || row.bank_after_tenths !== null) {
        fail(path, "an unsolved Wildcard scenario cannot include change count or bank");
      }
      if (baselinePoints !== null && !(typeof root.sequence_comparison === "object" && root.sequence_comparison !== null
        && "status" in root.sequence_comparison && root.sequence_comparison.status === "ready"
        && "sequences" in root.sequence_comparison && Array.isArray(root.sequence_comparison.sequences)
        && root.sequence_comparison.sequences.some(s => typeof s === "object" && s !== null && s.sequence_id === "wildcard"))) {
        reconcileNumber(
          baselinePoints,
          options.strategy.decision_value_points,
          `${path}.baseline_points`,
        );
      }
    } else if (chip === "freehit" && modelScope === "single_gameweek_counterfactual") {
      if (event === null || !metricsArePresent || squadValues.length !== 15) {
        fail(path, "a solved Free Hit must include an event, metrics and exactly 15 players");
      }
      squad = squadValues.map((player, playerIndex) =>
        validatePlayerReference(player, `${path}.squad[${playerIndex}]`),
      );
      if (new Set(squad.map((player) => player.id)).size !== 15) {
        fail(`${path}.squad`, "player ids must be unique");
      }
      validatePositionQuotas(squad, `${path}.squad`);
      const clubCounts = new Map<string, number>();
      for (const player of squad) clubCounts.set(player.team, (clubCounts.get(player.team) ?? 0) + 1);
      if ([...clubCounts.values()].some((count) => count > 3)) {
        fail(`${path}.squad`, "cannot contain more than three players from one club");
      }
      const permanentSquad = roadmapSquads.get(event);
      if (!permanentSquad) fail(`${path}.event`, "must reference a roadmap squad");
      changeCount = integer(row.change_count, `${path}.change_count`, 0, 15);
      const expectedChanges = permanentSquad.filter(
        (id) => !squad.some((player) => player.id === id),
      ).length;
      if (changeCount !== expectedChanges) {
        fail(`${path}.change_count`, "must match the roadmap players replaced by the Free Hit");
      }
      literal(row.bank_after_tenths, `${path}.bank_after_tenths`, null);
      const currentBudget = options.confirmedBank + [...options.confirmedSellingPrices.values()]
        .reduce((total, price) => total + price, 0);
      const freeHitCost = squad.reduce((total, player) => total + player.price_tenths, 0);
      if (freeHitCost > currentBudget) {
        fail(`${path}.squad`, "exceeds the confirmed Free Hit budget");
      }
      const roadmapGameweek = roadmapGameweeks.get(event);
      if (!roadmapGameweek) fail(`${path}.event`, "must reference a roadmap gameweek");
      reconcileNumber(baselinePoints!, roadmapGameweek.projected_points, `${path}.baseline_points`);
    } else {
      if (squadValues.length !== 0 || row.change_count !== null || row.bank_after_tenths !== null) {
        fail(path, "only a solved Wildcard or Free Hit scenario may include squad or change count");
      }
      if (chip === "freehit" && modelScope === "confirmed_blank_double_screen" && !metricsAreNull) {
        fail(path, "an unsolved Free Hit screen cannot include quantified points");
      }
      if (baselinePoints !== null && event !== null) {
        const roadmapGameweek = roadmapGameweeks.get(event);
        if (!roadmapGameweek) fail(`${path}.event`, "must reference a roadmap gameweek");
        reconcileNumber(baselinePoints, roadmapGameweek.projected_points, `${path}.baseline_points`);
      }
    }

    return {
      scenario_id: scenarioId,
      chip,
      event,
      signal,
      available,
      estimated_gain_points: estimatedGain,
      baseline_points: baselinePoints,
      chip_points: chipPoints,
      confidence,
      model_scope: modelScope,
      reason: row.reason as string,
      squad,
      change_count: changeCount,
      bank_after_tenths: bankAfter,
    } satisfies PlannerChipScenario;
  });
  if (
    new Set(scenarios.map((scenario) => scenario.chip)).size !== 4 ||
    new Set(scenarios.map((scenario) => scenario.scenario_id)).size !== 4
  ) {
    fail("planner.chip_strategy.scenarios", "must contain four unique chip scenarios");
  }

  const sequencePlayers = new Map<number, PlannerPlayerReference>([
    ...options.confirmedSquad,
    ...options.strategy.steps.flatMap(step => step.transfers.flatMap(move => [[move.out.id, move.out], [move.in.id, move.in]] as [number, PlannerPlayerReference][])),
    ...scenarios.flatMap(scenario => scenario.squad.map(player => [player.id, player] as [number, PlannerPlayerReference])),
  ]);
  const sequenceComparison = root.sequence_comparison == null ? null : parseChipSequences(root.sequence_comparison, {
    strategy: options.strategy, players: sequencePlayers, squad: options.confirmedSquad,
    selling: options.confirmedSellingPrices, bank: options.confirmedBank, inventory,
  });
  if (sequenceComparison?.status === "ready") {
    const normal = sequenceComparison.sequences.find(sequence => sequence.sequence_id === "normal");
    const wildcard = sequenceComparison.sequences.find(sequence => sequence.sequence_id === "wildcard");
    const scenario = scenarios.find(scenario => scenario.chip === "wildcard");
    if (normal && wildcard && scenario) {
      reconcileNumber(scenario.baseline_points!, normal.weighted_net_points, "planner.chip_strategy.wildcard.baseline");
      reconcileNumber(scenario.chip_points!, wildcard.weighted_net_points, "planner.chip_strategy.wildcard.points");
      if (!sameValues(scenario.squad.map(player => player.id), wildcard.actions[0].squad_ids)) fail("planner.chip_strategy.wildcard", "must match the paired Wildcard squad");
    }
  }
  const recommendationRow = exactRecord(
    root.recommendation,
    "planner.chip_strategy.recommendation",
    ["action", "scenario_id", "chip", "event", "reason"],
  );
  const action = oneOf(
    recommendationRow.action,
    "planner.chip_strategy.recommendation.action",
    ["hold", "consider"] as const,
  );
  text(recommendationRow.reason, "planner.chip_strategy.recommendation.reason");
  let recommendedScenarioId: string | null = null;
  let recommendedChip: ChipName | null = null;
  let recommendedEvent: number | null = null;
  if (action === "hold") {
    literal(recommendationRow.scenario_id, "planner.chip_strategy.recommendation.scenario_id", null);
    literal(recommendationRow.chip, "planner.chip_strategy.recommendation.chip", null);
    literal(recommendationRow.event, "planner.chip_strategy.recommendation.event", null);
  } else {
    recommendedScenarioId = text(
      recommendationRow.scenario_id,
      "planner.chip_strategy.recommendation.scenario_id",
    );
    recommendedChip = oneOf(
      recommendationRow.chip,
      "planner.chip_strategy.recommendation.chip",
      CHIP_NAMES,
    );
    recommendedEvent = integer(
      recommendationRow.event,
      "planner.chip_strategy.recommendation.event",
      options.strategy.gameweek_window[0],
      options.strategy.gameweek_window[0],
    );
    const scenario = scenarios.find((candidate) => candidate.scenario_id === recommendedScenarioId);
    if (
      !scenario || scenario.signal !== "consider" || !scenario.available ||
      scenario.chip !== recommendedChip || scenario.event !== recommendedEvent
    ) {
      fail(
        "planner.chip_strategy.recommendation",
        "must reference an existing available consider scenario at the target event",
      );
    }
  }

  return {
    horizon: options.strategy.horizon,
    target_event: options.strategy.gameweek_window[0],
    inventory,
    scenarios,
    recommendation: {
      action,
      scenario_id: recommendedScenarioId,
      chip: recommendedChip,
      event: recommendedEvent,
      reason: recommendationRow.reason as string,
    },
    model_scope: "bounded_chip_counterfactuals",
    globally_optimal: false,
    recalculate_each_deadline: true,
    ...(root.sequence_comparison === undefined ? {} : { sequence_comparison: sequenceComparison }),
  };
}

export function parsePlannerPayload(value: unknown): PlannerPayload {
  const root = exactRecord(value, "planner", [
    "manager_id", "state_fingerprint", "source_event", "target_event", "confirmed_state",
    "best_action", "alternatives", "sequential", "strategy", "chip_strategy", "method",
  ]);
  integer(root.manager_id, "planner.manager_id", 1);
  if (root.state_fingerprint !== null && (typeof root.state_fingerprint !== "string" || !HEX_64.test(root.state_fingerprint))) fail("planner.state_fingerprint", "must be null or a SHA-256 checksum");
  const sourceEvent = integer(root.source_event, "planner.source_event", 1, 38);
  const targetEvent = integer(root.target_event, "planner.target_event", 1, 38);
  if (sourceEvent >= targetEvent) fail("planner.source_event", "must be earlier than target_event");
  const confirmed = exactRecord(root.confirmed_state, "planner.confirmed_state", [
    "bank_tenths", "free_transfers", "no_active_chip_confirmed", "squad",
  ]);
  const bank = integer(confirmed.bank_tenths, "planner.confirmed_state.bank_tenths", 0, MAX_BANK_TENTHS);
  const freeTransfers = integer(confirmed.free_transfers, "planner.confirmed_state.free_transfers", 0, 5);
  literal(confirmed.no_active_chip_confirmed, "planner.confirmed_state.no_active_chip_confirmed", true);
  const squadValues = array(confirmed.squad, "planner.confirmed_state.squad");
  if (squadValues.length !== 15) fail("planner.confirmed_state.squad", "must contain exactly 15 players");
  const ownedSquad = squadValues.map((player, index) => validateOwnedPlayerReference(player, `planner.confirmed_state.squad[${index}]`));
  const squad = ownedSquad.map(({ purchase_price_tenths: _purchase, selling_price_tenths: _selling, ...player }) => player);
  if (new Set(squad.map((player) => player.id)).size !== 15) fail("planner.confirmed_state.squad", "player ids must be unique");
  validatePositionQuotas(squad, "planner.confirmed_state.squad");
  const confirmedMap = new Map(squad.map((player) => [player.id, player]));
  const confirmedPrices = new Map(ownedSquad.map((player) => [player.id, {
    purchase: player.purchase_price_tenths,
    selling: player.selling_price_tenths,
  }]));
  const confirmedPurchasePrices = new Map(ownedSquad.map((player) => [player.id, player.purchase_price_tenths]));
  const best = validateAction(root.best_action, "planner.best_action", confirmedMap, confirmedPrices, bank, freeTransfers, targetEvent);
  const alternatives = array(root.alternatives, "planner.alternatives");
  if (alternatives.length > 4) fail("planner.alternatives", "must contain at most four actions");
  const parsedAlternatives = alternatives.map((action, index) => validateAction(action, `planner.alternatives[${index}]`, confirmedMap, confirmedPrices, bank, freeTransfers, targetEvent));
  const horizon = best.gameweeks.length;
  const gameweekWindow = best.gameweeks.map((gameweek) => gameweek.gameweek);
  if (parsedAlternatives.some((action) => action.gameweeks.length !== horizon
    || action.gameweeks.some((gameweek, index) => gameweek.gameweek !== gameweekWindow[index]))) {
    fail("planner.alternatives", "all actions must use the best action's gameweek window");
  }

  const method = exactRecord(root.method, "planner.method", [
    "candidate_count", "plans_per_transfer_count", "higher_transfer_count_plans", "maximum_immediate_transfers",
    "roll_ft_value_points", "chips_modelled", "bounded_roadmap_modelled",
    "next_deadline_transfer_modelled", "future_transfers_modelled",
  ]);
  const candidateCount = integer(method.candidate_count, "planner.method.candidate_count", 15);
  literal(method.plans_per_transfer_count, "planner.method.plans_per_transfer_count", 5);
  literal(method.higher_transfer_count_plans, "planner.method.higher_transfer_count_plans", 1);
  const maximumImmediateTransfers = integer(method.maximum_immediate_transfers, "planner.method.maximum_immediate_transfers", 2, 5);
  if (maximumImmediateTransfers !== Math.max(2, freeTransfers)) fail("planner.method.maximum_immediate_transfers", "must cover all available free transfers");
  if ([best, ...parsedAlternatives].some((action) => action.transfer_count > maximumImmediateTransfers)) fail("planner.method.maximum_immediate_transfers", "must cover every returned action");
  const rollFtValuePoints = finiteNumber(method.roll_ft_value_points, "planner.method.roll_ft_value_points", 0);
  const chipsModelled = boolean(method.chips_modelled, "planner.method.chips_modelled");
  const boundedRoadmapModelled = boolean(
    method.bounded_roadmap_modelled,
    "planner.method.bounded_roadmap_modelled",
  );
  const nextDeadlineTransferModelled = boolean(
    method.next_deadline_transfer_modelled,
    "planner.method.next_deadline_transfer_modelled",
  );
  literal(method.future_transfers_modelled, "planner.method.future_transfers_modelled", false);
  if (root.sequential !== null && root.strategy !== null) {
    fail("planner", "cannot contain both the fallback sequential plan and the long-range strategy");
  }
  if (horizon === 1 && root.sequential !== null) {
    fail("planner.sequential", "must be null when the planner horizon is one gameweek");
  }
  let parsedSequential: PlannerSequentialPlan | null = null;
  if (root.sequential !== null) {
    parsedSequential = validateSequentialPlan(root.sequential, {
      horizon,
      targetEvent,
      confirmedSquad: confirmedMap,
      confirmedPurchasePrices,
      bank,
      freeTransfers,
      bestAction: best,
      alternatives: parsedAlternatives,
      rollFtValuePoints,
    });
  }
  let parsedStrategy: PlannerStrategyPlan | null = null;
  if (root.strategy !== null) {
    parsedStrategy = validateStrategyPlan(root.strategy, {
      targetEvent,
      confirmedSquad: confirmedMap,
      confirmedPurchasePrices,
      bank,
      freeTransfers,
      bestAction: best,
      alternatives: parsedAlternatives,
      rollFtValuePoints,
    });
  }
  let parsedChipStrategy: PlannerChipStrategy | null = null;
  if (root.chip_strategy !== null) {
    if (parsedStrategy === null) {
      fail("planner.chip_strategy", "requires a validated long-range strategy");
    }
    parsedChipStrategy = validateChipStrategy(root.chip_strategy, {
      strategy: parsedStrategy,
      confirmedSquad: confirmedMap,
      confirmedSellingPrices: new Map(
        [...confirmedPrices].map(([id, prices]) => [id, prices.selling]),
      ),
      confirmedBank: bank,
    });
  }
  if (boundedRoadmapModelled !== (parsedStrategy !== null)) {
    fail(
      "planner.method.bounded_roadmap_modelled",
      "must be true exactly when a strategy is present",
    );
  }
  if (chipsModelled !== (parsedChipStrategy !== null)) {
    fail(
      "planner.method.chips_modelled",
      "must be true exactly when a chip strategy is present",
    );
  }
  if (nextDeadlineTransferModelled !== (parsedSequential !== null || parsedStrategy !== null)) {
    fail(
      "planner.method.next_deadline_transfer_modelled",
      "must be true exactly when a sequential plan or strategy is present",
    );
  }
  const referencedIds = new Set([
    ...(best.transfers ?? []),
    ...parsedAlternatives.flatMap((action) => action.transfers),
    ...(parsedSequential?.best_sequence.steps.flatMap((step) => step.transfers) ?? []),
    ...(parsedStrategy?.steps.flatMap((step) => step.transfers) ?? []),
  ].flatMap((transfer) => [transfer.out_id, transfer.in_id]));
  const referencedPlayerIds = new Set([
    ...confirmedMap.keys(),
    ...referencedIds,
    ...(parsedChipStrategy?.scenarios.flatMap((scenario) => scenario.squad.map((player) => player.id)) ?? []),
  ]);
  if (candidateCount < referencedPlayerIds.size) fail("planner.method.candidate_count", "is smaller than the referenced candidate set");
  return root as unknown as PlannerPayload;
}

export function isPlannerPayload(value: unknown): value is PlannerPayload {
  try {
    parsePlannerPayload(value);
    return true;
  } catch {
    return false;
  }
}

export const validateManagerSyncResponse = parseManagerSyncResponse;
export const validatePlannerPayload = parsePlannerPayload;
export const parseComputePlannerPayload = parsePlannerPayload;
