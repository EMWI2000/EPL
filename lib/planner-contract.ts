export const MANAGER_SYNC_SCHEMA_VERSION = "fpl-manager-state-response-v1" as const;
export const PERSONAL_STATE_SCHEMA_VERSION = "fpl-personal-state-v1" as const;
export const SNAPSHOT_SCHEMA_VERSION = "fpl-deadline-state-snapshot-v1" as const;
export const PRICE_SIGNAL_SCHEMA_VERSION = "fpl-official-price-signals-v1" as const;

export type ForecastVersion = "v2" | "legacy";
export type PlayerPosition = "GKP" | "DEF" | "MID" | "FWD";
export type PlayerStatus = "a" | "d" | "i" | "s" | "u" | "n";
export type ChipName = "wildcard" | "freehit" | "bboost" | "3xc";
export type ChipStatus = "available" | "used" | "active" | "unavailable";

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
  limitations: PublicStateLimitation[];
  picks: ManagerSyncPick[];
}

export interface PlayerPriceState {
  element_id: number;
  purchase_price_tenths: number;
  selling_price_tenths: number;
}

export type ChipState = Partial<Record<ChipName, ChipStatus>>;

export interface ManualManagerState {
  current_squad_ids: number[];
  bank_tenths: number;
  free_transfers: number;
  player_prices: PlayerPriceState[];
  chips: ChipState;
  no_active_chip_confirmed: boolean;
  effective_event: number | null;
}

export type ConfirmationField =
  | "current_squad_ids"
  | "bank_tenths"
  | "free_transfers"
  | "player_prices"
  | "chips"
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

export interface PlannerPayload {
  manager_id: number;
  state_fingerprint: string | null;
  source_event: number;
  target_event: number;
  confirmed_state: {
    bank_tenths: number;
    free_transfers: number;
    no_active_chip_confirmed: true;
    squad: PlannerPlayerReference[];
  };
  best_action: PlannerAction;
  alternatives: PlannerAction[];
  method: {
    candidate_count: number;
    plans_per_transfer_count: 5;
    higher_transfer_count_plans: 1;
    maximum_immediate_transfers: number;
    roll_ft_value_points: number;
    chips_modelled: false;
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
const CHIP_STATUSES = ["available", "used", "active", "unavailable"] as const;
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

function literal<T extends string | number | boolean>(value: unknown, path: string, expected: T): T {
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

function validateChips(value: unknown, path: string): ChipState {
  const chips = record(value, path);
  const keys = Object.keys(chips);
  const unknown = keys.filter((key) => !CHIP_NAMES.includes(key as ChipName));
  if (unknown.length) fail(path, `contains unsupported chips: ${unknown.join(", ")}`);
  let active = 0;
  for (const key of keys) {
    const status = oneOf(chips[key], `${path}.${key}`, CHIP_STATUSES);
    if (status === "active") active += 1;
  }
  if (active > 1) fail(path, "at most one chip can be active");
  return chips as ChipState;
}

export function parseManualManagerState(value: unknown): ManualManagerState {
  const path = "manager_state";
  const row = exactRecord(value, path, [
    "current_squad_ids", "bank_tenths", "free_transfers", "player_prices", "chips",
    "no_active_chip_confirmed", "effective_event",
  ]);
  const squadIds = uniqueIntegers(row.current_squad_ids, `${path}.current_squad_ids`, 15);
  integer(row.bank_tenths, `${path}.bank_tenths`, 0, MAX_BANK_TENTHS);
  integer(row.free_transfers, `${path}.free_transfers`, 1, 5);
  validatePlayerPrices(row.player_prices, `${path}.player_prices`, squadIds);
  validateChips(row.chips, `${path}.chips`);
  boolean(row.no_active_chip_confirmed, `${path}.no_active_chip_confirmed`);
  nullable(row.effective_event, (item) => integer(item, `${path}.effective_event`, 1, 38));
  return row as unknown as ManualManagerState;
}

export function parseManagerSyncResponse(value: unknown): ManagerSyncResponse {
  const root = exactRecord(value, "sync", [
    "schema_version", "generated_at", "manager", "target", "last_deadline_state",
    "price_signals", "manual_state_template", "snapshot", "warnings",
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
    "limitations", "picks",
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
  if (Object.keys(manualState.chips).length !== 0) fail("sync.manual_state_template.state.chips", "must be empty until confirmed manually");
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
  if (!/^[1-5]$/.test(normalized)) fail("free_transfers", "must be a whole number from 1 to 5");
  return Number(normalized);
}

function cloneManualState(state: ManualManagerState): ManualManagerState {
  return {
    current_squad_ids: [...state.current_squad_ids],
    bank_tenths: state.bank_tenths,
    free_transfers: state.free_transfers,
    player_prices: state.player_prices.map((price) => ({ ...price })),
    chips: { ...state.chips },
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

function validateAction(value: unknown, path: string, confirmed: Map<number, PlannerPlayerReference>, bank: number, freeTransfers: number, targetEvent: number): PlannerAction {
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
  const ftBefore = integer(row.free_transfers_before, `${path}.free_transfers_before`, 1, 5);
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

export function parsePlannerPayload(value: unknown): PlannerPayload {
  const root = exactRecord(value, "planner", [
    "manager_id", "state_fingerprint", "source_event", "target_event", "confirmed_state",
    "best_action", "alternatives", "method",
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
  const freeTransfers = integer(confirmed.free_transfers, "planner.confirmed_state.free_transfers", 1, 5);
  literal(confirmed.no_active_chip_confirmed, "planner.confirmed_state.no_active_chip_confirmed", true);
  const squadValues = array(confirmed.squad, "planner.confirmed_state.squad");
  if (squadValues.length !== 15) fail("planner.confirmed_state.squad", "must contain exactly 15 players");
  const squad = squadValues.map((player, index) => validatePlayerReference(player, `planner.confirmed_state.squad[${index}]`));
  if (new Set(squad.map((player) => player.id)).size !== 15) fail("planner.confirmed_state.squad", "player ids must be unique");
  validatePositionQuotas(squad, "planner.confirmed_state.squad");
  const confirmedMap = new Map(squad.map((player) => [player.id, player]));
  const best = validateAction(root.best_action, "planner.best_action", confirmedMap, bank, freeTransfers, targetEvent);
  const alternatives = array(root.alternatives, "planner.alternatives");
  if (alternatives.length > 4) fail("planner.alternatives", "must contain at most four actions");
  const parsedAlternatives = alternatives.map((action, index) => validateAction(action, `planner.alternatives[${index}]`, confirmedMap, bank, freeTransfers, targetEvent));
  const horizon = best.gameweeks.length;
  const gameweekWindow = best.gameweeks.map((gameweek) => gameweek.gameweek);
  if (parsedAlternatives.some((action) => action.gameweeks.length !== horizon
    || action.gameweeks.some((gameweek, index) => gameweek.gameweek !== gameweekWindow[index]))) {
    fail("planner.alternatives", "all actions must use the best action's gameweek window");
  }

  const method = exactRecord(root.method, "planner.method", [
    "candidate_count", "plans_per_transfer_count", "higher_transfer_count_plans", "maximum_immediate_transfers",
    "roll_ft_value_points", "chips_modelled", "future_transfers_modelled",
  ]);
  const candidateCount = integer(method.candidate_count, "planner.method.candidate_count", 15);
  literal(method.plans_per_transfer_count, "planner.method.plans_per_transfer_count", 5);
  literal(method.higher_transfer_count_plans, "planner.method.higher_transfer_count_plans", 1);
  const maximumImmediateTransfers = integer(method.maximum_immediate_transfers, "planner.method.maximum_immediate_transfers", 2, 5);
  if (maximumImmediateTransfers !== Math.max(2, freeTransfers)) fail("planner.method.maximum_immediate_transfers", "must cover all available free transfers");
  if ([best, ...parsedAlternatives].some((action) => action.transfer_count > maximumImmediateTransfers)) fail("planner.method.maximum_immediate_transfers", "must cover every returned action");
  finiteNumber(method.roll_ft_value_points, "planner.method.roll_ft_value_points", 0);
  literal(method.chips_modelled, "planner.method.chips_modelled", false);
  literal(method.future_transfers_modelled, "planner.method.future_transfers_modelled", false);
  const referencedIds = new Set([...(best.transfers ?? []), ...parsedAlternatives.flatMap((action) => action.transfers)].flatMap((transfer) => [transfer.out_id, transfer.in_id]));
  if (candidateCount < new Set([...confirmedMap.keys(), ...referencedIds]).size) fail("planner.method.candidate_count", "is smaller than the referenced candidate set");
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
