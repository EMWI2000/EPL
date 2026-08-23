import type {
  ManagerSyncResponse,
  PlannerAction,
  PlannerPayload,
  PlannerPlayerReference,
  PlayerPosition,
  PlayerStatus,
  PublicStateLimitation,
} from "@/lib/planner-contract";

export const AI_REVIEW_REQUEST_SCHEMA_VERSION = "fpl-ai-review-request-v1" as const;
export const AI_REVIEW_RESPONSE_SCHEMA_VERSION = "fpl-ai-review-response-v1" as const;
export const MAX_AI_REVIEW_RECOMMENDATION_AGE_MS = 24 * 60 * 60 * 1_000;

export const AI_REVIEW_ALLOWED_SOURCE_DOMAINS = [
  "premierleague.com",
  "bbc.com",
  "bbc.co.uk",
] as const;

export type ManagerRankBand =
  | "top_10k"
  | "top_100k"
  | "top_500k"
  | "top_1m"
  | "outside_1m"
  | "unknown";

export interface AiReviewProjectionContext {
  gameweek: number;
  expected_points: number;
  expected_minutes: number | null;
  appearance_probability: number;
  sixty_probability: number;
  confidence: number;
  reliability: "low" | "medium" | "high";
  fixtures_count: number;
  is_blank: boolean;
  is_dgw: boolean;
}

export interface AiReviewPriceSignalContext {
  selected_by_percent: number;
  transfers_in_event: number;
  transfers_out_event: number;
  cost_change_event_tenths: number;
}

export interface AiReviewSquadPlayerContext {
  id: number;
  name: string;
  team: string;
  position: PlayerPosition;
  status: PlayerStatus;
  current_price_tenths: number;
  weighted_expected_points: number;
  projections: AiReviewProjectionContext[];
  price_signal: AiReviewPriceSignalContext | null;
}

export interface AiReviewRequest {
  schema_version: typeof AI_REVIEW_REQUEST_SCHEMA_VERSION;
  manager_id: number;
  recommendation_generated_at: string;
  target_deadline: string;
  state_observed_at: string;
  manager_rank_band: ManagerRankBand;
  state_limitations: PublicStateLimitation[];
  forecast: {
    version: "v2" | "legacy";
    horizon: number;
    include_doubtful: boolean;
    validation_status: string;
    price_signals_available: boolean;
    next_price_deadline: string | null;
  };
  planner: PlannerPayload;
  lineup: {
    gameweek: number;
    formation: string;
    starting_ids: number[];
    bench_ids: number[];
    captain_id: number;
    vice_captain_id: number;
  };
  squad_context: AiReviewSquadPlayerContext[];
}

export type AiReviewVerdict =
  | "confirm_best_action"
  | "wait_for_information"
  | "prefer_alternative";

export interface AiReviewModelOutput {
  verdict: AiReviewVerdict;
  alternative_index: number | null;
  headline: string;
  summary: string;
  rationale: string[];
  risks: string[];
  change_triggers: string[];
  deadline_checklist: string[];
  evidence_summary: string;
  data_gaps: string[];
  confidence: "low" | "medium" | "high";
}

export interface AiReviewSource {
  title: string;
  url: string;
}

export interface AiReviewResponse {
  schema_version: typeof AI_REVIEW_RESPONSE_SCHEMA_VERSION;
  generated_at: string;
  recommendation_generated_at: string;
  target_event: number;
  model: string;
  review: AiReviewModelOutput;
  research: {
    performed: boolean;
    sources: AiReviewSource[];
  };
}

export interface AiReviewRecommendationSource {
  meta: {
    generated_at: string;
    horizon: number;
    forecast_version: "v2" | "legacy";
    include_doubtful: boolean;
    validation: { status: string };
    data_sources: {
      price_signals: {
        available?: boolean;
        price_change_deadlines?: string[];
      };
    };
  };
  team: {
    squad: Array<{
      id: number;
      weighted_ep: number;
      projections: Array<{
        gameweek: number;
        ep: number;
        expected_minutes: number | null;
        appearance_probability: number;
        sixty_probability: number;
        confidence: number;
        reliability: "low" | "medium" | "high";
        fixtures_count: number;
        is_blank: boolean;
        is_dgw: boolean;
      }>;
    }>;
    gameweeks: Array<{
      gameweek: number;
      formation: string;
      starting_ids: number[];
      bench_ids: number[];
      captain_id: number;
      vice_captain_id: number;
    }>;
  };
  planner: PlannerPayload;
}

export const AI_REVIEW_OUTPUT_SCHEMA = {
  type: "object",
  properties: {
    verdict: {
      type: "string",
      enum: ["confirm_best_action", "wait_for_information", "prefer_alternative"],
    },
    alternative_index: {
      type: ["integer", "null"],
      description: "Null for confirm_best_action and wait_for_information; otherwise the zero-based index of an existing solver alternative.",
    },
    headline: { type: "string" },
    summary: { type: "string" },
    rationale: {
      type: "array",
      minItems: 2,
      maxItems: 4,
      items: { type: "string" },
    },
    risks: {
      type: "array",
      minItems: 0,
      maxItems: 4,
      items: { type: "string" },
    },
    change_triggers: {
      type: "array",
      minItems: 0,
      maxItems: 4,
      items: { type: "string" },
    },
    deadline_checklist: {
      type: "array",
      minItems: 2,
      maxItems: 5,
      items: { type: "string" },
    },
    evidence_summary: { type: "string" },
    data_gaps: {
      type: "array",
      minItems: 0,
      maxItems: 4,
      items: { type: "string" },
    },
    confidence: { type: "string", enum: ["low", "medium", "high"] },
  },
  required: [
    "verdict",
    "alternative_index",
    "headline",
    "summary",
    "rationale",
    "risks",
    "change_triggers",
    "deadline_checklist",
    "evidence_summary",
    "data_gaps",
    "confidence",
  ],
  additionalProperties: false,
} as const;

type UnknownRecord = Record<string, unknown>;

const POSITIONS = ["GKP", "DEF", "MID", "FWD"] as const;
const STATUSES = ["a", "d", "i", "s", "u", "n"] as const;
const RELIABILITIES = ["low", "medium", "high"] as const;
const RANK_BANDS: readonly ManagerRankBand[] = [
  "top_10k",
  "top_100k",
  "top_500k",
  "top_1m",
  "outside_1m",
  "unknown",
];
const LIMITATIONS: readonly PublicStateLimitation[] = [
  "state_is_locked_at_last_public_deadline",
  "current_free_transfers_not_public",
  "current_confirmed_transfers_not_public",
  "purchase_and_selling_prices_not_public",
  "next_deadline_chip_selection_not_public",
];
const ISO_UTC = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d{1,6})?Z$/;
const CONTROL_CHARACTERS = /[\u0000-\u001f\u007f]/;

export class AiReviewContractError extends TypeError {
  readonly path: string;

  constructor(path: string, message: string) {
    super(`${path}: ${message}`);
    this.name = "AiReviewContractError";
    this.path = path;
  }
}

function fail(path: string, message: string): never {
  throw new AiReviewContractError(path, message);
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

function array(value: unknown, path: string, minimum: number, maximum: number): unknown[] {
  if (!Array.isArray(value) || value.length < minimum || value.length > maximum) {
    fail(path, `must be an array with ${minimum}-${maximum} items`);
  }
  return value;
}

function integer(value: unknown, path: string, minimum: number, maximum = Number.MAX_SAFE_INTEGER): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum || (value as number) > maximum) {
    fail(path, `must be an integer from ${minimum} to ${maximum}`);
  }
  return value as number;
}

function finiteNumber(value: unknown, path: string, minimum: number, maximum: number): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum || value > maximum) {
    fail(path, `must be a finite number from ${minimum} to ${maximum}`);
  }
  return value;
}

function boolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") fail(path, "must be a boolean");
  return value;
}

function boundedText(value: unknown, path: string, maximum: number): string {
  if (
    typeof value !== "string" ||
    value.trim().length === 0 ||
    value.length > maximum ||
    CONTROL_CHARACTERS.test(value)
  ) {
    fail(path, `must be non-empty text without control characters, at most ${maximum} characters`);
  }
  return value;
}

function oneOf<T extends string>(value: unknown, path: string, choices: readonly T[]): T {
  if (typeof value !== "string" || !choices.includes(value as T)) {
    fail(path, `must be one of: ${choices.join(", ")}`);
  }
  return value as T;
}

function literal<T extends string>(value: unknown, path: string, expected: T): T {
  if (value !== expected) fail(path, `must be ${JSON.stringify(expected)}`);
  return expected;
}

function isoUtc(value: unknown, path: string): string {
  const result = boundedText(value, path, 40);
  if (!ISO_UTC.test(result) || !Number.isFinite(new Date(result).getTime())) {
    fail(path, "must be a UTC ISO-8601 timestamp ending in Z");
  }
  return result;
}

function uniqueIntegers(value: unknown, path: string, count: number): number[] {
  const values = array(value, path, count, count).map((item, index) =>
    integer(item, `${path}[${index}]`, 1),
  );
  if (new Set(values).size !== values.length) fail(path, "must contain unique values");
  return values;
}

function sameValues(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value) => right.includes(value));
}

function finalSquadReferences(planner: PlannerPayload): Map<number, PlannerPlayerReference> {
  const players = new Map(planner.confirmed_state.squad.map((player) => [player.id, player]));
  for (const transfer of planner.best_action.transfers) {
    players.delete(transfer.out_id);
    players.set(transfer.in_id, transfer.in);
  }
  return players;
}

function validatePriceSignal(value: unknown, path: string): AiReviewPriceSignalContext | null {
  if (value === null) return null;
  const row = exactRecord(value, path, [
    "selected_by_percent",
    "transfers_in_event",
    "transfers_out_event",
    "cost_change_event_tenths",
  ]);
  finiteNumber(row.selected_by_percent, `${path}.selected_by_percent`, 0, 100);
  integer(row.transfers_in_event, `${path}.transfers_in_event`, 0);
  integer(row.transfers_out_event, `${path}.transfers_out_event`, 0);
  integer(row.cost_change_event_tenths, `${path}.cost_change_event_tenths`, -30, 30);
  return row as unknown as AiReviewPriceSignalContext;
}

function validateProjection(value: unknown, path: string, gameweeks: ReadonlySet<number>): AiReviewProjectionContext {
  const row = exactRecord(value, path, [
    "gameweek",
    "expected_points",
    "expected_minutes",
    "appearance_probability",
    "sixty_probability",
    "confidence",
    "reliability",
    "fixtures_count",
    "is_blank",
    "is_dgw",
  ]);
  const gameweek = integer(row.gameweek, `${path}.gameweek`, 1, 38);
  if (!gameweeks.has(gameweek)) fail(`${path}.gameweek`, "must be inside the planner horizon");
  finiteNumber(row.expected_points, `${path}.expected_points`, 0, 100);
  if (row.expected_minutes !== null) {
    finiteNumber(row.expected_minutes, `${path}.expected_minutes`, 0, 270);
  }
  finiteNumber(row.appearance_probability, `${path}.appearance_probability`, 0, 1);
  finiteNumber(row.sixty_probability, `${path}.sixty_probability`, 0, 1);
  finiteNumber(row.confidence, `${path}.confidence`, 0, 1);
  oneOf(row.reliability, `${path}.reliability`, RELIABILITIES);
  integer(row.fixtures_count, `${path}.fixtures_count`, 0, 3);
  boolean(row.is_blank, `${path}.is_blank`);
  boolean(row.is_dgw, `${path}.is_dgw`);
  return row as unknown as AiReviewProjectionContext;
}

function validateSquadPlayer(
  value: unknown,
  path: string,
  reference: PlannerPlayerReference,
  gameweeks: ReadonlySet<number>,
  horizon: number,
): AiReviewSquadPlayerContext {
  const row = exactRecord(value, path, [
    "id",
    "name",
    "team",
    "position",
    "status",
    "current_price_tenths",
    "weighted_expected_points",
    "projections",
    "price_signal",
  ]);
  const id = integer(row.id, `${path}.id`, 1);
  const name = boundedText(row.name, `${path}.name`, 80);
  const team = boundedText(row.team, `${path}.team`, 80);
  const position = oneOf(row.position, `${path}.position`, POSITIONS);
  const status = oneOf(row.status, `${path}.status`, STATUSES);
  const price = integer(row.current_price_tenths, `${path}.current_price_tenths`, 1, 500);
  finiteNumber(row.weighted_expected_points, `${path}.weighted_expected_points`, 0, 500);
  const projections = array(row.projections, `${path}.projections`, 1, horizon).map((projection, index) =>
    validateProjection(projection, `${path}.projections[${index}]`, gameweeks),
  );
  const projectionGameweeks = projections.map((projection) => projection.gameweek);
  if (new Set(projectionGameweeks).size !== projectionGameweeks.length) {
    fail(`${path}.projections`, "gameweeks must be unique");
  }
  validatePriceSignal(row.price_signal, `${path}.price_signal`);
  if (
    id !== reference.id ||
    name !== reference.name ||
    team !== reference.team ||
    position !== reference.position ||
    status !== reference.status ||
    price !== reference.price_tenths
  ) {
    fail(path, "must match the validated best-action player reference");
  }
  return row as unknown as AiReviewSquadPlayerContext;
}

export function managerRankBand(overallRank: number | null): ManagerRankBand {
  if (overallRank === null || !Number.isSafeInteger(overallRank) || overallRank < 1) return "unknown";
  if (overallRank <= 10_000) return "top_10k";
  if (overallRank <= 100_000) return "top_100k";
  if (overallRank <= 500_000) return "top_500k";
  if (overallRank <= 1_000_000) return "top_1m";
  return "outside_1m";
}

function canonicalUtc(value: string, path: string): string {
  const parsed = new Date(value);
  if (!Number.isFinite(parsed.getTime())) {
    throw new AiReviewContractError(path, "must be a valid timestamp");
  }
  return parsed.toISOString();
}

function compactPriceSignal(reference: PlannerPlayerReference): AiReviewPriceSignalContext | null {
  const signal = reference.price_signal;
  if (!signal) return null;
  return {
    selected_by_percent: signal.selected_by_percent,
    transfers_in_event: signal.transfers_in_event,
    transfers_out_event: signal.transfers_out_event,
    cost_change_event_tenths: signal.cost_change_event_tenths,
  };
}

export function buildAiReviewRequest(
  sync: ManagerSyncResponse,
  recommendation: AiReviewRecommendationSource,
): AiReviewRequest {
  const planner = recommendation.planner;
  const lineup = recommendation.team.gameweeks[0];
  if (!lineup) throw new AiReviewContractError("recommendation.team.gameweeks", "must contain the target gameweek");
  const sourcePlayers = new Map(recommendation.team.squad.map((player) => [player.id, player]));
  const finalPlayers = finalSquadReferences(planner);
  const gameweeks = new Set(planner.best_action.gameweeks.map((gameweek) => gameweek.gameweek));

  const squadContext = planner.best_action.squad_ids.map((id) => {
    const reference = finalPlayers.get(id);
    const source = sourcePlayers.get(id);
    if (!reference || !source) {
      throw new AiReviewContractError("recommendation.team.squad", `is missing best-action player ${id}`);
    }
    return {
      id,
      name: reference.name,
      team: reference.team,
      position: reference.position,
      status: reference.status,
      current_price_tenths: reference.price_tenths,
      weighted_expected_points: source.weighted_ep,
      projections: source.projections
        .filter((projection) => gameweeks.has(projection.gameweek))
        .map((projection) => ({
          gameweek: projection.gameweek,
          expected_points: projection.ep,
          expected_minutes: projection.expected_minutes,
          appearance_probability: projection.appearance_probability,
          sixty_probability: projection.sixty_probability,
          confidence: projection.confidence,
          reliability: projection.reliability,
          fixtures_count: projection.fixtures_count,
          is_blank: projection.is_blank,
          is_dgw: projection.is_dgw,
        })),
      price_signal: compactPriceSignal(reference),
    } satisfies AiReviewSquadPlayerContext;
  });

  return {
    schema_version: AI_REVIEW_REQUEST_SCHEMA_VERSION,
    manager_id: sync.manager.id,
    recommendation_generated_at: canonicalUtc(
      recommendation.meta.generated_at,
      "recommendation.meta.generated_at",
    ),
    target_deadline: canonicalUtc(sync.target.deadline_time, "sync.target.deadline_time"),
    state_observed_at: canonicalUtc(sync.snapshot.observed_at, "sync.snapshot.observed_at"),
    manager_rank_band: managerRankBand(sync.manager.overall_rank),
    state_limitations: [...sync.last_deadline_state.limitations],
    forecast: {
      version: recommendation.meta.forecast_version,
      horizon: recommendation.meta.horizon,
      include_doubtful: recommendation.meta.include_doubtful,
      validation_status: recommendation.meta.validation.status,
      price_signals_available: Boolean(recommendation.meta.data_sources.price_signals.available),
      next_price_deadline: recommendation.meta.data_sources.price_signals.price_change_deadlines?.[0]
        ? canonicalUtc(
            recommendation.meta.data_sources.price_signals.price_change_deadlines[0],
            "recommendation.meta.data_sources.price_signals.price_change_deadlines[0]",
          )
        : null,
    },
    planner,
    lineup: {
      gameweek: lineup.gameweek,
      formation: lineup.formation,
      starting_ids: [...lineup.starting_ids],
      bench_ids: [...lineup.bench_ids],
      captain_id: lineup.captain_id,
      vice_captain_id: lineup.vice_captain_id,
    },
    squad_context: squadContext,
  };
}

export function parseAiReviewRequest(
  value: unknown,
  parsePlanner: (value: unknown) => PlannerPayload,
): AiReviewRequest {
  const root = exactRecord(value, "request", [
    "schema_version",
    "manager_id",
    "recommendation_generated_at",
    "target_deadline",
    "state_observed_at",
    "manager_rank_band",
    "state_limitations",
    "forecast",
    "planner",
    "lineup",
    "squad_context",
  ]);
  literal(root.schema_version, "request.schema_version", AI_REVIEW_REQUEST_SCHEMA_VERSION);
  const managerId = integer(root.manager_id, "request.manager_id", 1);
  const generatedAt = isoUtc(root.recommendation_generated_at, "request.recommendation_generated_at");
  const targetDeadline = isoUtc(root.target_deadline, "request.target_deadline");
  const observedAt = isoUtc(root.state_observed_at, "request.state_observed_at");
  if (new Date(observedAt) > new Date(generatedAt)) {
    fail("request.state_observed_at", "cannot be later than the recommendation");
  }
  if (new Date(generatedAt) > new Date(targetDeadline)) {
    fail("request.target_deadline", "must be later than the recommendation");
  }
  oneOf(root.manager_rank_band, "request.manager_rank_band", RANK_BANDS);

  const limitations = array(root.state_limitations, "request.state_limitations", 1, LIMITATIONS.length).map(
    (limitation, index) => oneOf(limitation, `request.state_limitations[${index}]`, LIMITATIONS),
  );
  if (new Set(limitations).size !== limitations.length) {
    fail("request.state_limitations", "must not contain duplicates");
  }

  let planner: PlannerPayload;
  try {
    planner = parsePlanner(root.planner);
  } catch {
    fail("request.planner", "must be a valid planner payload");
  }
  if (planner.manager_id !== managerId) fail("request.manager_id", "must match planner.manager_id");

  const forecast = exactRecord(root.forecast, "request.forecast", [
    "version",
    "horizon",
    "include_doubtful",
    "validation_status",
    "price_signals_available",
    "next_price_deadline",
  ]);
  oneOf(forecast.version, "request.forecast.version", ["v2", "legacy"] as const);
  const horizon = integer(forecast.horizon, "request.forecast.horizon", 1, 5);
  if (horizon !== planner.best_action.gameweeks.length) {
    fail("request.forecast.horizon", "must match the planner horizon");
  }
  boolean(forecast.include_doubtful, "request.forecast.include_doubtful");
  boundedText(forecast.validation_status, "request.forecast.validation_status", 40);
  boolean(forecast.price_signals_available, "request.forecast.price_signals_available");
  if (forecast.next_price_deadline !== null) {
    isoUtc(forecast.next_price_deadline, "request.forecast.next_price_deadline");
  }

  const lineup = exactRecord(root.lineup, "request.lineup", [
    "gameweek",
    "formation",
    "starting_ids",
    "bench_ids",
    "captain_id",
    "vice_captain_id",
  ]);
  const lineupGameweek = integer(lineup.gameweek, "request.lineup.gameweek", 1, 38);
  const firstGameweek = planner.best_action.gameweeks[0];
  if (lineupGameweek !== planner.target_event || lineupGameweek !== firstGameweek.gameweek) {
    fail("request.lineup.gameweek", "must match the planner target event");
  }
  const formation = boundedText(lineup.formation, "request.lineup.formation", 8);
  const startingIds = uniqueIntegers(lineup.starting_ids, "request.lineup.starting_ids", 11);
  const benchIds = uniqueIntegers(lineup.bench_ids, "request.lineup.bench_ids", 4);
  const captainId = integer(lineup.captain_id, "request.lineup.captain_id", 1);
  const viceCaptainId = integer(lineup.vice_captain_id, "request.lineup.vice_captain_id", 1);
  if (
    formation !== firstGameweek.formation ||
    !sameValues(startingIds, firstGameweek.starting_ids) ||
    !sameValues([...startingIds, ...benchIds], planner.best_action.squad_ids) ||
    captainId !== firstGameweek.captain_id ||
    !startingIds.includes(viceCaptainId) ||
    captainId === viceCaptainId
  ) {
    fail("request.lineup", "must match the validated target-gameweek best action");
  }

  const finalPlayers = finalSquadReferences(planner);
  const gameweeks = new Set(planner.best_action.gameweeks.map((gameweek) => gameweek.gameweek));
  const squad = array(root.squad_context, "request.squad_context", 15, 15).map((player, index) => {
    const candidate = record(player, `request.squad_context[${index}]`);
    const id = integer(candidate.id, `request.squad_context[${index}].id`, 1);
    const reference = finalPlayers.get(id);
    if (!reference) fail(`request.squad_context[${index}].id`, "must belong to the best-action squad");
    return validateSquadPlayer(
      player,
      `request.squad_context[${index}]`,
      reference,
      gameweeks,
      horizon,
    );
  });
  const squadIds = squad.map((player) => player.id);
  if (new Set(squadIds).size !== 15 || !sameValues(squadIds, planner.best_action.squad_ids)) {
    fail("request.squad_context", "must contain each best-action player exactly once");
  }

  return {
    ...root,
    planner,
  } as unknown as AiReviewRequest;
}

export function assertAiReviewRequestIsFresh(
  request: AiReviewRequest,
  now = new Date(),
): void {
  const nowMs = now.getTime();
  if (!Number.isFinite(nowMs)) fail("now", "must be a valid date");

  const deadlineMs = new Date(request.target_deadline).getTime();
  if (deadlineMs <= nowMs) {
    fail("request.target_deadline", "has passed; synchronize and calculate a new plan");
  }

  const generatedMs = new Date(request.recommendation_generated_at).getTime();
  if (generatedMs - nowMs > 5 * 60 * 1_000) {
    fail("request.recommendation_generated_at", "cannot be more than five minutes in the future");
  }
  if (nowMs - generatedMs > MAX_AI_REVIEW_RECOMMENDATION_AGE_MS) {
    fail("request.recommendation_generated_at", "is older than 24 hours; calculate a new plan");
  }

  const observedMs = new Date(request.state_observed_at).getTime();
  if (observedMs - nowMs > 5 * 60 * 1_000) {
    fail("request.state_observed_at", "cannot be more than five minutes in the future");
  }
  if (nowMs - observedMs > MAX_AI_REVIEW_RECOMMENDATION_AGE_MS) {
    fail("request.state_observed_at", "is older than 24 hours; synchronize the manager state");
  }
}

function boundedTextArray(
  value: unknown,
  path: string,
  minimum: number,
  maximum: number,
  itemMaximum: number,
): string[] {
  return array(value, path, minimum, maximum).map((item, index) =>
    boundedText(item, `${path}[${index}]`, itemMaximum),
  );
}

export function parseAiReviewModelOutput(value: unknown, alternativeCount: number): AiReviewModelOutput {
  const root = exactRecord(value, "review", [
    "verdict",
    "alternative_index",
    "headline",
    "summary",
    "rationale",
    "risks",
    "change_triggers",
    "deadline_checklist",
    "evidence_summary",
    "data_gaps",
    "confidence",
  ]);
  const verdict = oneOf(root.verdict, "review.verdict", [
    "confirm_best_action",
    "wait_for_information",
    "prefer_alternative",
  ] as const);
  const alternativeIndex = root.alternative_index === null
    ? null
    : integer(root.alternative_index, "review.alternative_index", 0, Math.max(0, alternativeCount - 1));
  if (verdict === "prefer_alternative" && (alternativeIndex === null || alternativeIndex >= alternativeCount)) {
    fail("review.alternative_index", "must reference a returned solver alternative");
  }
  if (verdict !== "prefer_alternative" && alternativeIndex !== null) {
    fail("review.alternative_index", "must be null unless an alternative is preferred");
  }
  boundedText(root.headline, "review.headline", 140);
  boundedText(root.summary, "review.summary", 700);
  boundedTextArray(root.rationale, "review.rationale", 2, 4, 280);
  boundedTextArray(root.risks, "review.risks", 0, 4, 280);
  boundedTextArray(root.change_triggers, "review.change_triggers", 0, 4, 280);
  boundedTextArray(root.deadline_checklist, "review.deadline_checklist", 2, 5, 240);
  boundedText(root.evidence_summary, "review.evidence_summary", 800);
  boundedTextArray(root.data_gaps, "review.data_gaps", 0, 4, 280);
  oneOf(root.confidence, "review.confidence", ["low", "medium", "high"] as const);
  return root as unknown as AiReviewModelOutput;
}

export function isAllowedAiReviewSourceUrl(value: string): boolean {
  try {
    const parsed = new URL(value);
    if (parsed.protocol !== "https:" || parsed.username || parsed.password) return false;
    const host = parsed.hostname.toLowerCase();
    return AI_REVIEW_ALLOWED_SOURCE_DOMAINS.some(
      (domain) => host === domain || host.endsWith(`.${domain}`),
    );
  } catch {
    return false;
  }
}

export function parseAiReviewResponse(value: unknown, alternativeCount = 4): AiReviewResponse {
  const root = exactRecord(value, "response", [
    "schema_version",
    "generated_at",
    "recommendation_generated_at",
    "target_event",
    "model",
    "review",
    "research",
  ]);
  literal(root.schema_version, "response.schema_version", AI_REVIEW_RESPONSE_SCHEMA_VERSION);
  isoUtc(root.generated_at, "response.generated_at");
  isoUtc(root.recommendation_generated_at, "response.recommendation_generated_at");
  integer(root.target_event, "response.target_event", 1, 38);
  boundedText(root.model, "response.model", 80);
  const research = exactRecord(root.research, "response.research", ["performed", "sources"]);
  const performed = boolean(research.performed, "response.research.performed");
  const sources = array(research.sources, "response.research.sources", 0, 8).map((source, index) => {
    const row = exactRecord(source, `response.research.sources[${index}]`, ["title", "url"]);
    boundedText(row.title, `response.research.sources[${index}].title`, 180);
    const url = boundedText(row.url, `response.research.sources[${index}].url`, 2_000);
    if (!isAllowedAiReviewSourceUrl(url)) {
      fail(`response.research.sources[${index}].url`, "must use an allowed HTTPS source");
    }
    return row as unknown as AiReviewSource;
  });
  if (!performed) fail("response.research.performed", "must be true for an AI review");
  if (!sources.length) fail("response.research.sources", "must contain at least one allowed research source");
  if (new Set(sources.map((source) => source.url)).size !== sources.length) {
    fail("response.research.sources", "must not contain duplicate URLs");
  }
  const review = parseAiReviewModelOutput(root.review, alternativeCount);
  return {
    ...root,
    review,
  } as unknown as AiReviewResponse;
}

export function actionForAiReview(
  planner: PlannerPayload,
  review: AiReviewModelOutput,
): PlannerAction | null {
  if (review.verdict === "wait_for_information") return null;
  if (review.verdict === "prefer_alternative") {
    return review.alternative_index === null
      ? null
      : planner.alternatives[review.alternative_index] ?? null;
  }
  return planner.best_action;
}
