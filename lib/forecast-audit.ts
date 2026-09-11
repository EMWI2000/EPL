/** Point-in-time, device-local forecast evaluation. This is not tamper-proof storage. */
export const FORECAST_AUDIT_STORAGE_KEY = "fpl-forecast-audit-v1";
export const FORECAST_AUDIT_SCHEMA_VERSION = "fpl-forecast-audit-v1" as const;
const MAX_ENTRIES = 76;
const MAX_BYTES = 2 * 1024 * 1024;

export interface ForecastAuditPlayer {
  id: number;
  name: string;
  model_points: number;
  model_minutes: number;
  baseline_points: number | null;
  baseline_minutes: number | null;
}

export interface ForecastAuditCapture {
  target_event: number;
  target_deadline: string;
  generated_at: string;
  model_version: string;
  players: ForecastAuditPlayer[];
}

export interface ForecastAuditResults {
  season: string;
  target_event: number;
  target_deadline: string;
  checked_at: string;
  finalized: true;
  players: { id: number; points: number; minutes: number }[];
}

export interface ForecastAuditEntry extends ForecastAuditCapture {
  season: string;
  captured_at: string;
  results: ForecastAuditResults | null;
}

export interface ForecastAuditJournal {
  schema_version: typeof FORECAST_AUDIT_SCHEMA_VERSION;
  entries: ForecastAuditEntry[];
}

export interface ForecastAuditStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

function object(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new TypeError("Expected an object");
  return value as Record<string, unknown>;
}

function exact(value: unknown, keys: string[]): Record<string, unknown> {
  const row = object(value);
  if (Object.keys(row).length !== keys.length || keys.some((key) => !Object.hasOwn(row, key))) {
    throw new TypeError("Unexpected or missing forecast audit fields");
  }
  return row;
}

function number(value: unknown, min: number, max: number, integer = false): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max || (integer && !Number.isSafeInteger(value))) {
    throw new TypeError("Invalid forecast audit number");
  }
  return value;
}

function text(value: unknown, max: number): string {
  if (typeof value !== "string" || !value.trim() || value.length > max || /[\u0000-\u001f\u007f]/.test(value)) {
    throw new TypeError("Invalid forecast audit text");
  }
  return value.trim();
}

function timestamp(value: unknown): string {
  const raw = text(value, 40);
  const date = new Date(raw);
  if (!/(?:Z|[+-]\d\d:\d\d)$/i.test(raw) || !Number.isFinite(date.getTime())) throw new TypeError("Invalid forecast audit timestamp");
  return date.toISOString();
}

/** FPL seasons span July–June. Exact event deadlines are also checked at evaluation. */
export function forecastAuditSeason(deadline: string): string {
  const date = new Date(timestamp(deadline));
  const start = date.getUTCFullYear() - (date.getUTCMonth() < 6 ? 1 : 0);
  return `${start}-${String(start + 1).slice(-2)}`;
}

function uniquePlayers<T extends { id: number }>(rows: T[], maximum: number): T[] {
  if (rows.length === 0 || rows.length > maximum || new Set(rows.map((row) => row.id)).size !== rows.length) {
    throw new TypeError("Invalid forecast audit player coverage");
  }
  return rows;
}

export function parseForecastAuditCapture(value: unknown): ForecastAuditCapture {
  const row = exact(value, ["target_event", "target_deadline", "generated_at", "model_version", "players"]);
  if (!Array.isArray(row.players)) throw new TypeError("Missing forecast audit players");
  const players = uniquePlayers(row.players.map((value) => {
    const player = exact(value, ["id", "name", "model_points", "model_minutes", "baseline_points", "baseline_minutes"]);
    return {
      id: number(player.id, 1, 100_000, true),
      name: text(player.name, 80),
      model_points: number(player.model_points, -50, 200),
      model_minutes: number(player.model_minutes, 0, 360),
      baseline_points: player.baseline_points === null ? null : number(player.baseline_points, -50, 200),
      baseline_minutes: player.baseline_minutes === null ? null : number(player.baseline_minutes, 0, 360),
    };
  }), 100);
  const result = {
    target_event: number(row.target_event, 1, 38, true),
    target_deadline: timestamp(row.target_deadline),
    generated_at: timestamp(row.generated_at),
    model_version: text(row.model_version, 40),
    players,
  };
  if (Date.parse(result.generated_at) >= Date.parse(result.target_deadline)) throw new TypeError("Forecast was generated after its deadline");
  return result;
}

export function parseForecastAuditResults(value: unknown): ForecastAuditResults {
  const row = exact(value, ["season", "target_event", "target_deadline", "checked_at", "finalized", "players"]);
  const deadline = timestamp(row.target_deadline);
  const checkedAt = timestamp(row.checked_at);
  if (row.season !== forecastAuditSeason(deadline) || row.finalized !== true || Date.parse(checkedAt) <= Date.parse(deadline)) {
    throw new TypeError("Results must be finalized for the matching season");
  }
  if (!Array.isArray(row.players)) throw new TypeError("Missing result players");
  return {
    season: row.season,
    target_event: number(row.target_event, 1, 38, true),
    target_deadline: deadline,
    checked_at: checkedAt,
    finalized: true,
    players: uniquePlayers(row.players.map((value) => {
      const player = exact(value, ["id", "points", "minutes"]);
      return {
        id: number(player.id, 1, 100_000, true),
        points: number(player.points, -50, 200, true),
        minutes: number(player.minutes, 0, 360, true),
      };
    }), 1_500),
  };
}

export function emptyForecastAudit(): ForecastAuditJournal {
  return { schema_version: FORECAST_AUDIT_SCHEMA_VERSION, entries: [] };
}

export function parseForecastAudit(value: unknown): ForecastAuditJournal {
  const root = exact(value, ["schema_version", "entries"]);
  if (root.schema_version !== FORECAST_AUDIT_SCHEMA_VERSION || !Array.isArray(root.entries) || root.entries.length > MAX_ENTRIES) {
    throw new TypeError("Invalid forecast audit journal");
  }
  const entries = root.entries.map((value) => {
    const row = exact(value, ["season", "captured_at", "results", "target_event", "target_deadline", "generated_at", "model_version", "players"]);
    const { season, captured_at, results, ...capture } = row;
    const clean = parseForecastAuditCapture(capture);
    const capturedAt = timestamp(captured_at);
    if (season !== forecastAuditSeason(clean.target_deadline) || Date.parse(capturedAt) >= Date.parse(clean.target_deadline)
      || Date.parse(capturedAt) < Date.parse(clean.generated_at)) {
      throw new TypeError("Forecast was not captured before the deadline");
    }
    const result = results === null ? null : parseForecastAuditResults(results);
    if (result !== null && (result.season !== season || result.target_event !== clean.target_event || result.target_deadline !== clean.target_deadline)) {
      throw new TypeError("Results do not match the saved forecast");
    }
    return { ...clean, season: season as string, captured_at: capturedAt, results: result };
  });
  const keys = entries.map((row) => `${row.season}:${row.target_event}`);
  if (new Set(keys).size !== keys.length) throw new TypeError("Duplicate forecast audit gameweeks");
  return { schema_version: FORECAST_AUDIT_SCHEMA_VERSION, entries };
}

/** First capture wins, even if a later recommendation or model version differs. */
export function captureForecastAudit(journal: ForecastAuditJournal, capture: ForecastAuditCapture, now = new Date().toISOString()): ForecastAuditJournal {
  const current = parseForecastAudit(journal);
  const clean = parseForecastAuditCapture(capture);
  const capturedAt = timestamp(now);
  const season = forecastAuditSeason(clean.target_deadline);
  if (current.entries.some((row) => row.season === season && row.target_event === clean.target_event)) return current;
  if (Date.parse(capturedAt) >= Date.parse(clean.target_deadline) || Date.parse(capturedAt) < Date.parse(clean.generated_at)) {
    throw new TypeError("Only forecasts available before the deadline may be saved");
  }
  if (current.entries.length >= MAX_ENTRIES) throw new TypeError("Forecast audit storage is full");
  return parseForecastAudit({
    schema_version: FORECAST_AUDIT_SCHEMA_VERSION,
    entries: [...current.entries, { ...clean, season, captured_at: capturedAt, results: null }]
      .sort((left, right) => Date.parse(right.target_deadline) - Date.parse(left.target_deadline)),
  });
}

export function addForecastAuditResults(journal: ForecastAuditJournal, results: ForecastAuditResults): ForecastAuditJournal {
  const current = parseForecastAudit(journal);
  const clean = parseForecastAuditResults(results);
  const matching = current.entries.find((row) => row.season === clean.season && row.target_event === clean.target_event);
  if (!matching || matching.target_deadline !== clean.target_deadline) throw new TypeError("No matching pre-deadline forecast");
  const owned = new Set(matching.players.map((row) => row.id));
  const trimmed = { ...clean, players: clean.players.filter((row) => owned.has(row.id)) };
  if (trimmed.players.length === 0) throw new TypeError("No matching result players");
  return parseForecastAudit({ ...current, entries: current.entries.map((row) => row === matching ? { ...row, results: trimmed } : row) });
}

export interface ForecastAuditMetric {
  count: number;
  model_mae: number;
  baseline_mae: number;
  model_bias: number;
  baseline_bias: number;
}

/** Compare both methods on exactly the same players; missing results are never zero-filled. */
export function evaluateForecastAudit(entry: ForecastAuditEntry): { total: number; points: ForecastAuditMetric | null; minutes: ForecastAuditMetric | null } {
  const clean = parseForecastAudit({ schema_version: FORECAST_AUDIT_SCHEMA_VERSION, entries: [entry] }).entries[0];
  const actuals = new Map(clean.results?.players.map((row) => [row.id, row]) ?? []);
  const metric = (kind: "points" | "minutes"): ForecastAuditMetric | null => {
    const errors = clean.players.flatMap((player) => {
      const actual = actuals.get(player.id);
      const baseline = player[kind === "points" ? "baseline_points" : "baseline_minutes"];
      if (!actual || baseline === null) return [];
      return [{ model: player[kind === "points" ? "model_points" : "model_minutes"] - actual[kind], baseline: baseline - actual[kind] }];
    });
    if (errors.length === 0) return null;
    const mean = (key: "model" | "baseline", absolute: boolean) => errors.reduce((sum, error) => sum + (absolute ? Math.abs(error[key]) : error[key]), 0) / errors.length;
    return { count: errors.length, model_mae: mean("model", true), baseline_mae: mean("baseline", true), model_bias: mean("model", false), baseline_bias: mean("baseline", false) };
  };
  return { total: clean.players.length, points: metric("points"), minutes: metric("minutes") };
}

export function readForecastAudit(storage: ForecastAuditStorage): { journal: ForecastAuditJournal; error: string | null } {
  try {
    const raw = storage.getItem(FORECAST_AUDIT_STORAGE_KEY);
    if (raw === null) return { journal: emptyForecastAudit(), error: null };
    if (new TextEncoder().encode(raw).byteLength > MAX_BYTES) throw new TypeError("Oversized journal");
    return { journal: parseForecastAudit(JSON.parse(raw)), error: null };
  } catch {
    return { journal: emptyForecastAudit(), error: "Den lokale målejournal kunne ikke læses. Eksisterende data er ikke overskrevet." };
  }
}

export function writeForecastAudit(storage: ForecastAuditStorage, journal: ForecastAuditJournal): boolean {
  try {
    const serialized = JSON.stringify(parseForecastAudit(journal));
    if (new TextEncoder().encode(serialized).byteLength > MAX_BYTES) return false;
    storage.setItem(FORECAST_AUDIT_STORAGE_KEY, serialized);
    return true;
  } catch {
    return false;
  }
}

/** Project only official finalized scores. Never use current bootstrap forecasts as baselines. */
export function buildForecastAuditResults(bootstrap: unknown, live: unknown, event: number, deadline: string, checkedAt: string): ForecastAuditResults {
  const root = object(bootstrap);
  if (!Array.isArray(root.events)) throw new TypeError("Missing FPL events");
  const matching = root.events.map(object).find((row) => row.id === event);
  if (!matching || timestamp(matching.deadline_time) !== timestamp(deadline)) throw new TypeError("The FPL season or deadline no longer matches");
  if (matching.finished !== true || matching.data_checked !== true) throw new TypeError("The gameweek has not been finalized");
  const scores = object(live);
  if (!Array.isArray(scores.elements)) throw new TypeError("Missing FPL live scores");
  return parseForecastAuditResults({
    season: forecastAuditSeason(deadline), target_event: event, target_deadline: deadline, checked_at: checkedAt, finalized: true,
    players: scores.elements.map((value) => {
      const player = object(value);
      const stats = object(player.stats);
      return { id: player.id, points: stats.total_points, minutes: stats.minutes };
    }),
  });
}
