/** Observed wildcard outcomes among peers, never a causal value for the chip. */
export type BenchmarkHorizon = 1 | 3 | 5;
export interface Distribution { p25: number; median: number; p75: number }
export interface BenchmarkExclusions {
  unfinished: number;
  season_end: number;
  other_chip: number;
  missing_data: number;
  insufficient_history: number;
  insufficient_peers: number;
}
export interface WildcardBenchmarkWindow {
  horizon: BenchmarkHorizon;
  sample_size: number;
  net_points: Distribution | null;
  peer_difference: Distribution | null;
  reference_count: { min: number; median: number; max: number } | null;
  exclusions: BenchmarkExclusions;
}
export type OwnWildcardStatus = "not_used" | "pending" | "season_end" | "other_chip" | "missing_data"
  | "insufficient_history" | "insufficient_peers" | "ready";
export interface OwnWildcardWindow {
  horizon: BenchmarkHorizon;
  status: OwnWildcardStatus;
  net_points: number | null;
  peer_difference: number | null;
  percentile: number | null;
  reference_count: number | null;
}
export interface WildcardBenchmarkAnalysis {
  half: 1 | 2;
  last_finalized_event: number | null;
  detected_wildcards: number;
  valid_peer_histories: number;
  invalid_peer_histories: number;
  windows: WildcardBenchmarkWindow[];
  own: { used_event: number | null; windows: OwnWildcardWindow[] };
}

type ChipName = "wildcard" | "freehit" | "bboost" | "3xc";
type ChipUse = { name: ChipName; event: number };
interface History { points: Map<number, number>; chips: ChipUse[] }
interface Observation {
  status: Exclude<OwnWildcardStatus, "not_used">;
  net: number | null;
  difference: number | null;
  references: number | null;
}
const HORIZONS: readonly BenchmarkHorizon[] = [1, 3, 5];
const CHIP_NAMES: readonly string[] = ["wildcard", "freehit", "bboost", "3xc"];

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown> : null;
}
function integer(value: unknown): value is number { return typeof value === "number" && Number.isSafeInteger(value); }
function eventNumber(value: unknown): value is number { return integer(value) && value >= 1 && value <= 38; }
function chipHalf(event: number): 1 | 2 { return event <= 19 ? 1 : 2; }

function finalizedEvents(bootstrap: unknown): Set<number> {
  const events = record(bootstrap)?.events;
  if (!Array.isArray(events) || events.length === 0) throw new TypeError("Missing FPL calendar");
  const seen = new Set<number>();
  const finalized = new Set<number>();
  for (const value of events) {
    const row = record(value);
    if (!row || !eventNumber(row.id) || seen.has(row.id)
      || typeof row.finished !== "boolean" || typeof row.data_checked !== "boolean") {
      throw new TypeError("Invalid FPL calendar");
    }
    seen.add(row.id);
    if (row.finished && row.data_checked) finalized.add(row.id);
  }
  return finalized;
}

function parseHistory(value: unknown): History | null {
  const source = record(value);
  if (!source || !Array.isArray(source.current) || !Array.isArray(source.chips)) return null;
  const points = new Map<number, number>();
  for (const value of source.current) {
    const row = record(value);
    if (!row || !eventNumber(row.event) || points.has(row.event) || !integer(row.points)
      || !integer(row.event_transfers_cost) || row.event_transfers_cost < 0 || row.event_transfers_cost % 4 !== 0) return null;
    const net = row.points - row.event_transfers_cost;
    if (!Number.isSafeInteger(net)) return null;
    points.set(row.event, net);
  }
  const chips: ChipUse[] = [];
  const usedEvents = new Set<number>();
  const usedHalves = new Set<string>();
  for (const value of source.chips) {
    const row = record(value);
    if (!row || typeof row.name !== "string" || !CHIP_NAMES.includes(row.name) || !eventNumber(row.event)) return null;
    if (row.event === 1 && (row.name === "wildcard" || row.name === "freehit")) return null;
    const identity = `${row.name}:${chipHalf(row.event)}`;
    if (usedEvents.has(row.event) || usedHalves.has(identity)) return null;
    usedEvents.add(row.event);
    usedHalves.add(identity);
    chips.push({ name: row.name as ChipName, event: row.event });
  }
  const freeHits = new Set(chips.filter(chip => chip.name === "freehit").map(chip => chip.event));
  if (freeHits.has(19) && freeHits.has(20)) return null;
  return { points, chips };
}

function total(history: History, first: number, last: number): number | null {
  let result = 0;
  for (let event = first; event <= last; event++) {
    const points = history.points.get(event);
    if (points === undefined) return null;
    result += points;
  }
  return Number.isSafeInteger(result) ? result : null;
}
function chipsIn(history: History, first: number, last: number): ChipUse[] {
  return history.chips.filter(chip => chip.event >= first && chip.event <= last);
}
function quantile(values: readonly number[], probability: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const fraction = position - lower;
  return sorted[lower] + fraction * (sorted[Math.ceil(position)] - sorted[lower]);
}
function distribution(values: readonly number[]): Distribution {
  return { p25: quantile(values, 0.25), median: quantile(values, 0.5), p75: quantile(values, 0.75) };
}
function wildcardEvent(history: History, half: 1 | 2): number | null {
  return history.chips.find(chip => chip.name === "wildcard" && chipHalf(chip.event) === half)?.event ?? null;
}

function observe(
  entry: number, history: History, start: number, horizon: BenchmarkHorizon,
  peers: ReadonlyMap<number, History>, finalized: ReadonlySet<number>,
): Observation {
  const empty = { net: null, difference: null, references: null };
  const end = start + horizon - 1;
  if (end > 38) return { ...empty, status: "season_end" };
  for (let event = start; event <= end; event++) {
    if (!finalized.has(event)) return { ...empty, status: "pending" };
  }
  const net = total(history, start, end);
  if (net === null) return { ...empty, status: "missing_data" };
  if (chipsIn(history, start, end).some(chip => chip.name !== "wildcard" || chip.event !== start)) {
    return { ...empty, net, status: "other_chip" };
  }
  const preLength = Math.min(3, start - 1);
  const before = total(history, start - preLength, start - 1);
  if (preLength < 2 || before === null) return { ...empty, net, status: "insufficient_history" };
  for (let event = start - preLength; event < start; event++) {
    if (!finalized.has(event)) return { ...empty, net, status: "insufficient_history" };
  }
  const matches: { entry: number; distance: number; points: number }[] = [];
  for (const [peerEntry, peer] of peers) {
    if (peerEntry === entry || chipsIn(peer, start, end).length) continue;
    const peerBefore = total(peer, start - preLength, start - 1);
    const peerAfter = total(peer, start, end);
    if (peerBefore === null || peerAfter === null) continue;
    // Equal calendar windows have equal lengths. Comparing integer totals
    // avoids floating-point edge cases at exactly five points per gameweek.
    const distance = Math.abs(peerBefore - before);
    if (distance <= 5 * preLength) matches.push({ entry: peerEntry, distance, points: peerAfter });
  }
  matches.sort((a, b) => a.distance - b.distance || a.entry - b.entry);
  const references = matches.slice(0, 10);
  if (references.length < 5) return { ...empty, net, references: references.length, status: "insufficient_peers" };
  return { status: "ready", net, difference: net - quantile(references.map(peer => peer.points), 0.5), references: references.length };
}

export function analyzeWildcardBenchmarks(input: {
  bootstrap: unknown;
  histories: ReadonlyMap<number, unknown>;
  ownManagerId: number;
  half: 1 | 2;
}): WildcardBenchmarkAnalysis {
  if (!integer(input.ownManagerId) || input.ownManagerId < 1 || (input.half !== 1 && input.half !== 2)) {
    throw new TypeError("Invalid wildcard benchmark input");
  }
  const finalized = finalizedEvents(input.bootstrap);
  const peers = new Map<number, History>();
  let invalidPeerHistories = 0;
  for (const [entry, value] of input.histories) {
    if (entry === input.ownManagerId) continue;
    const history = integer(entry) && entry > 0 ? parseHistory(value) : null;
    if (history) peers.set(entry, history);
    else invalidPeerHistories++;
  }
  const cases: { entry: number; history: History; start: number }[] = [];
  for (const [entry, history] of peers) {
    const start = wildcardEvent(history, input.half);
    if (start !== null) cases.push({ entry, history, start });
  }
  const ownHistory = parseHistory(input.histories.get(input.ownManagerId));
  const ownEvent = ownHistory ? wildcardEvent(ownHistory, input.half) : null;
  const ownWindows: OwnWildcardWindow[] = [];
  const windows = HORIZONS.map((horizon): WildcardBenchmarkWindow => {
    const exclusions: BenchmarkExclusions = { unfinished: 0, season_end: 0, other_chip: 0, missing_data: 0, insufficient_history: 0, insufficient_peers: 0 };
    const eligible: Observation[] = [];
    for (const example of cases) {
      const outcome = observe(example.entry, example.history, example.start, horizon, peers, finalized);
      if (outcome.status === "ready") eligible.push(outcome);
      else if (outcome.status === "pending") exclusions.unfinished++;
      else exclusions[outcome.status]++;
    }
    const net = eligible.map(outcome => outcome.net!);
    const differences = eligible.map(outcome => outcome.difference!);
    const references = eligible.map(outcome => outcome.references!);
    let own: OwnWildcardWindow = { horizon, status: ownHistory ? "not_used" : "missing_data", net_points: null,
      peer_difference: null, percentile: null, reference_count: null };
    if (ownHistory && ownEvent !== null) {
      const outcome = observe(input.ownManagerId, ownHistory, ownEvent, horizon, peers, finalized);
      own = { horizon, status: outcome.status, net_points: outcome.net, peer_difference: outcome.difference,
        percentile: null, reference_count: outcome.references };
      if (outcome.status === "ready" && differences.length >= 10) {
        const lower = differences.filter(value => value < outcome.difference!).length;
        const equal = differences.filter(value => value === outcome.difference!).length;
        own.percentile = 100 * (lower + equal / 2) / differences.length;
      }
    }
    ownWindows.push(own);
    const sufficient = eligible.length >= 5;
    return { horizon, sample_size: eligible.length, net_points: sufficient ? distribution(net) : null,
      peer_difference: sufficient ? distribution(differences) : null,
      reference_count: sufficient ? { min: Math.min(...references), median: quantile(references, 0.5), max: Math.max(...references) } : null,
      exclusions };
  });
  return { half: input.half, last_finalized_event: finalized.size ? Math.max(...finalized) : null,
    detected_wildcards: cases.length, valid_peer_histories: peers.size, invalid_peer_histories: invalidPeerHistories,
    windows, own: { used_event: ownEvent, windows: ownWindows } };
}
