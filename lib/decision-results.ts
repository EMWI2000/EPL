import { decisionSeason, selectedDecisionPlan, type DecisionHistoryEntry } from "./decision-history.ts";
import type { ChipName } from "./planner-contract.ts";

export type DecisionMatch = "followed" | "deviated" | "unknown";
export interface OfficialPlayerScore {
  id: number;
  position: "GKP" | "DEF" | "MID" | "FWD";
  minutes: number;
  points: number;
}
export interface ActualDecisionResult {
  schema_version: "fpl-decision-result-v1";
  manager_id: number;
  target_event: number;
  target_deadline: string;
  season: string;
  fetched_at: string;
  finalized: boolean;
  player_scores: OfficialPlayerScore[] | null;
  actual: {
    gross_points: number;
    transfer_cost: number;
    net_points: number;
    bank_tenths: number;
    active_chip: ChipName | null;
    squad_ids: number[];
    starting_ids: number[];
    bench_ids: number[];
    captain_id: number;
    vice_captain_id: number;
    automatic_substitutions: number;
    transfers: { out_id: number; in_id: number; time: string | null }[];
  } | null;
}
export interface DecisionComparison {
  status: DecisionMatch;
  transfer_match: DecisionMatch;
  squad_match: DecisionMatch;
  lineup_match: DecisionMatch;
  chip_match: DecisionMatch;
  reasons: string[];
  gross_points: number | null;
  transfer_cost: number | null;
  net_points: number | null;
}
type Obj = Record<string, unknown>;
const object = (v: unknown): Obj => {
  if (!v || typeof v !== "object" || Array.isArray(v)) throw new TypeError("Invalid official FPL data");
  return v as Obj;
};
function int(v: unknown, min: number, max: number): number {
  if (!Number.isSafeInteger(v) || (v as number) < min || (v as number) > max) throw new TypeError("Invalid official FPL number");
  return v as number;
}
function array(v: unknown): unknown[] {
  if (!Array.isArray(v)) throw new TypeError("Invalid official FPL collection");
  return v;
}
function time(v: unknown): string | null {
  return typeof v === "string" && Number.isFinite(Date.parse(v)) ? new Date(v).toISOString() : null;
}
function sameSet(a: number[], b: number[]): boolean {
  return a.length === b.length && new Set(a).size === a.length && a.every(id => b.includes(id));
}
const chipNames = ["wildcard", "freehit", "bboost", "3xc"] as const;

function publicPlayerScores(bootstrap: Obj, liveValue: unknown): OfficialPlayerScore[] {
  const positions = new Map(array(bootstrap.elements).map(value => {
    const p = object(value);
    return [int(p.id, 1, Number.MAX_SAFE_INTEGER), int(p.element_type, 1, 4)];
  }));
  const elements = array(object(liveValue).elements);
  if (elements.length > 1500) throw new TypeError("Official player score limit exceeded");
  const scores = elements.map(value => {
    const player = object(value), stats = object(player.stats);
    const id = int(player.id, 1, Number.MAX_SAFE_INTEGER);
    const position = positions.get(id);
    if (!position) throw new TypeError("Official score has no player position");
    return { id, position: (["GKP", "DEF", "MID", "FWD"] as const)[position - 1],
      minutes: int(stats.minutes, 0, 720), points: int(stats.total_points, -100, 300) };
  });
  if (new Set(scores.map(p => p.id)).size !== scores.length) throw new TypeError("Duplicate official player scores");
  return scores;
}

/** Restore the manager's submitted order before FPL's automatic substitutions. */
export function parseActualPicks(picksValue: unknown, bootstrapValue: unknown, target: number): NonNullable<ActualDecisionResult["actual"]> {
  const payload = object(picksValue), bootstrap = object(bootstrapValue);
  const history = object(payload.entry_history);
  if (history.event !== target) throw new TypeError("FPL returned a different gameweek");
  const positions = new Map(array(bootstrap.elements).map(p => {
    const row = object(p); return [int(row.id, 1, Number.MAX_SAFE_INTEGER), int(row.element_type, 1, 4)];
  }));
  const picks = array(payload.picks).map(v => {
    const p = object(v);
    if (typeof p.is_captain !== "boolean" || typeof p.is_vice_captain !== "boolean") throw new TypeError("Invalid captain flags");
    return { id: int(p.element, 1, Number.MAX_SAFE_INTEGER), position: int(p.position, 1, 15),
      captain: p.is_captain, vice: p.is_vice_captain };
  }).sort((a, b) => a.position - b.position);
  if (picks.length !== 15 || new Set(picks.map(p => p.id)).size !== 15
    || picks.some((p, i) => p.position !== i + 1 || !positions.has(p.id))
    || picks.filter(p => p.captain).length !== 1 || picks.filter(p => p.vice).length !== 1
    || picks.some(p => p.captain && p.vice)) throw new TypeError("Invalid official squad");
  const substitutions = array(payload.automatic_subs);
  if (substitutions.length > 4) throw new TypeError("Invalid automatic substitutions");
  for (const subValue of [...substitutions].reverse()) {
    const sub = object(subValue);
    if (sub.event !== target) throw new TypeError("Automatic substitution from wrong gameweek");
    const incoming = picks.findIndex(p => p.id === sub.element_in);
    const outgoing = picks.findIndex(p => p.id === sub.element_out);
    if (incoming < 0 || outgoing < 0 || incoming > 10 || outgoing < 11) throw new TypeError("Inconsistent automatic substitution");
    [picks[incoming], picks[outgoing]] = [picks[outgoing], picks[incoming]];
  }
  const bench = picks.slice(11);
  const benchIds = [...bench.filter(p => positions.get(p.id) !== 1), ...bench.filter(p => positions.get(p.id) === 1)].map(p => p.id);
  const gross = int(history.points, -1000, 1000), cost = int(history.event_transfers_cost, 0, 1000);
  if (cost % 4 !== 0) throw new TypeError("Invalid transfer cost");
  if (payload.active_chip !== null && !chipNames.includes(payload.active_chip as ChipName)) throw new TypeError("Unknown active chip");
  return { gross_points: gross, transfer_cost: cost, net_points: gross - cost,
    bank_tenths: int(history.bank, 0, 5000), active_chip: payload.active_chip as ChipName | null,
    squad_ids: picks.map(p => p.id), starting_ids: picks.slice(0, 11).map(p => p.id), bench_ids: benchIds,
    captain_id: picks.find(p => p.captain)!.id, vice_captain_id: picks.find(p => p.vice)!.id,
    automatic_substitutions: substitutions.length, transfers: [] };
}
export async function loadDecisionResult(managerId: number, event: number, fetcher: typeof fetch = fetch, now = Date.now()): Promise<ActualDecisionResult> {
  int(managerId, 1, Number.MAX_SAFE_INTEGER); int(event, 1, 38);
  const get = async (path: string): Promise<unknown> => {
    const response = await fetcher(`https://fantasy.premierleague.com/api/${path}`, {
      cache: "no-store", signal: AbortSignal.timeout(7000), headers: { Accept: "application/json" },
    });
    if (!response.ok) throw new Error("Official FPL request failed");
    return response.json();
  };
  const bootstrap = object(await get("bootstrap-static/"));
  const official = array(bootstrap.events).map(object).find(row => row.id === event);
  const deadline = time(official?.deadline_time);
  if (!official || !deadline) throw new TypeError("Official gameweek is unavailable");
  const finalized = official.finished === true && official.data_checked === true && Date.parse(deadline) < now;
  const result: ActualDecisionResult = {
    schema_version: "fpl-decision-result-v1", manager_id: managerId, target_event: event,
    target_deadline: deadline, season: decisionSeason(deadline), fetched_at: new Date(now).toISOString(), finalized, actual: null, player_scores: null,
  };
  if (!finalized) return result;
  const [picks, transfers, live] = await Promise.all([
    get(`entry/${managerId}/event/${event}/picks/`), get(`entry/${managerId}/transfers/`), get(`event/${event}/live/`),
  ]);
  const actual = parseActualPicks(picks, bootstrap, event);
  actual.transfers = array(transfers).map(object).filter(row => row.event === event).map(row => ({
    out_id: int(row.element_out, 1, Number.MAX_SAFE_INTEGER), in_id: int(row.element_in, 1, Number.MAX_SAFE_INTEGER), time: time(row.time),
  }));
  return { ...result, actual, player_scores: publicPlayerScores(bootstrap, live) };
}
export function compareDecisionResult(entry: DecisionHistoryEntry, result: ActualDecisionResult): DecisionComparison {
  const comparison: DecisionComparison = { status: "unknown", transfer_match: "unknown", squad_match: "unknown",
    lineup_match: "unknown", chip_match: "unknown", reasons: [],
    gross_points: null, transfer_cost: null, net_points: null };
  if (!result.finalized || !result.actual) { comparison.reasons.push("Runden er ikke færdigbehandlet hos FPL."); return comparison; }
  if (entry.manager_id !== result.manager_id || entry.season !== result.season || entry.target_event !== result.target_event) {
    comparison.reasons.push("Snapshot og resultat kan ikke knyttes sikkert til samme manager, sæson og runde."); return comparison;
  }
  comparison.gross_points = result.actual.gross_points;
  comparison.transfer_cost = result.actual.transfer_cost;
  comparison.net_points = result.actual.net_points;
  if (Date.parse(entry.saved_at) >= Date.parse(result.target_deadline)) {
    comparison.reasons.push("Snapshot blev ikke gemt før den officielle deadline."); return comparison;
  }
  const plan = selectedDecisionPlan(entry), actual = result.actual;
  if (entry.origin !== "recorded_v2" || !plan || !entry.context) {
    comparison.reasons.push(entry.selection.kind === "wait" ? "Der blev gemt et valg om at afvente, ikke en bestemt trup." : "Det ældre snapshot mangler de nødvendige detaljer.");
    return comparison;
  }
  const match = (same: boolean): DecisionMatch => same ? "followed" : "deviated";
  comparison.squad_match = match(sameSet(plan.squad_ids, actual.squad_ids));
  comparison.chip_match = entry.context.no_active_chip_confirmed || entry.context.planned_chip !== null
    ? match(entry.context.planned_chip === actual.active_chip) : "unknown";
  // The confirmed starting state can include transfers made earlier this same GW.
  // Only transactions after the save can implement a newly recorded intention.
  if (actual.transfers.every(t => t.time !== null)) {
    const later = actual.transfers.filter(t => Date.parse(t.time!) > Date.parse(entry.saved_at));
    // Pairing simultaneous sales to purchases is arbitrary. Compare both
    // multisets independently, retaining repeated trades and round trips.
    const amounts = (transfers: { out_id: number; in_id: number }[], key: "out_id" | "in_id") =>
      transfers.map(t => t[key]).sort((a, b) => a - b);
    const samePackage = (transfers: typeof later) =>
      JSON.stringify(amounts(transfers, "out_id")) === JSON.stringify(amounts(plan.action.transfers, "out_id"))
      && JSON.stringify(amounts(transfers, "in_id")) === JSON.stringify(amounts(plan.action.transfers, "in_id"));
    const sameTimestamp = actual.transfers.some(t => Date.parse(t.time!) === Date.parse(entry.saved_at));
    if (!sameTimestamp) {
      comparison.transfer_match = samePackage(later) ? "followed"
        : samePackage(actual.transfers) ? "unknown" : "deviated";
    }
  }
  const lineup = plan.gameweeks.find(g => g.gameweek === entry.target_event);
  if (lineup) {
    const startersMatch = sameSet(lineup.starting_ids, actual.starting_ids) && lineup.captain_id === actual.captain_id;
    if (!startersMatch) comparison.lineup_match = "deviated";
    else if (lineup.bench_ids && lineup.vice_captain_id !== null) {
      const positions = new Map(plan.players.map(p => [p.id, p.position]));
      const normalizedBench = [...lineup.bench_ids.filter(id => positions.get(id) !== "GKP"), ...lineup.bench_ids.filter(id => positions.get(id) === "GKP")];
      comparison.lineup_match = match(lineup.vice_captain_id === actual.vice_captain_id
        && JSON.stringify(normalizedBench) === JSON.stringify(actual.bench_ids));
    }
  }
  const matches = [comparison.squad_match, comparison.transfer_match, comparison.lineup_match, comparison.chip_match];
  comparison.status = matches.includes("deviated") ? "deviated" : matches.every(s => s === "followed") ? "followed" : "unknown";
  if (comparison.squad_match === "deviated") comparison.reasons.push("Den indsendte 15-mandstrup afviger fra det gemte valg.");
  if (comparison.transfer_match === "deviated") comparison.reasons.push("Handlerne efter snapshot afviger fra den gemte transferrute.");
  if (comparison.transfer_match === "unknown") comparison.reasons.push("Handlerne kan ikke tidsmæssigt eftervises som gennemført efter gemningen.");
  if (comparison.lineup_match === "deviated") comparison.reasons.push("Startellever, anførervalg eller bænkrækkefølge blev ændret.");
  if (comparison.lineup_match === "unknown") comparison.reasons.push("Fuld opstilling, viceanfører eller bænkrækkefølge mangler i snapshot.");
  if (comparison.chip_match === "deviated") comparison.reasons.push("Det faktiske chipvalg afviger fra snapshot.");
  if (actual.active_chip === "freehit" || actual.active_chip === "wildcard") {
    comparison.reasons.push(actual.active_chip === "freehit" ? "Free Hit var aktivt. Denne rundes midlertidige trup sammenlignes; den kan ikke videreføres til næste runde." : "Wildcard var aktivt. Den faktiske sluttrup og alle handler vises særskilt.");
  }
  if (comparison.status === "followed") comparison.reasons.push("Trup, handler efter snapshot, opstilling og chipvalg matcher.");
  comparison.reasons.push("Pointene er holdets faktiske resultat. Sammenligningen dokumenterer ikke, at appen forårsagede gevinsten eller tabet.");
  return comparison;
}
