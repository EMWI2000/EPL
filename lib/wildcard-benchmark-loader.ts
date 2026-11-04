import { createHash } from "node:crypto";
import { createLeagueFetcher } from "./league-fetch.ts";
import { loadLeagueStandings } from "./league-overview.ts";
import { analyzeWildcardBenchmarks, type WildcardBenchmarkAnalysis } from "./wildcard-benchmark.ts";

const MAX_PEERS = 200;

export interface WildcardBenchmarkResponse extends WildcardBenchmarkAnalysis {
  version: "fpl-wildcard-benchmark-v1";
  generated_at: string;
  season: string;
  target_event: number;
  coverage: {
    leagues_selected: number;
    leagues_complete: number;
    discovered_peers: number;
    sampled_peers: number;
    histories_loaded: number;
    missing_histories: number;
    selection: "all" | "stable_sample";
    complete: boolean;
  };
}

type Row = Record<string, unknown>;
const object = (value: unknown): Row => value !== null && typeof value === "object" && !Array.isArray(value)
  ? value as Row : {};
const rows = (value: unknown): Row[] => Array.isArray(value) ? value.map(object) : [];
const positiveInteger = (value: unknown): value is number => typeof value === "number"
  && Number.isSafeInteger(value) && value > 0;

/** Stable sampling is independent of score, rank and observed chip success. */
function selectPeers(ids: number[], season: string): number[] {
  if (ids.length <= MAX_PEERS) return ids.sort((a, b) => a - b);
  return ids.map(id => ({ id, hash: createHash("sha256").update(`${season}:${id}`).digest("hex") }))
    .sort((a, b) => a.hash.localeCompare(b.hash) || a.id - b.id)
    .slice(0, MAX_PEERS).map(row => row.id);
}

export async function loadWildcardBenchmark(
  managerId: number,
  fetcher: typeof fetch = fetch,
  now = Date.now(),
): Promise<WildcardBenchmarkResponse> {
  if (!positiveInteger(managerId)) throw new Error("Invalid manager");
  const get = createLeagueFetcher(fetcher);
  const [managerRaw, bootstrap] = await Promise.all([
    get(`entry/${managerId}/`), get("bootstrap-static/"),
  ]);
  const events = rows(object(bootstrap).events).filter(event => positiveInteger(event.id)
    && event.id <= 38 && typeof event.deadline_time === "string" && Number.isFinite(Date.parse(event.deadline_time)));
  const first = events.find(event => event.id === 1);
  const next = events.filter(event => Date.parse(String(event.deadline_time)) > now)
    .sort((a, b) => Number(a.id) - Number(b.id))[0];
  const seasonFinished = events.some(event => event.id === 38 && event.finished === true && event.data_checked === true);
  if (!first || (!next && !seasonFinished)) throw new Error("Current FPL season unavailable");
  const startYear = new Date(String(first.deadline_time)).getUTCFullYear();
  const season = `${startYear}/${String(startYear + 1).slice(-2)}`;
  const targetEvent = next ? Number(next.id) : 38;
  const half = targetEvent <= 19 ? 1 : 2;

  const memberships = rows(object(object(managerRaw).leagues).classic)
    .filter(league => league.league_type === "x" && positiveInteger(league.id) && positiveInteger(league.entry_rank));
  const selectedLeagues = [...new Map(memberships.map(league => [league.id, league])).values()].slice(0, 3);
  const standings = await Promise.all(selectedLeagues.map(league =>
    loadLeagueStandings(Number(league.id), Number(league.entry_rank), get)));
  const peerIds = [...new Set(standings.flatMap(table => table.rows.map(row => Number(row.entry))))]
    .filter(id => id !== managerId);
  const selectedPeers = selectPeers(peerIds, season);
  const historyResults = await Promise.allSettled([managerId, ...selectedPeers].map(async id =>
    [id, await get(`entry/${id}/history/`)] as const));
  const histories = new Map<number, unknown>();
  for (const result of historyResults) {
    if (result.status === "fulfilled") histories.set(...result.value);
  }
  const analysis = analyzeWildcardBenchmarks({ bootstrap, histories, ownManagerId: managerId, half });
  const completeLeagues = standings.filter(table => table.complete).length;
  const missingHistories = selectedPeers.length - analysis.valid_peer_histories;
  return {
    ...analysis,
    version: "fpl-wildcard-benchmark-v1",
    generated_at: new Date(now).toISOString(),
    season,
    target_event: targetEvent,
    coverage: {
      leagues_selected: selectedLeagues.length,
      leagues_complete: completeLeagues,
      discovered_peers: peerIds.length,
      sampled_peers: selectedPeers.length,
      histories_loaded: analysis.valid_peer_histories,
      missing_histories: missingHistories,
      selection: selectedPeers.length === peerIds.length ? "all" : "stable_sample",
      complete: selectedLeagues.length > 0 && completeLeagues === selectedLeagues.length
        && selectedPeers.length === peerIds.length && missingHistories === 0,
    },
  };
}
