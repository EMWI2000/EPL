import assert from "node:assert/strict";
import test from "node:test";
import { buildLeagueDiagnosis, buildLeagueSeasonSummary, summarizeLeagueSeasonDetails } from "../lib/league-diagnosis.ts";
import { loadLeagueOverview, loadLeagueSeasonDetails, LeagueScopeError } from "../lib/league-overview.ts";

const now = Date.parse("2026-09-20T12:00:00Z");
const events = Array.from({ length: 7 }, (_, i) => ({ id: i + 1,
  deadline_time: i < 6 ? `2026-09-0${i + 1}T12:00:00Z` : "2026-10-01T12:00:00Z",
  finished: i < 6, data_checked: i < 6 }));

function fixture(options: { start?: number; missing?: string; pages?: boolean; delay?: boolean; tiedRanks?: boolean } = {}) {
  const calls: string[] = [];
  let active = 0, maximum = 0;
  const starts = [100, 202, 201, 200];
  const points = (id: number) => id > 45 ? 3 : id > 30 ? id < 33 ? 4 : 2 : id > 15 ? 2 : 1;
  const players = Array.from({ length: 60 }, (_, i) => ({ id: i + 1, web_name: `Player ${i + 1}` }));
  const squad = (entry: number, event: number) => {
    const offset = Math.max(0, starts.indexOf(entry)) * 15;
    const picks = Array.from({ length: 15 }, (_, i) => ({ element: offset + i + 1, position: i + 1,
      multiplier: i === 0 ? 2 : i < 11 ? 1 : 0, is_captain: i === 0, is_vice_captain: i === 1 }));
    return { active_chip: null, picks, entry_history: { event,
      points: picks.reduce((sum, p) => sum + p.multiplier * points(p.element), 0), event_transfers_cost: 0 } };
  };
  const table = [200, 201, 202, 100].map((entry, i) => ({ entry, rank: i + 1, rank_sort: i + 1,
    total: squad(entry, 1).entry_history.points * (7 - (options.start ?? 1)), entry_name: `Team ${entry}` }));
  if (options.tiedRanks) for (const row of table.slice(0, 3)) { row.rank = 1; row.total = 216; }
  const full = options.pages ? Array.from({ length: 52 }, (_, i) => ({ entry: i === 50 ? 100 : 200 + i, rank: i + 1, rank_sort: i + 1,
    total: 200 - i, entry_name: `Team ${i}` })) : table;
  const fetcher = (async (url: string | URL | Request) => {
    const path = String(url).split("/api/")[1]; calls.push(path);
    active += 1; maximum = Math.max(maximum, active);
    try {
      if (options.delay) await new Promise(resolve => setTimeout(resolve, 4));
      if (path === options.missing) return new Response("missing", { status: 503 });
      if (path === "bootstrap-static/") return Response.json({ events, elements: players });
      if (path === "entry/100/") return Response.json({ summary_overall_points: 72,
        leagues: { classic: [{ id: 530, name: "Example", league_type: "x", entry_rank: options.pages ? 51 : 4 }] } });
      if (path.startsWith("leagues-classic/530/standings/")) {
        const page = Number(new URL(String(url)).searchParams.get("page_standings"));
        return Response.json({ league: { start_event: options.start ?? 1 },
          standings: { page, has_next: options.pages && page === 1, results: full.slice((page - 1) * 50, page * 50) } });
      }
      const entry = Number(path.split("/")[1]);
      if (path.endsWith("history/")) return Response.json({ chips: [], current: events.slice(0, 6).map(e => squad(entry, e.id).entry_history) });
      if (path.endsWith("picks/")) return Response.json(squad(entry, Number(path.split("/")[3])));
      if (path.endsWith("live/")) return Response.json({ elements: players.map(p => ({ id: p.id, stats: { total_points: points(p.id) } })) });
      throw new Error(`Unexpected ${path}`);
    } finally { active -= 1; }
  }) as typeof fetch;
  return { calls, fetcher, maximum: () => maximum };
}

test("all pages define the field median and select leader, nearest and podium once", async () => {
  const f = fixture({ pages: true });
  const result = await loadLeagueOverview(100, f.fetcher, now);
  const league = result.leagues[0];
  assert.deepEqual(league.field, { complete: true, observed_count: 52, median_points: 174.5, median_gap: 24.5, podium_points: 198 });
  assert.deepEqual(league.rivals.map(r => [r.rank, r.roles]), [[1, ["leader"]], [50, ["nearest"]], [3, ["podium"]]]);
  assert.equal(f.calls.filter(p => p.includes("standings/")).length, 2);
});

test("an interrupted page has no full-field median", async () => {
  const f = fixture({ pages: true, missing: "leagues-classic/530/standings/?page_standings=2" });
  const league = (await loadLeagueOverview(100, f.fetcher, now)).leagues[0];
  assert.equal(league.field?.complete, false);
  assert.equal(league.field?.median_points, null);
  assert.equal(league.rank, null);
});

test("podium and next-place selection respect the official ordering when point ranks tie", async () => {
  const f = fixture({ tiedRanks: true });
  const league = (await loadLeagueOverview(100, f.fetcher, now)).leagues[0];
  assert.equal(league.field?.podium_points, 216);
  assert.deepEqual(league.rivals.find(r => r.entry === 202)?.roles, ["nearest", "podium"]);
});

test("the completed season stays accessible after the final deadline", async () => {
  const f = fixture();
  const fetcher = (async (url: string | URL | Request, init?: RequestInit) => {
    const response = await f.fetcher(url, init);
    if (!String(url).endsWith("bootstrap-static/")) return response;
    const payload = await response.json();
    payload.events = [{ id: 38, finished: true, data_checked: true, deadline_time: "2026-09-01T12:00:00Z" }];
    return Response.json(payload);
  }) as typeof fetch;
  const overview = await loadLeagueOverview(100, fetcher, now);
  assert.equal(overview.source_event, 38);
  assert.equal(overview.leagues[0].rivals[0].season?.end_event, 38);
  assert.deepEqual(overview.own_chips_remaining, []);
});

test("season details reconcile each round and aggregate once, including late-start periods", async () => {
  const f = fixture({ start: 3, delay: true });
  const overview = await loadLeagueOverview(100, f.fetcher, now);
  const leader = overview.leagues[0].rivals.find(r => r.roles?.includes("leader"))!;
  assert.equal(leader.season?.gap_change, 96);
  assert.equal(leader.season?.matches_standings, true);
  const batch = await loadLeagueSeasonDetails(100, { leagueId: 530, rivalId: 200, fromEvent: 3, toEvent: 6 }, f.fetcher, now);
  assert.ok(batch.rounds.every(r => r.status === "available"));
  const total = summarizeLeagueSeasonDetails([3, 4, 5, 6], batch.rounds);
  assert.equal(total.gap_change, 96);
  assert.equal(total.own?.net, 48);
  assert.equal(total.rival?.net, 144);
  assert.equal(Object.values(total.contributions!).reduce((a, b) => a + b, 0), 96);
  assert.ok(f.maximum() <= 4);
  assert.equal(f.calls.filter(p => p === "event/6/live/").length, 1);
  const count = f.calls.length;
  await loadLeagueSeasonDetails(100, { leagueId: 530, rivalId: 200, fromEvent: 3, toEvent: 6 }, f.fetcher, now);
  assert.equal(f.calls.length, count, "cached reads should serve repeated detail views");
});

test("missing picks preserve complete history trend but withhold a misleading component total", async () => {
  const f = fixture({ missing: "entry/200/event/2/picks/" });
  const batch = await loadLeagueSeasonDetails(100, { leagueId: 530, rivalId: 200, fromEvent: 1, toEvent: 6 }, f.fetcher, now);
  const sum = summarizeLeagueSeasonDetails([1, 2, 3, 4, 5, 6], batch.rounds);
  assert.deepEqual(sum.missing_gameweeks, [2]);
  assert.equal(sum.gap_change, null);
  assert.equal(sum.own, null);
  assert.equal(batch.rounds[0].status, "available");
});

test("detail authorization rejects other managers, leagues and out-of-period ranges before their data is requested", async () => {
  const f = fixture({ start: 3 });
  for (const args of [
    { leagueId: 999, rivalId: 200, fromEvent: 3, toEvent: 6 },
    { leagueId: 530, rivalId: 999, fromEvent: 3, toEvent: 6 },
    { leagueId: 530, rivalId: 200, fromEvent: 1, toEvent: 2 },
    { leagueId: 530, rivalId: 200, fromEvent: 3, toEvent: 9 },
  ]) await assert.rejects(loadLeagueSeasonDetails(100, args, f.fetcher, now), LeagueScopeError);
  assert.ok(!f.calls.some(p => p.includes("entry/999") || p.includes("leagues-classic/999")));
  assert.ok(!f.calls.some(p => /event\/[1245789]\/picks/.test(p)));
});

test("missing and unfinalized season weeks never become zero, and current table mismatches are explicit", () => {
  const current = events.slice(0, 6).map(e => ({ event: e.id, points: 20, event_transfers_cost: 4 }));
  const base = { events, startEvent: 3, sourceEvent: 6, ownHistory: { current }, rivalHistory: { current }, ownLeaguePoints: 60, rivalLeaguePoints: 64 };
  const result = buildLeagueSeasonSummary(base);
  assert.equal(result.own_net, 64);
  assert.equal(result.matches_standings, false);
  const missing = buildLeagueSeasonSummary({ ...base, ownHistory: { current: current.filter(r => r.event !== 4) } });
  assert.deepEqual(missing.missing_gameweeks, [4]);
  assert.equal(missing.gap_change, null);
  assert.equal(missing.rounds[1].own_net, null);
  const unfinished = buildLeagueSeasonSummary({ ...base, events: events.map(e => e.id === 4 ? { ...e, data_checked: false } : e) });
  assert.deepEqual(unfinished.missing_gameweeks, [4]);
  assert.equal(unfinished.rounds[1].rival_net, null);
});

test("season totals separate Bench Boost, Triple Captain and signed hits without duplicate captain credit", () => {
  const live = { elements: Array.from({ length: 15 }, (_, i) => ({ id: i + 1, stats: { total_points: 2 } })) };
  const team = (event: number, chip: string | null, hits = 0) => {
    const picks = live.elements.map((p, i) => ({ element: p.id, position: i + 1, is_captain: i === 0, is_vice_captain: i === 1,
      multiplier: i === 0 ? chip === "3xc" ? 3 : 2 : i < 11 || chip === "bboost" ? 1 : 0 }));
    const entry_history = { event, points: picks.reduce((sum, p) => sum + p.multiplier * 2, 0), event_transfers_cost: hits };
    return { picks, active_chip: chip, entry_history };
  };
  const rounds = ["bboost", "3xc"].map((chip, i) => {
    const own = team(i + 1, null), rival = team(i + 1, chip, i === 0 ? 4 : 0);
    return buildLeagueDiagnosis({ events, sourceEvent: i + 1, ownHistory: { current: [own.entry_history] },
      rivalHistory: { current: [rival.entry_history] }, ownPicks: own, rivalPicks: rival, live, playerNames: new Map() }).latest;
  });
  const total = summarizeLeagueSeasonDetails([1, 2], rounds);
  assert.equal(total.gap_change, 6);
  assert.deepEqual(total.contributions, { core: 0, captain: 0, bench_boost: 8, triple_captain: 2, hits: -4 });
  assert.equal(summarizeLeagueSeasonDetails([1, 2], [...rounds, rounds[0]]).gap_change, null);
});
