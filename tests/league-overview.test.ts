import assert from "node:assert/strict";
import test from "node:test";
import { leagueAiContext, loadLeagueOverview, remainingLeagueChips } from "../lib/league-overview.ts";

test("chip inventory respects halves, malformed data is unknown rather than empty", () => {
  const history = { chips: [{ name: "bboost", event: 1 }, { name: "3xc", event: 3 }] };
  assert.deepEqual(remainingLeagueChips(history, 4), ["wildcard", "freehit"]);
  assert.equal(remainingLeagueChips({}, 4), null);
  assert.equal(remainingLeagueChips({ chips: [{ name: "bboost", event: 5 }] }, 4), null);
  assert.equal(remainingLeagueChips(history, 20)?.length, 4);
});

test("bounded league fetch compares public squads, pages near rank51, strips personal names from AI", async () => {
  const calls: string[] = [];
  const fetcher = (async (url: string | URL | Request) => {
    const path = String(url); calls.push(path);
    let body: unknown;
    if (path.endsWith("bootstrap-static/")) body = { events: [
      { id: 3, deadline_time: "2026-09-01T12:00:00Z", finished: true, data_checked: true }, { id: 4, deadline_time: "2026-09-12T12:30:00Z" },
    ], elements: Array.from({ length: 30 }, (_, i) => ({ id: i + 1, web_name: `Player${i + 1}` })) };
    else if (path.endsWith("entry/100/")) body = { summary_overall_points: 192,
      leagues: { classic: [{ id: 530, name: "Private league", league_type: "x", entry_rank: 51, entry_last_rank: 47 },
        { id: 314, league_type: "s", entry_rank: 300 }] } };
    else if (path.includes("standings/")) body = { standings: { results: [
      { entry: 200, rank: 1, total: 250, entry_name: "Rival team", player_name: "Private person" },
      { entry: 201, rank: 50, total: 195, entry_name: "Nearest team" },
      { entry: 100, rank: 51, total: 180, entry_name: "Own team" },
    ] } };
    else if (path.endsWith("history/")) body = { chips: [], current: [{ event: 3, points: 12, event_transfers_cost: 0 }] };
    else if (path.endsWith("event/3/live/")) body = { elements: Array.from({ length: 30 }, (_, i) => ({ id: i + 1, stats: { total_points: 1 } })) };
    else if (path.includes("picks/")) body = { active_chip: null, entry_history: { event: 3, points: 12, event_transfers_cost: 0 },
      picks: Array.from({ length: 15 }, (_, i) => ({
      element: i + (path.includes("entry/100/") ? 1 : 2), is_captain: i === 0, is_vice_captain: i === 1,
      position: i + 1, multiplier: i === 0 ? 2 : i < 11 ? 1 : 0 })) };
    else throw new Error("Unexpected path");
    return Response.json(body);
  }) as typeof fetch;
  const value = await loadLeagueOverview(100, fetcher, Date.parse("2026-09-11T12:00:00Z"));
  assert.equal(value.leagues.length, 1);
  assert.equal(value.leagues[0].leader_gap, 70);
  assert.equal(value.leagues[0].rivals[1].gap, 15);
  assert.deepEqual(value.leagues[0].rivals[0].different_players, ["Player16"]);
  assert.ok(calls.length <= 11);
  assert.equal(calls.filter(v => v.includes("event/3/live/")).length, 1);
  assert.ok(value.leagues[0].rivals.every(r => r.diagnosis.latest.status === "available"));
  assert.equal(calls.filter(v => v.includes("standings")).length, 2);
  const ai = JSON.stringify(leagueAiContext(value));
  assert.ok(!ai.includes("Private") && !ai.includes("Rival team") && !ai.includes('"entry"'));
  assert.ok(ai.includes("Player16"));
  assert.ok(ai.includes('"contributions"') && ai.includes("late-start league"));
});

test("standings outages preserve explicit unknown gaps", async () => {
  const fetcher = (async (url: string | URL | Request) => {
    if (String(url).endsWith("bootstrap-static/")) return Response.json({ events: [
      { id: 3, deadline_time: "2026-09-01T12:00:00Z" }, { id: 4, deadline_time: "2026-09-12T12:30:00Z" }] });
    if (String(url).endsWith("entry/100/")) return Response.json({ summary_overall_points: 192,
      leagues: { classic: [{ id: 530, name: "League", league_type: "x", entry_rank: 8 }] } });
    return new Response("unavailable", { status: 503 });
  }) as typeof fetch;
  const result = await loadLeagueOverview(100, fetcher, Date.parse("2026-09-11T12:00:00Z"));
  assert.equal(result.leagues[0].partial, true);
  assert.equal(result.leagues[0].leader_gap, null);
  assert.equal(result.own_chips_remaining, null);
});

function regressionFixture(options: {
  summaryRank?: number; ownRank?: number | null; finalized?: boolean; priorFinalized?: boolean;
  liveUnavailable?: boolean; captain?: "vice" | "none";
  standings?: { entry: number; rank: number; total: number }[];
} = {}) {
  const calls: string[] = [];
  const score = (id: number) => options.captain && id === 16 ? 0 : 1;
  const squad = (own: boolean) => {
    const picks = Array.from({ length: 15 }, (_, i) => ({ element: i + (own ? 1 : 16), position: i + 1,
      multiplier: i === 0 ? 2 : i < 11 ? 1 : 0, is_captain: i === 0, is_vice_captain: i === 1 }));
    if (!own && options.captain) {
      picks[0].multiplier = 0;
      picks[1].multiplier = options.captain === "vice" ? 2 : 1;
      picks[11].multiplier = 1;
    }
    return { active_chip: null, picks, entry_history: { event: 3,
      points: picks.reduce((sum, p) => sum + score(p.element) * p.multiplier, 0), event_transfers_cost: 0 } };
  };
  const fetcher = (async (url: string | URL | Request) => {
    const path = String(url); calls.push(path);
    if (path.endsWith("bootstrap-static/")) return Response.json({ events: [
      ...(options.priorFinalized ? [{ id: 2, deadline_time: "2026-08-28T12:00:00Z", finished: true, data_checked: true }] : []),
      { id: 3, deadline_time: "2026-09-01T12:00:00Z", finished: options.finalized !== false, data_checked: options.finalized !== false },
      { id: 4, deadline_time: "2026-09-12T12:30:00Z" },
    ], elements: Array.from({ length: 30 }, (_, i) => ({ id: i + 1, web_name: `Player${i + 1}` })) });
    if (path.endsWith("entry/100/")) return Response.json({ summary_overall_points: 100,
      leagues: { classic: [{ id: 530, name: "Private league", league_type: "x", entry_rank: options.summaryRank ?? 3, entry_last_rank: 8 }] } });
    if (path.includes("standings/")) return Response.json({ standings: { results: options.standings ?? [
      { entry: 200, rank: 1, total: 150 }, { entry: 201, rank: 2, total: 110 },
      ...(options.ownRank === null ? [] : [{ entry: 100, rank: options.ownRank ?? 3, total: 100 }]),
    ] } });
    if (path.endsWith("event/3/live/")) return options.liveUnavailable ? new Response("Unavailable", { status: 503 })
      : Response.json({ elements: Array.from({ length: 30 }, (_, i) => ({ id: i + 1, stats: { total_points: score(i + 1) } })) });
    if (path.includes("picks/")) return Response.json(squad(path.includes("entry/100/")));
    if (path.endsWith("history/")) return Response.json({ chips: [], current: [squad(path.includes("entry/100/")).entry_history] });
    throw new Error("Unexpected regression path");
  }) as typeof fetch;
  return { calls, run: () => loadLeagueOverview(100, fetcher, Date.parse("2026-09-11T12:00:00Z")) };
}

test("fresh own standings rank determines displayed rank and nearest rival, not stale membership rank", async () => {
  const f = regressionFixture({ summaryRank: 8, standings: [
    { entry: 200, rank: 1, total: 150 }, { entry: 201, rank: 4, total: 110 },
    { entry: 100, rank: 5, total: 100 }, { entry: 202, rank: 7, total: 90 },
  ] });
  const league = (await f.run()).leagues[0];
  assert.equal(league.rank, 5);
  assert.deepEqual(league.rivals.map(r => r.rank), [1, 4]);
  assert.equal(league.partial, false);
});

test("missing own standings row is unknown and never selects a false nearest rival", async () => {
  const f = regressionFixture({ summaryRank: 8, ownRank: null });
  const result = await f.run(), league = result.leagues[0];
  assert.equal(league.rank, null);
  assert.equal(league.leader_gap, null);
  assert.equal(league.partial, true);
  assert.deepEqual(league.rivals.map(r => r.rank), [1]);
  const ai = leagueAiContext(result);
  assert.ok(ai.available);
  assert.equal(ai.leagues?.[0].rank, null);
});

test("stale side-boundary rank cannot label a remote fetched page as the nearest rival", async () => {
  const f = regressionFixture({ summaryRank: 102, standings: [
    { entry: 200, rank: 1, total: 150 }, { entry: 201, rank: 50, total: 110 },
    { entry: 100, rank: 101, total: 100 },
  ] });
  const league = (await f.run()).leagues[0];
  assert.equal(league.rank, 101);
  assert.equal(league.partial, true);
  assert.deepEqual(league.rivals.map(r => r.rank), [1]);
  assert.equal(f.calls.filter(path => path.includes("standings/")).length, 2);
});

test("missing finalized live scores or history windows propagate the partial league flag", async () => {
  const noLive = (await regressionFixture({ liveUnavailable: true }).run()).leagues[0];
  assert.equal(noLive.partial, true);
  assert.ok(noLive.rivals.every(r => r.diagnosis.latest.status === "unavailable"));
  const noHistory = (await regressionFixture({ priorFinalized: true }).run()).leagues[0];
  assert.equal(noHistory.partial, true);
  assert.ok(noHistory.rivals.every(r => r.diagnosis.latest.status === "available" && r.diagnosis.trend.status === "partial"));
});

test("an unfinished source alone is expected, skips live fetching and retains the nominated captain", async () => {
  const f = regressionFixture({ finalized: false, captain: "vice" });
  const league = (await f.run()).leagues[0];
  assert.equal(league.partial, false);
  assert.ok(league.rivals.every(r => r.captain === "Player16"));
  assert.ok(!f.calls.some(path => path.includes("event/3/live/")));
});

test("finalized captain display and AI context use the effective vice-captain", async () => {
  const result = await regressionFixture({ captain: "vice" }).run();
  assert.equal(result.leagues[0].partial, false);
  assert.ok(result.leagues[0].rivals.every(r => r.captain === "Player17"));
  const ai = leagueAiContext(result);
  assert.ok(ai.leagues?.[0].rivals.every(r => r.last_captain !== null && ai.player_labels?.[r.last_captain] === "Player17"));
});

test("finalized captain is unknown when no player received an effective captain multiplier", async () => {
  const result = await regressionFixture({ captain: "none" }).run();
  assert.equal(result.leagues[0].partial, false);
  assert.ok(result.leagues[0].rivals.every(r => r.captain === null));
  const ai = leagueAiContext(result);
  assert.ok(ai.leagues?.[0].rivals.every(r => r.last_captain === null));
});
