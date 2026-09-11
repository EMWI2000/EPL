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
      { id: 3, deadline_time: "2026-09-01T12:00:00Z" }, { id: 4, deadline_time: "2026-09-12T12:30:00Z" },
    ], elements: Array.from({ length: 30 }, (_, i) => ({ id: i + 1, web_name: `Player${i + 1}` })) };
    else if (path.endsWith("entry/100/")) body = { summary_overall_points: 192,
      leagues: { classic: [{ id: 530, name: "Private league", league_type: "x", entry_rank: 51, entry_last_rank: 47 },
        { id: 314, league_type: "s", entry_rank: 300 }] } };
    else if (path.includes("standings/")) body = { standings: { results: [
      { entry: 200, rank: 1, total: 250, entry_name: "Rival team", player_name: "Private person" },
      { entry: 201, rank: 50, total: 195, entry_name: "Nearest team" },
      { entry: 100, rank: 51, total: 180, entry_name: "Own team" },
    ] } };
    else if (path.endsWith("history/")) body = { chips: [] };
    else if (path.includes("picks/")) body = { picks: Array.from({ length: 15 }, (_, i) => ({
      element: i + (path.includes("entry/100/") ? 1 : 2), is_captain: i === 0 })) };
    else throw new Error("Unexpected path");
    return Response.json(body);
  }) as typeof fetch;
  const value = await loadLeagueOverview(100, fetcher, Date.parse("2026-09-11T12:00:00Z"));
  assert.equal(value.leagues.length, 1);
  assert.equal(value.leagues[0].leader_gap, 70);
  assert.equal(value.leagues[0].rivals[1].gap, 15);
  assert.deepEqual(value.leagues[0].rivals[0].different_players, ["Player16"]);
  assert.ok(calls.length <= 10);
  assert.equal(calls.filter(v => v.includes("standings")).length, 2);
  const ai = JSON.stringify(leagueAiContext(value));
  assert.ok(!ai.includes("Private") && !ai.includes("Rival team") && !ai.includes('"entry"'));
  assert.ok(ai.includes("Player16"));
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
