import assert from "node:assert/strict";
import test from "node:test";
import { buildLeagueDiagnosis } from "../lib/league-diagnosis.ts";

type Pick = { element: number; position: number; multiplier: number; is_captain: boolean; is_vice_captain: boolean };
function fixture() {
  const live = { elements: Array.from({ length: 30 }, (_, i) => ({ id: i + 1, stats: { total_points: (i % 5) + 1 } })) };
  const points = new Map(live.elements.map(p => [p.id, p.stats.total_points]));
  const team = (chip: string | null = null, offset = 0, hits = 0) => {
    const picks: Pick[] = Array.from({ length: 15 }, (_, i) => ({ element: i + 1 + offset, position: i + 1,
      multiplier: i === 0 ? chip === "3xc" ? 3 : 2 : i < 11 || chip === "bboost" ? 1 : 0,
      is_captain: i === 0, is_vice_captain: i === 1 }));
    const payload = { picks, active_chip: chip, entry_history: { event: 3, points: 0, event_transfers_cost: hits } };
    const history = { current: [{ event: 1, points: 40, event_transfers_cost: 0 }, { event: 2, points: 45, event_transfers_cost: 0 },
      { event: 3, points: 0, event_transfers_cost: hits }] };
    const sync = () => {
      payload.entry_history.points = picks.reduce((sum, p) => sum + p.multiplier * points.get(p.element)!, 0);
      history.current[2].points = payload.entry_history.points;
    };
    sync();
    return { payload, history, sync };
  };
  const own = team(), rival = team();
  const events = [1, 2, 3].map(id => ({ id, finished: true, data_checked: true }));
  const playerNames = new Map(live.elements.map(p => [p.id, `Player${p.id}`]));
  const run = (other = rival) => buildLeagueDiagnosis({ events, sourceEvent: 3, ownHistory: own.history, rivalHistory: other.history,
    ownPicks: own.payload, rivalPicks: other.payload, live, playerNames });
  const score = (id: number, value: number) => { points.set(id, value); live.elements.find(p => p.id === id)!.stats.total_points = value; };
  return { own, rival, team, run, events, live, score, playerNames };
}

test("identical squads reconcile exactly and shared core players cancel", () => {
  const f = fixture(), result = f.run();
  assert.equal(result.latest.status, "available");
  assert.equal(result.latest.gap_change, 0);
  assert.equal(result.latest.own!.gross, 32);
  assert.equal(result.latest.own!.core, 31);
  assert.equal(result.latest.own!.captain, 1);
  assert.deepEqual(result.latest.core_player_gains, []);
  assert.deepEqual(result.latest.core_player_losses, []);
  assert.equal(result.trend.status, "complete");
  assert.deepEqual(result.trend.requested_gameweeks, [1, 2, 3]);
});

test("effective vice-captain promotion and negative captain points are respected", () => {
  const f = fixture();
  f.score(1, 0); f.score(2, -2);
  f.own.payload.picks[0].multiplier = 0;
  f.own.payload.picks[1].multiplier = 2;
  f.own.payload.picks[11].multiplier = 1; // Actual automatic substitute, not unplayed bench points.
  f.own.sync(); f.rival.sync();
  const result = f.run();
  assert.equal(result.latest.status, "available");
  assert.equal(result.latest.own!.captain, -2);
  assert.equal(result.latest.contributions!.captain, 2);
  assert.ok(result.latest.core_player_losses.some(p => p.name === "Player12" && p.gap_change === -2));
});

test("triple captain separates the regular extra copy from the chip extra copy", () => {
  const f = fixture(), rival = f.team("3xc");
  const result = f.run(rival);
  assert.equal(result.latest.status, "available");
  assert.equal(result.latest.rival!.captain, 1);
  assert.equal(result.latest.rival!.triple_captain, 1);
  assert.equal(result.latest.contributions!.triple_captain, 1);
  assert.equal(result.latest.gap_change, 1);
});

test("bench boost and transfer hits are distinct, signed and sum exactly to the net gap", () => {
  const f = fixture(), rival = f.team("bboost", 0, 4);
  f.score(12, -1); f.own.sync(); rival.sync();
  const result = f.run(rival);
  assert.equal(result.latest.status, "available");
  assert.equal(result.latest.rival!.bench_boost, 11);
  assert.equal(result.latest.contributions!.hits, -4);
  assert.equal(result.latest.contributions!.core, 0);
  assert.equal(result.latest.gap_change, 7);
  assert.equal(Object.values(result.latest.contributions!).reduce((a, b) => a + b, 0), result.latest.gap_change);
  assert.equal(result.trend.rounds[2].gap_change, 7);
});

test("Wildcard and Free Hit are recorded, never assigned a counterfactual point gain", () => {
  for (const chip of ["wildcard", "freehit"]) {
    const f = fixture(), result = f.run(f.team(chip));
    assert.equal(result.latest.rival!.active_chip, chip);
    assert.equal(result.latest.gap_change, 0);
    assert.equal(result.latest.contributions!.bench_boost, 0);
    assert.equal(result.latest.contributions!.triple_captain, 0);
  }
});

test("nonzero core differences have sanitized names, signs and bounded lists, but no IDs", () => {
  const f = fixture();
  f.playerNames.set(1, "Player\u0000One");
  const result = f.run(f.team(null, 15));
  assert.equal(result.latest.status, "available");
  assert.equal(result.latest.core_player_gains.length, 3);
  assert.equal(result.latest.core_player_losses.length, 3);
  assert.ok(result.latest.core_player_gains.every(p => p.gap_change > 0));
  assert.ok(result.latest.core_player_losses.every(p => p.gap_change < 0));
  assert.ok(!JSON.stringify(result).includes('"element"') && !JSON.stringify(result).includes('"id"'));
  f.score(1, 100); f.own.sync();
  assert.equal(f.run(f.team(null, 15)).latest.core_player_losses[0].name, "PlayerOne");
});

test("malformed picks, missing scores and contradictory official totals fail closed", () => {
  const mutations: ((f: ReturnType<typeof fixture>) => void)[] = [
    f => { f.own.payload.picks[0].multiplier = 3; f.own.sync(); },
    f => { f.own.payload.picks[1].multiplier = 2; f.own.sync(); },
    f => { f.own.payload.picks[1].position = 1; },
    f => { f.own.payload.picks[1].element = 1; f.own.sync(); },
    f => { f.own.payload.picks[0].is_vice_captain = true; },
    f => { f.own.payload.active_chip = "unknown"; },
    f => { f.own.payload.entry_history.event = 2; },
    f => { f.own.payload.entry_history.points += 1; },
    f => { f.own.history.current[2].points += 1; },
    f => { f.own.history.current[2].event_transfers_cost = -4; },
    f => { f.own.history.current.push({ ...f.own.history.current[2] }); },
    f => { f.live.elements.shift(); },
    f => { f.live.elements.push({ ...f.live.elements[0] }); },
    f => { f.live.elements[0].stats.total_points = Number.NaN; },
  ];
  for (const mutate of mutations) {
    const f = fixture(); mutate(f);
    const result = f.run();
    assert.equal(result.latest.status, "unavailable", mutate.toString());
    assert.equal(result.latest.own, null);
    assert.equal(result.latest.gap_change, null);
    assert.equal(result.latest.contributions, null);
  }
});

test("unfinished source is never attributed, while prior finalized net trends remain usable", () => {
  const f = fixture(); f.events[2].data_checked = false;
  const result = f.run();
  assert.equal(result.latest.status, "unavailable");
  assert.match(result.latest.reason!, /not finalized/);
  assert.deepEqual(result.trend.requested_gameweeks, [1, 2]);
  assert.equal(result.trend.status, "complete");
});

test("partial history does not zero-fill or present an incomplete window total", () => {
  const f = fixture(); f.rival.history.current.splice(1, 1);
  const result = f.run();
  assert.equal(result.trend.status, "partial");
  assert.deepEqual(result.trend.missing_gameweeks, [2]);
  assert.equal(result.trend.gap_change, null);
  assert.equal(result.trend.rounds.length, 2);
  assert.match(result.scope, /late-start league/);
  assert.match(result.scope, /not whether a past decision was good/);
  assert.equal(result.latest.status, "available");
});
