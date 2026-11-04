import assert from "node:assert/strict";
import test from "node:test";
import { buildDecisionHistoryEntry, parseDecisionHistoryEntry, type DecisionHistoryEntry } from "../lib/decision-history.ts";
import { compareDecisionResult, loadDecisionResult, parseActualPicks, type ActualDecisionResult } from "../lib/decision-results.ts";
import { decisionResultsResponse } from "../lib/decision-results-response.ts";
import { replayFrozenDecisionPlan } from "../lib/frozen-plan-score.ts";
import type { PlannerAction, PlannerPayload, PlayerPosition } from "../lib/planner-contract.ts";

const starters = [1, 2, 3, 4, 7, 8, 9, 10, 12, 13, 14];
const bench = [11, 5, 6, 15];
const squad = Array.from({ length: 15 }, (_, i) => i + 1);
function position(id: number): PlayerPosition {
  return [1, 15].includes(id) ? "GKP" : id <= 6 ? "DEF" : id <= 11 ? "MID" : "FWD";
}
function entry(): DecisionHistoryEntry {
  const players = squad.map(id => ({ id, name: `Player ${id}`, position: position(id), team: "TST",
    price_tenths: 50, purchase_price_tenths: 50, selling_price_tenths: 50, status: "a" as const, price_signal: null }));
  const action: PlannerAction = { kind: "roll", transfer_count: 0, transfers: [], squad_ids: squad,
    gameweeks: [{ gameweek: 2, starting_ids: starters, captain_id: 14, vice_captain_id: 12, bench_ids: bench, formation: "3-4-3", projected_points: 50 }],
    bank_before_tenths: 10, bank_after_tenths: 10, free_transfers_before: 1, free_transfers_next_gameweek: 2,
    hit_points: 0, projected_points: 50, banked_ft_value_points: 1, decision_value_points: 51,
    net_points_vs_roll: 0, decision_value_vs_roll: 0, explanation: "Roll" };
  const planner = { manager_id: 123, state_fingerprint: "a".repeat(64), source_event: 1, target_event: 2,
    confirmed_state: { bank_tenths: 10, free_transfers: 1, no_active_chip_confirmed: true, squad: players },
    best_action: action, alternatives: [], sequential: null, method: { candidate_count: 45, plans_per_transfer_count: 5,
      higher_transfer_count_plans: 1, maximum_immediate_transfers: 2, roll_ft_value_points: 1, chips_modelled: false,
      next_deadline_transfer_modelled: false, future_transfers_modelled: false } } satisfies PlannerPayload;
  return buildDecisionHistoryEntry({ saved_at: "2026-08-23T12:00:00Z", target_deadline: "2026-08-28T18:30:00Z",
    state_observed_at: "2026-08-23T11:00:00Z", recommendation_generated_at: "2026-08-23T11:30:00Z",
    forecast: { version: "v2", horizon: 1, validation_status: "unvalidated" }, planner, ai_review: null });
}
function bootstrap(finalized = true) {
  return { events: [{ id: 2, deadline_time: "2026-08-28T18:30:00Z", finished: finalized, data_checked: finalized }],
    elements: squad.map(id => ({ id, element_type: { GKP: 1, DEF: 2, MID: 3, FWD: 4 }[position(id)] })) };
}
function picks() {
  return { entry_history: { event: 2, points: 45, event_transfers_cost: 8, bank: 10 }, active_chip: null,
    automatic_subs: [] as { event: number; element_in: number; element_out: number }[],
    picks: [...starters, 15, 11, 5, 6].map((id, i) => ({ element: id, position: i + 1,
      is_captain: id === 14, is_vice_captain: id === 12 })) };
}
function result(): ActualDecisionResult {
  return { schema_version: "fpl-decision-result-v1", manager_id: 123, target_event: 2, target_deadline: "2026-08-28T18:30:00Z",
    season: "2026/27", fetched_at: "2026-08-30T20:00:00Z", finalized: true,
    player_scores: squad.map(id => ({ id, position: position(id), minutes: 90, points: 2 })),
    actual: parseActualPicks(picks(), bootstrap(), 2) };
}

test("compares the originally submitted XI and bench after automatic substitutions, with correct net points", () => {
  const payload = picks();
  const incoming = payload.picks.find(p => p.element === 11)!;
  const outgoing = payload.picks.find(p => p.element === 14)!;
  [incoming.position, outgoing.position] = [outgoing.position, incoming.position];
  payload.automatic_subs = [{ event: 2, element_in: 11, element_out: 14 }];
  const r = result();
  r.actual = parseActualPicks(payload, bootstrap(), 2);
  assert.deepEqual(r.actual.starting_ids, starters);
  assert.deepEqual(r.actual.bench_ids, bench);
  assert.equal(r.actual.captain_id, 14, "Keeps the chosen captain even when the vice actually scores");
  const comparison = compareDecisionResult(entry(), r);
  assert.equal(comparison.status, "followed");
  assert.equal(comparison.gross_points, 45);
  assert.equal(comparison.transfer_cost, 8);
  assert.equal(comparison.net_points, 37);
});

test("ignores transactions already present in the confirmed state, but catches later extra trades", () => {
  const r = result();
  r.actual!.transfers = [{ out_id: 99, in_id: 8, time: "2026-08-22T12:00:00Z" }];
  assert.equal(compareDecisionResult(entry(), r).status, "followed");
  r.actual!.transfers.push({ out_id: 8, in_id: 99, time: "2026-08-24T12:00:00Z" });
  assert.equal(compareDecisionResult(entry(), r).status, "deviated");
  r.actual!.transfers = [{ out_id: 99, in_id: 8, time: null }];
  assert.equal(compareDecisionResult(entry(), r).status, "unknown");
});

test("equivalent transfer packages match despite arbitrary pairing while round trips stay visible", () => {
  const e = entry();
  const plan = e.plans!.best!;
  const replace = (id: number) => id === 8 ? 16 : id === 9 ? 17 : id;
  plan.action.kind = "transfer";
  plan.action.transfers = [
    { out_id: 8, in_id: 16, out_name: "Player 8", in_name: "Player 16", position: "MID" },
    { out_id: 9, in_id: 17, out_name: "Player 9", in_name: "Player 17", position: "MID" },
  ];
  plan.squad_ids = plan.squad_ids.map(replace);
  plan.players = plan.players.map(p => ({ ...p, id: replace(p.id), name: `Player ${replace(p.id)}` }));
  plan.gameweeks[0].starting_ids = plan.gameweeks[0].starting_ids.map(replace);
  e.action = structuredClone(plan.action);
  const clean = parseDecisionHistoryEntry(e);
  const r = result();
  r.actual!.squad_ids = r.actual!.squad_ids.map(replace);
  r.actual!.starting_ids = r.actual!.starting_ids.map(replace);
  r.actual!.transfers = [
    { out_id: 8, in_id: 17, time: "2026-08-24T12:00:00Z" },
    { out_id: 9, in_id: 16, time: "2026-08-24T12:00:00Z" },
  ];
  assert.equal(compareDecisionResult(clean, r).status, "followed");
  r.actual!.transfers.forEach(t => { t.time = "2026-08-22T12:00:00Z"; });
  assert.equal(compareDecisionResult(clean, r).status, "unknown", "A matching trade already made before recording cannot establish following the saved intention");
  r.actual!.transfers.forEach(t => { t.time = clean.saved_at; });
  assert.equal(compareDecisionResult(clean, r).status, "unknown", "Equal timestamps do not establish ordering");
  r.actual!.transfers.forEach(t => { t.time = "2026-08-24T12:00:00Z"; });
  r.actual!.transfers.push(
    { out_id: 16, in_id: 8, time: "2026-08-24T13:00:00Z" },
    { out_id: 8, in_id: 16, time: "2026-08-24T14:00:00Z" },
  );
  assert.equal(compareDecisionResult(clean, r).transfer_match, "deviated");
});

test("requires complete bench and vice information before saying the whole recommendation was followed", () => {
  const e = entry();
  e.plans!.best!.gameweeks[0].bench_ids = null;
  assert.equal(compareDecisionResult(e, result()).status, "unknown");
  const changed = result();
  changed.actual!.vice_captain_id = 13;
  assert.equal(compareDecisionResult(entry(), changed).lineup_match, "deviated");
});

test("different manager, season, or a post-deadline save cannot be attributed to a snapshot", () => {
  const e = entry();
  assert.equal(compareDecisionResult({ ...e, manager_id: 999 }, result()).net_points, null);
  assert.equal(compareDecisionResult({ ...e, season: "2025/26" }, result()).status, "unknown");
  assert.equal(compareDecisionResult({ ...e, saved_at: "2026-08-28T18:30:00Z" }, result()).status, "unknown");
});

test("handles Free Hit and Wildcard explicitly rather than pretending they are normal transfers", () => {
  for (const chip of ["freehit", "wildcard"] as const) {
    const r = result();
    r.actual!.active_chip = chip;
    const c = compareDecisionResult(entry(), r);
    assert.equal(c.chip_match, "deviated");
    assert.equal(c.status, "deviated");
    assert.ok(c.reasons.some(reason => reason.includes(chip === "freehit" ? "Free Hit" : "Wildcard")));
  }
});

test("does not fetch unfinalized picks or declare in-progress points final", async () => {
  const urls: string[] = [];
  const fetcher: typeof fetch = async input => {
    urls.push(String(input)); return Response.json(bootstrap(false));
  };
  const r = await loadDecisionResult(123, 2, fetcher, Date.parse("2026-08-30T20:00:00Z"));
  assert.equal(r.finalized, false);
  assert.equal(r.actual, null);
  assert.equal(urls.length, 1);
});

test("uses the fixed configured manager and public source after finalization", async () => {
  const urls: string[] = [];
  const fetcher: typeof fetch = async input => {
    const url = String(input); urls.push(url);
    const live = { elements: squad.map(id => ({ id, stats: { minutes: 90, total_points: 2 } })) };
    return Response.json(url.endsWith("bootstrap-static/") ? bootstrap() : url.endsWith("/picks/") ? picks() : url.endsWith("/live/") ? live : []);
  };
  const r = await loadDecisionResult(123, 2, fetcher, Date.parse("2026-08-30T20:00:00Z"));
  assert.equal(r.actual?.net_points, 37);
  assert.equal(r.player_scores?.length, 15);
  assert.ok(urls.includes("https://fantasy.premierleague.com/api/entry/123/event/2/picks/"));
  assert.ok(urls.every(url => url.startsWith("https://fantasy.premierleague.com/api/")));
});

test("route denies anonymous callers and rejects client-supplied managers before fetching", async () => {
  let calls = 0;
  const load: typeof loadDecisionResult = async (manager, event) => { calls++; assert.equal(manager, 123); assert.equal(event, 2); return result(); };
  const anonymous = await decisionResultsResponse(new Request("https://app.test/api/decision-results?event=2"), null, () => { throw new Error("Must not access config"); }, load);
  assert.equal(anonymous.status, 401);
  for (const query of ["event=2&manager_id=999", "event=2&event=3", "event=39", "event=0"]) {
    const bad = await decisionResultsResponse(new Request("https://app.test/api/decision-results?" + query), {}, () => 123, load);
    assert.equal(bad.status, 400);
  }
  assert.equal(calls, 0);
  const ok = await decisionResultsResponse(new Request("https://app.test/api/decision-results?event=2"), {}, () => 123, load);
  assert.equal(ok.status, 200);
  assert.equal(ok.headers.get("Cache-Control"), "no-store");
  assert.equal(calls, 1);
});

test("frozen replay uses the saved vice and first eligible substitute when the captain never plays", () => {
  const e = entry(), r = result();
  Object.assign(r.player_scores!.find(p => p.id === 14)!, { minutes: 0, points: 0 });
  r.player_scores!.find(p => p.id === 12)!.points = 7;
  r.player_scores!.find(p => p.id === 11)!.points = 5;
  const replay = replayFrozenDecisionPlan(e, e.plans!.best, r);
  assert.equal(replay.available, true);
  assert.equal(replay.effective_captain_id, 12);
  assert.deepEqual(replay.automatic_subs, [{ out_id: 14, in_id: 11 }]);
  assert.equal(replay.net_points, 37);
});

test("frozen replay keeps three defenders and respects bench priority rather than actual points", () => {
  const e = entry(), r = result();
  Object.assign(r.player_scores!.find(p => p.id === 2)!, { minutes: 0, points: 0 });
  r.player_scores!.find(p => p.id === 11)!.points = 100;
  r.player_scores!.find(p => p.id === 5)!.points = 3;
  r.player_scores!.find(p => p.id === 6)!.points = 15;
  const replay = replayFrozenDecisionPlan(e, e.plans!.best, r);
  assert.deepEqual(replay.automatic_subs, [{ out_id: 2, in_id: 5 }]);
  assert.equal(replay.net_points, 25);
  assert.ok(!replay.scoring_ids.includes(11));
  assert.ok(!replay.scoring_ids.includes(6));
});

test("frozen replay treats a zero-point cameo as participation and a missing score as unknown", () => {
  const e = entry(), r = result();
  Object.assign(r.player_scores!.find(p => p.id === 14)!, { minutes: 1, points: 0 });
  r.player_scores!.find(p => p.id === 11)!.points = 20;
  const replay = replayFrozenDecisionPlan(e, e.plans!.best, r);
  assert.equal(replay.effective_captain_id, 14);
  assert.deepEqual(replay.automatic_subs, []);
  assert.equal(replay.net_points, 20);
  r.player_scores = r.player_scores!.filter(p => p.id !== 11);
  assert.equal(replayFrozenDecisionPlan(e, e.plans!.best, r).available, false);
});

test("frozen replay can return a genuine all-zero gameweek without a captain or playing bench", () => {
  const e = entry(), r = result();
  r.player_scores!.forEach(p => { p.minutes = 0; p.points = 0; });
  const replay = replayFrozenDecisionPlan(e, e.plans!.best, r);
  assert.equal(replay.available, true);
  assert.equal(replay.net_points, 0);
  assert.equal(replay.effective_captain_id, null);
  assert.deepEqual(replay.automatic_subs, []);
});

test("frozen replay deducts the plan's new hit once and does not deduct the official historic hit baseline", () => {
  const e = entry(), r = result();
  const plan = structuredClone(e.plans!.best!);
  plan.action.hit_points = 4;
  const replay = replayFrozenDecisionPlan(e, plan, r);
  assert.equal(replay.gross_points, 24);
  assert.equal(replay.planned_hit_points, 4);
  assert.equal(replay.net_points, 20);
  assert.equal(r.actual!.transfer_cost, 8, "Official cost can include previously incurred hits and is not deducted again");
});

test("frozen replay supports explicit TC and BB but refuses a rebuild or incomplete saved lineup", () => {
  const e = entry(), r = result();
  e.context!.planned_chip = "3xc";
  assert.equal(replayFrozenDecisionPlan(e, e.plans!.best, r).net_points, 26);
  e.context!.planned_chip = "bboost";
  assert.equal(replayFrozenDecisionPlan(e, e.plans!.best, r).net_points, 32);
  e.context!.planned_chip = "freehit";
  assert.equal(replayFrozenDecisionPlan(e, e.plans!.best, r).available, false);
  e.context!.planned_chip = null;
  e.plans!.best!.gameweeks[0].bench_ids = null;
  assert.equal(replayFrozenDecisionPlan(e, e.plans!.best, r).available, false);
});
