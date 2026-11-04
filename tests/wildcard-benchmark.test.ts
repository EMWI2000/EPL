import assert from "node:assert/strict";
import test from "node:test";
import { analyzeWildcardBenchmarks } from "../lib/wildcard-benchmark.ts";

const OWN = 999;
function calendar(last = 9) {
  return { events: Array.from({ length: 38 }, (_, i) => ({ id: i + 1, finished: i < last, data_checked: i < last })) };
}
function history(options: {
  end?: number; score?: (event: number) => number;
  chips?: { name: string; event: number }[]; hits?: (event: number) => number;
} = {}) {
  return {
    current: Array.from({ length: options.end ?? 9 }, (_, i) => ({
      event: i + 1, points: options.score?.(i + 1) ?? 50, event_transfers_cost: options.hits?.(i + 1) ?? 0,
    })),
    chips: options.chips ?? [],
  };
}
function sample(caseCount = 10) {
  const histories = new Map<number, unknown>();
  for (let id = 1; id <= caseCount; id++) {
    histories.set(id, history({ score: event => event < 3 ? 50 : 60, chips: [{ name: "wildcard", event: 3 }] }));
  }
  for (let id = 101; id <= 112; id++) histories.set(id, history());
  histories.set(OWN, history({ score: event => event < 3 ? 50 : 60, chips: [{ name: "wildcard", event: 3 }] }));
  return histories;
}
function analyze(histories: ReadonlyMap<number, unknown>, last = 9, half: 1 | 2 = 1) {
  return analyzeWildcardBenchmarks({ bootstrap: calendar(last), histories, ownManagerId: OWN, half });
}

test("complete aligned windows use unweighted net points, same calendar peers, and exact midrank", () => {
  const result = analyze(sample());
  assert.equal(result.detected_wildcards, 10);
  assert.equal(result.valid_peer_histories, 22);
  assert.equal(result.invalid_peer_histories, 0);
  assert.equal(result.last_finalized_event, 9);
  for (const window of result.windows) {
    assert.equal(window.sample_size, 10);
    assert.deepEqual(window.net_points, { p25: 60 * window.horizon, median: 60 * window.horizon, p75: 60 * window.horizon });
    assert.deepEqual(window.peer_difference, { p25: 10 * window.horizon, median: 10 * window.horizon, p75: 10 * window.horizon });
    assert.deepEqual(window.reference_count, { min: 10, median: 10, max: 10 });
    assert.ok(Object.values(window.exclusions).every(count => count === 0));
  }
  assert.equal(result.own.used_event, 3);
  assert.ok(result.own.windows.every(window => window.status === "ready" && window.percentile === 50));
});

test("changing calendar scores never substitutes a season average for the same window", () => {
  const histories = sample();
  for (let id = 101; id <= 112; id++) histories.set(id, history({ score: event => event === 3 ? 100 : 50 }));
  histories.set(OWN, history({ score: event => event === 3 ? 110 : 50, chips: [{ name: "wildcard", event: 3 }] }));
  const own = analyze(histories).own.windows;
  assert.equal(own[0].peer_difference, 10);
  assert.equal(own[1].net_points, 210);
  assert.equal(own[1].peer_difference, 10);
});

test("matching chooses at most ten nearest pre-form peers with stable numeric-ID ties", () => {
  const histories = new Map<number, unknown>([[OWN, history({ score: event => event < 3 ? 50 : 20, chips: [{ name: "wildcard", event: 3 }] })]]);
  for (let i = 1; i <= 12; i++) histories.set(100 + i, history({ score: event => event < 3 ? 50 : i }));
  const own = analyze(histories).own.windows[0];
  assert.equal(own.reference_count, 10);
  assert.equal(own.peer_difference, 14.5);
  assert.deepEqual(analyze(new Map([...histories].reverse())), analyze(histories));
});

test("five-point caliper includes its exact boundary and requires five references", () => {
  const histories = new Map<number, unknown>([[OWN, history({ chips: [{ name: "wildcard", event: 3 }] })]]);
  for (let i = 1; i <= 4; i++) histories.set(i, history());
  histories.set(5, history({ score: event => event < 3 ? 55 : 60 }));
  histories.set(6, history({ score: event => event < 3 ? 56 : 100 }));
  assert.equal(analyze(histories).own.windows[0].reference_count, 5);
  histories.delete(5);
  const own = analyze(histories).own.windows[0];
  assert.equal(own.status, "insufficient_peers");
  assert.equal(own.reference_count, 4);
  assert.equal(own.peer_difference, null);
  assert.equal(own.net_points, 50);
});

test("own manager cannot rescue or change peer references and is excluded as a case", () => {
  const histories = new Map<number, unknown>([[OWN, history()], [1, history({ chips: [{ name: "wildcard", event: 3 }] })]]);
  for (let i = 101; i <= 104; i++) histories.set(i, history());
  const result = analyze(histories);
  assert.equal(result.detected_wildcards, 1);
  assert.equal(result.windows[0].sample_size, 0);
  assert.equal(result.windows[0].exclusions.insufficient_peers, 1);
  assert.equal(result.own.windows[0].status, "not_used");
});

test("follow-up spans may cross chip halves but any additional chip excludes the window", () => {
  const histories = new Map<number, unknown>();
  histories.set(OWN, history({ end: 22, chips: [{ name: "wildcard", event: 18 }] }));
  for (let i = 1; i <= 6; i++) histories.set(i, history({ end: 22 }));
  assert.equal(analyze(histories, 22).own.windows[2].status, "ready");
  histories.set(OWN, history({ end: 22, chips: [{ name: "wildcard", event: 18 }, { name: "bboost", event: 20 }] }));
  const result = analyze(histories, 22);
  assert.equal(result.own.windows[0].status, "ready");
  assert.equal(result.own.windows[1].status, "other_chip");
  assert.equal(result.own.windows[1].net_points, 150);
  assert.equal(result.own.windows[1].percentile, null);
});

test("controls can have a prior WC or use a later chip beyond the evaluated window", () => {
  const histories = new Map<number, unknown>([[OWN, history({ chips: [{ name: "wildcard", event: 6 }] })]]);
  for (let i = 1; i <= 6; i++) histories.set(i, history({ chips: [{ name: "wildcard", event: 3 }, { name: "bboost", event: 9 }] }));
  const own = analyze(histories).own.windows;
  assert.equal(own[1].status, "ready");
  assert.equal(own[1].reference_count, 6);
  assert.equal(own[2].status, "pending");
});

test("all post and pre gameweeks must be finalized and present", () => {
  const histories = sample();
  const incomplete = history({ end: 4, chips: [{ name: "wildcard", event: 3 }] });
  histories.set(OWN, incomplete);
  assert.equal(analyze(histories).own.windows[1].status, "missing_data");
  assert.equal(analyze(histories, 4).own.windows[1].status, "pending");
  const bootstrap = calendar();
  bootstrap.events[0].data_checked = false;
  assert.equal(analyzeWildcardBenchmarks({ bootstrap, histories: sample(), ownManagerId: OWN, half: 1 }).own.windows[0].status, "insufficient_history");
});

test("a WC in GW2 lacks the required two pre-period gameweeks", () => {
  const histories = sample();
  histories.set(OWN, history({ chips: [{ name: "wildcard", event: 2 }] }));
  const own = analyze(histories).own.windows[0];
  assert.equal(own.status, "insufficient_history");
  assert.equal(own.net_points, 50);
});

test("GW20 uses second-half WC and preserves previously used first-half chip history", () => {
  const histories = new Map<number, unknown>([[OWN, history({ end: 24, chips: [{ name: "wildcard", event: 6 }, { name: "wildcard", event: 20 }] })]]);
  for (let i = 1; i <= 6; i++) histories.set(i, history({ end: 24 }));
  assert.equal(analyze(histories, 24, 2).own.used_event, 20);
  assert.ok(analyze(histories, 24, 2).own.windows.every(window => window.status === "ready"));
});


test("late-season horizons distinguish impossible future rounds from pending data", () => {
  const histories = new Map<number, unknown>();
  histories.set(OWN, history({ end: 38, chips: [{ name: "wildcard", event: 38 }] }));
  histories.set(1, history({ end: 38, chips: [{ name: "wildcard", event: 38 }] }));
  for (let id = 101; id <= 106; id++) histories.set(id, history({ end: 38 }));
  const last = analyze(histories, 38, 2);
  assert.deepEqual(last.own.windows.map(window => window.status), ["ready", "season_end", "season_end"]);
  assert.equal(last.windows[1].exclusions.season_end, 1);
  assert.equal(last.windows[1].exclusions.unfinished, 0);
  histories.set(OWN, history({ end: 38, chips: [{ name: "wildcard", event: 36 }] }));
  assert.deepEqual(analyze(histories, 38, 2).own.windows.map(window => window.status), ["ready", "ready", "season_end"]);
});

test("invalid scores, duplicate rounds and illegal chip histories never become zeros", () => {
  const histories = sample();
  const invalid = history();
  invalid.current[0].points = NaN;
  histories.set(200, invalid);
  const duplicate = history();
  duplicate.current.push(duplicate.current[0]);
  histories.set(201, duplicate);
  histories.set(202, history({ chips: [{ name: "wildcard", event: 3 }, { name: "wildcard", event: 6 }] }));
  histories.set(203, history({ chips: [{ name: "wildcard", event: 3 }, { name: "bboost", event: 3 }] }));
  histories.set(204, history({ chips: [{ name: "unknown", event: 5 }] }));
  histories.set(205, history({ hits: () => -4 }));
  histories.set(206, history({ chips: [{ name: "wildcard", event: 1 }] }));
  histories.set(OWN, null);
  const result = analyze(histories);
  assert.equal(result.invalid_peer_histories, 7);
  assert.equal(result.valid_peer_histories, 22);
  assert.equal(result.windows[0].peer_difference?.median, 10);
  assert.equal(result.own.windows[0].status, "missing_data");
});

test("distributions need five cases and percentile needs ten independent peer WC cases", () => {
  const four = analyze(sample(4));
  assert.equal(four.windows[0].sample_size, 4);
  assert.equal(four.windows[0].net_points, null);
  assert.equal(four.windows[0].peer_difference, null);
  assert.equal(four.windows[0].reference_count, null);
  assert.equal(four.own.windows[0].percentile, null);
  const five = analyze(sample(5));
  assert.equal(five.windows[0].peer_difference?.median, 10);
  assert.equal(five.own.windows[0].percentile, null);
});

test("signed outcomes and transfer hits are retained in the observed score", () => {
  const histories = sample();
  histories.set(OWN, history({ score: event => event < 3 ? 50 : -1, hits: event => event === 4 ? 4 : 0,
    chips: [{ name: "wildcard", event: 3 }] }));
  const own = analyze(histories).own.windows[1];
  assert.equal(own.net_points, -7);
  assert.equal(own.peer_difference, -157);
  assert.equal(own.percentile, 0);
});

test("missing or malformed calendars fail rather than turning unfinished results into data", () => {
  assert.throws(() => analyzeWildcardBenchmarks({ bootstrap: {}, histories: sample(), ownManagerId: OWN, half: 1 }), /calendar/);
  const duplicate = calendar();
  duplicate.events.push(duplicate.events[0]);
  assert.throws(() => analyzeWildcardBenchmarks({ bootstrap: duplicate, histories: sample(), ownManagerId: OWN, half: 1 }), /calendar/);
  const first = analyze(sample(), 0);
  assert.equal(first.last_finalized_event, null);
  assert.equal(first.windows[0].sample_size, 0);
  assert.equal(first.windows[0].exclusions.unfinished, 10);
});
