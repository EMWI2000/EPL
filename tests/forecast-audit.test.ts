import assert from "node:assert/strict";
import test from "node:test";
import {
  FORECAST_AUDIT_SCHEMA_VERSION,
  FORECAST_AUDIT_STORAGE_KEY,
  addForecastAuditResults,
  buildForecastAuditResults,
  captureForecastAudit,
  emptyForecastAudit,
  evaluateForecastAudit,
  forecastAuditSeason,
  parseForecastAudit,
  parseForecastAuditCapture,
  parseForecastAuditResults,
  readForecastAudit,
  writeForecastAudit,
  type ForecastAuditCapture,
  type ForecastAuditResults,
} from "../lib/forecast-audit.ts";

function forecast(): ForecastAuditCapture {
  return {
    target_event: 4,
    target_deadline: "2026-09-12T12:30:00Z",
    generated_at: "2026-09-11T12:00:00Z",
    model_version: "v2.1",
    players: [
      { id: 1, name: "Raya", model_points: 4, model_minutes: 80, baseline_points: 3, baseline_minutes: 90 },
      { id: 2, name: "Player two", model_points: 7, model_minutes: 70, baseline_points: 5, baseline_minutes: 60 },
      { id: 3, name: "Player three", model_points: 10, model_minutes: 80, baseline_points: null, baseline_minutes: 80 },
    ],
  };
}

function results(): ForecastAuditResults {
  return {
    season: "2026-27",
    target_event: 4,
    target_deadline: "2026-09-12T12:30:00Z",
    checked_at: "2026-09-15T12:00:00Z",
    finalized: true,
    players: [
      { id: 1, points: 6, minutes: 90 },
      { id: 2, points: 3, minutes: 60 },
      { id: 3, points: 1, minutes: 90 },
      { id: 4, points: 20, minutes: 90 },
    ],
  };
}

const saved = () => captureForecastAudit(emptyForecastAudit(), forecast(), "2026-09-11T12:01:00Z");

test("forecast audit captures before deadline and never overwrites the first forecast", () => {
  const initial = saved();
  const later = { ...forecast(), model_version: "v2.2", players: forecast().players.map((row) => ({ ...row, model_points: 100 })) };
  assert.deepEqual(captureForecastAudit(initial, later, "2026-09-11T14:00:00Z"), initial);
  assert.deepEqual(captureForecastAudit(initial, later, "2026-09-15T14:00:00Z"), initial);
  assert.equal(initial.entries[0].season, "2026-27");
});

test("forecast audit refuses retrospective capture, deadline equality and future generation", () => {
  assert.throws(() => captureForecastAudit(emptyForecastAudit(), forecast(), "2026-09-12T12:30:00Z"));
  assert.throws(() => captureForecastAudit(emptyForecastAudit(), forecast(), "2026-09-15T12:00:00Z"));
  assert.throws(() => captureForecastAudit(emptyForecastAudit(), forecast(), "2026-09-11T11:59:59Z"));
  assert.throws(() => parseForecastAuditCapture({ ...forecast(), generated_at: forecast().target_deadline }));
  assert.throws(() => parseForecastAudit({ ...saved(), entries: [{ ...saved().entries[0], captured_at: forecast().target_deadline }] }));
});

test("forecast audit keeps January in the previous year's season and isolates reused IDs", () => {
  assert.equal(forecastAuditSeason("2027-01-10T12:00:00Z"), "2026-27");
  assert.equal(forecastAuditSeason("2027-08-10T12:00:00Z"), "2027-28");
  const secondSeason = { ...forecast(), generated_at: "2027-09-11T12:00:00Z", target_deadline: "2027-09-12T12:30:00Z" };
  const journal = captureForecastAudit(saved(), secondSeason, "2027-09-11T12:01:00Z");
  assert.equal(journal.entries.length, 2);
  assert.throws(() => addForecastAuditResults(saved(), { ...results(), season: "2027-28", target_deadline: secondSeason.target_deadline, checked_at: "2027-09-15T12:00:00Z" }));
  assert.throws(() => addForecastAuditResults(saved(), { ...results(), target_deadline: "2026-09-12T13:30:00Z" }));
});

test("forecast audit enforces unique gameweeks even when deadlines differ", () => {
  const entry = saved().entries[0];
  assert.throws(() => parseForecastAudit({ schema_version: FORECAST_AUDIT_SCHEMA_VERSION, entries: [entry, { ...entry, target_deadline: "2026-09-13T12:30:00Z" }] }));
});

test("forecast audit metrics use common coverage, stored baselines and no captain multiplier", () => {
  const journal = addForecastAuditResults(saved(), results());
  assert.equal(journal.entries[0].results?.players.length, 3, "drops irrelevant player results");
  assert.equal(journal.entries[0].players[0].baseline_points, 3);
  const metric = evaluateForecastAudit(journal.entries[0]);
  assert.deepEqual(metric.points, { count: 2, model_mae: 3, baseline_mae: 2.5, model_bias: 1, baseline_bias: -0.5 });
  assert.equal(metric.minutes?.count, 3);
  assert.equal(metric.minutes?.model_mae, 10);
  assert.equal(metric.minutes?.baseline_mae, 10 / 3);
  assert.equal(metric.total, 3);
});

test("forecast audit does not treat absent actuals or baselines as zero", () => {
  const partial = { ...results(), players: [results().players[0]] };
  const metric = evaluateForecastAudit(addForecastAuditResults(saved(), partial).entries[0]);
  assert.equal(metric.points?.count, 1);
  assert.equal(metric.points?.model_mae, 2);
  assert.equal(metric.points?.baseline_mae, 3);
  const noBaseline = { ...forecast(), players: forecast().players.map((row) => ({ ...row, baseline_points: null, baseline_minutes: null })) };
  const journal = captureForecastAudit(emptyForecastAudit(), noBaseline, "2026-09-11T12:01:00Z");
  const noMetrics = evaluateForecastAudit(addForecastAuditResults(journal, results()).entries[0]);
  assert.equal(noMetrics.points, null);
  assert.equal(noMetrics.minutes, null);
  assert.equal(evaluateForecastAudit(saved().entries[0]).points, null);
});

test("forecast audit rejects unfinalized results, unknown rounds and mismatched seasons", () => {
  assert.throws(() => parseForecastAuditResults({ ...results(), finalized: false }));
  assert.throws(() => parseForecastAuditResults({ ...results(), season: "2025-26" }));
  assert.throws(() => parseForecastAuditResults({ ...results(), checked_at: "2026-09-11T12:00:00Z" }));
  assert.throws(() => addForecastAuditResults(saved(), { ...results(), target_event: 5 }));
  assert.throws(() => addForecastAuditResults(saved(), { ...results(), players: [{ id: 999, points: 4, minutes: 90 }] }));
});

test("forecast audit projects only sanitized official scores after both finalization flags", () => {
  const bootstrap = { events: [{ id: 4, deadline_time: forecast().target_deadline, finished: true, data_checked: true }], secret: "not forwarded" };
  const live = { elements: [{ id: 1, stats: { total_points: 6, minutes: 90, other: "not forwarded" } }] };
  const projected = buildForecastAuditResults(bootstrap, live, 4, forecast().target_deadline, results().checked_at);
  assert.deepEqual(projected.players, [{ id: 1, points: 6, minutes: 90 }]);
  assert.equal(JSON.stringify(projected).includes("not forwarded"), false);
  assert.throws(() => buildForecastAuditResults({ events: [{ ...bootstrap.events[0], data_checked: false }] }, live, 4, forecast().target_deadline, results().checked_at));
  assert.throws(() => buildForecastAuditResults({ events: [{ ...bootstrap.events[0], finished: false }] }, live, 4, forecast().target_deadline, results().checked_at));
  assert.throws(() => buildForecastAuditResults(bootstrap, live, 4, "2025-09-12T12:30:00Z", results().checked_at));
  assert.throws(() => buildForecastAuditResults(bootstrap, { elements: [{ id: 1, stats: { minutes: 90 } }] }, 4, forecast().target_deadline, results().checked_at));
});

test("forecast audit validates bounds, duplicates, timestamps and rejects unexpected private fields", () => {
  const baseline = forecast();
  assert.throws(() => parseForecastAuditCapture({ ...baseline, manager_id: 8425806 }));
  assert.throws(() => parseForecastAuditCapture({ ...baseline, players: [...baseline.players, baseline.players[0]] }));
  assert.throws(() => parseForecastAuditCapture({ ...baseline, target_event: 39 }));
  assert.throws(() => parseForecastAuditCapture({ ...baseline, players: [] }));
  assert.throws(() => parseForecastAuditCapture({ ...baseline, generated_at: "2026-09-11T12:00:00" }));
  assert.throws(() => parseForecastAuditCapture({ ...baseline, players: [{ ...baseline.players[0], model_points: NaN }] }));
  assert.throws(() => parseForecastAuditCapture({ ...baseline, players: [{ ...baseline.players[0], baseline_minutes: -1 }] }));
  assert.throws(() => parseForecastAuditResults({ ...results(), players: [...results().players, results().players[0]] }));
});

test("forecast audit storage round-trips and preserves corrupt or inaccessible data", () => {
  const data = new Map<string, string>();
  const storage = { getItem: (key: string) => data.get(key) ?? null, setItem: (key: string, value: string) => { data.set(key, value); } };
  assert.deepEqual(readForecastAudit(storage), { journal: emptyForecastAudit(), error: null });
  assert.equal(writeForecastAudit(storage, saved()), true);
  assert.deepEqual(readForecastAudit(storage).journal, saved());
  data.set(FORECAST_AUDIT_STORAGE_KEY, "corrupt");
  assert.notEqual(readForecastAudit(storage).error, null);
  assert.equal(data.get(FORECAST_AUDIT_STORAGE_KEY), "corrupt");
  const unavailable = { getItem() { throw new Error("blocked"); }, setItem() { throw new Error("full"); } };
  assert.notEqual(readForecastAudit(unavailable).error, null);
  assert.equal(writeForecastAudit(unavailable, saved()), false);
  data.set(FORECAST_AUDIT_STORAGE_KEY, "x".repeat(2 * 1024 * 1024 + 1));
  assert.notEqual(readForecastAudit(storage).error, null);
});

test("forecast audit results are repeatable and do not mutate the captured inputs", () => {
  const journal = saved();
  const before = JSON.stringify(journal);
  const once = addForecastAuditResults(journal, results());
  const twice = addForecastAuditResults(once, results());
  assert.deepEqual(once, twice);
  assert.equal(JSON.stringify(journal), before);
  assert.deepEqual(once.entries[0].players, journal.entries[0].players);
});
