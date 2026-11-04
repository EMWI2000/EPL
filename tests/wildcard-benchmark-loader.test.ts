import assert from "node:assert/strict";
import test from "node:test";
import { loadWildcardBenchmark } from "../lib/wildcard-benchmark-loader.ts";

const OWN = 999;
const now = Date.UTC(2026, 7, 1 + 9 * 7) - 1_000;

function fixture(options: { missing?: number; invalid?: number; partial?: boolean; size?: number; reverse?: boolean } = {}) {
  const calls: string[] = [];
  const peers = Array.from({ length: options.size ?? 45 }, (_, index) => index + 1);
  if (options.reverse) peers.reverse();
  const fetcher = (async (url: string | URL | Request) => {
    const path = String(url).replace("https://fantasy.premierleague.com/api/", "");
    calls.push(path);
    if (path === `entry/${OWN}/`) return Response.json({ leagues: { classic: [
      { id: 1, league_type: "x", entry_rank: 10 },
      { id: 2, league_type: "x", entry_rank: 10 },
    ] } });
    if (path === "bootstrap-static/") return Response.json({ events: Array.from({ length: 38 }, (_, index) => ({
      id: index + 1,
      deadline_time: new Date(Date.UTC(2026, 7, 1 + index * 7)).toISOString(),
      finished: index < 9,
      data_checked: index < 9,
    })) });
    if (path.startsWith("leagues-classic/")) {
      if (path.includes("page_standings=2")) return new Response("Unavailable", { status: 503 });
      return Response.json({ league: { start_event: 1 }, standings: {
        page: 1, has_next: options.partial ?? false,
        results: [...peers, OWN].map((entry, index) => ({ entry, rank: index + 1, total: 450 })),
      } });
    }
    const id = Number(/^entry\/(\d+)\/history\/$/.exec(path)?.[1]);
    if (!Number.isSafeInteger(id)) throw new Error(`Unexpected path ${path}`);
    if (id === options.missing) return new Response("Unavailable", { status: 503 });
    let total = 0;
    return Response.json({ chips: id <= 12 ? [{ name: "wildcard", event: 4 }] : [],
      current: Array.from({ length: 9 }, (_, index) => {
        const points = 40 + id % 5 + (id <= 12 && index >= 3 ? 10 : 0);
        total += points;
        return { event: index + 1, points: id === options.invalid ? null : points,
          event_transfers_cost: 0, total_points: total };
      }),
    });
  }) as typeof fetch;
  return { fetcher, calls };
}

test("benchmark loader deduplicates league members and excludes own team from references", async () => {
  const { fetcher, calls } = fixture();
  const result = await loadWildcardBenchmark(OWN, fetcher, now);
  assert.equal(result.season, "2026/27");
  assert.equal(result.target_event, 10);
  assert.equal(result.half, 1);
  assert.equal(result.coverage.complete, true);
  assert.equal(result.coverage.discovered_peers, 45);
  assert.equal(result.coverage.histories_loaded, 45);
  assert.equal(result.coverage.leagues_complete, 2);
  assert.equal(result.detected_wildcards, 12);
  assert.equal(calls.filter(path => path.includes("/history/")).length, 46);
  assert.equal(calls.filter(path => path === `entry/${OWN}/history/`).length, 1);
  assert.equal(result.windows[2].sample_size, 12);
});

test("missing and malformed histories remain visible as incomplete coverage", async () => {
  for (const options of [{ missing: 2 }, { invalid: 2 }]) {
    const { fetcher } = fixture(options);
    const result = await loadWildcardBenchmark(OWN, fetcher, now);
    assert.equal(result.coverage.complete, false);
    assert.equal(result.coverage.histories_loaded, 44);
    assert.equal(result.coverage.missing_histories, 1);
    assert.equal(result.detected_wildcards, 11);
  }
});

test("incomplete standings never produce a complete league benchmark", async () => {
  const { fetcher } = fixture({ partial: true });
  const result = await loadWildcardBenchmark(OWN, fetcher, now);
  assert.equal(result.coverage.complete, false);
  assert.equal(result.coverage.leagues_complete, 0);
  assert.equal(result.coverage.histories_loaded, 45);
});

test("large cohorts use a stable bounded sample independent of league rank ordering", async () => {
  const first = fixture({ size: 230 });
  const second = fixture({ size: 230, reverse: true });
  const result = await loadWildcardBenchmark(OWN, first.fetcher, now);
  await loadWildcardBenchmark(OWN, second.fetcher, now);
  assert.equal(result.coverage.selection, "stable_sample");
  assert.equal(result.coverage.discovered_peers, 230);
  assert.equal(result.coverage.sampled_peers, 200);
  assert.equal(result.coverage.histories_loaded, 200);
  assert.equal(result.coverage.complete, false);
  const histories = (calls: string[]) => calls.filter(path => path.endsWith("/history/")).sort();
  assert.deepEqual(histories(first.calls), histories(second.calls));
  assert.equal(histories(first.calls).length, 201);
});
