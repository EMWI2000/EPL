import assert from "node:assert/strict";
import test from "node:test";

import {
  DECISION_HISTORY_STORAGE_KEY,
  MAX_DECISION_HISTORY_ENTRIES,
  buildDecisionHistoryEntry,
  clearDecisionHistory,
  emptyDecisionHistory,
  parseDecisionHistory,
  parseDecisionHistoryEntry,
  readDecisionHistory,
  upsertDecisionHistory,
  writeDecisionHistory,
  type DecisionHistoryEntry,
  type StorageLike,
} from "../lib/decision-history.ts";
import type { AiReviewResponse } from "../lib/ai-review-contract.ts";
import type { PlannerAction, PlannerPayload } from "../lib/planner-contract.ts";

function player(id: number, name = `Player ${id}`) {
  return {
    id,
    name,
    team: "TST",
    position: "MID" as const,
    price_tenths: 50,
    status: "a" as const,
    price_signal: null,
  };
}

function ownedPlayer(id: number, name = `Player ${id}`) {
  return {
    ...player(id, name),
    purchase_price_tenths: 50,
    selling_price_tenths: 50,
  };
}

function action(kind: "roll" | "transfer" = "roll", incomingId = 100): PlannerAction {
  const transfers = kind === "roll" ? [] : [{
    out_id: 8,
    in_id: incomingId,
    position: "MID" as const,
    out_purchase_price_tenths: 50,
    out_current_price_tenths: 50,
    out_selling_price_tenths: 50,
    in_price_tenths: 50,
    out: player(8),
    in: player(incomingId),
  }];
  const squadIds = Array.from({ length: 15 }, (_, index) => index + 1);
  const startingIds = Array.from({ length: 11 }, (_, index) => index + 1);
  if (kind === "transfer") {
    squadIds[squadIds.indexOf(8)] = incomingId;
    startingIds[startingIds.indexOf(8)] = incomingId;
  }
  return {
    kind,
    transfer_count: transfers.length,
    transfers,
    squad_ids: squadIds,
    gameweeks: [{
      gameweek: 2,
      starting_ids: startingIds,
      captain_id: kind === "transfer" ? incomingId : 1,
      formation: "3-5-2",
      projected_points: 60,
    }],
    bank_before_tenths: 10,
    bank_after_tenths: 10,
    free_transfers_before: 1,
    free_transfers_next_gameweek: kind === "roll" ? 2 : 1,
    hit_points: 0,
    projected_points: kind === "roll" ? 60 : 64,
    banked_ft_value_points: kind === "roll" ? 1 : 0,
    decision_value_points: kind === "roll" ? 61 : 64,
    net_points_vs_roll: kind === "roll" ? 0 : 4,
    decision_value_vs_roll: kind === "roll" ? 0 : 3,
    explanation: "Test action",
  };
}

function planner(): PlannerPayload {
  return {
    manager_id: 8425806,
    state_fingerprint: "a".repeat(64),
    source_event: 1,
    target_event: 2,
    confirmed_state: {
      bank_tenths: 10,
      free_transfers: 1,
      no_active_chip_confirmed: true,
      squad: Array.from({ length: 15 }, (_, index) => ownedPlayer(index + 1)),
    },
    best_action: action("roll"),
    alternatives: [action("transfer")],
    sequential: null,
    method: {
      candidate_count: 45,
      plans_per_transfer_count: 5,
      higher_transfer_count_plans: 1,
      maximum_immediate_transfers: 2,
      roll_ft_value_points: 1,
      chips_modelled: false,
      next_deadline_transfer_modelled: false,
      future_transfers_modelled: false,
    },
  };
}

function aiReview(verdict: "confirm_best_action" | "wait_for_information" | "prefer_alternative"): AiReviewResponse {
  return {
    schema_version: "fpl-ai-review-response-v2",
    generated_at: "2026-08-23T12:03:00Z",
    recommendation_generated_at: "2026-08-23T12:01:00Z",
    target_event: 2,
    model: "gpt-5.6-sol",
    reasoning_effort: "xhigh",
    review: {
      verdict,
      alternative_index: verdict === "prefer_alternative" ? 0 : null,
      execution_timing: verdict === "wait_for_information" ? "wait_for_team_news" : "act_now",
      headline: "Afvent holdnyt",
      summary: "Test",
      rationale: ["A", "B"],
      risks: [],
      change_triggers: ["Nyt"],
      deadline_checklist: ["A", "B"],
      evidence_summary: "Test",
      data_gaps: [],
      confidence: "medium",
      strategic_outlook: {
        horizon_gameweeks: 2,
        posture: "preserve_flexibility",
        summary: "Test",
        priorities: ["A"],
        watchpoints: [{ subject: "A", reason: "B", trigger: "C", earliest_gameweek: 2 }],
        scope: "advisory_only_no_unmodelled_transfers_or_chips",
      },
      qualitative_evidence: [{
        subject: "A",
        category: "other",
        finding: "B",
        basis: "inference",
        impact: "neutral",
        freshness: "unknown",
        confidence: "low",
      }],
    },
    research: { performed: true, sources: [] },
  };
}

function build(review: AiReviewResponse | null = null, savedAt = "2026-08-23T12:05:00Z") {
  return buildDecisionHistoryEntry({
    saved_at: savedAt,
    target_deadline: "2026-08-28T18:30:00Z",
    state_observed_at: "2026-08-23T12:00:00Z",
    recommendation_generated_at: "2026-08-23T12:01:00Z",
    forecast: { version: "v2", horizon: 2, validation_status: "unvalidated" },
    planner: planner(),
    ai_review: review,
  });
}

class MemoryStorage implements StorageLike {
  values = new Map<string, string>();
  removed: string[] = [];

  getItem(key: string) { return this.values.get(key) ?? null; }
  setItem(key: string, value: string) { this.values.set(key, value); }
  removeItem(key: string) { this.removed.push(key); this.values.delete(key); }
}

test("builds an allowlisted compact entry without identity or secret-shaped fields", () => {
  const source = planner() as PlannerPayload & Record<string, unknown>;
  source.github_id = "199608244";
  source.internal_token = "secret-token";
  const entry = buildDecisionHistoryEntry({
    saved_at: "2026-08-23T12:05:00Z",
    target_deadline: "2026-08-28T18:30:00Z",
    state_observed_at: "2026-08-23T12:00:00Z",
    recommendation_generated_at: "2026-08-23T12:01:00Z",
    forecast: { version: "v2", horizon: 2, validation_status: "unvalidated" },
    planner: source,
    ai_review: null,
  });
  const serialized = JSON.stringify(entry);
  assert.equal(entry.selection.kind, "best");
  assert.equal(entry.action?.kind, "roll");
  assert.doesNotMatch(serialized, /manager_id|github_id|internal_token|secret-token/);
});

test("preserves zero remaining free transfers for the current deadline", () => {
  const entry = build();
  entry.confirmed.free_transfers = 0;

  assert.equal(parseDecisionHistoryEntry(entry).confirmed.free_transfers, 0);
});

test("stores a validated alternative or wait verdict without raw AI output", () => {
  const alternative = build(aiReview("prefer_alternative"));
  assert.deepEqual(alternative.selection, { kind: "alternative", alternative_index: 0 });
  assert.equal(alternative.action?.transfers[0]?.in_id, 100);
  assert.deepEqual(alternative.lineup, { captain_id: 100, captain_name: "Player 100" });
  assert.equal(alternative.ai?.verdict, "prefer_alternative");
  assert.equal((alternative.ai as Record<string, unknown>).summary, undefined);

  const wait = build(aiReview("wait_for_information"));
  assert.deepEqual(wait.selection, { kind: "wait", alternative_index: null });
  assert.equal(wait.action, null);
  assert.equal(wait.lineup, null);
  assert.equal(wait.ai?.execution_timing, "wait_for_team_news");
});

test("rejects extra fields and inconsistent selection data", () => {
  const entry = build();
  assert.throws(
    () => parseDecisionHistoryEntry({ ...entry, token: "should-not-survive" }),
    /unexpected or missing fields/,
  );
  assert.throws(
    () => parseDecisionHistoryEntry({ ...entry, selection: { kind: "wait", alternative_index: null } }),
    /action does not match/,
  );
  assert.throws(
    () => parseDecisionHistoryEntry({
      ...entry,
      selection: { kind: "alternative", alternative_index: 0 },
      ai: null,
    }),
    /ai is required/,
  );
});

test("deduplicates deadlines, sorts newest first and keeps at most one season", () => {
  let history = emptyDecisionHistory();
  for (let event = 1; event <= MAX_DECISION_HISTORY_ENTRIES + 2; event += 1) {
    const entry = build(null, `2026-08-${String(Math.min(event, 28)).padStart(2, "0")}T12:00:00Z`) as DecisionHistoryEntry;
    entry.source_event = Math.max(1, Math.min(37, event));
    entry.target_event = Math.min(38, entry.source_event + 1);
    entry.target_deadline = new Date(Date.UTC(2026, 7, event + 1)).toISOString();
    history = upsertDecisionHistory(history, parseDecisionHistoryEntry(entry));
  }
  assert.equal(history.entries.length, MAX_DECISION_HISTORY_ENTRIES);
  assert.ok(Date.parse(history.entries[0].target_deadline) > Date.parse(history.entries.at(-1)!.target_deadline));

  const replacement = { ...history.entries[0], saved_at: "2026-09-30T12:00:00Z" };
  const updated = upsertDecisionHistory(history, replacement);
  assert.equal(updated.entries.length, MAX_DECISION_HISTORY_ENTRIES);
  assert.equal(updated.entries.find((entry) => entry.target_deadline === replacement.target_deadline)?.saved_at, "2026-09-30T12:00:00.000Z");
});

test("storage failures are nonfatal and clear only removes the history key", () => {
  const storage = new MemoryStorage();
  const history = upsertDecisionHistory(emptyDecisionHistory(), build());
  assert.equal(writeDecisionHistory(storage, history), true);
  assert.deepEqual(readDecisionHistory(storage), parseDecisionHistory(JSON.parse(storage.getItem(DECISION_HISTORY_STORAGE_KEY)!)));
  storage.setItem("unrelated", "keep");
  assert.equal(clearDecisionHistory(storage), true);
  assert.deepEqual(storage.removed, [DECISION_HISTORY_STORAGE_KEY]);
  assert.equal(storage.getItem("unrelated"), "keep");

  const broken: StorageLike = {
    getItem() { throw new Error("disabled"); },
    setItem() { throw new Error("quota"); },
    removeItem() { throw new Error("disabled"); },
  };
  assert.deepEqual(readDecisionHistory(broken), emptyDecisionHistory());
  assert.equal(writeDecisionHistory(broken, history), false);
  assert.equal(clearDecisionHistory(broken), false);
});
