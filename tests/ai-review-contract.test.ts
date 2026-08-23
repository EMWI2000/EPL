import assert from "node:assert/strict";
import test from "node:test";

import {
  AiReviewContractError,
  actionForAiReview,
  assertAiReviewRequestIsFresh,
  buildAiReviewRequest,
  managerRankBand,
  parseAiReviewModelOutput,
  parseAiReviewRequest,
  parseAiReviewResponse,
  type AiReviewModelOutput,
  type AiReviewRecommendationSource,
} from "../lib/ai-review-contract.ts";
import type { ManagerSyncResponse, PlannerPayload } from "../lib/planner-contract.ts";
import { parsePlannerPayload } from "../lib/planner-contract.ts";

function position(id: number): "GKP" | "DEF" | "MID" | "FWD" {
  if (id <= 2) return "GKP";
  if (id <= 7 || id === 100) return "DEF";
  if (id <= 12) return "MID";
  return "FWD";
}

function player(id: number, forcedPosition = position(id)) {
  return {
    id,
    name: `Player ${id}`,
    team: `Club ${((id - 1) % 5) + 1}`,
    position: forcedPosition,
    price_tenths: 50,
    status: "a",
    price_signal: null,
  };
}

function action(kind: "roll" | "transfer", replacementId = 100) {
  const isTransfer = kind === "transfer";
  const transfers = isTransfer
    ? [{
        out_id: 3,
        in_id: replacementId,
        position: "DEF",
        out_purchase_price_tenths: 50,
        out_current_price_tenths: 50,
        out_selling_price_tenths: 50,
        in_price_tenths: 50,
        out: player(3, "DEF"),
        in: player(replacementId, "DEF"),
      }]
    : [];
  const squadIds = Array.from({ length: 15 }, (_, index) => index + 1);
  const starters = [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15];
  if (isTransfer) {
    squadIds[squadIds.indexOf(3)] = replacementId;
    starters[starters.indexOf(3)] = replacementId;
  }
  return {
    kind,
    transfer_count: transfers.length,
    transfers,
    squad_ids: squadIds,
    gameweeks: [7, 8].map((gameweek, index) => ({
      gameweek,
      starting_ids: starters,
      captain_id: 13,
      formation: "3-4-3",
      projected_points: index === 0 ? 72.5 : 70,
    })),
    bank_before_tenths: 10,
    bank_after_tenths: 10,
    free_transfers_before: 1,
    free_transfers_next_gameweek: isTransfer ? 1 : 2,
    hit_points: 0,
    projected_points: 132,
    banked_ft_value_points: isTransfer ? 0 : 0.8,
    decision_value_points: isTransfer ? 132 : 132.8,
    net_points_vs_roll: isTransfer ? -0.2 : 0,
    decision_value_vs_roll: isTransfer ? -0.8 : 0,
    explanation: isTransfer ? "Sælg Player 3 og køb Player 100." : "Gem transferen.",
  };
}

function plannerFixture(): PlannerPayload {
  return parsePlannerPayload({
    manager_id: 123,
    state_fingerprint: "a".repeat(64),
    source_event: 6,
    target_event: 7,
    confirmed_state: {
      bank_tenths: 10,
      free_transfers: 1,
      no_active_chip_confirmed: true,
      squad: Array.from({ length: 15 }, (_, index) => player(index + 1)),
    },
    best_action: action("roll"),
    alternatives: [action("transfer")],
    method: {
      candidate_count: 45,
      plans_per_transfer_count: 5,
      higher_transfer_count_plans: 1,
      maximum_immediate_transfers: 2,
      roll_ft_value_points: 0.8,
      chips_modelled: false,
      future_transfers_modelled: false,
    },
  });
}

const limitations = [
  "state_is_locked_at_last_public_deadline",
  "current_free_transfers_not_public",
  "current_confirmed_transfers_not_public",
  "purchase_and_selling_prices_not_public",
  "next_deadline_chip_selection_not_public",
] as const;

function syncFixture(): ManagerSyncResponse {
  return {
    manager: { id: 123, overall_rank: 42_000 },
    target: { deadline_time: "2026-08-30T17:30:00Z" },
    snapshot: { observed_at: "2026-08-23T12:00:00.123456+00:00" },
    last_deadline_state: { limitations: [...limitations] },
  } as unknown as ManagerSyncResponse;
}

function recommendationFixture(): AiReviewRecommendationSource {
  const planner = plannerFixture();
  const starters = planner.best_action.gameweeks[0].starting_ids;
  const bench = planner.best_action.squad_ids.filter((id) => !starters.includes(id));
  return {
    meta: {
      generated_at: "2026-08-23T12:01:00.654321+00:00",
      horizon: 2,
      forecast_version: "v2",
      include_doubtful: true,
      validation: { status: "unvalidated" },
      data_sources: {
        price_signals: {
          available: true,
          price_change_deadlines: ["2026-08-23T23:00:00Z"],
        },
      },
    },
    team: {
      squad: planner.best_action.squad_ids.map((id) => ({
        id,
        weighted_ep: 12.5,
        projections: [7, 8].map((gameweek) => ({
          gameweek,
          ep: 6.2,
          expected_minutes: 83,
          appearance_probability: 0.96,
          sixty_probability: 0.9,
          confidence: 0.72,
          reliability: "medium" as const,
          fixtures_count: 1,
          is_blank: false,
          is_dgw: false,
        })),
      })),
      gameweeks: [{
        gameweek: 7,
        formation: "3-4-3",
        starting_ids: starters,
        bench_ids: bench,
        captain_id: 13,
        vice_captain_id: 14,
      }],
    },
    planner,
  };
}

function reviewFixture(): AiReviewModelOutput {
  return {
    verdict: "confirm_best_action",
    alternative_index: null,
    execution_timing: "act_now",
    headline: "Rul transferen og behold fleksibiliteten",
    summary: "Ingen af de kontrollerede nyheder ændrer solverens konklusion.",
    rationale: [
      "Rul har højere beslutningsværdi end de beregnede alternativer.",
      "Den forventede startopstilling har høj sandsynlighed for minutter.",
    ],
    risks: ["Holdnyt kan ændre sig efter pressemøderne."],
    change_triggers: ["Genberegn ved en skade i start-XI."],
    deadline_checklist: ["Kontrollér holdnyt.", "Bekræft bank og frie transfers."],
    evidence_summary: "Der blev ikke fundet en bekræftet ændring i spillernes status.",
    data_gaps: ["Fremtidige transfersekvenser er ikke modelleret."],
    confidence: "medium",
    strategic_outlook: {
      horizon_gameweeks: 2,
      posture: "preserve_flexibility",
      summary: "Rulningen holder flest muligheder åbne gennem GW8.",
      priorities: ["Bevar fleksibilitet.", "Overvåg start-XI-minutter."],
      watchpoints: [{
        subject: "Player 13",
        reason: "Kaptajnens minutter er afgørende.",
        trigger: "Genberegn ved holdnyt.",
        earliest_gameweek: 7,
      }],
      scope: "advisory_only_no_unmodelled_transfers_or_chips",
    },
    qualitative_evidence: [{
      subject: "Player 13",
      category: "minutes_role",
      finding: "Solveren estimerer en stabil rolle.",
      basis: "solver_interpretation",
      impact: "supports_best_action",
      freshness: "today",
      confidence: "medium",
    }],
  };
}

test("builds and validates a compact review request tied to the solver result", () => {
  const request = buildAiReviewRequest(syncFixture(), recommendationFixture());
  const parsed = parseAiReviewRequest(request, parsePlannerPayload);

  assert.equal(parsed.manager_id, 123);
  assert.equal(parsed.manager_rank_band, "top_100k");
  assert.equal(parsed.recommendation_generated_at, "2026-08-23T12:01:00.654Z");
  assert.equal(parsed.state_observed_at, "2026-08-23T12:00:00.123Z");
  assert.equal(parsed.squad_context.length, 15);
  assert.deepEqual(parsed.lineup.starting_ids, parsed.planner.best_action.gameweeks[0].starting_ids);
});

test("rejects reviews after the deadline or when the recommendation is older than 24 hours", () => {
  const request = buildAiReviewRequest(syncFixture(), recommendationFixture());

  assert.doesNotThrow(() => assertAiReviewRequestIsFresh(request, new Date("2026-08-24T11:59:59Z")));
  assert.throws(
    () => assertAiReviewRequestIsFresh(request, new Date("2026-08-24T12:01:01Z")),
    /older than 24 hours/,
  );
  assert.throws(
    () => assertAiReviewRequestIsFresh(request, new Date("2026-08-30T17:30:00Z")),
    /has passed/,
  );

  const staleManagerState = structuredClone(request);
  staleManagerState.recommendation_generated_at = "2026-08-24T12:00:00Z";
  assert.throws(
    () => assertAiReviewRequestIsFresh(staleManagerState, new Date("2026-08-24T12:00:01Z")),
    /synchronize the manager state/,
  );
});

test("request validation rejects extra fields, mismatched manager and inconsistent lineup", () => {
  const base = buildAiReviewRequest(syncFixture(), recommendationFixture());
  const withSecret = structuredClone(base) as typeof base & { access_token: string };
  withSecret.access_token = "secret";
  assert.throws(() => parseAiReviewRequest(withSecret, parsePlannerPayload), AiReviewContractError);

  const wrongManager = structuredClone(base);
  wrongManager.manager_id = 999;
  assert.throws(() => parseAiReviewRequest(wrongManager, parsePlannerPayload), /manager_id/);

  const wrongLineup = structuredClone(base);
  wrongLineup.lineup.captain_id = wrongLineup.lineup.bench_ids[0];
  assert.throws(() => parseAiReviewRequest(wrongLineup, parsePlannerPayload), /lineup/);
});

test("model output can only select an existing solver action", () => {
  const best = reviewFixture();
  assert.strictEqual(parseAiReviewModelOutput(best, 1, 2, 7), best);

  const alternative = { ...reviewFixture(), verdict: "prefer_alternative", alternative_index: 0 } as const;
  const parsed = parseAiReviewModelOutput(alternative, 1, 2, 7);
  assert.equal(parsed.alternative_index, 0);
  assert.equal(actionForAiReview(plannerFixture(), parsed)?.transfers[0]?.in.name, "Player 100");

  assert.throws(
    () => parseAiReviewModelOutput({ ...alternative, alternative_index: 1 }, 1),
    /alternative_index/,
  );
  assert.throws(
    () => parseAiReviewModelOutput({ ...best, invented_transfer: "Player X" }, 1),
    /unsupported invented_transfer/,
  );

  assert.throws(
    () => parseAiReviewModelOutput({
      ...best,
      verdict: "wait_for_information",
      execution_timing: "act_now",
    }, 1, 2, 7),
    /execution_timing/,
  );
  assert.throws(
    () => parseAiReviewModelOutput({
      ...best,
      strategic_outlook: { ...best.strategic_outlook, horizon_gameweeks: 3 },
    }, 1, 2, 7),
    /solver horizon/,
  );
});

test("browser response validation accepts only deduplicated allowed HTTPS sources", () => {
  const response = {
    schema_version: "fpl-ai-review-response-v2",
    generated_at: "2026-08-23T12:02:00Z",
    recommendation_generated_at: "2026-08-23T12:01:00Z",
    target_event: 7,
    model: "gpt-5.6-sol",
    reasoning_effort: "xhigh",
    review: reviewFixture(),
    research: {
      performed: true,
      sources: [{ title: "Premier League", url: "https://www.premierleague.com/news/123" }],
    },
  };
  assert.deepEqual(parseAiReviewResponse(response, 1, 2), response);

  const unsafe = structuredClone(response);
  unsafe.research.sources[0].url = "https://attacker.example/fake-news";
  assert.throws(() => parseAiReviewResponse(unsafe, 1), /allowed HTTPS source/);

  const nonDefaultPort = structuredClone(response);
  nonDefaultPort.research.sources[0].url = "https://www.premierleague.com:444/news/123";
  assert.throws(() => parseAiReviewResponse(nonDefaultPort, 1), /allowed HTTPS source/);

  const withoutEvidence = structuredClone(response);
  withoutEvidence.research.sources = [];
  assert.throws(() => parseAiReviewResponse(withoutEvidence, 1), /at least one allowed research source/);
});

test("rank is coarsened before it can enter the OpenAI context", () => {
  assert.equal(managerRankBand(1), "top_10k");
  assert.equal(managerRankBand(10_001), "top_100k");
  assert.equal(managerRankBand(900_000), "top_1m");
  assert.equal(managerRankBand(2_000_000), "outside_1m");
  assert.equal(managerRankBand(null), "unknown");
});
