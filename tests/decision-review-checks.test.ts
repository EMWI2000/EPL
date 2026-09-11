import assert from "node:assert/strict";
import test from "node:test";
import type { AiReviewRequest, AiReviewSquadPlayerContext } from "../lib/ai-review-contract.ts";
import type { EnrichedTransfer, PlannerAction, PlannerOwnedPlayerReference } from "../lib/planner-contract.ts";
import { buildDecisionReviewChecks } from "../lib/decision-review-checks.ts";
import { buildOpenAiReviewContext } from "../lib/openai-ai-review.ts";

function player(id: number): PlannerOwnedPlayerReference {
  return {
    id, name: `Player ${id}`, team: `Club ${(id % 5) + 1}`,
    position: id <= 2 ? "GKP" : id <= 7 ? "DEF" : id <= 12 || id >= 100 ? "MID" : "FWD",
    price_tenths: 50, purchase_price_tenths: 50, selling_price_tenths: 50,
    status: "a", price_signal: null,
  };
}

function transfer(outId = 8, inId = 100): EnrichedTransfer {
  return {
    out_id: outId, in_id: inId, position: "MID", out: player(outId), in: player(inId),
    out_purchase_price_tenths: 50, out_current_price_tenths: 50,
    out_selling_price_tenths: 50, in_price_tenths: 50,
  };
}

function action(netEp: number, decisionValue: number, nextGameweekPoints: number, transfers: EnrichedTransfer[] = []): PlannerAction {
  return {
    kind: transfers.length ? "transfer" : "roll", transfer_count: transfers.length, transfers,
    squad_ids: Array.from({ length: 15 }, (_, index) => {
      const id = index + 1;
      return transfers.find(t => t.out_id === id)?.in_id ?? id;
    }),
    gameweeks: [{
      gameweek: 4, starting_ids: [1, 3, 4, 5, 9, 10, 11, transfers[0]?.in_id ?? 8, 13, 14, 15],
      captain_id: transfers[0]?.in_id ?? 8, formation: "3-4-3", projected_points: nextGameweekPoints,
    }],
    bank_before_tenths: 17, bank_after_tenths: 17, free_transfers_before: 1,
    free_transfers_next_gameweek: transfers.length ? 1 : 2, hit_points: 0,
    projected_points: 200 + netEp, banked_ft_value_points: 1,
    decision_value_points: 200 + decisionValue, net_points_vs_roll: netEp,
    decision_value_vs_roll: decisionValue, explanation: "Test calculation",
  };
}

function requestFixture(): AiReviewRequest {
  const best = action(3.8, 3, 55, [transfer()]);
  const squad: AiReviewSquadPlayerContext[] = best.squad_ids.map(id => ({
    ...player(id), current_price_tenths: 50, weighted_expected_points: 20,
    projections: [{
      gameweek: 4, expected_points: 6, expected_minutes: 80,
      appearance_probability: 0.95, sixty_probability: 0.8, confidence: 0.7,
      reliability: "medium", fixtures_count: 1, is_blank: false, is_dgw: false,
    }],
  }));
  return {
    schema_version: "fpl-ai-review-request-v1", manager_id: 8425806,
    recommendation_generated_at: "2026-09-11T12:00:00Z",
    target_deadline: "2026-09-12T12:30:00Z", state_observed_at: "2026-09-11T11:59:00Z",
    manager_rank_band: "outside_1m", state_limitations: ["state_is_locked_at_last_public_deadline"],
    forecast: {
      version: "v2", horizon: 1, include_doubtful: false, validation_status: "unvalidated",
      price_signals_available: false, next_price_deadline: null,
    },
    planner: {
      manager_id: 8425806, state_fingerprint: "a".repeat(64), source_event: 3, target_event: 4,
      confirmed_state: {
        bank_tenths: 17, free_transfers: 1, no_active_chip_confirmed: true,
        squad: Array.from({ length: 15 }, (_, index) => player(index + 1)),
      },
      best_action: best, alternatives: [action(0, 0, 52), action(4.6, 2.5, 57, [transfer(8, 101)])],
      sequential: null, strategy: null, chip_strategy: null,
      method: {
        candidate_count: 10, plans_per_transfer_count: 5, higher_transfer_count_plans: 1,
        maximum_immediate_transfers: 1, roll_ft_value_points: 1, chips_modelled: false,
        bounded_roadmap_modelled: false, next_deadline_transfer_modelled: false, future_transfers_modelled: false,
      },
    },
    lineup: {
      gameweek: 4, formation: "3-4-3", starting_ids: best.gameweeks[0].starting_ids,
      bench_ids: [2, 6, 7, 12], captain_id: 100, vice_captain_id: 14,
    },
    squad_context: squad,
  };
}

test("compares the proposed team plan with roll and each numbered alternative", () => {
  const checks = buildDecisionReviewChecks(requestFixture());
  assert.equal(checks.best_net_ep_vs_roll, 3.8);
  assert.deepEqual(checks.comparisons, [
    { alternative_index: 0, best_minus_alternative_net_ep: 3.8, best_minus_alternative_decision_value: 3, next_gameweek_points_difference: 3 },
    { alternative_index: 1, best_minus_alternative_net_ep: -0.8, best_minus_alternative_decision_value: 0.5, next_gameweek_points_difference: -2 },
  ]);
  assert.match(checks.interpretation, /entire weighted team plan/);
  assert.match(checks.interpretation, /No counterfactual minute simulation was run/);
});

test("selects the highest net-EP challenger independently of saved-transfer decision value", () => {
  const request = requestFixture();
  request.planner.alternatives = [action(2, 5, 60), action(4, 1, 54)];
  const checks = buildDecisionReviewChecks(request);
  assert.equal(checks.strongest_points_alternative_index, 1);
  assert.equal(checks.comparisons[0].best_minus_alternative_decision_value, -2);
  assert.equal(checks.comparisons[1].best_minus_alternative_net_ep, -0.2);
});

test("uses the original alternative index to resolve exact EP ties deterministically without mutation", () => {
  const request = requestFixture();
  request.planner.alternatives = [action(4, 1, 54), action(4, 5, 60)];
  const before = structuredClone(request);
  assert.equal(buildDecisionReviewChecks(request).strongest_points_alternative_index, 0);
  assert.deepEqual(request, before);
});

test("does not turn small but distinct raw EP values into a false challenger tie", () => {
  const request = requestFixture();
  request.planner.alternatives = [action(3.8001, 2, 55), action(3.8004, 2, 55)];
  const checks = buildDecisionReviewChecks(request);
  assert.equal(checks.strongest_points_alternative_index, 1);
  assert.ok(checks.comparisons.every(c => c.best_minus_alternative_net_ep === 0));
});

test("returns no challenger when the solver provides no alternatives", () => {
  const request = requestFixture();
  request.planner.alternatives = [];
  const checks = buildDecisionReviewChecks(request);
  assert.equal(checks.strongest_points_alternative_index, null);
  assert.deepEqual(checks.comparisons, []);
});

test("keeps the confirmed current squad separate from proposed POST-transfer ownership", () => {
  const request = requestFixture();
  const checks = buildDecisionReviewChecks(request);
  assert.equal(checks.current_owned_players.length, 15);
  assert.ok(checks.current_owned_players.some(p => p.player === "Player 8"));
  assert.ok(!checks.current_owned_players.some(p => p.player === "Player 100"));
  const context = buildOpenAiReviewContext(request);
  assert.ok(context.squad_outlook.some(p => p.player === "Player 100"));
  assert.ok(!context.squad_outlook.some(p => p.player === "Player 8"));
  assert.equal(context.next_gameweek_lineup.captain, "Player 100");
  assert.match(checks.ownership_scope, /confirmed current squad/);
  assert.match(checks.ownership_scope, /POST-transfer squad/);
  assert.ok(!JSON.stringify(checks).includes("8425806"));
  assert.ok(!JSON.stringify(checks).includes('"id":'));
});

test("marks absent outgoing and strongest-alternative profiles unknown and deduplicates by player ID", () => {
  const request = requestFixture();
  request.planner.alternatives.push(action(1, 1, 54, [transfer(9, 102)]));
  const checks = buildDecisionReviewChecks(request);
  assert.deepEqual(checks.missing_individual_forecasts, [
    { player: "Player 8", club: player(8).team },
    { player: "Player 101", club: player(101).team },
  ]);
  assert.match(checks.ownership_scope, /unknown, not zero/);
  assert.ok(checks.missing_individual_forecasts.every(p => !("expected_points" in p)));
});

test("retains low captain reliability, unknown minutes and a genuine zero appearance probability", () => {
  const request = requestFixture();
  const captain = request.squad_context.find(p => p.id === 100)!;
  captain.status = "d";
  captain.projections[0] = { ...captain.projections[0], expected_minutes: null, appearance_probability: 0, reliability: "low" };
  assert.deepEqual(buildDecisionReviewChecks(request).captain_check, {
    player: "Player 100", club: captain.team, status: "d",
    expected_points: 6, expected_minutes: null, appearance_probability: 0, reliability: "low",
  });
});

test("does not invent captain evidence when the player or target-gameweek projection is missing", () => {
  const request = requestFixture();
  request.lineup.captain_id = 999;
  assert.equal(buildDecisionReviewChecks(request).captain_check, null);
  request.lineup.captain_id = 100;
  request.squad_context.find(p => p.id === 100)!.projections[0].gameweek = 5;
  assert.equal(buildDecisionReviewChecks(request).captain_check, null);
});

test("sanitizes bounded display names without leaking player metadata into arithmetic checks", () => {
  const request = requestFixture();
  request.planner.confirmed_state.squad[0].name = `  Keeper\u0000\n ${"x".repeat(100)}  `;
  const checks = buildDecisionReviewChecks(request);
  assert.equal(checks.current_owned_players[0].player.length, 80);
  assert.ok(checks.current_owned_players[0].player.startsWith("Keeper x"));
  assert.doesNotMatch(checks.current_owned_players[0].player, /[\u0000-\u001f\u007f]/);
  assert.deepEqual(Object.keys(checks.current_owned_players[0]).sort(), ["club", "player", "position"]);
});
