import assert from "node:assert/strict";
import test from "node:test";
import { buildChipPolicy, type ChipPolicyInput } from "../lib/chip-policy.ts";
import type { ChipName, ChipState, PlannerChipStrategy } from "../lib/planner-contract.ts";
import type { ChipSequence, ChipSequenceId } from "../lib/chip-sequences.ts";

const available = (): ChipState => ({ wildcard: "available", freehit: "available", bboost: "available", "3xc": "available" });
const used = (): ChipState => ({ wildcard: "used", freehit: "used", bboost: "used", "3xc": "used" });
function strategy(options: { target?: number; gain?: number; changes?: number; boostGain?: number } = {}): PlannerChipStrategy {
  const target = options.target ?? 10;
  const horizon = Math.min(8, 39 - target);
  const sequence = (id: ChipSequenceId, points: number): ChipSequence => ({
    sequence_id: id, label: id, weighted_net_points: points, gain_vs_normal_points: points - 300, total_hit_points: 0,
    actions: Array.from({ length: horizon }, (_, index) => ({
      event: target + index,
      chip: index === 0 && id.startsWith("wildcard") ? "wildcard"
        : index === 1 && id.endsWith("bboost") ? "bboost" : null,
      transfer_out_ids: [], transfer_in_ids: [], squad_ids: Array.from({ length: 15 }, (_, i) => i + 1),
      bank_after_tenths: 10, free_transfers_before: Math.min(5, index + 1),
      free_transfers_next_gameweek: Math.min(5, index + 2), hit_points: 0,
    })),
  });
  return {
    horizon, target_event: target,
    inventory: (["wildcard", "freehit", "bboost", "3xc"] as ChipName[])
      .map(chip => ({ chip, used_events: [], available_for_target: true })),
    scenarios: [{ scenario_id: `wildcard-gw${target}`, chip: "wildcard", event: target,
      signal: "hold", available: true, estimated_gain_points: options.gain ?? 3,
      baseline_points: 300, chip_points: 300 + (options.gain ?? 3), confidence: "low", model_scope: "multiweek_rebuild",
      reason: "Test", squad: [], change_count: options.changes ?? 7, bank_after_tenths: 1 }],
    recommendation: { action: "hold", scenario_id: null, chip: null, event: null, reason: "Test" },
    model_scope: "bounded_chip_counterfactuals", globally_optimal: false, recalculate_each_deadline: true,
    sequence_comparison: {
      status: "ready", horizon, model_scope: "paired_bounded_chip_sequences", globally_optimal: false,
      recalculate_each_deadline: true, original_roadmap_weighted_net_points: 300,
      highest_projected_sequence_id: "wildcard-bboost", assumptions: [], reason: "Test",
      sequences: [sequence("normal", 300), sequence("wildcard", 300 + (options.gain ?? 3)),
        sequence("normal-bboost", 315), sequence("wildcard-bboost", 315 + (options.boostGain ?? 8))],
    },
  };
}
function policy(overrides: Partial<ChipPolicyInput> = {}) {
  return buildChipPolicy({ targetEvent: 10, chips: available(), chipUsage: [], strategy: null, evidenceCurrent: false, ...overrides });
}

test("calendar policy works before a solver response and leaves benefit unknown", () => {
  const p = policy();
  assert.equal(p.expiryEvent, 19);
  assert.equal(p.remainingDeadlines, 10);
  assert.equal(p.unusedCount, 4);
  assert.equal(p.slack, 6);
  assert.equal(p.status, "review");
  assert.equal(p.wildcard.reviewEvent, 13);
  assert.equal(p.wildcard.latestPlanningEvent, 16);
  assert.equal(p.wildcard.promptEvent, 15);
  assert.equal(p.wildcard.gainVsNormal, null);
  assert.equal(p.wildcard.evidence, "unknown");
});

test("four unused chips at GW16 require a decision now, even when numeric gain is low", () => {
  const p = policy({ targetEvent: 16, strategy: strategy({ target: 16 }), evidenceCurrent: true });
  assert.equal(p.remainingDeadlines, 4);
  assert.equal(p.slack, 0);
  assert.equal(p.urgent, true);
  assert.equal(p.allCanStillBeUsed, true);
  assert.equal(p.status, "urgent");
  assert.equal(p.wildcard.status, "urgent");
  assert.match(p.headline, /kræves/);
});

test("fewer slots than chips explicitly says that using them all is impossible", () => {
  const p = policy({ targetEvent: 18 });
  assert.equal(p.remainingDeadlines, 2);
  assert.equal(p.unusedCount, 4);
  assert.equal(p.slack, -2);
  assert.equal(p.allCanStillBeUsed, false);
  assert.equal(p.status, "urgent");
  assert.match(p.headline, /kan ikke nås/);
});

test("GW19 includes its own deadline and preserves one slot for BB after WC when needed", () => {
  const last = policy({ targetEvent: 19, chips: { ...used(), wildcard: "available" } });
  assert.equal(last.remainingDeadlines, 1);
  assert.equal(last.wildcard.latestPlanningEvent, 19);
  assert.equal(last.wildcard.promptEvent, 18);
  assert.equal(last.status, "urgent");
  const pair = policy({ targetEvent: 19, chips: { ...used(), wildcard: "available", bboost: "available" } });
  assert.equal(pair.wildcard.latestPlanningEvent, 18);
  assert.equal(pair.allCanStillBeUsed, false);
  assert.ok(pair.reasons.some(reason => reason.includes("konkurrerer om sidste deadline")));
  assert.ok(!pair.reasons.some(reason => reason.includes("planlæg Wildcard først")));
});

test("GW20 resets used halves and counts temporarily unavailable Free Hit as unused", () => {
  const p = policy({ targetEvent: 20, chips: { ...available(), freehit: "unavailable" },
    chipUsage: [{ name: "wildcard", event: 4 }, { name: "bboost", event: 14 }, { name: "3xc", event: 16 }, { name: "freehit", event: 19 }] });
  assert.equal(p.half, 2);
  assert.equal(p.expiryEvent, 38);
  assert.equal(p.remainingDeadlines, 19);
  assert.equal(p.unusedCount, 4);
  assert.ok(p.unusedChips.includes("freehit"));
  assert.ok(!p.availableNow.includes("freehit"));
  assert.ok(p.availableNow.includes("wildcard"));
  assert.equal(p.wildcard.reviewEvent, null);
  assert.equal(p.wildcard.latestPlanningEvent, 35);
});

test("Free Hit after GW19 stays unavailable in GW20 even if supplied availability is inconsistent", () => {
  const p = policy({ targetEvent: 20, chipUsage: [{ name: "freehit", event: 19 }] });
  assert.equal(p.unusedCount, 4);
  assert.ok(!p.availableNow.includes("freehit"));
});

test("only current-half usage removes chips from the budget", () => {
  const p = policy({ targetEvent: 24, chipUsage: [{ name: "wildcard", event: 5 }, { name: "bboost", event: 22 }] });
  assert.equal(p.unusedCount, 3);
  assert.ok(p.unusedChips.includes("wildcard"));
  assert.ok(!p.unusedChips.includes("bboost"));
});

test("late-season policy remains visible when the long-range model is absent", () => {
  for (const targetEvent of [33, 34, 35, 36, 37, 38]) {
    const p = policy({ targetEvent });
    assert.equal(p.remainingDeadlines, 39 - targetEvent);
    assert.equal(p.wildcard.gainVsNormal, null);
    assert.equal(p.urgent, targetEvent >= 35);
  }
  const last = policy({ targetEvent: 38, chips: { ...used(), bboost: "available" } });
  assert.equal(last.status, "urgent");
  assert.equal(last.wildcard.status, "used");
  assert.deepEqual(last.availableNow, ["bboost"]);
});

test("fully used inventory has no false expiry alarm", () => {
  const p = policy({ targetEvent: 38, chips: used() });
  assert.equal(p.status, "used");
  assert.equal(p.urgent, false);
  assert.equal(p.unusedCount, 0);
  assert.equal(p.wildcard.latestPlanningEvent, null);
});

test("an unavailable or used WC is never considered even if a stale scenario claims a large gain", () => {
  for (const state of ["used", "unavailable"] as const) {
    const p = policy({ chips: { ...available(), wildcard: state }, strategy: strategy({ gain: 100 }), evidenceCurrent: true });
    assert.equal(p.wildcard.availableNow, false);
    assert.notEqual(p.wildcard.status, "consider");
    assert.equal(p.wildcard.gainVsNormal, null);
  }
  const urgent = policy({ targetEvent: 19, chips: { ...used(), wildcard: "unavailable" } });
  assert.equal(urgent.status, "urgent");
  assert.equal(urgent.wildcard.status, "hold");
  assert.ok(!urgent.availableNow.includes("wildcard"));
});

test("stale evidence or a mismatching model target cannot supply live gains", () => {
  for (const overrides of [
    { targetEvent: 11, strategy: strategy(), evidenceCurrent: true },
    { strategy: strategy({ gain: 100 }), evidenceCurrent: false },
  ]) {
    const p = policy(overrides);
    assert.equal(p.wildcard.evidence, "stale");
    assert.equal(p.wildcard.gainVsNormal, null);
    assert.equal(p.wildcard.changeCount, null);
    assert.notEqual(p.status, "consider");
  }
});

test("WC consideration requires both four changes and fifteen comparable weighted points", () => {
  assert.equal(policy({ strategy: strategy({ gain: 15, changes: 4 }), evidenceCurrent: true }).status, "consider");
  assert.equal(policy({ strategy: strategy({ gain: 14.999, changes: 4 }), evidenceCurrent: true }).status, "hold");
  assert.equal(policy({ strategy: strategy({ gain: 30, changes: 3 }), evidenceCurrent: true }).status, "hold");
});

test("the WC plus BB calculation subtracts a baseline that also spends BB", () => {
  const p = policy({ strategy: strategy({ gain: 3, boostGain: 8.403 }), evidenceCurrent: true });
  assert.equal(p.wildcard.gainVsNormal, 3);
  assert.equal(p.wildcard.gainWithBenchBoost, 8.403);
  assert.equal(p.status, "hold", "A combined 23.403 points vs no chips must not be called WC value");
  assert.equal(policy({ strategy: strategy({ gain: 3, boostGain: 15, changes: 4 }), evidenceCurrent: true }).status, "consider");
});

test("missing or incomplete paired paths stay unknown instead of using a static scenario gain", () => {
  const s = strategy({ gain: 100, boostGain: 100 });
  s.sequence_comparison!.sequences = s.sequence_comparison!.sequences.filter(path => path.sequence_id !== "normal" && path.sequence_id !== "normal-bboost");
  const p = policy({ strategy: s, evidenceCurrent: true });
  assert.equal(p.wildcard.gainVsNormal, null);
  assert.equal(p.wildcard.gainWithBenchBoost, null);
  assert.equal(p.wildcard.evidence, "unavailable");
  assert.notEqual(p.status, "consider");
});

test("a WC path with an extra chip cannot be compared against no-chip normal transfers", () => {
  const s = strategy({ gain: 100, boostGain: 2 });
  s.sequence_comparison!.sequences.find(path => path.sequence_id === "wildcard")!.actions[2].chip = "bboost";
  const p = policy({ strategy: s, evidenceCurrent: true });
  assert.equal(p.wildcard.gainVsNormal, null);
  assert.equal(p.wildcard.gainWithBenchBoost, 2);
  assert.equal(p.status, "hold");
});

test("BB in the next chip half cannot justify spending the current WC by a paired comparison", () => {
  const s = strategy({ target: 18, gain: 3, boostGain: 100 });
  for (const path of s.sequence_comparison!.sequences.filter(path => path.sequence_id.endsWith("bboost"))) {
    path.actions[1].chip = null;
    path.actions[2].chip = "bboost";
  }
  const p = policy({ targetEvent: 18, strategy: s, evidenceCurrent: true });
  assert.equal(p.wildcard.gainVsNormal, 3);
  assert.equal(p.wildcard.gainWithBenchBoost, null);
});

test("the personal checkpoint and one-round-early planning prompt do not assert an optimized date", () => {
  const p = policy({ targetEvent: 13, strategy: strategy({ target: 13 }), evidenceCurrent: true });
  assert.equal(p.status, "review");
  assert.equal(p.wildcard.reviewEvent, 13);
  const prompt = policy({ targetEvent: 34, strategy: null });
  assert.equal(prompt.wildcard.promptEvent, 34);
  assert.equal(prompt.wildcard.latestPlanningEvent, 35);
  assert.ok(prompt.reasons.some(reason => reason.includes("Fastlæg Wildcard-planen nu")));
});

test("criteria remain guidance and building the policy does not mutate state or model evidence", () => {
  const input: ChipPolicyInput = { targetEvent: 10, chips: available(), chipUsage: [], strategy: strategy(), evidenceCurrent: true };
  const original = structuredClone(input);
  const p = buildChipPolicy(input);
  assert.deepEqual(input, original);
  assert.match(p.criteria.find(rule => rule.chip === "bboost")!.rule, /12.*4 reserver/);
  assert.match(p.criteria.find(rule => rule.chip === "3xc")!.rule, /8 ekstra.*middel/);
  assert.match(p.criteria.find(rule => rule.chip === "freehit")!.rule, /manuel vurdering/);
  p.criteria[0].rule = "changed";
  assert.notEqual(buildChipPolicy(input).criteria[0].rule, "changed");
});
