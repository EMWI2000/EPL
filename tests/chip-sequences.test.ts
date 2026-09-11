import assert from "node:assert/strict";
import test from "node:test";
import { CHIP_SEQUENCE_ASSUMPTIONS, parseChipSequences, type ChipSequence, type ChipSequenceAction, type ChipSequenceComparison, type ChipSequenceContext } from "../lib/chip-sequences.ts";
import type { PlannerPlayerReference, PlannerStrategyPlan } from "../lib/planner-contract.ts";

function setup(target = 4): { context: ChipSequenceContext; value: ChipSequenceComparison } {
  const initial = Array.from({ length: 15 }, (_, i) => i + 1);
  const players = new Map<number, PlannerPlayerReference>(initial.map((id) => [id, {
    id, name: `Player ${id}`, team: `T${Math.floor((id - 1) / 3)}`, position: id <= 2 ? "GKP" : id <= 7 ? "DEF" : id <= 12 ? "MID" : "FWD",
    price_tenths: id === 1 ? 60 : 50, status: "a", price_signal: null,
  }]));
  players.set(16, { ...players.get(1)!, id: 16, name: "New keeper", team: "OTHER" });
  const path = (wildcard: boolean): ChipSequenceAction[] => {
    let ft = 1;
    return Array.from({ length: 6 }, (_, index) => {
      const next = wildcard && index === 0 ? ft : Math.min(5, ft + 1);
      const action: ChipSequenceAction = { event: target + index, chip: wildcard && index === 0 ? "wildcard" : null,
        transfer_out_ids: wildcard && index === 0 ? [1] : [], transfer_in_ids: wildcard && index === 0 ? [16] : [],
        squad_ids: wildcard ? initial.filter(id => id !== 1).concat(16).sort((a, b) => a - b) : initial,
        bank_after_tenths: wildcard ? 15 : 20, free_transfers_before: ft, free_transfers_next_gameweek: next, hit_points: 0 };
      ft = next;
      return action;
    });
  };
  const normal: ChipSequence = { sequence_id: "normal", label: "Normal", weighted_net_points: 100, gain_vs_normal_points: 0, total_hit_points: 0, actions: path(false) };
  const wildcard: ChipSequence = { sequence_id: "wildcard", label: "Wildcard", weighted_net_points: 110, gain_vs_normal_points: 10, total_hit_points: 0, actions: path(true) };
  const normalBb: ChipSequence = { ...structuredClone(normal), sequence_id: "normal-bboost", weighted_net_points: 105, gain_vs_normal_points: 5 };
  const wildcardBb: ChipSequence = { ...structuredClone(wildcard), sequence_id: "wildcard-bboost", weighted_net_points: 116, gain_vs_normal_points: 16 };
  normalBb.actions[1].chip = "bboost";
  wildcardBb.actions[1].chip = "bboost";
  const strategy = {
    horizon: 6, gameweek_window: Array.from({ length: 6 }, (_, i) => target + i), gw_weights: [1, 1, 1, 1, 1, 1],
    weighted_projected_points: 104, weighted_hit_cost_points: 2,
    steps: [{ transfers: [], squad_ids: initial, bank_after_tenths: 20, free_transfers_before: 1, free_transfers_next_gameweek: 2, hit_points: 0 }],
  } as unknown as PlannerStrategyPlan;
  return {
    context: { strategy, players, squad: new Map(initial.map(id => [id, players.get(id)!])),
      selling: new Map(initial.map(id => [id, id === 1 ? 55 : 50])), bank: 20,
      inventory: ["wildcard", "bboost", "freehit", "3xc"].map(chip => ({ chip: chip as "wildcard", used_events: [], available_for_target: true })) },
    value: { status: "ready", horizon: 6, model_scope: "paired_bounded_chip_sequences", globally_optimal: false,
      recalculate_each_deadline: true, original_roadmap_weighted_net_points: 102,
      highest_projected_sequence_id: "wildcard-bboost", sequences: [normal, normalBb, wildcard, wildcardBb],
      assumptions: [...CHIP_SEQUENCE_ASSUMPTIONS], reason: "Bounded same-policy comparison" },
  };
}

test("chip sequences accept complete same-policy paths and real selling prices", () => {
  const { value, context } = setup();
  assert.deepEqual(parseChipSequences(value, context), value);
  assert.equal(value.sequences[2].actions[0].bank_after_tenths, 15, "original sale55 not market60 finances the WC");
  assert.equal(value.sequences[2].actions[0].free_transfers_next_gameweek, 1, "Wildcard preserves bank without adding a transfer");
});

test("chip sequences fail closed for unavailable output or mismatched score bases", () => {
  const { value, context } = setup();
  const unavailable = { ...value, status: "unavailable", sequences: [], highest_projected_sequence_id: null };
  assert.equal(parseChipSequences(unavailable, context).status, "unavailable");
  assert.throws(() => parseChipSequences({ ...unavailable, sequences: [value.sequences[0]] }, context));
  assert.throws(() => parseChipSequences({ ...value, original_roadmap_weighted_net_points: 104 }, context));
  assert.throws(() => parseChipSequences({ ...value, globally_optimal: true }, context));
  assert.throws(() => parseChipSequences({ ...value, horizon: 8 }, context));
  assert.throws(() => parseChipSequences({ ...value, assumptions: [] }, context));
});

test("chip sequences reject incorrect gains, claimed winner and duplicate IDs", () => {
  const { value, context } = setup();
  const bad = structuredClone(value);
  bad.sequences[2].gain_vs_normal_points = 99;
  assert.throws(() => parseChipSequences(bad, context));
  assert.throws(() => parseChipSequences({ ...value, highest_projected_sequence_id: "normal" }, context));
  assert.throws(() => parseChipSequences({ ...value, sequences: [value.sequences[0], value.sequences[0]] }, context));
  assert.throws(() => parseChipSequences({ ...value, sequences: [value.sequences[3]] }, context));
});

test("chip sequences reject selling inflation, wrong free transfers and hidden hits", () => {
  for (const change of [
    (action: ChipSequenceAction) => { action.bank_after_tenths = 20; },
    (action: ChipSequenceAction) => { action.free_transfers_next_gameweek = 2; },
    (action: ChipSequenceAction) => { action.hit_points = 4; },
  ]) {
    const { value, context } = setup();
    change(value.sequences[2].actions[0]);
    assert.throws(() => parseChipSequences(value, context));
  }
  const { value, context } = setup();
  value.sequences[0].total_hit_points = 4;
  assert.throws(() => parseChipSequences(value, context));
});

test("chip sequences reject non-owned sales, wrong positions, duplicate and over-quota squads", () => {
  for (const change of [
    (action: ChipSequenceAction) => { action.transfer_out_ids = [99]; },
    (action: ChipSequenceAction) => { action.transfer_in_ids = [3]; },
    (action: ChipSequenceAction) => { action.squad_ids[0] = action.squad_ids[1]; },
    (action: ChipSequenceAction) => { action.squad_ids.pop(); },
  ]) {
    const { value, context } = setup();
    change(value.sequences[2].actions[0]);
    assert.throws(() => parseChipSequences(value, context));
  }
  const { value, context } = setup();
  const refs = new Map(context.players);
  refs.set(16, { ...refs.get(16)!, position: "FWD" });
  assert.throws(() => parseChipSequences(value, { ...context, players: refs }));
  refs.set(16, { ...refs.get(16)!, position: "GKP", team: "T2" });
  assert.throws(() => parseChipSequences(value, { ...context, players: refs }));
});

test("chip sequences preserve FT0 on Wildcard and one free transfer on a normal roll", () => {
  const { value, context } = setup();
  const selected = value.sequences.filter(sequence => !sequence.sequence_id.endsWith("-bboost"));
  for (const sequence of selected) {
    let ft = 0;
    for (const action of sequence.actions) {
      action.free_transfers_before = ft;
      ft = action.chip === "wildcard" ? ft : Math.min(5, ft + 1);
      action.free_transfers_next_gameweek = ft;
    }
  }
  context.strategy.steps[0].free_transfers_before = 0;
  context.strategy.steps[0].free_transfers_next_gameweek = 1;
  assert.doesNotThrow(() => parseChipSequences({ ...value, sequences: selected, highest_projected_sequence_id: "wildcard" }, context));
});

test("chip sequences cannot reuse a chip, combine it on one GW, or cross half-season expiry", () => {
  const { value, context } = setup();
  const unavailable = { ...context, inventory: context.inventory.map(row => row.chip === "wildcard" ? { ...row, available_for_target: false } : row) };
  assert.throws(() => parseChipSequences(value, unavailable));
  const used = { ...context, inventory: context.inventory.map(row => row.chip === "bboost" ? { ...row, used_events: [1] } : row) };
  assert.throws(() => parseChipSequences(value, used));
  const bad = structuredClone(value);
  bad.sequences[3].actions[0].chip = "bboost";
  assert.throws(() => parseChipSequences(bad, context));
  const rollover = setup(19);
  assert.throws(() => parseChipSequences(rollover.value, rollover.context), "BB GW20 cannot be treated as use of the GW19 chip set");
  const halfTwo = setup(20);
  halfTwo.context.inventory = halfTwo.context.inventory.map(row => ({ ...row, used_events: [1] }));
  assert.doesNotThrow(() => parseChipSequences(halfTwo.value, halfTwo.context));
});

test("chip sequences reject incomplete paths, later Wildcards and changed BB transfer paths", () => {
  const { value, context } = setup();
  const short = structuredClone(value); short.sequences[0].actions.pop();
  assert.throws(() => parseChipSequences(short, context));
  const later = structuredClone(value); later.sequences[2].actions[0].chip = null; later.sequences[2].actions[1].chip = "wildcard";
  assert.throws(() => parseChipSequences(later, context));
  const changed = structuredClone(value); changed.sequences[1].actions[0].squad_ids.reverse();
  assert.throws(() => parseChipSequences(changed, context), "BB path is exactly its saved normal path except for the chip");
  const wrongEvent = structuredClone(value); wrongEvent.sequences[0].actions[2].event = 20;
  assert.throws(() => parseChipSequences(wrongEvent, context));
});

test("chip sequence transfers after deadline four are forbidden", () => {
  const { value, context } = setup();
  value.sequences[0].actions[4].transfer_out_ids = [1];
  value.sequences[0].actions[4].transfer_in_ids = [16];
  assert.throws(() => parseChipSequences(value, context));
});

test("chip sequence first normal action must match the existing roadmap", () => {
  const { value, context } = setup();
  context.strategy.steps[0].squad_ids = context.strategy.steps[0].squad_ids.filter(id => id !== 1).concat(16);
  assert.throws(() => parseChipSequences(value, context));
});
