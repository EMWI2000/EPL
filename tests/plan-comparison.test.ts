import assert from "node:assert/strict";
import test from "node:test";
import { actionFinances, netPointsOver, planOptions } from "../lib/plan-comparison.ts";
import type { PlannerAction, PlannerPayload } from "../lib/planner-contract.ts";

const plan = (points: number[], hit = 0) => ({
  hit_points: hit,
  gameweeks: points.map((projected_points, index) => ({ gameweek: 10 + index, projected_points })),
}) as PlannerAction;

test("horizon comparisons subtract one initial hit and do not substitute a shorter horizon", () => {
  const roll = plan([50, 50, 50, 50, 50]);
  const buy = plan([52, 54, 53, 52, 51], 4);
  assert.equal(netPointsOver(buy, roll, 1), -2);
  assert.equal(netPointsOver(buy, roll, 3), 5);
  assert.equal(netPointsOver(buy, roll, 5), 8);
  assert.equal(netPointsOver(plan([51, 52]), roll, 3), null);
  assert.equal(netPointsOver(buy, undefined, 1), null);
  const wrongRound = structuredClone(roll);
  wrongRound.gameweeks[1].gameweek = 20;
  assert.equal(netPointsOver(buy, wrongRound, 3), null);
});

test("separate sales funding the same purchase remain separate selectable plans", () => {
  const first = { ...plan([50]), transfers: [{ out_id: 1, in_id: 3 }] } as PlannerAction;
  const second = { ...plan([51]), transfers: [{ out_id: 2, in_id: 3 }] } as PlannerAction;
  const options = planOptions({ best_action: first, alternatives: [second] } as PlannerPayload);
  assert.equal(options.length, 2);
  assert.deepEqual(options[1].choice, { kind: "alternative", alternative_index: 0 });
  assert.equal(options[1].action.transfers[0].out_id, 2);
});

test("funding totals use selling price rather than market or purchase price", () => {
  const action = { transfers: [
    { out_selling_price_tenths: 55, out_current_price_tenths: 60, in_price_tenths: 70 },
    { out_selling_price_tenths: 65, out_current_price_tenths: 70, in_price_tenths: 50 },
  ], bank_after_tenths: 7 } as PlannerAction;
  assert.deepEqual(actionFinances(action), { sold: 120, bought: 120, bank: 7 });
});
