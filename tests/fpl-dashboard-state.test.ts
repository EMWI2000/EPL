import assert from "node:assert/strict";
import test from "node:test";

import { canSubmitWeeklyPlanner } from "../components/fpl-dashboard-state.ts";

const ready = {
  hasManagerSync: true,
  squadConfirmed: true,
  hasFreeHitWarning: false,
  isSyncing: false,
  syncStateValid: true,
  hasSyncError: false,
};

test("weekly planner fails closed after a resync failure", () => {
  assert.equal(canSubmitWeeklyPlanner(ready), true);
  assert.equal(canSubmitWeeklyPlanner({ ...ready, syncStateValid: false }), false);
  assert.equal(canSubmitWeeklyPlanner({ ...ready, hasSyncError: true }), false);
  assert.equal(canSubmitWeeklyPlanner({ ...ready, isSyncing: true }), false);
});
