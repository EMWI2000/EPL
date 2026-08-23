import assert from "node:assert/strict";
import test from "node:test";

import {
  configuredFplManagerId,
  hasConfiguredFplManagerMismatch,
  resolveInitialFplManagerId,
} from "../lib/fpl-manager-config.ts";

test("parses an optional configured FPL manager id", () => {
  assert.equal(configuredFplManagerId(undefined), null);
  assert.equal(configuredFplManagerId(""), null);
  assert.equal(configuredFplManagerId(" 1234567 "), 1234567);
});

test("rejects malformed or unsafe configured FPL manager ids", () => {
  for (const value of [
    "0",
    "01",
    "-1",
    "1.5",
    "entry/123",
    "00000000-0000-4000-8000-000000000000",
    "9007199254740992",
  ]) {
    assert.throws(() => configuredFplManagerId(value), /positive integer|safe positive integer/);
  }
});

test("can require the manager id in a hosted environment", () => {
  assert.throws(
    () => configuredFplManagerId(undefined, { required: true }),
    /must be configured in Vercel/,
  );
});

test("detects only an explicit mismatching manager id", () => {
  assert.equal(hasConfiguredFplManagerMismatch({ manager_id: 123 }, 123), false);
  assert.equal(hasConfiguredFplManagerMismatch({ horizon: 5 }, 123), false);
  assert.equal(hasConfiguredFplManagerMismatch({ manager_id: 456 }, 123), true);
  assert.equal(hasConfiguredFplManagerMismatch({ manager_id: "123" }, 123), true);
  assert.equal(hasConfiguredFplManagerMismatch(null, 123), false);
  assert.equal(
    hasConfiguredFplManagerMismatch({ horizon: 5 }, 123, { required: true }),
    true,
  );
  assert.equal(
    hasConfiguredFplManagerMismatch({ manager_id: 123 }, 123, { required: true }),
    false,
  );
});

test("configured manager id overrides stale browser state", () => {
  assert.equal(resolveInitialFplManagerId(123, "456"), 123);
  assert.equal(resolveInitialFplManagerId(null, "456"), 456);
  assert.equal(resolveInitialFplManagerId(null, "not-an-id"), null);
});
