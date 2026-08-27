import assert from "node:assert/strict";
import test from "node:test";

import {
  MAX_AI_REVIEW_INBOUND_REQUEST_BYTES,
  hasInvalidAiReviewContentLength,
  isAiReviewBodyTooLarge,
} from "../lib/ai-review-body-limit.ts";

test("allows the bounded full planner request and rejects larger declared bodies", () => {
  assert.equal(hasInvalidAiReviewContentLength(null), false);
  assert.equal(hasInvalidAiReviewContentLength(String(MAX_AI_REVIEW_INBOUND_REQUEST_BYTES)), false);
  assert.equal(hasInvalidAiReviewContentLength(String(MAX_AI_REVIEW_INBOUND_REQUEST_BYTES + 1)), true);
  assert.equal(hasInvalidAiReviewContentLength("invalid"), true);
  assert.equal(hasInvalidAiReviewContentLength("1.5"), true);
  assert.equal(hasInvalidAiReviewContentLength("-1"), true);
});

test("measures the actual UTF-8 body at the same 256 KiB boundary", () => {
  assert.equal(isAiReviewBodyTooLarge("a".repeat(MAX_AI_REVIEW_INBOUND_REQUEST_BYTES)), false);
  assert.equal(isAiReviewBodyTooLarge("a".repeat(MAX_AI_REVIEW_INBOUND_REQUEST_BYTES + 1)), true);
  assert.equal(isAiReviewBodyTooLarge("ø".repeat(MAX_AI_REVIEW_INBOUND_REQUEST_BYTES / 2)), false);
  assert.equal(isAiReviewBodyTooLarge(`ø${"a".repeat(MAX_AI_REVIEW_INBOUND_REQUEST_BYTES - 1)}`), true);
});
