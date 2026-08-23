import assert from "node:assert/strict";
import test from "node:test";

import {
  hasValidPublicOrigin,
  resolvePublicRequestOrigin,
} from "../lib/public-request-origin.ts";

test("uses the forwarded public origin behind Vercel's local proxy", () => {
  const request = new Request("http://127.0.0.1:43121/api/planner/sync", {
    headers: {
      host: "127.0.0.1:43121",
      origin: "http://localhost:3000",
      "x-forwarded-host": "localhost:3000",
      "x-forwarded-proto": "http",
    },
  });

  assert.equal(resolvePublicRequestOrigin(request), "http://localhost:3000");
  assert.equal(hasValidPublicOrigin(request), true);
});

test("accepts an ordinary same-origin production request", () => {
  const request = new Request("https://epl.example/api/recommend", {
    headers: {
      host: "epl.example",
      origin: "https://epl.example",
    },
  });

  assert.equal(resolvePublicRequestOrigin(request), "https://epl.example");
  assert.equal(hasValidPublicOrigin(request), true);
});

test("rejects a cross-origin or malformed browser request", () => {
  const crossOrigin = new Request("https://epl.example/api/recommend", {
    headers: {
      host: "epl.example",
      origin: "https://attacker.example",
    },
  });
  const malformed = new Request("https://epl.example/api/recommend", {
    headers: { host: "epl.example", origin: "not an origin" },
  });

  assert.equal(hasValidPublicOrigin(crossOrigin), false);
  assert.equal(hasValidPublicOrigin(malformed), false);
});

test("falls back safely when forwarded headers are invalid", () => {
  const request = new Request("https://epl.example/api/recommend", {
    headers: {
      origin: "https://epl.example",
      "x-forwarded-host": "attacker.example/path",
      "x-forwarded-proto": "javascript",
    },
  });

  assert.equal(resolvePublicRequestOrigin(request), "https://epl.example");
  assert.equal(hasValidPublicOrigin(request), true);
});
