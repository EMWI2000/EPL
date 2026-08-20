import assert from "node:assert/strict";
import test from "node:test";

import { isAllowedGitHubIdentity } from "../lib/auth-policy.ts";

test("allows the configured immutable GitHub identity", () => {
  assert.equal(isAllowedGitHubIdentity("199608244", "github", { id: 199608244 }), true);
  assert.equal(isAllowedGitHubIdentity("199608244", "github", { id: "199608244" }), true);
});

test("fails closed for another provider, identity, or missing configuration", () => {
  assert.equal(isAllowedGitHubIdentity("199608244", "gitlab", { id: 199608244 }), false);
  assert.equal(isAllowedGitHubIdentity("199608244", "github", { id: 42 }), false);
  assert.equal(isAllowedGitHubIdentity("199608244", "github", {}), false);
  assert.equal(isAllowedGitHubIdentity(undefined, "github", { id: 199608244 }), false);
});

test("can revalidate the identity stored in an existing stateless session", () => {
  const sessionUser = { id: "199608244" };

  assert.equal(isAllowedGitHubIdentity("199608244", "github", sessionUser), true);
  assert.equal(isAllowedGitHubIdentity("42", "github", sessionUser), false);
});
