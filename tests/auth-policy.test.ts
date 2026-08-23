import assert from "node:assert/strict";
import test from "node:test";

import {
  AUTH_DISABLED_PATHS,
  isAllowedGitHubIdentity,
  isAllowedGitHubSessionUser,
} from "../lib/auth-policy.ts";

test("blocks provider-claim mutation and browser OAuth-token endpoints", () => {
  assert.deepEqual(AUTH_DISABLED_PATHS, [
    "/update-user",
    "/get-access-token",
    "/refresh-token",
    "/link-social",
    "/unlink-account",
    "/list-accounts",
    "/account-info",
  ]);
});

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
  const sessionUser = { id: "internal-user-id", githubId: "199608244" };

  assert.equal(isAllowedGitHubSessionUser("199608244", sessionUser), true);
  assert.equal(isAllowedGitHubSessionUser("42", sessionUser), false);
  assert.equal(isAllowedGitHubSessionUser("199608244", { id: "199608244" }), false);
});
