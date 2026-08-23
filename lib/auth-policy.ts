export type OAuthProfile = {
  id?: string | number;
};

export type GitHubSessionUser = {
  githubId?: string | number;
};

// Better Auth requires provider-mapped additional fields to remain input-enabled.
// Keep the provider-owned claim immutable and never expose GitHub OAuth tokens
// through browser-callable account endpoints that this app does not use.
export const AUTH_DISABLED_PATHS = [
  "/update-user",
  "/get-access-token",
  "/refresh-token",
  "/link-social",
  "/unlink-account",
  "/list-accounts",
  "/account-info",
] as const;

export function isAllowedGitHubIdentity(
  allowedGitHubId: string | undefined,
  providerId: string | undefined,
  profile: unknown,
) {
  const githubProfile = profile as OAuthProfile | undefined;

  return Boolean(
    allowedGitHubId &&
      providerId === "github" &&
      githubProfile?.id !== undefined &&
      String(githubProfile.id) === allowedGitHubId,
  );
}

export function isAllowedGitHubSessionUser(
  allowedGitHubId: string | undefined,
  sessionUser: GitHubSessionUser,
) {
  return isAllowedGitHubIdentity(allowedGitHubId, "github", {
    id: sessionUser.githubId,
  });
}
