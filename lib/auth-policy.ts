export type OAuthProfile = {
  id?: string | number;
};

export type GitHubSessionUser = {
  githubId?: string | number;
};

// Better Auth requires provider-mapped additional fields to remain input-enabled.
// Disabling this route keeps authenticated clients from rewriting the access claim.
export const AUTH_DISABLED_PATHS = ["/update-user"] as const;

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
