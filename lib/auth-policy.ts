export type OAuthProfile = {
  id?: string | number;
};

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
