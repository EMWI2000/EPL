import "server-only";

import { betterAuth } from "better-auth";

import { isAllowedGitHubIdentity } from "@/lib/auth-policy";

function authEnvironment(name: string, localFallback?: string) {
  const value = process.env[name];
  if (value) return value;
  if (process.env.VERCEL === "1") {
    throw new Error(`${name} must be configured in Vercel before deployment.`);
  }
  return localFallback;
}

const allowedGitHubId = authEnvironment("ALLOWED_GITHUB_ID");

export const auth = betterAuth({
  appName: "FPL HoldPlanner",
  baseURL: authEnvironment("BETTER_AUTH_URL", "http://localhost:3000"),
  secret: authEnvironment("BETTER_AUTH_SECRET", "local-development-secret-change-me"),
  socialProviders: {
    github: {
      clientId: authEnvironment("GITHUB_CLIENT_ID", "local-unconfigured-client")!,
      clientSecret: authEnvironment("GITHUB_CLIENT_SECRET", "local-unconfigured-secret")!,
      scope: ["read:user", "user:email"],
    },
  },
  user: {
    validateUserInfo: ({ source }) => {
      if (
        isAllowedGitHubIdentity(
          allowedGitHubId,
          source.oauth?.providerId,
          source.oauth?.profile,
        )
      ) {
        return;
      }

      return {
        error: "access_denied",
        errorDescription: "Denne GitHub-konto har ikke adgang.",
      };
    },
  },
  session: {
    cookieCache: {
      enabled: true,
      maxAge: 7 * 24 * 60 * 60,
      strategy: "jwe",
      refreshCache: true,
      version: process.env.SESSION_VERSION ?? "1",
    },
  },
  account: {
    storeStateStrategy: "cookie",
    storeAccountCookie: true,
  },
});
