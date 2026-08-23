import "server-only";

import { betterAuth } from "better-auth";

import {
  AUTH_DISABLED_PATHS,
  isAllowedGitHubIdentity,
} from "@/lib/auth-policy";

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
  disabledPaths: [...AUTH_DISABLED_PATHS],
  socialProviders: {
    github: {
      clientId: authEnvironment("GITHUB_CLIENT_ID", "local-unconfigured-client")!,
      clientSecret: authEnvironment("GITHUB_CLIENT_SECRET", "local-unconfigured-secret")!,
      scope: ["read:user", "user:email"],
      mapProfileToUser: (profile) => ({
        githubId: String(profile.id),
      }),
    },
  },
  user: {
    additionalFields: {
      githubId: {
        type: "string",
        required: true,
        input: true,
      },
    },
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
      version: authEnvironment("SESSION_VERSION", "1")!,
    },
  },
  account: {
    storeStateStrategy: "cookie",
    // The GitHub token is only needed during the OAuth callback. The app never
    // calls GitHub on the user's behalf after sign-in, so do not retain token
    // material in a browser cookie.
    storeAccountCookie: false,
  },
});
