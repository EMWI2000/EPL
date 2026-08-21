import "server-only";

import { headers } from "next/headers";
import { redirect } from "next/navigation";

import { auth } from "@/lib/auth";
import { isAllowedGitHubSessionUser } from "@/lib/auth-policy";

export async function getCurrentSession() {
  const session = await auth.api.getSession({
    headers: await headers(),
  });
  if (
    !session ||
    !isAllowedGitHubSessionUser(process.env.ALLOWED_GITHUB_ID, session.user)
  ) {
    return null;
  }
  return session;
}

export async function requireUser() {
  const session = await getCurrentSession();

  if (!session) {
    redirect("/login");
  }

  return session;
}
