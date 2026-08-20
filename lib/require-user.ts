import "server-only";

import { headers } from "next/headers";
import { redirect } from "next/navigation";

import { auth } from "@/lib/auth";
import { isAllowedGitHubIdentity } from "@/lib/auth-policy";

export async function getCurrentSession() {
  const session = await auth.api.getSession({
    headers: await headers(),
  });
  if (
    !session ||
    !isAllowedGitHubIdentity(process.env.ALLOWED_GITHUB_ID, "github", {
      id: session.user.id,
    })
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
