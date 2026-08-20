import { redirect } from "next/navigation";

import { LoginCard } from "@/components/login-card";
import { getCurrentSession } from "@/lib/require-user";

type LoginPageProps = {
  searchParams: Promise<{ error?: string | string[] }>;
};

export default async function LoginPage({ searchParams }: LoginPageProps) {
  const session = await getCurrentSession();
  if (session) {
    redirect("/");
  }

  const params = await searchParams;
  return <LoginCard rejected={Boolean(params.error)} />;
}
