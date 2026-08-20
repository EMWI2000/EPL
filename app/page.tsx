import { FplDashboard } from "@/components/fpl-dashboard";
import { requireUser } from "@/lib/require-user";

export default async function HomePage() {
  const session = await requireUser();

  return <FplDashboard userName={session.user.name || "Emil"} />;
}
