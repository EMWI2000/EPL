import { FplDashboard } from "@/components/fpl-dashboard";
import { configuredFplManagerId } from "@/lib/fpl-manager-config";
import { requireUser } from "@/lib/require-user";

export default async function HomePage() {
  const session = await requireUser();

  return (
    <FplDashboard
      userName={session.user.name || "Emil"}
      initialManagerId={configuredFplManagerId(process.env.FPL_MANAGER_ID, {
        required: process.env.VERCEL_ENV === "production",
      })}
    />
  );
}
