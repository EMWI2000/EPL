import { configuredFplManagerId } from "@/lib/fpl-manager-config";
import { proxyInternalJson } from "@/lib/internal-function-proxy";

export const runtime = "nodejs";
export const maxDuration = 60;

export async function POST(request: Request) {
  const managerId = configuredFplManagerId(process.env.FPL_MANAGER_ID, {
    required: process.env.VERCEL_ENV === "production",
  });
  return proxyInternalJson(request, {
    endpoint: "/api/manager_state",
    unauthorizedMessage: "Du skal være logget ind for at hente et FPL-hold.",
    unavailableMessage: "FPL-holdet kunne ikke hentes lige nu.",
    requestBody: managerId === null
      ? undefined
      : JSON.stringify({ manager_id: managerId }),
    configuredManagerId: managerId ?? undefined,
  });
}
