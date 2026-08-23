import { configuredFplManagerId } from "@/lib/fpl-manager-config";
import { proxyInternalJson } from "@/lib/internal-function-proxy";

export const runtime = "nodejs";
export const maxDuration = 60;

export async function POST(request: Request) {
  const managerId = configuredFplManagerId(process.env.FPL_MANAGER_ID, {
    required: process.env.VERCEL_ENV === "production",
  });
  return proxyInternalJson(request, {
    endpoint: "/api/compute",
    unauthorizedMessage: "Du skal være logget ind for at beregne et hold.",
    unavailableMessage: "Beregningstjenesten kunne ikke kontaktes.",
    configuredManagerId: managerId ?? undefined,
    requireConfiguredManagerId: managerId !== null,
  });
}
