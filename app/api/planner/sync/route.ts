import { proxyInternalJson } from "@/lib/internal-function-proxy";

export const runtime = "nodejs";
export const maxDuration = 60;

export async function POST(request: Request) {
  return proxyInternalJson(request, {
    endpoint: "/api/manager_state",
    unauthorizedMessage: "Du skal være logget ind for at hente et FPL-hold.",
    unavailableMessage: "FPL-holdet kunne ikke hentes lige nu.",
  });
}
