import { decisionResultsResponse } from "@/lib/decision-results-response";
import { configuredFplManagerId } from "@/lib/fpl-manager-config";
import { getCurrentSession } from "@/lib/require-user";

export const runtime = "nodejs";
export const maxDuration = 30;

export async function GET(request: Request) {
  return decisionResultsResponse(request, await getCurrentSession(),
    () => configuredFplManagerId(process.env.FPL_MANAGER_ID, { required: true }));
}
