import { loadWildcardBenchmark } from "@/lib/wildcard-benchmark-loader";
import { configuredFplManagerId } from "@/lib/fpl-manager-config";
import { getCurrentSession } from "@/lib/require-user";

export const runtime = "nodejs";
export const maxDuration = 30;

export async function GET() {
  const headers = { "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" };
  if (!await getCurrentSession()) {
    return Response.json({ error: "Du skal være logget ind." }, { status: 401, headers });
  }
  try {
    const managerId = configuredFplManagerId(process.env.FPL_MANAGER_ID, { required: true });
    return Response.json(await loadWildcardBenchmark(managerId!), { headers });
  } catch {
    return Response.json({ error: "Wildcard-sammenligningen kunne ikke hentes. Prøv igen om lidt." }, { status: 503, headers });
  }
}
