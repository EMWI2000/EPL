import { loadLeagueOverview, loadLeagueSeasonDetails, LeagueScopeError } from "@/lib/league-overview";
import { configuredFplManagerId } from "@/lib/fpl-manager-config";
import { getCurrentSession } from "@/lib/require-user";

export const runtime = "nodejs";
export const maxDuration = 30;
export async function GET(request: Request) {
  const headers = { "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" };
  if (!await getCurrentSession()) return Response.json({ error: "Du skal være logget ind." }, { status: 401, headers });
  try {
    const managerId = configuredFplManagerId(process.env.FPL_MANAGER_ID, { required: true });
    const params = new URL(request.url).searchParams;
    if (params.has("league") || params.has("rival") || params.has("from") || params.has("to")) {
      const number = (key: string) => /^\d+$/.test(params.get(key) ?? "") ? Number(params.get(key)) : NaN;
      return Response.json(await loadLeagueSeasonDetails(managerId!, {
        leagueId: number("league"), rivalId: number("rival"), fromEvent: number("from"), toEvent: number("to"),
      }), { headers });
    }
    return Response.json(await loadLeagueOverview(managerId!), { headers });
  } catch (error) {
    if (error instanceof LeagueScopeError) return Response.json({ error: error.message }, { status: 400, headers });
    return Response.json({ error: "Ligaerne kunne ikke hentes. Prøv igen om lidt." }, { status: 503, headers });
  }
}
