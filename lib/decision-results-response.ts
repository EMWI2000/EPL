import { loadDecisionResult } from "./decision-results.ts";

/** Auth and manager selection live on the server; the browser supplies only a GW. */
export async function decisionResultsResponse(
  request: Request,
  session: unknown,
  configuredManager: () => number | null,
  load: typeof loadDecisionResult = loadDecisionResult,
): Promise<Response> {
  const headers = { "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" };
  if (!session) return Response.json({ error: "Du skal være logget ind." }, { status: 401, headers });
  const params = new URL(request.url).searchParams;
  if ([...params.keys()].some(key => key !== "event") || params.getAll("event").length !== 1
    || !/^(?:[1-9]|[12]\d|3[0-8])$/.test(params.get("event") ?? "")) {
    return Response.json({ error: "Vælg en gyldig spillerunde." }, { status: 400, headers });
  }
  try {
    const manager = configuredManager();
    if (!manager) throw new TypeError("Missing configured manager");
    return Response.json(await load(manager, Number(params.get("event"))), { headers });
  } catch {
    return Response.json({ error: "Rundens officielle resultat kunne ikke hentes. Prøv igen." }, { status: 503, headers });
  }
}
