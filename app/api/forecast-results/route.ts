import { buildForecastAuditResults } from "@/lib/forecast-audit";
import { configuredFplManagerId } from "@/lib/fpl-manager-config";
import { getCurrentSession } from "@/lib/require-user";

export const runtime = "nodejs";
export const maxDuration = 30;

const HEADERS = { "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" };

function error(status: number, code: string, message: string) {
  return Response.json({ error: { code, message } }, { status, headers: HEADERS });
}

async function officialJson(path: "bootstrap-static/" | `event/${number}/live/`): Promise<unknown> {
  const response = await fetch(`https://fantasy.premierleague.com/api/${path}`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
    redirect: "error",
    signal: AbortSignal.timeout(10_000),
  });
  if (!response.ok) throw new Error("FPL unavailable");
  const raw = await response.text();
  if (Buffer.byteLength(raw, "utf8") > 6 * 1024 * 1024) throw new Error("FPL response too large");
  return JSON.parse(raw) as unknown;
}

export async function GET(request: Request) {
  if (!await getCurrentSession()) return error(401, "unauthorized", "Du skal være logget ind for at måle prognoserne.");
  try {
    configuredFplManagerId(process.env.FPL_MANAGER_ID, { required: process.env.VERCEL_ENV === "production" });
  } catch {
    return error(503, "manager_unconfigured", "Det personlige FPL-hold er ikke konfigureret.");
  }
  const params = new URL(request.url).searchParams;
  const eventText = params.get("event");
  const deadline = params.get("deadline");
  if (params.size !== 2 || params.getAll("event").length !== 1 || params.getAll("deadline").length !== 1
    || !eventText || !/^(?:[1-9]|[12]\d|3[0-8])$/.test(eventText) || !deadline || deadline.length > 40
    || !/(?:Z|[+-]\d\d:\d\d)$/i.test(deadline) || !Number.isFinite(Date.parse(deadline))) {
    return error(400, "invalid_request", "Vælg en gemt spillerunde med en gyldig deadline.");
  }
  const event = Number(eventText);
  try {
    const bootstrap = await officialJson("bootstrap-static/");
    const events = (bootstrap as { events?: { id?: unknown; deadline_time?: unknown; finished?: unknown; data_checked?: unknown }[] }).events;
    if (!Array.isArray(events)) throw new Error("Missing FPL events");
    const matching = events.find((row) => row?.id === event);
    if (!matching || typeof matching.deadline_time !== "string" || Date.parse(matching.deadline_time) !== Date.parse(deadline)) {
      return error(409, "season_unavailable", "FPL leverer ikke længere resultater for den gemte sæson og deadline.");
    }
    if (matching.finished !== true || matching.data_checked !== true) {
      return error(409, "results_pending", "Spillerunden er ikke færdigkontrolleret af FPL endnu.");
    }
    const live = await officialJson(`event/${event}/live/`);
    return Response.json(buildForecastAuditResults(bootstrap, live, event, deadline, new Date().toISOString()), { headers: HEADERS });
  } catch {
    return error(502, "results_unavailable", "Slutresultaterne kunne ikke hentes. De gemte prognoser er bevaret; prøv igen senere.");
  }
}
