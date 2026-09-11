/** Public, bounded league context. Never receives FPL login credentials. */
import { buildLeagueDiagnosis, type LeagueDiagnosis } from "./league-diagnosis.ts";

export type LeagueRival = {
  entry: number; team: string; rank: number; points: number; gap: number | null;
  captain: string | null; different_players: string[];
  chips_remaining: string[] | null;
  diagnosis: LeagueDiagnosis;
};
export type LeagueOverview = {
  generated_at: string; source_event: number; target_event: number;
  own_points: number; own_chips_remaining: string[] | null;
  leagues: { id: number; name: string; rank: number | null; previous_rank: number | null;
    leader_gap: number | null; rivals: LeagueRival[]; partial: boolean }[];
  warnings: string[];
};
type Obj = Record<string, unknown>;
const obj = (v: unknown): Obj => v !== null && typeof v === "object" && !Array.isArray(v) ? v as Obj : {};
const rows = (v: unknown): Obj[] => Array.isArray(v) ? v.map(obj) : [];
const int = (v: unknown): number | null => typeof v === "number" && Number.isSafeInteger(v) && v >= 0 ? v : null;
const signedInt = (v: unknown): number | null => typeof v === "number" && Number.isSafeInteger(v) ? v : null;
const label = (v: unknown): string => typeof v === "string" ? v.replace(/[\u0000-\u001f\u007f]/g, "").slice(0, 100) : "Ukendt";
const chipNames = ["wildcard", "freehit", "bboost", "3xc"];

export function remainingLeagueChips(history: unknown, target: number): string[] | null {
  const value = obj(history).chips;
  if (!Array.isArray(value)) return null;
  const first = target <= 19 ? 1 : 20;
  const used = new Set<string>();
  for (const row of rows(value)) {
    const event = int(row.event);
    if (event === null || event < 1 || event >= target || !chipNames.includes(String(row.name))) return null;
    if (event >= first) used.add(String(row.name));
  }
  return chipNames.filter(chip => !used.has(chip));
}

export async function loadLeagueOverview(
  managerId: number, fetcher: typeof fetch = fetch, now = Date.now(),
): Promise<LeagueOverview> {
  if (!Number.isSafeInteger(managerId) || managerId <= 0) throw new Error("Invalid manager");
  const started = Date.now();
  const get = async (path: string): Promise<unknown> => {
    const remaining = 18_000 - (Date.now() - started);
    if (remaining <= 0) throw new Error("League time budget exceeded");
    const response = await fetcher(`https://fantasy.premierleague.com/api/${path}`, {
      headers: { Accept: "application/json" }, cache: "no-store",
      signal: AbortSignal.timeout(Math.max(1, Math.min(5_000, remaining))), redirect: "error",
    });
    if (!response.ok) throw new Error("Official league data unavailable");
    return response.json();
  };
  const [rawManager, rawBootstrap] = await Promise.all([get(`entry/${managerId}/`), get("bootstrap-static/")]);
  const manager = obj(rawManager), bootstrap = obj(rawBootstrap);
  const events = rows(bootstrap.events);
  const target = events.filter(e => int(e.id) && typeof e.deadline_time === "string" && Date.parse(e.deadline_time) > now)
    .sort((a, b) => Number(a.id) - Number(b.id))[0];
  const source = events.filter(e => int(e.id) && typeof e.deadline_time === "string" && Date.parse(e.deadline_time) <= now)
    .sort((a, b) => Number(b.id) - Number(a.id))[0];
  const ownPoints = signedInt(manager.summary_overall_points);
  if (!target || !source || ownPoints === null) throw new Error("No current league context");
  const targetEvent = Number(target.id), sourceEvent = Number(source.id);
  const sourceFinalized = source.finished === true && source.data_checked === true;
  const players = new Map(rows(bootstrap.elements).filter(p => int(p.id)).map(p => [Number(p.id), label(p.web_name)]));
  const memberships = rows(obj(manager.leagues).classic).filter(l => l.league_type === "x" && int(l.id) && int(l.entry_rank));
  const selected = memberships.slice(0, 3);
  const warnings = ["Offentlige hold er låst ved seneste deadline. Nye transfers og næste kaptajn er ikke synlige.",
    "Udvalget viser ligalederen og nærmeste hold foran dig, ikke en beregnet sandsynlighed for at vinde."];
  if (memberships.length > 3) warnings.push("Kun de første tre invitationelle ligaer er med i dette overblik.");
  const picksCache = new Map<number, Promise<unknown>>();
  const historyCache = new Map<number, Promise<unknown>>();
  const picks = (id: number) => {
    if (!picksCache.has(id)) picksCache.set(id, get(`entry/${id}/event/${sourceEvent}/picks/`).catch(() => null));
    return picksCache.get(id)!;
  };
  const history = (id: number) => {
    if (!historyCache.has(id)) historyCache.set(id, get(`entry/${id}/history/`).catch(() => null));
    return historyCache.get(id)!;
  };
  const [ownPicksRaw, ownHistory, standings, live] = await Promise.all([
    picks(managerId), history(managerId),
    Promise.all(selected.map(async l => {
      try {
        // FPL returns 50 standings per page; page above the user also covers rank51 etc.
        const nearPage = Math.floor((Number(l.entry_rank) - 2) / 50) + 1;
        const ownPage = Math.floor((Number(l.entry_rank) - 1) / 50) + 1;
        const paths = [...new Set([1, Math.max(1, nearPage), ownPage])];
        const pages = await Promise.all(paths.map(page => get(`leagues-classic/${l.id}/standings/?page_standings=${page}`)));
        return { rows: pages.flatMap(p => rows(obj(obj(p).standings).results)), pages: paths };
      } catch { return null; }
    })),
    sourceFinalized
      ? get(`event/${sourceEvent}/live/`).catch(() => null) : Promise.resolve(null),
  ]);
  const validSquad = (payload: unknown): Obj[] => {
    const squad = rows(obj(payload).picks);
    return squad.length === 15 && squad.every(p => int(p.element) && players.has(Number(p.element)))
      && new Set(squad.map(p => p.element)).size === 15 && squad.filter(p => p.is_captain === true).length === 1 ? squad : [];
  };
  const ownIds = new Set(validSquad(ownPicksRaw).map(p => Number(p.element)));
  const leagues = await Promise.all(selected.map(async (l, index) => {
    const leagueRows = standings[index]?.rows ?? null;
    const valid = leagueRows?.filter(r => int(r.entry) && int(r.rank) && signedInt(r.total) !== null) ?? [];
    const leader = valid.find(r => r.rank === 1);
    const ownRows = valid.filter(r => r.entry === managerId);
    // Entry-summary ranks can lag behind standings. Conflicting pages are also unknown.
    const ownRow = new Set(ownRows.map(r => `${r.rank}:${r.total}`)).size === 1 ? ownRows[0] : undefined;
    const ownRank = int(ownRow?.rank), ownLeaguePoints = signedInt(ownRow?.total);
    const nearPage = ownRank === null ? null : Math.max(1, Math.floor((ownRank - 2) / 50) + 1);
    const near = ownRank !== null && nearPage !== null && standings[index]?.pages.includes(nearPage)
      ? valid.filter(r => Number(r.rank) < ownRank).sort((a, b) => Number(b.rank) - Number(a.rank))[0] : undefined;
    const unique = [...new Map([leader, near].filter((r): r is Obj => Boolean(r) && r?.entry !== managerId).map(r => [r.entry, r])).values()];
    let partial = leagueRows === null || !leader || ownLeaguePoints === null || (ownRank !== null && ownRank > 1 && !near);
    const rivals = await Promise.all(unique.map(async r => {
      const [p, h] = await Promise.all([picks(Number(r.entry)), history(Number(r.entry))]);
      const squad = validSquad(p);
      const chips = remainingLeagueChips(h, targetEvent);
      if (squad.length !== 15 || chips === null || ownIds.size !== 15) partial = true;
      const effectiveCaptains = squad.filter(v => v.multiplier === 2 || v.multiplier === 3);
      const captain = sourceFinalized ? effectiveCaptains.length === 1 ? effectiveCaptains[0] : undefined
        : squad.find(v => v.is_captain === true);
      const diagnosis = buildLeagueDiagnosis({ events, sourceEvent, ownHistory, rivalHistory: h,
        ownPicks: ownPicksRaw, rivalPicks: p, live, playerNames: players });
      if ((sourceFinalized && diagnosis.latest.status !== "available")
        || (diagnosis.trend.requested_gameweeks.length > 0 && diagnosis.trend.status !== "complete")) partial = true;
      return {
        entry: Number(r.entry), team: label(r.entry_name), rank: Number(r.rank), points: Number(r.total),
        gap: ownLeaguePoints === null ? null : Number(r.total) - ownLeaguePoints,
        captain: captain ? players.get(Number(captain.element)) ?? null : null,
        different_players: ownIds.size === 15 && squad.length === 15
          ? squad.filter(v => !ownIds.has(Number(v.element))).map(v => players.get(Number(v.element)) ?? "Ukendt") : [],
        chips_remaining: chips,
        diagnosis,
      };
    }));
    return { id: Number(l.id), name: label(l.name), rank: ownRank, previous_rank: int(l.entry_last_rank) ?? ownRank,
      leader_gap: leader && ownLeaguePoints !== null ? Number(leader.total) - ownLeaguePoints : null, rivals, partial };
  }));
  return { generated_at: new Date(now).toISOString(), source_event: sourceEvent, target_event: targetEvent,
    own_points: ownPoints, own_chips_remaining: remainingLeagueChips(ownHistory, targetEvent), leagues, warnings };
}

/** Deliberately excludes manager/league IDs and user-controlled team names. */
export function leagueAiContext(overview: LeagueOverview | null) {
  if (!overview) return { available: false, limitation: "League data unavailable; do not infer opponents or gaps." };
  const ownLatest = overview.leagues.flatMap(l => l.rivals)
    .find(r => r.diagnosis.latest.status === "available")?.diagnosis.latest.own ?? null;
  const playerKeys = new Map<string, string>();
  const playerKey = (name: string) => {
    if (!playerKeys.has(name)) playerKeys.set(name, `P${playerKeys.size + 1}`);
    return playerKeys.get(name)!;
  };
  return {
    available: true, observed_at: overview.generated_at, source_gameweek: overview.source_event,
    target_gameweek: overview.target_event, own_chips_remaining: overview.own_chips_remaining,
    own_latest_points: ownLatest,
    diagnosis_scope: "Latest three finalized season rounds, not necessarily the full league gap or a late-start league's scoring period. Positive changes favor the rival. Outcome decomposition, not decision-quality attribution. Wildcard/Free Hit effects and avoidable bench losses are unknown. Latest own points are shared across rivals; contributions are rival minus own, including signed hit impact.",
    leagues: overview.leagues.map((l, index) => ({ league: index + 1, rank: l.rank, leader_gap: l.leader_gap, partial: l.partial,
      rivals: l.rivals.map(r => ({ rank: r.rank, gap: r.gap, last_captain: r.captain === null ? null : playerKey(r.captain),
        different_players: r.different_players.map(playerKey), chips_remaining: r.chips_remaining,
        diagnosis: {
          trend: r.diagnosis.trend,
          latest: {
            status: r.diagnosis.latest.status, gameweek: r.diagnosis.latest.gameweek,
            reason: r.diagnosis.latest.reason, gap_change: r.diagnosis.latest.gap_change,
            rival_net_points: r.diagnosis.latest.rival?.net ?? null,
            rival_active_chip: r.diagnosis.latest.rival?.active_chip ?? null,
            contributions: r.diagnosis.latest.contributions,
            core_player_gains: r.diagnosis.latest.core_player_gains.map(p => ({ player: playerKey(p.name), gap_change: p.gap_change })),
            core_player_losses: r.diagnosis.latest.core_player_losses.map(p => ({ player: playerKey(p.name), gap_change: p.gap_change })),
          },
        } })) })),
    player_labels: Object.fromEntries([...playerKeys].map(([name, key]) => [key, name])),
    limitation: "Public last-deadline squads only. All three leagues matter; no selected priority. No opponent transfer forecast or win-probability model. Gaps alone do not justify hits or sacrificing expected points.",
  };
}
