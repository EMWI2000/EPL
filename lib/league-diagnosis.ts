/** Descriptive, reconciled public scores. Not a counterfactual measure of decision quality. */
type Obj = Record<string, unknown>;
const obj = (value: unknown): Obj => value !== null && typeof value === "object" && !Array.isArray(value) ? value as Obj : {};
const integer = (value: unknown): value is number => typeof value === "number" && Number.isSafeInteger(value);
const chipNames = ["wildcard", "freehit", "bboost", "3xc"];

export type LeaguePointComponents = {
  core: number; captain: number; bench_boost: number; triple_captain: number;
  hit_cost: number; gross: number; net: number; active_chip: string | null;
};
type GapComponents = { core: number; captain: number; bench_boost: number; triple_captain: number; hits: number };
type PlayerGap = { name: string; gap_change: number };
export type LeagueDiagnosis = {
  scope: string;
  trend: {
    status: "complete" | "partial" | "unavailable";
    requested_gameweeks: number[]; missing_gameweeks: number[];
    rounds: { gameweek: number; own_net: number; rival_net: number; gap_change: number }[];
    gap_change: number | null;
  };
  latest: {
    status: "available" | "unavailable"; gameweek: number; reason: string | null;
    own: LeaguePointComponents | null; rival: LeaguePointComponents | null;
    gap_change: number | null; contributions: GapComponents | null;
    core_player_gains: PlayerGap[]; core_player_losses: PlayerGap[];
  };
};

function historyRound(history: unknown, event: number): { points: number; hits: number } | null {
  const current = obj(history).current;
  if (!Array.isArray(current)) return null;
  const matching = current.map(obj).filter(row => row.event === event);
  if (matching.length !== 1) return null;
  const row = matching[0];
  if (!integer(row.points) || !integer(row.event_transfers_cost) || row.event_transfers_cost < 0) return null;
  return { points: row.points, hits: row.event_transfers_cost };
}

function decompose(
  picksPayload: unknown, history: unknown, live: unknown, event: number,
): { points: LeaguePointComponents; corePlayers: Map<number, number> } | null {
  const payload = obj(picksPayload), entry = obj(payload.entry_history);
  const historyScore = historyRound(history, event);
  const chip = payload.active_chip;
  if (!historyScore || entry.event !== event || entry.points !== historyScore.points
    || entry.event_transfers_cost !== historyScore.hits || (chip !== null && !chipNames.includes(String(chip)))) return null;
  if (!Array.isArray(payload.picks) || payload.picks.length !== 15 || !Array.isArray(obj(live).elements)) return null;
  const picks = payload.picks.map(obj);
  if (picks.some(p => !integer(p.element) || p.element < 1 || !integer(p.position) || p.position < 1 || p.position > 15
    || !integer(p.multiplier) || p.multiplier < 0 || p.multiplier > 3
    || typeof p.is_captain !== "boolean" || typeof p.is_vice_captain !== "boolean")
    || new Set(picks.map(p => p.element)).size !== 15 || new Set(picks.map(p => p.position)).size !== 15
    || picks.filter(p => p.is_captain).length !== 1 || picks.filter(p => p.is_vice_captain).length !== 1
    || picks.some(p => p.is_captain && p.is_vice_captain)) return null;
  const scoring = picks.filter(p => Number(p.multiplier) > 0);
  const captains = picks.filter(p => Number(p.multiplier) > 1);
  if (scoring.length > (chip === "bboost" ? 15 : 11) || captains.length > 1
    || captains.some(p => !p.is_captain && !p.is_vice_captain)
    || captains.some(p => p.multiplier !== (chip === "3xc" ? 3 : 2))) return null;
  const scores = new Map<number, number>();
  const owned = new Set(picks.map(p => Number(p.element)));
  for (const raw of obj(live).elements as unknown[]) {
    const player = obj(raw), stats = obj(player.stats);
    if (!owned.has(Number(player.id))) continue;
    if (!integer(player.id) || !integer(stats.total_points) || scores.has(player.id)) return null;
    scores.set(player.id, stats.total_points);
  }
  if (scores.size !== 15) return null;
  const points: LeaguePointComponents = {
    core: 0, captain: 0, bench_boost: 0, triple_captain: 0,
    hit_cost: historyScore.hits, gross: 0, net: 0, active_chip: chip as string | null,
  };
  const corePlayers = new Map<number, number>();
  for (const pick of picks) {
    const id = Number(pick.element), multiplier = Number(pick.multiplier), score = scores.get(id)!;
    points.gross += multiplier * score;
    points.captain += Math.min(Math.max(multiplier - 1, 0), 1) * score;
    points.triple_captain += Math.max(multiplier - 2, 0) * score;
    if (multiplier > 0) {
      if (chip === "bboost" && Number(pick.position) > 11) points.bench_boost += score;
      else { points.core += score; corePlayers.set(id, score); }
    }
  }
  points.net = points.gross - points.hit_cost;
  if (points.gross !== historyScore.points
    || points.core + points.captain + points.triple_captain + points.bench_boost !== points.gross) return null;
  return { points, corePlayers };
}

export function buildLeagueDiagnosis(input: {
  events: unknown; sourceEvent: number; ownHistory: unknown; rivalHistory: unknown;
  ownPicks: unknown; rivalPicks: unknown; live: unknown; playerNames: ReadonlyMap<number, string>;
}): LeagueDiagnosis {
  const events = Array.isArray(input.events) ? input.events.map(obj) : [];
  const finalized = [...new Set(events.filter(e => integer(e.id) && e.id > 0 && e.id <= input.sourceEvent
    && e.finished === true && e.data_checked === true).map(e => Number(e.id)))].sort((a, b) => b - a).slice(0, 3).reverse();
  const rounds: LeagueDiagnosis["trend"]["rounds"] = [];
  const missing: number[] = [];
  for (const event of finalized) {
    const own = historyRound(input.ownHistory, event), rival = historyRound(input.rivalHistory, event);
    if (!own || !rival) { missing.push(event); continue; }
    const ownNet = own.points - own.hits, rivalNet = rival.points - rival.hits;
    rounds.push({ gameweek: event, own_net: ownNet, rival_net: rivalNet, gap_change: rivalNet - ownNet });
  }
  const result: LeagueDiagnosis = {
    scope: "Season points in the latest three finalized gameweeks, compared with this currently selected rival. This window does not necessarily explain the full league gap, especially in a late-start league. Positive gap changes favor the rival. Scores describe outcomes, not whether a past decision was good; Wildcard/Free Hit effects and avoidable bench losses are not inferred.",
    trend: {
      status: !rounds.length ? "unavailable" : missing.length ? "partial" : "complete",
      requested_gameweeks: finalized, missing_gameweeks: missing, rounds,
      gap_change: rounds.length && !missing.length ? rounds.reduce((sum, row) => sum + row.gap_change, 0) : null,
    },
    latest: {
      status: "unavailable", gameweek: input.sourceEvent, reason: "Official finalized score data is missing or does not reconcile.",
      own: null, rival: null, gap_change: null, contributions: null, core_player_gains: [], core_player_losses: [],
    },
  };
  const sourceEvents = events.filter(e => e.id === input.sourceEvent);
  if (sourceEvents.length !== 1 || sourceEvents[0].finished !== true || sourceEvents[0].data_checked !== true) {
    result.latest.reason = "The latest deadline's gameweek is not finalized.";
    return result;
  }
  const own = decompose(input.ownPicks, input.ownHistory, input.live, input.sourceEvent);
  const rival = decompose(input.rivalPicks, input.rivalHistory, input.live, input.sourceEvent);
  if (!own || !rival) return result;
  const contributions: GapComponents = {
    core: rival.points.core - own.points.core,
    captain: rival.points.captain - own.points.captain,
    bench_boost: rival.points.bench_boost - own.points.bench_boost,
    triple_captain: rival.points.triple_captain - own.points.triple_captain,
    hits: own.points.hit_cost - rival.points.hit_cost,
  };
  const players = [...new Set([...own.corePlayers.keys(), ...rival.corePlayers.keys()])].map(id => ({
    name: (input.playerNames.get(id) ?? "Ukendt spiller").replace(/[\u0000-\u001f\u007f]/g, "").slice(0, 80),
    gap_change: (rival.corePlayers.get(id) ?? 0) - (own.corePlayers.get(id) ?? 0),
  }));
  result.latest = {
    status: "available", gameweek: input.sourceEvent, reason: null, own: own.points, rival: rival.points,
    gap_change: rival.points.net - own.points.net, contributions,
    core_player_gains: players.filter(p => p.gap_change > 0).sort((a, b) => b.gap_change - a.gap_change || a.name.localeCompare(b.name)).slice(0, 3),
    core_player_losses: players.filter(p => p.gap_change < 0).sort((a, b) => a.gap_change - b.gap_change || a.name.localeCompare(b.name)).slice(0, 3),
  };
  return result;
}
