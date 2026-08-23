"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AccountControl } from "@/components/account-control";
import {
  actionForAiReview,
  buildAiReviewRequest,
  parseAiReviewResponse,
  type AiReviewResponse,
} from "@/lib/ai-review-contract";
import { resolveInitialFplManagerId } from "@/lib/fpl-manager-config";
import {
  type ManagerSyncResponse,
  type ManualManagerState,
  type PlannerAction,
  type PlannerPayload,
  isPlannerPayload,
  parseBankTenths,
  parseFreeTransfers,
  parseManagerId,
  parseManagerSyncResponse,
  serializePlannerRequest,
} from "@/lib/planner-contract";
import {
  ArrowRightIcon,
  CheckIcon,
  ChevronIcon,
  DatabaseIcon,
  DownloadIcon,
  ExternalLinkIcon,
  InfoIcon,
  RefreshIcon,
  ShieldIcon,
  SparkIcon,
  UsersIcon,
} from "@/components/icons";

type Horizon = 1 | 2 | 3 | 4 | 5;
type ForecastVersion = "v2" | "legacy";

function managerIdFromBrowser(configuredManagerId: number | null): number | null {
  try {
    const value = window.localStorage.getItem("fpl-manager-id");
    const managerId = resolveInitialFplManagerId(configuredManagerId, value);
    if (configuredManagerId !== null) {
      rememberManagerId(configuredManagerId);
    } else if (value && managerId === null) {
      window.localStorage.removeItem("fpl-manager-id");
    }
    return managerId;
  } catch {
    return configuredManagerId;
  }
}

function rememberManagerId(managerId: number) {
  try {
    window.localStorage.setItem("fpl-manager-id", String(managerId));
  } catch {
    // Browser storage is only a convenience; server configuration is authoritative.
  }
}

type Settings = {
  horizon: Horizon;
  includeDoubtful: boolean;
  forecastVersion: ForecastVersion;
};

type PlayerProjection = {
  gameweek: number;
  ep: number;
  source: string;
  expected_minutes: number | null;
  appearance_probability: number;
  sixty_probability: number;
  confidence: number;
  reliability: "low" | "medium" | "high";
  fixtures_count: number;
  is_blank: boolean;
  is_dgw: boolean;
  components: Record<string, number>;
};

type Player = {
  id: number;
  name: string;
  team_id: number;
  team: string;
  position: "GKP" | "DEF" | "MID" | "FWD";
  price: number;
  price_tenths: number;
  status: string;
  role: string;
  is_captain: boolean;
  is_vice_captain: boolean;
  bench_order: number | null;
  weighted_ep: number;
  price_signal: Record<string, unknown> | null;
  projections: PlayerProjection[];
};

type SolioMetadata = {
  requested: boolean;
  applied: boolean;
  gameweek: number | null;
  generated_at: string | null;
  matched: number;
  usable: number;
  unmatched: number;
  ambiguous: number;
  warning: string | null;
};

type GameweekLineup = {
  gameweek: number;
  formation: string;
  starting_ids: number[];
  bench_ids: number[];
  captain_id: number;
  vice_captain_id: number;
  projected_xi_points: number;
  projected_captain_bonus: number;
  projected_bench_contribution: number;
  objective_points: number;
};

type RecommendationResponse = {
  meta: {
    generated_at: string;
    gameweek_window: number[];
    horizon: number;
    forecast_version: ForecastVersion;
    include_doubtful: boolean;
    use_solio_requested: boolean;
    validation: {
      status: string;
      message: string;
    };
    data_sources: {
      fpl: {
        player_count: number;
        eligible_count: number;
        optimizer_candidate_count: number;
        shortlist_method: string;
      };
      solio: SolioMetadata;
      price_signals: {
        available?: boolean;
        player_count?: number;
        price_change_deadlines?: string[];
        warning?: string | null;
      };
    };
    mode: "initial_squad" | "weekly";
  };
  summary: {
    total_cost: number;
    total_cost_tenths: number;
    bank: number;
    bank_tenths: number;
    formation: string;
    objective_points: number;
    projected_xi_points: number;
    projected_captain_bonus: number;
    projected_bench_contribution: number;
  };
  team: {
    squad: Player[];
    starters: Player[];
    bench: Player[];
    captain_id: number;
    vice_captain_id: number;
    gameweeks: GameweekLineup[];
  };
  planner?: PlannerPayload;
  experimental_notice: string;
};

type ApiError = {
  error?: {
    code?: string;
    message?: string;
    details?: unknown;
  };
};

const DEFAULT_SETTINGS: Settings = {
  horizon: 5,
  includeDoubtful: true,
  forecastVersion: "v2",
};

const positionNames: Record<Player["position"], string> = {
  GKP: "Målmand",
  DEF: "Forsvar",
  MID: "Midtbane",
  FWD: "Angreb",
};

const compactNumber = new Intl.NumberFormat("da-DK", {
  maximumFractionDigits: 1,
  minimumFractionDigits: 1,
});

function classNames(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(" ");
}

function formatDateTime(value: string | null | undefined) {
  if (!value) return "Ikke oplyst";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("da-DK", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "Europe/Copenhagen",
  }).format(date);
}

function formatPoints(value: number) {
  return compactNumber.format(value);
}

function formatPrice(value: number) {
  return `£${compactNumber.format(value)}m`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isProjection(value: unknown): value is PlayerProjection {
  return Boolean(
    isRecord(value) &&
      isFiniteNumber(value.gameweek) &&
      isFiniteNumber(value.ep) &&
      typeof value.source === "string" &&
      (value.expected_minutes === null || isFiniteNumber(value.expected_minutes)) &&
      isFiniteNumber(value.appearance_probability) &&
      isFiniteNumber(value.sixty_probability) &&
      isFiniteNumber(value.confidence) &&
      ["low", "medium", "high"].includes(String(value.reliability)) &&
      isFiniteNumber(value.fixtures_count) &&
      typeof value.is_blank === "boolean" &&
      typeof value.is_dgw === "boolean" &&
      isRecord(value.components) &&
      Object.values(value.components).every(isFiniteNumber),
  );
}

function isGameweekLineup(value: unknown): value is GameweekLineup {
  return Boolean(
    isRecord(value) &&
      isFiniteNumber(value.gameweek) &&
      typeof value.formation === "string" &&
      Array.isArray(value.starting_ids) &&
      value.starting_ids.every(isFiniteNumber) &&
      Array.isArray(value.bench_ids) &&
      value.bench_ids.every(isFiniteNumber) &&
      isFiniteNumber(value.captain_id) &&
      isFiniteNumber(value.vice_captain_id) &&
      isFiniteNumber(value.projected_xi_points) &&
      isFiniteNumber(value.projected_captain_bonus) &&
      isFiniteNumber(value.projected_bench_contribution) &&
      isFiniteNumber(value.objective_points),
  );
}

function isPlayer(value: unknown): value is Player {
  if (!isRecord(value)) return false;
  return Boolean(
    isFiniteNumber(value.id) &&
      typeof value.name === "string" &&
      isFiniteNumber(value.team_id) &&
      typeof value.team === "string" &&
      ["GKP", "DEF", "MID", "FWD"].includes(String(value.position)) &&
      isFiniteNumber(value.price) &&
      isFiniteNumber(value.price_tenths) &&
      typeof value.status === "string" &&
      typeof value.role === "string" &&
      typeof value.is_captain === "boolean" &&
      typeof value.is_vice_captain === "boolean" &&
      (value.bench_order === null || isFiniteNumber(value.bench_order)) &&
      isFiniteNumber(value.weighted_ep) &&
      (value.price_signal === null || isRecord(value.price_signal)) &&
      Array.isArray(value.projections) &&
      value.projections.every(isProjection),
  );
}

function isRecommendation(value: unknown): value is RecommendationResponse {
  if (!isRecord(value) || !isRecord(value.meta) || !isRecord(value.summary) || !isRecord(value.team)) {
    return false;
  }
  const { meta, summary, team } = value;
  if (
    !isRecord(meta.data_sources) ||
    !isRecord(meta.data_sources.fpl) ||
    !isRecord(meta.data_sources.solio) ||
    !isRecord(meta.data_sources.price_signals) ||
    !isRecord(meta.validation)
  ) {
    return false;
  }
  const { fpl, solio } = meta.data_sources;
  const hasValidShape = Boolean(
      typeof meta.generated_at === "string" &&
      ["initial_squad", "weekly"].includes(String(meta.mode)) &&
      Array.isArray(meta.gameweek_window) &&
      meta.gameweek_window.every(isFiniteNumber) &&
      isFiniteNumber(meta.horizon) &&
      ["v2", "legacy"].includes(String(meta.forecast_version)) &&
      typeof meta.validation.status === "string" &&
      typeof meta.validation.message === "string" &&
      isFiniteNumber(fpl.player_count) &&
      isFiniteNumber(fpl.eligible_count) &&
      isFiniteNumber(fpl.optimizer_candidate_count) &&
      typeof fpl.shortlist_method === "string" &&
      typeof solio.requested === "boolean" &&
      typeof solio.applied === "boolean" &&
      isFiniteNumber(solio.matched) &&
      isFiniteNumber(solio.usable) &&
      isFiniteNumber(solio.unmatched) &&
      isFiniteNumber(summary.total_cost) &&
      isFiniteNumber(summary.bank) &&
      typeof summary.formation === "string" &&
      isFiniteNumber(summary.objective_points) &&
      isFiniteNumber(summary.projected_captain_bonus) &&
      Array.isArray(team.squad) &&
      team.squad.every(isPlayer) &&
      Array.isArray(team.starters) &&
      team.starters.every(isPlayer) &&
      Array.isArray(team.bench) &&
      team.bench.every(isPlayer) &&
      Array.isArray(team.gameweeks) &&
      team.gameweeks.every(isGameweekLineup) &&
      typeof value.experimental_notice === "string",
  );
  if (!hasValidShape) return false;
  if (value.planner !== undefined && !isPlannerPayload(value.planner)) return false;
  if (meta.mode === "weekly" && !isPlannerPayload(value.planner)) return false;

  const squad = team.squad as Player[];
  const starters = team.starters as Player[];
  const bench = team.bench as Player[];
  const lineups = team.gameweeks as GameweekLineup[];
  const squadIds = new Set(squad.map((player) => player.id));
  if (
    squad.length !== 15 ||
    squadIds.size !== 15 ||
    starters.length !== 11 ||
    bench.length !== 4 ||
    lineups.length === 0
  ) {
    return false;
  }

  return lineups.every((lineup) => {
    const startingIds = new Set(lineup.starting_ids);
    const benchIds = new Set(lineup.bench_ids);
    const allLineupIds = new Set([...startingIds, ...benchIds]);
    return (
      lineup.starting_ids.length === 11 &&
      startingIds.size === 11 &&
      lineup.bench_ids.length === 4 &&
      benchIds.size === 4 &&
      allLineupIds.size === 15 &&
      [...allLineupIds].every((id) => squadIds.has(id)) &&
      startingIds.has(lineup.captain_id) &&
      startingIds.has(lineup.vice_captain_id) &&
      lineup.captain_id !== lineup.vice_captain_id
    );
  });
}

function sourceKind(source: string) {
  return source.toLocaleLowerCase("da-DK").includes("solio") ? "solio" : "internal";
}

function projectionFor(player: Player, gameweek: number) {
  return player.projections.find((projection) => projection.gameweek === gameweek);
}

function expectedMinutesLabel(projection: PlayerProjection) {
  return projection.expected_minutes === null
    ? "xMin ikke oplyst"
    : `${Math.round(projection.expected_minutes)} xMin`;
}

function lineupRole(player: Player, lineup: GameweekLineup) {
  if (player.id === lineup.captain_id) return "Kaptajn";
  if (player.id === lineup.vice_captain_id) return "Vicekaptajn";
  if (lineup.starting_ids.includes(player.id)) return "Start-XI";
  const benchIndex = lineup.bench_ids.indexOf(player.id);
  return benchIndex >= 0 ? `Bænk ${benchIndex + 1}` : "Ikke udtaget";
}

function confidenceLabel(confidence: number) {
  if (confidence >= 0.72) return "Høj";
  if (confidence >= 0.42) return "Middel";
  return "Lav";
}

function lineupPlayers(
  squad: Player[],
  lineup: GameweekLineup,
  role: "starter" | "bench",
) {
  const ids = role === "starter" ? lineup.starting_ids : lineup.bench_ids;
  const byId = new Map(squad.map((player) => [player.id, player]));
  return ids.flatMap((id, index) => {
    const player = byId.get(id);
    if (!player) return [];
    return [{
      ...player,
      role,
      is_captain: id === lineup.captain_id,
      is_vice_captain: id === lineup.vice_captain_id,
      bench_order: role === "bench" ? index + 1 : null,
    }];
  });
}

function SourcePill({ source }: { source: string }) {
  const isSolio = sourceKind(source) === "solio";
  return (
    <span className={classNames("source-pill", isSolio ? "source-pill--solio" : "source-pill--internal")}>
      <span className="source-pill__dot" aria-hidden="true" />
      {isSolio ? "Solio" : "Intern"}
    </span>
  );
}

function PlayerTile({ player, gameweek }: { player: Player; gameweek: number }) {
  const leadProjection = projectionFor(player, gameweek);
  return (
    <article
      className={classNames("player-tile", player.is_captain && "player-tile--captain")}
      aria-label={`${player.name}, ${player.team}, ${player.position}${
        player.is_captain ? ", kaptajn" : player.is_vice_captain ? ", vicekaptajn" : ""
      }`}
    >
      <div className="player-tile__top">
        <span className="player-tile__team">{player.team}</span>
        {player.is_captain && <span className="role-badge role-badge--captain">C</span>}
        {player.is_vice_captain && <span className="role-badge">VC</span>}
      </div>
      <strong className="player-tile__name">{player.name}</strong>
      <div className="player-tile__meta">
        <span>{formatPrice(player.price)}</span>
        <span aria-label={`${formatPoints(leadProjection?.ep ?? 0)} forventede point i gameweek ${gameweek}`}>
          {formatPoints(leadProjection?.ep ?? 0)} EP
        </span>
      </div>
      {leadProjection && (
        <span className="player-tile__forecast">
          {expectedMinutesLabel(leadProjection)} · {confidenceLabel(leadProjection.confidence)} sikkerhed
        </span>
      )}
      {leadProjection && <span className={classNames("player-tile__source", `player-tile__source--${sourceKind(leadProjection.source)}`)} />}
    </article>
  );
}

function Pitch({ starters, gameweek }: { starters: Player[]; gameweek: number }) {
  const rows = (["FWD", "MID", "DEF", "GKP"] as const)
    .map((position) => ({
      position,
      players: starters
        .filter((player) => player.position === position)
        .sort((a, b) => b.weighted_ep - a.weighted_ep),
    }))
    .filter((row) => row.players.length > 0);

  return (
    <section className="pitch" aria-label="Anbefalet startopstilling">
      <div className="pitch__markings" aria-hidden="true">
        <span className="pitch__centre-line" />
        <span className="pitch__centre-circle" />
        <span className="pitch__box pitch__box--top" />
        <span className="pitch__box pitch__box--bottom" />
      </div>
      <div className="pitch__content">
        {rows.map((row) => (
          <div className="pitch-row" key={row.position}>
            <span className="pitch-row__label">{positionNames[row.position]}</span>
            <div className="pitch-row__players">
              {row.players.map((player) => (
                <PlayerTile key={player.id} player={player} gameweek={gameweek} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function Bench({ players, gameweek }: { players: Player[]; gameweek: number }) {
  const sorted = [...players].sort((a, b) => (a.bench_order ?? 99) - (b.bench_order ?? 99));
  return (
    <section className="bench-section" aria-labelledby="bench-heading">
      <div className="section-heading section-heading--compact">
        <div>
          <p className="eyebrow">Reserver</p>
          <h3 id="bench-heading">Bænken</h3>
        </div>
        <p>Prioriteret fra venstre · målmanden står sidst</p>
      </div>
      <div className="bench-grid">
        {sorted.map((player, index) => (
          <article className="bench-card" key={player.id}>
            <span className="bench-card__order">{index + 1}</span>
            <div className="bench-card__identity">
              <span className="position-chip">{player.position}</span>
              <div>
                <strong>{player.name}</strong>
                <span>{player.team}</span>
              </div>
            </div>
            <div className="bench-card__numbers">
              <strong>{formatPoints(projectionFor(player, gameweek)?.ep ?? 0)} EP</strong>
              <span>{projectionFor(player, gameweek) ? expectedMinutesLabel(projectionFor(player, gameweek)!) : "Ingen prognose"} · {formatPrice(player.price)}</span>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function MetricCard({
  label,
  value,
  detail,
  featured = false,
}: {
  label: string;
  value: string;
  detail: string;
  featured?: boolean;
}) {
  return (
    <article className={classNames("metric-card", featured && "metric-card--featured")}>
      <span className="metric-card__label">{label}</span>
      <strong>{value}</strong>
      <span className="metric-card__detail">{detail}</span>
    </article>
  );
}

function signedPoints(value: number) {
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${formatPoints(value)}`;
}

function plannerActionLabel(action: PlannerAction) {
  if (action.kind === "roll") return "Rul transferen";
  const moves = action.transfers
    .map((transfer) => `${transfer.out.name} → ${transfer.in.name}`)
    .join(" + ");
  return action.hit_points > 0 ? `${moves} (−${action.hit_points})` : moves;
}

function TransferDecision({ action, horizon }: { action: PlannerAction; horizon: number }) {
  const title = action.kind === "roll"
    ? "Rul transferen"
    : action.transfer_count === 1
      ? "Lav én transfer"
      : `Lav ${action.transfer_count} transfers`;
  return (
    <section className={classNames("transfer-decision", `transfer-decision--${action.kind}`)} aria-labelledby="decision-heading">
      <div className="transfer-decision__main">
        <p className="transfer-decision__eyebrow">Anbefalet træk</p>
        <h3 id="decision-heading">{title}</h3>
        {action.transfers.length > 0 && (
          <div className="transfer-board">
            {action.transfers.flatMap((transfer) => [
              <div className="transfer-move" key={`out-${transfer.out_id}`}>
                <span className="transfer-move__direction transfer-move__direction--out">UD</span>
                <strong>{transfer.out.name}</strong>
                <small>{formatPrice(transfer.out_selling_price_tenths / 10)}</small>
              </div>,
              <div className="transfer-move" key={`in-${transfer.in_id}`}>
                <span className="transfer-move__direction transfer-move__direction--in">IND</span>
                <strong>{transfer.in.name}</strong>
                <small>{formatPrice(transfer.in_price_tenths / 10)}</small>
              </div>,
            ])}
          </div>
        )}
        <p className="transfer-decision__explanation">{action.explanation}</p>
      </div>
      <div className="transfer-decision__score">
        <strong>{action.kind === "roll" ? `FT ${action.free_transfers_next_gameweek}` : `${signedPoints(action.net_points_vs_roll)} EP`}</strong>
        <span>{action.kind === "roll" ? "næste gameweek" : `nettogevinst mod rul over ${horizon} GW`}</span>
        <div className="transfer-decision__facts">
          <span>{action.hit_points === 0 ? "Intet hit" : `−${action.hit_points} point i hit`}</span>
          <span>{formatPrice(action.bank_after_tenths / 10)} tilbage</span>
          <span>{action.free_transfers_next_gameweek} FT næste runde</span>
        </div>
      </div>
    </section>
  );
}

function AiReviewList({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="ai-review__list">
      <h4>{title}</h4>
      {items.length > 0
        ? <ul>{items.map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}</ul>
        : <p>Ingen konkrete punkter i den aktuelle review.</p>}
    </div>
  );
}

function AiReviewPanel({
  response,
  planner,
  deadline,
  isLoading,
  canReview,
  error,
  onReview,
}: {
  response: AiReviewResponse | null;
  planner: PlannerPayload;
  deadline: string | null;
  isLoading: boolean;
  canReview: boolean;
  error: string | null;
  onReview: () => void;
}) {
  const review = response?.review;
  const selectedAction = review ? actionForAiReview(planner, review) : null;
  const verdictLabel = review?.verdict === "confirm_best_action"
    ? "Bekræft planen"
    : review?.verdict === "wait_for_information"
      ? "Vent på nyt"
      : "Vælg plan B";
  const actionLabel = review?.verdict === "wait_for_information"
    ? "Vent og genberegn, når den manglende information er kendt"
    : selectedAction
      ? plannerActionLabel(selectedAction)
      : "Behold solverens plan";

  return (
    <section
      className={classNames(
        "ai-review",
        response && `ai-review--${response.review.verdict}`,
      )}
      aria-labelledby="ai-review-heading"
    >
      <div className="ai-review__rail" aria-hidden="true">
        <span>AI</span>
        <small>2. vurdering</small>
      </div>
      <div className="ai-review__body">
        <div className="ai-review__header">
          <div>
            <p className="eyebrow">Deadlinebrief · GW{planner.target_event}</p>
            <h2 id="ai-review-heading">Kvalificér næste træk med aktuel kontekst</h2>
            <p>En uafhængig reviewer udfordrer den beregnede plan med holdnyt, minutrisiko, prisvindue og de allerede løste alternativer.</p>
          </div>
          <button className="ai-review__button" type="button" onClick={onReview} disabled={isLoading || !canReview}>
            {isLoading
              ? <><span className="spinner spinner--button" /> Researcher …</>
              : <><SparkIcon /> {!canReview ? "Opdatér planen først" : response ? "Opdatér brief" : "Kvalificér planen"}</>}
          </button>
        </div>

        {error && (
          <div className="ai-review__error" role="alert">
            <InfoIcon />
            <p><strong>AI-reviewet stoppede</strong>{error} Den deterministiske plan ovenfor er stadig tilgængelig.</p>
          </div>
        )}

        {!response && !isLoading && (
          <div className="ai-review__empty">
            <div className="ai-review__empty-mark"><SparkIcon /></div>
            <div>
              <strong>Få en second opinion før deadline</strong>
              <p>Reviewet sender kun en begrænset fodboldkontekst til OpenAI. OpenAI modtager ikke GitHub-identitet, manager-ID, sessioner eller nøgler.</p>
            </div>
            <span>{deadline ? `Deadline ${formatDateTime(deadline)}` : "Næste deadline"}</span>
          </div>
        )}

        {isLoading && !response && (
          <div className="ai-review__loading" role="status">
            <span className="spinner spinner--dark" />
            <p><strong>Kontrollerer planen</strong>Sammenholder solverens resultat med aktuelle, kildebegrænsede nyheder.</p>
          </div>
        )}

        {response && review && (
          <div className={classNames("ai-review__result", isLoading && "ai-review__result--updating")}>
            <div className="ai-review__verdict">
              <span>{verdictLabel}</span>
              <div>
                <p>AI-kvalificeret handling</p>
                <strong>{actionLabel}</strong>
              </div>
              <small>{review.confidence === "high" ? "Høj" : review.confidence === "medium" ? "Middel" : "Lav"} sikkerhed</small>
            </div>

            <div className="ai-review__lead">
              <h3>{review.headline}</h3>
              <p>{review.summary}</p>
            </div>

            <div className="ai-review__grid">
              <AiReviewList title="Hvorfor nu" items={review.rationale} />
              <AiReviewList title="Risici" items={review.risks} />
              <AiReviewList title="Det ændrer rådet" items={review.change_triggers} />
            </div>

            <div className="ai-review__deadline">
              <div>
                <p className="eyebrow">Inden du trykker gem i FPL</p>
                <h3>Deadline-check</h3>
                <span>{deadline ? formatDateTime(deadline) : `GW${planner.target_event}`}</span>
              </div>
              <ol>
                {review.deadline_checklist.map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}
              </ol>
            </div>

            <div className="ai-review__evidence">
              <div>
                <p className="eyebrow">Aktuel research</p>
                <p>{review.evidence_summary}</p>
              </div>
              {response.research.sources.length > 0 ? (
                <div className="ai-review__sources" aria-label="Kilder til AI-reviewet">
                  {response.research.sources.map((source, index) => (
                    <a href={source.url} key={source.url} target="_blank" rel="noopener noreferrer">
                      <span>{index + 1}</span>{source.title}<ExternalLinkIcon />
                    </a>
                  ))}
                </div>
              ) : (
                <p className="ai-review__no-sources">Webresearchen returnerede ingen tilladte kildehenvisninger. Brug reviewet med ekstra forsigtighed.</p>
              )}
            </div>

            {review.data_gaps.length > 0 && (
              <div className="ai-review__gaps">
                <InfoIcon />
                <p><strong>Stadig ukendt</strong>{review.data_gaps.join(" · ")}</p>
              </div>
            )}
            <div className="ai-review__meta">
              <span>Research {formatDateTime(response.generated_at)}</span>
              <span>{response.model} · server-side · ingen automatisk transfer</span>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

function PlannerAlternatives({ actions }: { actions: PlannerAction[] }) {
  if (!actions.length) return null;
  return (
    <section className="panel alternatives-panel" aria-labelledby="alternatives-heading">
      <div className="section-heading section-heading--compact">
        <div>
          <p className="eyebrow">Plan B</p>
          <h2 id="alternatives-heading">Nærmeste alternativer</h2>
        </div>
        <p>Sammenlignet på samme horisont, budget og regelsæt.</p>
      </div>
      <div className="alternatives-list">
        {actions.map((action, index) => (
          <article className="alternative-card" key={`${action.kind}-${action.transfers.map((move) => `${move.out_id}-${move.in_id}`).join("-")}-${index}`}>
            <div className="alternative-card__top">
              <strong>{action.kind === "roll" ? "Rul transferen" : action.transfers.map((move) => `${move.out.name} → ${move.in.name}`).join(" + ")}</strong>
              <small>{signedPoints(action.net_points_vs_roll)} EP</small>
            </div>
            <p>{action.hit_points ? `${action.hit_points} point i hit · ` : "Intet hit · "}{formatPrice(action.bank_after_tenths / 10)} tilbage</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function ForecastTable({ data, gameweek }: { data: RecommendationResponse; gameweek: number }) {
  const gameweeks = data.meta.gameweek_window;
  const selectedLineup = data.team.gameweeks.find((lineup) => lineup.gameweek === gameweek) ?? data.team.gameweeks[0];
  const players = [...data.team.squad].sort((a, b) => {
    const aStarts = selectedLineup.starting_ids.includes(a.id);
    const bStarts = selectedLineup.starting_ids.includes(b.id);
    if (aStarts !== bStarts) return aStarts ? -1 : 1;
    return b.weighted_ep - a.weighted_ep;
  });

  return (
    <section className="panel forecast-panel" aria-labelledby="forecast-heading">
      <div className="section-heading section-heading--with-action">
        <div>
          <p className="eyebrow">Transparens</p>
          <h2 id="forecast-heading">Spillerprognoser</h2>
          <p>Se pointestimat, forventede minutter og usikkerhed for hver gameweek.</p>
        </div>
        <span className="row-count">15 spillere</span>
      </div>
      <div className="table-scroll" tabIndex={0} aria-label="Vandret scrollbar til spillerprognoser">
        <table>
          <thead>
            <tr>
              <th scope="col">Spiller</th>
              <th scope="col">Rolle i GW{selectedLineup.gameweek}</th>
              <th scope="col">Pris</th>
              {gameweeks.map((gameweek) => (
                <th scope="col" key={gameweek}>
                  GW{gameweek}
                </th>
              ))}
              <th scope="col">Vægtet EP</th>
              <th scope="col">Primær kilde</th>
            </tr>
          </thead>
          <tbody>
            {players.map((player) => {
              const primarySource = projectionFor(player, selectedLineup.gameweek)?.source ?? player.projections[0]?.source ?? "Intern baseline";
              const role = lineupRole(player, selectedLineup);
              return (
                <tr key={player.id}>
                  <th scope="row">
                    <span className="table-player">
                      <span className="position-chip">{player.position}</span>
                      <span>
                        <strong>{player.name}</strong>
                        <small>{player.team}</small>
                      </span>
                    </span>
                  </th>
                  <td>
                    <span className={classNames("role-label", selectedLineup.starting_ids.includes(player.id) && "role-label--start")}>
                      {role}
                    </span>
                  </td>
                  <td>{formatPrice(player.price)}</td>
                  {gameweeks.map((gameweek) => {
                    const projection = player.projections.find((item) => item.gameweek === gameweek);
                    return (
                      <td key={gameweek}>
                        <span className="projection-cell projection-cell--stacked">
                          <span>
                            <strong>{projection ? formatPoints(projection.ep) : "–"}</strong>
                            {projection && (
                              <span
                                className={classNames(
                                  "projection-cell__source",
                                  sourceKind(projection.source) === "solio" && "projection-cell__source--solio",
                                )}
                                aria-hidden="true"
                                title={projection.source}
                              />
                            )}
                          </span>
                          {projection && (
                            <small title={`${Math.round(projection.appearance_probability * 100)}% sandsynlighed for minutter`}>
                              {expectedMinutesLabel(projection)} · {confidenceLabel(projection.confidence)}
                            </small>
                          )}
                          {projection && (
                            <span className="sr-only">
                              {sourceKind(projection.source) === "solio" ? "Solio-kilde" : "Intern kilde"}
                            </span>
                          )}
                        </span>
                      </td>
                    );
                  })}
                  <td className="weighted-cell">{formatPoints(player.weighted_ep)}</td>
                  <td>
                    <SourcePill source={primarySource} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="table-legend" aria-label="Forklaring af datakilder">
        <span><i className="legend-dot" /> Intern {data.meta.forecast_version === "v2" ? "minutjusteret v2" : "legacy-baseline"}</span>
        <span>xMin = forventede spilleminutter · sikkerhed afspejler datamængde og rollevished</span>
      </div>
      <div className={classNames("model-status", data.meta.validation.status !== "validated" && "model-status--warning")}>
        <InfoIcon />
        <p><strong>Modelstatus: {data.meta.validation.status}</strong>{data.meta.validation.message}</p>
      </div>
    </section>
  );
}

function ProgressBar({ value, max, label }: { value: number; max: number; label: string }) {
  const percentage = max > 0 ? Math.min(100, Math.max(0, (value / max) * 100)) : 0;
  return (
    <div className="progress" aria-label={`${label}: ${value} af ${max}`}>
      <div className="progress__track" aria-hidden="true">
        <span style={{ width: `${percentage}%` }} />
      </div>
      <span>{compactNumber.format(percentage)}%</span>
    </div>
  );
}

function DataCoverage({ data }: { data: RecommendationResponse }) {
  const { fpl, price_signals: priceSignals } = data.meta.data_sources;
  const priceSignalPlayers = data.team.squad.filter((player) => player.price_signal !== null).length;

  return (
    <section className="panel coverage-panel" aria-labelledby="coverage-heading">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Datakvalitet</p>
          <h2 id="coverage-heading">Dækning og aktualitet</h2>
          <p>Et ærligt billede af, hvad anbefalingen bygger på.</p>
        </div>
      </div>
      <div className="coverage-grid">
        <article className="coverage-card">
          <div className="coverage-card__header">
            <span className="coverage-icon"><UsersIcon /></span>
            <span className="status-badge status-badge--ok"><CheckIcon /> FPL live</span>
          </div>
          <h3>Officielle FPL-data</h3>
          <strong>{fpl.eligible_count} <small>valgbare spillere</small></strong>
          <ProgressBar value={fpl.eligible_count} max={fpl.player_count} label="Valgbare FPL-spillere" />
          <p>{fpl.player_count} spillere i datasættet · {fpl.optimizer_candidate_count} i det auditerbare MILP-shortlist.</p>
        </article>

        <article className="coverage-card">
          <div className="coverage-card__header">
            <span className="coverage-icon coverage-icon--accent"><DatabaseIcon /></span>
            <span className={classNames("status-badge", priceSignals.available ? "status-badge--ok" : "status-badge--warning")}>
              {priceSignals.available ? <CheckIcon /> : <InfoIcon />}
              {priceSignals.available ? "FPL live" : "Ikke tilgængelig"}
            </span>
          </div>
          <h3>Officielle prissignaler</h3>
          <strong>{priceSignalPlayers} <small>/15 i truppen</small></strong>
          <ProgressBar value={priceSignalPlayers} max={15} label="Spillere med officielle prissignaler" />
          <p>Aktuel pris, nettotransfers og FPLs egne prisændringsfelter. Sandsynlighedskoder fortolkes ikke.</p>
        </article>

        <article className="coverage-card coverage-card--dark">
          <div className="coverage-card__header">
            <span className="coverage-icon coverage-icon--dark"><SparkIcon /></span>
            <span className="status-badge status-badge--dark">Eksakt solve</span>
          </div>
          <h3>{data.planner ? "Transferplaner" : "Trupoptimering"}</h3>
          <strong>{data.planner ? data.planner.method.candidate_count : fpl.optimizer_candidate_count}<small> kandidater</small></strong>
          <div className="source-split" aria-label="Optimeringen er gennemført">
            <span style={{ width: "100%" }} />
          </div>
          <p>{data.planner
            ? `Rul og op til ${data.planner.method.maximum_immediate_transfers} transfers er sammenlignet med bank, salgspriser og hits.`
            : "En lovlig 15-mandstrup er løst inden for budget og klubkvoter."}</p>
        </article>
      </div>
      {priceSignals.warning && (
        <div className="inline-notice inline-notice--warning">
          <InfoIcon />
          <p><strong>Prissignaler</strong>{priceSignals.warning}</p>
        </div>
      )}
      <div className="freshness-row">
        <span><span className="live-dot" /> Beregnet {formatDateTime(data.meta.generated_at)}</span>
        {priceSignals.price_change_deadlines?.[0] && <span>Næste prisvindue {formatDateTime(priceSignals.price_change_deadlines[0])}</span>}
      </div>
    </section>
  );
}

function ResultSkeleton() {
  return (
    <div className="result-skeleton" aria-live="polite" aria-label="Beregner holdforslag">
      <div className="skeleton-status"><span className="spinner" /> Henter live-data og løser holdoptimeringen …</div>
      <div className="skeleton-metrics">
        {[0, 1, 2, 3].map((item) => <span className="skeleton-block" key={item} />)}
      </div>
      <span className="skeleton-block skeleton-block--pitch" />
    </div>
  );
}

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <section className="error-state" role="alert">
      <span className="error-state__icon"><InfoIcon /></span>
      <div>
        <p className="eyebrow">Beregningen stoppede</p>
        <h2>Vi kunne ikke bygge holdet</h2>
        <p>{message}</p>
        <button className="secondary-button" type="button" onClick={onRetry}>
          <RefreshIcon /> Prøv igen
        </button>
      </div>
    </section>
  );
}

function EmptyState({
  onStart,
  mode,
  ready,
}: {
  onStart: () => void;
  mode: "weekly" | "initial";
  ready: boolean;
}) {
  return (
    <section className="empty-state">
      <span className="empty-state__icon"><SparkIcon /></span>
      <p className="eyebrow">{mode === "weekly" ? "Ugens beslutning" : "Klar til analyse"}</p>
      <h2>{mode === "weekly" ? "Planlæg dit næste FPL-træk" : "Dit første holdforslag er ét klik væk"}</h2>
      <p>{mode === "weekly"
        ? ready
          ? "Truppen er bekræftet. Sammenlign nu rul og mulige transfers med korrekte salgspriser og pointfradrag."
          : "Hent dit offentlige hold i venstre side, og bekræft bank samt frie transfers."
        : "Vi henter de seneste FPL-data, vurderer spillerpuljen og finder en lovlig 15-mandstrup."}</p>
      {ready && (
        <button className="primary-button" type="button" onClick={onStart}>
          {mode === "weekly" ? "Beregn næste træk" : "Beregn mit hold"} <ArrowRightIcon />
        </button>
      )}
    </section>
  );
}

export function FplDashboard({
  userName,
  initialManagerId,
}: {
  userName: string;
  initialManagerId: number | null;
}) {
  const [analysisMode, setAnalysisMode] = useState<"weekly" | "initial">("weekly");
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [appliedSettings, setAppliedSettings] = useState<Settings | null>(null);
  const [appliedMode, setAppliedMode] = useState<"weekly" | "initial" | null>(null);
  const [recommendation, setRecommendation] = useState<RecommendationResponse | null>(null);
  const [selectedGameweek, setSelectedGameweek] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [managerIdInput, setManagerIdInput] = useState(
    initialManagerId === null ? "" : String(initialManagerId),
  );
  const [managerSync, setManagerSync] = useState<ManagerSyncResponse | null>(null);
  const [bankInput, setBankInput] = useState("0,0");
  const [freeTransfers, setFreeTransfers] = useState(1);
  const [squadConfirmed, setSquadConfirmed] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [syncError, setSyncError] = useState<string | null>(null);
  const [aiReview, setAiReview] = useState<AiReviewResponse | null>(null);
  const [isAiReviewing, setIsAiReviewing] = useState(false);
  const [aiReviewError, setAiReviewError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const syncAbortRef = useRef<AbortController | null>(null);
  const aiAbortRef = useRef<AbortController | null>(null);
  const aiRequestKeyRef = useRef<string | null>(null);

  const clearAiReview = useCallback(() => {
    aiAbortRef.current?.abort();
    aiAbortRef.current = null;
    aiRequestKeyRef.current = null;
    setAiReview(null);
    setAiReviewError(null);
    setIsAiReviewing(false);
  }, []);

  const requestRecommendation = useCallback(async (
    requestSettings: Settings,
    plannerInput?: { sync: ManagerSyncResponse; state: ManualManagerState },
  ) => {
    clearAiReview();
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setIsLoading(true);
    setError(null);

    try {
      const requestBody = plannerInput
        ? serializePlannerRequest({
            manager_id: plannerInput.sync.manager.id,
            source_event: plannerInput.sync.last_deadline_state.event,
            state_fingerprint: plannerInput.sync.snapshot.checksum_sha256,
            manager_state: plannerInput.state,
            horizon: requestSettings.horizon,
            include_doubtful: requestSettings.includeDoubtful,
            forecast_version: requestSettings.forecastVersion,
          })
        : {
            horizon: requestSettings.horizon,
            include_doubtful: requestSettings.includeDoubtful,
            use_solio: false,
            forecast_version: requestSettings.forecastVersion,
          };
      const response = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(requestBody),
        cache: "no-store",
        signal: controller.signal,
      });
      const body: unknown = await response.json().catch(() => null);

      if (!response.ok) {
        const apiError = body as ApiError | null;
        throw new Error(apiError?.error?.message || `Serveren svarede med status ${response.status}.`);
      }
      if (!isRecommendation(body)) {
        throw new Error("Serveren returnerede et uventet dataformat.");
      }

      setRecommendation(body);
      setSelectedGameweek(body.team.gameweeks[0]?.gameweek ?? body.meta.gameweek_window[0] ?? null);
      setAppliedSettings(requestSettings);
      setAppliedMode(plannerInput ? "weekly" : "initial");
    } catch (requestError) {
      if (requestError instanceof DOMException && requestError.name === "AbortError") return;
      setError(requestError instanceof Error ? requestError.message : "Der opstod en ukendt fejl.");
    } finally {
      if (abortRef.current === controller) {
        setIsLoading(false);
      }
    }
  }, [clearAiReview]);

  const syncManagerById = useCallback(async (managerId: number) => {
    clearAiReview();
    syncAbortRef.current?.abort();
    const controller = new AbortController();
    syncAbortRef.current = controller;
    setIsSyncing(true);
    setSyncError(null);
    try {
      const response = await fetch("/api/planner/sync", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ manager_id: managerId }),
        cache: "no-store",
        signal: controller.signal,
      });
      const body: unknown = await response.json().catch(() => null);
      if (!response.ok) {
        const apiError = body as ApiError | null;
        throw new Error(apiError?.error?.message || `Serveren svarede med status ${response.status}.`);
      }
      const parsed = parseManagerSyncResponse(body);
      setManagerSync(parsed);
      setBankInput((parsed.manual_state_template.state.bank_tenths / 10).toFixed(1).replace(".", ","));
      setFreeTransfers(parsed.manual_state_template.state.free_transfers);
      setSquadConfirmed(false);
      setRecommendation(null);
      setAppliedMode(null);
      rememberManagerId(parsed.manager.id);
    } catch (requestError) {
      if (requestError instanceof DOMException && requestError.name === "AbortError") return;
      setSyncError(requestError instanceof Error ? requestError.message : "Holdet kunne ikke hentes.");
    } finally {
      if (syncAbortRef.current === controller) setIsSyncing(false);
    }
  }, [clearAiReview]);

  useEffect(() => {
    const managerId = managerIdFromBrowser(initialManagerId);

    if (managerId !== null) {
      setManagerIdInput(String(managerId));
      void syncManagerById(managerId);
    }
    return () => {
      abortRef.current?.abort();
      syncAbortRef.current?.abort();
      aiAbortRef.current?.abort();
    };
  }, [initialManagerId, syncManagerById]);

  useEffect(() => {
    clearAiReview();
  }, [
    bankInput,
    clearAiReview,
    freeTransfers,
    managerIdInput,
    settings.forecastVersion,
    settings.horizon,
    settings.includeDoubtful,
    squadConfirmed,
  ]);

  async function syncManagerState() {
    let managerId: number;
    try {
      managerId = parseManagerId(managerIdInput);
    } catch (parseError) {
      setSyncError(parseError instanceof Error ? parseError.message : "Indtast et gyldigt FPL-team-ID.");
      return;
    }
    await syncManagerById(managerId);
  }

  function confirmedPlannerState(): ManualManagerState | null {
    if (!managerSync || !squadConfirmed) return null;
    try {
      return {
        ...managerSync.manual_state_template.state,
        current_squad_ids: [...managerSync.manual_state_template.state.current_squad_ids],
        bank_tenths: parseBankTenths(bankInput),
        free_transfers: parseFreeTransfers(String(freeTransfers)),
        player_prices: managerSync.manual_state_template.state.player_prices.map((price) => ({ ...price })),
        chips: { ...managerSync.manual_state_template.state.chips },
        no_active_chip_confirmed: true,
      };
    } catch (parseError) {
      setSyncError(parseError instanceof Error ? parseError.message : "Bekræft bank, frie transfers og chipstatus.");
      return null;
    }
  }

  const hasUnappliedChanges = useMemo(() => {
    if (!appliedSettings) return false;
    return (
      analysisMode !== appliedMode ||
      settings.horizon !== appliedSettings.horizon ||
      settings.includeDoubtful !== appliedSettings.includeDoubtful ||
      settings.forecastVersion !== appliedSettings.forecastVersion ||
      Boolean(recommendation?.planner && !squadConfirmed) ||
      (recommendation?.planner?.confirmed_state.bank_tenths !== undefined &&
        (() => {
          try {
            return parseBankTenths(bankInput) !== recommendation.planner?.confirmed_state.bank_tenths ||
              freeTransfers !== recommendation.planner?.confirmed_state.free_transfers;
          } catch {
            return true;
          }
        })())
    );
  }, [analysisMode, appliedMode, settings, appliedSettings, recommendation, bankInput, freeTransfers, squadConfirmed]);

  function runCurrentAnalysis() {
    if (analysisMode === "weekly") {
      const state = confirmedPlannerState();
      if (!managerSync || !state) return;
      void requestRecommendation(settings, { sync: managerSync, state });
      return;
    }
    void requestRecommendation(settings);
  }

  async function requestAiQualification() {
    if (!managerSync || !recommendation?.planner) return;
    if (!squadConfirmed) {
      setAiReviewError("Bekræft først, at truppen, banken, de frie transfers og chipstatus stadig er korrekte.");
      return;
    }
    if (hasUnappliedChanges) {
      setAiReviewError("Opdatér først den beregnede plan med dine nye indstillinger.");
      return;
    }

    aiAbortRef.current?.abort();
    const controller = new AbortController();
    aiAbortRef.current = controller;
    const requestKey = [
      recommendation.meta.generated_at,
      recommendation.planner.state_fingerprint,
      recommendation.planner.target_event,
    ].join(":");
    aiRequestKeyRef.current = requestKey;
    setIsAiReviewing(true);
    setAiReview(null);
    setAiReviewError(null);

    try {
      const requestBody = buildAiReviewRequest(managerSync, {
        ...recommendation,
        planner: recommendation.planner,
      });
      const response = await fetch("/api/ai-review", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(requestBody),
        cache: "no-store",
        signal: controller.signal,
      });
      const body: unknown = await response.json().catch(() => null);
      if (!response.ok) {
        const apiError = body as ApiError | null;
        throw new Error(apiError?.error?.message || `Serveren svarede med status ${response.status}.`);
      }
      const parsed = parseAiReviewResponse(body, recommendation.planner.alternatives.length);
      if (
        new Date(parsed.recommendation_generated_at).getTime() !== new Date(recommendation.meta.generated_at).getTime() ||
        parsed.target_event !== recommendation.planner.target_event
      ) {
        throw new Error("AI-reviewet matcher ikke længere den viste anbefaling.");
      }
      if (aiRequestKeyRef.current === requestKey) setAiReview(parsed);
    } catch (requestError) {
      if (requestError instanceof DOMException && requestError.name === "AbortError") return;
      if (aiRequestKeyRef.current === requestKey) {
        setAiReviewError(requestError instanceof Error ? requestError.message : "AI-reviewet kunne ikke hentes.");
      }
    } finally {
      if (aiAbortRef.current === controller) {
        setIsAiReviewing(false);
        aiAbortRef.current = null;
      }
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    runCurrentAnalysis();
  }

  function downloadJson() {
    if (!recommendation) return;
    const blob = new Blob([JSON.stringify(recommendation, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `fpl-holdforslag-gw${recommendation.meta.gameweek_window[0] ?? "na"}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }

  function selectAnalysisMode(mode: "weekly" | "initial") {
    clearAiReview();
    setAnalysisMode(mode);
    setRecommendation(null);
    setError(null);
    setAppliedMode(null);
  }

  const windowLabel = recommendation?.meta.gameweek_window.length
    ? `GW${recommendation.meta.gameweek_window[0]}–GW${recommendation.meta.gameweek_window.at(-1)}`
    : `${settings.horizon} gameweeks`;
  const activeLineup = recommendation?.team.gameweeks.find(
    (lineup) => lineup.gameweek === selectedGameweek,
  ) ?? recommendation?.team.gameweeks[0];
  const activeStarters = recommendation && activeLineup
    ? lineupPlayers(recommendation.team.squad, activeLineup, "starter")
    : recommendation?.team.starters ?? [];
  const activeBench = recommendation && activeLineup
    ? lineupPlayers(recommendation.team.squad, activeLineup, "bench")
    : recommendation?.team.bench ?? [];
  const recommendationIsStale = Boolean(error && recommendation);
  const hasFreeHitWarning = managerSync?.warnings.some(
    (warning) => warning.code === "free_hit_squad_is_temporary",
  ) ?? false;
  const plannerCanSubmit = Boolean(
    managerSync && squadConfirmed && !hasFreeHitWarning && !isSyncing,
  );

  return (
    <>
      <a className="skip-link" href="#main-content">Gå til indhold</a>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="FPL HoldPlanner, forsiden">
          <span className="brand__mark">F</span>
          <span><strong>FPL</strong> HoldPlanner</span>
        </a>
        <nav className="topbar__nav" aria-label="Hovednavigation">
          <a href="#hold">Hold</a>
          <a href="#data">Datakvalitet</a>
          <a href="#metode">Metode</a>
        </nav>
        <AccountControl userName={userName} />
      </header>

      <main id="main-content">
        <section className="hero" id="top">
          <div className="hero__copy">
            <p className="eyebrow eyebrow--lime"><span /> FPL 2026/27 · Beslutningsmotor</p>
            <h1>Byg et stærkere hold.<br /><em>På et bedre grundlag.</em></h1>
            <p className="hero__lead">
              Synkronisér din trup, bekræft bank og frie transfers, og få ét gennemsigtigt træk før næste deadline.
            </p>
            <div className="hero__trust">
              <span><ShieldIcon /> FPL-regler valideret</span>
              <span><DatabaseIcon /> Kilder synlige pr. spiller</span>
            </div>
          </div>
          <div className="hero__signal" aria-hidden="true">
            <span className="signal-orbit signal-orbit--one" />
            <span className="signal-orbit signal-orbit--two" />
            <div className="signal-card signal-card--primary"><strong>15</strong><span>spillere</span></div>
            <div className="signal-card signal-card--secondary"><SparkIcon /><span>Optimeret</span></div>
            <span className="signal-dot signal-dot--one" />
            <span className="signal-dot signal-dot--two" />
          </div>
        </section>

        <div className="workspace">
          <aside className="settings-panel" aria-labelledby="settings-heading">
            <form onSubmit={handleSubmit}>
              <div className="settings-panel__heading">
                <div>
                  <p className="eyebrow">Opsætning</p>
                  <h2 id="settings-heading">{analysisMode === "weekly" ? "Næste træk" : "Ny trup"}</h2>
                </div>
                <span className="settings-step">01</span>
              </div>

              <div className="planner-mode" aria-label="Vælg analysetype">
                <button
                  type="button"
                  aria-pressed={analysisMode === "weekly"}
                  onClick={() => selectAnalysisMode("weekly")}
                >
                  Mit hold
                </button>
                {initialManagerId === null && (
                  <button
                    type="button"
                    aria-pressed={analysisMode === "initial"}
                    onClick={() => selectAnalysisMode("initial")}
                  >
                    Byg ny trup
                  </button>
                )}
              </div>

              {analysisMode === "weekly" && !managerSync && (
                <section className="planner-state" aria-labelledby="sync-heading">
                  <div className="planner-state__header">
                    <div>
                      <p className="eyebrow">Trin 1</p>
                      <strong id="sync-heading">Hent sidste deadline</strong>
                    </div>
                  </div>
                  <div className="planner-field">
                    <label htmlFor="manager-id">FPL-team-ID</label>
                    <div className="planner-sync-row">
                      <input
                        id="manager-id"
                        type="text"
                        inputMode="numeric"
                        autoComplete="off"
                        value={managerIdInput}
                        readOnly={initialManagerId !== null}
                        onChange={(event) => {
                          setManagerIdInput(event.target.value);
                          setSyncError(null);
                        }}
                        placeholder="fx 1499152"
                        aria-invalid={Boolean(syncError)}
                      />
                      <button
                        className="secondary-button"
                        type="button"
                        onClick={() => void syncManagerState()}
                        disabled={isSyncing}
                      >
                        {isSyncing ? <span className="spinner" /> : <RefreshIcon />}
                        Hent
                      </button>
                    </div>
                    <small>{initialManagerId === null
                      ? "Nummeret står i adressen på din offentlige FPL-side."
                      : "Dit offentlige FPL-hold er knyttet til denne private app og hentes automatisk."}</small>
                  </div>
                  {syncError && <p className="planner-warning" role="alert"><InfoIcon /> {syncError}</p>}
                  <p className="planner-warning"><ShieldIcon /> Vi henter kun offentlige holddata. Vi beder aldrig om din FPL-adgangskode og foretager ingen transfers.</p>
                </section>
              )}

              {analysisMode === "weekly" && managerSync && (
                <section className="planner-state planner-state--synced" aria-labelledby="synced-heading">
                  <div className="planner-state__header">
                    <div>
                      <p className="eyebrow">Trin 2 · GW{managerSync.last_deadline_state.event} hentet</p>
                      <strong id="synced-heading">{managerSync.manager.team_name}</strong>
                      <small>Deadline GW{managerSync.target.event}: {formatDateTime(managerSync.target.deadline_time)}</small>
                    </div>
                    <button className="download-button" type="button" onClick={() => void syncManagerState()} disabled={isSyncing}>
                      <RefreshIcon /> Hent igen
                    </button>
                  </div>

                  <div className="deadline-strip" aria-label="Status for ugens plan">
                    <span className="deadline-strip__step is-complete"><strong>1 · Hentet</strong><small>GW{managerSync.last_deadline_state.event}</small></span>
                    <span className={classNames("deadline-strip__step", squadConfirmed ? "is-complete" : "is-active")}><strong>2 · Bekræft</strong><small>Bank og FT</small></span>
                    <span className={classNames("deadline-strip__step", squadConfirmed && "is-active")}><strong>3 · Beregn</strong><small>Næste træk</small></span>
                  </div>

                  <div className="planner-confirm-grid">
                    <div className="planner-field">
                      <label htmlFor="bank-now">Bank nu (£m)</label>
                      <input
                        id="bank-now"
                        type="text"
                        inputMode="decimal"
                        value={bankInput}
                        onChange={(event) => {
                          setBankInput(event.target.value);
                          setSquadConfirmed(false);
                          setSyncError(null);
                        }}
                      />
                    </div>
                    <div className="planner-field">
                      <span>Frie transfers nu</span>
                      <div className="ft-selector" aria-label="Antal frie transfers">
                        {[1, 2, 3, 4, 5].map((value) => (
                          <button
                            type="button"
                            key={value}
                            aria-pressed={freeTransfers === value}
                            onClick={() => {
                              setFreeTransfers(value);
                              setSquadConfirmed(false);
                            }}
                          >
                            {value}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>

                  {hasFreeHitWarning && (
                    <p className="planner-warning" role="alert"><InfoIcon /> Sidste offentlige hold var et Free Hit-hold og er midlertidigt. Denne version kan ikke sikkert genskabe den permanente trup endnu.</p>
                  )}
                  {!hasFreeHitWarning && (
                    <label className="planner-confirmation">
                      <input
                        type="checkbox"
                        checked={squadConfirmed}
                        onChange={(event) => {
                          setSquadConfirmed(event.target.checked);
                          setSyncError(null);
                        }}
                      />
                      <span>Jeg bekræfter, at truppen og spillerpriserne er uændrede siden GW{managerSync.last_deadline_state.event}, at bank samt frie transfers er korrekte, og at ingen chip er aktiv til næste deadline.</span>
                    </label>
                  )}
                  {syncError && <p className="planner-warning" role="alert"><InfoIcon /> {syncError}</p>}
                </section>
              )}

              <fieldset className="control-group">
                <legend>Horisont</legend>
                <p>Hvor langt frem skal holdet vægtes?</p>
                <div className="horizon-selector">
                  {([1, 2, 3, 4, 5] as Horizon[]).map((value) => (
                    <label key={value}>
                      <input
                        type="radio"
                        name="horizon"
                        value={value}
                        checked={settings.horizon === value}
                        onChange={() => setSettings((current) => ({ ...current, horizon: value }))}
                      />
                      <span>{value}</span>
                    </label>
                  ))}
                </div>
                <span className="control-hint">gameweeks</span>
              </fieldset>

              <fieldset className="control-group">
                <legend>Prognosemodel</legend>
                <p>V2 dæmper små stikprøver og modellerer spilletid.</p>
                <div className="model-selector">
                  <label>
                    <input
                      type="radio"
                      name="forecast-version"
                      checked={settings.forecastVersion === "v2"}
                      onChange={() => setSettings((current) => ({ ...current, forecastVersion: "v2" }))}
                    />
                    <span><strong>V2</strong><small>Anbefalet</small></span>
                  </label>
                  <label>
                    <input
                      type="radio"
                      name="forecast-version"
                      checked={settings.forecastVersion === "legacy"}
                      onChange={() => setSettings((current) => ({ ...current, forecastVersion: "legacy" }))}
                    />
                    <span><strong>Baseline</strong><small>Til sammenligning</small></span>
                  </label>
                </div>
              </fieldset>

              <div className="toggle-list">
                <label className="toggle-control">
                  <span>
                    <strong>Tvivlsomme spillere</strong>
                    <small>Medtag status “doubtful”</small>
                  </span>
                  <input
                    type="checkbox"
                    checked={settings.includeDoubtful}
                    onChange={(event) => setSettings((current) => ({ ...current, includeDoubtful: event.target.checked }))}
                  />
                  <span className="toggle" aria-hidden="true"><span /></span>
                </label>
              </div>

              <div className="rule-summary">
                <span><CheckIcon /> {analysisMode === "weekly" ? "Salgspriser valideret" : "£100,0m budget"}</span>
                <span><CheckIcon /> Maks. 3 pr. klub</span>
                <span><CheckIcon /> Lovlig 15-mandstrup</span>
              </div>

              <button
                className="primary-button primary-button--full"
                type="submit"
                disabled={isLoading || (analysisMode === "weekly" && !plannerCanSubmit)}
              >
                {isLoading
                  ? <><span className="spinner spinner--button" /> Sammenligner planer …</>
                  : <>{recommendation ? "Opdatér anbefaling" : analysisMode === "weekly" ? "Bekræft trup og beregn" : "Byg ny trup"}<ArrowRightIcon /></>}
              </button>
              {hasUnappliedChanges && !isLoading && (
                <p className="changed-hint" role="status"><span /> Indstillingerne er ændret</p>
              )}
            </form>

            <div className="settings-panel__footer">
              <InfoIcon />
              <p>Prognosen er eksperimentel og bør genberegnes efter skader, pressemøder og prisændringer.</p>
            </div>
          </aside>

          <div className="results" id="hold">
            {isLoading && !recommendation && <ResultSkeleton />}
            {error && !recommendation && <ErrorState message={error} onRetry={runCurrentAnalysis} />}
            {!isLoading && !error && !recommendation && (
              <EmptyState onStart={runCurrentAnalysis} mode={analysisMode} ready={analysisMode === "initial" || plannerCanSubmit} />
            )}

            {recommendation && (
              <div className={classNames("result-content", isLoading && "result-content--updating")}>
                {isLoading && (
                  <div className="updating-banner" role="status"><span className="spinner spinner--dark" /> Opdaterer anbefalingen med dine nye valg …</div>
                )}
                {error && <ErrorState message={error} onRetry={runCurrentAnalysis} />}

                <section className="result-header" aria-labelledby="team-heading">
                  <div>
                    <p className="eyebrow">{recommendation.planner ? `Deadline-plan · GW${recommendation.planner.target_event}` : "Anbefaling"} · {windowLabel}</p>
                    <h2 id="team-heading">{recommendation.planner ? "Din plan til næste deadline" : "Dit optimerede hold"}</h2>
                    <p>{recommendation.planner
                      ? "Transfer, start-XI, kaptajn og alternativer baseret på din bekræftede trup."
                      : "Start-XI, kaptajn og bænk inden for de officielle trupbegrænsninger."}</p>
                  </div>
                  <div className="result-actions">
                    <span className={classNames("live-status", recommendationIsStale && "live-status--stale")}>
                      <span /> {recommendationIsStale ? "Forældet resultat" : "Live-data"}
                    </span>
                    <button className="download-button" type="button" onClick={downloadJson}>
                      <DownloadIcon /> Download JSON
                    </button>
                  </div>
                </section>

                {recommendation.planner && (
                  <>
                    <TransferDecision action={recommendation.planner.best_action} horizon={recommendation.meta.horizon} />
                    <AiReviewPanel
                      response={aiReview}
                      planner={recommendation.planner}
                      deadline={managerSync?.target.deadline_time ?? null}
                      isLoading={isAiReviewing}
                      canReview={squadConfirmed && !hasUnappliedChanges && !isLoading}
                      error={aiReviewError}
                      onReview={() => void requestAiQualification()}
                    />
                  </>
                )}

                <div className="metrics-grid">
                  <MetricCard label="Vægtet modelscore" value={formatPoints(recommendation.summary.objective_points)} detail="XI + kaptajn + bænk" featured />
                  <MetricCard label="Holdpris" value={formatPrice(recommendation.summary.total_cost)} detail={`${formatPrice(recommendation.summary.bank)} i banken`} />
                  <MetricCard label="Formation" value={activeLineup?.formation ?? recommendation.summary.formation} detail={`Valgt for GW${activeLineup?.gameweek ?? recommendation.meta.gameweek_window[0]}`} />
                  <MetricCard label="Analysevindue" value={windowLabel} detail={`${recommendation.meta.horizon} vægtede gameweeks`} />
                </div>

                <section className="lineup-panel" aria-labelledby="lineup-heading">
                  <div className="section-heading section-heading--pitch">
                    <div>
                      <p className="eyebrow">Startopstilling</p>
                      <h2 id="lineup-heading">Anbefalet XI · GW{activeLineup?.gameweek ?? recommendation.meta.gameweek_window[0]}</h2>
                    </div>
                    <div className="captain-summary">
                      <span>Kaptajnbonus</span>
                      <strong>+{formatPoints(activeLineup?.projected_captain_bonus ?? recommendation.summary.projected_captain_bonus)} EP</strong>
                    </div>
                  </div>
                  <div className="gameweek-tabs" aria-label="Vælg gameweek-opstilling">
                    {recommendation.team.gameweeks.map((lineup) => (
                      <button
                        key={lineup.gameweek}
                        type="button"
                        aria-pressed={lineup.gameweek === activeLineup?.gameweek}
                        className={lineup.gameweek === activeLineup?.gameweek ? "is-active" : undefined}
                        onClick={() => setSelectedGameweek(lineup.gameweek)}
                      >
                        GW{lineup.gameweek}<small>{lineup.formation}</small>
                      </button>
                    ))}
                  </div>
                  <Pitch starters={activeStarters} gameweek={activeLineup?.gameweek ?? recommendation.meta.gameweek_window[0]} />
                  <Bench players={activeBench} gameweek={activeLineup?.gameweek ?? recommendation.meta.gameweek_window[0]} />
                </section>

                <ForecastTable
                  data={recommendation}
                  gameweek={activeLineup?.gameweek ?? recommendation.meta.gameweek_window[0]}
                />
                {recommendation.planner && <PlannerAlternatives actions={recommendation.planner.alternatives} />}
                <div id="data"><DataCoverage data={recommendation} /></div>

                <section className="method-panel" id="metode" aria-labelledby="method-heading">
                  <div className="method-panel__number">03</div>
                  <div className="method-panel__copy">
                    <p className="eyebrow eyebrow--lime">Sådan skal du læse resultatet</p>
                    <h2 id="method-heading">Beslutningsstøtte, ikke en facitliste</h2>
                    <p>{recommendation.experimental_notice}</p>
                    <div className="method-steps">
                      <span><strong>01</strong> Live FPL-data</span>
                      <i aria-hidden="true" />
                      <span><strong>02</strong> EP-projektioner</span>
                      <i aria-hidden="true" />
                      <span><strong>03</strong> Transferoptimering</span>
                    </div>
                  </div>
                </section>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="footer">
        <span><strong>FPL HoldPlanner</strong> · Privat beslutningsværktøj</span>
        <span>Overfør altid holdet manuelt i Fantasy Premier League.</span>
        <a href="https://github.com/EMWI2000/EPL" target="_blank" rel="noreferrer">
          GitHub <ExternalLinkIcon />
        </a>
      </footer>
    </>
  );
}
