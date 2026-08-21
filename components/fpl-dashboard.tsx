"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AccountControl } from "@/components/account-control";
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

type Settings = {
  horizon: Horizon;
  includeDoubtful: boolean;
  useSolio: boolean;
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
    };
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
  useSolio: true,
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
    !isRecord(meta.validation)
  ) {
    return false;
  }
  const { fpl, solio } = meta.data_sources;
  const hasValidShape = Boolean(
    typeof meta.generated_at === "string" &&
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
        <span><i className="legend-dot legend-dot--solio" /> Solio Analytics</span>
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
  const { fpl, solio } = data.meta.data_sources;
  const solioSquadPlayers = data.team.squad.filter((player) =>
    player.projections.some((projection) => sourceKind(projection.source) === "solio"),
  ).length;

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
            <span className={classNames("status-badge", solio.applied ? "status-badge--ok" : "status-badge--warning")}>
              {solio.applied ? <CheckIcon /> : <InfoIcon />}
              {solio.applied ? "Anvendt" : solio.requested ? "Ikke anvendt" : "Fravalgt"}
            </span>
          </div>
          <h3>Solio Analytics</h3>
          <strong>{solio.matched} <small>entydige matches</small></strong>
          <ProgressBar value={solio.matched} max={solio.usable} label="Matchede Solio-projektioner" />
          <p>{solio.usable} brugbare projektioner · {solio.unmatched} uden match.</p>
        </article>

        <article className="coverage-card coverage-card--dark">
          <div className="coverage-card__header">
            <span className="coverage-icon coverage-icon--dark"><SparkIcon /></span>
            <span className="status-badge status-badge--dark">Holdmix</span>
          </div>
          <h3>Projektionskilder i truppen</h3>
          <strong>{solioSquadPlayers}<small>/15 med Solio-signal</small></strong>
          <div className="source-split" aria-label={`${solioSquadPlayers} Solio-spillere og ${15 - solioSquadPlayers} interne spillere`}>
            <span style={{ width: `${(solioSquadPlayers / 15) * 100}%` }} />
          </div>
          <p>{15 - solioSquadPlayers} spillere bruger alene den interne baseline.</p>
        </article>
      </div>
      {solio.warning && (
        <div className="inline-notice inline-notice--warning">
          <InfoIcon />
          <p><strong>Bemærk om Solio</strong>{solio.warning}</p>
        </div>
      )}
      <div className="freshness-row">
        <span><span className="live-dot" /> Beregnet {formatDateTime(data.meta.generated_at)}</span>
        {solio.generated_at && <span>Solio genereret {formatDateTime(solio.generated_at)}</span>}
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

function EmptyState({ onStart }: { onStart: () => void }) {
  return (
    <section className="empty-state">
      <span className="empty-state__icon"><SparkIcon /></span>
      <p className="eyebrow">Klar til analyse</p>
      <h2>Dit første holdforslag er ét klik væk</h2>
      <p>Vi henter de seneste FPL-data, vurderer spillerpuljen og finder en lovlig 15-mandstrup.</p>
      <button className="primary-button" type="button" onClick={onStart}>
        Beregn mit hold <ArrowRightIcon />
      </button>
    </section>
  );
}

export function FplDashboard({ userName }: { userName: string }) {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [appliedSettings, setAppliedSettings] = useState<Settings | null>(null);
  const [recommendation, setRecommendation] = useState<RecommendationResponse | null>(null);
  const [selectedGameweek, setSelectedGameweek] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const requestRecommendation = useCallback(async (requestSettings: Settings) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({
          horizon: requestSettings.horizon,
          include_doubtful: requestSettings.includeDoubtful,
          use_solio: requestSettings.useSolio,
          forecast_version: requestSettings.forecastVersion,
        }),
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
    } catch (requestError) {
      if (requestError instanceof DOMException && requestError.name === "AbortError") return;
      setError(requestError instanceof Error ? requestError.message : "Der opstod en ukendt fejl.");
    } finally {
      if (abortRef.current === controller) {
        setIsLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    void requestRecommendation(DEFAULT_SETTINGS);
    return () => abortRef.current?.abort();
  }, [requestRecommendation]);

  const hasUnappliedChanges = useMemo(() => {
    if (!appliedSettings) return false;
    return (
      settings.horizon !== appliedSettings.horizon ||
      settings.includeDoubtful !== appliedSettings.includeDoubtful ||
      settings.useSolio !== appliedSettings.useSolio ||
      settings.forecastVersion !== appliedSettings.forecastVersion
    );
  }, [settings, appliedSettings]);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void requestRecommendation(settings);
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
              Kombinér officielle FPL-data, eksterne projektioner og eksakt optimering i ét gennemsigtigt holdforslag.
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
                  <h2 id="settings-heading">Din analyse</h2>
                </div>
                <span className="settings-step">01</span>
              </div>

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
                    <strong>Solio-projektioner</strong>
                    <small>Overlay til næste gameweek</small>
                  </span>
                  <input
                    type="checkbox"
                    checked={settings.useSolio}
                    onChange={(event) => setSettings((current) => ({ ...current, useSolio: event.target.checked }))}
                  />
                  <span className="toggle" aria-hidden="true"><span /></span>
                </label>
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
                <span><CheckIcon /> £100,0m budget</span>
                <span><CheckIcon /> Maks. 3 pr. klub</span>
                <span><CheckIcon /> Lovlig 15-mandstrup</span>
              </div>

              <button className="primary-button primary-button--full" type="submit" disabled={isLoading}>
                {isLoading ? <><span className="spinner spinner--button" /> Beregner …</> : <>{recommendation ? "Opdatér anbefaling" : "Beregn mit hold"}<ArrowRightIcon /></>}
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
            {error && !recommendation && <ErrorState message={error} onRetry={() => void requestRecommendation(settings)} />}
            {!isLoading && !error && !recommendation && <EmptyState onStart={() => void requestRecommendation(settings)} />}

            {recommendation && (
              <div className={classNames("result-content", isLoading && "result-content--updating")}>
                {isLoading && (
                  <div className="updating-banner" role="status"><span className="spinner spinner--dark" /> Opdaterer anbefalingen med dine nye valg …</div>
                )}
                {error && <ErrorState message={error} onRetry={() => void requestRecommendation(settings)} />}

                <section className="result-header" aria-labelledby="team-heading">
                  <div>
                    <p className="eyebrow">Anbefaling · {windowLabel}</p>
                    <h2 id="team-heading">Dit optimerede hold</h2>
                    <p>Start-XI, kaptajn og bænk inden for de officielle trupbegrænsninger.</p>
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
                <div id="data"><DataCoverage data={recommendation} /></div>

                <section className="method-panel" id="metode" aria-labelledby="method-heading">
                  <div className="method-panel__number">03</div>
                  <div className="method-panel__copy">
                    <p className="eyebrow eyebrow--lime">Sådan skal du læse resultatet</p>
                    <h2 id="method-heading">En beslutningsstøtte—ikke en facitliste</h2>
                    <p>{recommendation.experimental_notice}</p>
                    <div className="method-steps">
                      <span><strong>01</strong> Live FPL-data</span>
                      <i aria-hidden="true" />
                      <span><strong>02</strong> EP-projektioner</span>
                      <i aria-hidden="true" />
                      <span><strong>03</strong> Eksakt optimering</span>
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
