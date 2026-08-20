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

type Settings = {
  horizon: Horizon;
  includeDoubtful: boolean;
  useSolio: boolean;
};

type PlayerProjection = {
  gameweek: number;
  ep: number;
  source: string;
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

type RecommendationResponse = {
  meta: {
    generated_at: string;
    gameweek_window: number[];
    horizon: number;
    include_doubtful: boolean;
    use_solio_requested: boolean;
    data_sources: {
      fpl: {
        player_count: number;
        eligible_count: number;
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
      typeof value.source === "string",
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
  if (!isRecord(meta.data_sources) || !isRecord(meta.data_sources.fpl) || !isRecord(meta.data_sources.solio)) {
    return false;
  }
  const { fpl, solio } = meta.data_sources;
  return Boolean(
    typeof meta.generated_at === "string" &&
      Array.isArray(meta.gameweek_window) &&
      meta.gameweek_window.every(isFiniteNumber) &&
      isFiniteNumber(meta.horizon) &&
      isFiniteNumber(fpl.player_count) &&
      isFiniteNumber(fpl.eligible_count) &&
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
      typeof value.experimental_notice === "string",
  );
}

function sourceKind(source: string) {
  return source.toLocaleLowerCase("da-DK").includes("solio") ? "solio" : "internal";
}

function isStarter(player: Player) {
  return player.role.toLocaleLowerCase("da-DK").includes("start");
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

function PlayerTile({ player }: { player: Player }) {
  const leadProjection = player.projections[0];
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
        <span aria-label={`${formatPoints(player.weighted_ep)} vægtede forventede point`}>
          {formatPoints(player.weighted_ep)} EP
        </span>
      </div>
      {leadProjection && <span className={classNames("player-tile__source", `player-tile__source--${sourceKind(leadProjection.source)}`)} />}
    </article>
  );
}

function Pitch({ starters }: { starters: Player[] }) {
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
                <PlayerTile key={player.id} player={player} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function Bench({ players }: { players: Player[] }) {
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
              <strong>{formatPoints(player.weighted_ep)} EP</strong>
              <span>{formatPrice(player.price)}</span>
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

function ForecastTable({ data }: { data: RecommendationResponse }) {
  const gameweeks = data.meta.gameweek_window;
  const players = [...data.team.squad].sort((a, b) => {
    if (a.role !== b.role) return isStarter(a) ? -1 : 1;
    return b.weighted_ep - a.weighted_ep;
  });

  return (
    <section className="panel forecast-panel" aria-labelledby="forecast-heading">
      <div className="section-heading section-heading--with-action">
        <div>
          <p className="eyebrow">Transparens</p>
          <h2 id="forecast-heading">Spillerprognoser</h2>
          <p>Se pointestimat og datakilde for hver gameweek.</p>
        </div>
        <span className="row-count">15 spillere</span>
      </div>
      <div className="table-scroll" tabIndex={0} aria-label="Vandret scrollbar til spillerprognoser">
        <table>
          <thead>
            <tr>
              <th scope="col">Spiller</th>
              <th scope="col">Rolle</th>
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
              const primarySource = player.projections[0]?.source ?? "Intern baseline";
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
                    <span className={classNames("role-label", isStarter(player) && "role-label--start")}>
                      {player.is_captain
                        ? "Kaptajn"
                        : player.is_vice_captain
                          ? "Vicekaptajn"
                          : isStarter(player)
                            ? "Start-XI"
                            : `Bænk ${player.bench_order ?? ""}`}
                    </span>
                  </td>
                  <td>{formatPrice(player.price)}</td>
                  {gameweeks.map((gameweek) => {
                    const projection = player.projections.find((item) => item.gameweek === gameweek);
                    return (
                      <td key={gameweek}>
                        <span className="projection-cell">
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
        <span><i className="legend-dot" /> Intern eksperimentel baseline</span>
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
          <p>{fpl.player_count} spillere i det samlede datasæt.</p>
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
      settings.useSolio !== appliedSettings.useSolio
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
                    <span className="live-status"><span /> Live-data</span>
                    <button className="download-button" type="button" onClick={downloadJson}>
                      <DownloadIcon /> Download JSON
                    </button>
                  </div>
                </section>

                <div className="metrics-grid">
                  <MetricCard label="Vægtet modelscore" value={formatPoints(recommendation.summary.objective_points)} detail="XI + kaptajn + bænk" featured />
                  <MetricCard label="Holdpris" value={formatPrice(recommendation.summary.total_cost)} detail={`${formatPrice(recommendation.summary.bank)} i banken`} />
                  <MetricCard label="Formation" value={recommendation.summary.formation} detail="Valgt af optimeringen" />
                  <MetricCard label="Analysevindue" value={windowLabel} detail={`${recommendation.meta.horizon} vægtede gameweeks`} />
                </div>

                <section className="lineup-panel" aria-labelledby="lineup-heading">
                  <div className="section-heading section-heading--pitch">
                    <div>
                      <p className="eyebrow">Startopstilling</p>
                      <h2 id="lineup-heading">Anbefalet XI</h2>
                    </div>
                    <div className="captain-summary">
                      <span>Kaptajnbonus</span>
                      <strong>+{formatPoints(recommendation.summary.projected_captain_bonus)} EP</strong>
                    </div>
                  </div>
                  <Pitch starters={recommendation.team.starters} />
                  <Bench players={recommendation.team.bench} />
                </section>

                <ForecastTable data={recommendation} />
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
