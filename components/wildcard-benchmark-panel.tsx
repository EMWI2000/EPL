"use client";

import { useEffect, useId, useState } from "react";
import type { WildcardBenchmarkResponse } from "@/lib/wildcard-benchmark-loader";
import styles from "./wildcard-benchmark-panel.module.css";

const format = (value: number) => value.toLocaleString("da-DK", { maximumFractionDigits: 1 });
const signed = (value: number) => `${value > 0 ? "+" : ""}${format(value)}`;
const windowLabel = (horizon: number) => `${horizon} ${horizon === 1 ? "runde" : "runder"}`;
const ownStatus = {
  not_used: "Dit Wildcard er endnu ikke registreret som brugt i dette chipsæt.",
  pending: "Dit forløb mangler afsluttede runder i den valgte periode.",
  season_end: "Der er ikke nok runder tilbage i sæsonen til at måle hele den valgte periode efter dit Wildcard.",
  other_chip: "Du brugte også en anden chip i perioden. Det forløb indgår ikke i denne Wildcard-sammenligning.",
  missing_data: "Der mangler officielle rundedata til at vurdere dit forløb.",
  insufficient_history: "Der er for få runder før dit Wildcard til at finde sammenlignelige hold.",
  insufficient_peers: "Der er færre end fem egnede referencehold til dit forløb.",
  ready: "",
};

export function WildcardBenchmarkPanel({ targetEvent, half }: { targetEvent: number; half: 1 | 2 }) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const [horizon, setHorizon] = useState<1 | 3 | 5>(5);
  const [data, setData] = useState<WildcardBenchmarkResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setLoading(true); setData(null); setError(null);
    const load = async () => {
      const response = await fetch("/api/wildcard-benchmark", { cache: "no-store", signal: controller.signal });
      if (!response.ok) throw new Error("Sammenligningen kunne ikke hentes. Prøv igen om lidt.");
      const body = await response.json() as WildcardBenchmarkResponse;
      if (body.version !== "fpl-wildcard-benchmark-v1" || body.half !== half || body.target_event !== targetEvent || !Array.isArray(body.windows)
        || body.windows.length !== 3 || !body.coverage || !body.own || !Array.isArray(body.own.windows)) {
        throw new Error("Data matcher ikke det aktuelle chipsæt. Hent holdet igen, og prøv derefter sammenligningen.");
      }
      if (!controller.signal.aborted) setData(body);
    };
    void load().catch(reason => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Sammenligningen kunne ikke hentes.");
    }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [open, targetEvent, half, attempt]);

  const selected = data?.windows.find(window => window.horizon === horizon);
  const own = data?.own.windows.find(window => window.horizon === horizon);
  const difference = selected?.peer_difference;
  const canGrade = Boolean(data?.coverage.complete && selected && selected.sample_size >= 10 && difference);
  const missingSources = Boolean(data && (data.coverage.missing_histories > 0
    || data.coverage.leagues_complete < data.coverage.leagues_selected || data.invalid_peer_histories > 0));
  return <details className={styles.panel} onToggle={event => setOpen(event.currentTarget.open)}>
    <summary><span>Sammenlign med andre Wildcard-hold</span><small>Faktiske resultater efter 1, 3 og 5 runder</small></summary>
    {open && <div className={styles.body}>
      <header className={styles.header}><div>
        <p className={styles.eyebrow}>Efter Wildcard · {data?.season ?? `${half}. chipsæt`}</p>
        <h3 id={`${id}-heading`}>Sådan gik det for andre Wildcard-hold</h3>
      </div><button type="button" className={styles.refresh} disabled={loading} onClick={() => setAttempt(value => value + 1)}>
        {loading ? "Henter data…" : "Opdatér sammenligning"}
      </button></header>
      <p className={styles.intro}>Vi sammenligner Wildcard-hold fra de op til tre private ligaer i dit overblik med hold, der havde lignende pointform inden skiftet og spillede de samme runder uden chips.</p>
      {loading && <p className={styles.notice} role="status">Henter officielle rundehistorikker og beregner sammenlignelige forløb…</p>}
      {error && <p className={styles.error} role="alert">{error}</p>}
      {data && <>
        {data.coverage.complete && <p className={styles.sample}>Datagrundlaget dækker {data.coverage.histories_loaded} af {data.coverage.sampled_peers} andre hold og {data.coverage.leagues_complete} af {data.coverage.leagues_selected} udvalgte ligaer. Der er registreret {data.detected_wildcards} Wildcard-forløb i dette chipsæt før frasortering.</p>}
        {missingSources && <p className={styles.warning} role="status">Der mangler komplette kilder. {data.coverage.histories_loaded} af {data.coverage.sampled_peers} udvalgte holds historikker er hentet, og {data.coverage.leagues_complete} af {data.coverage.leagues_selected} ligatabeller er komplette. {data.invalid_peer_histories > 0 ? `${data.invalid_peer_histories} historikker kunne ikke valideres. ` : ""}Tallene er foreløbige, og vurderingen af et godt resultat er slået fra.</p>}
        {data.coverage.selection === "stable_sample" && <p className={styles.notice}>Udvalget omfatter {data.coverage.sampled_peers} af {data.coverage.discovered_peers} andre hold. Det er en fast stikprøve på tværs af ligaerne. Der vises derfor ingen grænse for et godt resultat i hele feltet.</p>}
        {!data.coverage.complete && !missingSources && data.coverage.selection === "all" && <p className={styles.notice}>Dækningen kan endnu ikke bekræftes for hele sammenligningen. Vurderingen af et godt resultat er slået fra.</p>}
        <fieldset className={styles.horizons}><legend>Periode fra og med Wildcard-runden</legend>
          {([1, 3, 5] as const).map(value => <label key={value}>
            <input type="radio" name={`${id}-horizon`} value={value} checked={horizon === value} onChange={() => setHorizon(value)} />
            <span>{windowLabel(value)}</span>
          </label>)}
        </fieldset>
        {selected && <>
          <p className={styles.sample}>{selected.sample_size} sammenlignelige Wildcard-forløb · {windowLabel(horizon)} · senest afsluttede GW{data.last_finalized_event ?? "ukendt"}</p>
          {difference ? <>
            <h4 className={styles.metricHeading}>Forskel til referencehold</h4>
            <dl className={styles.metrics}>
              <div><dt>{data.coverage.complete ? "Typisk resultat (median)" : "Median i det hentede udsnit"}</dt><dd>{signed(difference.median)} <small>point</small></dd></div>
              <div><dt>Midterste halvdel</dt><dd className={styles.range}>{signed(difference.p25)} til {signed(difference.p75)}</dd><small>25.–75. percentil</small></div>
              {canGrade && <div><dt>Bedste fjerdedel</dt><dd>{signed(difference.p75)} <small>point eller mere</small></dd></div>}
            </dl>
            <p className={styles.notice}>Et positivt tal betyder flere nettopoint end medianen af referenceholdene i de samme runder.</p>
            {selected.sample_size < 10 && <p className={styles.notice}>Færre end ti forløb giver et spinkelt grundlag. Derfor vises ingen grænse for et godt resultat.</p>}
          </> : <p className={styles.notice}>Der er endnu ikke fem sammenlignelige forløb med komplette data til den valgte periode.</p>}
          {selected.net_points && <p className={styles.secondary}>Wildcard-holdenes samlede nettopoint havde median {format(selected.net_points.median)} over {windowLabel(horizon)}. Den midterste halvdel lå mellem {format(selected.net_points.p25)} og {format(selected.net_points.p75)} point.</p>}
          {own && <div className={styles.own}>
            <h4>{data.own.used_event === null ? "Dit eget forløb" : `Dit Wildcard i GW${data.own.used_event}`}</h4>
            {own.status === "ready" && own.net_points !== null && own.peer_difference !== null ? <>
              <p><strong>{signed(own.peer_difference)} point</strong> i forhold til dine referencehold over {windowLabel(horizon)}. Dit hold scorede {format(own.net_points)} nettopoint.</p>
              {canGrade && own.percentile !== null && <p className={styles.notice}>Dit forløb ligger ved {format(own.percentile)}-percentilen blandt de sammenlignelige Wildcard-forløb. Ved ens resultater bruges midterplaceringen.</p>}
            </> : <p>{ownStatus[own.status]}</p>}
          </div>}
          <details className={styles.method}><summary>Sådan beregnes sammenligningen</summary>
            <p>Hvert forløb måles fra Wildcard-runden. Alle runder skal være afsluttet og kontrolleret af FPL. Vi bruger faktiske point efter transferfradrag med samme vægt på hver runde.</p>
            <p>Referenceholdene vælges efter gennemsnitlige nettopoint i op til tre runder før Wildcard, med mindst to tidligere runder. Vi vælger op til ti nærmeste hold inden for fem point pr. runde og kræver mindst fem. Referenceholdene må have brugt Wildcard tidligere, men ingen chip i selve sammenligningsperioden.</p>
            <p>Dit eget hold indgår hverken i referencegruppen eller fordelingen af andre Wildcard-forløb. Et Wildcard-forløb med en anden chip i perioden udelades.</p>
            <p>Udvalget kommer fra de første tre private ligaer i dit overblik. Sammenligningen beskriver resultaterne i dette felt.</p>
            {selected.reference_count && <p>For de medregnede forløb var der mellem {selected.reference_count.min} og {selected.reference_count.max} referencehold, med median {format(selected.reference_count.median)}.</p>}
            <ul>
              <li>{selected.exclusions.unfinished} forløb afventer afsluttede runder.</li>
              <li>{selected.exclusions.season_end} forløb ligger for sent til, at hele perioden kan nås inden sæsonslut.</li>
              <li>{selected.exclusions.other_chip} forløb havde en anden chip i perioden.</li>
              <li>{selected.exclusions.missing_data} forløb manglede rundedata.</li>
              <li>{selected.exclusions.insufficient_history} forløb havde for kort forhistorie.</li>
              <li>{selected.exclusions.insufficient_peers} forløb havde for få egnede referencehold.</li>
            </ul>
          </details>
        </>}
        <p className={styles.limit}>Resultatet omfatter også efterfølgende transfers, kaptajnvalg og kampudfald. Det viser ikke Wildcardets isolerede gevinst. Disse faktiske, uvægtede point kan ikke sammenlignes direkte med chipreglens 15 vægtede modelpoint.</p>
        <p className={styles.source}>Kilden er FPL’s offentlige liga- og rundehistorikker. Hentet {new Date(data.generated_at).toLocaleString("da-DK", { timeZone: "Europe/Copenhagen", dateStyle: "short", timeStyle: "short" })}.</p>
      </>}
    </div>}
  </details>;
}
