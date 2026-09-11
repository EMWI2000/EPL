"use client";
import { useEffect, useState } from "react";
import type { LeagueOverview } from "@/lib/league-overview";
import type { LeagueDiagnosis } from "@/lib/league-diagnosis";
import styles from "./league-overview-panel.module.css";

const chipLabel = (chip: string) => ({ wildcard: "WC", bboost: "BB", freehit: "FH", "3xc": "TC" })[chip] ?? chip;
const chipsText = (chips: string[] | null) => chips === null ? "Ukendt beholdning" : chips.length ? chips.map(chipLabel).join(", ") : "Ingen tilbage";
const signed = (value: number) => value > 0 ? `+${value}` : String(value);
const gapText = (value: number) => value > 0 ? `Du tabte ${value} point` : value < 0 ? `Du hentede ${-value} point` : "I fik lige mange point";

function RivalDiagnosis({ diagnosis }: { diagnosis: LeagueDiagnosis | undefined }) {
  if (!diagnosis) return <p className={styles.note}>Pointregnskab mangler. Opdatér ligaerne for at prøve igen.</p>;
  const { latest, trend } = diagnosis;
  const available = latest.status === "available" && latest.own !== null && latest.rival !== null;
  const own = latest.own, rival = latest.rival;
  const rows = available && own && rival ? [
    { label: "Spillerpoint (én kopi)", own: own.core, rival: rival.core },
    { label: "Kaptajnbonus", own: own.captain, rival: rival.captain },
    { label: "Triple Captain, ekstra", own: own.triple_captain, rival: rival.triple_captain },
    { label: "Bench Boost", own: own.bench_boost, rival: rival.bench_boost },
    { label: "Transferhits", own: -own.hit_cost, rival: -rival.hit_cost },
    { label: "Nettopoint", own: own.net, rival: rival.net },
  ] : [];
  return <details className={styles.diagnosis}>
    <summary>{available && latest.gap_change !== null ? `GW${latest.gameweek}: ${gapText(latest.gap_change)}` : "Se pointregnskab og udvikling"}</summary>
    {trend.rounds.length > 0 ? <>
      <p>{trend.gap_change !== null ? `${gapText(trend.gap_change)} samlet mod dette hold i de viste runder.` : "Der mangler runder, så en samlet udvikling kan ikke beregnes."}</p>
      <p className={styles.note}>{trend.rounds.map(round => `GW${round.gameweek}: ${round.own_net} til dig, ${round.rival_net} til rivalen (${signed(round.gap_change)})`).join("; ")}.</p>
      {trend.missing_gameweeks.length > 0 && <p className={styles.note}>Manglende runder: {trend.missing_gameweeks.map(gw => `GW${gw}`).join(", ")}.</p>}
    </> : <p>Der mangler afsluttede, kontrollerede runder til en sammenligning.</p>}
    {available ? <>
      <div className={styles.tableWrap}>
        <table className={styles.pointTable}>
          <caption>Pointregnskab for GW{latest.gameweek}</caption>
          <thead><tr><th scope="col">Point fra</th><th scope="col">Dig</th><th scope="col">Rival</th><th scope="col">Forskel</th></tr></thead>
          <tbody>{rows.map(row => <tr key={row.label}><th scope="row">{row.label}</th><td>{row.own}</td><td>{row.rival}</td><td>{signed(row.rival - row.own)}</td></tr>)}</tbody>
        </table>
      </div>
      <p className={styles.note}>Plus i forskellen er point til rivalens fordel. Kaptajnbonus er den anden kopi af anførerens point; Triple Captain er den tredje. Spillerpoint medtager gennemførte automatiske indskiftninger.</p>
      {latest.core_player_gains.length > 0 && <p className={styles.note}>Største spillerbidrag til rivalens fordel: {latest.core_player_gains.map(p => `${p.name} (${signed(p.gap_change)})`).join(", ")}.</p>}
      {latest.core_player_losses.length > 0 && <p className={styles.note}>Største spillerbidrag til din fordel: {latest.core_player_losses.map(p => `${p.name} (${signed(p.gap_change)})`).join(", ")}.</p>}
      <p className={styles.note}>Spillerbidragene er uden ekstra kaptajn- og chippoint. Regnskabet stemmer med FPL's officielle rundepoint.</p>
    </> : <p>GW{latest.gameweek} kan ikke opdeles endnu: Runden er ikke afsluttet og kontrolleret, eller de officielle data mangler eller stemmer ikke overens.</p>}
    <p className={styles.note}>Dette viser udfaldet, ikke om en beslutning var god på forhånd. Effekten af Wildcard og Free Hit kan ikke isoleres. Runderne forklarer ikke nødvendigvis hele ligaafstanden, især hvis ligaen startede senere.</p>
  </details>;
}
export function LeagueOverviewPanel() {
  const [data, setData] = useState<LeagueOverview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setError(null);
    void fetch("/api/leagues", { cache: "no-store", signal: controller.signal }).then(async response => {
      if (!response.ok) throw new Error("Ligaerne kunne ikke hentes. Dine øvrige analyser kan stadig bruges.");
      const body = await response.json() as LeagueOverview;
      if (!Array.isArray(body.leagues) || !Number.isInteger(body.source_event)) throw new Error("Ugyldigt ligaoverblik.");
      if (!controller.signal.aborted) setData(body);
    }).catch(e => { if (!controller.signal.aborted) setError(e instanceof Error ? e.message : "Ligaerne kunne ikke hentes."); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [refresh]);
  return <section className={styles.panel} aria-labelledby="league-heading">
    <div className={styles.heading}><div><p className="eyebrow">Din ligakontekst</p><h2 id="league-heading">Afstanden og dine muligheder</h2></div>
      <button type="button" className="download-button" disabled={loading} onClick={() => setRefresh(n => n + 1)}>{loading ? "Henter ligaer…" : "Opdatér ligaer"}</button></div>
    {error && <p role="alert">{error}{data ? " Nedenstående er det senest hentede overblik." : ""}</p>}
    {data && <>
      <p>{data.own_points} point. Dine chips i den aktuelle sæsonhalvdel: {chipsText(data.own_chips_remaining)}. Modstanderhold fra GW{data.source_event}.</p>
      {data.leagues.length === 0 && <p>Ingen invitationelle klassiske ligaer fundet.</p>}
      {data.leagues.map(league => <article key={league.id} className={styles.league}>
        <h3>{league.name}</h3><p>Nr. {league.rank ?? "ukendt"} (før {league.previous_rank ?? "ukendt"}). {league.leader_gap === null ? "Afstand ukendt." : league.leader_gap <= 0 ? "Du ligger på førstepladsen eller deler den." : `${league.leader_gap} point op til førstepladsen.`}</p>
        {league.rivals.map(rival => <div className={styles.rival} key={rival.entry}>
          <strong>#{rival.rank} {rival.team} · {rival.gap === null ? "Ukendt afstand" : `${rival.gap >= 0 ? "+" : ""}${rival.gap} point`}</strong>
          <span>Kaptajn i GW{data.source_event}: {rival.captain ?? "ukendt"}. Chips tilbage: {chipsText(rival.chips_remaining)}.</span>
          <span>Spillere, som ikke var på dit deadline-hold: {rival.different_players.join(", ") || (league.partial ? "ukendt" : "ingen")}.</span>
          <RivalDiagnosis diagnosis={rival.diagnosis} />
        </div>)}
        {league.partial && <p>Dele af ligadata mangler. Ukendt er ikke det samme som nul.</p>}
      </article>)}
      <p className={styles.note}>AI-reviewet henter frisk ligakontekst. Et pointgab er ikke i sig selv en grund til at tage hits eller bruge en chip.</p>
      {data.warnings.map(w => <p className={styles.note} key={w}>{w}</p>)}
    </>}
  </section>;
}
