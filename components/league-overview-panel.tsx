"use client";
import { useEffect, useState } from "react";
import type { LeagueOverview } from "@/lib/league-overview";
import styles from "./league-overview-panel.module.css";

const chipLabel = (chip: string) => ({ wildcard: "WC", bboost: "BB", freehit: "FH", "3xc": "TC" })[chip] ?? chip;
const chipsText = (chips: string[] | null) => chips === null ? "Ukendt beholdning" : chips.length ? chips.map(chipLabel).join(", ") : "Ingen tilbage";
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
        <h3>{league.name}</h3><p>Nr. {league.rank} (før {league.previous_rank}). {league.leader_gap === null ? "Afstand ukendt." : league.leader_gap <= 0 ? "Du ligger på førstepladsen eller deler den." : `${league.leader_gap} point op til førstepladsen.`}</p>
        {league.rivals.map(rival => <div className={styles.rival} key={rival.entry}>
          <strong>#{rival.rank} {rival.team} · {rival.gap === null ? "Ukendt afstand" : `${rival.gap >= 0 ? "+" : ""}${rival.gap} point`}</strong>
          <span>Kaptajn i GW{data.source_event}: {rival.captain ?? "ukendt"}. Chips tilbage: {chipsText(rival.chips_remaining)}.</span>
          <span>Spillere, som ikke var på dit deadline-hold: {rival.different_players.join(", ") || (league.partial ? "ukendt" : "ingen")}.</span>
        </div>)}
        {league.partial && <p>Dele af ligadata mangler. Ukendt er ikke det samme som nul.</p>}
      </article>)}
      <p className={styles.note}>AI-reviewet henter frisk ligakontekst. Et pointgab er ikke i sig selv en grund til at tage hits eller bruge en chip.</p>
      {data.warnings.map(w => <p className={styles.note} key={w}>{w}</p>)}
    </>}
  </section>;
}
