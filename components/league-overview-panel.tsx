"use client";
import { useEffect, useState } from "react";
import type { LeagueOverview, LeagueRival, LeagueSeasonBatch } from "@/lib/league-overview";
import { LEAGUE_DETAIL_BATCH_SIZE, summarizeLeagueSeasonDetails, type LeagueDiagnosis, type LeaguePointComponents, type LeagueRoundDiagnosis } from "@/lib/league-diagnosis";
import styles from "./league-overview-panel.module.css";

const chipLabel = (chip: string) => ({ wildcard: "WC", bboost: "BB", freehit: "FH", "3xc": "TC" })[chip] ?? chip;
const chipsText = (chips: string[] | null) => chips === null ? "Ukendt beholdning" : chips.length ? chips.map(chipLabel).join(", ") : "Ingen tilbage";
const numberText = (value: number) => value.toLocaleString("da-DK", { maximumFractionDigits: 1 });
const signed = (value: number) => value > 0 ? `+${numberText(value)}` : numberText(value);
const gapText = (value: number) => value > 0 ? `Du tabte ${value} point` : value < 0 ? `Du hentede ${-value} point` : "I fik lige mange point";
const roleText = (roles: LeagueRival["roles"]) => roles?.map(role => ({ leader: "Fører", nearest: "Næste plads", podium: "Podium" })[role]).join(" · ");

function PointsTable({ own, rival, caption }: { own: LeaguePointComponents; rival: LeaguePointComponents; caption: string }) {
  const components = [
    ["Spillerpoint (én kopi)", own.core, rival.core], ["Kaptajnbonus", own.captain, rival.captain],
    ["Triple Captain, ekstra", own.triple_captain, rival.triple_captain],
    ["Bench Boost", own.bench_boost, rival.bench_boost], ["Transferhits", -own.hit_cost, -rival.hit_cost],
    ["Nettopoint", own.net, rival.net],
  ] as const;
  return <div className={styles.tableWrap}><table className={styles.pointTable}>
    <caption>{caption}</caption><thead><tr><th scope="col">Point fra</th><th scope="col">Dig</th><th scope="col">Rival</th><th scope="col">Forskel</th></tr></thead>
    <tbody>{components.map(([label, a, b]) => <tr key={label}><th scope="row">{label}</th><td>{a}</td><td>{b}</td><td>{signed(b - a)}</td></tr>)}</tbody>
  </table></div>;
}

function SeasonReview({ leagueId, rival, updatedAt }: { leagueId: number; rival: LeagueRival; updatedAt: string }) {
  const [open, setOpen] = useState(false);
  const [rounds, setRounds] = useState<Record<number, LeagueRoundDiagnosis>>({});
  const [period, setPeriod] = useState("season");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const season = rival.season;
  const start = season?.start_event, end = season?.end_event;
  useEffect(() => {
    if (!open || !start || !end) return;
    const controller = new AbortController();
    setRounds({}); setLoading(true); setError(null);
    const load = async () => {
      for (let from = start; from <= end; from += LEAGUE_DETAIL_BATCH_SIZE) {
        const to = Math.min(end, from + LEAGUE_DETAIL_BATCH_SIZE - 1);
        const params = new URLSearchParams({ league: String(leagueId), rival: String(rival.entry), from: String(from), to: String(to) });
        const response = await fetch(`/api/leagues?${params}`, { cache: "no-store", signal: controller.signal });
        if (!response.ok) throw new Error("Nogle rundedetaljer kunne ikke hentes. Prøv igen.");
        const batch = await response.json() as LeagueSeasonBatch;
        if (batch.league_id !== leagueId || batch.rival_entry !== rival.entry || batch.from_event !== from || batch.to_event !== to
          || !Array.isArray(batch.rounds) || batch.rounds.length !== to - from + 1
          || batch.rounds.some((r, i) => r.gameweek !== from + i || !["available", "unavailable"].includes(r.status))) {
          throw new Error("Rundedetaljerne svarede ikke til den valgte sammenligning.");
        }
        if (controller.signal.aborted) return;
        setRounds(previous => ({ ...previous, ...Object.fromEntries(batch.rounds.map(r => [r.gameweek, r])) }));
      }
    };
    void load().catch(e => { if (!controller.signal.aborted) setError(e instanceof Error ? e.message : "Pointregnskabet kunne ikke hentes."); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [open, start, end, leagueId, rival.entry, updatedAt, attempt]);
  if (!season || !end) return <p className={styles.note}>Der er endnu ingen afsluttede runder i ligaens periode.</p>;
  const expected = season.rounds.map(r => r.gameweek);
  const total = summarizeLeagueSeasonDetails(expected, Object.values(rounds));
  const chosen = period === "season" ? null : rounds[Number(period)];
  const own = period === "season" ? total.own : chosen?.own;
  const other = period === "season" ? total.rival : chosen?.rival;
  const available = expected.length - total.missing_gameweeks.length;
  return <details className={styles.diagnosis} onToggle={e => setOpen(e.currentTarget.open)}>
    <summary>GW{season.start_event}–{end}: {season.gap_change === null ? "Se sæsonens pointregnskab" : gapText(season.gap_change)}</summary>
    {open && <>
      <p>{season.matches_standings === true ? "Rundernes nettopoint stemmer med ligaens samlede stilling."
        : season.matches_standings === false ? "Rundernes sum matcher endnu ikke den aktuelle ligatabel. Tabellen kan være under opdatering."
          : "Perioden slutter ved seneste afsluttede og kontrollerede runde. Den aktuelle tabel kan medtage en igangværende runde."}</p>
      <label className={styles.period}>Vis point fra
        <select value={period} onChange={e => setPeriod(e.target.value)}>
          <option value="season">Hele perioden, GW{season.start_event}–{end}</option>
          {season.rounds.map(r => <option key={r.gameweek} value={r.gameweek}>GW{r.gameweek}{r.gap_change === null ? " · ukendt" : ` · ${signed(r.gap_change)}`}</option>)}
        </select>
      </label>
      <p className={styles.note} role="status">{available} af {expected.length} runder er fuldt afstemt.{loading ? " Henter flere rundedetaljer…" : ""}</p>
      {error && <p role="alert">{error}</p>}
      {own && other ? <PointsTable own={own} rival={other} caption={period === "season" ? `Samlet GW${season.start_event}–${end}` : `Pointregnskab GW${period}`} />
        : <p>{period === "season" ? "Den samlede opdeling vises, når alle runders data stemmer. Vælg en enkelt runde for at se de hentede detaljer." : "Denne rundes opdeling mangler eller kunne ikke afstemmes med FPL."}</p>}
      {chosen?.status === "available" && <>
        {chosen.core_player_gains.length > 0 && <p className={styles.note}>Største spillerbidrag til rivalens fordel: {chosen.core_player_gains.map(p => `${p.name} (${signed(p.gap_change)})`).join(", ")}.</p>}
        {chosen.core_player_losses.length > 0 && <p className={styles.note}>Største spillerbidrag til din fordel: {chosen.core_player_losses.map(p => `${p.name} (${signed(p.gap_change)})`).join(", ")}.</p>}
      </>}
      {!loading && total.missing_gameweeks.length > 0 && <>
        <p className={styles.note}>Uafstemte runder: {total.missing_gameweeks.map(g => `GW${g}`).join(", ")}.</p>
        <button type="button" className="download-button" onClick={() => setAttempt(n => n + 1)}>Prøv at hente manglende runder igen</button>
      </>}
      <div className={styles.tableWrap}><table className={styles.pointTable}>
        <caption>Udvikling runde for runde</caption><thead><tr><th scope="col">Runde</th><th scope="col">Dig</th><th scope="col">Rival</th><th scope="col">Forskel</th></tr></thead>
        <tbody>{season.rounds.map(r => <tr key={r.gameweek}><th scope="row">GW{r.gameweek}</th><td>{r.own_net ?? "Ukendt"}</td><td>{r.rival_net ?? "Ukendt"}</td><td>{r.gap_change === null ? "Ukendt" : signed(r.gap_change)}</td></tr>)}</tbody>
      </table></div>
      <p className={styles.note}>Plus i forskellen er til rivalens fordel. Spillerpoint medtager automatiske indskiftninger. Kaptajnbonus er anden kopi, Triple Captain tredje kopi. Wildcard og Free Hit indgår gennem den spillede trup; deres selvstændige effekt kan ikke isoleres.</p>
    </>}
  </details>;
}

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
        {league.field && <p className={styles.note}>{league.field.complete && league.field.median_points !== null
          ? `Hele feltet: ${league.field.observed_count} hold. Medianen er ${numberText(league.field.median_points)} point${league.field.median_gap === null ? "" : ` (${signed(league.field.median_gap)} i forhold til dig)`}. Medianen inkluderer dit hold.`
          : `Ligatabellen er ufuldstændig (${league.field.observed_count} hold hentet). Medianen kan ikke fastslås.`}
          {league.start_event ? ` Ligaen tæller fra GW${league.start_event}.` : " Ligaens startuge er ukendt."}</p>}
        {league.rivals.map(rival => <div className={styles.rival} key={rival.entry}>
          {rival.roles && <span className={styles.role}>{roleText(rival.roles)}</span>}
          <strong>#{rival.rank} {rival.team} · {rival.gap === null ? "Ukendt afstand" : `${rival.gap >= 0 ? "+" : ""}${rival.gap} point`}</strong>
          <span>Kaptajn i GW{data.source_event}: {rival.captain ?? "ukendt"}. Chips tilbage: {chipsText(rival.chips_remaining)}.</span>
          <span>Spillere, som ikke var på dit deadline-hold: {rival.different_players.join(", ") || (league.partial ? "ukendt" : "ingen")}.</span>
          {rival.season ? <SeasonReview leagueId={league.id} rival={rival} updatedAt={data.generated_at} /> : <RivalDiagnosis diagnosis={rival.diagnosis} />}
        </div>)}
        {league.partial && <p>Dele af ligadata mangler. Ukendt er ikke det samme som nul.</p>}
      </article>)}
      <p className={styles.note}>AI-reviewet henter frisk ligakontekst. Et pointgab er ikke i sig selv en grund til at tage hits eller bruge en chip.</p>
      {data.warnings.map(w => <p className={styles.note} key={w}>{w}</p>)}
    </>}
  </section>;
}
