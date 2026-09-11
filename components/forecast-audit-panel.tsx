"use client";

import { useEffect, useState } from "react";
import {
  addForecastAuditResults,
  captureForecastAudit,
  emptyForecastAudit,
  evaluateForecastAudit,
  forecastAuditSeason,
  parseForecastAuditResults,
  readForecastAudit,
  writeForecastAudit,
  type ForecastAuditCapture,
  type ForecastAuditJournal,
} from "@/lib/forecast-audit";
import styles from "./forecast-audit-panel.module.css";

const format = (value: number) => value.toLocaleString("da-DK", { maximumFractionDigits: 2 });

export function ForecastAuditPanel({ capture }: { capture: ForecastAuditCapture | null }) {
  const [journal, setJournal] = useState<ForecastAuditJournal>(emptyForecastAudit);
  const [ready, setReady] = useState(false);
  const [storageError, setStorageError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    try {
      const result = readForecastAudit(window.localStorage);
      setJournal(result.journal);
      setStorageError(result.error);
    } catch {
      setStorageError("Browseren tillader ikke lokal lagring. Prognoserne bliver ikke gemt.");
    }
    setReady(true);
  }, []);

  useEffect(() => {
    if (!ready || storageError || !capture || busy) return;
    try {
      // Read again to preserve the first capture if another tab has already saved it.
      const current = readForecastAudit(window.localStorage);
      if (current.error) { setStorageError(current.error); return; }
      const next = captureForecastAudit(current.journal, capture);
      if (!writeForecastAudit(window.localStorage, next)) {
        setStorageError("Prognosen kunne ikke gemmes i browseren. Tidligere målinger er bevaret.");
        return;
      }
      setJournal(next);
    } catch {
      setStatus("Denne prognose blev ikke gemt. Der kræves en ny beregning før deadline og ledig plads i browseren.");
    }
  }, [capture, ready, storageError, busy]);

  async function refreshResults() {
    setBusy(true);
    setStatus(null);
    try {
      const current = readForecastAudit(window.localStorage);
      if (current.error) { setStorageError(current.error); return; }
      let next = current.journal;
      const now = Date.now();
      const season = forecastAuditSeason(new Date(now).toISOString());
      const pending = next.entries.filter((row) => row.results === null && Date.parse(row.target_deadline) < now && row.season === season).slice(0, 4);
      if (pending.length === 0) { setStatus("Ingen afsluttede, umålte runder i den aktuelle sæson. Prognoser gemmes først, når du beregner et hold før deadline."); return; }
      let completed = 0;
      let waiting = 0;
      for (const entry of pending) {
        const params = new URLSearchParams({ event: String(entry.target_event), deadline: entry.target_deadline });
        const response = await fetch(`/api/forecast-results?${params}`, { cache: "no-store", signal: AbortSignal.timeout(25_000) });
        const body: unknown = await response.json();
        if (!response.ok) {
          const code = (body as { error?: { code?: unknown } })?.error?.code;
          if (code === "results_pending") { waiting++; continue; }
          throw new Error(response.status === 401 ? "Log ind igen for at hente slutresultater. De gemte prognoser er bevaret." : "Resultater kunne ikke hentes. De gemte prognoser er bevaret; prøv igen senere.");
        }
        const results = parseForecastAuditResults(body);
        // Preserve forecasts saved in another tab while the request was in flight.
        const latest = readForecastAudit(window.localStorage);
        if (latest.error) { setStorageError(latest.error); return; }
        next = addForecastAuditResults(latest.journal, results);
        if (!writeForecastAudit(window.localStorage, next)) throw new Error("Slutresultatet kunne ikke gemmes lokalt. Den oprindelige prognose er bevaret.");
        setJournal(next);
        completed++;
      }
      setStatus(`${completed} runder målt.${waiting ? ` ${waiting} afventer FPL's kontrol af slutresultatet.` : ""}`);
    } catch (error) {
      setStatus(error instanceof Error && !["TimeoutError", "AbortError", "TypeError", "SyntaxError"].includes(error.name)
        ? error.message : "Resultater kunne ikke hentes. De gemte prognoser er bevaret; prøv igen senere.");
    } finally {
      setBusy(false);
    }
  }

  function download() {
    const blob = new Blob([JSON.stringify(journal, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `fpl-prognosemaaling-${new Date().toISOString().slice(0, 10)}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <section className={`panel ${styles.panel}`} aria-labelledby="forecast-audit-heading">
      <div className={styles.header}>
        <div>
          <h2 id="forecast-audit-heading">Holder prognoserne?</h2>
          <p className={styles.description}>Den første prognose for dit bekræftede hold gemmes automatisk før hver deadline. Efter runden måler vi appens fejl mod FPL&apos;s daværende pointprognose og et enkelt minutgennemsnit.</p>
        </div>
      </div>
      <div className={styles.actions}>
        <button type="button" className="secondary-button" disabled={!ready || busy || !!storageError || journal.entries.length === 0} onClick={refreshResults}>{busy ? "Henter slutresultater…" : "Hent slutresultater"}</button>
        <button type="button" className="secondary-button" disabled={journal.entries.length === 0 || busy} onClick={download}>Eksportér målinger</button>
      </div>
      {storageError && <p className={`${styles.notice} ${styles.error}`} role="alert">{storageError}</p>}
      {status && <p className={styles.notice} role="status">{status}</p>}
      {journal.entries.length === 0 ? <p className={styles.empty}>Ingen prognoser gemt endnu. Beregn din næste runde før deadline. Tidligere runder genskabes ikke bagefter, fordi det ville give et misvisende resultat.</p> : (
        <div className={styles.tableWrap} tabIndex={0} role="region" aria-label="Prognosefejl pr. spillerunde">
          <table className={styles.table}>
            <thead><tr><th scope="col">Runde</th><th scope="col">Pointfejl<br />App / FPL</th><th scope="col">Minutfejl<br />App / snit</th><th scope="col">Dækning</th></tr></thead>
            <tbody>{journal.entries.slice(0, 12).map((entry) => {
              const metrics = evaluateForecastAudit(entry);
              return <tr key={`${entry.season}:${entry.target_event}`}>
                <th scope="row">GW {entry.target_event}<small>{entry.season}, {entry.model_version}<br />Gemt {new Date(entry.captured_at).toLocaleString("da-DK", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}</small></th>
                <td>{metrics.points ? `${format(metrics.points.model_mae)} / ${format(metrics.points.baseline_mae)}` : entry.results ? "Data mangler" : "Afventer"}</td>
                <td>{metrics.minutes ? `${format(metrics.minutes.model_mae)} / ${format(metrics.minutes.baseline_mae)}` : entry.results ? "Data mangler" : "Afventer"}</td>
                <td>{metrics.points ? `${metrics.points.count}/${metrics.total} point` : "Ikke målt"}<small>{metrics.minutes ? `${metrics.minutes.count}/${metrics.total} minutter` : ""}</small></td>
              </tr>;
            })}</tbody>
          </table>
        </div>
      )}
      <p className={styles.footnote}>Fejl er den gennemsnitlige absolutte afvigelse pr. spiller; lavere er bedre. Begge metoder måles på de samme spillere med tilgængelige data, uden kaptajnbonus. Minutbaseline er sæsonminutter delt med antal færdigkontrollerede runder og korrigeres ikke for blanke eller dobbelte runder. Resultater bruges først, når FPL har færdigkontrolleret runden. Få runder beviser ikke, at en model er bedre.</p>
      <p className={styles.footnote}>Kun i denne browser. Journalen følger ikke automatisk med til andre enheder og kan ikke genskabes, hvis browserdata slettes. Eksportér en kopi. Første prognose låses i appen, men lokal lagring er ikke et uafhængigt, manipulationssikkert arkiv. {journal.entries.length > 12 ? "De seneste 12 runder vises; eksporten indeholder alle gemte runder." : ""}</p>
    </section>
  );
}
