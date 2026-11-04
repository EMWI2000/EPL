"use client";

import { useId, useState } from "react";
import { DECISION_HISTORY_STORAGE_KEY, LEGACY_DECISION_HISTORY_STORAGE_KEY, MAX_DECISION_HISTORY_ENTRIES, selectedDecisionPlan,
  type DecisionHistory, type DecisionHistoryEntry, type DecisionPlan } from "@/lib/decision-history";
import { compareDecisionResult, type ActualDecisionResult, type DecisionMatch } from "@/lib/decision-results";
import { replayFrozenDecisionPlan } from "@/lib/frozen-plan-score";
import styles from "./decision-history-panel.module.css";

const matchLabel: Record<DecisionMatch, string> = { followed: "Matcher", deviated: "Afviger", unknown: "Kan ikke afgøres" };
const dates = new Intl.DateTimeFormat("da-DK", { dateStyle: "short", timeStyle: "short", timeZone: "Europe/Copenhagen" });
const date = (value: string) => dates.format(new Date(value));
const number = (value: number) => value.toLocaleString("da-DK", { maximumFractionDigits: 2 });
function actionLabel(entry: DecisionHistoryEntry) {
  if (!entry.action) return "Afvent information";
  if (!entry.action.transfers.length) return "Behold truppen";
  return entry.action.transfers.map(t => `${t.out_name} → ${t.in_name}`).join(" · ");
}
function planNames(plan: DecisionPlan, ids: number[] | null): string {
  if (!ids) return "Ikke gemt";
  const names = new Map(plan.players.map(p => [p.id, p.name]));
  return ids.map(id => names.get(id) ?? `Spiller ${id}`).join(" · ");
}
function RecordedPlan({ plan, title }: { plan: DecisionPlan; title: string }) {
  const first = plan.gameweeks[0];
  return <div className={styles.plan}>
    <strong>{title}</strong>
    <p>{plan.action.transfers.length ? plan.action.transfers.map(t => `${t.out_name} → ${t.in_name}`).join(" · ") : "Behold truppen"} · {plan.action.hit_points} i transferfradrag</p>
    <dl>
      <div><dt>Trup</dt><dd>{planNames(plan, plan.squad_ids)}</dd></div>
      <div><dt>Startellever</dt><dd>{planNames(plan, first.starting_ids)}</dd></div>
      <div><dt>Anfører / vice</dt><dd>{planNames(plan, [first.captain_id])} / {first.vice_captain_id === null ? "Ikke gemt" : planNames(plan, [first.vice_captain_id])}</dd></div>
      <div><dt>Bænkrækkefølge</dt><dd>{planNames(plan, first.bench_ids)}</dd></div>
      <div><dt>Forventede point</dt><dd>{number(plan.action.projected_points)} over den vægtede planhorisont</dd></div>
      <div><dt>Efter planen</dt><dd>£{number(plan.action.bank_after_tenths / 10)} mio. · {plan.action.free_transfers_next_gameweek} fri(e) transfer(s) i næste runde</dd></div>
    </dl>
  </div>;
}
export function DecisionHistoryPanel({ history, canSave, status, onSave, onExport, onClear }: {
  history: DecisionHistory; canSave: boolean; status: string | null;
  onSave: () => void; onExport: () => void; onClear: () => void;
}) {
  const heading = useId();
  const [showAll, setShowAll] = useState(false);
  const [results, setResults] = useState<Record<number, ActualDecisionResult>>({});
  const [loading, setLoading] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  async function load(event: number) {
    setLoading(event); setError(null);
    try {
      const response = await fetch(`/api/decision-results?event=${event}`, { cache: "no-store" });
      if (!response.ok) throw new Error("Resultatet kunne ikke hentes. Kontrollér login, og prøv igen.");
      const result = await response.json() as ActualDecisionResult;
      if (result.schema_version !== "fpl-decision-result-v1" || result.target_event !== event) throw new Error("Resultatet kunne ikke genkendes.");
      setResults(previous => ({ ...previous, [event]: result }));
    } catch (e) { setError(e instanceof Error ? e.message : "Resultatet kunne ikke hentes."); }
    finally { setLoading(null); }
  }
  function exportOriginal() {
    try {
      const originals = {
        [DECISION_HISTORY_STORAGE_KEY]: window.localStorage.getItem(DECISION_HISTORY_STORAGE_KEY),
        [LEGACY_DECISION_HISTORY_STORAGE_KEY]: window.localStorage.getItem(LEGACY_DECISION_HISTORY_STORAGE_KEY),
      };
      const url = URL.createObjectURL(new Blob([JSON.stringify(originals, null, 2)], { type: "application/json" }));
      const link = document.createElement("a");
      link.href = url; link.download = "fpl-journal-original.json"; link.click();
      URL.revokeObjectURL(url);
    } catch { setError("Browseren tillader ikke adgang til den oprindelige journal."); }
  }
  const visible = showAll ? history.entries : history.entries.slice(0, 4);
  return <section className={styles.section} aria-labelledby={heading}>
    <div className={styles.header}>
      <div><p className={styles.eyebrow}>Fra plan til resultat</p><h2 id={heading}>Din beslutningsjournal</h2>
        <p>Gem et snapshot før deadline. Hver version bevares lokalt i denne browser. Efter runden kan du sammenligne med dit offentliggjorte FPL-hold.</p></div>
      <button type="button" onClick={onSave} disabled={!canSave || Boolean(history.storage_error)}>Gem beslutning</button>
    </div>
    {status && <p className={styles.notice} role="status">{status}</p>}
    {history.storage_error && <div className={styles.notice} role="alert"><p>{history.storage_error}</p><button type="button" onClick={exportOriginal}>Hent oprindelig journal</button></div>}
    {error && <p className={styles.notice} role="alert">{error}</p>}
    {!visible.length && !history.storage_error && <p>Ingen gemte beslutninger endnu. Gem efter at have kontrolleret trup, bank, transfers og den valgte plan.</p>}
    <div className={styles.entries}>{visible.map(entry => {
      const result = results[entry.target_event];
      const comparison = result ? compareDecisionResult(entry, result) : null;
      const selected = selectedDecisionPlan(entry);
      const frozen = result?.finalized ? [
        { title: "Valgt plan", score: replayFrozenDecisionPlan(entry, selected, result) },
        { title: "Modellens bedste", score: replayFrozenDecisionPlan(entry, entry.plans?.best ?? null, result) },
        { title: "Behold truppen", score: replayFrozenDecisionPlan(entry, entry.plans?.roll ?? null, result) },
      ] : null;
      return <article key={entry.snapshot_id}>
        <div className={styles.entryHeader}>
          <div><span className={styles.eyebrow}>{entry.season} · GW{entry.target_event}</span><h3>{actionLabel(entry)}</h3>
            <p>Gemt {date(entry.saved_at)} · deadline {date(entry.target_deadline)}</p></div>
          <span className={styles.badge}>{entry.origin === "migrated_v1" ? "Ældre version" : entry.selection_source === "user" ? "Dit valg" : entry.selection_source === "ai" ? "AI-vurderingens valg" : "Modellens valg"}</span>
        </div>
        <p className={styles.muted}>£{number(entry.confirmed.bank_tenths / 10)} mio. i banken · {entry.confirmed.free_transfers} fri(e) transfer(s) · {entry.forecast.version} · {entry.forecast.horizon} runders horisont</p>
        {entry.origin === "migrated_v1" && <p className={styles.muted}>Den gamle version gemte kun et kort resumé. Manglende trup- og opstillingsdata kan ikke genskabes.</p>}
        <details><summary>Vis den gemte beslutning og alternativer</summary>
          <p className={styles.muted}>Grundlag hentet {date(entry.state_observed_at)}. Anbefaling beregnet {date(entry.recommendation_generated_at)}. Modellens valideringsstatus var {entry.forecast.validation_status}.</p>
          {entry.context && <p>Tilgængeligt budget ved gemning £{number(entry.context.budget_tenths / 10)} mio. inklusive salgspriser. {entry.context.chips ? `Tilgængelige chips er ${Object.entries(entry.context.chips).filter(([, value]) => value === "available").map(([name]) => name).join(", ") || "ingen"}.` : "Chipbeholdning blev ikke gemt."}</p>}
          {selected && <RecordedPlan plan={selected} title="Den valgte plan" />}
          {entry.plans?.best && selected !== entry.plans.best && <RecordedPlan plan={entry.plans.best} title="Modellens bedste plan" />}
          {entry.plans?.roll && selected?.action.kind !== "roll" && <RecordedPlan plan={entry.plans.roll} title="Alternativet med uændret trup" />}
          {entry.plans?.alternatives.map((plan, i) => <RecordedPlan key={i} plan={plan} title={`Alternativ ${i + 1}`} />)}
          {!entry.plans && <p>{entry.lineup ? `Den gemte anfører er ${entry.lineup.captain_name}.` : "Ingen gemt opstilling."}</p>}
        </details>
        <div className={styles.result}>
          <button type="button" disabled={loading !== null || Date.now() < Date.parse(entry.target_deadline)} onClick={() => void load(entry.target_event)}>
            {loading === entry.target_event ? "Henter officielt resultat…" : result ? "Opdatér resultat" : "Sammenlign med faktisk hold"}
          </button>
          {comparison && <>
            <strong>{result.finalized ? matchLabel[comparison.status] : "Afventer færdigbehandlet runde"}</strong>
            {comparison.net_points !== null && <p>{comparison.gross_points} point − {comparison.transfer_cost} i transferfradrag = <b>{comparison.net_points} nettopoint</b></p>}
            {comparison.net_points !== null && <p className={styles.muted}>Trup {matchLabel[comparison.squad_match].toLowerCase()} · handler {matchLabel[comparison.transfer_match].toLowerCase()} · opstilling {matchLabel[comparison.lineup_match].toLowerCase()} · chip {matchLabel[comparison.chip_match].toLowerCase()}</p>}
            <ul>{comparison.reasons.map(reason => <li key={reason}>{reason}</li>)}</ul>
            <p className={styles.muted}>Officielle data hentet {date(result.fetched_at)}. Automatiske indskiftninger er ført tilbage ved sammenligning af din indsendte opstilling.</p>
          </>}
          {frozen && <div className={styles.frozen}>
            <h4>Fastholdt plan · samme GW</h4>
            <p>De gemte opstillinger genberegnes lokalt med rundens officielle spillerpoint, bænkrækkefølge og anførerskift.</p>
            <div className={styles.tableScroll}><table>
              <thead><tr><th scope="col">Gemt plan</th><th scope="col">Point</th><th scope="col">Planens hit</th><th scope="col">Netto</th></tr></thead>
              <tbody>{frozen.map(({ title, score }) => <tr key={title}>
                <th scope="row">{title}</th><td>{score.gross_points ?? "—"}</td>
                <td>{score.planned_hit_points ?? "—"}</td><td>{score.net_points ?? "—"}</td>
              </tr>)}</tbody>
            </table></div>
            {frozen[0].score.net_points !== null && frozen[2].score.net_points !== null
              && <p>Den fastholdte valgte plan gav <b>{number(frozen[0].score.net_points - frozen[2].score.net_points)} point</b> i forskel mod den gemte plan med uændret trup.</p>}
            {frozen.some(item => !item.score.available) && <ul>{[...new Set(frozen.filter(item => !item.score.available).map(item => item.score.reason))].map(reason => <li key={reason}>{reason}</li>)}</ul>}
            <p className={styles.muted}>Planens nye transferfradrag er trukket fra én gang. Eventuelle hit pådraget før gemningen er en fælles, ukendt baseline og indgår ikke her. Sammenlign forskellen mellem de fastholdte planer. Dette er én runde med uændrede valg, ikke et senere transferforløb eller et mål for appens årsagsvirkning.</p>
          </div>}
        </div>
      </article>;
    })}</div>
    <div className={styles.footer}>
      <p>Op til {MAX_DECISION_HISTORY_ENTRIES} gemte versioner. Eksportér jævnligt til en fil. Journalen følger browseren og synkroniseres ikke mellem enheder.</p>
      <div>{history.entries.length > 4 && <button type="button" aria-expanded={showAll} onClick={() => setShowAll(v => !v)}>{showAll ? "Vis seneste" : `Vis alle ${history.entries.length}`}</button>}
        <button type="button" disabled={!history.entries.length} onClick={onExport}>Eksportér journal</button>
        <button type="button" disabled={!history.entries.length && !history.storage_error} onClick={onClear}>Slet journal</button></div>
    </div>
  </section>;
}
