"use client";

import { useId } from "react";
import type { ChipPolicy } from "@/lib/chip-policy";
import type { ChipName } from "@/lib/planner-contract";
import styles from "./chip-rules-panel.module.css";

const chipNames: Record<ChipName, string> = {
  wildcard: "Wildcard", freehit: "Free Hit", bboost: "Bench Boost", "3xc": "Triple Captain",
};
const statusNames: Record<ChipPolicy["status"], string> = {
  hold: "Følg planen", review: "Kontrollér planen", consider: "Overvej chippen", urgent: "Tag stilling nu", used: "Sættet er brugt",
};
const points = (value: number | null) => value === null ? "Ikke beregnet"
  : `${value > 0 ? "+" : ""}${value.toLocaleString("da-DK", { maximumFractionDigits: 1 })} point`;

export function ChipRulesPanel({ policy, inventoryConfirmed, inventoryRefreshing = false, deadlinePassed = false }: {
  policy: ChipPolicy;
  inventoryConfirmed: boolean;
  inventoryRefreshing?: boolean;
  deadlinePassed?: boolean;
}) {
  const id = useId();
  const wildcard = policy.wildcard;
  const checkpoints = [wildcard.reviewEvent, wildcard.promptEvent].filter((event): event is number => event !== null);
  const nextCheckpoint = checkpoints.length ? Math.max(policy.targetEvent, Math.min(...checkpoints)) : null;
  return <section className={`${styles.panel} ${policy.urgent ? styles.urgent : ""}`} aria-labelledby={`${id}-heading`}>
    <header className={styles.header}>
      <div><p className={styles.eyebrow}>Chipregler · GW{policy.targetEvent}</p>
        <h2 id={`${id}-heading`}>{deadlinePassed ? "Chipplanen skal opdateres" : `Din chipplan frem mod GW${policy.expiryEvent}`}</h2></div>
      <span className={styles.badge} data-status={deadlinePassed ? "review" : policy.status}>{deadlinePassed ? "Opdatér holddata" : statusNames[policy.status]}</span>
    </header>
    <div className={styles.message} aria-live="polite">
      <strong>{deadlinePassed ? "Den viste deadline er passeret" : policy.headline}</strong>
      {deadlinePassed ? <p>Hent holdet igen for at få opdateret din chipstatus efter deadline.</p>
        : policy.reasons.length > 0 && <ul>{policy.reasons.map((reason, index) => <li key={`${index}-${reason}`}>{reason}</li>)}</ul>}
    </div>
    {!inventoryConfirmed && !deadlinePassed && <p className={styles.confirmation} role="status">
      {inventoryRefreshing ? "Chipbeholdningen bliver opdateret. " : "Chipbeholdningen kommer fra det senest offentliggjorte hold. "}
      Bekræft din aktuelle trup og chipstatus, før du følger en chipplan.
    </p>}
    <dl className={styles.metrics}>
      <div><dt>{deadlinePassed ? "Chips ved seneste hentning" : "Chips tilbage"}</dt><dd>{policy.unusedCount}</dd></div>
      <div><dt>Deadlines tilbage</dt><dd>{deadlinePassed ? "—" : policy.remainingDeadlines}</dd><small>{deadlinePassed ? "Kræver nye holddata" : `Inklusive GW${policy.targetEvent}`}</small></div>
      <div><dt>Sidste mulighed</dt><dd>GW{policy.expiryEvent}</dd><small>{policy.half === 1 ? "Første chipsæt" : "Andet chipsæt"}</small></div>
    </dl>
    {policy.unusedChips.length > 0 && <ul className={styles.chips} aria-label="Ubrugte chips i dette sæt">
      {policy.unusedChips.map(chip => <li key={chip}>
        <strong>{chipNames[chip]}</strong>
        <span>{deadlinePassed ? "Status skal opdateres" : policy.availableNow.includes(chip) ? "Tilgængelig nu" : "Kan ikke vælges i denne runde"}</span>
      </li>)}
    </ul>}
    {!deadlinePassed && !policy.allCanStillBeUsed && policy.unusedCount > 0 && <p className={styles.warning}>
      Der er flere ubrugte chips end ledige deadlines. Prioritér dem med størst forventet værdi for dit hold.
    </p>}
    {!deadlinePassed && policy.unusedChips.includes("wildcard") && <div className={styles.wildcard}>
      <div className={styles.wildcardHeading}><h3>Wildcard-kontrol</h3>
        <span>{nextCheckpoint !== null ? `Næste holdepunkt · GW${nextCheckpoint}` : "Kontrollér ved næste deadline"}</span>
      </div>
      {wildcard.evidence !== "current" && <p className={styles.note}>
        Der mangler en aktuel chipberegning for den bekræftede trup. Tidsfristerne ovenfor gælder stadig.
      </p>}
      <dl className={styles.evidence}>
        <div><dt>Ændringer i Wildcard-truppen</dt><dd>{wildcard.changeCount ?? "Ikke beregnet"}</dd></div>
        <div><dt>Wildcard mod normale transfers</dt><dd>{points(wildcard.gainVsNormal)}</dd></div>
        <div><dt>Wildcard + BB mod normale transfers + BB</dt><dd>{points(wildcard.gainWithBenchBoost)}</dd></div>
      </dl>
      <p className={styles.note}>Gevinsterne er vægtede nettopoint fra modelscenarier med samme horisont. Sammenhold dem med mindst fire planlagte spillerændringer og grænsen på 15 point.</p>
    </div>}
    <details className={styles.rules}>
      <summary>Se chipreglerne, der bruges ved hver deadline</summary>
      <p>Én chip pr. runde. Ubrugte chips udløber med sættet. Modellen kvalificerer tidspunktet med din trup og de kendte kampe.</p>
      <p>Genåbn vurderingen ved skader, ændrede spillerroller og nye kamptidspunkter. Hent holdet, bekræft oplysningerne og beregn igen.</p>
      <dl>{policy.criteria.map(rule => <div key={rule.chip}><dt>{rule.label}</dt><dd>{rule.rule}</dd></div>)}</dl>
      {wildcard.reviewEvent !== null && <p>Wildcard-planen gennemgås fra GW{wildcard.reviewEvent}.</p>}
      {wildcard.latestPlanningEvent !== null && <p>Seneste planlægningspunkt for dette chipsæt er GW{wildcard.latestPlanningEvent}.</p>}
    </details>
  </section>;
}
