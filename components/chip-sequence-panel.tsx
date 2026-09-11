import type { ChipSequenceComparison } from "@/lib/chip-sequences";
import styles from "./chip-sequence-panel.module.css";

const format = (value: number) => value.toLocaleString("da-DK", { maximumFractionDigits: 1 });

export function ChipSequencePanel({ comparison, playerNames }: {
  comparison: ChipSequenceComparison;
  playerNames: ReadonlyMap<number, string>;
}) {
  const name = (id: number) => playerNames.get(id) ?? `Spiller ${id}`;
  return (
    <section className={styles.panel} aria-labelledby="chip-sequence-heading">
      <h3 id="chip-sequence-heading">Wildcard og Bench Boost som samlet forløb</h3>
      <p className={styles.intro}>{comparison.reason}</p>
      {comparison.status === "ready" && <>
        <div className={styles.tableWrap} tabIndex={0} role="region" aria-label="Sammenligning af chipforløb">
          <table className={styles.table}>
            <thead><tr><th scope="col">Forløb</th><th scope="col">Vægtede nettopoint</th><th scope="col">Mod normale transfers</th><th scope="col">Hits</th></tr></thead>
            <tbody>{comparison.sequences.map((sequence) => <tr key={sequence.sequence_id}>
              <th scope="row">{sequence.label}{sequence.sequence_id === comparison.highest_projected_sequence_id && <small>Højeste prognose i denne sammenligning</small>}</th>
              <td>{format(sequence.weighted_net_points)}</td>
              <td>{sequence.gain_vs_normal_points > 0 ? "+" : ""}{format(sequence.gain_vs_normal_points)}</td>
              <td>{sequence.total_hit_points === 0 ? "0" : `−${sequence.total_hit_points}`}</td>
            </tr>)}</tbody>
          </table>
        </div>
        {comparison.sequences.map((sequence) => <details className={styles.details} key={sequence.sequence_id}>
          <summary>Se handlinger: {sequence.label}</summary>
          <ol className={styles.steps}>
            {sequence.actions.map((action, index) => <li key={action.event}>
              <strong>GW {action.event}{index > 0 ? " (foreløbig)" : ""}: </strong>
              {action.chip === "wildcard" ? "Wildcard. " : action.chip === "bboost" ? "Bench Boost. " : ""}
              {action.transfer_out_ids.length === 0 ? "Ingen transfers." : action.transfer_out_ids.map((out, i) => `${name(out)} ud, ${name(action.transfer_in_ids[i])} ind`).join("; ") + "."}
              <small>Bank £{format(action.bank_after_tenths / 10)}m. {action.free_transfers_next_gameweek} frie transfers til næste runde.{action.hit_points > 0 ? ` Hit: −${action.hit_points} point.` : ""}</small>
            </li>)}
          </ol>
        </details>)}
      </>}
      <p className={styles.limits}>Sammenligningen dækker {comparison.horizon} runder med samme vægtning og transferpolitik på de tilgængelige starttrupper. Fremtidige transfers er en afgrænset søgning blandt spillere fra den eksisterende plan og Wildcard-holdet, højst to pr. deadline efter den første og frem til runde fire. Derefter beholdes truppen. Priser fastholdes. Bench Boost vurderes kun inden chip-sættets udløb.</p>
      <p className={styles.limits}>Nettopoint fratrækker hits med rundernes vægtning. Værdien af at gemme en chip til senere er ikke prissat, og der søges ikke efter den optimale Wildcard-dato. Wildcard-holdet er ikke særskilt bygget til Bench Boost. Tallene er derfor et beslutningsgrundlag, ikke en automatisk besked om at spille chippen. Genberegn før næste deadline.</p>
    </section>
  );
}
