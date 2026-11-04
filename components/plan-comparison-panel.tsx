"use client";

import { useId } from "react";
import type { PlannerPayload } from "@/lib/planner-contract";
import { actionFinances, netPointsOver, planOptions, playersForAction, type PlanChoice } from "@/lib/plan-comparison";
import styles from "./plan-comparison-panel.module.css";

const money = (tenths: number) => `£${(tenths / 10).toLocaleString("da-DK", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}`;
const points = (value: number | null) => value === null ? "—" : `${value > 0 ? "+" : ""}${value.toLocaleString("da-DK", { maximumFractionDigits: 1, minimumFractionDigits: 1 })}`;

export function PlanComparisonPanel({ planner, selection, onSelect, disabled = false }: {
  planner: PlannerPayload;
  selection: PlanChoice;
  onSelect: (choice: PlanChoice) => void;
  disabled?: boolean;
}) {
  const id = useId();
  const options = planOptions(planner);
  const roll = options.find(option => option.action.kind === "roll")?.action;
  const selected = options.find(option => option.choice.kind === selection.kind && option.choice.alternative_index === selection.alternative_index);
  const action = selected?.action;
  const names = action ? playersForAction(planner, action) : new Map<number, string>();
  const lineup = action?.gameweeks[0];
  const finance = action ? actionFinances(action) : null;
  const playerList = (ids: number[]) => ids.map(playerId => names.get(playerId) ?? `Spiller ${playerId}`).join(" · ");
  return <section className={styles.panel} aria-labelledby={`${id}-heading`}>
    <header>
      <p className="eyebrow">Hele holdet og økonomien</p>
      <h2 id={`${id}-heading`}>Sammenlign planerne</h2>
      <p>Vælg den plan, du vil gemme i journalen. Hver række viser hele pakken og samme udgangspunkt.</p>
    </header>
    <div className={styles.scroll} tabIndex={0} role="region" aria-label="Transferplaner med økonomi og forventede nettopoint">
      <table>
        <caption className="sr-only">Forventede nettopoint mod at rulle. En, tre og fem runder vægtes ens og inkluderer transferfradraget én gang.</caption>
        <thead><tr><th scope="col">Plan til journalen</th><th scope="col">Bank</th><th scope="col">Hit</th><th scope="col">FT næste</th><th scope="col">1 GW</th><th scope="col">3 GW</th><th scope="col">5 GW</th></tr></thead>
        <tbody>{options.map(option => <tr key={option.key} className={selected?.key === option.key ? styles.selected : undefined}>
          <th scope="row"><label className={styles.choice}>
            <input type="radio" name={`${id}-plan`} checked={selected?.key === option.key} disabled={disabled} onChange={() => onSelect(option.choice)} />
            <span>{option.action.transfers.length ? option.action.transfers.map(move => `${move.out.name} → ${move.in.name}`).join(" + ") : "Rul transferen"}
              <small>{option.key === "best" ? "Modellens hovedplan" : option.action.hit_points ? "Med pointfradrag" : "Uden pointfradrag"}</small>
            </span>
          </label></th>
          <td>{money(option.action.bank_after_tenths)}</td><td>{option.action.hit_points ? `−${option.action.hit_points}` : "0"}</td><td>{option.action.free_transfers_next_gameweek}</td>
          {[1, 3, 5].map(weeks => <td key={weeks}>{points(netPointsOver(option.action, roll, weeks))}</td>)}
        </tr>)}</tbody>
      </table>
    </div>
    <p className={styles.note}>Tallene er forventede nettopoint mod at rulle, inklusive modellens bænkbidrag. Runderne har samme vægt her. Modellens hovedvalg vægter de nærmeste runder højere og medregner værdien af en gemt transfer. — betyder, at perioden eller sammenligningen mangler.</p>
    {action && lineup && finance ? <div className={styles.detail} aria-live="polite">
      <h3>Valgt pakke · GW{lineup.gameweek} · {lineup.formation}</h3>
      <p>{money(action.bank_before_tenths)} i banken + {money(finance.sold)} fra salg − {money(finance.bought)} i køb = <strong>{money(finance.bank)} tilbage</strong>.</p>
      <dl>
        <div><dt>Startellever</dt><dd>{playerList(lineup.starting_ids)}</dd></div>
        <div><dt>Kaptajn og vice</dt><dd>{names.get(lineup.captain_id)} · {lineup.vice_captain_id ? names.get(lineup.vice_captain_id) : "Vice ikke gemt i dette resultat"}</dd></div>
        <div><dt>Bænk</dt><dd>{lineup.bench_ids ? `${playerList(lineup.bench_ids.slice(0, 3))}. ${names.get(lineup.bench_ids[3])} er reservemålmand.` : "Bænkrækkefølge ikke gemt i dette resultat"}</dd></div>
      </dl>
      <p className={styles.note}>Planen nedenfor på banen er modellens hovedvalg. Dit valg her gemmes, når du trykker Gem beslutning i journalen.</p>
    </div> : <p className={styles.note}>Det aktuelle valg er at afvente. Vælg en konkret plan, hvis du vil gemme den som din beslutning.</p>}
  </section>;
}
