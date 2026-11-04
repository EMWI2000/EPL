import type { ChipName, ChipState, ChipUsage, PlannerChipStrategy } from "./planner-contract.ts";
import type { ChipSequence, ChipSequenceComparison, ChipSequenceId } from "./chip-sequences.ts";

export type ChipPolicyStatus = "hold" | "review" | "consider" | "urgent" | "used";
export type ChipPolicyEvidence = "current" | "unknown" | "stale" | "unavailable";

export interface ChipPolicy {
  version: "fpl-chip-policy-v1";
  targetEvent: number;
  half: 1 | 2;
  expiryEvent: 19 | 38;
  /** Includes the current deadline. Calendar facts do not depend on a solver. */
  remainingDeadlines: number;
  unusedChips: ChipName[];
  availableNow: ChipName[];
  unusedCount: number;
  slack: number;
  urgent: boolean;
  allCanStillBeUsed: boolean;
  status: ChipPolicyStatus;
  headline: string;
  reasons: string[];
  wildcard: {
    status: ChipPolicyStatus;
    availableNow: boolean;
    /** GW13 is a personal first-half checkpoint, not an optimized play date. */
    reviewEvent: number | null;
    /** Latest planning round if WC goes before the other remaining chips. */
    latestPlanningEvent: number | null;
    promptEvent: number | null;
    changeCount: number | null;
    gainVsNormal: number | null;
    gainWithBenchBoost: number | null;
    evidence: ChipPolicyEvidence;
  };
  /** Guidance only. This module cannot verify four reserves' expected minutes. */
  criteria: Array<{ chip: ChipName; label: string; rule: string }>;
}

export interface ChipPolicyInput {
  targetEvent: number;
  chips: ChipState;
  chipUsage: ChipUsage[];
  strategy: PlannerChipStrategy | null;
  evidenceCurrent: boolean;
}

const CHIPS: readonly ChipName[] = ["wildcard", "freehit", "bboost", "3xc"];
const WILDCARD_MINIMUM_CHANGES = 4;
const WILDCARD_MINIMUM_GAIN = 15;
const CRITERIA: ChipPolicy["criteria"] = [
  {
    chip: "wildcard",
    label: "Wildcard",
    rule: "Mindst 4 ændringer og 15 vægtede merpoint mod normale transfers over samme horisont. Ved en plan med Bench Boost sammenlignes to forløb, der begge bruger Bench Boost.",
  },
  {
    chip: "bboost",
    label: "Bench Boost",
    rule: "Mindst 12 marginale point, 4 reserver med forventet spilletid og mindst middel sikkerhed. Kontrollér selv reservernes aktuelle roller og holdnyheder.",
  },
  {
    chip: "3xc",
    label: "Triple Captain",
    rule: "Mindst 8 ekstra kaptajnpoint og mindst middel sikkerhed. Bekræft kaptajnens forventede spilletid efter seneste holdnyheder.",
  },
  {
    chip: "freehit",
    label: "Free Hit",
    rule: "Vurdér ved en bekræftet blank eller double gameweek. Hvis ingen sådan runde er bekræftet, kræves en manuel vurdering inden chipsættets udløb.",
  },
];

function completeSequence(
  comparison: ChipSequenceComparison,
  id: ChipSequenceId,
  target: number,
  expiry: number,
): ChipSequence | null {
  const matches = comparison.sequences.filter(sequence => sequence.sequence_id === id);
  if (matches.length !== 1) return null;
  const sequence = matches[0];
  if (!Number.isFinite(sequence.weighted_net_points)
    || sequence.actions.length !== comparison.horizon
    || sequence.actions.some((action, index) => action.event !== target + index || action.event > 38)) {
    return null;
  }
  const expectedChips = id === "normal" ? [] : id === "wildcard" ? ["wildcard"]
    : id === "normal-bboost" ? ["bboost"] : ["bboost", "wildcard"];
  const chipActions = sequence.actions.filter(action => action.chip !== null);
  if (chipActions.some(action => action.event > expiry || (action.chip === "wildcard" && action.event !== target))
    || JSON.stringify(chipActions.map(action => action.chip).sort()) !== JSON.stringify(expectedChips)) {
    return null;
  }
  return sequence;
}

function pairedBenefits(input: ChipPolicyInput, expiry: number, wildcardAvailable: boolean, boostAvailable: boolean) {
  const missing = { evidence: "unknown" as ChipPolicyEvidence, changeCount: null as number | null,
    gainVsNormal: null as number | null, gainWithBenchBoost: null as number | null };
  if (!wildcardAvailable) return { ...missing, evidence: "unavailable" as const };
  if (!input.strategy) return missing;
  const strategy = input.strategy;
  if (!input.evidenceCurrent || strategy.target_event !== input.targetEvent) {
    return { ...missing, evidence: "stale" as const };
  }
  const comparison = strategy.sequence_comparison;
  const scenario = strategy.scenarios.find(s => s.chip === "wildcard" && s.event === input.targetEvent && s.available);
  const inventory = strategy.inventory.find(item => item.chip === "wildcard");
  if (!inventory?.available_for_target || !comparison || comparison.status !== "ready"
    || comparison.horizon !== strategy.horizon || !scenario) {
    return { ...missing, evidence: "unavailable" as const };
  }
  const normal = completeSequence(comparison, "normal", input.targetEvent, expiry);
  const wildcard = completeSequence(comparison, "wildcard", input.targetEvent, expiry);
  const normalBoost = boostAvailable ? completeSequence(comparison, "normal-bboost", input.targetEvent, expiry) : null;
  const wildcardBoost = boostAvailable ? completeSequence(comparison, "wildcard-bboost", input.targetEvent, expiry) : null;
  const difference = (chip: ChipSequence | null, baseline: ChipSequence | null) =>
    chip && baseline ? Math.round((chip.weighted_net_points - baseline.weighted_net_points) * 1000) / 1000 : null;
  const gainVsNormal = difference(wildcard, normal);
  const gainWithBenchBoost = difference(wildcardBoost, normalBoost);
  return {
    evidence: gainVsNormal !== null || gainWithBenchBoost !== null ? "current" as const : "unavailable" as const,
    changeCount: Number.isInteger(scenario.change_count) && scenario.change_count! >= 0 && scenario.change_count! <= 15
      ? scenario.change_count : null,
    gainVsNormal,
    gainWithBenchBoost,
  };
}

/** Apply transparent house rules to validated FPL state, never activate a chip. */
export function buildChipPolicy(input: ChipPolicyInput): ChipPolicy {
  const { targetEvent, chips, chipUsage } = input;
  if (!Number.isInteger(targetEvent) || targetEvent < 1 || targetEvent > 38) {
    throw new RangeError("Chip policy requires a target gameweek from 1 to 38");
  }
  const half = targetEvent <= 19 ? 1 : 2;
  const firstEvent = half === 1 ? 1 : 20;
  const expiryEvent = half === 1 ? 19 : 38;
  const usedThisHalf = new Set(chipUsage
    .filter(use => use.event >= firstEvent && use.event < targetEvent)
    .map(use => use.name));
  const unusedChips = CHIPS.filter(chip => chips[chip] !== "used" && !usedThisHalf.has(chip));
  const previousFreeHit = chipUsage.some(use => use.name === "freehit" && use.event === targetEvent - 1);
  const availableNow = unusedChips.filter(chip => chips[chip] === "available"
    && !((chip === "wildcard" || chip === "freehit") && targetEvent === 1)
    && !(chip === "freehit" && previousFreeHit));
  const unusedCount = unusedChips.length;
  const remainingDeadlines = expiryEvent - targetEvent + 1;
  const slack = remainingDeadlines - unusedCount;
  const urgent = unusedCount > 0 && slack <= 0;
  const wildcardUnused = unusedChips.includes("wildcard");
  const wildcardAvailable = availableNow.includes("wildcard");
  const boostUnused = unusedChips.includes("bboost");
  const latestPlanningEvent = wildcardUnused
    ? Math.min(expiryEvent - (unusedCount - 1), boostUnused ? expiryEvent - 1 : expiryEvent) : null;
  const promptEvent = latestPlanningEvent === null ? null : Math.max(firstEvent, latestPlanningEvent - 1);
  const reviewEvent = wildcardUnused && half === 1 ? 13 : null;
  const benefits = pairedBenefits(input, expiryEvent, wildcardAvailable, availableNow.includes("bboost"));
  const gains = [benefits.gainVsNormal, benefits.gainWithBenchBoost].filter((gain): gain is number => gain !== null);
  const meetsThreshold = benefits.evidence === "current"
    && benefits.changeCount !== null && benefits.changeCount >= WILDCARD_MINIMUM_CHANGES
    && gains.some(gain => gain >= WILDCARD_MINIMUM_GAIN);
  const atCheckpoint = (reviewEvent !== null && targetEvent >= reviewEvent)
    || (promptEvent !== null && targetEvent >= promptEvent);

  let wildcardStatus: ChipPolicyStatus = !wildcardUnused ? "used" : !wildcardAvailable ? "hold"
    : meetsThreshold ? "consider"
    : atCheckpoint || benefits.evidence !== "current" ? "review" : "hold";
  let status: ChipPolicyStatus = unusedCount === 0 ? "used"
    : wildcardStatus === "used" ? "hold" : wildcardStatus;
  const reasons: string[] = [];
  let headline = "Planlæg de resterende chips";

  if (!wildcardUnused) reasons.push("Wildcard i dette chipsæt er allerede brugt.");
  else {
    if (reviewEvent !== null) {
      reasons.push(targetEvent < reviewEvent
        ? "GW13 er dit personlige tidspunkt for en ny Wildcard-vurdering."
        : "Dit personlige Wildcard-tjek fra GW13 skal følges op ved hver deadline.");
    } else reasons.push("Vælg et Wildcard-vindue efter de bekræftede kampe og holdets behov i anden sæsonhalvdel.");
    if (!wildcardAvailable) reasons.push("Wildcard er ikke tilgængeligt i den aktuelle runde.");
    else if (meetsThreshold) {
      headline = "Wildcard opfylder grænsen for overvejelse";
      reasons.push("Mindst ét sammenligneligt forløb viser 15 vægtede merpoint, og mindst 4 spillere ændres.");
    } else if (benefits.evidence === "current") {
      headline = atCheckpoint ? "Tag stilling til Wildcard ved denne deadline" : "Gem Wildcard til næste vurdering";
      reasons.push("Den aktuelle beregning opfylder ikke begge krav om mindst 4 ændringer og 15 vægtede merpoint.");
    } else {
      headline = "Wildcard kræver en aktuel vurdering";
      reasons.push(benefits.evidence === "stale"
        ? "Chipberegningen matcher ikke det aktuelle grundlag. Beregn igen, før du vurderer gevinsten."
        : "Der er ingen fuldført, sammenlignelig chipberegning. Den beregnede gevinst er ukendt.");
    }
    if (boostUnused) reasons.push(remainingDeadlines >= 2
      ? "Hvis du prioriterer både Wildcard og Bench Boost, planlæg Wildcard først og brug hver sin deadline."
      : "Wildcard og Bench Boost konkurrerer om sidste deadline. Vælg blandt de chips, der er tilgængelige nu.");
    if (promptEvent !== null && targetEvent >= promptEvent && latestPlanningEvent !== null) {
      reasons.push(targetEvent > latestPlanningEvent
        ? `Planlægningspunktet GW${latestPlanningEvent} er passeret. Revidér rækkefølgen af de resterende chips nu.`
        : `Fastlæg Wildcard-planen nu. GW${latestPlanningEvent} er seneste planlægningsrunde, hvis Wildcard skal komme før de øvrige resterende chips.`);
    }
  }

  if (urgent) {
    status = "urgent";
    if (wildcardAvailable) wildcardStatus = "urgent";
    headline = slack < 0 ? "Alle resterende chips kan ikke nås" : "Chipbeslutning kræves ved denne deadline";
    reasons.unshift(slack < 0
      ? `Du har ${unusedCount} ubrugte chips og ${remainingDeadlines} ${remainingDeadlines === 1 ? "deadline" : "deadlines"} til og med GW${expiryEvent}. Prioritér de chips, der stadig kan bruges.`
      : `Antallet af ubrugte chips matcher de ${remainingDeadlines} resterende muligheder til og med GW${expiryEvent}. Hvis alle skal bruges, kræves én ved hver resterende deadline, inklusive denne.`);
    reasons.push("Vælg blandt de tilgængelige chips efter aktuel vurdering. Udløb giver ikke i sig selv et dokumenteret pointoverskud.");
  } else if (unusedCount === 0) {
    headline = "Alle chips i dette sæt er brugt";
    reasons.push(half === 1 ? "Næste chipsæt åbner i GW20." : "Der er ingen flere chips i denne sæson.");
  } else {
    reasons.push(`Der er ${remainingDeadlines} deadlines til ${unusedCount} ubrugte chips inden udløbet ved GW${expiryEvent}.`);
  }
  if (unusedCount > 0) reasons.push("Bekræft trup, chipbeholdning, priser og seneste holdnyheder ved hver deadline. Tærsklerne er beslutningsregler og er endnu ikke kalibreret på historiske anbefalinger.");

  return { version: "fpl-chip-policy-v1", targetEvent, half, expiryEvent, remainingDeadlines, unusedChips,
    availableNow, unusedCount, slack, urgent, allCanStillBeUsed: slack >= 0, status, headline, reasons,
    wildcard: { status: wildcardStatus, availableNow: wildcardAvailable, reviewEvent, latestPlanningEvent, promptEvent, ...benefits },
    criteria: CRITERIA.map(criterion => ({ ...criterion })) };
}
