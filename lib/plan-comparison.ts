import type { PlannerAction, PlannerPayload } from "./planner-contract.ts";

export interface PlanChoice {
  kind: "best" | "alternative" | "wait";
  alternative_index: number | null;
}

export function planOptions(planner: PlannerPayload) {
  return [
    { key: "best", choice: { kind: "best", alternative_index: null } as PlanChoice, action: planner.best_action },
    ...planner.alternatives.map((action, index) => ({
      key: `alternative-${index}`,
      choice: { kind: "alternative", alternative_index: index } as PlanChoice,
      action,
    })),
  ];
}

/** Equal-weight horizon comparison, including the first-deadline hit exactly once. */
export function netPointsOver(action: PlannerAction, roll: PlannerAction | undefined, weeks: number): number | null {
  if (!roll || action.gameweeks.length < weeks || roll.gameweeks.length < weeks) return null;
  const selected = action.gameweeks.slice(0, weeks);
  if (selected.some((week, index) => week.gameweek !== roll.gameweeks[index].gameweek)) return null;
  return selected.reduce((sum, week, index) => sum + week.projected_points - roll.gameweeks[index].projected_points, 0)
    - action.hit_points + roll.hit_points;
}

export function actionFinances(action: PlannerAction) {
  return {
    sold: action.transfers.reduce((sum, move) => sum + move.out_selling_price_tenths, 0),
    bought: action.transfers.reduce((sum, move) => sum + move.in_price_tenths, 0),
    bank: action.bank_after_tenths,
  };
}

export function playersForAction(planner: PlannerPayload, action: PlannerAction) {
  const players = new Map(planner.confirmed_state.squad.map(player => [player.id, player]));
  const names = new Map([...players].map(([id, player]) => [id, player.name]));
  for (const move of action.transfers) names.set(move.in_id, move.in.name);
  return names;
}
