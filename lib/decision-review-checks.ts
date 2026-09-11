import type { AiReviewRequest } from "./ai-review-contract.ts";

const text = (value: string) => value.replace(/[\u0000-\u001f\u007f]/g, " ").replace(/\s+/g, " ").trim().slice(0, 80);
const difference = (a: number, b: number) => Number((a - b).toFixed(3));

/** Arithmetic checks, not a second forecast or a confidence probability. */
export function buildDecisionReviewChecks(request: AiReviewRequest) {
  const { best_action: best, alternatives, confirmed_state: state } = request.planner;
  const profiles = new Map(request.squad_context.map(p => [p.id, p]));
  const comparisons = alternatives.map((alternative, index) => ({
    alternative_index: index,
    best_minus_alternative_net_ep: difference(best.net_points_vs_roll, alternative.net_points_vs_roll),
    best_minus_alternative_decision_value: difference(best.decision_value_vs_roll, alternative.decision_value_vs_roll),
    next_gameweek_points_difference: difference(best.gameweeks[0].projected_points, alternative.gameweeks[0].projected_points),
  }));
  const strongest = comparisons.toSorted((a, b) => alternatives[b.alternative_index].net_points_vs_roll - alternatives[a.alternative_index].net_points_vs_roll
    || a.alternative_index - b.alternative_index)[0] ?? null;
  const examined = [best, ...(strongest ? [alternatives[strongest.alternative_index]] : [])];
  const missingProfiles = [...new Map(examined.flatMap(a => a.transfers.flatMap(t => [t.out, t.in]))
    .filter(p => !profiles.has(p.id)).map(p => [p.id, { player: text(p.name), club: text(p.team) }])).values()];
  const captain = profiles.get(request.lineup.captain_id);
  const projection = captain?.projections.find(p => p.gameweek === request.planner.target_event);
  return {
    current_owned_players: (state.squad ?? []).map(p => ({ player: text(p.name), club: text(p.team), position: p.position })),
    best_net_ep_vs_roll: best.net_points_vs_roll,
    comparisons,
    strongest_points_alternative_index: strongest?.alternative_index ?? null,
    captain_check: captain && projection ? {
      player: text(captain.name), club: text(captain.team), status: captain.status,
      expected_points: projection.expected_points, expected_minutes: projection.expected_minutes,
      appearance_probability: projection.appearance_probability, reliability: projection.reliability,
    } : null,
    missing_individual_forecasts: missingProfiles,
    interpretation: "Positive margins favour the proposed action; zero is a tie; negative means that alternative has more forecast points. Decision value also prices saved transfers. Margins concern the entire weighted team plan, not one player's minutes or a win probability. No counterfactual minute simulation was run.",
    ownership_scope: "current_owned_players is the confirmed current squad; squad_outlook and next_gameweek_lineup describe the proposed POST-transfer squad. Missing outgoing/alternative individual forecasts are unknown, not zero. Research those players before concluding the transfer is right.",
  };
}
