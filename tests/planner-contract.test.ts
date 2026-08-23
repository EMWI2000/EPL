import assert from "node:assert/strict";
import test from "node:test";

import {
  PlannerContractError,
  isManagerSyncResponse,
  isPlannerPayload,
  parseBankTenths,
  parseFreeTransfers,
  parseManagerId,
  parseManagerSyncResponse,
  parsePlannerPayload,
  serializePlannerRequest,
  stringifyPlannerRequest,
} from "../lib/planner-contract.ts";

const limitations = [
  "state_is_locked_at_last_public_deadline",
  "current_free_transfers_not_public",
  "current_confirmed_transfers_not_public",
  "purchase_and_selling_prices_not_public",
  "next_deadline_chip_selection_not_public",
];

function playerPosition(id: number): "GKP" | "DEF" | "MID" | "FWD" {
  if (id <= 2) return "GKP";
  if (id <= 7) return "DEF";
  if (id <= 12) return "MID";
  return "FWD";
}

function priceSignal(id: number, price = 50) {
  return {
    element_id: id,
    now_cost_tenths: price,
    cost_change_start_tenths: 0,
    cost_change_event_tenths: 0,
    selected_by_percent: 10,
    transfers_in_event: 100,
    transfers_out_event: 50,
    price_change_percent: 95.5,
    price_change_hourly_rate: 4,
    projections: [
      { offset_days: 0, projected_percent: 96, likelihood_code: 2 },
      { offset_days: 1, projected_percent: 98, likelihood_code: 3 },
    ],
    locked_until: null,
    calibrating: false,
  };
}

function syncResponse(): Record<string, any> {
  const picks = Array.from({ length: 15 }, (_, index) => {
    const id = index + 1;
    return {
      element_id: id,
      name: `Player ${id}`,
      club: `C${((id - 1) % 5) + 1}`,
      club_name: `Club ${((id - 1) % 5) + 1}`,
      club_id: ((id - 1) % 5) + 1,
      position: playerPosition(id),
      lineup_position: id,
      multiplier: id === 1 ? 2 : id <= 11 ? 1 : 0,
      is_captain: id === 1,
      is_vice_captain: id === 2,
      current_price_tenths: 50,
      estimated_purchase_price_tenths: 50,
      estimated_selling_price_tenths: 50,
      purchase_price_basis: { kind: "season_start_price" },
      official_price_signal: priceSignal(id),
    };
  });
  const playerPrices = picks.map((pick) => ({
    element_id: pick.element_id,
    purchase_price_tenths: pick.estimated_purchase_price_tenths,
    selling_price_tenths: pick.estimated_selling_price_tenths,
  }));
  return {
    schema_version: "fpl-manager-state-response-v2",
    generated_at: "2026-08-23T12:00:00Z",
    manager: {
      id: 123,
      team_name: "Test XI",
      overall_points: 321,
      overall_rank: 4567,
    },
    target: {
      event: 7,
      name: "Gameweek 7",
      deadline_time: "2026-08-30T17:30:00Z",
    },
    last_deadline_state: {
      schema_version: "fpl-personal-state-v2",
      state_kind: "public_last_deadline",
      event: 6,
      bank_tenths: 8,
      squad_value_tenths: 1012,
      total_transfers_at_deadline: 4,
      event_transfers: 1,
      event_transfer_cost: 0,
      active_chip: null,
      chip_usage: [],
      limitations,
      picks,
    },
    price_signals: {
      schema_version: "fpl-official-price-signals-v1",
      available: true,
      player_count: 15,
      price_change_deadlines: [
        "2026-08-23T23:00:00Z",
        "2026-08-24T23:00:00Z",
      ],
      warning: null,
    },
    manual_state_template: {
      state: {
        current_squad_ids: picks.map((pick) => pick.element_id),
        bank_tenths: 8,
        free_transfers: 1,
        player_prices: playerPrices,
        chips: {
          wildcard: "available",
          freehit: "available",
          bboost: "available",
          "3xc": "available",
        },
        chip_usage: [],
        no_active_chip_confirmed: false,
        effective_event: 7,
      },
      confirmation_required: true,
      fields_requiring_confirmation: [
        "current_squad_ids",
        "bank_tenths",
        "free_transfers",
        "player_prices",
        "chips",
        "chip_usage",
        "no_active_chip_confirmed",
      ],
      free_transfers_default_reason: "current_free_transfers_are_not_public",
    },
    snapshot: {
      schema_version: "fpl-deadline-state-snapshot-v2",
      observed_at: "2026-08-23T12:00:00.000000Z",
      source: "fpl_public_last_deadline",
      checksum_sha256: "a".repeat(64),
      persisted: false,
    },
    warnings: [
      {
        code: "last_deadline_state_requires_confirmation",
        message: "Confirm the current state before planning.",
      },
    ],
  };
}

function playerReference(id: number, position = playerPosition(id)) {
  return {
    id,
    name: `Player ${id}`,
    team: `Club ${((id - 1) % 5) + 1}`,
    position,
    price_tenths: 50,
    status: "a",
    price_signal: null,
  };
}

function ownedPlayerReference(id: number, position = playerPosition(id)) {
  return {
    ...playerReference(id, position),
    purchase_price_tenths: 50,
    selling_price_tenths: 50,
  };
}

function action(
  kind: "roll" | "transfer" = "roll",
  replacementId = 100,
): Record<string, any> {
  const isTransfer = kind === "transfer";
  const transfers = isTransfer
    ? [
        {
          out_id: 3,
          in_id: replacementId,
          position: "DEF",
          out_purchase_price_tenths: 50,
          out_current_price_tenths: 50,
          out_selling_price_tenths: 50,
          in_price_tenths: 50,
          out: playerReference(3, "DEF"),
          in: playerReference(replacementId, "DEF"),
        },
      ]
    : [];
  const squadIds = Array.from({ length: 15 }, (_, index) => index + 1);
  const starters = [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15];
  if (isTransfer) {
    squadIds[squadIds.indexOf(3)] = replacementId;
    starters[starters.indexOf(3)] = replacementId;
  }
  return {
    kind,
    transfer_count: transfers.length,
    transfers,
    squad_ids: squadIds,
    gameweeks: [
      {
        gameweek: 7,
        starting_ids: starters,
        captain_id: 13,
        formation: "3-4-3",
        projected_points: 72.5,
      },
      {
        gameweek: 8,
        starting_ids: starters,
        captain_id: 13,
        formation: "3-4-3",
        projected_points: 70,
      },
    ],
    bank_before_tenths: 10,
    bank_after_tenths: 10,
    free_transfers_before: 1,
    free_transfers_next_gameweek: isTransfer ? 1 : 2,
    hit_points: 0,
    projected_points: 132,
    banked_ft_value_points: isTransfer ? 0 : 0.8,
    decision_value_points: isTransfer ? 132 : 132.8,
    net_points_vs_roll: isTransfer ? -0.2 : 0,
    decision_value_vs_roll: isTransfer ? -0.8 : 0,
    explanation: isTransfer ? "Sælg Player 3 og køb Player 100." : "Gem transferen.",
  };
}

function plannerPayload(): Record<string, any> {
  return {
    manager_id: 123,
    state_fingerprint: "a".repeat(64),
    source_event: 6,
    target_event: 7,
    confirmed_state: {
      bank_tenths: 10,
      free_transfers: 1,
      no_active_chip_confirmed: true,
      squad: Array.from({ length: 15 }, (_, index) => ownedPlayerReference(index + 1)),
    },
    best_action: action("roll"),
    alternatives: [action("transfer")],
    sequential: null,
    strategy: null,
    chip_strategy: null,
    method: {
      candidate_count: 45,
      plans_per_transfer_count: 5,
      higher_transfer_count_plans: 1,
      maximum_immediate_transfers: 2,
      roll_ft_value_points: 0.8,
      chips_modelled: false,
      bounded_roadmap_modelled: false,
      next_deadline_transfer_modelled: false,
      future_transfers_modelled: false,
    },
  };
}

function sequentialPlan(payload: Record<string, any>): Record<string, any> {
  const secondTransfer = {
    out_id: 4,
    in_id: 101,
    position: "DEF",
    out_purchase_price_tenths: 50,
    out_current_price_tenths: 50,
    out_selling_price_tenths: 50,
    in_price_tenths: 50,
    out: playerReference(4, "DEF"),
    in: playerReference(101, "DEF"),
  };
  const secondSquad = [...payload.best_action.squad_ids];
  secondSquad[secondSquad.indexOf(4)] = 101;
  const secondStarters = [...payload.best_action.gameweeks[1].starting_ids];
  secondStarters[secondStarters.indexOf(4)] = 101;
  const firstWeightedPoints = 72.5;
  const secondWeightedPoints = 73 * 0.85;
  return {
    horizon: 2,
    gw_weights: [1, 0.85],
    first_step_candidate_count: 2,
    first_step_search: "explicit_bounded_transfer_plans",
    future_price_assumption: "fixed_current_prices",
    solver_proven_optimal_within_bounds: true,
    globally_optimal: false,
    best_sequence: {
      first_action: {
        source: "best_action",
        alternative_index: null,
      },
      steps: [
        {
          deadline_offset: 1,
          provisional: false,
          target_event: 7,
          kind: "roll",
          transfer_count: 0,
          transfers: [],
          squad_ids: [...payload.best_action.squad_ids],
          gameweeks: [structuredClone(payload.best_action.gameweeks[0])],
          bank_before_tenths: 10,
          bank_after_tenths: 10,
          free_transfers_before: 1,
          free_transfers_next_gameweek: 2,
          hit_points: 0,
          weighted_projected_points: firstWeightedPoints,
          weighted_hit_cost_points: 0,
        },
        {
          deadline_offset: 2,
          provisional: true,
          target_event: 8,
          kind: "transfer",
          transfer_count: 1,
          transfers: [secondTransfer],
          squad_ids: secondSquad,
          gameweeks: [{
            gameweek: 8,
            starting_ids: secondStarters,
            captain_id: 13,
            formation: "3-4-3",
            projected_points: 73,
          }],
          bank_before_tenths: 10,
          bank_after_tenths: 10,
          free_transfers_before: 2,
          free_transfers_next_gameweek: 2,
          hit_points: 0,
          weighted_projected_points: secondWeightedPoints,
          weighted_hit_cost_points: 0,
        },
      ],
      weighted_projected_points: firstWeightedPoints + secondWeightedPoints,
      total_hit_points: 0,
      weighted_hit_cost_points: 0,
      terminal_banked_ft_value_points: 0.8,
      decision_value_points: firstWeightedPoints + secondWeightedPoints + 0.8,
    },
  };
}

function strategyPlan(payload: Record<string, any>): Record<string, any> {
  const gameweekWindow = Array.from({ length: 8 }, (_, index) => 7 + index);
  const weights = gameweekWindow.map((_, index) => 0.85 ** index);
  const points = [72.5, 70, 68, 66, 64, 62, 60, 58];
  const starters = [...payload.best_action.gameweeks[0].starting_ids];
  const gameweek = (index: number) => ({
    gameweek: gameweekWindow[index],
    starting_ids: [...starters],
    captain_id: 13,
    formation: "3-4-3",
    projected_points: points[index],
  });
  const step = (
    deadlineOffset: number,
    freeTransfersBefore: number,
    governedOffsets: number[],
  ) => ({
    deadline_offset: deadlineOffset,
    provisional: deadlineOffset > 1,
    target_event: gameweekWindow[deadlineOffset - 1],
    kind: "roll",
    transfer_count: 0,
    transfers: [],
    squad_ids: [...payload.best_action.squad_ids],
    gameweeks: governedOffsets.map(gameweek),
    bank_before_tenths: 10,
    bank_after_tenths: 10,
    free_transfers_before: freeTransfersBefore,
    free_transfers_next_gameweek: Math.min(5, freeTransfersBefore + 1),
    hit_points: 0,
    weighted_projected_points: governedOffsets.reduce(
      (total, offset) => total + points[offset] * weights[offset],
      0,
    ),
    weighted_hit_cost_points: 0,
  });
  const steps = [
    step(1, 1, [0]),
    step(2, 2, [1]),
    step(3, 3, [2]),
    step(4, 4, [3, 4, 5, 6, 7]),
  ];
  const weightedProjectedPoints = steps.reduce(
    (total, value) => total + value.weighted_projected_points,
    0,
  );
  const terminalFtValue = 3.2;
  return {
    horizon: 8,
    gameweek_window: gameweekWindow,
    gw_weights: weights,
    modelled_deadlines: 4,
    maximum_provisional_transfers: 2,
    first_step_candidate_count: 2,
    first_step_search: "explicit_bounded_transfer_plans",
    search_scope: "four_deadlines_max_two_provisional_transfers",
    future_price_assumption: "fixed_current_prices",
    assumptions: [
      "fixed_current_prices",
      "four_transfer_deadlines_modelled",
      "maximum_two_transfers_at_each_provisional_deadline",
      "first_action_restricted_to_supplied_explicit_plans",
      "no_transfers_after_deadline_four_assumed",
      "recalculate_at_every_real_deadline",
    ],
    solver_proven_optimal_within_bounds: true,
    globally_optimal: false,
    recalculate_each_deadline: true,
    first_action: { source: "best_action", alternative_index: null },
    steps,
    weighted_projected_points: weightedProjectedPoints,
    total_hit_points: 0,
    weighted_hit_cost_points: 0,
    terminal_banked_ft_value_points: terminalFtValue,
    decision_value_points: weightedProjectedPoints + terminalFtValue,
  };
}

function chipStrategy(payload: Record<string, any>): Record<string, any> {
  const strategy = payload.strategy;
  const wildcardSquad = Array.from({ length: 15 }, (_, index) => playerReference(index + 1));
  return {
    horizon: strategy.horizon,
    target_event: 7,
    inventory: ["wildcard", "freehit", "bboost", "3xc"].map((chip) => ({
      chip,
      used_events: [],
      available_for_target: true,
    })),
    scenarios: [
      {
        scenario_id: "wildcard-gw7",
        chip: "wildcard",
        event: 7,
        signal: "hold",
        available: true,
        estimated_gain_points: 0,
        baseline_points: strategy.decision_value_points,
        chip_points: strategy.decision_value_points,
        confidence: "low",
        model_scope: "multiweek_rebuild",
        reason: "Bounded Wildcard comparison.",
        squad: wildcardSquad,
        change_count: 0,
        bank_after_tenths: 10,
      },
      {
        scenario_id: "freehit-no-confirmed-trigger",
        chip: "freehit",
        event: null,
        signal: "hold",
        available: true,
        estimated_gain_points: null,
        baseline_points: null,
        chip_points: null,
        confidence: "low",
        model_scope: "confirmed_blank_double_screen",
        reason: "No confirmed blank or double.",
        squad: [],
        change_count: null,
        bank_after_tenths: null,
      },
      {
        scenario_id: "bboost-gw8",
        chip: "bboost",
        event: 8,
        signal: "watch",
        available: true,
        estimated_gain_points: 5,
        baseline_points: 70,
        chip_points: 75,
        confidence: "medium",
        model_scope: "bench_marginal",
        reason: "Best bench margin.",
        squad: [],
        change_count: null,
        bank_after_tenths: null,
      },
      {
        scenario_id: "3xc-gw7",
        chip: "3xc",
        event: 7,
        signal: "consider",
        available: true,
        estimated_gain_points: 8,
        baseline_points: 72.5,
        chip_points: 80.5,
        confidence: "medium",
        model_scope: "captain_marginal",
        reason: "Best captain margin.",
        squad: [],
        change_count: null,
        bank_after_tenths: null,
      },
    ],
    recommendation: {
      action: "consider",
      scenario_id: "3xc-gw7",
      chip: "3xc",
      event: 7,
      reason: "Review the quantified current scenario manually.",
    },
    model_scope: "bounded_chip_counterfactuals",
    globally_optimal: false,
    recalculate_each_deadline: true,
  };
}

test("validates the exact manager sync response and its cross-field invariants", () => {
  const payload = syncResponse();
  assert.strictEqual(parseManagerSyncResponse(payload), payload);
  assert.equal(isManagerSyncResponse(payload), true);

  const freeHit = syncResponse();
  freeHit.last_deadline_state.active_chip = "freehit";
  freeHit.last_deadline_state.chip_usage = [{ name: "freehit", event: 6 }];
  freeHit.manual_state_template.state.chip_usage = [{ name: "freehit", event: 6 }];
  freeHit.manual_state_template.state.chips.freehit = "used";
  freeHit.warnings.push({
    code: "free_hit_squad_is_temporary",
    message: "Confirm the permanent squad.",
  });
  assert.equal(isManagerSyncResponse(freeHit), true);
});

test("sync validation fails closed for unknown, contradictory, or credential-shaped fields", () => {
  const unknown = syncResponse();
  unknown.manager.access_token = "secret";
  assert.throws(() => parseManagerSyncResponse(unknown), PlannerContractError);

  const wrongSignal = syncResponse();
  wrongSignal.last_deadline_state.picks[0].official_price_signal.element_id = 999;
  assert.equal(isManagerSyncResponse(wrongSignal), false);

  const wrongDraft = syncResponse();
  wrongDraft.manual_state_template.state.player_prices[0].selling_price_tenths = 49;
  assert.throws(
    () => parseManagerSyncResponse(wrongDraft),
    /must match the public price estimates/,
  );

  const wrongChipStatus = syncResponse();
  wrongChipStatus.manual_state_template.state.chips.wildcard = "used";
  assert.throws(
    () => parseManagerSyncResponse(wrongChipStatus),
    /inconsistent with chip_usage/,
  );

  const missingChip = syncResponse();
  delete missingChip.manual_state_template.state.chips["3xc"];
  assert.throws(() => parseManagerSyncResponse(missingChip), PlannerContractError);

  const duplicateChipEvent = syncResponse();
  duplicateChipEvent.last_deadline_state.chip_usage = [
    { name: "freehit", event: 2 },
    { name: "wildcard", event: 2 },
  ];
  assert.throws(() => parseManagerSyncResponse(duplicateChipEvent), /more than one chip/);

  const consecutiveBoundaryFreeHits = syncResponse();
  consecutiveBoundaryFreeHits.last_deadline_state.event = 20;
  consecutiveBoundaryFreeHits.target.event = 21;
  consecutiveBoundaryFreeHits.last_deadline_state.active_chip = "freehit";
  consecutiveBoundaryFreeHits.last_deadline_state.chip_usage = [
    { name: "freehit", event: 19 },
    { name: "freehit", event: 20 },
  ];
  consecutiveBoundaryFreeHits.manual_state_template.state.effective_event = 21;
  consecutiveBoundaryFreeHits.manual_state_template.state.chip_usage = [
    { name: "freehit", event: 19 },
    { name: "freehit", event: 20 },
  ];
  consecutiveBoundaryFreeHits.manual_state_template.state.chips.freehit = "used";
  consecutiveBoundaryFreeHits.warnings.push({
    code: "free_hit_squad_is_temporary",
    message: "Confirm the permanent squad.",
  });
  assert.throws(
    () => parseManagerSyncResponse(consecutiveBoundaryFreeHits),
    /consecutive gameweeks 19 and 20/,
  );

  const missingWarning = syncResponse();
  missingWarning.warnings = [];
  assert.equal(isManagerSyncResponse(missingWarning), false);

  const degradedPrices = syncResponse();
  degradedPrices.price_signals.available = false;
  degradedPrices.price_signals.player_count = 0;
  degradedPrices.price_signals.price_change_deadlines = [];
  degradedPrices.price_signals.warning = "Optional price signals are unavailable.";
  for (const pick of degradedPrices.last_deadline_state.picks) {
    pick.official_price_signal = null;
  }
  assert.equal(isManagerSyncResponse(degradedPrices), true);
});

test("parses manager id, bank in tenths, and free transfers without coercion", () => {
  assert.equal(parseManagerId(" 123456 "), 123456);
  assert.equal(parseBankTenths("1,2"), 12);
  assert.equal(parseBankTenths("1.2"), 12);
  assert.equal(parseBankTenths("£0.1m"), 1);
  assert.equal(parseBankTenths("10"), 100);
  assert.equal(parseFreeTransfers(" 5 "), 5);

  for (const value of ["", "0", "1.2", "+7", "01"]) {
    assert.throws(() => parseManagerId(value), PlannerContractError);
  }
  for (const value of ["-0.1", "1,25", "1.000", "£1.2bn", "NaN"]) {
    assert.throws(() => parseBankTenths(value), PlannerContractError);
  }
  assert.throws(() => parseBankTenths("100.1"), PlannerContractError);
  for (const value of ["0", "6", "1.0", "two"]) {
    assert.throws(() => parseFreeTransfers(value), PlannerContractError);
  }
});

test("serializes only the allowlisted compute request and always disables Solio", () => {
  const sync = syncResponse();
  const input = {
    manager_id: sync.manager.id,
    source_event: sync.last_deadline_state.event,
    state_fingerprint: sync.snapshot.checksum_sha256,
    manager_state: {
      ...sync.manual_state_template.state,
      bank_tenths: 12,
      free_transfers: 3,
      no_active_chip_confirmed: true,
    },
    horizon: 4,
    include_doubtful: true,
    forecast_version: "v2" as const,
    password: "never-copy-me",
    access_token: "never-copy-me-either",
    use_solio: true,
  };
  const serialized = serializePlannerRequest(input);
  assert.deepEqual(Object.keys(serialized), [
    "manager_id",
    "source_event",
    "state_fingerprint",
    "manager_state",
    "horizon",
    "include_doubtful",
    "use_solio",
    "forecast_version",
  ]);
  assert.equal(serialized.use_solio, false);
  assert.notStrictEqual(serialized.manager_state, input.manager_state);
  assert.doesNotMatch(JSON.stringify(serialized), /password|access_token|never-copy-me/);
  assert.deepEqual(JSON.parse(stringifyPlannerRequest(input)), serialized);

  const chipNotConfirmed = structuredClone(input);
  chipNotConfirmed.manager_state.no_active_chip_confirmed = false;
  assert.throws(() => serializePlannerRequest(chipNotConfirmed), /must be confirmed/);

  const unsafeNested = structuredClone(input);
  unsafeNested.manager_state.player_prices[0].token = "secret";
  assert.throws(() => serializePlannerRequest(unsafeNested), PlannerContractError);
});

test("validates planner actions, enriched transfers, confirmed state, and method", () => {
  const payload = plannerPayload();
  assert.strictEqual(parsePlannerPayload(payload), payload);
  assert.equal(isPlannerPayload(payload), true);
  assert.equal(parsePlannerPayload(payload).best_action.kind, "roll");
  assert.equal(parsePlannerPayload(payload).alternatives[0].transfers[0].in.id, 100);

  const fiveFreeTransfers = plannerPayload();
  fiveFreeTransfers.confirmed_state.free_transfers = 5;
  fiveFreeTransfers.best_action.free_transfers_before = 5;
  fiveFreeTransfers.best_action.free_transfers_next_gameweek = 5;
  fiveFreeTransfers.alternatives[0].free_transfers_before = 5;
  fiveFreeTransfers.alternatives[0].free_transfers_next_gameweek = 5;
  fiveFreeTransfers.method.maximum_immediate_transfers = 5;
  assert.equal(isPlannerPayload(fiveFreeTransfers), true);
});

test("validates an additive bounded two-deadline sequence and reconciles its totals", () => {
  const payload = plannerPayload();
  payload.sequential = sequentialPlan(payload);
  payload.method.next_deadline_transfer_modelled = true;

  const parsed = parsePlannerPayload(payload);

  assert.strictEqual(parsed, payload);
  assert.equal(parsed.sequential?.best_sequence.first_action.source, "best_action");
  assert.equal(parsed.sequential?.best_sequence.steps[1].target_event, 8);
  assert.equal(parsed.sequential?.best_sequence.steps[1].transfers[0].in_id, 101);
  assert.equal(parsed.method.future_transfers_modelled, false);
});

test("validates a bounded eight-gameweek roadmap and four personalized chip scenarios", () => {
  const payload = plannerPayload();
  payload.strategy = strategyPlan(payload);
  payload.chip_strategy = chipStrategy(payload);
  payload.method.bounded_roadmap_modelled = true;
  payload.method.chips_modelled = true;
  payload.method.next_deadline_transfer_modelled = true;

  const parsed = parsePlannerPayload(payload);

  assert.strictEqual(parsed, payload);
  assert.equal(parsed.strategy?.horizon, 8);
  assert.equal(parsed.strategy?.steps.length, 4);
  assert.equal(parsed.strategy?.steps[3].gameweeks.length, 5);
  assert.equal(parsed.strategy?.globally_optimal, false);
  assert.equal(parsed.chip_strategy?.inventory.length, 4);
  assert.equal(parsed.chip_strategy?.recommendation.scenario_id, "3xc-gw7");
  assert.equal(parsed.method.future_transfers_modelled, false);
});

test("validates a solved Free Hit counterfactual with its temporary squad", () => {
  const payload = plannerPayload();
  payload.strategy = strategyPlan(payload);
  payload.chip_strategy = chipStrategy(payload);
  payload.chip_strategy.scenarios[1] = {
    scenario_id: "freehit-gw8",
    chip: "freehit",
    event: 8,
    signal: "watch",
    available: true,
    estimated_gain_points: 5,
    baseline_points: 70,
    chip_points: 75,
    confidence: "low",
    model_scope: "single_gameweek_counterfactual",
    reason: "One temporary squad solved against the permanent roadmap team.",
    squad: Array.from({ length: 15 }, (_, index) => playerReference(index + 1)),
    change_count: 0,
    bank_after_tenths: null,
  };
  payload.method.bounded_roadmap_modelled = true;
  payload.method.chips_modelled = true;
  payload.method.next_deadline_transfer_modelled = true;

  const parsed = parsePlannerPayload(payload);
  const freeHit = parsed.chip_strategy?.scenarios.find((scenario) => scenario.chip === "freehit");
  assert.equal(freeHit?.model_scope, "single_gameweek_counterfactual");
  assert.equal(freeHit?.squad.length, 15);
  assert.equal(freeHit?.change_count, 0);
});

test("roadmap validation rejects a broken window, first action, provisional flag and totals", () => {
  const brokenWindow = plannerPayload();
  brokenWindow.strategy = strategyPlan(brokenWindow);
  brokenWindow.strategy.gameweek_window[4] += 1;
  brokenWindow.method.bounded_roadmap_modelled = true;
  brokenWindow.method.next_deadline_transfer_modelled = true;
  assert.throws(() => parsePlannerPayload(brokenWindow), /contiguous gameweek window/);

  const wrongFirstAction = plannerPayload();
  wrongFirstAction.strategy = strategyPlan(wrongFirstAction);
  wrongFirstAction.strategy.steps[0].free_transfers_next_gameweek = 1;
  wrongFirstAction.method.bounded_roadmap_modelled = true;
  wrongFirstAction.method.next_deadline_transfer_modelled = true;
  assert.throws(() => parsePlannerPayload(wrongFirstAction), /free-transfer rule|referenced visible action/);

  const executableFuture = plannerPayload();
  executableFuture.strategy = strategyPlan(executableFuture);
  executableFuture.strategy.steps[2].provisional = false;
  executableFuture.method.bounded_roadmap_modelled = true;
  executableFuture.method.next_deadline_transfer_modelled = true;
  assert.throws(() => parsePlannerPayload(executableFuture), /provisional/);

  const tooManyFutureTransfers = plannerPayload();
  tooManyFutureTransfers.strategy = strategyPlan(tooManyFutureTransfers);
  tooManyFutureTransfers.strategy.steps[1].transfer_count = 3;
  tooManyFutureTransfers.method.bounded_roadmap_modelled = true;
  tooManyFutureTransfers.method.next_deadline_transfer_modelled = true;
  assert.throws(() => parsePlannerPayload(tooManyFutureTransfers), /exceeds the bounded/);

  const wrongTotal = plannerPayload();
  wrongTotal.strategy = strategyPlan(wrongTotal);
  wrongTotal.strategy.decision_value_points += 1;
  wrongTotal.method.bounded_roadmap_modelled = true;
  wrongTotal.method.next_deadline_transfer_modelled = true;
  assert.throws(() => parsePlannerPayload(wrongTotal), /does not reconcile/);
});

test("chip strategy validation rejects inventory, scenario and recommendation contradictions", () => {
  const wrongAvailability = plannerPayload();
  wrongAvailability.strategy = strategyPlan(wrongAvailability);
  wrongAvailability.chip_strategy = chipStrategy(wrongAvailability);
  wrongAvailability.chip_strategy.inventory[0].available_for_target = false;
  wrongAvailability.method.bounded_roadmap_modelled = true;
  wrongAvailability.method.chips_modelled = true;
  wrongAvailability.method.next_deadline_transfer_modelled = true;
  assert.throws(() => parsePlannerPayload(wrongAvailability), /available_for_target/);

  const duplicateScenario = plannerPayload();
  duplicateScenario.strategy = strategyPlan(duplicateScenario);
  duplicateScenario.chip_strategy = chipStrategy(duplicateScenario);
  duplicateScenario.chip_strategy.scenarios[1] = structuredClone(
    duplicateScenario.chip_strategy.scenarios[0],
  );
  duplicateScenario.method.bounded_roadmap_modelled = true;
  duplicateScenario.method.chips_modelled = true;
  duplicateScenario.method.next_deadline_transfer_modelled = true;
  assert.throws(() => parsePlannerPayload(duplicateScenario), /four unique chip scenarios/);

  const partialWildcard = plannerPayload();
  partialWildcard.strategy = strategyPlan(partialWildcard);
  partialWildcard.chip_strategy = chipStrategy(partialWildcard);
  partialWildcard.chip_strategy.scenarios[0].squad.pop();
  partialWildcard.method.bounded_roadmap_modelled = true;
  partialWildcard.method.chips_modelled = true;
  partialWildcard.method.next_deadline_transfer_modelled = true;
  assert.throws(() => parsePlannerPayload(partialWildcard), /zero or exactly 15/);

  const inventedRecommendation = plannerPayload();
  inventedRecommendation.strategy = strategyPlan(inventedRecommendation);
  inventedRecommendation.chip_strategy = chipStrategy(inventedRecommendation);
  inventedRecommendation.chip_strategy.recommendation.scenario_id = "wildcard-gw7";
  inventedRecommendation.chip_strategy.recommendation.chip = "wildcard";
  inventedRecommendation.method.bounded_roadmap_modelled = true;
  inventedRecommendation.method.chips_modelled = true;
  inventedRecommendation.method.next_deadline_transfer_modelled = true;
  assert.throws(() => parsePlannerPayload(inventedRecommendation), /available consider scenario/);

  const falseMethodFlag = plannerPayload();
  falseMethodFlag.strategy = strategyPlan(falseMethodFlag);
  falseMethodFlag.method.next_deadline_transfer_modelled = true;
  assert.throws(() => parsePlannerPayload(falseMethodFlag), /bounded_roadmap_modelled/);
});

test("sequential validation rejects unknown fields, a mismatched first action and broken chaining", () => {
  const withSecret = plannerPayload();
  withSecret.sequential = sequentialPlan(withSecret);
  withSecret.sequential.api_token = "never-accept";
  withSecret.method.next_deadline_transfer_modelled = true;
  assert.throws(() => parsePlannerPayload(withSecret), PlannerContractError);

  const firstActionMismatch = plannerPayload();
  firstActionMismatch.sequential = sequentialPlan(firstActionMismatch);
  firstActionMismatch.method.next_deadline_transfer_modelled = true;
  firstActionMismatch.sequential.best_sequence.steps[0].gameweeks[0].projected_points = 73;
  firstActionMismatch.sequential.best_sequence.steps[0].weighted_projected_points = 73;
  assert.throws(
    () => parsePlannerPayload(firstActionMismatch),
    /must exactly match the selected existing action/,
  );

  const brokenSecondBank = plannerPayload();
  brokenSecondBank.sequential = sequentialPlan(brokenSecondBank);
  brokenSecondBank.method.next_deadline_transfer_modelled = true;
  brokenSecondBank.sequential.best_sequence.steps[1].bank_before_tenths = 11;
  brokenSecondBank.sequential.best_sequence.steps[1].bank_after_tenths = 11;
  assert.throws(() => parsePlannerPayload(brokenSecondBank), /reconcile sequentially/);

  const wrongTailGameweek = plannerPayload();
  wrongTailGameweek.sequential = sequentialPlan(wrongTailGameweek);
  wrongTailGameweek.method.next_deadline_transfer_modelled = true;
  wrongTailGameweek.sequential.best_sequence.steps[1].gameweeks[0].gameweek = 9;
  assert.throws(() => parsePlannerPayload(wrongTailGameweek), /must be 8/);

  const changedOriginalPurchaseBasis = plannerPayload();
  changedOriginalPurchaseBasis.sequential = sequentialPlan(changedOriginalPurchaseBasis);
  changedOriginalPurchaseBasis.method.next_deadline_transfer_modelled = true;
  const secondMove = changedOriginalPurchaseBasis.sequential.best_sequence.steps[1].transfers[0];
  secondMove.out_purchase_price_tenths = 49;
  secondMove.out_selling_price_tenths = 49;
  changedOriginalPurchaseBasis.sequential.best_sequence.steps[1].bank_after_tenths = 9;
  assert.throws(() => parsePlannerPayload(changedOriginalPurchaseBasis), /first-step purchase price/);

  const undercountedCandidatePool = plannerPayload();
  undercountedCandidatePool.sequential = sequentialPlan(undercountedCandidatePool);
  undercountedCandidatePool.method.next_deadline_transfer_modelled = true;
  undercountedCandidatePool.method.candidate_count = 16;
  assert.throws(() => parsePlannerPayload(undercountedCandidatePool), /referenced candidate set/);
});

test("sequential validation requires reconciled totals and a matching method flag", () => {
  const wrongTotal = plannerPayload();
  wrongTotal.sequential = sequentialPlan(wrongTotal);
  wrongTotal.method.next_deadline_transfer_modelled = true;
  wrongTotal.sequential.best_sequence.decision_value_points += 1;
  assert.throws(() => parsePlannerPayload(wrongTotal), /does not reconcile/);

  const missingFlag = plannerPayload();
  missingFlag.sequential = sequentialPlan(missingFlag);
  assert.throws(() => parsePlannerPayload(missingFlag), /true exactly when/);

  const falsePositiveFlag = plannerPayload();
  falsePositiveFlag.method.next_deadline_transfer_modelled = true;
  assert.throws(() => parsePlannerPayload(falsePositiveFlag), /true exactly when/);
});

test("a one-gameweek planner must not expose a sequential plan", () => {
  const payload = plannerPayload();
  payload.sequential = sequentialPlan(payload);
  payload.method.next_deadline_transfer_modelled = true;
  payload.best_action.gameweeks = payload.best_action.gameweeks.slice(0, 1);
  payload.alternatives[0].gameweeks = payload.alternatives[0].gameweeks.slice(0, 1);
  assert.throws(() => parsePlannerPayload(payload), /must be null/);

  payload.sequential = null;
  payload.method.next_deadline_transfer_modelled = false;
  assert.equal(isPlannerPayload(payload), true);
});

test("planner validation rejects action and budget contradictions or extra fields", () => {
  const wrongKind = plannerPayload();
  wrongKind.best_action.kind = "transfer";
  assert.throws(() => parsePlannerPayload(wrongKind), /must be roll/);

  const wrongBank = plannerPayload();
  wrongBank.alternatives[0].bank_after_tenths = 11;
  assert.equal(isPlannerPayload(wrongBank), false);

  const changedConfirmedPriceBasis = plannerPayload();
  const changedTransfer = changedConfirmedPriceBasis.alternatives[0].transfers[0];
  changedTransfer.out_purchase_price_tenths = 49;
  changedTransfer.out_selling_price_tenths = 49;
  changedConfirmedPriceBasis.alternatives[0].bank_after_tenths = 9;
  assert.throws(() => parsePlannerPayload(changedConfirmedPriceBasis), /confirmed purchase and selling prices/);

  const wrongSquad = plannerPayload();
  wrongSquad.alternatives[0].squad_ids[0] = 999;
  assert.throws(() => parsePlannerPayload(wrongSquad), /confirmed squad after transfers/);

  const leakedToken = plannerPayload();
  leakedToken.method.token = "secret";
  assert.equal(isPlannerPayload(leakedToken), false);
});
