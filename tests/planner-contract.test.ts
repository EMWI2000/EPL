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
    schema_version: "fpl-manager-state-response-v1",
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
      schema_version: "fpl-personal-state-v1",
      state_kind: "public_last_deadline",
      event: 6,
      bank_tenths: 8,
      squad_value_tenths: 1012,
      total_transfers_at_deadline: 4,
      event_transfers: 1,
      event_transfer_cost: 0,
      active_chip: null,
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
        chips: {},
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
        "no_active_chip_confirmed",
      ],
      free_transfers_default_reason: "current_free_transfers_are_not_public",
    },
    snapshot: {
      schema_version: "fpl-deadline-state-snapshot-v1",
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
      squad: Array.from({ length: 15 }, (_, index) => playerReference(index + 1)),
    },
    best_action: action("roll"),
    alternatives: [action("transfer")],
    method: {
      candidate_count: 45,
      plans_per_transfer_count: 5,
      higher_transfer_count_plans: 1,
      maximum_immediate_transfers: 2,
      roll_ft_value_points: 0.8,
      chips_modelled: false,
      future_transfers_modelled: false,
    },
  };
}

test("validates the exact manager sync response and its cross-field invariants", () => {
  const payload = syncResponse();
  assert.strictEqual(parseManagerSyncResponse(payload), payload);
  assert.equal(isManagerSyncResponse(payload), true);

  const freeHit = syncResponse();
  freeHit.last_deadline_state.active_chip = "freehit";
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

test("planner validation rejects action and budget contradictions or extra fields", () => {
  const wrongKind = plannerPayload();
  wrongKind.best_action.kind = "transfer";
  assert.throws(() => parsePlannerPayload(wrongKind), /must be roll/);

  const wrongBank = plannerPayload();
  wrongBank.alternatives[0].bank_after_tenths = 11;
  assert.equal(isPlannerPayload(wrongBank), false);

  const wrongSquad = plannerPayload();
  wrongSquad.alternatives[0].squad_ids[0] = 999;
  assert.throws(() => parsePlannerPayload(wrongSquad), /confirmed squad after transfers/);

  const leakedToken = plannerPayload();
  leakedToken.method.token = "secret";
  assert.equal(isPlannerPayload(leakedToken), false);
});
