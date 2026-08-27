import assert from "node:assert/strict";
import test from "node:test";

import type {
  ManagerSyncPlayerCatalogEntry,
  PlayerPriceState,
  PlannerOwnedPlayerReference,
} from "../lib/planner-contract.ts";
import {
  applySquadReplacement,
  parsePlayerPriceTenths,
  reconciledSquadMatchesConfirmed,
  sellingPriceTenths,
} from "../lib/squad-reconciliation.ts";

function catalog(): ManagerSyncPlayerCatalogEntry[] {
  return Array.from({ length: 16 }, (_, index) => ({
    id: index + 1,
    display_name: `Player ${index + 1}`,
    position: index < 2 ? "GKP" : index < 8 ? "DEF" : index < 13 ? "MID" : "FWD",
    team_id: (index % 10) + 1,
    team_name: `Club ${(index % 10) + 1}`,
    current_price_tenths: index === 15 ? 57 : 50,
  }));
}

function prices(): PlayerPriceState[] {
  return Array.from({ length: 15 }, (_, index) => ({
    element_id: index + 1,
    purchase_price_tenths: 50,
    selling_price_tenths: 50,
  }));
}

test("replaces one owned player while preserving the 15-player slot and derives selling price", () => {
  const state = applySquadReplacement(
    { current_squad_ids: Array.from({ length: 15 }, (_, index) => index + 1), player_prices: prices() },
    catalog(),
    { out_id: 15, in_id: 16, purchase_price_tenths: 55 },
  );

  assert.deepEqual(state.current_squad_ids.slice(-2), [14, 16]);
  assert.deepEqual(state.player_prices.slice(-1), [{
    element_id: 16,
    purchase_price_tenths: 55,
    selling_price_tenths: 56,
  }]);
});

test("rejects an unowned outgoing player, an already-owned incoming player, and a position mismatch", () => {
  const state = {
    current_squad_ids: Array.from({ length: 15 }, (_, index) => index + 1),
    player_prices: prices(),
  };
  assert.throws(
    () => applySquadReplacement(state, catalog(), { out_id: 16, in_id: 15, purchase_price_tenths: 50 }),
    /not in the current squad|ikke i den aktuelle trup/,
  );
  assert.throws(
    () => applySquadReplacement(state, catalog(), { out_id: 15, in_id: 14, purchase_price_tenths: 50 }),
    /already in the current squad|allerede i den aktuelle trup/,
  );
  assert.throws(
    () => applySquadReplacement(state, catalog(), { out_id: 3, in_id: 16, purchase_price_tenths: 50 }),
    /same position|samme position/,
  );
});

test("rejects a manual correction that would create a fourth player from one club", () => {
  const players = catalog();
  players[0].team_id = 1;
  players[1].team_id = 1;
  players[2].team_id = 1;
  players[15].team_id = 1;
  const state = {
    current_squad_ids: Array.from({ length: 15 }, (_, index) => index + 1),
    player_prices: prices(),
  };
  assert.throws(
    () => applySquadReplacement(state, players, { out_id: 15, in_id: 16, purchase_price_tenths: 50 }),
    /mere end tre spillere/,
  );
});

test("parses Danish FPL prices and applies the half-profit rule", () => {
  assert.equal(parsePlayerPriceTenths("5,5"), 55);
  assert.equal(parsePlayerPriceTenths("5"), 50);
  assert.equal(parsePlayerPriceTenths("12.0"), 120);
  assert.equal(parsePlayerPriceTenths("12"), 120);
  assert.equal(sellingPriceTenths(55, 57), 56);
  assert.equal(sellingPriceTenths(57, 55), 55);
  assert.throws(() => parsePlayerPriceTenths("5,55"), /højst én decimal/);
});

test("detects whether the recommendation used the exact reconciled ownership and prices", () => {
  const state = {
    current_squad_ids: Array.from({ length: 15 }, (_, index) => index + 1),
    player_prices: prices(),
  };
  const confirmed = catalog().slice(0, 15).map((player) => ({
    id: player.id,
    name: player.display_name,
    team: player.team_name,
    position: player.position,
    price_tenths: player.current_price_tenths,
    status: "a",
    price_signal: null,
    purchase_price_tenths: 50,
    selling_price_tenths: 50,
  })) satisfies PlannerOwnedPlayerReference[];

  assert.equal(reconciledSquadMatchesConfirmed(state, confirmed), true);
  confirmed[0].selling_price_tenths = 49;
  assert.equal(reconciledSquadMatchesConfirmed(state, confirmed), false);
});
