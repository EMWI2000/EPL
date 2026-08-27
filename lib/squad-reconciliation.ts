import type {
  ManagerSyncPlayerCatalogEntry,
  PlayerPriceState,
  PlannerOwnedPlayerReference,
} from "@/lib/planner-contract";

export interface ReconciledSquad {
  current_squad_ids: number[];
  player_prices: PlayerPriceState[];
}

export class SquadReconciliationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SquadReconciliationError";
  }
}

export function sellingPriceTenths(purchasePriceTenths: number, currentPriceTenths: number): number {
  if (!Number.isInteger(purchasePriceTenths) || purchasePriceTenths < 1 || purchasePriceTenths > 500) {
    throw new SquadReconciliationError("Købsprisen skal være en gyldig FPL-pris.");
  }
  if (!Number.isInteger(currentPriceTenths) || currentPriceTenths < 1 || currentPriceTenths > 500) {
    throw new SquadReconciliationError("Den aktuelle spillerpris er ugyldig.");
  }
  if (currentPriceTenths <= purchasePriceTenths) return currentPriceTenths;
  return purchasePriceTenths + Math.floor((currentPriceTenths - purchasePriceTenths) / 2);
}

export function parsePlayerPriceTenths(value: string): number {
  if (typeof value !== "string") {
    throw new SquadReconciliationError("Købsprisen skal være tekst.");
  }
  const normalized = value.trim().replace(",", ".");
  if (!/^\d{1,2}(?:\.\d)?$/.test(normalized)) {
    throw new SquadReconciliationError("Skriv købsprisen med højst én decimal, fx 5,0.");
  }
  const [whole, decimal = "0"] = normalized.split(".");
  const tenths = Number(whole) * 10 + Number(decimal);
  if (!Number.isInteger(tenths) || tenths < 1 || tenths > 500) {
    throw new SquadReconciliationError("Købsprisen skal være mellem £0,1m og £50,0m.");
  }
  return tenths;
}

function validateCurrentState(state: ReconciledSquad): void {
  if (
    state.current_squad_ids.length !== 15 ||
    new Set(state.current_squad_ids).size !== 15 ||
    state.current_squad_ids.some((id) => !Number.isInteger(id) || id < 1)
  ) {
    throw new SquadReconciliationError("Den aktuelle trup skal indeholde 15 unikke spillere.");
  }
  const priceIds = state.player_prices.map((row) => row.element_id);
  if (
    state.player_prices.length !== 15 ||
    new Set(priceIds).size !== 15 ||
    priceIds.some((id) => !state.current_squad_ids.includes(id))
  ) {
    throw new SquadReconciliationError("Der skal være én pris for hver spiller i truppen.");
  }
}

export function applySquadReplacement(
  state: ReconciledSquad,
  playerCatalog: readonly ManagerSyncPlayerCatalogEntry[],
  replacement: {
    out_id: number;
    in_id: number;
    purchase_price_tenths: number;
  },
): ReconciledSquad {
  validateCurrentState(state);
  const catalogById = new Map(playerCatalog.map((player) => [player.id, player]));
  if (catalogById.size !== playerCatalog.length) {
    throw new SquadReconciliationError("Spillerkataloget indeholder dubletter.");
  }
  const outgoing = catalogById.get(replacement.out_id);
  const incoming = catalogById.get(replacement.in_id);
  if (!outgoing || !incoming) {
    throw new SquadReconciliationError("Vælg to gyldige spillere.");
  }
  const outIndex = state.current_squad_ids.indexOf(outgoing.id);
  if (outIndex < 0) {
    throw new SquadReconciliationError(`${outgoing.display_name} er ikke i den aktuelle trup.`);
  }
  if (state.current_squad_ids.includes(incoming.id)) {
    throw new SquadReconciliationError(`${incoming.display_name} er allerede i den aktuelle trup.`);
  }
  if (outgoing.position !== incoming.position) {
    throw new SquadReconciliationError("Spiller ud og spiller ind skal have samme position.");
  }

  const sellingPrice = sellingPriceTenths(
    replacement.purchase_price_tenths,
    incoming.current_price_tenths,
  );
  const currentSquadIds = [...state.current_squad_ids];
  currentSquadIds[outIndex] = incoming.id;
  const clubCounts = new Map<number, number>();
  for (const playerId of currentSquadIds) {
    const player = catalogById.get(playerId);
    if (!player) throw new SquadReconciliationError("Truppen indeholder en ukendt spiller.");
    clubCounts.set(player.team_id, (clubCounts.get(player.team_id) ?? 0) + 1);
  }
  if ([...clubCounts.values()].some((count) => count > 3)) {
    throw new SquadReconciliationError("Ændringen giver mere end tre spillere fra samme klub.");
  }
  const priceIndex = state.player_prices.findIndex((row) => row.element_id === outgoing.id);
  if (priceIndex < 0) {
    throw new SquadReconciliationError(`Prisen for ${outgoing.display_name} mangler.`);
  }
  const playerPrices = state.player_prices.map((row) => ({ ...row }));
  playerPrices[priceIndex] = {
    element_id: incoming.id,
    purchase_price_tenths: replacement.purchase_price_tenths,
    selling_price_tenths: sellingPrice,
  };
  const result = { current_squad_ids: currentSquadIds, player_prices: playerPrices };
  validateCurrentState(result);
  return result;
}

export function reconciledSquadMatchesConfirmed(
  state: ReconciledSquad,
  confirmedSquad: readonly PlannerOwnedPlayerReference[],
): boolean {
  if (confirmedSquad.length !== 15) return false;
  const confirmedById = new Map(confirmedSquad.map((player) => [player.id, player]));
  if (confirmedById.size !== 15 || state.current_squad_ids.some((id) => !confirmedById.has(id))) {
    return false;
  }
  return state.player_prices.every((price) => {
    const confirmed = confirmedById.get(price.element_id);
    return confirmed?.purchase_price_tenths === price.purchase_price_tenths &&
      confirmed.selling_price_tenths === price.selling_price_tenths;
  });
}
