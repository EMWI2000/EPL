# logic/optimizer.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import pulp
from itertools import combinations

POS_343 = {"GKP": 1, "DEF": 3, "MID": 4, "FWD": 3}
POS_352 = {"GKP": 1, "DEF": 3, "MID": 5, "FWD": 2}
POS_442 = {"GKP": 1, "DEF": 4, "MID": 4, "FWD": 2}
POS_433 = {"GKP": 1, "DEF": 4, "MID": 3, "FWD": 3}
POS_451 = {"GKP": 1, "DEF": 4, "MID": 5, "FWD": 1}
POS_532 = {"GKP": 1, "DEF": 5, "MID": 3, "FWD": 2}
POS_541 = {"GKP": 1, "DEF": 5, "MID": 4, "FWD": 1}

FORMATIONS = {
    "343": POS_343,
    "352": POS_352,
    "442": POS_442,
    "433": POS_433,
    "451": POS_451,
    "532": POS_532,
    "541": POS_541,
}

SQUAD_QUOTA = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}

# Point hit for ekstra transfers
TRANSFER_HIT = 4.0


def select_starting_xi(players_df: pd.DataFrame, formation: str = "343") -> List[int]:
    """Vælger optimal starting XI baseret på formation og EP."""
    need = FORMATIONS.get(formation, POS_343)
    df = players_df.copy()
    df["ep_next_gw"] = pd.to_numeric(df.get("ep_next_gw", 0.0), errors="coerce").fillna(0.0)
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").fillna(-1).astype(int)

    df["var"] = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat="Binary") for i in df.index]
    prob = pulp.LpProblem("startXI", pulp.LpMaximize)

    prob += pulp.lpSum(df["var"] * df["ep_next_gw"])
    prob += pulp.lpSum(df["var"]) == 11

    for pos, cnt in need.items():
        prob += pulp.lpSum(df.loc[df["pos"] == pos, "var"]) == cnt

    for team_id, grp in df.groupby("team_id"):
        prob += pulp.lpSum(grp["var"]) <= 3

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    chosen = df[df["var"].apply(lambda v: v.value()) > 0.5]
    return chosen.index.tolist()


def find_best_formation(players_df: pd.DataFrame) -> Tuple[str, List[int], float]:
    """Finder den optimale formation for holdet."""
    best_formation = "343"
    best_xi = []
    best_ep = 0.0

    for formation in FORMATIONS.keys():
        try:
            xi = select_starting_xi(players_df, formation)
            ep = players_df.loc[xi, "ep_next_gw"].sum()
            if ep > best_ep:
                best_ep = ep
                best_xi = xi
                best_formation = formation
        except Exception:
            continue

    return best_formation, best_xi, best_ep


def _meets_squad_quota(base_counts: Dict[str, int], out_pos: str, in_pos: str) -> bool:
    """Tjekker om positionskvoter (2/5/5/3) overholdes."""
    counts = base_counts.copy()
    counts[out_pos] = counts.get(out_pos, 0) - 1
    counts[in_pos] = counts.get(in_pos, 0) + 1
    for pos, need in SQUAD_QUOTA.items():
        if counts.get(pos, 0) != need:
            return False
    return True


def _meets_squad_quota_multi(
        base_counts: Dict[str, int],
        outs: List[Tuple[str, int]],  # List of (pos, team_id)
        ins: List[Tuple[str, int]]  # List of (pos, team_id)
) -> bool:
    """Tjekker kvoter for multi-transfers."""
    counts = base_counts.copy()
    for pos, _ in outs:
        counts[pos] = counts.get(pos, 0) - 1
    for pos, _ in ins:
        counts[pos] = counts.get(pos, 0) + 1

    for pos, need in SQUAD_QUOTA.items():
        if counts.get(pos, 0) != need:
            return False
    return True


def _check_team_limit(
        base_team_counts: Dict[int, int],
        outs: List[int],  # team_ids to remove
        ins: List[int]  # team_ids to add
) -> bool:
    """Tjekker max 3 spillere per hold."""
    counts = base_team_counts.copy()
    for tid in outs:
        counts[tid] = counts.get(tid, 0) - 1
    for tid in ins:
        counts[tid] = counts.get(tid, 0) + 1
    return all(c <= 3 for c in counts.values())


def best_one_transfer_with_quotas(
        my15: pd.DataFrame,
        candidates: pd.DataFrame,
        bank_tenths: float,
        team_value_m: float,
        horizon: int = 5
) -> Optional[Dict[str, Any]]:
    """Finder det bedste enkelt-transfer."""
    results = best_n_transfers_with_quotas(my15, candidates, bank_tenths, team_value_m, horizon, top_n=1)
    return results[0] if results else None


def best_n_transfers_with_quotas(
        my15: pd.DataFrame,
        candidates: pd.DataFrame,
        bank_tenths: float,
        team_value_m: float,
        horizon: int = 5,
        top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Finder de top N bedste enkelt-transfers.

    Args:
        my15: Dit nuværende hold
        candidates: Mulige spillere at købe
        bank_tenths: Bank i tiendedele (£0.1m enheder)
        team_value_m: Samlet holdværdi i millioner
        horizon: Antal GW at optimere for
        top_n: Antal forslag at returnere

    Returns:
        Liste af transfer-forslag sorteret efter delta EP
    """
    ep_col = f"ep_next{horizon}"
    df_out = my15.copy()
    df_in = candidates.copy()

    # Sikr nødvendige kolonner
    for col in ["now_cost", "sell_price", ep_col]:
        if col not in df_out.columns:
            if col == "sell_price":
                df_out["sell_price"] = df_out.get("now_cost", 0.0)
            else:
                df_out[col] = 0.0
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce").fillna(0.0)

    for col in ["now_cost", ep_col]:
        if col not in df_in.columns:
            df_in[col] = 0.0
        df_in[col] = pd.to_numeric(df_in[col], errors="coerce").fillna(0.0)

    df_out["team_id"] = pd.to_numeric(df_out["team_id"], errors="coerce").fillna(-1).astype(int)
    df_in["team_id"] = pd.to_numeric(df_in["team_id"], errors="coerce").fillna(-1).astype(int)
    df_out["pos"] = df_out["pos"].astype(str)
    df_in["pos"] = df_in["pos"].astype(str)

    have_ids = set(df_out["id"].tolist())
    base_pos_counts = df_out["pos"].value_counts().to_dict()
    base_team_counts = df_out["team_id"].value_counts().to_dict()

    all_transfers = []

    for _, out_row in df_out.iterrows():
        price_out_tenths = float(out_row.get("sell_price", out_row.get("now_cost", 0.0)))
        out_pos = out_row["pos"]
        out_team = int(out_row["team_id"])

        for _, in_row in df_in.iterrows():
            if in_row["id"] in have_ids:
                continue

            in_pos = in_row["pos"]
            in_team = int(in_row["team_id"])

            # Check position quota
            if not _meets_squad_quota(base_pos_counts, out_pos, in_pos):
                continue

            # Check team limit
            if not _check_team_limit(base_team_counts, [out_team], [in_team]):
                continue

            # Check budget
            cost_diff = (float(in_row["now_cost"]) - price_out_tenths) / 10.0
            if cost_diff > float(bank_tenths) / 10.0 + 0.001:
                continue

            delta = float(in_row[ep_col]) - float(out_row[ep_col])

            all_transfers.append({
                "out": out_row.to_dict(),
                "in": in_row.to_dict(),
                "delta": delta,
                "cost_change": cost_diff,
                "new_bank": (float(bank_tenths) / 10.0) - cost_diff
            })

    # Sortér efter delta (faldende) og returner top N
    all_transfers.sort(key=lambda x: x["delta"], reverse=True)

    # Filtrér kun positive forbedringer
    positive = [t for t in all_transfers if t["delta"] > 0]

    return positive[:top_n]


def best_two_transfers(
        my15: pd.DataFrame,
        candidates: pd.DataFrame,
        bank_tenths: float,
        team_value_m: float,
        horizon: int = 5,
        top_n: int = 3,
        include_hit: bool = True
) -> List[Dict[str, Any]]:
    """
    Finder de bedste 2-transfer kombinationer.

    Args:
        my15: Dit nuværende hold
        candidates: Mulige spillere at købe
        bank_tenths: Bank i tiendedele
        team_value_m: Samlet holdværdi
        horizon: Antal GW at optimere for
        top_n: Antal forslag at returnere
        include_hit: Om -4 point hit skal fratrækkes delta

    Returns:
        Liste af 2-transfer forslag
    """
    ep_col = f"ep_next{horizon}"
    df_out = my15.copy()
    df_in = candidates.copy()

    # Sikr kolonner
    for col in ["now_cost", "sell_price", ep_col]:
        if col not in df_out.columns:
            if col == "sell_price":
                df_out["sell_price"] = df_out.get("now_cost", 0.0)
            else:
                df_out[col] = 0.0
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce").fillna(0.0)

    for col in ["now_cost", ep_col]:
        if col not in df_in.columns:
            df_in[col] = 0.0
        df_in[col] = pd.to_numeric(df_in[col], errors="coerce").fillna(0.0)

    df_out["team_id"] = pd.to_numeric(df_out["team_id"], errors="coerce").fillna(-1).astype(int)
    df_in["team_id"] = pd.to_numeric(df_in["team_id"], errors="coerce").fillna(-1).astype(int)
    df_out["pos"] = df_out["pos"].astype(str)
    df_in["pos"] = df_in["pos"].astype(str)

    have_ids = set(df_out["id"].tolist())
    base_pos_counts = df_out["pos"].value_counts().to_dict()
    base_team_counts = df_out["team_id"].value_counts().to_dict()

    all_combos = []
    out_list = list(df_out.iterrows())
    in_list = list(df_in.iterrows())

    # Begræns kandidater til top performers for at reducere beregninger
    df_in_top = df_in.nlargest(50, ep_col) if len(df_in) > 50 else df_in
    in_list_top = list(df_in_top.iterrows())

    for (idx1, out1), (idx2, out2) in combinations(out_list, 2):
        if idx1 == idx2:
            continue

        out1_price = float(out1.get("sell_price", out1.get("now_cost", 0.0)))
        out2_price = float(out2.get("sell_price", out2.get("now_cost", 0.0)))
        total_out_price = out1_price + out2_price
        total_out_ep = float(out1[ep_col]) + float(out2[ep_col])

        out_positions = [(out1["pos"], int(out1["team_id"])), (out2["pos"], int(out2["team_id"]))]
        out_teams = [int(out1["team_id"]), int(out2["team_id"])]

        for (_, in1), (_, in2) in combinations(in_list_top, 2):
            if in1["id"] in have_ids or in2["id"] in have_ids:
                continue
            if in1["id"] == in2["id"]:
                continue

            in1_price = float(in1["now_cost"])
            in2_price = float(in2["now_cost"])
            total_in_price = in1_price + in2_price

            # Budget check
            cost_diff = (total_in_price - total_out_price) / 10.0
            if cost_diff > float(bank_tenths) / 10.0 + 0.001:
                continue

            in_positions = [(in1["pos"], int(in1["team_id"])), (in2["pos"], int(in2["team_id"]))]
            in_teams = [int(in1["team_id"]), int(in2["team_id"])]

            # Position quota check
            if not _meets_squad_quota_multi(base_pos_counts, out_positions, in_positions):
                continue

            # Team limit check
            if not _check_team_limit(base_team_counts, out_teams, in_teams):
                continue

            # Beregn delta
            total_in_ep = float(in1[ep_col]) + float(in2[ep_col])
            delta = total_in_ep - total_out_ep

            if include_hit:
                delta -= TRANSFER_HIT

            all_combos.append({
                "out": [out1.to_dict(), out2.to_dict()],
                "in": [in1.to_dict(), in2.to_dict()],
                "delta": delta,
                "delta_before_hit": delta + (TRANSFER_HIT if include_hit else 0),
                "cost_change": cost_diff,
                "new_bank": (float(bank_tenths) / 10.0) - cost_diff,
                "hit": TRANSFER_HIT if include_hit else 0
            })

    # Sortér og returner
    all_combos.sort(key=lambda x: x["delta"], reverse=True)
    positive = [c for c in all_combos if c["delta"] > 0]

    return positive[:top_n]


def suggest_wildcard_team(
        all_players: pd.DataFrame,
        budget_tenths: float,
        horizon: int = 5,
        formation: str = "343"
) -> Optional[pd.DataFrame]:
    """
    Foreslår et optimalt wildcard-hold inden for budget.

    Args:
        all_players: Alle spillere
        budget_tenths: Samlet budget i tiendedele (typisk 1000 = £100m)
        horizon: Antal GW at optimere for
        formation: Ønsket formation for start XI

    Returns:
        DataFrame med 15 spillere eller None
    """
    ep_col = f"ep_next{horizon}"
    df = all_players.copy()

    # Sikr kolonner
    df["now_cost"] = pd.to_numeric(df.get("now_cost", 0), errors="coerce").fillna(0)
    df[ep_col] = pd.to_numeric(df.get(ep_col, 0), errors="coerce").fillna(0)
    df["pos"] = df.get("singular_name_short", df.get("pos", "MID")).astype(str)
    df["team_id"] = pd.to_numeric(df.get("team_id", df.get("team", 0)), errors="coerce").fillna(0).astype(int)

    # Filtrér utilgængelige spillere
    if "status" in df.columns:
        df = df[~df["status"].isin(["i", "s", "u"])]

    # Opret LP problem
    prob = pulp.LpProblem("wildcard", pulp.LpMaximize)

    # Binær variabel for hver spiller
    df["var"] = [pulp.LpVariable(f"p_{i}", cat="Binary") for i in df.index]

    # Maksimer EP
    prob += pulp.lpSum(df["var"] * df[ep_col])

    # Budget constraint
    prob += pulp.lpSum(df["var"] * df["now_cost"]) <= budget_tenths

    # Squad size = 15
    prob += pulp.lpSum(df["var"]) == 15

    # Position quotas
    for pos, need in SQUAD_QUOTA.items():
        prob += pulp.lpSum(df.loc[df["pos"] == pos, "var"]) == need

    # Max 3 per team
    for team_id in df["team_id"].unique():
        prob += pulp.lpSum(df.loc[df["team_id"] == team_id, "var"]) <= 3

    # Løs
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if prob.status != 1:
        return None

    # Hent valgte spillere
    chosen = df[df["var"].apply(lambda v: v.value()) > 0.5].copy()
    chosen = chosen.drop(columns=["var"], errors="ignore")

    return chosen


def evaluate_transfer_options(
        my15: pd.DataFrame,
        candidates: pd.DataFrame,
        bank_tenths: float,
        team_value_m: float,
        horizon: int = 5,
        free_transfers: int = 1
) -> Dict[str, Any]:
    """
    Evaluerer alle transfer-muligheder og returnerer struktureret overblik.

    Returns:
        Dict med:
        - best_single: Liste af top 5 enkelt-transfers
        - best_double: Liste af top 3 dobbelt-transfers (hvis FT >= 2 eller værd at tage hit)
        - recommendation: Anbefaling ("hold", "single", "double")
    """
    # Enkelt-transfers (altid gratis med mindst 1 FT)
    singles = best_n_transfers_with_quotas(my15, candidates, bank_tenths, team_value_m, horizon, top_n=5)

    # Dobbelt-transfers
    doubles_with_hit = best_two_transfers(my15, candidates, bank_tenths, team_value_m, horizon, top_n=3,
                                          include_hit=True)
    doubles_free = best_two_transfers(my15, candidates, bank_tenths, team_value_m, horizon, top_n=3,
                                      include_hit=False) if free_transfers >= 2 else []

    # Bestem anbefaling
    best_single_delta = singles[0]["delta"] if singles else 0
    best_double_delta = doubles_with_hit[0]["delta"] if doubles_with_hit else 0
    best_double_free_delta = doubles_free[0]["delta"] if doubles_free else 0

    if free_transfers >= 2 and best_double_free_delta > best_single_delta:
        recommendation = "double_free"
        reason = f"2 gratis transfers giver +{best_double_free_delta:.1f} EP vs +{best_single_delta:.1f} for enkelt"
    elif best_double_delta > best_single_delta and best_double_delta > 0:
        recommendation = "double_hit"
        reason = f"Dobbelt transfer med -4 hit giver stadig +{best_double_delta:.1f} EP netto"
    elif best_single_delta > 2.0:
        recommendation = "single"
        reason = f"Enkelt transfer giver solid +{best_single_delta:.1f} EP forbedring"
    elif best_single_delta > 0:
        recommendation = "single_marginal"
        reason = f"Lille forbedring på +{best_single_delta:.1f} EP - overvej at spare transfer"
    else:
        recommendation = "hold"
        reason = "Ingen positive forbedringer - spar din transfer"

    return {
        "best_single": singles,
        "best_double": doubles_free if free_transfers >= 2 else doubles_with_hit,
        "doubles_include_hit": free_transfers < 2,
        "recommendation": recommendation,
        "reason": reason,
        "free_transfers": free_transfers
    }
