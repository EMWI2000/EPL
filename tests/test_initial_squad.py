from __future__ import annotations

import unittest

import pandas as pd

from fpl_app.logic.initial_squad import (
    EXPERIMENTAL_NOTICE_DA,
    InitialSquadError,
    InitialSquadInfeasibleError,
    optimize_initial_squad,
)


def _candidate_pool() -> pd.DataFrame:
    rows = []
    specifications = (
        ("GKP", 4, 1, 45, 8.0),
        ("DEF", 8, 20, 50, 10.0),
        ("MID", 8, 40, 65, 13.0),
        ("FWD", 6, 60, 75, 12.0),
    )
    for position, count, first_id, cost, peak in specifications:
        for offset in range(count):
            rows.append(
                {
                    "id": first_id + offset,
                    "name": f"{position}-{offset}",
                    "team_id": (offset % 8) + 1,
                    "pos": position,
                    "now_cost": cost,
                    "ep_gw1": peak - offset * 0.35,
                    "ep_gw2": peak * 0.80 - offset * 0.20,
                    "ep_gw3": peak * 0.65 - offset * 0.10,
                }
            )
    return pd.DataFrame(rows)


class InitialSquadOptimizerTests(unittest.TestCase):
    def test_builds_complete_valid_and_deterministic_squad(self) -> None:
        players = _candidate_pool()
        result = optimize_initial_squad(players, horizon=3, gw_weights=(1.0, 0.8, 0.6))
        shuffled_result = optimize_initial_squad(
            players.sample(frac=1.0, random_state=42),
            horizon=3,
            gw_weights=(1.0, 0.8, 0.6),
        )

        self.assertEqual(result, shuffled_result)
        self.assertEqual(len(result.squad_ids), 15)
        self.assertEqual(len(set(result.squad_ids)), 15)
        self.assertEqual(len(result.starting_ids), 11)
        self.assertEqual(len(result.bench_ids), 4)
        self.assertEqual(set(result.starting_ids) | set(result.bench_ids), set(result.squad_ids))
        self.assertFalse(set(result.starting_ids) & set(result.bench_ids))
        self.assertLessEqual(result.total_cost_tenths, 1000)
        self.assertEqual(result.bank_tenths, 1000 - result.total_cost_tenths)
        self.assertIn(result.captain_id, result.starting_ids)
        self.assertIn(result.vice_captain_id, result.starting_ids)
        self.assertNotEqual(result.captain_id, result.vice_captain_id)
        self.assertEqual(result.experimental_notice, EXPERIMENTAL_NOTICE_DA)

        selected = players.set_index("id").loc[list(result.squad_ids)]
        self.assertEqual(selected["pos"].value_counts().to_dict(), {"MID": 5, "DEF": 5, "FWD": 3, "GKP": 2})
        self.assertLessEqual(int(selected["team_id"].value_counts().max()), 3)

        starters = players.set_index("id").loc[list(result.starting_ids)]
        counts = starters["pos"].value_counts()
        self.assertEqual(int(counts["GKP"]), 1)
        self.assertGreaterEqual(int(counts["DEF"]), 3)
        self.assertLessEqual(int(counts["DEF"]), 5)
        self.assertGreaterEqual(int(counts["MID"]), 2)
        self.assertLessEqual(int(counts["MID"]), 5)
        self.assertGreaterEqual(int(counts["FWD"]), 1)
        self.assertLessEqual(int(counts["FWD"]), 3)
        self.assertEqual(result.formation, f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}")

        # FPL's three substitution slots are outfield players; reserve GK is last.
        by_id = players.set_index("id")
        self.assertTrue(all(by_id.at[player_id, "pos"] != "GKP" for player_id in result.bench_ids[:3]))
        self.assertEqual(by_id.at[result.bench_ids[3], "pos"], "GKP")
        outfield_bench_scores = [
            by_id.at[player_id, "ep_gw1"]
            + 0.8 * by_id.at[player_id, "ep_gw2"]
            + 0.6 * by_id.at[player_id, "ep_gw3"]
            for player_id in result.bench_ids[:3]
        ]
        self.assertEqual(outfield_bench_scores, sorted(outfield_bench_scores, reverse=True))

        starter_scores = {
            player_id: (
                by_id.at[player_id, "ep_gw1"]
                + 0.8 * by_id.at[player_id, "ep_gw2"]
                + 0.6 * by_id.at[player_id, "ep_gw3"]
            )
            for player_id in result.starting_ids
        }
        self.assertEqual(result.captain_id, max(starter_scores, key=starter_scores.get))
        without_captain = {key: value for key, value in starter_scores.items() if key != result.captain_id}
        self.assertEqual(result.vice_captain_id, max(without_captain, key=without_captain.get))

    def test_budget_can_exclude_higher_projection(self) -> None:
        players = _candidate_pool()
        # Make the pool's fifth-best midfielder unaffordable, but leave a cheap sixth option.
        midfield_ids = players.loc[players["pos"] == "MID", "id"].tolist()
        expensive_id = midfield_ids[4]
        cheap_id = midfield_ids[5]
        players.loc[players["id"] == expensive_id, ["now_cost", "ep_gw1"]] = [300, 40.0]
        players.loc[players["id"] == cheap_id, "now_cost"] = 45

        result = optimize_initial_squad(players, horizon=1, budget_tenths=1000)

        self.assertLessEqual(result.total_cost_tenths, 1000)
        self.assertNotIn(expensive_id, result.squad_ids)
        self.assertIn(cheap_id, result.squad_ids)

    def test_never_selects_more_than_three_from_one_club(self) -> None:
        players = _candidate_pool()
        star_ids = (1, 20, 21, 40, 60)
        players.loc[players["id"].isin(star_ids), "team_id"] = 99
        players.loc[players["id"].isin(star_ids), "ep_gw1"] = 50.0

        result = optimize_initial_squad(players, horizon=1)
        selected = players.set_index("id").loc[list(result.squad_ids)]

        self.assertEqual(int((selected["team_id"] == 99).sum()), 3)

    def test_gameweek_weights_change_the_selected_player(self) -> None:
        players = _candidate_pool()
        midfield_ids = players.loc[players["pos"] == "MID", "id"].tolist()
        early_id, late_id = midfield_ids[4], midfield_ids[5]
        players.loc[players["id"] == early_id, ["ep_gw1", "ep_gw2"]] = [15.0, 0.0]
        players.loc[players["id"] == late_id, ["ep_gw1", "ep_gw2"]] = [0.0, 15.0]
        # The remaining optional midfielders should not compete with either test player.
        players.loc[players["id"].isin(midfield_ids[6:]), ["ep_gw1", "ep_gw2"]] = 0.0

        early = optimize_initial_squad(players, horizon=2, gw_weights=(1.0, 0.1))
        late = optimize_initial_squad(players, horizon=2, gw_weights=(0.1, 1.0))

        self.assertIn(early_id, early.squad_ids)
        self.assertNotIn(late_id, early.squad_ids)
        self.assertIn(late_id, late.squad_ids)
        self.assertNotIn(early_id, late.squad_ids)

    def test_derives_per_gameweek_values_from_cumulative_columns(self) -> None:
        players = _candidate_pool().drop(columns=["ep_gw1", "ep_gw2", "ep_gw3"])
        players["ep_next_gw"] = 2.0
        players["ep_next2"] = 5.0

        result = optimize_initial_squad(players, horizon=2, gw_weights=(1.0, 0.5))

        self.assertEqual(result.forecast_columns, ("ep_next_gw", "ep_next2"))
        # Each player's weighted value is 2 + 0.5 * (5 - 2) = 3.5.
        self.assertGreater(result.projected_xi_points, 0)

    def test_rejects_invalid_input_and_reports_infeasibility(self) -> None:
        players = _candidate_pool()
        duplicated = pd.concat([players, players.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(InitialSquadError, "unique"):
            optimize_initial_squad(duplicated, horizon=1)

        with self.assertRaisesRegex(InitialSquadError, "per-gameweek columns"):
            optimize_initial_squad(players.drop(columns=["ep_gw1"]), horizon=1)

        too_few_forwards = players[players["pos"] != "FWD"].copy()
        with self.assertRaises(InitialSquadInfeasibleError):
            optimize_initial_squad(too_few_forwards, horizon=1)

        with self.assertRaises(InitialSquadInfeasibleError):
            optimize_initial_squad(players, horizon=1, budget_tenths=100)


if __name__ == "__main__":
    unittest.main()
