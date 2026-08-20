from __future__ import annotations

from copy import deepcopy
import unittest

import pandas as pd

from fpl_app.logic.projections import (
    extract_solio_player_projections,
    normalize_player_name,
    overlay_solio_projections,
)


def official_players() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "first_name": "Bruno Miguel",
                "second_name": "Borges Fernandes",
                "web_name": "Fernandes",
                "short_name": "MUN",
                "singular_name_short": "MID",
                "now_cost": 120,
            },
            {
                "id": 2,
                "first_name": "Martin",
                "second_name": "Ødegaard",
                "web_name": "Ødegaard",
                "short_name": "ARS",
                "singular_name_short": "MID",
                "now_cost": 65,
            },
            {
                "id": 3,
                "first_name": "Enzo",
                "second_name": "Le Fée",
                "web_name": "E.Le Fée",
                "short_name": "SUN",
                "singular_name_short": "MID",
                "now_cost": 60,
            },
        ]
    )


def payload(*rows: dict) -> dict:
    return {
        "gameweek": 1,
        "topProjected": list(rows),
        "topCaptains": [],
        "bestCleanSheets": [{"team": "ARS", "csProb": 0.55}],
    }


class ProjectionExtractionTests(unittest.TestCase):
    def test_name_normalization_handles_initials_accents_and_special_letters(self) -> None:
        self.assertEqual(normalize_player_name("E. Le-Fée"), "e le fee")
        self.assertEqual(normalize_player_name("M. Ødegaard"), "m odegaard")

    def test_duplicate_player_rows_are_consolidated_across_all_sections(self) -> None:
        row = {
            "name": "Ødegaard",
            "team": "ARS",
            "position": "MID",
            "price": 65,
            "prPoints": 4.69,
        }
        data = payload(row)
        data["topGoals"] = [deepcopy(row)]
        data["newAdditiveLeaderboard"] = [deepcopy(row)]

        result = extract_solio_player_projections(data)

        self.assertEqual(result.source_rows_seen, 3)
        self.assertEqual(len(result.projections), 1)
        projection = result.projections.iloc[0]
        self.assertEqual(projection["occurrences"], 3)
        self.assertEqual(
            projection["sections"],
            ("newAdditiveLeaderboard", "topGoals", "topProjected"),
        )
        self.assertFalse(projection["is_conflicting"])

    def test_conflicting_points_are_visible_but_marked_unusable(self) -> None:
        row = {
            "name": "Ødegaard",
            "team": "ARS",
            "position": "MID",
            "price": 65,
            "prPoints": 4.69,
        }
        data = payload(row)
        changed = deepcopy(row)
        changed["prPoints"] = 5.1
        data["topCaptains"] = [changed]

        result = extract_solio_player_projections(data)

        self.assertEqual(len(result.projections), 1)
        self.assertTrue(result.projections.iloc[0]["is_conflicting"])
        self.assertTrue(pd.isna(result.projections.iloc[0]["prPoints"]))
        self.assertEqual(len(result.conflicts), 1)


class ProjectionOverlayTests(unittest.TestCase):
    def test_exact_alias_overlay_matches_initials_accents_and_punctuation(self) -> None:
        data = payload(
            {
                "name": "B.Fernandes",
                "team": "Manchester United",
                "position": "MID",
                "price": 120,
                "prPoints": 6.92,
            },
            {
                "name": "M.Odegaard",
                "team": "ARS",
                "position": "Midfielder",
                "price": 65,
                "prPoints": 4.69,
            },
            {
                "name": "E. Le-Fee",
                "team": "SUN",
                "position": "MID",
                "price": 60,
                "prPoints": 4.44,
            },
        )
        original = official_players()

        result = overlay_solio_projections(original, data)

        self.assertEqual(result.output_column, "solio_ep_gw1")
        self.assertEqual(result.diagnostics.matched_projection_count, 3)
        self.assertEqual(result.diagnostics.projection_coverage, 1.0)
        self.assertEqual(result.diagnostics.match_methods, (("exact_alias", 3),))
        self.assertEqual(
            result.players.set_index("id")["solio_ep_gw1"].to_dict(),
            {1: 6.92, 2: 4.69, 3: 4.44},
        )
        self.assertNotIn("solio_ep_gw1", original.columns)

    def test_team_position_and_price_are_hard_match_constraints(self) -> None:
        data = payload(
            {
                "name": "B.Fernandes",
                "team": "MUN",
                "position": "MID",
                "price": 121,
                "prPoints": 6.92,
            }
        )
        result = overlay_solio_projections(official_players(), data)
        self.assertEqual(result.diagnostics.matched_projection_count, 0)
        self.assertEqual(len(result.diagnostics.unmatched), 1)
        self.assertTrue(result.players["solio_ep_gw1"].isna().all())

    def test_a_minor_typo_can_match_one_structurally_unique_candidate(self) -> None:
        data = payload(
            {
                "name": "Odegard",
                "team": "ARS",
                "position": "MID",
                "price": 65,
                "prPoints": 4.69,
            }
        )
        result = overlay_solio_projections(official_players(), data)
        self.assertEqual(result.diagnostics.match_methods, (("fuzzy_name", 1),))
        self.assertEqual(result.matches.iloc[0]["fpl_player_id"], 2)

    def test_multiple_exact_candidates_are_ambiguous_and_not_overlaid(self) -> None:
        players = pd.DataFrame(
            [
                {
                    "id": 10,
                    "first_name": "Alex",
                    "second_name": "Smith",
                    "web_name": "Smith",
                    "short_name": "ARS",
                    "singular_name_short": "DEF",
                    "now_cost": 50,
                },
                {
                    "id": 11,
                    "first_name": "Adam",
                    "second_name": "Smith",
                    "web_name": "Smith",
                    "short_name": "ARS",
                    "singular_name_short": "DEF",
                    "now_cost": 50,
                },
            ]
        )
        data = payload(
            {
                "name": "Smith",
                "team": "ARS",
                "position": "DEF",
                "price": 50,
                "prPoints": 4.0,
            }
        )
        result = overlay_solio_projections(players, data)
        self.assertEqual(result.diagnostics.matched_projection_count, 0)
        self.assertEqual(len(result.diagnostics.ambiguous), 1)
        self.assertEqual(result.diagnostics.ambiguous[0].candidate_ids, (10, 11))

    def test_conflicts_and_invalid_rows_are_reported_not_silently_overlaid(self) -> None:
        valid = {
            "name": "Ødegaard",
            "team": "ARS",
            "position": "MID",
            "price": 65,
            "prPoints": 4.69,
        }
        data = payload(valid, {"name": "Broken", "prPoints": "unknown"})
        conflicting = deepcopy(valid)
        conflicting["prPoints"] = 5.0
        data["topGoals"] = [conflicting]

        result = overlay_solio_projections(official_players(), data)

        self.assertEqual(result.diagnostics.matched_projection_count, 0)
        self.assertEqual(result.diagnostics.usable_projection_count, 0)
        self.assertEqual(len(result.diagnostics.conflicts), 1)
        self.assertEqual(len(result.diagnostics.invalid_rows), 1)

    def test_distinct_source_identities_claiming_one_player_are_rejected(self) -> None:
        players = pd.DataFrame(
            [
                {
                    "id": 20,
                    "first_name": "João",
                    "second_name": "Pedro",
                    "web_name": "João Pedro",
                    "short_name": "CHE",
                    "singular_name_short": "FWD",
                    "now_cost": 75,
                }
            ]
        )
        data = payload(
            {
                "name": "Joao Pedro",
                "team": "CHE",
                "position": "FWD",
                "price": 75,
                "prPoints": 4.9,
            },
            {
                "name": "J.Pedro",
                "team": "CHE",
                "position": "FWD",
                "price": 75,
                "prPoints": 4.9,
            },
        )
        result = overlay_solio_projections(players, data)
        self.assertEqual(result.diagnostics.matched_projection_count, 0)
        self.assertEqual(len(result.diagnostics.ambiguous), 2)
        self.assertTrue(result.players["solio_ep_gw1"].isna().all())


if __name__ == "__main__":
    unittest.main()
