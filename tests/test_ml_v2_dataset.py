from __future__ import annotations

import pandas as pd

from fpl_app.ml.v2_dataset import PointInTimeFeatureBuilder, build_point_in_time_dataset


def _history():
    return pd.DataFrame([
        {"season": "23", "player_id": 7, "event": 38, "total_points": 99, "minutes": 90, "goals_scored": 9, "assists": 1, "clean_sheets": 0, "bonus": 0, "bps": 0, "ict_index": 0, "expected_goals": 1, "expected_assists": 0, "effective_at": "2024-05-01T00:00:00Z"},
        {"season": "24", "player_id": 7, "event": 1, "total_points": 2, "minutes": 90, "goals_scored": 0, "assists": 0, "clean_sheets": 0, "bonus": 0, "bps": 1, "ict_index": 1, "expected_goals": .1, "expected_assists": .2, "effective_at": "2025-08-01T00:00:00Z"},
        # Revision after the prospective cutoff must not enter features.
        {"season": "24", "player_id": 7, "event": 1, "total_points": 20, "minutes": 90, "goals_scored": 4, "assists": 0, "clean_sheets": 0, "bonus": 0, "bps": 1, "ict_index": 1, "expected_goals": .1, "expected_assists": .2, "effective_at": "2025-08-04T00:00:00Z"},
    ])


def test_builder_is_season_keyed_and_excludes_same_event_and_later_revision():
    candidate = pd.DataFrame({"season": ["24"], "player_id": [7], "event": [2]})
    row = PointInTimeFeatureBuilder().build(_history(), candidate, "2025-08-03T00:00:00Z").iloc[0]
    assert row["cum_total_points"] == 2
    assert row["history_events"] == 1
    assert row["history_minutes"] == 90


def test_builder_retains_gw1_cold_start_when_only_same_event_is_known():
    """Same-event data is not valid history, but must not delete the player."""
    candidate = pd.DataFrame({"season": ["24"], "player_id": [7], "event": [1]})
    result = PointInTimeFeatureBuilder().build(_history(), candidate, "2025-08-03T00:00:00Z")
    assert len(result) == 1
    row = result.iloc[0]
    assert row["player_id"] == 7
    assert row["history_events"] == 0
    assert row["history_minutes"] == 0
    assert row["cum_total_points"] == 0
    assert row["sample_weight"] == 0


def test_train_dataset_and_serving_builder_are_the_same_point_in_time_path():
    decisions = pd.DataFrame({"season": ["24"], "player_id": [7], "event": [2], "deadline": ["2025-08-03T00:00:00Z"]})
    built = build_point_in_time_dataset(_history(), decisions)
    direct = PointInTimeFeatureBuilder().build(_history(), decisions, "2025-08-03T00:00:00Z")
    assert built["cum_total_points"].tolist() == direct["cum_total_points"].tolist()
