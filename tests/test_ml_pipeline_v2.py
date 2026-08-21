from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from fpl_app.ml import predict
from fpl_app.ml.predict import FEATURE_CONTRACT, FeatureContractError, prepare_point_in_time_features
from fpl_app.ml.train_model import deadline_folds, engineer_features


STATS = {
    "minutes": 90, "goals_scored": 0, "assists": 0, "clean_sheets": 0,
    "bonus": 0, "bps": 10, "ict_index": 2, "expected_goals": 0.1,
    "expected_assists": 0.1,
}


def _raw() -> pd.DataFrame:
    rows = []
    for season, player, points in (("23", 7, 99), ("24", 7, 2), ("24", 8, 4)):
        for event in range(1, 8):
            rows.append({
                "season": season, "element": player, "GW": event,
                "kickoff_time": f"20{season}-08-{event + 1:02d}T12:00:00Z",
                "total_points": points if event == 1 else event,
                "position": "MID", "was_home": event % 2 == 0, "value": 75,
                "selected": 100, "transfers_in": 3, "transfers_out": 1,
                "opponent_team": 2, **STATS,
            })
    return pd.DataFrame(rows)


def test_training_pipeline_is_season_player_isolated_and_targets_current_decision():
    dataset = engineer_features(_raw())
    first = dataset.loc[(dataset["season"] == "24") & (dataset["player_id"] == 7) & (dataset["event"] == 1)].iloc[0]
    second = dataset.loc[(dataset["season"] == "24") & (dataset["player_id"] == 7) & (dataset["event"] == 2)].iloc[0]
    assert first["cum_total_points"] == 0
    assert first["target"] == 2
    assert second["cum_total_points"] == 2


def test_deadline_folds_are_global_and_unchanged_by_row_order():
    dataset = engineer_features(_raw())
    original = [(fold.test_deadline, int(train.sum()), int(test.sum())) for fold, train, test in deadline_folds(dataset)]
    shuffled = dataset.sample(frac=1, random_state=3).reset_index(drop=True)
    reordered = [(fold.test_deadline, int(train.sum()), int(test.sum())) for fold, train, test in deadline_folds(shuffled)]
    assert original == reordered
    assert all((dataset.loc[train, "deadline"] < fold.test_deadline).all() for fold, train, _ in deadline_folds(dataset))


def test_existing_or_candidate_artifact_is_not_available(monkeypatch, tmp_path):
    model = tmp_path / "model.joblib"
    features = tmp_path / "features.joblib"
    meta = tmp_path / "meta.json"
    model.touch()
    features.touch()
    monkeypatch.setattr(predict, "MODEL_PATH", model)
    monkeypatch.setattr(predict, "FEATURES_PATH", features)
    monkeypatch.setattr(predict, "META_PATH", meta)

    meta.write_text(json.dumps({"avg_mae": 1.3}), encoding="utf-8")
    assert not predict.model_available(FEATURE_CONTRACT)
    meta.write_text(json.dumps({
        "schema_version": 2, "feature_contract": FEATURE_CONTRACT,
        "validation_status": "candidate",
    }), encoding="utf-8")
    assert not predict.model_available(FEATURE_CONTRACT)
    meta.write_text(json.dumps({
        "schema_version": 2, "feature_contract": FEATURE_CONTRACT,
        "validation_status": "validated",
        "artifact_sha256": {
            "model": hashlib.sha256(b"").hexdigest(),
            "features": hashlib.sha256(b"").hexdigest(),
        },
    }), encoding="utf-8")
    assert predict.model_available(FEATURE_CONTRACT)
    assert not predict.model_available()

    model.write_bytes(b"changed")
    assert not predict.model_available(FEATURE_CONTRACT)


def test_serving_uses_point_in_time_builder_and_fails_on_missing_contract_feature():
    history = _raw().rename(columns={"element": "player_id", "GW": "event"})
    candidate = pd.DataFrame({"season": ["24"], "player_id": [7], "event": [2]})
    features = prepare_point_in_time_features(history, candidate, "2024-08-10T00:00:00Z", ["cum_total_points"])
    assert features.iloc[0, 0] == 2
    with pytest.raises(FeatureContractError, match="cannot reproduce"):
        prepare_point_in_time_features(history, candidate, "2024-08-10T00:00:00Z", ["not_a_feature"])
