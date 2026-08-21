from __future__ import annotations

import pandas as pd
import pytest

from fpl_app.evaluation.backtest import ExpandingDeadlineSplit, evaluate_predictions, paired_gw_bootstrap_ci, select_snapshot_as_of


def test_snapshot_selection_uses_latest_value_available_at_cutoff():
    rows = pd.DataFrame([
        {"season": "24", "player_id": 1, "event": 1, "prediction": 3.0, "effective_at": "2025-08-01T10:00:00Z"},
        # This is a backfill: it belongs to GW1, but was not knowable before deadline.
        {"season": "24", "player_id": 1, "event": 1, "prediction": 9.0, "effective_at": "2025-08-02T10:00:00Z"},
    ])
    chosen = select_snapshot_as_of(rows, "2025-08-01T12:00:00Z")
    assert chosen.iloc[0].prediction == 3.0


def test_expanding_deadlines_are_strictly_before_test_cutoff():
    decisions = pd.DataFrame({"season": ["24"] * 4, "event": [1, 2, 3, 4], "deadline": pd.date_range("2025-08-01", periods=4, tz="UTC")})
    folds = list(ExpandingDeadlineSplit(min_train_deadlines=2).split(decisions))
    assert [f.test_deadline for f in folds] == [pd.Timestamp("2025-08-03", tz="UTC"), pd.Timestamp("2025-08-04", tz="UTC")]
    train, test = ExpandingDeadlineSplit(min_train_deadlines=2).masks(decisions, folds[0])
    assert decisions.loc[train, "event"].tolist() == [1, 2]
    assert decisions.loc[test, "event"].tolist() == [3]


def test_expanding_deadline_test_size_keeps_every_deadline_in_block():
    decisions = pd.DataFrame({
        "season": ["24"] * 6,
        "event": [1, 2, 3, 4, 5, 6],
        "deadline": pd.date_range("2025-08-01", periods=6, tz="UTC"),
    })
    splitter = ExpandingDeadlineSplit(min_train_deadlines=2, test_size=2)
    fold = next(splitter.split(decisions))
    train, test = splitter.masks(decisions, fold)
    assert decisions.loc[train, "event"].tolist() == [1, 2]
    assert decisions.loc[test, "event"].tolist() == [3, 4]


def test_metrics_include_calibration_brier_and_decision_regret():
    df = pd.DataFrame([
        {"season": "24", "event": 1, "player_id": 1, "prediction": 8, "actual_points": 2, "minutes": 90, "p_start_60": .9},
        {"season": "24", "event": 1, "player_id": 2, "prediction": 7, "actual_points": 10, "minutes": 90, "p_start_60": .8},
        {"season": "24", "event": 1, "player_id": 3, "prediction": 1, "actual_points": 1, "minutes": 10, "p_start_60": .1},
    ])
    result = evaluate_predictions(df, calibration_bins=2, decision_top_n=1)
    assert result["mae"] == pytest.approx(3.0)
    assert result["start_60_brier"] == pytest.approx((.1 ** 2 + .2 ** 2 + .1 ** 2) / 3)
    assert result["captain_regret"] == 8.0
    assert result["decision_utility"] == 2.0
    assert result["decision_regret"] == 8.0
    assert sum(bin_["count"] for bin_ in result["calibration"]) == 3


def test_brier_ignores_missing_minutes_and_rejects_invalid_choice_cardinality():
    rows = pd.DataFrame([
        {"season": "24", "event": 1, "player_id": 1, "prediction": 2, "actual_points": 2, "minutes": 90, "p_start_60": .8, "is_captain": True},
        {"season": "24", "event": 1, "player_id": 2, "prediction": 1, "actual_points": 1, "minutes": None, "p_start_60": .9, "is_captain": False},
    ])
    result = evaluate_predictions(rows, decision_top_n=1)
    assert result["start_60_n"] == 1
    assert result["start_60_brier"] == pytest.approx(.04)

    rows["is_captain"] = True
    with pytest.raises(ValueError, match="exactly 1"):
        evaluate_predictions(rows, decision_top_n=1)


def test_paired_gw_bootstrap_is_deterministic_and_requires_same_outcomes():
    actual = [1, 5, 4, 0]
    base = pd.DataFrame({"season": ["24"] * 4, "event": [1, 1, 2, 2], "player_id": [1, 2, 1, 2], "prediction": [4, 4, 4, 4], "actual_points": actual})
    candidate = base.copy()
    candidate["prediction"] = [1, 5, 3, 1]
    one = paired_gw_bootstrap_ci(candidate, base, n_bootstrap=100, seed=9)
    two = paired_gw_bootstrap_ci(candidate, base, n_bootstrap=100, seed=9)
    assert one == two
    assert one["delta"] < 0
    bad = base.copy(); bad.loc[0, "actual_points"] = 999
    with pytest.raises(ValueError, match="outcomes differ"):
        paired_gw_bootstrap_ci(candidate, bad)
