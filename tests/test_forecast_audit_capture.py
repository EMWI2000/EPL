from api.compute import _forecast_audit_capture
import pandas as pd


def test_capture_uses_confirmed_cohort_and_same_bootstrap_baselines():
    bootstrap = {"events": [
        {"id": 1, "finished": True, "data_checked": True},
        {"id": 2, "finished": True, "data_checked": False},
        {"id": 3, "deadline_time": "2026-09-12T12:30:00Z"},
    ], "elements": [{"id": 1, "minutes": 90, "ep_next": "4.1"}, {"id": 2, "ep_next": "99"}]}
    pool = pd.DataFrame([{"id": 1, "name": "Original", "ep_gw1": 3.2, "expected_minutes_gw1": 80},
                         {"id": 2, "name": "Transfer in", "ep_gw1": 6, "expected_minutes_gw1": 85}])
    planner = {"confirmed_state": {"squad": [{"id": 1}]}}
    value = _forecast_audit_capture(bootstrap, pool, planner, "2026-09-11T12:00:00Z", 3, "v2")
    assert value["model_version"] == "v2.1"
    assert value["players"] == [{"id": 1, "name": "Original", "model_points": 3.2,
                                 "model_minutes": 80, "baseline_points": 4.1, "baseline_minutes": None}]
    bootstrap["events"][1]["data_checked"] = True
    assert _forecast_audit_capture(bootstrap, pool, planner, "2026-09-11T12:00:00Z", 3, "v2")["players"][0]["baseline_minutes"] == 45
    bootstrap["elements"][0]["ep_next"] = "NaN"
    assert _forecast_audit_capture(bootstrap, pool, planner, "2026-09-11T12:00:00Z", 3, "v2")["players"][0]["baseline_points"] is None


def test_initial_optimiser_cannot_be_mistaken_for_personal_forecast_audit():
    assert _forecast_audit_capture({}, pd.DataFrame(), None, "", 4, "v2") is None
