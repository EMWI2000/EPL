from __future__ import annotations

from datetime import datetime, timezone
import json

import pandas as pd
import pytest

from api import compute, health


def test_health_payload_is_json_serializable():
    payload = health.health_payload()

    assert payload["status"] == "ok"
    assert payload["service"] == "epl-fpl-api"
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"horizon": 0}, "horizon"),
        ({"horizon": True}, "horizon"),
        ({"horizon": 2.5}, "horizon"),
        ({"include_doubtful": 1}, "include_doubtful"),
        ({"use_solio": "yes"}, "use_solio"),
        ({"unexpected": True}, "unsupported"),
        ([], "JSON object"),
    ],
)
def test_request_validation_rejects_invalid_options(payload, message):
    with pytest.raises(compute.RequestValidationError, match=message):
        compute.validate_request_payload(payload)


def test_request_validation_applies_defaults_without_mutating_input():
    supplied = {"horizon": 2}

    result = compute.validate_request_payload(supplied)

    assert result == {"horizon": 2, "include_doubtful": True, "use_solio": True}
    assert supplied == {"horizon": 2}


def test_next_open_gameweek_skips_locked_but_unfinished_event():
    bootstrap = {
        "events": [
            {
                "id": 7,
                "is_current": True,
                "is_next": False,
                "finished": False,
                "deadline_time": "2026-09-01T17:30:00Z",
            },
            {
                "id": 8,
                "is_current": False,
                "is_next": True,
                "finished": False,
                "deadline_time": "2026-09-08T17:30:00Z",
            },
        ]
    }

    result = compute.next_open_gameweek(
        bootstrap,
        now=datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc),
    )

    assert result == 8


def test_next_open_gameweek_fails_closed_without_future_deadline():
    result = compute.next_open_gameweek(
        {
            "events": [
                {
                    "id": 38,
                    "is_next": True,
                    "deadline_time": "2026-05-24T13:30:00Z",
                }
            ]
        },
        now=datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc),
    )

    assert result is None


@pytest.mark.parametrize(
    ("provided", "configured", "expected"),
    [
        ("private-token", "private-token", True),
        ("wrong-token", "private-token", False),
        (None, "private-token", False),
        ("", "private-token", False),
        ("private-token", "", False),
    ],
)
def test_internal_authentication_fails_closed(provided, configured, expected):
    assert (
        compute.is_internal_request_authorized(
            provided,
            configured_token=configured,
        )
        is expected
    )


def test_internal_authentication_reads_environment(monkeypatch):
    monkeypatch.delenv(compute.INTERNAL_TOKEN_ENV, raising=False)
    assert compute.is_internal_request_authorized("anything") is False

    monkeypatch.setenv(compute.INTERNAL_TOKEN_ENV, "configured-secret")
    assert compute.is_internal_request_authorized("configured-secret") is True
    assert compute.is_internal_request_authorized("different-secret") is False


def test_internal_authentication_uses_constant_time_comparison(monkeypatch):
    calls = []

    def capture_compare(left, right):
        calls.append((left, right))
        return True

    monkeypatch.setattr(compute.hmac, "compare_digest", capture_compare)

    assert compute.is_internal_request_authorized(
        "provided",
        configured_token="configured",
    )
    assert calls == [(b"provided", b"configured")]


def test_solio_overlay_uses_bounded_serverless_timeout(monkeypatch):
    captured_timeouts = []

    class UnavailableSolioClient:
        def __init__(self, *, timeout):
            captured_timeouts.append(timeout)

        def fetch_latest(self):
            raise compute.SolioError("offline")

    monkeypatch.setattr(compute, "SolioClient", UnavailableSolioClient)
    pool = pd.DataFrame({"id": [1], "ep_gw1": [1.0]})

    unchanged, metadata = compute._apply_solio_overlay(pool, pool, first_gameweek=1)

    assert captured_timeouts == [compute.SOLIO_TIMEOUT_SECONDS]
    assert unchanged.equals(pool)
    assert metadata["applied"] is False
    assert "offline" in metadata["warning"]


def _bare_handler(monkeypatch, *, provided_token=None, configured_token=None):
    instance = object.__new__(compute.handler)
    instance.headers = {}
    if provided_token is not None:
        instance.headers[compute.INTERNAL_TOKEN_HEADER] = provided_token
    if configured_token is None:
        monkeypatch.delenv(compute.INTERNAL_TOKEN_ENV, raising=False)
    else:
        monkeypatch.setenv(compute.INTERNAL_TOKEN_ENV, configured_token)
    responses = []
    instance._send_json = lambda status, payload: responses.append((status, payload))
    return instance, responses


@pytest.mark.parametrize(
    ("provided_token", "configured_token"),
    [
        (None, "configured-secret"),
        ("wrong-secret", "configured-secret"),
        ("configured-secret", None),
    ],
)
def test_compute_handler_rejects_before_reading_body(
    monkeypatch,
    provided_token,
    configured_token,
):
    instance, responses = _bare_handler(
        monkeypatch,
        provided_token=provided_token,
        configured_token=configured_token,
    )
    instance._read_payload = lambda: pytest.fail("unauthorized request body was read")
    monkeypatch.setattr(
        compute,
        "generate_recommendation",
        lambda payload: pytest.fail("unauthorized recommendation was generated"),
    )

    instance.do_POST()

    assert responses[0][0] == 401
    assert responses[0][1]["error"]["code"] == "unauthorized"


def test_compute_handler_accepts_valid_internal_token(monkeypatch):
    instance, responses = _bare_handler(
        monkeypatch,
        provided_token="configured-secret",
        configured_token="configured-secret",
    )
    instance._read_payload = lambda: {"horizon": 1}
    monkeypatch.setattr(
        compute,
        "generate_recommendation",
        lambda payload: {"accepted": payload},
    )

    instance.do_POST()

    assert responses == [(200, {"accepted": {"horizon": 1}})]


def _synthetic_pool(horizon: int = 2) -> pd.DataFrame:
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    rows = []
    for player_id, position in enumerate(positions, start=1):
        row = {
            "id": player_id,
            "name": f"Player {player_id}",
            "team_id": ((player_id - 1) % 5) + 1,
            "team": f"T{((player_id - 1) % 5) + 1}",
            "pos": position,
            "now_cost": 50,
            "status": "a",
        }
        for offset in range(1, horizon + 1):
            row[f"ep_gw{offset}"] = float(20 - player_id + offset)
            row[f"source_gw{offset}"] = "internal_heuristic"
        rows.append(row)
    return pd.DataFrame(rows)


def test_generate_recommendation_returns_frontend_contract(monkeypatch):
    pool = _synthetic_pool(horizon=2)
    official = pd.DataFrame({"id": pool["id"]})
    fixtures = pd.DataFrame(
        {
            "event": [7, 8],
            "home_team": [1, 2],
            "away_team": [2, 3],
            "home_fdr": [3, 3],
            "away_fdr": [3, 3],
        }
    )
    teams = pd.DataFrame(
        {
            "team_id": range(1, 6),
            "name": [f"Team {index}" for index in range(1, 6)],
            "short_name": [f"T{index}" for index in range(1, 6)],
        }
    )

    bootstrap = {
        "events": [
            {
                "id": 7,
                "is_next": True,
                "deadline_time": "2099-08-01T17:30:00Z",
            }
        ]
    }
    monkeypatch.setattr(
        compute,
        "_load_official_data",
        lambda: (bootstrap, official, fixtures, teams),
    )
    monkeypatch.setattr(compute, "_build_forecast_pool", lambda *args: pool.copy())

    response = compute.generate_recommendation(
        {"horizon": 2, "include_doubtful": False, "use_solio": False}
    )

    assert response["meta"]["gameweek_window"] == [7, 8]
    assert response["meta"]["data_sources"]["solio"]["requested"] is False
    assert response["summary"]["total_cost_tenths"] == 750
    assert len(response["team"]["squad"]) == 15
    assert len(response["team"]["starters"]) == 11
    assert len(response["team"]["bench"]) == 4
    assert sum(player["is_captain"] for player in response["team"]["squad"]) == 1
    assert sum(player["is_vice_captain"] for player in response["team"]["squad"]) == 1
    assert all(len(player["projections"]) == 2 for player in response["team"]["squad"])
    json.dumps(response, allow_nan=False)
