from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from io import BytesIO
import json

import pytest

from api import manager_state
from fpl_app.services.personal_fpl_state import (
    PublicLastDeadlineState,
    PublicPick,
    PublicTransfer,
)


VALID_INTERNAL_TOKEN = "manager-sync-token-with-at-least-32-bytes"


def _element_type(element_id: int) -> int:
    if element_id <= 2:
        return 1
    if element_id <= 7:
        return 2
    if element_id <= 12:
        return 3
    return 4


def _public_state(*, active_chip: str | None = None) -> PublicLastDeadlineState:
    picks = tuple(
        PublicPick(
            element_id=element_id,
            position=element_id,
            multiplier=2 if element_id == 1 else (0 if element_id > 11 else 1),
            is_captain=element_id == 1,
            is_vice_captain=element_id == 2,
            element_type=_element_type(element_id),
        )
        for element_id in range(1, 16)
    )
    return PublicLastDeadlineState(
        entry_id=123,
        event=6,
        picks=picks,
        bank_tenths=8,
        squad_value_tenths=1012,
        total_transfers_at_deadline=4,
        event_transfers=1,
        event_transfer_cost=0,
        active_chip=active_chip,
        public_transfers=(
            PublicTransfer(
                event=2,
                element_in=1,
                element_out=99,
                element_in_cost_tenths=50,
                element_out_cost_tenths=48,
                confirmed_at="2026-08-10T12:00:00Z",
            ),
            PublicTransfer(
                event=4,
                element_in=98,
                element_out=1,
                element_in_cost_tenths=51,
                element_out_cost_tenths=51,
                confirmed_at="2026-08-16T12:00:00Z",
            ),
            PublicTransfer(
                event=6,
                element_in=1,
                element_out=98,
                element_in_cost_tenths=52,
                element_out_cost_tenths=51,
                confirmed_at="2026-08-22T12:00:00Z",
            ),
        ),
        source_urls=("https://internal-source.example/entry/123/",),
    )


def _bootstrap() -> dict:
    elements = []
    for element_id in range(1, 16):
        now_cost = 56 if element_id == 1 else (55 if element_id == 2 else 50)
        change_start = 2 if element_id == 1 else (1 if element_id == 2 else 0)
        elements.append(
            {
                "id": element_id,
                "web_name": f"Player {element_id}",
                "team": ((element_id - 1) % 5) + 1,
                "element_type": _element_type(element_id),
                "now_cost": now_cost,
                "cost_change_start": change_start,
                "cost_change_event": 0,
                "selected_by_percent": "10.0",
                "transfers_in_event": 100 + element_id,
                "transfers_out_event": 50 + element_id,
                "price_change_percent": "95.5",
                "price_change_hourly_rate": 4,
                "price_change_projections": [
                    {
                        "offset": 0,
                        "projected_percent": "96.0",
                        "likelihood": 2,
                    }
                ],
                "price_change_locked_until": None,
                "price_change_calibrating": False,
            }
        )
    return {
        "events": [
            {
                "id": 6,
                "name": "Gameweek 6",
                "deadline_time": "2026-08-22T17:30:00Z",
            },
            {
                "id": 7,
                "name": "Gameweek 7",
                "deadline_time": "2026-08-30T17:30:00Z",
            },
        ],
        "teams": [
            {"id": team_id, "name": f"Club {team_id}", "short_name": f"C{team_id}"}
            for team_id in range(1, 6)
        ],
        "element_types": [
            {"id": 1, "singular_name_short": "GKP"},
            {"id": 2, "singular_name_short": "DEF"},
            {"id": 3, "singular_name_short": "MID"},
            {"id": 4, "singular_name_short": "FWD"},
        ],
        "game_config": {
            "settings": {
                "price_change_deadlines": [
                    "2026-08-23T23:00:00Z",
                    "2026-08-24T23:00:00Z",
                ]
            }
        },
        "elements": elements,
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "required"),
        ({"manager_id": True}, "positive integer"),
        ({"manager_id": 0}, "positive integer"),
        ({"manager_id": "123"}, "positive integer"),
        ({"manager_id": 123, "extra": 1}, "unsupported"),
        ([], "JSON object"),
        (None, "JSON object"),
    ],
)
def test_request_contract_accepts_only_manager_id(payload, message: str) -> None:
    with pytest.raises(manager_state.RequestValidationError, match=message):
        manager_state.validate_request_payload(payload)

    assert manager_state.validate_request_payload({"manager_id": 123}) == 123


def test_manager_sync_enriches_prices_and_returns_confirmation_draft(
    monkeypatch,
) -> None:
    public_state = _public_state(active_chip="freehit")
    calls: list[tuple[str, int | None]] = []

    class FakeClient:
        def fetch_last_deadline_state(self, manager_id: int, *, validated_summary):
            calls.append(("public_state", manager_id))
            assert validated_summary.entry_id == manager_id
            assert validated_summary.team_name == "Test XI"
            return public_state

    monkeypatch.setattr(manager_state, "PublicManagerStateClient", FakeClient)
    monkeypatch.setattr(
        manager_state,
        "manager_summary",
        lambda manager_id: calls.append(("summary", manager_id))
        or {
            "id": manager_id,
            "name": "Test XI",
            "started_event": 1,
            "current_event": 6,
            "last_deadline_total_transfers": 4,
            "summary_overall_points": 321,
            "summary_overall_rank": 4567,
        },
    )
    monkeypatch.setattr(
        manager_state,
        "bootstrap_static",
        lambda: calls.append(("bootstrap", None)) or _bootstrap(),
    )

    response = manager_state.generate_manager_state(
        {"manager_id": 123},
        now=datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc),
    )

    assert calls[0] == ("summary", 123)
    assert sorted(calls[1:]) == [("bootstrap", None), ("public_state", 123)]
    assert response["manager"] == {
        "id": 123,
        "team_name": "Test XI",
        "overall_points": 321,
        "overall_rank": 4567,
    }
    assert response["target"] == {
        "event": 7,
        "name": "Gameweek 7",
        "deadline_time": "2026-08-30T17:30:00Z",
    }

    picks = response["last_deadline_state"]["picks"]
    assert len(picks) == 15
    assert picks[0]["name"] == "Player 1"
    assert picks[0]["club"] == "C1"
    assert picks[0]["position"] == "GKP"
    assert picks[0]["current_price_tenths"] == 56
    assert picks[0]["estimated_purchase_price_tenths"] == 52
    assert picks[0]["estimated_selling_price_tenths"] == 54
    assert picks[0]["purchase_price_basis"] == {
        "kind": "latest_public_transfer_in",
        "event": 6,
        "confirmed_at": "2026-08-22T12:00:00Z",
    }
    assert picks[0]["official_price_signal"]["price_change_percent"] == 95.5
    assert response["price_signals"]["available"] is True
    assert response["price_signals"]["warning"] is None

    # With no public acquisition, purchase price falls back conservatively to
    # current price minus the official season change.  A one-tenth rise yields
    # no realised profit because FPL rounds the manager's half down.
    assert picks[1]["estimated_purchase_price_tenths"] == 54
    assert picks[1]["estimated_selling_price_tenths"] == 54
    assert picks[1]["purchase_price_basis"] == {"kind": "season_start_price"}

    template = response["manual_state_template"]
    assert template["state"]["free_transfers"] == 1
    assert template["state"]["no_active_chip_confirmed"] is False
    assert template["state"]["effective_event"] == 7
    assert template["confirmation_required"] is True
    assert "free_transfers" in template["fields_requiring_confirmation"]
    assert "no_active_chip_confirmed" in template["fields_requiring_confirmation"]
    assert len(template["state"]["player_prices"]) == 15
    assert response["snapshot"]["persisted"] is False
    assert len(response["snapshot"]["checksum_sha256"]) == 64
    assert {warning["code"] for warning in response["warnings"]} >= {
        "last_deadline_state_requires_confirmation",
        "free_hit_squad_is_temporary",
    }

    encoded = json.dumps(response, allow_nan=False)
    assert "source_urls" not in encoded
    assert "internal-source.example" not in encoded
    assert "X-Internal-Token" not in encoded


def test_acquisition_replay_fails_closed_when_owned_players_latest_move_is_out() -> None:
    state = _public_state()
    inconsistent = replace(
        state,
        public_transfers=(
            PublicTransfer(
                event=2,
                element_in=1,
                element_out=99,
                element_in_cost_tenths=50,
                element_out_cost_tenths=48,
                confirmed_at="2026-08-10T12:00:00Z",
            ),
            PublicTransfer(
                event=6,
                element_in=98,
                element_out=1,
                element_in_cost_tenths=52,
                element_out_cost_tenths=51,
                confirmed_at="2026-08-22T12:00:00Z",
            ),
        ),
    )
    bootstrap = _bootstrap()
    players, _, _ = manager_state._catalogue_indexes(bootstrap)
    core_prices = manager_state._core_player_prices(inconsistent, players)

    with pytest.raises(manager_state.UpstreamPayloadError, match="latest.*is out"):
        manager_state._estimated_player_prices(inconsistent, core_prices)


def test_undocumented_price_signals_degrade_without_losing_core_prices(
    monkeypatch,
) -> None:
    public_state = _public_state()
    degraded_bootstrap = _bootstrap()
    degraded_bootstrap["elements"][0]["price_change_calibrating"] = 0

    class FakeClient:
        def fetch_last_deadline_state(self, manager_id: int, *, validated_summary):
            assert manager_id == validated_summary.entry_id == 123
            return public_state

    monkeypatch.setattr(manager_state, "PublicManagerStateClient", FakeClient)
    monkeypatch.setattr(
        manager_state,
        "manager_summary",
        lambda manager_id: {
            "id": manager_id,
            "name": "Test XI",
            "started_event": 1,
            "current_event": 6,
            "last_deadline_total_transfers": 4,
            "summary_overall_points": 321,
            "summary_overall_rank": 4567,
        },
    )
    monkeypatch.setattr(
        manager_state,
        "bootstrap_static",
        lambda: degraded_bootstrap,
    )

    response = manager_state.generate_manager_state(
        {"manager_id": 123},
        now=datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc),
    )

    assert response["price_signals"]["available"] is False
    assert response["price_signals"]["warning"]
    assert response["price_signals"]["price_change_deadlines"] == []
    first_pick = response["last_deadline_state"]["picks"][0]
    assert first_pick["official_price_signal"] is None
    assert first_pick["current_price_tenths"] == 56
    assert first_pick["estimated_purchase_price_tenths"] == 52
    assert first_pick["estimated_selling_price_tenths"] == 54
    json.dumps(response, allow_nan=False)


@pytest.mark.parametrize(
    ("provided", "configured", "expected"),
    [
        (VALID_INTERNAL_TOKEN, VALID_INTERNAL_TOKEN, True),
        ("wrong", VALID_INTERNAL_TOKEN, False),
        (None, VALID_INTERNAL_TOKEN, False),
        ("", VALID_INTERNAL_TOKEN, False),
        (VALID_INTERNAL_TOKEN, "short", False),
        (VALID_INTERNAL_TOKEN, "", False),
    ],
)
def test_internal_authentication_fails_closed(provided, configured, expected) -> None:
    assert (
        manager_state.is_internal_request_authorized(
            provided,
            configured_token=configured,
        )
        is expected
    )


def test_internal_authentication_enforces_utf8_byte_length_from_environment(
    monkeypatch,
) -> None:
    monkeypatch.delenv(manager_state.INTERNAL_TOKEN_ENV, raising=False)
    assert manager_state.is_internal_request_authorized(VALID_INTERNAL_TOKEN) is False

    monkeypatch.setenv(manager_state.INTERNAL_TOKEN_ENV, "too-short")
    assert manager_state.is_internal_request_authorized("too-short") is False

    multibyte_token = "ø" * 16
    assert len(multibyte_token) < manager_state.MIN_INTERNAL_TOKEN_BYTES
    assert len(multibyte_token.encode("utf-8")) == manager_state.MIN_INTERNAL_TOKEN_BYTES
    monkeypatch.setenv(manager_state.INTERNAL_TOKEN_ENV, multibyte_token)
    assert manager_state.is_internal_request_authorized(multibyte_token) is True


def _bare_handler(
    monkeypatch,
    *,
    provided_token=None,
    configured_token=VALID_INTERNAL_TOKEN,
):
    instance = object.__new__(manager_state.handler)
    instance.headers = {}
    if provided_token is not None:
        instance.headers[manager_state.INTERNAL_TOKEN_HEADER] = provided_token
    if configured_token is None:
        monkeypatch.delenv(manager_state.INTERNAL_TOKEN_ENV, raising=False)
    else:
        monkeypatch.setenv(manager_state.INTERNAL_TOKEN_ENV, configured_token)
    responses = []
    instance._send_json = lambda status, payload: responses.append((status, payload))
    return instance, responses


def test_handler_authenticates_before_reading_request(monkeypatch) -> None:
    instance, responses = _bare_handler(
        monkeypatch,
        provided_token="wrong",
        configured_token=VALID_INTERNAL_TOKEN,
    )
    instance._read_payload = lambda: pytest.fail("unauthorized body was read")
    monkeypatch.setattr(
        manager_state,
        "generate_manager_state",
        lambda payload: pytest.fail("unauthorized upstream call was made"),
    )

    instance.do_POST()

    assert responses[0][0] == 401
    assert responses[0][1]["error"]["code"] == "unauthorized"


@pytest.mark.parametrize(
    ("exception", "status", "code"),
    [
        (
            manager_state.RequestValidationError("bad manager_id"),
            422,
            "invalid_request",
        ),
        (manager_state.ManagerNotFoundError("missing"), 404, "manager_not_found"),
        (
            manager_state.ManagerStateUnavailableError("season over"),
            503,
            "manager_state_unavailable",
        ),
        (manager_state.UpstreamPayloadError("bad upstream"), 502, "upstream_error"),
    ],
)
def test_handler_maps_expected_failures_to_clear_statuses(
    monkeypatch,
    exception: Exception,
    status: int,
    code: str,
) -> None:
    instance, responses = _bare_handler(
        monkeypatch,
        provided_token=VALID_INTERNAL_TOKEN,
        configured_token=VALID_INTERNAL_TOKEN,
    )
    instance._read_payload = lambda: {"manager_id": 123}

    def fail(_payload):
        raise exception

    monkeypatch.setattr(manager_state, "generate_manager_state", fail)

    instance.do_POST()

    assert responses[0][0] == status
    assert responses[0][1]["error"]["code"] == code


def test_http_json_responses_are_never_cached() -> None:
    instance = object.__new__(manager_state.handler)
    response_statuses: list[int] = []
    headers: list[tuple[str, str]] = []
    instance.send_response = response_statuses.append
    instance.send_header = lambda name, value: headers.append((name, value))
    instance.end_headers = lambda: None
    instance.wfile = BytesIO()

    instance._send_json(200, {"ok": True})

    assert response_statuses == [200]
    assert ("Cache-Control", "no-store") in headers
    assert json.loads(instance.wfile.getvalue()) == {"ok": True}
