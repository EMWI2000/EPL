from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from fpl_app.services.personal_fpl_state import (
    ChipUsage,
    DeadlineSnapshotError,
    MANUAL_SOURCE,
    ManualStateValidationError,
    PUBLIC_SOURCE,
    PublicFplNotFound,
    PublicFplPayloadError,
    PublicManagerStateClient,
    chip_statuses_for_event,
    parse_manual_current_state,
    parse_public_manager_summary,
    serialize_deadline_snapshot,
    verify_deadline_snapshot,
)


def _manual_payload() -> dict:
    return {
        "current_squad_ids": list(range(1, 16)),
        "bank_tenths": 7,
        "free_transfers": 3,
        "player_prices": [
            {
                "element_id": element_id,
                "purchase_price_tenths": 45 + element_id,
                "selling_price_tenths": 45 + element_id,
            }
            for element_id in range(1, 16)
        ],
        "no_active_chip_confirmed": True,
        "chips": {
            "3xc": "available",
            "bboost": "used",
            "freehit": "available",
            "wildcard": "available",
        },
        "chip_usage": [{"name": "bboost", "event": 1}],
        "effective_event": 7,
    }


def _public_payloads(entry_id: int = 123) -> tuple[dict[str, object], dict[str, str]]:
    base = "https://fantasy.premierleague.com/api"
    summary_url = f"{base}/entry/{entry_id}/"
    history_url = f"{base}/entry/{entry_id}/history/"
    transfers_url = f"{base}/entry/{entry_id}/transfers/"
    gw1_url = f"{base}/entry/{entry_id}/event/1/picks/"
    gw2_url = f"{base}/entry/{entry_id}/event/2/picks/"
    event_history = {
        "event": 1,
        "bank": 9,
        "value": 1004,
        "event_transfers": 0,
        "event_transfers_cost": 0,
    }
    picks = []
    for position in range(1, 16):
        picks.append(
            {
                "element": 100 + position,
                "position": position,
                "multiplier": 2 if position == 1 else (0 if position > 11 else 1),
                "is_captain": position == 1,
                "is_vice_captain": position == 2,
                "element_type": 1 if position in {1, 15} else 2,
            }
        )
    payloads: dict[str, object] = {
        summary_url: {
            "id": entry_id,
            "name": "Test XI",
            "started_event": 1,
            "current_event": 2,
            "last_deadline_total_transfers": 0,
            "summary_overall_points": 55,
            "summary_overall_rank": 1234,
        },
        history_url: {
            "current": [event_history],
            "past": [],
            "chips": [
                {
                    "name": "bboost",
                    "event": 1,
                    "time": "2026-08-12T03:54:21.742807Z",
                }
            ],
        },
        transfers_url: [
            {
                "event": 1,
                "element_in": 101,
                "element_out": 99,
                "element_in_cost": 50,
                "element_out_cost": 49,
                "time": "2026-08-20T12:00:00Z",
            },
            # A future row must never leak into a last-deadline DTO even if an
            # upstream response unexpectedly includes it.
            {
                "event": 2,
                "element_in": 200,
                "element_out": 201,
                "element_in_cost": 55,
                "element_out_cost": 54,
                "time": "2026-08-22T12:00:00Z",
            },
        ],
        gw1_url: {
            "active_chip": "bboost",
            "automatic_subs": [],
            "entry_history": event_history,
            "picks": picks,
        },
    }
    urls = {
        "summary": summary_url,
        "history": history_url,
        "transfers": transfers_url,
        "gw1": gw1_url,
        "gw2": gw2_url,
    }
    return payloads, urls


def test_manual_state_is_strict_complete_and_json_safe() -> None:
    state = parse_manual_current_state(
        _manual_payload(), known_player_ids=set(range(1, 100))
    )

    assert state.current_squad_ids == tuple(range(1, 16))
    assert state.schema_version == "fpl-personal-state-v2"
    assert state.free_transfers == 3
    assert state.no_active_chip_confirmed is True
    assert state.player_prices[0].element_id == 1
    assert dict(state.chips)["wildcard"] == "available"
    assert [(row.name, row.event) for row in state.chip_usage] == [("bboost", 1)]
    encoded = json.dumps(state.to_server_dict(), allow_nan=False)
    assert '"chip_usage": [{"name": "bboost", "event": 1}]' in encoded
    assert "password" not in encoded.casefold()
    assert "cookie" not in encoded.casefold()


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda value: value.update({"password": "never"}), "credentials"),
        (lambda value: value.update({"unknown": True}), "unsupported"),
        (lambda value: value.update({"free_transfers": 0}), "free_transfers"),
        (lambda value: value.update({"free_transfers": True}), "integer"),
        (
            lambda value: value.update({"no_active_chip_confirmed": 1}),
            "boolean",
        ),
        (
            lambda value: value.pop("no_active_chip_confirmed"),
            "missing required",
        ),
        (lambda value: value.update({"bank_tenths": 1001}), "bank_tenths"),
        (
            lambda value: value["current_squad_ids"].__setitem__(14, 1),
            "15 unique",
        ),
        (
            lambda value: value["player_prices"].pop(),
            "exactly one row",
        ),
        (lambda value: value["chips"].pop("3xc"), "exactly all four"),
        (
            lambda value: value["chips"].update({"bboost": "available"}),
            "inconsistent",
        ),
        (
            lambda value: value.update(
                {"chip_usage": [{"name": "wildcard", "event": 1}]}
            ),
            "legal window",
        ),
        (
            lambda value: value["chip_usage"].append(
                {"name": "bboost", "event": 2}
            ),
            "duplicate bboost usage",
        ),
        (
            lambda value: value["chip_usage"].append(
                {"name": "3xc", "event": 1}
            ),
            "more than one chip",
        ),
        (
            lambda value: value.update(
                {"chip_usage": [{"name": "bboost", "event": 7}]}
            ),
            "earlier than effective_event",
        ),
    ],
)
def test_manual_state_fails_closed(mutator, message: str) -> None:
    payload = _manual_payload()
    mutator(payload)
    with pytest.raises(ManualStateValidationError, match=message):
        parse_manual_current_state(payload)


def test_manual_state_rejects_ids_outside_official_player_pool() -> None:
    with pytest.raises(ManualStateValidationError, match="unknown player ids"):
        parse_manual_current_state(_manual_payload(), known_player_ids=set(range(1, 15)))


def test_chip_statuses_follow_current_half_and_free_hit_boundary() -> None:
    usage = (ChipUsage(name="freehit", event=19),)

    assert chip_statuses_for_event(usage, 20) == {
        "3xc": "available",
        "bboost": "available",
        "freehit": "unavailable",
        "wildcard": "available",
    }
    assert chip_statuses_for_event(usage, 21)["freehit"] == "available"


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (
            1,
            {
                "3xc": "available",
                "bboost": "available",
                "freehit": "unavailable",
                "wildcard": "unavailable",
            },
        ),
        (19, {name: "available" for name in ("3xc", "bboost", "freehit", "wildcard")}),
        (20, {name: "available" for name in ("3xc", "bboost", "freehit", "wildcard")}),
        (38, {name: "available" for name in ("3xc", "bboost", "freehit", "wildcard")}),
    ],
)
def test_empty_chip_inventory_statuses_cover_window_boundaries(
    event: int,
    expected: dict[str, str],
) -> None:
    assert chip_statuses_for_event((), event) == expected


def test_manual_state_rejects_consecutive_free_hits_across_halves() -> None:
    payload = _manual_payload()
    payload["effective_event"] = 21
    payload["chip_usage"] = [
        {"name": "freehit", "event": 19},
        {"name": "freehit", "event": 20},
    ]
    payload["chips"] = {
        "3xc": "available",
        "bboost": "available",
        "freehit": "used",
        "wildcard": "available",
    }

    with pytest.raises(ManualStateValidationError, match="consecutive events 19 and 20"):
        parse_manual_current_state(payload)


def test_manual_state_accepts_one_use_per_chip_in_each_half() -> None:
    payload = _manual_payload()
    payload["effective_event"] = 21
    payload["chip_usage"] = [
        {"name": "bboost", "event": 1},
        {"name": "bboost", "event": 20},
    ]
    payload["chips"] = {
        "3xc": "available",
        "bboost": "used",
        "freehit": "available",
        "wildcard": "available",
    }

    state = parse_manual_current_state(payload)

    assert [(row.name, row.event) for row in state.chip_usage] == [
        ("bboost", 1),
        ("bboost", 20),
    ]


def test_public_client_uses_latest_history_event_without_probing_future_picks() -> None:
    payloads, urls = _public_payloads()
    calls: list[str] = []

    def transport(url: str):
        calls.append(url)
        return deepcopy(payloads[url])

    state = PublicManagerStateClient(transport).fetch_last_deadline_state(123)

    assert state.event == 1
    assert state.squad_ids == tuple(range(101, 116))
    assert state.bank_tenths == 9
    assert state.squad_value_tenths == 1004
    assert state.active_chip == "bboost"
    assert [(row.name, row.event) for row in state.chip_usage] == [("bboost", 1)]
    assert len(state.public_transfers) == 1
    assert "current_free_transfers_not_public" in state.limitations
    assert calls == [
        urls["summary"],
        urls["history"],
        urls["gw1"],
        urls["transfers"],
    ]
    assert urls["gw2"] not in calls
    json.dumps(state.to_server_dict(), allow_nan=False)
    assert state.to_server_dict()["chip_usage"] == [
        {"name": "bboost", "event": 1}
    ]
    assert "2026-08-12T03:54:21.742807Z" not in json.dumps(
        state.to_server_dict(), allow_nan=False
    )


@pytest.mark.parametrize(
    ("chip_rows", "active_chip", "message"),
    [
        (
            [{"name": "assistant_manager", "event": 1, "time": "2026-08-12T00:00:00Z"}],
            None,
            "must be one of",
        ),
        (
            [{"name": "wildcard", "event": 1, "time": "2026-08-12T00:00:00Z"}],
            None,
            "legal window",
        ),
        (
            [
                {"name": "bboost", "event": 1, "time": "2026-08-12T00:00:00Z"},
                {"name": "bboost", "event": 2, "time": "2026-08-22T00:00:00Z"},
            ],
            None,
            "duplicate bboost usage",
        ),
        ([], "bboost", "inconsistent"),
        (
            [{"name": "bboost", "event": 1, "time": "2026-08-12T00:00:00Z"}],
            None,
            "inconsistent",
        ),
        (
            [{"name": "bboost", "event": 20, "time": "2027-01-06T18:30:00Z"}],
            None,
            "outside the public manager event range",
        ),
        (
            [{"name": "bboost", "event": 1, "time": "not-a-timestamp"}],
            "bboost",
            "ISO UTC timestamp",
        ),
        (
            [
                {
                    "name": "bboost",
                    "event": 1,
                    "time": "2026-08-12T00:00:00Z",
                    "extra": True,
                }
            ],
            "bboost",
            "contain exactly",
        ),
        (
            [
                {"name": "freehit", "event": 19, "time": "2027-01-01T00:00:00Z"},
                {"name": "freehit", "event": 20, "time": "2027-01-03T00:00:00Z"},
            ],
            None,
            "consecutive events 19 and 20",
        ),
    ],
)
def test_public_client_rejects_invalid_chip_history(
    chip_rows: list[dict[str, object]],
    active_chip: str | None,
    message: str,
) -> None:
    payloads, urls = _public_payloads()
    history = payloads[urls["history"]]
    picks = payloads[urls["gw1"]]
    assert isinstance(history, dict)
    assert isinstance(picks, dict)
    history["chips"] = chip_rows
    picks["active_chip"] = active_chip

    def transport(url: str):
        return deepcopy(payloads[url])

    with pytest.raises(PublicFplPayloadError, match=message):
        PublicManagerStateClient(transport).fetch_last_deadline_state(123)


def test_public_client_selects_max_event_from_unsorted_history() -> None:
    payloads, urls = _public_payloads()
    event_two_history = {
        "event": 2,
        "bank": 7,
        "value": 1008,
        "event_transfers": 1,
        "event_transfers_cost": 0,
    }
    history = payloads[urls["history"]]
    assert isinstance(history, dict)
    history["current"] = [event_two_history, *history["current"]]
    gw_two = deepcopy(payloads[urls["gw1"]])
    assert isinstance(gw_two, dict)
    gw_two["entry_history"] = event_two_history
    gw_two["active_chip"] = None
    payloads[urls["gw2"]] = gw_two
    calls: list[str] = []

    def transport(url: str):
        calls.append(url)
        return deepcopy(payloads[url])

    state = PublicManagerStateClient(transport).fetch_last_deadline_state(123)

    assert state.event == 2
    assert state.bank_tenths == 7
    assert state.event_transfers == 1
    assert calls == [
        urls["summary"],
        urls["history"],
        urls["gw2"],
        urls["transfers"],
    ]
    assert urls["gw1"] not in calls


def test_public_client_reuses_validated_summary_without_fetching_it_again() -> None:
    payloads, urls = _public_payloads()
    validated = parse_public_manager_summary(
        payloads[urls["summary"]],
        expected_entry_id=123,
    )
    calls: list[str] = []

    def transport(url: str):
        calls.append(url)
        if url == urls["summary"]:
            pytest.fail("validated manager summary was fetched again")
        return deepcopy(payloads[url])

    state = PublicManagerStateClient(transport).fetch_last_deadline_state(
        123,
        validated_summary=validated,
    )

    assert state.entry_id == 123
    assert validated.to_server_dict() == {
        "id": 123,
        "team_name": "Test XI",
        "overall_points": 55,
        "overall_rank": 1234,
    }
    assert calls == [urls["history"], urls["gw1"], urls["transfers"]]


def test_public_client_rejects_incomplete_or_inconsistent_picks() -> None:
    payloads, urls = _public_payloads()
    payloads[urls["gw1"]]["picks"][1]["is_captain"] = True  # type: ignore[index]

    def transport(url: str):
        return deepcopy(payloads[url])

    with pytest.raises(PublicFplPayloadError, match="one captain"):
        PublicManagerStateClient(transport).fetch_last_deadline_state(123)


def test_deadline_snapshot_is_deterministic_and_detects_tampering() -> None:
    state = parse_manual_current_state(_manual_payload())
    observed_at = datetime(2026, 8, 22, 12, 30, tzinfo=timezone.utc)

    first = serialize_deadline_snapshot(
        state, observed_at=observed_at, source=MANUAL_SOURCE
    )
    second = serialize_deadline_snapshot(
        state, observed_at=observed_at, source=MANUAL_SOURCE
    )

    assert first == second
    assert first["schema_version"] == "fpl-deadline-state-snapshot-v2"
    assert first["observed_at"] == "2026-08-22T12:30:00.000000Z"
    assert len(first["checksum_sha256"]) == 64
    assert verify_deadline_snapshot(first) == first

    tampered = deepcopy(first)
    tampered["state"]["bank_tenths"] = 999
    with pytest.raises(DeadlineSnapshotError, match="checksum"):
        verify_deadline_snapshot(tampered)


def test_deadline_snapshot_rejects_naive_time_and_wrong_source() -> None:
    state = parse_manual_current_state(_manual_payload())
    with pytest.raises(DeadlineSnapshotError, match="timezone-aware"):
        serialize_deadline_snapshot(
            state,
            observed_at=datetime(2026, 8, 22, 12, 30),
            source=MANUAL_SOURCE,
        )
    with pytest.raises(DeadlineSnapshotError, match="state kind"):
        serialize_deadline_snapshot(
            state,
            observed_at=datetime.now(timezone.utc),
            source=PUBLIC_SOURCE,
        )
