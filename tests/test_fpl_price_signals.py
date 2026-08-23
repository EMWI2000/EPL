from __future__ import annotations

from copy import deepcopy
import json

import pytest

from fpl_app.services.fpl_price_signals import (
    OfficialPricePayloadError,
    parse_bootstrap_price_feed,
)


def _bootstrap() -> dict:
    return {
        "game_config": {
            "settings": {
                "price_change_deadlines": [
                    "2026-08-22T23:00:00Z",
                    "2026-08-23T23:00:00Z",
                ]
            }
        },
        "elements": [
            {
                "id": 7,
                "now_cost": 60,
                "cost_change_start": 1,
                "cost_change_event": 1,
                "selected_by_percent": "37.7",
                "transfers_in_event": 6652,
                "transfers_out_event": 5059,
                "price_change_percent": "101.1",
                "price_change_hourly_rate": 208,
                "price_change_projections": [
                    {"offset": 0, "projected_percent": "102.2", "likelihood": 4},
                    {"offset": 1, "projected_percent": "105.3", "likelihood": 4},
                ],
                "price_change_locked_until": "2026-08-24T00:00:00+01:00",
                "price_change_calibrating": False,
                "ignored_future_field": {"not": "leaked"},
            },
            {
                "id": 2,
                "now_cost": 45,
                "cost_change_start": 0,
                "cost_change_event": 0,
                "selected_by_percent": "0.1",
                "transfers_in_event": 12,
                "transfers_out_event": 30,
                "price_change_percent": None,
                "price_change_hourly_rate": None,
                "price_change_projections": [],
                "price_change_locked_until": None,
                "price_change_calibrating": True,
            },
        ],
    }


def test_price_feed_normalises_allowlisted_official_fields() -> None:
    feed = parse_bootstrap_price_feed(_bootstrap())

    assert [player.element_id for player in feed.players] == [2, 7]
    signal = feed.players[1]
    assert signal.now_cost_tenths == 60
    assert signal.selected_by_percent == pytest.approx(37.7)
    assert signal.price_change_percent == pytest.approx(101.1)
    assert signal.projections[0].offset_days == 0
    assert signal.projections[0].likelihood_code == 4
    assert signal.locked_until == "2026-08-23T23:00:00Z"
    payload = feed.to_server_dict()
    assert "ignored_future_field" not in json.dumps(payload)
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize(
    "mutator, message",
    [
        (
            lambda data: data["elements"].append(deepcopy(data["elements"][0])),
            "duplicate element",
        ),
        (
            lambda data: data["elements"][0].update(
                {"price_change_percent": float("nan")}
            ),
            "finite",
        ),
        (
            lambda data: data["elements"][0].update(
                {"price_change_calibrating": 0}
            ),
            "boolean",
        ),
        (
            lambda data: data["elements"][0].update(
                {"price_change_locked_until": "2026-08-23T23:00:00"}
            ),
            "timezone",
        ),
        (
            lambda data: data["elements"][0]["price_change_projections"].append(
                {"offset": 0, "projected_percent": "99", "likelihood": 1}
            ),
            "offsets",
        ),
        (
            lambda data: data["game_config"]["settings"].update(
                {
                    "price_change_deadlines": [
                        "2026-08-23T23:00:00Z",
                        "2026-08-22T23:00:00Z",
                    ]
                }
            ),
            "chronologically",
        ),
    ],
)
def test_price_feed_fails_closed_on_invalid_signals(mutator, message: str) -> None:
    payload = _bootstrap()
    mutator(payload)
    with pytest.raises(OfficialPricePayloadError, match=message):
        parse_bootstrap_price_feed(payload)


def test_price_feed_requires_official_container_shape() -> None:
    with pytest.raises(OfficialPricePayloadError, match="bootstrap.elements"):
        parse_bootstrap_price_feed({"game_config": {"settings": {}}})
