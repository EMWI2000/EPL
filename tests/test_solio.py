from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import unittest

from fpl_app.domain.sources import SOLIO_SOURCE_ID
from fpl_app.services.solio import (
    HttpResponse,
    SOLIO_SCHEMA_VERSION,
    SolioClient,
    SolioHTTPError,
    SolioValidationError,
    validate_solio_payload,
)


def valid_payload() -> dict:
    payload = {
        "generatedAt": "2026-08-20T08:35:56.210Z",
        "gameweek": 1,
        "deadlineIso": "2026-08-21T17:30:00.000Z",
        "source": "https://fpl.solioanalytics.com/api/data/latest",
        "topProjected": [
            {
                "name": "Haaland",
                "team": "MCI",
                "position": "FWD",
                "price": 155,
                "opponents": [{"opponent": "BOU", "isHome": True}],
                "ownership": 69.5,
                "prPoints": 6.15,
            }
        ],
    }
    for section in (
        "topCaptains",
        "topDifferentials",
        "topGoals",
        "topAssists",
        "bestCleanSheets",
        "topBonus",
        "topDefCon",
        "bestAttackingFixtures",
        "topTransfersIn",
        "topTransfersOut",
    ):
        payload[section] = []
    return payload


class SolioValidationTests(unittest.TestCase):
    def test_valid_payload_is_returned_without_transforming_additive_fields(self) -> None:
        payload = valid_payload()
        payload["futureField"] = {"kept": True}
        validated = validate_solio_payload(payload)
        self.assertIs(validated, payload)
        self.assertEqual(validated["futureField"], {"kept": True})

    def test_missing_section_reports_its_path(self) -> None:
        payload = valid_payload()
        del payload["topDefCon"]
        with self.assertRaisesRegex(SolioValidationError, "topDefCon"):
            validate_solio_payload(payload)

    def test_naive_upstream_timestamp_is_rejected(self) -> None:
        payload = valid_payload()
        payload["generatedAt"] = "2026-08-20T08:35:56"
        with self.assertRaisesRegex(SolioValidationError, "timezone"):
            validate_solio_payload(payload)

    def test_non_finite_projection_is_rejected(self) -> None:
        payload = valid_payload()
        payload["topProjected"][0]["prPoints"] = float("nan")
        with self.assertRaisesRegex(SolioValidationError, "prPoints"):
            validate_solio_payload(payload)

    def test_empty_projection_list_can_be_allowed_for_offseason_capture(self) -> None:
        payload = valid_payload()
        payload["topProjected"] = []
        validate_solio_payload(payload, require_projection_rows=False)


class SolioClientTests(unittest.TestCase):
    def test_fetch_uses_injected_transport_and_builds_provenance(self) -> None:
        payload = valid_payload()
        calls = []

        def transport(url, timeout, headers):
            calls.append((url, timeout, headers))
            return HttpResponse(200, json.dumps(payload).encode(), {"Content-Type": "application/json"})

        now = datetime(2026, 8, 20, 9, 1, tzinfo=timezone.utc)
        result = SolioClient(
            timeout=7,
            transport=transport,
            clock=lambda: now,
        ).fetch_latest()

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], 7.0)
        self.assertEqual(calls[0][2]["Accept"], "application/json")
        self.assertEqual(result.gameweek, 1)
        self.assertEqual(result.observed_at, now)
        self.assertEqual(result.provenance.source_id, SOLIO_SOURCE_ID)
        self.assertEqual(result.provenance.schema_version, SOLIO_SCHEMA_VERSION)
        self.assertEqual(
            result.generated_at,
            datetime(2026, 8, 20, 8, 35, 56, 210000, tzinfo=timezone.utc),
        )

    def test_http_failure_does_not_attempt_to_parse_body(self) -> None:
        client = SolioClient(
            transport=lambda *_: HttpResponse(503, b"maintenance", {}),
        )
        with self.assertRaisesRegex(SolioHTTPError, "503"):
            client.fetch_latest()

    def test_invalid_json_is_rejected(self) -> None:
        client = SolioClient(
            transport=lambda *_: HttpResponse(200, b"not-json", {}),
        )
        with self.assertRaisesRegex(SolioValidationError, "valid UTF-8 JSON"):
            client.fetch_latest()


if __name__ == "__main__":
    unittest.main()
