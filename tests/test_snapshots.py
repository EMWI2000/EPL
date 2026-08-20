from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from fpl_app.domain.sources import SOLIO_SOURCE_ID, Provenance, get_source
from fpl_app.services.snapshots import (
    JsonSnapshotStore,
    SnapshotError,
    SnapshotIntegrityError,
)


class JsonSnapshotStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.store = JsonSnapshotStore(self.root)
        self.observed_at = datetime(
            2026, 8, 20, 10, 15, 30, 123456, tzinfo=timezone.utc
        )
        self.provenance = Provenance(
            source_id=SOLIO_SOURCE_ID,
            source_url=get_source(SOLIO_SOURCE_ID).data_url,
            observed_at=self.observed_at,
            effective_at=datetime(2026, 8, 20, 8, 35, tzinfo=timezone.utc),
            schema_version="test-v1",
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_write_preserves_payload_and_records_point_in_time_metadata(self) -> None:
        payload = {"gameweek": 1, "players": [{"name": "Ødegaard", "xPts": 5.2}]}
        ref = self.store.write(payload, self.provenance)

        self.assertTrue(ref.data_path.is_file())
        self.assertTrue(ref.metadata_path.is_file())
        self.assertIn("/solio/2026/08/20/", ref.data_path.as_posix())
        self.assertEqual(self.store.read(ref), payload)

        metadata = self.store.read_metadata(ref)
        self.assertEqual(metadata["provenance"]["source_id"], SOLIO_SOURCE_ID)
        self.assertEqual(
            metadata["provenance"]["observed_at"],
            "2026-08-20T10:15:30.123456Z",
        )
        self.assertEqual(metadata["source"]["attribution"], "Solio Analytics")
        self.assertEqual(len(metadata["content_sha256"]), 64)
        self.assertTrue(metadata["ingested_at"].endswith("Z"))

    def test_identical_write_is_idempotent_and_leaves_no_staging_directories(self) -> None:
        first = self.store.write({"a": 1}, self.provenance)
        second = self.store.write({"a": 1}, self.provenance)
        self.assertEqual(first, second)
        self.assertFalse(any(self.root.rglob(".tmp-snapshot-*")))

    def test_latest_orders_by_utc_observation_timestamp(self) -> None:
        first = self.store.write({"value": 1}, self.provenance)
        later_provenance = Provenance(
            source_id=SOLIO_SOURCE_ID,
            source_url=get_source(SOLIO_SOURCE_ID).data_url,
            observed_at=datetime(2026, 8, 21, 9, 0, tzinfo=timezone.utc),
            schema_version="test-v1",
        )
        second = self.store.write({"value": 2}, later_provenance)
        self.assertEqual(self.store.latest(SOLIO_SOURCE_ID), second)
        self.assertNotEqual(first.snapshot_id, second.snapshot_id)

    def test_tampered_payload_fails_integrity_check(self) -> None:
        ref = self.store.write({"safe": True}, self.provenance)
        ref.data_path.write_text(json.dumps({"safe": False}), encoding="utf-8")
        with self.assertRaises(SnapshotIntegrityError):
            self.store.read(ref)

    def test_idempotent_write_does_not_accept_a_corrupted_destination(self) -> None:
        ref = self.store.write({"safe": True}, self.provenance)
        ref.data_path.write_text(json.dumps({"safe": False}), encoding="utf-8")
        with self.assertRaises(SnapshotIntegrityError):
            self.store.write({"safe": True}, self.provenance)

    def test_non_json_values_are_rejected_before_any_directory_is_created(self) -> None:
        with self.assertRaisesRegex(SnapshotError, "strict JSON"):
            self.store.write({"bad": float("nan")}, self.provenance)
        self.assertFalse((self.root / SOLIO_SOURCE_ID).exists())


if __name__ == "__main__":
    unittest.main()
