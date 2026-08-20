from __future__ import annotations

from datetime import datetime
import unittest

from fpl_app.domain.sources import (
    SOLIO_SOURCE_ID,
    Provenance,
    SourceDefinition,
    SourceRegistry,
    get_source,
    list_sources,
)


class SourceRegistryTests(unittest.TestCase):
    def test_builtin_registry_contains_solio_attribution_and_endpoint(self) -> None:
        source = get_source(SOLIO_SOURCE_ID)
        self.assertEqual(source.attribution, "Solio Analytics")
        self.assertEqual(
            source.data_url,
            "https://fpl.solioanalytics.com/api/data/latest.json",
        )
        self.assertGreaterEqual(len(list_sources()), 3)

    def test_duplicate_ids_are_rejected(self) -> None:
        source = get_source(SOLIO_SOURCE_ID)
        with self.assertRaisesRegex(ValueError, "Duplicate source_id"):
            SourceRegistry((source, source))

    def test_source_id_cannot_escape_snapshot_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_id"):
            SourceDefinition(
                source_id="../outside",
                name="Unsafe",
                homepage_url="https://example.com",
                data_url="https://example.com/data.json",
                access_mode="Public",
                attribution="Example",
                refresh_cadence="Daily",
            )

    def test_provenance_requires_timezone_aware_observation(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone"):
            Provenance(
                source_id=SOLIO_SOURCE_ID,
                source_url=get_source(SOLIO_SOURCE_ID).data_url,
                observed_at=datetime(2026, 8, 20, 9, 0),
                schema_version="1",
            )


if __name__ == "__main__":
    unittest.main()
