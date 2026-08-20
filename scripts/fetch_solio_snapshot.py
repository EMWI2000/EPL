"""Fetch, validate and atomically store the latest public Solio feed."""

from __future__ import annotations

import argparse
from pathlib import Path

from services.snapshots import JsonSnapshotStore
from services.solio import SolioClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/snapshots"),
        help="Snapshot root (default: data/snapshots)",
    )
    args = parser.parse_args()

    ref = SolioClient().fetch_and_store(JsonSnapshotStore(args.root))
    print(f"Stored {ref.source_id}/{ref.snapshot_id}")
    print(ref.data_path)
    print(ref.metadata_path)


if __name__ == "__main__":
    main()
