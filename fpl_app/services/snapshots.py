"""Atomic, point-in-time storage for untransformed JSON payloads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Iterator, Mapping, Optional

try:  # Supports both ``python -m fpl_app...`` and Streamlit from fpl_app/.
    from ..domain.sources import Provenance, datetime_to_iso, get_source
except ImportError:  # pragma: no cover - exercised by the deployed import layout
    from domain.sources import Provenance, datetime_to_iso, get_source


SNAPSHOT_FORMAT_VERSION = 1


class SnapshotError(RuntimeError):
    """Base error for invalid or corrupted snapshots."""


class SnapshotIntegrityError(SnapshotError):
    """Raised when stored content no longer matches its recorded digest."""


@dataclass(frozen=True)
class SnapshotRef:
    source_id: str
    snapshot_id: str
    data_path: Path
    metadata_path: Path


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    try:
        if pretty:
            text = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ) + "\n"
        else:
            # Canonical encoding makes de-duplication and integrity checks stable.
            text = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
    except (TypeError, ValueError) as exc:
        raise SnapshotError(f"Payload is not strict JSON: {exc}") from exc
    return text.encode("utf-8")


def _write_and_sync(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _sync_directory(path: Path) -> None:
    """Best-effort directory fsync for durable atomic renames on POSIX."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError:  # pragma: no cover - platform/filesystem dependent
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - platform/filesystem dependent
        pass
    finally:
        os.close(descriptor)


def _verify_existing_destination(
    data_path: Path,
    metadata_path: Path,
    expected_digest: str,
) -> None:
    """Accept an idempotent/concurrent write only if the winner is intact."""

    if not data_path.is_file() or not metadata_path.is_file():
        raise SnapshotError(
            f"Snapshot destination exists but is incomplete: {data_path.parent}"
        )
    try:
        data_bytes = data_path.read_bytes()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotIntegrityError(
            f"Existing snapshot is unreadable: {data_path.parent}"
        ) from exc
    actual_digest = hashlib.sha256(data_bytes).hexdigest()
    if actual_digest != expected_digest or metadata.get("content_sha256") != expected_digest:
        raise SnapshotIntegrityError(
            f"Existing snapshot has conflicting content: {data_path.parent}"
        )


class JsonSnapshotStore:
    """Store a payload and its provenance as one atomically visible directory.

    Layout::

        <root>/<source>/YYYY/MM/DD/<observed timestamp>_<digest>/
            data.json
            metadata.json

    A hidden staging directory is fully written and synced before a single rename
    makes the snapshot visible.  This avoids consumers seeing half-written JSON or
    a payload without provenance after a process interruption.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def write(self, payload: Any, provenance: Provenance) -> SnapshotRef:
        source = get_source(provenance.source_id)
        data_bytes = _json_bytes(payload)
        digest = hashlib.sha256(data_bytes).hexdigest()
        observed_at = provenance.observed_at.astimezone(timezone.utc)
        timestamp = observed_at.strftime("%Y%m%dT%H%M%S.%fZ")
        snapshot_id = f"{timestamp}_{digest[:12]}"
        day_dir = (
            self.root
            / provenance.source_id
            / observed_at.strftime("%Y")
            / observed_at.strftime("%m")
            / observed_at.strftime("%d")
        )
        final_dir = day_dir / snapshot_id
        data_path = final_dir / "data.json"
        metadata_path = final_dir / "metadata.json"

        metadata: Dict[str, Any] = {
            "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
            "snapshot_id": snapshot_id,
            "content_sha256": digest,
            "content_bytes": len(data_bytes),
            "ingested_at": datetime_to_iso(datetime.now(timezone.utc)),
            "source": source.to_dict(),
            "provenance": provenance.to_dict(),
        }
        metadata_bytes = _json_bytes(metadata, pretty=True)

        day_dir.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix=".tmp-snapshot-", dir=day_dir))
        try:
            _write_and_sync(staging_dir / "data.json", data_bytes)
            _write_and_sync(staging_dir / "metadata.json", metadata_bytes)
            _sync_directory(staging_dir)
            if final_dir.exists():
                # Same source, timestamp and payload is an idempotent write.
                _verify_existing_destination(data_path, metadata_path, digest)
            else:
                try:
                    os.rename(staging_dir, final_dir)
                except OSError as exc:
                    # Another writer may have won the race between exists() and
                    # rename(). Accept that only when it completed both files.
                    try:
                        _verify_existing_destination(data_path, metadata_path, digest)
                    except SnapshotError:
                        raise SnapshotError(
                            f"Could not publish snapshot atomically: {final_dir}"
                        ) from exc
            _sync_directory(day_dir)
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)

        return SnapshotRef(
            source_id=provenance.source_id,
            snapshot_id=snapshot_id,
            data_path=data_path,
            metadata_path=metadata_path,
        )

    def read_metadata(self, ref: SnapshotRef) -> Mapping[str, Any]:
        try:
            metadata = json.loads(ref.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotError(f"Cannot read snapshot metadata: {ref.metadata_path}") from exc
        if not isinstance(metadata, dict):
            raise SnapshotError("Snapshot metadata must be a JSON object")
        if metadata.get("snapshot_format_version") != SNAPSHOT_FORMAT_VERSION:
            raise SnapshotError("Unsupported snapshot format version")
        if metadata.get("snapshot_id") != ref.snapshot_id:
            raise SnapshotIntegrityError("Snapshot identifier does not match metadata")
        if metadata.get("provenance", {}).get("source_id") != ref.source_id:
            raise SnapshotIntegrityError("Snapshot source does not match metadata")
        return metadata

    def read(self, ref: SnapshotRef, *, verify: bool = True) -> Any:
        metadata = self.read_metadata(ref)
        try:
            data_bytes = ref.data_path.read_bytes()
        except OSError as exc:
            raise SnapshotError(f"Cannot read snapshot data: {ref.data_path}") from exc
        if verify:
            actual_digest = hashlib.sha256(data_bytes).hexdigest()
            if actual_digest != metadata.get("content_sha256"):
                raise SnapshotIntegrityError(
                    f"Digest mismatch for snapshot {ref.snapshot_id}"
                )
            if len(data_bytes) != metadata.get("content_bytes"):
                raise SnapshotIntegrityError(
                    f"Byte-count mismatch for snapshot {ref.snapshot_id}"
                )
        try:
            return json.loads(data_bytes)
        except json.JSONDecodeError as exc:
            raise SnapshotError(f"Stored payload is invalid JSON: {ref.data_path}") from exc

    def iter_snapshots(self, source_id: str) -> Iterator[SnapshotRef]:
        # Registry lookup validates the source and protects the filesystem path.
        get_source(source_id)
        source_dir = self.root / source_id
        if not source_dir.exists():
            return
        metadata_paths = sorted(source_dir.glob("*/*/*/*/metadata.json"), reverse=True)
        for metadata_path in metadata_paths:
            snapshot_dir = metadata_path.parent
            if snapshot_dir.name.startswith("."):
                continue
            yield SnapshotRef(
                source_id=source_id,
                snapshot_id=snapshot_dir.name,
                data_path=snapshot_dir / "data.json",
                metadata_path=metadata_path,
            )

    def latest(self, source_id: str) -> Optional[SnapshotRef]:
        return next(self.iter_snapshots(source_id), None)
