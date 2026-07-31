from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from src.common.config import SPOT_DATA_DIR

MANIFEST_PATH = SPOT_DATA_DIR / "manifest.json"
MANIFEST_SCHEMA_VERSION = 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest() -> dict[str, object]:
    if not MANIFEST_PATH.exists():
        return {"schema_version": MANIFEST_SCHEMA_VERSION, "datasets": {}}
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        manifest = cast(dict[str, object], json.load(handle))
    manifest.setdefault("schema_version", MANIFEST_SCHEMA_VERSION)
    manifest.setdefault("datasets", {})
    return manifest


def _save_manifest(manifest: dict[str, object]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = MANIFEST_PATH.with_suffix(".tmp.json")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path.replace(MANIFEST_PATH)


def load_spot_manifest() -> dict[str, object]:
    """Return the canonical spot manifest (empty datasets when absent)."""
    return _load_manifest()


def _update_manifest_record(dataset: str, symbol: str, record: dict[str, object]) -> None:
    """Replace only the matching manifest record after the new file is durable."""
    manifest = _load_manifest()
    datasets = manifest["datasets"]
    assert isinstance(datasets, dict)
    datasets.setdefault(dataset, {})
    dataset_records = datasets[dataset]
    assert isinstance(dataset_records, dict)
    dataset_records[symbol] = record
    _save_manifest(manifest)


def _manifest_record(
    *,
    venue: str,
    instrument: str,
    source_locator: str,
    retrieved_at: str,
    requested_range: str,
    row_count: int,
    min_ts: str,
    max_ts: str,
    sha256: str,
    conversion: dict[str, object] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "venue": venue,
        "instrument": instrument,
        "source_locator": source_locator,
        "retrieved_at": retrieved_at,
        "requested_range": requested_range,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "row_count": row_count,
        "min_ts": min_ts,
        "max_ts": max_ts,
        "sha256": sha256,
    }
    if conversion is not None:
        record["conversion"] = conversion
    return record


def _prior_quality_metadata(dataset: str, symbol: str) -> dict[str, object]:
    """Carry synthetic-bar provenance across ordinary incremental refreshes."""
    manifest = _load_manifest()
    datasets = manifest.get("datasets", {})
    if not isinstance(datasets, dict):
        return {}
    previous = datasets.get(dataset, {})
    if not isinstance(previous, dict) or not isinstance(previous.get(symbol), dict):
        return {}
    old = previous[symbol]
    quality: dict[str, object] = {}
    imputations = old.get("imputations")
    if isinstance(imputations, list):
        quality["imputations"] = imputations
    elif isinstance(old.get("imputation"), dict):
        quality["imputations"] = [old["imputation"]]
    if isinstance(old.get("data_quality"), dict):
        quality["data_quality"] = old["data_quality"]
    return quality
