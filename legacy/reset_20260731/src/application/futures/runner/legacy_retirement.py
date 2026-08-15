from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

_logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class LegacyRetirementReport:
    migration_hash_match: bool
    snapshot_complete: bool
    smoke_run_passed: bool
    unresolved_references: tuple[str, ...]
    deletion_targets: tuple[Path, ...]


def migrate_legacy_universe_state(*, source_ledger: Path, catalog: object, root: Path) -> str:
    from src.domain.futures.data_lake.ingestion import migrate_legacy_universe_state as _ingest_migrate
    return _ingest_migrate(source_ledger=source_ledger, catalog=catalog, root=root)  # type: ignore[arg-type]


def retire_legacy_storage(*, report: LegacyRetirementReport, approved: bool) -> tuple[Path, ...]:
    if not approved:
        raise RuntimeError("retirement preflight: not approved")

    if not report.migration_hash_match:
        raise RuntimeError("retirement preflight: migration_hash_match=False")
    if not report.snapshot_complete:
        raise RuntimeError("retirement preflight: snapshot_complete=False")
    if not report.smoke_run_passed:
        raise RuntimeError("retirement preflight: smoke_run_passed=False")
    if report.unresolved_references:
        raise RuntimeError(
            f"retirement preflight: unresolved_references={report.unresolved_references}"
        )
    if not report.deletion_targets:
        _logger.warning("retirement preflight: no deletion targets")
        return ()

    deleted: list[Path] = []
    for target in report.deletion_targets:
        if target.is_dir():
            import shutil
            shutil.rmtree(target)
            _logger.info("deleted directory: %s", target)
        elif target.exists():
            target.unlink()
            _logger.info("deleted file: %s", target)
        else:
            _logger.warning("target not found, skipping: %s", target)
            continue
        deleted.append(target)

    return tuple(deleted)
