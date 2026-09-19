"""Compact inventory summaries with lossless content-addressed detail bundles.

The compact summary preserves every financial scalar of the legacy full JSON
payload; only the large daily-return dictionaries and gap arrays are replaced
by content-identity references. Complete details live in one atomic bundle of
Zstd Parquet files addressed by the canonical manifest hash, so identical
evidence is stored once and reused by identity.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.common.errors import DataIntegrityError
from src.mhs.backtest.inventory import (
    _UNPRICED_TERMINAL_CODES,
    _inventory_daily_returns,
    _inventory_ledger_summary,
)
from src.mhs.execution.contracts import TerminalPositionEvidence
from src.mhs.execution.integrity import replay_ledger_certified
from src.mhs.reporting.process import _tier_payload

if TYPE_CHECKING:
    from src.mhs.backtest.certification import ProcessValidationResult
    from src.mhs.backtest.contracts import ProcessInventoryFailureReport, ProcessInventoryReport
    from src.mhs.execution.contracts import ExecutionDataGap, FundingCoverageGap, StrategyExecutionReplayResult

_VALIDATION_SCHEMA = "process_validation/1"
_LEGACY_VALIDATION_ABSENT = "LEGACY_VALIDATION_ABSENT"

_MANIFEST_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_SCHEMA_VERSION = 1
_ROW_GROUP_SIZE = 10_000
_LEASE_ROLE = "publication_lease"
_LEDGER_SERIES_FIELDS: tuple[str, ...] = (
    "equity",
    "net_returns",
    "mark_to_market_pnl",
    "funding_charge",
    "fee_charge",
    "fill_turnover",
)


def _utc_index(values: Sequence[Any]) -> pd.DatetimeIndex:
    stamps = pd.to_datetime(list(values), utc=True)
    return pd.DatetimeIndex(stamps)


def _daily_table(daily: pd.Series) -> pa.Table:
    values = daily.to_numpy(dtype="float64")
    return pa.table(
        {
            "timestamp": pa.array(_utc_index(daily.index), type=pa.timestamp("ns", tz="UTC")),
            "daily_return": pa.array(values, type=pa.float64()),
        }
    )


def _ledger_table(series: pd.Series, field: str) -> pa.Table:
    name = series.name if isinstance(series.name, str) else field
    values = series.to_numpy()
    return pa.table(
        {
            "timestamp": pa.array(_utc_index(series.index), type=pa.timestamp("ns", tz="UTC")),
            name: pa.array(values, type=pa.from_numpy_dtype(values.dtype)),
        }
    )


def _fills_table(fills: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(fills, preserve_index=True)


def _gaps_table(gaps: Sequence[ExecutionDataGap]) -> pa.Table:
    return pa.table(
        {
            "ordinal": pa.array(range(len(gaps)), type=pa.int64()),
            "code": pa.array([str(gap.code) for gap in gaps], type=pa.string()),
            "symbol": pa.array([str(gap.symbol) for gap in gaps], type=pa.string()),
            "timestamp": pa.array(_utc_index([gap.timestamp for gap in gaps]), type=pa.timestamp("ns", tz="UTC")),
            "decision_time": pa.array(
                _utc_index([gap.decision_time for gap in gaps]), type=pa.timestamp("ns", tz="UTC")
            ),
            "signal_time": pa.array(_utc_index([gap.signal_time for gap in gaps]), type=pa.timestamp("ns", tz="UTC")),
            "execution_bound": pa.array([str(gap.execution_bound) for gap in gaps], type=pa.string()),
        }
    )


def _coverage_table(gaps: Sequence[FundingCoverageGap]) -> pa.Table:
    return pa.table(
        {
            "ordinal": pa.array(range(len(gaps)), type=pa.int64()),
            "symbol": pa.array([str(gap.symbol) for gap in gaps], type=pa.string()),
            "start": pa.array(_utc_index([gap.start for gap in gaps]), type=pa.timestamp("ns", tz="UTC")),
            "end": pa.array(_utc_index([gap.end for gap in gaps]), type=pa.timestamp("ns", tz="UTC")),
            "reason": pa.array([str(gap.reason) for gap in gaps], type=pa.string()),
        }
    )


def _coverage_payload(gap: FundingCoverageGap) -> dict[str, object]:
    return {
        "symbol": str(gap.symbol),
        "start": gap.start.isoformat(),
        "end": gap.end.isoformat(),
        "reason": str(gap.reason),
    }


def _tier_tables(tier: str, result: StrategyExecutionReplayResult, daily: pd.Series) -> dict[str, pa.Table]:
    tables = {f"{tier}_daily.parquet": _daily_table(daily)}
    for field in _LEDGER_SERIES_FIELDS:
        tables[f"{tier}_ledger_{field}.parquet"] = _ledger_table(getattr(result.ledger, field), field)
    tables[f"{tier}_fills.parquet"] = _fills_table(result.simulated_fills)
    tables[f"{tier}_gaps.parquet"] = _gaps_table(list(result.ledger.data_gaps))
    tables[f"{tier}_coverage.parquet"] = _coverage_table(list(result.funding_coverage_gaps))
    return tables


def _write_parquet(path: Path, table: pa.Table) -> None:
    with pq.ParquetWriter(str(path), table.schema, compression="zstd") as writer:
        for offset in range(0, table.num_rows, _ROW_GROUP_SIZE):
            writer.write_table(table.slice(offset, _ROW_GROUP_SIZE), row_group_size=_ROW_GROUP_SIZE)


def _file_record(bundle: Path, name: str) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    with open(bundle / name, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"role": name, "name": name, "size": size, "sha256": digest.hexdigest()}


def _canonical_manifest(files: list[dict[str, object]]) -> dict[str, object]:
    ordered = sorted(files, key=lambda entry: str(entry["name"]))
    return {"manifest_version": _MANIFEST_VERSION, "files": ordered}


def _manifest_identity(canonical: dict[str, object]) -> str:
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_bundle(bundle: Path, identity: str) -> None:
    try:
        manifest = json.loads((bundle / _MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataIntegrityError(f"unreadable evidence manifest: {bundle}") from exc
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("manifest_version") != _MANIFEST_VERSION
        or manifest.get("evidence_id") != identity
        or not isinstance(files, list)
    ):
        raise DataIntegrityError(f"corrupt evidence manifest: {bundle}")
    claimed: list[dict[str, object]] = []
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise DataIntegrityError(f"corrupt evidence manifest: {bundle}")
        name = str(entry["name"])
        try:
            record = _file_record(bundle, name)
        except OSError as exc:
            raise DataIntegrityError(f"missing evidence file: {name}") from exc
        if record["size"] != entry.get("size") or record["sha256"] != entry.get("sha256"):
            raise DataIntegrityError(f"conflicting evidence file: {name}")
        claimed.append({"role": name, "name": name, "size": record["size"], "sha256": record["sha256"]})
    if _manifest_identity(_canonical_manifest(claimed)) != identity:
        raise DataIntegrityError(f"conflicting evidence bundle: {bundle}")


def _publish_bundle(evidence_root: Path, tables: dict[str, pa.Table]) -> tuple[Path, str]:
    staging = evidence_root / f".staging_{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        for name, table in tables.items():
            _write_parquet(staging / name, table)
        files = [_file_record(staging, name) for name in sorted(tables)]
        canonical = _canonical_manifest(files)
        identity = _manifest_identity(canonical)
        manifest = dict(canonical)
        manifest["evidence_id"] = identity
        (staging / _MANIFEST_NAME).write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
        target = evidence_root / identity
        if target.is_symlink() or target.is_file():
            raise DataIntegrityError(f"evidence slot occupied by non-bundle: {target}")
        if target.is_dir():
            _verify_bundle(target, identity)
            return target, identity
        os.rename(staging, target)
        return target, identity
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or len(run_id) != 32 or any(c not in "0123456789abcdefABCDEF" for c in run_id):
        raise ValueError(f"run_id must be UUID hex, got {run_id!r}")


def _record_publication_lease(registry_path: Path, run_id: str, bundle: Path, evidence_id: str) -> None:
    _validate_run_id(run_id)
    total_bytes = sum(entry.stat().st_size for entry in bundle.iterdir() if entry.is_file())
    conn = sqlite3.connect(str(registry_path), timeout=5.0, isolation_level="DEFERRED")
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO artifacts (run_id, role, path, sha256, byte_count, managed, evidence_id, retained)"
                " VALUES (?, ?, ?, ?, ?, 1, ?, 1)",
                (run_id, _LEASE_ROLE, str(bundle / _MANIFEST_NAME), evidence_id, total_bytes, evidence_id),
            )
    finally:
        conn.close()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, path)
    except Exception:
        with suppress(OSError):
            os.unlink(tmp_name)
        raise


def _detail_reference(identity: str, role: str, count: int) -> dict[str, object]:
    return {"evidence_id": identity, "role": role, "count": count}


def persist_inventory_evidence(
    report: ProcessInventoryReport,
    output: Path,
    *,
    evidence_root: Path,
    registry_path: Path | None = None,
    run_id: str | None = None,
) -> tuple[Path, str]:
    """Persist compact three-minute inventory evidence and its lossless content-addressed details. Execution completion does not certify the financial ledger. Args: evaluated report, fresh summary destination, evidence root and paired optional registry/execution identity for managed publication protection. Returns: summary destination and verified evidence identity. Raises: ValueError for invalid paths or incomplete ownership context; DataIntegrityError for conflicting evidence; OSError or sqlite3.Error for persistence failure."""

    out = Path(output)
    if out.suffix != ".json":
        raise ValueError(f"destination must be a JSON path, got {output}")
    if out.exists():
        raise ValueError(f"summary destination must be fresh, got {out}")
    if (registry_path is None) != (run_id is None):
        raise ValueError("registry_path and run_id must be provided together or both be None")
    root = Path(evidence_root)
    if root.is_symlink():
        raise ValueError(f"evidence_root must be an owned directory, got {root}")
    root.mkdir(parents=True, exist_ok=True)
    legacy = _inventory_summary_payload(report)
    daily_base = _inventory_daily_returns(report.base)
    daily_stress = _inventory_daily_returns(report.stress)
    tables = _tier_tables("base", report.base, daily_base)
    tables.update(_tier_tables("stress", report.stress, daily_stress))
    tables["coverage.parquet"] = _coverage_table(list(report.funding_coverage_gaps))
    bundle, identity = _publish_bundle(root, tables)
    if registry_path is not None and run_id is not None:
        _record_publication_lease(Path(registry_path), run_id, bundle, identity)
    summary = dict(legacy)
    summary["schema_version"] = _SCHEMA_VERSION
    summary["evidence_id"] = identity
    summary["evidence"] = {"evidence_id": identity, "bundle_path": str(bundle), "manifest": _MANIFEST_NAME}
    summary["funding_coverage_gaps"] = _detail_reference(identity, "coverage", len(report.funding_coverage_gaps))
    for tier_key, tier_role, daily, gaps in (
        ("base", "base", daily_base, report.base.ledger.data_gaps),
        ("stress", "stress", daily_stress, report.stress.ledger.data_gaps),
    ):
        tier = cast(dict[str, object], legacy[tier_key])
        terminal = cast(dict[str, object], tier["terminal"])
        tier["daily_returns"] = _detail_reference(identity, f"{tier_role}_daily", len(daily))
        terminal["data_gaps"] = _detail_reference(identity, f"{tier_role}_gaps", len(gaps))
        tier["terminal"] = terminal
        summary[tier_key] = tier
    for tier_key, tier_role, result in (
        ("base", "base", report.base),
        ("stress", "stress", report.stress),
    ):
        tier = cast(dict[str, object], summary[tier_key])
        tier["funding_coverage_gaps"] = _detail_reference(
            identity, f"{tier_role}_coverage", len(result.funding_coverage_gaps)
        )
        summary[tier_key] = tier
    _atomic_write_json(out, summary)
    return out, identity


def export_inventory_json(summary_path: Path, output: Path) -> Path:
    """Export the versioned inventory evidence as explicit full JSON without modifying managed originals. Args: readable summary and fresh export destination. Returns: export path. Raises: DataIntegrityError for missing or corrupt detail evidence; ValueError for an occupied destination; OSError for I/O failure."""
    out = Path(output)
    if out.exists():
        raise ValueError(f"export destination must be fresh, got {out}")
    try:
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataIntegrityError(f"unreadable inventory summary: {summary_path}") from exc
    if not isinstance(summary, dict):
        raise DataIntegrityError(f"corrupt inventory summary: {summary_path}")
    evidence = summary.get("evidence")
    evidence_id = summary.get("evidence_id")
    if not isinstance(evidence, dict) or not isinstance(evidence_id, str) or not evidence_id:
        raise DataIntegrityError(f"missing evidence reference: {summary_path}")
    bundle_path = evidence.get("bundle_path")
    if not isinstance(bundle_path, str):
        raise DataIntegrityError(f"missing evidence reference: {summary_path}")
    bundle = Path(bundle_path)
    _verify_bundle(bundle, evidence_id)
    payload = {key: value for key, value in summary.items() if key not in ("schema_version", "evidence_id", "evidence")}
    if "validation" not in payload:
        payload["validation"] = process_validation_payload(None)
        gate = payload.get("gate")
        if isinstance(gate, dict):
            raw_metrics = gate.get("metrics")
            raw_reasons = gate.get("reason_codes")
            gate_metrics: dict[object, object] = dict(raw_metrics) if isinstance(raw_metrics, dict) else {}
            gate_reasons: list[object] = list(raw_reasons) if isinstance(raw_reasons, list) else []
            payload["gate"] = {
                "go": False,
                "reason_codes": sorted(set(gate_reasons) | {_LEGACY_VALIDATION_ABSENT}),
                "metrics": dict(gate_metrics),
            }
            financial = payload.get("financial")
            if isinstance(financial, dict):
                payload["financial"] = {"gate": dict(cast(dict[object, object], payload["gate"]))}
    payload["funding_coverage_gaps"] = _restore_coverage(bundle, "coverage")
    for tier in ("base", "stress"):
        tier_payload = cast(dict[str, object], payload[tier])
        terminal = cast(dict[str, object], tier_payload["terminal"])
        tier_payload["daily_returns"] = _restore_daily_returns(bundle, tier)
        terminal["data_gaps"] = _restore_gap_provenance(bundle, tier)
        tier_payload["terminal"] = terminal
        tier_payload["funding_coverage_gaps"] = _restore_coverage(bundle, f"{tier}_coverage")
        payload[tier] = tier_payload
    _atomic_write_json(out, payload)
    return out


def _restore_daily_returns(bundle: Path, tier: str) -> dict[str, float]:
    try:
        table = pq.read_table(bundle / f"{tier}_daily.parquet", columns=["timestamp", "daily_return"])
    except Exception as exc:
        raise DataIntegrityError(f"missing detail evidence: {tier}_daily.parquet") from exc
    stamps = table.column("timestamp").to_pylist()
    values = table.column("daily_return").to_pylist()
    return {pd.Timestamp(stamp).isoformat(): float(value) for stamp, value in zip(stamps, values, strict=True)}


def _restore_gap_provenance(bundle: Path, tier: str) -> list[dict[str, object]]:
    try:
        table = pq.read_table(bundle / f"{tier}_gaps.parquet")
    except Exception as exc:
        raise DataIntegrityError(f"missing detail evidence: {tier}_gaps.parquet") from exc
    frame = table.to_pandas().sort_values("ordinal", kind="stable")
    return [
        {
            "code": str(row.code),
            "symbol": str(row.symbol),
            "timestamp": pd.Timestamp(row.timestamp).isoformat(),
        }
        for row in frame.itertuples()
    ]


def _restore_coverage(bundle: Path, role: str) -> list[dict[str, object]]:
    try:
        table = pq.read_table(bundle / f"{role}.parquet")
    except Exception as exc:
        raise DataIntegrityError(f"missing detail evidence: {role}.parquet") from exc
    frame = table.to_pandas().sort_values("ordinal", kind="stable")
    return [
        {
            "symbol": str(row.symbol),
            "start": pd.Timestamp(row.start).isoformat(),
            "end": pd.Timestamp(row.end).isoformat(),
            "reason": str(row.reason),
        }
        for row in frame.itertuples()
    ]


__all__ = [
    "export_inventory_json",
    "persist_inventory_evidence",
    "persist_process_inventory_failure",
    "persist_process_inventory_report",
    "process_validation_payload",
]


PROCESS_INVENTORY_CERTIFICATION_LEVEL: str = "process_inventory_3m"


def _require_json_finite(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        raise DataIntegrityError("validation numeric output must be finite")
    return value


def _terminal_position_payload(position: TerminalPositionEvidence) -> dict[str, object]:
    return {
        "symbol": position.symbol,
        "quantity": float(position.quantity),
        "cutoff": position.cutoff.isoformat(),
        "status": position.status,
        "mark": None if position.mark is None else float(position.mark),
        "mark_available_at": None if position.mark_available_at is None else position.mark_available_at.isoformat(),
        "funding_complete": bool(position.funding_complete),
        "reason_codes": list(position.reason_codes),
    }


def process_validation_payload(validation: ProcessValidationResult | None) -> dict[str, object]:
    """Serialize evidence-state acceptance without converting absent proof into success.

    Args:
        validation: Source-owned assessment or absent legacy validation evidence.
    Returns:
        Versioned requirement statuses, reasons and diagnostic/inferential values;
        missing legacy validation is explicitly unverified and deployment-ineligible.
    """
    if validation is None:
        return {
            "schema": _VALIDATION_SCHEMA,
            "status": "legacy_unverified",
            "go": False,
            "accounting_valid": None,
            "historical_acceptance": "unverified",
            "forward_acceptance": "unverified",
            "requirements": [],
            "reason_codes": [_LEGACY_VALIDATION_ABSENT],
            "diagnostics": {},
            "gate": {"go": False, "reason_codes": [_LEGACY_VALIDATION_ABSENT], "metrics": {}},
            "procedure_digest": None,
            "code_digest": None,
            "input_manifest_digest": None,
            "interval_start": None,
            "interval_end": None,
            "evidence_role": None,
            "actual_capital": None,
            "look_ordinal": None,
        }
    requirements = [
        {
            "requirement": check.requirement,
            "status": check.status,
            "procedure_digest": check.procedure_digest,
            "input_manifest_digest": check.input_manifest_digest,
            "code_digest": check.code_digest,
            "interval_start": check.interval_start.isoformat(),
            "interval_end": check.interval_end.isoformat(),
            "artifact_digest": check.artifact_digest,
            "reason_codes": list(check.reason_codes),
        }
        for check in validation.requirements
    ]
    digests = {check.procedure_digest for check in validation.requirements}
    codes = {check.code_digest for check in validation.requirements}
    manifests = {check.input_manifest_digest for check in validation.requirements}
    starts = {check.interval_start.isoformat() for check in validation.requirements}
    ends = {check.interval_end.isoformat() for check in validation.requirements}
    diagnostics = {key: _require_json_finite(value) for key, value in dict(validation.diagnostic_metrics).items()}
    gate_metrics = {key: _require_json_finite(value) for key, value in dict(validation.gate.metrics).items()}
    return {
        "schema": _VALIDATION_SCHEMA,
        "status": "assessed",
        "go": bool(validation.gate.go),
        "accounting_valid": bool(validation.accounting_valid),
        "historical_acceptance": validation.historical_acceptance,
        "forward_acceptance": validation.forward_acceptance,
        "requirements": requirements,
        "reason_codes": list(validation.reason_codes),
        "diagnostics": diagnostics,
        "gate": {
            "go": bool(validation.gate.go),
            "reason_codes": list(validation.gate.reason_codes),
            "metrics": gate_metrics,
        },
        "procedure_digest": next(iter(digests)) if len(digests) == 1 else None,
        "code_digest": next(iter(codes)) if len(codes) == 1 else None,
        "input_manifest_digest": next(iter(manifests)) if len(manifests) == 1 else None,
        "interval_start": next(iter(starts)) if len(starts) == 1 else None,
        "interval_end": next(iter(ends)) if len(ends) == 1 else None,
        "evidence_role": None,
        "actual_capital": None,
        "look_ordinal": None,
    }


def _inventory_terminal_state(result: StrategyExecutionReplayResult) -> dict[str, object]:
    """Serialize engine-native priced-open, settled and unresolved terminal evidence.
    Historical economic gaps remain visible even if a later terminal mark is known;
    no serialization step may invent a closing trade or forgive missing financing.
    """
    gaps = list(result.ledger.data_gaps)
    unpriced = sorted({g.symbol for g in gaps if g.code in _UNPRICED_TERMINAL_CODES})
    unpriced_set = set(unpriced)
    open_inventory: dict[str, float] = {}
    fills = result.simulated_fills
    if len(fills) and "symbol" in fills.columns and "quantity_delta" in fills.columns:
        totals = fills.groupby("symbol")["quantity_delta"].sum()
        for symbol, quantity in totals.items():
            if float(quantity) != 0.0 and str(symbol) not in unpriced_set:
                open_inventory[str(symbol)] = float(quantity)
    positions = [_terminal_position_payload(p) for p in list(result.terminal_positions)]
    return {
        "primary_valid": bool(result.ledger.primary_valid),
        "invalid_reasons": list(result.ledger.invalid_reasons),
        "terminal_certified": bool(replay_ledger_certified(result)),
        "open_inventory": open_inventory,
        "unpriced_terminal_symbols": unpriced,
        "terminal_positions": positions,
        "priced_open_symbols": sorted({str(p["symbol"]) for p in positions if p["status"] == "open_marked"}),
        "settled_symbols": sorted({str(p["symbol"]) for p in positions if p["status"] == "settled"}),
        "unresolved_symbols": sorted({str(p["symbol"]) for p in positions if p["status"] == "unresolved"}),
        "data_gaps": [{"code": g.code, "symbol": g.symbol, "timestamp": g.timestamp.isoformat()} for g in gaps],
    }


def _inventory_result_payload(result: StrategyExecutionReplayResult) -> dict[str, object]:
    """Serializable 3m ledger evidence with engine-native fill accounting."""
    daily = _inventory_daily_returns(result)
    summary = _inventory_ledger_summary(result)
    fills = result.simulated_fills
    certified = bool(replay_ledger_certified(result))
    return {
        "daily_returns": {ts.isoformat(): float(v) for ts, v in daily.items()},
        "cagr": summary["cagr"],
        "max_drawdown": summary["max_drawdown"],
        "annualized_turnover": summary["annualized_turnover"],
        "diagnostic_only": not certified,
        "total_fees": summary["total_fees"],
        "total_funding": summary["total_funding"],
        "total_fills": len(fills),
        "passive_fills": int(result.fill_count),
        "unfilled_count": int(result.unfilled_count),
        "fallback_count": int(result.fallback_count),
        "forced_exit_count": int(result.forced_exit_count),
        "forced_exit_notional": float(result.forced_exit_notional),
        "termination_counts": dict(result.termination_counts),
        "terminal": _inventory_terminal_state(result),
        "funding_coverage_gaps": [_coverage_payload(g) for g in result.funding_coverage_gaps],
    }


def _inventory_summary_payload(report: ProcessInventoryReport) -> dict[str, object]:
    """Assemble the full-JSON inventory payload with engine-native fill accounting."""
    decisions = report.proxy.base.target_weights.index
    validation_payload = process_validation_payload(report.validation)
    gate = report.validation.gate if report.validation is not None else report.gate
    gate_payload: dict[str, object] = {
        "go": gate.go,
        "reason_codes": list(gate.reason_codes),
        "metrics": dict(gate.metrics),
    }
    return {
        "status": "completed",
        "execution_timeframe": "3m",
        "start": report.proxy.start.isoformat(),
        "end": report.proxy.end.isoformat(),
        "certification_level": PROCESS_INVENTORY_CERTIFICATION_LEVEL,
        "source_policy": {"tracking_error_threshold": report.proxy.base.execution_policy.tracking_error_threshold},
        "periods": {
            "decision_start": decisions[0].isoformat() if len(decisions) else None,
            "decision_end": decisions[-1].isoformat() if len(decisions) else None,
            "n_decisions": len(decisions),
        },
        "gate": gate_payload,
        "financial": {"gate": dict(gate_payload)},
        "validation": validation_payload,
        "base": _inventory_result_payload(report.base),
        "stress": _inventory_result_payload(report.stress),
        "proxy": {
            "scope": "hourly_proxy_comparison",
            "certification_level": report.proxy.certification_level,
            "base": _tier_payload(report.proxy.base),
            "stress": _tier_payload(report.proxy.stress),
        },
        "resource_measurements": [dataclasses.asdict(m) for m in report.resource_measurements],
        "memory_stats": dataclasses.asdict(report.memory_stats),
        "funding_coverage_gaps": [_coverage_payload(g) for g in report.funding_coverage_gaps],
    }


def persist_process_inventory_report(
    report: ProcessInventoryReport,
    output: Path,
    *,
    evidence_root: Path | None = None,
    registry_path: Path | None = None,
    run_id: str | None = None,
) -> Path:
    """Persist separately identified 3m execution evidence.

    The summary is compact: large daily-return dictionaries and gap arrays live
    in a content-addressed detail bundle under ``evidence_root`` (defaulting to
    ``output.parent / '.evidence'``), and the summary carries the evidence
    identity and references. Financial scalars match the legacy full payload.

    Args:
        report: Inventory evaluation and explicit comparison evidence.
        output: Fresh JSON summary destination distinct from reserved hourly evidence.
        evidence_root: Owned detail-bundle root or the standalone default beside output.
        registry_path: Paired optional execution registry for managed publication protection.
        run_id: Paired optional registered running execution owning the lease.

    Returns:
        The persisted summary destination.

    Raises:
        ValueError: Output format is unsupported or ownership context is incomplete.
        DataIntegrityError: Destination would overwrite reserved evidence or conflicts.
    """
    root = Path(evidence_root) if evidence_root is not None else Path(output).parent / ".evidence"
    summary_path, _evidence_id = persist_inventory_evidence(
        report,
        output,
        evidence_root=root,
        registry_path=registry_path,
        run_id=run_id,
    )
    return summary_path


def _failure_gap_payload(gap: ExecutionDataGap) -> dict[str, object]:
    return {
        "code": gap.code,
        "symbol": gap.symbol,
        "timestamp": gap.timestamp.isoformat(),
        "decision_time": gap.decision_time.isoformat() if gap.decision_time is not None else None,
        "signal_time": gap.signal_time.isoformat() if gap.signal_time is not None else None,
        "execution_bound": gap.execution_bound,
    }


def persist_process_inventory_failure(
    report: ProcessInventoryFailureReport,
    output: Path,
) -> Path:
    """Persist failure diagnostics without overwriting certified evidence.

    Args:
        report: Typed failed evaluation evidence, never a performance report.
        output: Dedicated JSON failure destination.

    Returns:
        Persisted failure artifact path.

    Raises:
        ValueError: Destination format is unsupported.
        DataIntegrityError: Destination already holds completed success evidence.
        OSError: Persistence fails; the caller must also retain evaluation cause.
    """
    out = Path(output)
    if out.suffix != ".json":
        raise ValueError(f"destination must be a JSON path, got {output}")
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            existing = None
        if isinstance(existing, dict) and existing.get("status") == "completed":
            raise DataIntegrityError(f"destination {out} already holds completed evidence")
    payload: dict[str, object] = {
        "status": "failed",
        "execution_timeframe": "3m",
        "certification_level": PROCESS_INVENTORY_CERTIFICATION_LEVEL,
        "start": report.start.isoformat(),
        "end": report.end.isoformat(),
        "data_root": report.data_root,
        "execution_policy": {"tracking_error_threshold": report.execution_policy.tracking_error_threshold},
        "stage": report.stage,
        "error_code": report.error_code,
        "error_type": report.error_type,
        "error_message": report.error_message,
        "total_decisions": report.total_decisions,
        "validated_decisions": report.validated_decisions,
        "completed_decisions": report.completed_decisions,
        "completed_windows": report.completed_windows,
        "completed_decision_start": (
            report.completed_decision_start.isoformat() if report.completed_decision_start is not None else None
        ),
        "completed_decision_end": (
            report.completed_decision_end.isoformat() if report.completed_decision_end is not None else None
        ),
        "source_gaps": [_failure_gap_payload(g) for g in report.source_gaps],
        "source_gap_excluded_symbols": list(report.source_gap_excluded_symbols),
        "funding_coverage_gaps": [_coverage_payload(g) for g in report.funding_coverage_gaps],
        "resource_measurements": [dataclasses.asdict(m) for m in report.resource_measurements],
        "memory_stats": (dataclasses.asdict(report.memory_stats) if report.memory_stats is not None else None),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=str(out.parent), suffix=".tmp", delete=False, encoding="utf-8"
        ) as tmp:
            tmp_path = tmp.name
            tmp.write(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, out)
    except Exception:
        if tmp_path is not None:
            with suppress(Exception):
                os.unlink(tmp_path)
        raise
    return out
