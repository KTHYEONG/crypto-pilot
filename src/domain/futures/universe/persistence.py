"""Persistence helpers for universe snapshot and manifest hashing."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import (
    FilterReport,
    ManifestRow,
    RejectCode,
    SymbolMeta,
    UniverseSnapshot,
)

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_json(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _symbol_meta_to_dict(meta: SymbolMeta) -> dict[str, Any]:
    payload = asdict(meta)
    payload["capacity_clip_usdt_list"] = list(meta.capacity_clip_usdt_list)
    return payload


def _symbol_meta_from_dict(payload: dict[str, Any]) -> SymbolMeta:
    return SymbolMeta(
        symbol=str(payload["symbol"]),
        role=str(payload["role"]),
        adv_usdt=float(payload["adv_usdt"]),
        execution_cost_bps=float(payload["execution_cost_bps"]),
        funding_carry_8h=float(payload["funding_carry_8h"]),
        beta_vs_market=float(payload["beta_vs_market"]),
        cluster_id=int(payload["cluster_id"]),
        tradeable_rank=int(payload["tradeable_rank"]),
        basis_annualized_mean=(
            float(payload["basis_annualized_mean"])
            if payload["basis_annualized_mean"] is not None
            else None
        ),
        basis_vol=float(payload["basis_vol"]) if payload["basis_vol"] is not None else None,
        capacity_clip_usdt_list=tuple(float(item) for item in payload["capacity_clip_usdt_list"]),
    )


def _filter_report_to_dict(report: FilterReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["stage1_reason"] = (
        report.stage1_reason.value if report.stage1_reason is not None else None
    )
    payload["stage2_reason"] = (
        report.stage2_reason.value if report.stage2_reason is not None else None
    )
    payload["stage3_reason"] = (
        report.stage3_reason.value if report.stage3_reason is not None else None
    )
    payload["stage4_reason"] = (
        report.stage4_reason.value if report.stage4_reason is not None else None
    )
    payload["stage5_reason"] = (
        report.stage5_reason.value if report.stage5_reason is not None else None
    )
    payload["stage6_reason"] = (
        report.stage6_reason.value if report.stage6_reason is not None else None
    )
    payload["audit_trail"] = list(report.audit_trail)
    return payload


def _reject_code_or_none(value: Any) -> RejectCode | None:
    if value is None:
        return None
    return RejectCode(str(value))


def _filter_report_from_dict(payload: dict[str, Any]) -> FilterReport:
    return FilterReport(
        symbol=str(payload["symbol"]),
        stage0_pass=bool(payload["stage0_pass"]),
        stage1_reason=_reject_code_or_none(payload["stage1_reason"]),
        stage1_metrics={str(k): float(v) for k, v in dict(payload["stage1_metrics"]).items()},
        stage2_reason=_reject_code_or_none(payload["stage2_reason"]),
        stage2_metrics={str(k): float(v) for k, v in dict(payload["stage2_metrics"]).items()},
        stage3_reason=_reject_code_or_none(payload["stage3_reason"]),
        stage3_metrics={str(k): float(v) for k, v in dict(payload["stage3_metrics"]).items()},
        stage4_reason=_reject_code_or_none(payload["stage4_reason"]),
        stage4_metrics={str(k): float(v) for k, v in dict(payload["stage4_metrics"]).items()},
        stage5_reason=_reject_code_or_none(payload["stage5_reason"]),
        stage5_metrics={str(k): float(v) for k, v in dict(payload["stage5_metrics"]).items()},
        stage6_reason=_reject_code_or_none(payload["stage6_reason"]),
        stage6_metrics={str(k): float(v) for k, v in dict(payload["stage6_metrics"]).items()},
        final_rank=int(payload["final_rank"]) if payload["final_rank"] is not None else None,
        final_cluster_id=(
            int(payload["final_cluster_id"])
            if payload["final_cluster_id"] is not None
            else None
        ),
        audit_trail=tuple(str(item) for item in payload["audit_trail"]),
    )


def snapshot_to_payload(snapshot: UniverseSnapshot) -> dict[str, Any]:
    """Serialize snapshot into a JSON-safe payload."""
    return {
        "as_of": snapshot.as_of,
        "tf": snapshot.tf,
        "schema_version": snapshot.schema_version,
        "config_hash": snapshot.config_hash,
        "data_manifest_hash": snapshot.data_manifest_hash,
        "basket_ref": list(snapshot.basket_ref),
        "basket_weights": list(snapshot.basket_weights),
        "selected": [_symbol_meta_to_dict(item) for item in snapshot.selected],
        "rejected": {
            symbol: _filter_report_to_dict(report)
            for symbol, report in snapshot.rejected.items()
        },
        "generated_at_utc": snapshot.generated_at_utc,
        "ledger_confidence": snapshot.ledger_confidence,
        "n_stage0": snapshot.n_stage0,
        "n_stage1_pass": snapshot.n_stage1_pass,
        "n_stage2_pass": snapshot.n_stage2_pass,
        "n_stage3_pass": snapshot.n_stage3_pass,
        "n_stage4_pass": snapshot.n_stage4_pass,
        "n_stage5_pass": snapshot.n_stage5_pass,
        "n_stage6_selected": snapshot.n_stage6_selected,
    }


def snapshot_from_payload(payload: dict[str, Any]) -> UniverseSnapshot:
    """Deserialize snapshot payload into contract object."""
    rejected_payload = {str(k): dict(v) for k, v in dict(payload["rejected"]).items()}
    return UniverseSnapshot(
        as_of=str(payload["as_of"]),
        tf=str(payload["tf"]),
        schema_version=int(payload["schema_version"]),
        config_hash=str(payload["config_hash"]),
        data_manifest_hash=str(payload["data_manifest_hash"]),
        basket_ref=tuple(str(item) for item in payload["basket_ref"]),
        basket_weights=tuple(float(item) for item in payload["basket_weights"]),
        selected=tuple(_symbol_meta_from_dict(dict(item)) for item in payload["selected"]),
        rejected={
            symbol: _filter_report_from_dict(report)
            for symbol, report in rejected_payload.items()
        },
        generated_at_utc=str(payload["generated_at_utc"]),
        ledger_confidence=str(payload["ledger_confidence"]),
        n_stage0=int(payload["n_stage0"]),
        n_stage1_pass=int(payload["n_stage1_pass"]),
        n_stage2_pass=int(payload["n_stage2_pass"]),
        n_stage3_pass=int(payload["n_stage3_pass"]),
        n_stage4_pass=int(payload["n_stage4_pass"]),
        n_stage5_pass=int(payload["n_stage5_pass"]),
        n_stage6_selected=int(payload["n_stage6_selected"]),
    )


def save_snapshot_json(snapshot: UniverseSnapshot, path: str | Path) -> Path:
    """Persist UniverseSnapshot as JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = snapshot_to_payload(snapshot)
    target.write_text(_canonical_json(payload), encoding="utf-8")
    logger.info("universe_snapshot_json_saved path=%s", target)
    return target


def load_snapshot_json(path: str | Path) -> UniverseSnapshot:
    """Load UniverseSnapshot from JSON."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    logger.info("universe_snapshot_json_loaded path=%s", source)
    return snapshot_from_payload(dict(payload))


def save_snapshot_parquet(snapshot: UniverseSnapshot, path: str | Path) -> Path:
    """Persist UniverseSnapshot as one-row Parquet with JSON payload."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = snapshot_to_payload(snapshot)
    payload_json = _canonical_json(payload)
    frame = pd.DataFrame(
        [
            {
                "as_of": snapshot.as_of,
                "tf": snapshot.tf,
                "schema_version": snapshot.schema_version,
                "config_hash": snapshot.config_hash,
                "data_manifest_hash": snapshot.data_manifest_hash,
                "generated_at_utc": snapshot.generated_at_utc,
                "saved_at_utc": _utc_now_iso(),
                "payload_json": payload_json,
            }
        ]
    )
    frame.to_parquet(target, index=False)
    logger.info("universe_snapshot_parquet_saved path=%s", target)
    return target


def load_snapshot_parquet(path: str | Path) -> UniverseSnapshot:
    """Load UniverseSnapshot from one-row Parquet payload."""
    source = Path(path)
    frame = pd.read_parquet(source)
    if frame.empty:
        raise ValueError(f"Snapshot parquet is empty: {source}")
    payload_json = str(frame.iloc[0]["payload_json"])
    payload = json.loads(payload_json)
    logger.info("universe_snapshot_parquet_loaded path=%s", source)
    return snapshot_from_payload(dict(payload))


def hash_manifest_rows(rows: list[ManifestRow] | tuple[ManifestRow, ...]) -> str:
    """Compute deterministic SHA256 hash over manifest row set."""
    normalized = [
        {
            "symbol": row.symbol,
            "period": row.period,
            "sha256": row.sha256,
        }
        for row in rows
    ]
    normalized.sort(
        key=lambda item: (
            item["symbol"],
            item["period"],
            item["sha256"],
        )
    )
    return _hash_json({"rows": normalized})
