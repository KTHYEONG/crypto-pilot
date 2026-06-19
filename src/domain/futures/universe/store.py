"""Normalized persistence store for futures universe runs."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.settings import LOG_DIR

from .models import (
    FilterReport,
    RejectCode,
    SymbolMeta,
    UniverseRunManifest,
    UniverseSnapshot,
)

DEFAULT_UNIVERSE_STORE_ROOT = LOG_DIR / "futures/universe/store/v1"
UNIVERSE_DECISION_COLUMNS = (
    "as_of",
    "tf",
    "run_id",
    "config_hash",
    "data_manifest_hash",
    "symbol",
    "stage5_pass",
    "stage6_selected",
    "stage",
    "selection_reason",
    "role",
    "rank",
    "tradeable_score",
    "execution_pool_score",
    "adv_usdt_median",
    "execution_cost_bps",
    "funding_rate_8h",
    "beta_vs_market",
    "cluster_id",
    "cluster_size",
    "anchor_cluster_member",
    "basis_annualized_mean",
    "basis_vol",
    "capacity_clip_usdt_list",
    "reject_code",
    "final_rank",
    "generated_at_utc",
)
_BASE_REPORT_COLUMNS = {"symbol", "stage", "passed", "reason"}


def _to_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def _run_dir(*, as_of: str | date, tf: str, run_id: str, root: Path) -> Path:
    return root / "runs" / f"tf={tf}" / f"as_of={_to_date(as_of).isoformat()}" / f"run_id={run_id}"


def _manifest_to_frame(manifest: UniverseRunManifest) -> pd.DataFrame:
    payload = asdict(manifest)
    payload["basket_ref"] = list(manifest.basket_ref)
    payload["basket_weights"] = list(manifest.basket_weights)
    return pd.DataFrame([payload])


def _manifest_from_frame(frame: pd.DataFrame) -> UniverseRunManifest:
    row = frame.iloc[0].to_dict()
    return UniverseRunManifest(
        as_of=str(row["as_of"]),
        tf=str(row["tf"]),
        schema_version=int(row["schema_version"]),
        run_id=str(row["run_id"]),
        config_hash=str(row["config_hash"]),
        data_manifest_hash=str(row["data_manifest_hash"]),
        generated_at_utc=str(row["generated_at_utc"]),
        ledger_confidence=str(row["ledger_confidence"]),
        basket_ref=tuple(str(item) for item in row.get("basket_ref", [])),
        basket_weights=tuple(float(item) for item in row.get("basket_weights", [])),
        n_stage0=int(row["n_stage0"]),
        n_stage1_pass=int(row["n_stage1_pass"]),
        n_stage2_pass=int(row["n_stage2_pass"]),
        n_stage3_pass=int(row["n_stage3_pass"]),
        n_stage4_pass=int(row["n_stage4_pass"]),
        n_stage5_pass=int(row["n_stage5_pass"]),
        n_stage6_selected=int(row["n_stage6_selected"]),
    )


def _empty_filter_report(symbol: str) -> FilterReport:
    return FilterReport(
        symbol=symbol,
        stage0_pass=True,
        stage1_reason=None,
        stage1_metrics={},
        stage2_reason=None,
        stage2_metrics={},
        stage3_reason=None,
        stage3_metrics={},
        stage4_reason=None,
        stage4_metrics={},
        stage5_reason=None,
        stage5_metrics={},
        stage6_reason=None,
        stage6_metrics={},
        final_rank=None,
        final_cluster_id=None,
        audit_trail=(),
    )


def _reason_to_reject_code(*, stage: str, reason: str) -> RejectCode | None:
    normalized = reason.strip().lower()
    if normalized in {"", "pass", "selected", "anchor_selected"}:
        return None
    if normalized == "not_trading":
        return RejectCode.NOT_TRADING
    if normalized == "not_listed":
        return RejectCode.NOT_LISTED
    if stage.startswith("stage1"):
        return RejectCode.INVALID_STRUCTURE
    if normalized in {"insufficient_coverage_60d", "insufficient_is_coverage", "missing_kline"}:
        return RejectCode.LOW_COVERAGE
    if normalized == "too_many_zero_volume_bars":
        return RejectCode.TOO_MANY_ZERO_VOLUME_BARS
    if normalized in {"too_many_gaps", "gap_too_wide"}:
        return RejectCode.EXCESSIVE_GAPS
    if stage.startswith("stage3"):
        return RejectCode.LOW_LIQUIDITY
    if stage.startswith("stage4"):
        return RejectCode.HIGH_EXECUTION_COST
    if normalized == "funding_anomaly":
        return RejectCode.FUNDING_ANOMALY
    if normalized == "basis_anomaly":
        return RejectCode.BASIS_ANOMALY
    if normalized in {"manual_risk_override", "manual_override_fail_closed_missing_knowledge_date"}:
        return RejectCode.RISK_EVENT_OVERRIDE
    if normalized == "listing_age_too_young":
        return RejectCode.LISTING_TOO_YOUNG
    if normalized in {"vol_too_low", "vol_too_high"}:
        return RejectCode.VOL_BAND_VIOLATION
    return RejectCode.RANKED_OUT


def _stage_metrics_from_row(row: pd.Series) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for column, value in row.items():
        if column in _BASE_REPORT_COLUMNS or pd.isna(value):
            continue
        if isinstance(value, bool):
            metrics[str(column)] = 1.0 if value else 0.0
            continue
        if isinstance(value, int | float):
            metrics[str(column)] = float(value)
            continue
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric):
            metrics[str(column)] = float(numeric)
    return metrics


def _to_rejected(frame: pd.DataFrame) -> dict[str, FilterReport]:
    if frame.empty or "symbol" not in frame.columns:
        return {}
    reports: dict[str, FilterReport] = {}
    grouped = frame.groupby(frame["symbol"].astype("string"), sort=False)
    for symbol_key, symbol_rows in grouped:
        symbol = str(symbol_key)
        if not symbol:
            continue
        base = _empty_filter_report(symbol)
        ordered = symbol_rows.reset_index(drop=True)
        audit_steps: list[str] = []
        for _, row in ordered.iterrows():
            stage = str(row.get("stage", ""))
            if not stage.startswith("stage"):
                continue
            passed = bool(row.get("passed", False))
            reason = str(row.get("reason", "pass"))
            reject = None if passed else _reason_to_reject_code(stage=stage, reason=reason)
            metrics = _stage_metrics_from_row(row)
            metric_fragment = ",".join(f"{k}={v:.6g}" for k, v in sorted(metrics.items()))
            audit_steps.append(
                f"{stage}:{'PASS' if passed else 'FAIL'}:{reason}"
                + (f":{metric_fragment}" if metric_fragment else "")
            )
            if stage.startswith("stage1"):
                base = base.__class__(**{**asdict(base), "stage1_reason": reject, "stage1_metrics": metrics})
            elif stage.startswith("stage2"):
                base = base.__class__(**{**asdict(base), "stage2_reason": reject, "stage2_metrics": metrics})
            elif stage.startswith("stage3"):
                base = base.__class__(**{**asdict(base), "stage3_reason": reject, "stage3_metrics": metrics})
            elif stage.startswith("stage4"):
                base = base.__class__(**{**asdict(base), "stage4_reason": reject, "stage4_metrics": metrics})
            elif stage.startswith("stage5"):
                base = base.__class__(**{**asdict(base), "stage5_reason": reject, "stage5_metrics": metrics})
            elif stage.startswith("stage6"):
                final_rank_raw = row.get("rank")
                final_rank = (
                    int(final_rank_raw)
                    if final_rank_raw is not None and pd.notna(final_rank_raw)
                    else base.final_rank
                )
                cluster_raw = row.get("cluster_id")
                final_cluster_id = (
                    int(cluster_raw)
                    if cluster_raw is not None and pd.notna(cluster_raw)
                    else base.final_cluster_id
                )
                base = base.__class__(
                    **{
                        **asdict(base),
                        "stage6_reason": reject,
                        "stage6_metrics": metrics,
                        "final_rank": final_rank,
                        "final_cluster_id": final_cluster_id,
                    }
                )
        base = base.__class__(**{**asdict(base), "audit_trail": tuple(audit_steps)})
        if any(not bool(item.get("passed", False)) for _, item in ordered.iterrows()):
            reports[symbol] = base
    return reports


def compute_universe_run_id(
    *,
    as_of: str | date,
    tf: str,
    config_hash: str,
    data_manifest_hash: str,
) -> str:
    payload = f"{tf}|{_to_date(as_of).isoformat()}|{config_hash}|{data_manifest_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_decision_frame(
    *,
    manifest: UniverseRunManifest,
    stage5_frame: pd.DataFrame,
    stage6_frame: pd.DataFrame,
    report: pd.DataFrame,
) -> pd.DataFrame:
    if report.empty and stage5_frame.empty and stage6_frame.empty:
        return pd.DataFrame(columns=UNIVERSE_DECISION_COLUMNS)

    stage5_symbols = set(stage5_frame.get("symbol", pd.Series(dtype="string")).astype(str).tolist())
    stage6_symbols = set(stage6_frame.get("symbol", pd.Series(dtype="string")).astype(str).tolist())
    report_latest = (
        report.sort_values(["symbol", "stage"])
        .groupby("symbol", as_index=False)
        .tail(1)
        .set_index("symbol", drop=False)
        if not report.empty and "symbol" in report.columns
        else pd.DataFrame()
    )
    report_by_symbol = report_latest.to_dict(orient="index") if not report_latest.empty else {}
    stage5_indexed = stage5_frame.set_index("symbol", drop=False) if not stage5_frame.empty else pd.DataFrame()
    stage6_indexed = stage6_frame.set_index("symbol", drop=False) if not stage6_frame.empty else pd.DataFrame()

    rows: list[dict[str, Any]] = []
    all_symbols = sorted(set(report_by_symbol) | stage5_symbols | stage6_symbols)
    for symbol in all_symbols:
        source_row: pd.Series | None = None
        if not stage6_indexed.empty and symbol in stage6_indexed.index:
            source_row = stage6_indexed.loc[symbol]
        elif not stage5_indexed.empty and symbol in stage5_indexed.index:
            source_row = stage5_indexed.loc[symbol]
        report_row = report_by_symbol.get(symbol, {})
        rank_raw = None
        if source_row is not None and "rank" in source_row.index:
            rank_raw = source_row.get("rank")
        elif "rank" in report_row:
            rank_raw = report_row.get("rank")
        final_rank = int(rank_raw) if rank_raw is not None and pd.notna(rank_raw) else None
        selection_reason = str(report_row.get("reason", "selected" if symbol in stage6_symbols else "rejected"))
        stage_value = str(report_row.get("stage", "stage6_selection" if symbol in stage6_symbols else ""))
        reject_code = None
        if stage_value and selection_reason:
            reject = _reason_to_reject_code(stage=stage_value, reason=selection_reason)
            reject_code = reject.value if reject is not None else None
        rows.append(
            {
                "as_of": manifest.as_of,
                "tf": manifest.tf,
                "run_id": manifest.run_id,
                "config_hash": manifest.config_hash,
                "data_manifest_hash": manifest.data_manifest_hash,
                "symbol": symbol,
                "stage5_pass": symbol in stage5_symbols,
                "stage6_selected": symbol in stage6_symbols,
                "stage": stage_value,
                "selection_reason": selection_reason,
                "role": (
                    str(source_row.get("role", "regular"))
                    if source_row is not None
                    else "regular"
                ),
                "rank": final_rank,
                "tradeable_score": (
                    float(source_row.get("tradeable_score", 0.0))
                    if source_row is not None
                    else 0.0
                ),
                "execution_pool_score": (
                    float(source_row.get("execution_pool_score", 0.0))
                    if source_row is not None
                    else 0.0
                ),
                "adv_usdt_median": (
                    float(source_row.get("adv_usdt_median", 0.0))
                    if source_row is not None
                    else 0.0
                ),
                "execution_cost_bps": (
                    float(source_row.get("execution_cost_bps", 0.0))
                    if source_row is not None
                    else 0.0
                ),
                "funding_rate_8h": (
                    float(source_row.get("funding_rate_8h", 0.0))
                    if source_row is not None
                    else 0.0
                ),
                "beta_vs_market": (
                    float(source_row.get("beta_vs_market", 0.0))
                    if source_row is not None
                    else 0.0
                ),
                "cluster_id": (
                    int(source_row.get("cluster_id", -1))
                    if source_row is not None
                    else -1
                ),
                "cluster_size": (
                    float(source_row.get("cluster_size", 1.0))
                    if source_row is not None
                    else 1.0
                ),
                "anchor_cluster_member": (
                    float(source_row.get("anchor_cluster_member", 0.0))
                    if source_row is not None
                    else 0.0
                ),
                "basis_annualized_mean": (
                    None
                    if source_row is None or pd.isna(source_row.get("basis_annualized_mean"))
                    else float(source_row.get("basis_annualized_mean"))
                ),
                "basis_vol": (
                    None
                    if source_row is None or pd.isna(source_row.get("basis_vol"))
                    else float(source_row.get("basis_vol"))
                ),
                "capacity_clip_usdt_list": (
                    tuple(float(item) for item in source_row.get("capacity_clip_usdt_list", ()))
                    if source_row is not None
                    else ()
                ),
                "reject_code": reject_code,
                "final_rank": final_rank,
                "generated_at_utc": manifest.generated_at_utc,
            }
        )
    decisions = pd.DataFrame(rows)
    return decisions.loc[:, list(UNIVERSE_DECISION_COLUMNS)]


def write_universe_store_run(
    *,
    manifest: UniverseRunManifest,
    decisions: pd.DataFrame,
    report: pd.DataFrame,
    root: Path = DEFAULT_UNIVERSE_STORE_ROOT,
) -> Path:
    # PIT 경로는 Stage6 decisions 스키마 없음 — 빈 DataFrame 허용
    if decisions.empty:
        run_dir = _run_dir(as_of=manifest.as_of, tf=manifest.tf, run_id=manifest.run_id, root=root)
        run_dir.mkdir(parents=True, exist_ok=True)
        _manifest_to_frame(manifest).to_parquet(run_dir / "manifest.parquet", index=False)
        return run_dir
    missing_columns = [column for column in UNIVERSE_DECISION_COLUMNS if column not in decisions.columns]
    if missing_columns:
        raise ValueError(f"universe decisions missing columns: {missing_columns}")
    run_dir = _run_dir(as_of=manifest.as_of, tf=manifest.tf, run_id=manifest.run_id, root=root)
    run_dir.mkdir(parents=True, exist_ok=True)
    _manifest_to_frame(manifest).to_parquet(run_dir / "manifest.parquet", index=False)
    decisions.loc[:, list(UNIVERSE_DECISION_COLUMNS)].to_parquet(run_dir / "decisions.parquet", index=False)
    report.to_parquet(run_dir / "filter_report.parquet", index=False)
    return run_dir


def load_universe_store_run(
    *,
    as_of: str | date,
    tf: str,
    config_hash: str,
    data_manifest_hash: str,
    root: Path = DEFAULT_UNIVERSE_STORE_ROOT,
) -> tuple[UniverseRunManifest, pd.DataFrame, pd.DataFrame] | None:
    run_id = compute_universe_run_id(
        as_of=as_of,
        tf=tf,
        config_hash=config_hash,
        data_manifest_hash=data_manifest_hash,
    )
    run_dir = _run_dir(as_of=as_of, tf=tf, run_id=run_id, root=root)
    manifest_path = run_dir / "manifest.parquet"
    decisions_path = run_dir / "decisions.parquet"
    report_path = run_dir / "filter_report.parquet"
    if not (manifest_path.exists() and decisions_path.exists() and report_path.exists()):
        return None
    manifest = _manifest_from_frame(pd.read_parquet(manifest_path))
    decisions = pd.read_parquet(decisions_path)
    report = pd.read_parquet(report_path)
    if manifest.config_hash != config_hash or manifest.data_manifest_hash != data_manifest_hash:
        return None
    return manifest, decisions, report


def materialize_snapshot_from_store(
    *,
    manifest: UniverseRunManifest,
    decisions: pd.DataFrame,
    report: pd.DataFrame,
) -> tuple[UniverseSnapshot, pd.DataFrame, pd.DataFrame]:
    selected = decisions.loc[decisions["stage6_selected"].astype(bool)].copy()
    selected["_anchor_priority"] = (
        selected["role"].astype(str).str.lower().eq("anchor").astype(int) * -1
    )
    selected = selected.sort_values(
        ["_anchor_priority", "rank", "symbol"],
        na_position="last",
    ).reset_index(drop=True)
    selected_meta = tuple(
        SymbolMeta(
            symbol=str(row.symbol),
            role=str(row.role),
            adv_usdt=float(row.adv_usdt_median),
            execution_cost_bps=float(row.execution_cost_bps),
            funding_carry_8h=float(row.funding_rate_8h),
            beta_vs_market=float(row.beta_vs_market),
            cluster_id=int(row.cluster_id),
            tradeable_rank=int(row.rank) if pd.notna(row.rank) else 0,
            basis_annualized_mean=(
                None if pd.isna(row.basis_annualized_mean) else float(row.basis_annualized_mean)
            ),
            basis_vol=None if pd.isna(row.basis_vol) else float(row.basis_vol),
            capacity_clip_usdt_list=tuple(float(item) for item in row.capacity_clip_usdt_list),
            cluster_size=float(row.cluster_size),
            anchor_cluster_member=float(row.anchor_cluster_member),
        )
        for row in selected.itertuples(index=False)
    )
    snapshot = UniverseSnapshot(
        as_of=manifest.as_of,
        tf=manifest.tf,
        schema_version=manifest.schema_version,
        config_hash=manifest.config_hash,
        data_manifest_hash=manifest.data_manifest_hash,
        basket_ref=manifest.basket_ref,
        basket_weights=manifest.basket_weights,
        selected=selected_meta,
        rejected=_to_rejected(report),
        generated_at_utc=manifest.generated_at_utc,
        ledger_confidence=manifest.ledger_confidence,
        n_stage0=manifest.n_stage0,
        n_stage1_pass=manifest.n_stage1_pass,
        n_stage2_pass=manifest.n_stage2_pass,
        n_stage3_pass=manifest.n_stage3_pass,
        n_stage4_pass=manifest.n_stage4_pass,
        n_stage5_pass=manifest.n_stage5_pass,
        n_stage6_selected=manifest.n_stage6_selected,
    )
    selected_frame = selected.loc[
        :,
        [
            "symbol",
            "tradeable_score",
            "execution_pool_score",
            "rank",
            "role",
            "adv_usdt_median",
            "execution_cost_bps",
            "funding_rate_8h",
            "beta_vs_market",
            "cluster_id",
            "cluster_size",
            "anchor_cluster_member",
            "basis_annualized_mean",
            "basis_vol",
            "capacity_clip_usdt_list",
        ],
    ].copy()
    return snapshot, selected_frame, report.copy()
