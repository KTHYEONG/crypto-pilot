"""Offline-first PIT universe build pipeline."""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow.parquet as pq

from src.core.settings import FUTURES_DATA_DIR, LOG_DIR

from .config import UniverseConfig, hash_config
from .data_quality import apply_data_quality_stage
from .filters import (
    apply_cost_model_stage,
    apply_liquidity_stage,
    apply_risk_events_stage,
)
from .models import (
    DEFAULT_LEDGER_PATH,
    FilterReport,
    ManifestRow,
    RejectCode,
    SymbolMeta,
    UniverseRunManifest,
    UniverseSnapshot,
    apply_structure_stage,
    load_ledger_slice,
)
from .selection import apply_selection_stage
from .storage import (
    hash_manifest_rows,
    load_snapshot_json,
    save_snapshot_json,
    save_snapshot_parquet,
)
from .store import (
    DEFAULT_UNIVERSE_STORE_ROOT,
    build_decision_frame,
    compute_universe_run_id,
    load_universe_store_run,
    materialize_snapshot_from_store,
    write_universe_store_run,
)

DEFAULT_SNAPSHOT_ROOT = LOG_DIR / "futures/universe/snapshots"
SCHEMA_VERSION = 1
DEFAULT_MANIFEST_PATH = FUTURES_DATA_DIR / "data_manifest.parquet"
_BASE_REPORT_COLUMNS = {"symbol", "stage", "passed", "reason"}
_log = logging.getLogger(__name__)


def _to_date(as_of: str | date) -> date:
    return as_of if isinstance(as_of, date) else date.fromisoformat(as_of)


def _normalize_cfg(cfg: dict[str, Any] | UniverseConfig | None) -> UniverseConfig:
    if cfg is None:
        return UniverseConfig()
    if isinstance(cfg, UniverseConfig):
        return cfg
    return UniverseConfig(**cfg)


def _snapshot_dir(*, tf: str, as_of: date, root: Path) -> Path:
    return root / f"tf={tf}" / f"as_of={as_of.isoformat()}"


def _snapshot_file_stem(*, tf: str, as_of: date) -> str:
    return f"snapshot_{tf}_{as_of.isoformat()}"


def _snapshot_paths(*, tf: str, as_of: date, root: Path) -> tuple[Path, Path]:
    stem = _snapshot_file_stem(tf=tf, as_of=as_of)
    return root / f"{stem}.parquet", root / f"{stem}.json"


def _filter_report_path(*, tf: str, as_of: date, root: Path) -> Path:
    return root / f"filter_report_{tf}_{as_of.isoformat()}.parquet"


def _existing_ledger_columns(*, ledger_path: Path) -> tuple[str, ...]:
    if not ledger_path.exists():
        return ()
    parquet_file = cast(Any, pq.ParquetFile)(ledger_path)
    return tuple(str(name) for name in parquet_file.schema.names)


def _to_symbol_meta(frame: pd.DataFrame) -> tuple[SymbolMeta, ...]:
    metas: list[SymbolMeta] = []
    if frame.empty:
        return ()
    ranked = frame.copy()
    if "rank" not in ranked.columns:
        ranked["rank"] = pd.Series(range(1, len(ranked) + 1), dtype="int64")
    for _, row in ranked.iterrows():
        role_raw = row.get("role", "regular")
        role = "regular" if pd.isna(role_raw) else str(role_raw)
        if role not in {"anchor", "regular"}:
            role = "regular"
        metas.append(
            SymbolMeta(
                symbol=str(row.get("symbol", "")),
                role=role,
                adv_usdt=float(row.get("adv_usdt_median", 0.0)),
                execution_cost_bps=float(row.get("execution_cost_bps", 0.0)),
                funding_carry_8h=float(row.get("funding_rate_8h", 0.0)),
                beta_vs_market=float(row.get("beta_vs_market", 0.0)),
                cluster_id=int(row.get("cluster_id", -1)),
                tradeable_rank=int(row.get("rank", 0)),
                basis_annualized_mean=(
                    float(row["basis_annualized_mean"])
                    if row.get("basis_annualized_mean") is not None
                    else None
                ),
                basis_vol=float(row["basis_vol"]) if row.get("basis_vol") is not None else None,
                capacity_clip_usdt_list=tuple(
                    float(x) for x in row.get("capacity_clip_usdt_list", ())
                ),
                cluster_size=float(row.get("cluster_size", 1.0)),
                anchor_cluster_member=float(row.get("anchor_cluster_member", 0.0)),
            )
        )
    return tuple(metas)


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
    if normalized in {"not_listed"}:
        return RejectCode.NOT_LISTED
    if stage.startswith("stage1"):
        return RejectCode.INVALID_STRUCTURE
    if normalized in {"insufficient_coverage_60d", "insufficient_is_coverage", "missing_kline"}:
        return RejectCode.LOW_COVERAGE
    if normalized in {"too_many_zero_volume_bars"}:
        return RejectCode.TOO_MANY_ZERO_VOLUME_BARS
    if normalized in {"too_many_gaps", "gap_too_wide"}:
        return RejectCode.EXCESSIVE_GAPS
    if stage.startswith("stage3"):
        return RejectCode.LOW_LIQUIDITY
    if stage.startswith("stage4"):
        return RejectCode.HIGH_EXECUTION_COST
    if normalized in {"funding_anomaly"}:
        return RejectCode.FUNDING_ANOMALY
    if normalized in {"basis_anomaly"}:
        return RejectCode.BASIS_ANOMALY
    if normalized in {"manual_risk_override", "manual_override_fail_closed_missing_knowledge_date"}:
        return RejectCode.RISK_EVENT_OVERRIDE
    if normalized in {"listing_age_too_young"}:
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
                base = replace(base, stage1_reason=reject, stage1_metrics=metrics)
            elif stage.startswith("stage2"):
                base = replace(base, stage2_reason=reject, stage2_metrics=metrics)
            elif stage.startswith("stage3"):
                base = replace(base, stage3_reason=reject, stage3_metrics=metrics)
            elif stage.startswith("stage4"):
                base = replace(base, stage4_reason=reject, stage4_metrics=metrics)
            elif stage.startswith("stage5"):
                base = replace(base, stage5_reason=reject, stage5_metrics=metrics)
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
                base = replace(
                    base,
                    stage6_reason=reject,
                    stage6_metrics=metrics,
                    final_rank=final_rank,
                    final_cluster_id=final_cluster_id,
                )
        base = replace(base, audit_trail=tuple(audit_steps))
        if any(not bool(item.get("passed", False)) for _, item in ordered.iterrows()):
            reports[symbol] = base
    return reports


def _compute_manifest_hash(*, as_of: date, tf: str, manifest_path: Path) -> str:
    if not manifest_path.exists():
        return str(hash_manifest_rows(()))
    manifest = pd.read_parquet(manifest_path)
    if manifest.empty:
        return str(hash_manifest_rows(()))

    scoped = manifest.copy()
    if "tf" in scoped.columns:
        scoped = scoped.loc[scoped["tf"].astype("string") == tf]
    if scoped.empty:
        return str(hash_manifest_rows(()))

    if "knowledge_date" in scoped.columns:
        cutoff_raw = pd.to_datetime(scoped["knowledge_date"], errors="coerce")
    else:
        cutoff_raw = pd.to_datetime(scoped.get("period"), errors="coerce")
    scoped = scoped.loc[cutoff_raw.dt.date <= as_of]
    if scoped.empty:
        return str(hash_manifest_rows(()))

    rows: list[ManifestRow] = []
    for _, row in scoped.iterrows():
        rows.append(
            ManifestRow(
                symbol=str(row.get("symbol", "")),
                period=str(row.get("period", "")),
                source=str(row.get("source", "")),
                sha256=str(row.get("sha256", "")),
                is_final=bool(row.get("is_final", True)),
                updated_at_utc=str(row.get("updated_at_utc", "")),
                tf=str(row.get("tf", "")),
                url=str(row.get("url", "")),
                bytes=int(row.get("bytes", 0) or 0),
                fetched_at_utc=str(row.get("fetched_at_utc", "")),
            )
        )
    return str(hash_manifest_rows(rows))


def _stage_counts_from_report(
    report: pd.DataFrame,
    *,
    selected: pd.DataFrame | None = None,
) -> tuple[int, int, int, int, int, int, int]:
    if report.empty or "stage" not in report.columns or "symbol" not in report.columns:
        n_selected = (
            int(selected["symbol"].nunique())
            if selected is not None and "symbol" in selected.columns
            else 0
        )
        return (0, 0, 0, 0, 0, 0, n_selected)

    stage_col = report["stage"].astype("string")
    symbol_col = report["symbol"].astype("string")
    pass_col = report.get("passed", pd.Series(False, index=report.index)).astype(bool)

    def _count(stage_prefix: str, *, passed_only: bool) -> int:
        mask = stage_col.str.startswith(stage_prefix)
        if passed_only:
            mask = mask & pass_col
        return int(symbol_col.loc[mask].nunique())

    n_stage0 = _count("stage1", passed_only=False)
    n_stage1_pass = _count("stage1", passed_only=True)
    n_stage2_pass = _count("stage2", passed_only=True)
    n_stage3_pass = _count("stage3", passed_only=True)
    n_stage4_pass = _count("stage4", passed_only=True)
    n_stage5_pass = _count("stage5", passed_only=True)
    n_stage6_selected = _count("stage6", passed_only=True)
    if n_stage6_selected == 0 and selected is not None and "symbol" in selected.columns:
        n_stage6_selected = int(selected["symbol"].nunique())
    return (
        n_stage0,
        n_stage1_pass,
        n_stage2_pass,
        n_stage3_pass,
        n_stage4_pass,
        n_stage5_pass,
        n_stage6_selected,
    )


def _stage5_symbols_from_report(report: pd.DataFrame) -> tuple[str, ...]:
    """Extract Stage5-passed symbols from a filter report DataFrame."""
    if report.empty or "stage" not in report.columns or "passed" not in report.columns:
        return ()
    mask = report["stage"].str.startswith("stage5") & report["passed"].astype(bool)
    syms = report.loc[mask, "symbol"].dropna().astype(str).unique().tolist()
    return tuple(sorted(syms))


def _save_snapshot(
    manifest: UniverseRunManifest,
    snapshot: UniverseSnapshot,
    selected: pd.DataFrame,
    decisions: pd.DataFrame,
    report: pd.DataFrame,
    *,
    root: Path,
) -> None:
    as_of_date = date.fromisoformat(snapshot.as_of)
    write_universe_store_run(
        manifest=manifest,
        decisions=decisions,
        report=report,
        root=DEFAULT_UNIVERSE_STORE_ROOT,
    )
    out_dir = _snapshot_dir(tf=snapshot.tf, as_of=as_of_date, root=root)
    flat_parquet, flat_json = _snapshot_paths(tf=snapshot.tf, as_of=as_of_date, root=root)
    flat_report = _filter_report_path(tf=snapshot.tf, as_of=as_of_date, root=root)
    root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(flat_parquet, index=False)
    report.to_parquet(flat_report, index=False)
    save_snapshot_json(snapshot, flat_json)
    save_snapshot_parquet(snapshot, out_dir / "snapshot_meta.parquet")
    # Legacy compatibility artifacts.
    selected.to_parquet(out_dir / "snapshot.parquet", index=False)
    report.to_parquet(out_dir / "filter_report.parquet", index=False)
    save_snapshot_json(snapshot, out_dir / "snapshot.json")


def load_universe_snapshot(
    *,
    as_of: str | date,
    tf: str,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
) -> pd.DataFrame | None:
    """Load selected universe symbols from snapshot, if already materialized."""
    as_of_date = _to_date(as_of)
    flat_parquet, _ = _snapshot_paths(tf=tf, as_of=as_of_date, root=snapshot_root)
    legacy_parquet = _snapshot_dir(tf=tf, as_of=as_of_date, root=snapshot_root) / "snapshot.parquet"
    target = flat_parquet if flat_parquet.exists() else legacy_parquet
    if not target.exists():
        return None
    return pd.read_parquet(target)


def build_universe(
    *,
    as_of: str | date,
    tf: str,
    cfg: dict[str, Any] | UniverseConfig | None = None,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    previous_selection: tuple[str, ...] | None = None,
) -> tuple[UniverseSnapshot, pd.DataFrame, pd.DataFrame]:
    """Build universe with offline-first PIT stage pipeline.

    Returns:
        Tuple of (snapshot metadata, selected symbols frame, filter report frame).

    """
    as_of_date = _to_date(as_of)
    config = _normalize_cfg(cfg)
    manifest_hash = _compute_manifest_hash(
        as_of=as_of_date,
        tf=tf,
        manifest_path=DEFAULT_MANIFEST_PATH,
    )
    columns = (
        "symbol",
        "tf",
        "date",
        "knowledge_date",
        "contract_type",
        "quote_asset",
        "margin_asset",
        "status",
        "contract_multiplier",
        "has_kline",
        "has_funding",
        "is_coverage",
        "n_is_bars",
        "expected_is_bars",
        "n_bar_gaps",
        "last_60d_coverage",
        "n_zero_volume_bars_60d",
        "frozen_bars",
        "has_nan",
        "has_inf",
        "has_timestamp_issues",
        "adv_usdt_median",
        "amihud_30d",
        "vol_30d",
        "screening_clip_usdt",
        "taker_fee_bps",
        "half_spread_bps",
        "impact_bps",
        "tick_cost_bps",
        "tick_size",
        "mark_price",
        "listing_age_days",
        "funding_rate_8h",
        "funding_zscore",
        "risk_event_override",
    )
    ledger_columns = _existing_ledger_columns(ledger_path=ledger_path)
    optional_oi_cols = tuple(
        col
        for col in ("oi_usdt_median", "sum_open_interest_value", "open_interest_usdt")
        if col in ledger_columns
    )
    stage0 = load_ledger_slice(
        as_of=as_of_date,
        tf=tf,
        columns=columns + optional_oi_cols,
        ledger_path=ledger_path,
    )
    if stage0.empty:
        empty = pd.DataFrame(columns=["symbol"])
        report = pd.DataFrame(columns=["symbol", "stage", "passed", "reason"])
        generated_at_utc = datetime.now(tz=UTC).isoformat()
        manifest = UniverseRunManifest(
            as_of=as_of_date.isoformat(),
            tf=tf,
            schema_version=SCHEMA_VERSION,
            run_id=compute_universe_run_id(
                as_of=as_of_date,
                tf=tf,
                config_hash=hash_config(config),
                data_manifest_hash=manifest_hash,
            ),
            config_hash=hash_config(config),
            data_manifest_hash=manifest_hash,
            generated_at_utc=generated_at_utc,
            ledger_confidence=config.ledger_confidence,
            basket_ref=config.stage6.basket_ref,
            basket_weights=config.stage6.basket_weights,
            n_stage0=0,
            n_stage1_pass=0,
            n_stage2_pass=0,
            n_stage3_pass=0,
            n_stage4_pass=0,
            n_stage5_pass=0,
            n_stage6_selected=0,
        )
        decisions = build_decision_frame(
            manifest=manifest,
            stage5_frame=empty,
            stage6_frame=empty,
            report=report,
        )
        snapshot, selected, report = materialize_snapshot_from_store(
            manifest=manifest,
            decisions=decisions,
            report=report,
        )
        _save_snapshot(manifest, snapshot, selected, decisions, report, root=snapshot_root)
        return snapshot, selected, report

    latest = (
        stage0.sort_values(["symbol", "date", "knowledge_date"])
        .groupby("symbol", as_index=False)
        .tail(1)
    )
    s1, r1 = apply_structure_stage(latest)
    s2, r2 = apply_data_quality_stage(s1, config=config.stage2)
    s3, r3 = apply_liquidity_stage(s2, config=config.stage3)
    s4, r4 = apply_cost_model_stage(
        s3,
        config=config.stage4,
        as_of=as_of_date,
    )
    s5, r5 = apply_risk_events_stage(s4, config=config.stage5)
    s6, r6 = apply_selection_stage(
        s5,
        config=config.stage6,
        max_symbols=int(config.stage6.k_in),
        previous_selection=previous_selection,
        k_in=int(config.stage6.k_in),
        k_out=int(config.stage6.k_out),
    )
    report = pd.concat([r1, r2, r3, r4, r5, r6], ignore_index=True)

    generated_at_utc = datetime.now(tz=UTC).isoformat()
    manifest = UniverseRunManifest(
        as_of=as_of_date.isoformat(),
        tf=tf,
        schema_version=SCHEMA_VERSION,
        run_id=compute_universe_run_id(
            as_of=as_of_date,
            tf=tf,
            config_hash=hash_config(config),
            data_manifest_hash=manifest_hash,
        ),
        config_hash=hash_config(config),
        data_manifest_hash=manifest_hash,
        generated_at_utc=generated_at_utc,
        ledger_confidence=config.ledger_confidence,
        basket_ref=config.stage6.basket_ref,
        basket_weights=config.stage6.basket_weights,
        n_stage0=int(latest["symbol"].nunique()),
        n_stage1_pass=int(s1["symbol"].nunique()),
        n_stage2_pass=int(s2["symbol"].nunique()),
        n_stage3_pass=int(s3["symbol"].nunique()),
        n_stage4_pass=int(s4["symbol"].nunique()),
        n_stage5_pass=int(s5["symbol"].nunique()),
        n_stage6_selected=int(s6["symbol"].nunique()),
    )
    decisions = build_decision_frame(
        manifest=manifest,
        stage5_frame=s5,
        stage6_frame=s6,
        report=report,
    )
    snapshot, selected, report = materialize_snapshot_from_store(
        manifest=manifest,
        decisions=decisions,
        report=report,
    )

    # [P1-3] 유니버스 충원율(fill-rate) 경보 게이트
    _n_selected = snapshot.n_stage6_selected
    _k_in = int(config.stage6.k_in)
    _fill_rate = _n_selected / max(1, _k_in)
    if _fill_rate < 0.25:
        _log.error(
            "Universe fill-rate critical: %d/%d (%.0f%%) — stage filter over-rejection suspected",
            _n_selected, _k_in, _fill_rate * 100,
        )
    elif _fill_rate < 0.50:
        _log.warning(
            "Universe fill-rate low: %d/%d (%.0f%%) — check stage filters",
            _n_selected, _k_in, _fill_rate * 100,
        )

    _save_snapshot(manifest, snapshot, selected, decisions, report, root=snapshot_root)
    return snapshot, selected, report


def load_or_build_universe_snapshot(
    *,
    as_of: str | date,
    tf: str,
    cfg: dict[str, Any] | UniverseConfig | None = None,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    previous_selection: tuple[str, ...] | None = None,
) -> tuple[UniverseSnapshot, pd.DataFrame, pd.DataFrame]:
    """Load cached snapshot, otherwise build and persist it."""
    as_of_date = _to_date(as_of)
    config = _normalize_cfg(cfg)
    expected_config_hash = hash_config(config)
    expected_manifest_hash = _compute_manifest_hash(
        as_of=as_of_date,
        tf=tf,
        manifest_path=DEFAULT_MANIFEST_PATH,
    )
    store_run = load_universe_store_run(
        as_of=as_of_date,
        tf=tf,
        config_hash=expected_config_hash,
        data_manifest_hash=expected_manifest_hash,
        root=DEFAULT_UNIVERSE_STORE_ROOT,
    )
    if store_run is not None:
        manifest, decisions, report = store_run
        return materialize_snapshot_from_store(
            manifest=manifest,
            decisions=decisions,
            report=report,
        )

    loaded_snapshot: UniverseSnapshot | None = None
    _flat_parquet, flat_json = _snapshot_paths(tf=tf, as_of=as_of_date, root=snapshot_root)
    legacy_json = _snapshot_dir(tf=tf, as_of=as_of_date, root=snapshot_root) / "snapshot.json"
    snapshot_meta_path = flat_json if flat_json.exists() else legacy_json
    if snapshot_meta_path.exists():
        try:
            loaded_snapshot = load_snapshot_json(snapshot_meta_path)
        except Exception as exc:
            _log.warning("Cached universe snapshot metadata load failed: %s", exc)

    if loaded_snapshot is not None:
        cache_mismatches: list[str] = []
        if loaded_snapshot.as_of != as_of_date.isoformat():
            cache_mismatches.append("as_of")
        if loaded_snapshot.tf != tf:
            cache_mismatches.append("tf")
        if loaded_snapshot.schema_version != SCHEMA_VERSION:
            cache_mismatches.append("schema")
        if loaded_snapshot.config_hash != expected_config_hash:
            cache_mismatches.append("config_hash")
        if loaded_snapshot.data_manifest_hash != expected_manifest_hash:
            cache_mismatches.append("manifest_hash")
        if cache_mismatches:
            _log.info("Cached universe snapshot stale (%s); rebuilding", ",".join(cache_mismatches))
        else:
            loaded = load_universe_snapshot(as_of=as_of, tf=tf, snapshot_root=snapshot_root)
            if loaded is not None:
                if not loaded.empty and "adv_usdt_median" not in loaded.columns:
                    # Join from ledger to restore metrics for backwards compatibility
                    # Only query physical columns in load_ledger_slice
                    physical_cols = (
                        "adv_usdt_median",
                        "funding_rate_8h",
                        "taker_fee_bps",
                        "half_spread_bps",
                        "impact_bps",
                        "tick_cost_bps",
                    )
                    try:
                        ledger_slice = load_ledger_slice(
                            as_of=as_of_date,
                            tf=tf,
                            columns=physical_cols,
                            symbols=tuple(loaded["symbol"].tolist()),
                            ledger_path=ledger_path,
                        )
                        if not ledger_slice.empty:
                            latest_ledger = (
                                ledger_slice.sort_values(["symbol", "date", "knowledge_date"])
                                .groupby("symbol", as_index=False)
                                .tail(1)
                            ).copy()

                            # Compute execution_cost_bps from physical ledger metrics
                            latest_ledger["execution_cost_bps"] = (
                                2.0 * latest_ledger["taker_fee_bps"].fillna(0.0)
                                + 2.0 * latest_ledger["half_spread_bps"].fillna(0.0)
                                + latest_ledger["impact_bps"].fillna(0.0)
                                + latest_ledger["tick_cost_bps"].fillna(0.0)
                            )

                            missing_cols = [
                                c
                                for c in (*physical_cols, "execution_cost_bps")
                                if c in latest_ledger.columns and c not in loaded.columns
                            ]
                            if missing_cols:
                                loaded = loaded.merge(
                                    latest_ledger[["symbol", *missing_cols]],
                                    on="symbol",
                                    how="left",
                                )
                    except Exception as exc:
                        _log.warning("Backwards compatible ledger join failed: %s", exc)

                    # Safely backfill all virtual/derived columns to avoid KeyError or 0.0 anomalies
                    fallback_defaults = {
                        "adv_usdt_median": 0.0,
                        "execution_cost_bps": 0.0,
                        "funding_rate_8h": 0.0,
                        "beta_vs_market": 0.0,
                        "cluster_id": -1,
                        "cluster_size": 1.0,
                        "anchor_cluster_member": 0.0,
                        "basis_annualized_mean": None,
                        "basis_vol": None,
                    }
                    for col, default_val in fallback_defaults.items():
                        if col not in loaded.columns:
                            if default_val is None:
                                loaded[col] = None
                            else:
                                loaded[col] = default_val

                    if "capacity_clip_usdt_list" not in loaded.columns:
                        loaded["capacity_clip_usdt_list"] = [() for _ in range(len(loaded))]

                report_path = _filter_report_path(tf=tf, as_of=as_of_date, root=snapshot_root)
                if not report_path.exists():
                    report_path = (
                        _snapshot_dir(tf=tf, as_of=as_of_date, root=snapshot_root)
                        / "filter_report.parquet"
                    )
                report = pd.read_parquet(report_path) if report_path.exists() else pd.DataFrame()
                selected_symbols = tuple(
                    str(symbol).strip()
                    for symbol in loaded["symbol"].astype(str).tolist()
                    if str(symbol).strip()
                )
                selected_set = set(selected_symbols)
                inference_panel = tuple(loaded_snapshot.inference_panel)
                stage5_research_panel = (
                    tuple(loaded_snapshot.stage5_research_panel)
                    or _stage5_symbols_from_report(report)
                )
                training_panel = tuple(loaded_snapshot.training_panel)
                live_inference_panel = tuple(loaded_snapshot.live_inference_panel)
                has_required_metadata = {"cluster_size", "anchor_cluster_member"}.issubset(
                    set(loaded.columns)
                )
                if not training_panel or set(training_panel) != selected_set:
                    training_panel = selected_symbols
                if not live_inference_panel or set(live_inference_panel) != selected_set:
                    live_inference_panel = selected_symbols
                if (
                    has_required_metadata
                    and loaded_snapshot.schema_version == SCHEMA_VERSION
                    and loaded_snapshot.config_hash == expected_config_hash
                    and loaded_snapshot.data_manifest_hash == expected_manifest_hash
                    and (not inference_panel or set(inference_panel) == selected_set)
                ):
                    manifest = UniverseRunManifest(
                        as_of=loaded_snapshot.as_of,
                        tf=loaded_snapshot.tf,
                        schema_version=loaded_snapshot.schema_version,
                        run_id=compute_universe_run_id(
                            as_of=loaded_snapshot.as_of,
                            tf=loaded_snapshot.tf,
                            config_hash=loaded_snapshot.config_hash,
                            data_manifest_hash=loaded_snapshot.data_manifest_hash,
                        ),
                        config_hash=loaded_snapshot.config_hash,
                        data_manifest_hash=loaded_snapshot.data_manifest_hash,
                        generated_at_utc=loaded_snapshot.generated_at_utc,
                        ledger_confidence=loaded_snapshot.ledger_confidence,
                        basket_ref=loaded_snapshot.basket_ref,
                        basket_weights=loaded_snapshot.basket_weights,
                        n_stage0=loaded_snapshot.n_stage0,
                        n_stage1_pass=loaded_snapshot.n_stage1_pass,
                        n_stage2_pass=loaded_snapshot.n_stage2_pass,
                        n_stage3_pass=loaded_snapshot.n_stage3_pass,
                        n_stage4_pass=loaded_snapshot.n_stage4_pass,
                        n_stage5_pass=loaded_snapshot.n_stage5_pass,
                        n_stage6_selected=loaded_snapshot.n_stage6_selected,
                    )
                    stage5_frame = loaded.loc[
                        loaded["symbol"].astype(str).isin(stage5_research_panel)
                    ].copy()
                    decisions = build_decision_frame(
                        manifest=manifest,
                        stage5_frame=stage5_frame,
                        stage6_frame=loaded,
                        report=report,
                    )
                    snapshot, selected, normalized_report = materialize_snapshot_from_store(
                        manifest=manifest,
                        decisions=decisions,
                        report=report,
                    )
                    snapshot = replace(
                        snapshot,
                        inference_panel=inference_panel,
                        historical_trading_panel=tuple(loaded_snapshot.historical_trading_panel),
                        inference_panel_quarter_membership=dict(
                            loaded_snapshot.inference_panel_quarter_membership
                        ),
                    )
                    _save_snapshot(
                        manifest,
                        snapshot,
                        selected,
                        decisions,
                        normalized_report,
                        root=snapshot_root,
                    )
                    return snapshot, selected, normalized_report
    return build_universe(
        as_of=as_of,
        tf=tf,
        cfg=cfg,
        ledger_path=ledger_path,
        snapshot_root=snapshot_root,
        previous_selection=previous_selection,
    )
