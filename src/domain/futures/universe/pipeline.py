"""Offline-first PIT universe build pipeline."""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.settings import FUTURES_DATA_DIR, LOG_DIR

from .config import UniverseConfig, hash_config
from .contracts import DataConfidence, ExecutionRules
from .eligibility import (
    ExecutionEligibilityConfig,
    RuleFallbackPolicy,
    build_universe_state_cube,
    evaluate_execution_eligibility,
    resolve_execution_rules,
)
from .models import (
    DEFAULT_LEDGER_PATH,
    FilterReport,
    ManifestRow,
    RejectCode,
    SymbolMeta,
    UniverseRunManifest,
    UniverseSnapshot,
    load_ledger_slice,
)
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
    import sqlite3
    conn = sqlite3.connect(str(ledger_path))
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(ledger)")
        cols = [row[1] for row in cursor.fetchall()]
        return tuple(cols)
    except Exception:
        return ()
    finally:
        conn.close()


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
                vol_30d=float(row.get("vol_30d", 0.0)),
                friction_score=float(row.get("friction_score", 0.0)),
                alpha_capacity_score=float(row.get("alpha_capacity_score", 0.0)),
                diversification_score=float(row.get("diversification_score", 0.0)),
                tradeable_score=float(row.get("tradeable_score", 0.0)),
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


# ---------------------------------------------------------------------------
# PIT path helpers (Phase 1)
# ---------------------------------------------------------------------------

def _instrument_df_from_ledger(latest_df: pd.DataFrame) -> pd.DataFrame:
    """Convert latest ledger rows to InstrumentRecord-schema DataFrame.

    Args:
        latest_df: One row per symbol from the ledger (latest knowledge_date).

    Returns:
        DataFrame with instrument registry columns.
    """
    rows: list[dict[str, object]] = []
    for _, row in latest_df.iterrows():
        sym = str(row["symbol"])
        iid = f"binance_usdt_perpetual:{sym}"
        kd = row.get("knowledge_date") or row.get("date")
        if hasattr(kd, "to_pydatetime"):
            kd = kd.to_pydatetime()
        elif isinstance(kd, str):
            kd = datetime.fromisoformat(kd).replace(tzinfo=UTC)
        elif isinstance(kd, date) and not isinstance(kd, datetime):
            kd = datetime(kd.year, kd.month, kd.day, tzinfo=UTC)
        if hasattr(kd, "tzinfo") and kd.tzinfo is None:
            kd = kd.replace(tzinfo=UTC)
        rows.append(
            {
                "instrument_id": iid,
                "symbol": sym,
                "pair": sym.replace("USDT", ""),
                "quote_asset": str(row.get("quote_asset", "USDT")),
                "margin_asset": str(row.get("margin_asset", "USDT")),
                "contract_type": str(row.get("contract_type", "PERPETUAL")),
                "onboard_at": kd,
                "status": str(row.get("status", "TRADING")),
                "state_valid_from": kd,
                "available_at": kd,
                "confidence": DataConfidence.RECONSTRUCTED.value,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "instrument_id", "symbol", "pair", "quote_asset", "margin_asset",
                "contract_type", "onboard_at", "status", "state_valid_from",
                "available_at", "confidence",
            ]
        )
    return pd.DataFrame(rows)


def _observation_df_from_ledger(
    latest_df: pd.DataFrame, *, decision_at: datetime
) -> pd.DataFrame:
    """Convert ledger metrics to MarketObservation-schema DataFrame.

    Maps: adv_usdt_median→adv30_usdt, amihud_30d→amihud30, vol_30d→vol30,
    mark_price→last_price. available_at is set to decision_at (bootstrapped).

    Args:
        latest_df: One row per symbol from the ledger.
        decision_at: Point-in-time decision timestamp (UTC).

    Returns:
        DataFrame with market observation columns.
    """
    metric_map: dict[str, str] = {
        "adv_usdt_median": "adv30_usdt",
        "amihud_30d": "amihud30",
        "vol_30d": "vol30",
        "mark_price": "last_price",
    }
    rows: list[dict[str, object]] = []
    for _, row in latest_df.iterrows():
        sym = str(row["symbol"])
        iid = f"binance_usdt_perpetual:{sym}"
        kd = row.get("knowledge_date") or row.get("date")
        if hasattr(kd, "to_pydatetime"):
            observed_at: datetime = kd.to_pydatetime()
        elif isinstance(kd, str):
            observed_at = datetime.fromisoformat(kd)
        else:
            observed_at = decision_at
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=UTC)
        for ledger_col, metric_name in metric_map.items():
            val = row.get(ledger_col)
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if pd.isna(fval):
                continue
            rows.append(
                {
                    "instrument_id": iid,
                    "metric": metric_name,
                    "observed_at": observed_at,
                    "available_at": min(observed_at, decision_at),
                    "value": fval,
                    "source": "ledger_bootstrap",
                    "confidence": DataConfidence.RECONSTRUCTED.value,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "instrument_id", "metric", "observed_at", "available_at",
                "value", "source", "confidence",
            ]
        )
    return pd.DataFrame(rows)


def _rules_from_ledger(
    latest_df: pd.DataFrame, *, decision_at: datetime
) -> dict[str, ExecutionRules]:
    """Build ExecutionRules mapping from ledger tick_size and taker_fee_bps.

    Args:
        latest_df: One row per symbol from the ledger.
        decision_at: Point-in-time decision timestamp (UTC).

    Returns:
        Mapping of instrument_id → ExecutionRules.
    """
    fallback = RuleFallbackPolicy(allow_reconstructed=True)
    rule_history_rows: list[dict[str, object]] = []
    for _, row in latest_df.iterrows():
        sym = str(row["symbol"])
        iid = f"binance_usdt_perpetual:{sym}"
        tick = row.get("tick_size")
        fee = row.get("taker_fee_bps")
        try:
            tick_val = float(tick) if tick is not None else None
            tick_val = fallback.conservative_tick_size if (tick_val is None or pd.isna(tick_val)) else tick_val
        except (TypeError, ValueError):
            tick_val = fallback.conservative_tick_size
        try:
            fee_val = float(fee) if fee is not None else None
            fee_val = fallback.conservative_taker_fee_bps if (fee_val is None or pd.isna(fee_val)) else fee_val
        except (TypeError, ValueError):
            fee_val = fallback.conservative_taker_fee_bps
        rule_history_rows.append(
            {
                "instrument_id": iid,
                "available_at": decision_at,
                "tick_size": tick_val,
                "step_size": fallback.conservative_step_size,
                "min_qty": fallback.conservative_min_qty,
                "min_notional": fallback.conservative_min_notional,
                "taker_fee_bps": fee_val,
                "confidence": DataConfidence.RECONSTRUCTED.value,
            }
        )
    if not rule_history_rows:
        return {}
    rule_history = pd.DataFrame(rule_history_rows)
    result: dict[str, ExecutionRules] = {}
    for iid in rule_history["instrument_id"].unique():
        result[iid] = resolve_execution_rules(
            iid,
            decision_at=decision_at,
            rule_history=rule_history,
            fallback_policy=fallback,
        )
    return result


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
            basket_ref=(),
            basket_weights=(),
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
    decision_at = datetime.combine(as_of_date, datetime.min.time()).replace(tzinfo=UTC)

    instruments_df = _instrument_df_from_ledger(latest)
    observations_df = _observation_df_from_ledger(latest, decision_at=decision_at)
    rules = _rules_from_ledger(latest, decision_at=decision_at)

    elig_config = ExecutionEligibilityConfig(
        max_staleness_bars=config.pit_config.max_market_data_staleness_bars,
        min_metric_observations=config.pit_config.min_metric_observations,
        max_round_trip_cost_bps=config.pit_config.max_round_trip_cost_bps,
        max_participation_rate=config.pit_config.max_participation_rate,
        min_data_confidence=DataConfidence(config.pit_config.min_data_confidence),
        default_intended_notional_usdt=config.pit_config.default_intended_notional_usdt,
    )
    intended_notional: dict[str, float] = {
        f"binance_usdt_perpetual:{sym}": config.pit_config.default_intended_notional_usdt
        for sym in latest["symbol"].tolist()
    }

    snapshot_elig = evaluate_execution_eligibility(
        decision_at=decision_at,
        instruments=instruments_df,
        observations=observations_df,
        rules=rules,
        intended_notional_usdt=intended_notional,
        config=elig_config,
    )

    calendar = pd.DatetimeIndex([pd.Timestamp(decision_at)])
    all_instrument_ids: tuple[str, ...] = tuple(
        f"binance_usdt_perpetual:{sym}" for sym in sorted(latest["symbol"].tolist())
    )
    state_cube = build_universe_state_cube(
        calendar=calendar,
        instruments=all_instrument_ids,
        snapshots=[snapshot_elig],
    )

    eligible_ids = {e.instrument_id for e in snapshot_elig.eligibilities if e.eligible}
    _cap_lookup: dict[str, float] = {
        e.instrument_id: e.capacity_usdt
        for e in snapshot_elig.eligibilities
    }
    eligible_syms_all = [
        sym
        for sym in latest["symbol"].tolist()
        if f"binance_usdt_perpetual:{sym}" in eligible_ids
    ]
    # Sort by capacity_usdt descending → apply k_in cap if > 0
    eligible_syms_all.sort(
        key=lambda s: _cap_lookup.get(f"binance_usdt_perpetual:{s}", 0.0),
        reverse=True,
    )
    _k = config.pit_config.k_in
    eligible_syms: list[str] = eligible_syms_all[:_k] if _k > 0 else eligible_syms_all

    # Build SymbolMeta for eligible instruments using actual SymbolMeta fields.
    def _cost_for(sym: str) -> float:
        iid = f"binance_usdt_perpetual:{sym}"
        for e in snapshot_elig.eligibilities:
            if e.instrument_id == iid:
                return e.cost_bps
        return 0.0

    def _capacity_for(sym: str) -> float:
        iid = f"binance_usdt_perpetual:{sym}"
        for e in snapshot_elig.eligibilities:
            if e.instrument_id == iid:
                return e.capacity_usdt
        return 0.0

    selected_meta: tuple[SymbolMeta, ...] = tuple(
        SymbolMeta(
            symbol=sym,
            role="regular",
            adv_usdt=float(
                latest.loc[latest["symbol"] == sym, "adv_usdt_median"].iloc[0]
                if sym in latest["symbol"].values and "adv_usdt_median" in latest.columns
                else 0.0
            ),
            execution_cost_bps=_cost_for(sym),
            funding_carry_8h=float(
                latest.loc[latest["symbol"] == sym, "funding_rate_8h"].iloc[0]
                if sym in latest["symbol"].values and "funding_rate_8h" in latest.columns
                else 0.0
            ),
            beta_vs_market=0.0,
            cluster_id=-1,
            tradeable_rank=idx + 1,
            basis_annualized_mean=None,
            basis_vol=None,
            capacity_clip_usdt_list=(_capacity_for(sym),),
            vol_30d=float(
                latest.loc[latest["symbol"] == sym, "vol_30d"].iloc[0]
                if sym in latest["symbol"].values and "vol_30d" in latest.columns
                else 0.0
            ),
        )
        for idx, sym in enumerate(eligible_syms)
    )

    n_eligible = len(eligible_syms)
    generated_at_utc = datetime.now(tz=UTC).isoformat()
    run_id = compute_universe_run_id(
        as_of=as_of_date,
        tf=tf,
        config_hash=hash_config(config),
        data_manifest_hash=manifest_hash,
    )
    pit_manifest = UniverseRunManifest(
        as_of=as_of_date.isoformat(),
        tf=tf,
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        config_hash=hash_config(config),
        data_manifest_hash=manifest_hash,
        generated_at_utc=generated_at_utc,
        ledger_confidence=config.ledger_confidence,
        basket_ref=(),
        basket_weights=(),
        n_stage0=int(latest["symbol"].nunique()),
        n_stage1_pass=0,
        n_stage2_pass=0,
        n_stage3_pass=0,
        n_stage4_pass=0,
        n_stage5_pass=0,
        n_stage6_selected=n_eligible,
    )
    pit_snapshot = UniverseSnapshot(
        as_of=as_of_date.isoformat(),
        tf=tf,
        schema_version=SCHEMA_VERSION,
        config_hash=hash_config(config),
        data_manifest_hash=manifest_hash,
        generated_at_utc=generated_at_utc,
        ledger_confidence=config.ledger_confidence,
        basket_ref=(),
        basket_weights=(),
        selected=selected_meta,
        rejected={},
        n_stage0=pit_manifest.n_stage0,
        n_stage1_pass=0,
        n_stage2_pass=0,
        n_stage3_pass=0,
        n_stage4_pass=0,
        n_stage5_pass=0,
        n_stage6_selected=n_eligible,
        state_transition_summary={"n_eligible": n_eligible},
        pit_state_cube=state_cube,
    )
    eligible_df = pd.DataFrame(
        {
            "symbol": eligible_syms,
            "instrument_id": [f"binance_usdt_perpetual:{s}" for s in eligible_syms],
        }
    )
    report_df = pd.DataFrame(
        [
            {
                "symbol": (
                    e.instrument_id.split(":")[-1]
                    if ":" in e.instrument_id
                    else e.instrument_id
                ),
                "stage": "pit_eligibility",
                "passed": e.eligible,
                "reason": e.code.value,
            }
            for e in snapshot_elig.eligibilities
        ]
    )
    _log.info(
        "PIT universe built: as_of=%s n_instruments=%d n_eligible=%d",
        as_of_date.isoformat(),
        pit_manifest.n_stage0,
        n_eligible,
    )
    _save_snapshot(pit_manifest, pit_snapshot, eligible_df, pd.DataFrame(), report_df, root=snapshot_root)
    return pit_snapshot, eligible_df, report_df


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
                stage5_research_panel = _stage5_symbols_from_report(report)
                has_required_metadata = {"cluster_size", "anchor_cluster_member"}.issubset(
                    set(loaded.columns)
                )
                if (
                    has_required_metadata
                    and loaded_snapshot.schema_version == SCHEMA_VERSION
                    and loaded_snapshot.config_hash == expected_config_hash
                    and loaded_snapshot.data_manifest_hash == expected_manifest_hash
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
