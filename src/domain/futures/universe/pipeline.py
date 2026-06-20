"""Offline-first PIT universe build pipeline."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.settings import FUTURES_DATA_DIR, LOG_DIR

from .config import UniverseConfig, hash_config
from .contracts import DataConfidence, ExecutionRules, UniverseStateCube
from .eligibility import (
    ExecutionEligibilityConfig,
    RuleFallbackPolicy,
    build_universe_state_cube,
    evaluate_execution_eligibility,
    resolve_execution_rules,
)
from .models import (
    DEFAULT_LEDGER_PATH,
    ManifestRow,
    SymbolMeta,
    UniverseRunManifest,
    UniverseSnapshot,
    load_ledger_slice,
)
from .storage import hash_manifest_rows
from .store import (
    DEFAULT_UNIVERSE_STORE_ROOT,
    build_decision_frame,
    compute_universe_run_id,
    load_universe_store_run,
    materialize_snapshot_from_store,
    validate_materializable_pit_store_run,
    write_universe_store_run,
)

SCHEMA_VERSION = 1
DEFAULT_MANIFEST_PATH = FUTURES_DATA_DIR / "data_manifest.parquet"
_BASE_REPORT_COLUMNS = {"symbol", "stage", "passed", "reason"}
_log = logging.getLogger(__name__)

# Deprecated — kept only for backward-compat function signatures.
DEFAULT_SNAPSHOT_ROOT = LOG_DIR / "futures/universe/snapshots"


def _to_date(as_of: str | date) -> date:
    return as_of if isinstance(as_of, date) else date.fromisoformat(as_of)


def _normalize_cfg(cfg: dict[str, Any] | UniverseConfig | None) -> UniverseConfig:
    if cfg is None:
        return UniverseConfig()
    if isinstance(cfg, UniverseConfig):
        return cfg
    return UniverseConfig(**cfg)


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





def _save_snapshot(
    manifest: UniverseRunManifest,
    snapshot: UniverseSnapshot,
    selected: pd.DataFrame,
    decisions: pd.DataFrame,
    report: pd.DataFrame,
    *,
    root: Path = DEFAULT_SNAPSHOT_ROOT,
) -> None:
    write_universe_store_run(
        manifest=manifest,
        decisions=decisions,
        report=report,
        snapshot=snapshot,
        root=DEFAULT_UNIVERSE_STORE_ROOT,
    )


def _selected_meta_to_frame(selected_meta: tuple[SymbolMeta, ...]) -> pd.DataFrame:
    rows = [
        {
            "symbol": meta.symbol,
            "role": meta.role,
            "rank": meta.tradeable_rank,
            "tradeable_score": meta.tradeable_score,
            "alpha_capacity_score": meta.alpha_capacity_score,
            "vol_30d": meta.vol_30d,
            "friction_score": meta.friction_score,
            "diversification_score": meta.diversification_score,
            "adv_usdt_median": meta.adv_usdt,
            "execution_cost_bps": meta.execution_cost_bps,
            "funding_rate_8h": meta.funding_carry_8h,
            "beta_vs_market": meta.beta_vs_market,
            "cluster_id": meta.cluster_id,
            "cluster_size": meta.cluster_size,
            "anchor_cluster_member": meta.anchor_cluster_member,
            "basis_annualized_mean": meta.basis_annualized_mean,
            "basis_vol": meta.basis_vol,
            "capacity_clip_usdt_list": meta.capacity_clip_usdt_list,
        }
        for meta in selected_meta
    ]
    return pd.DataFrame(rows)


def _metric_from_row(
    row: pd.Series | None,
    column: str,
    *,
    default: float = 0.0,
) -> float:
    if row is None or column not in row.index:
        return default
    value = row.get(column)
    if pd.isna(value):
        return default
    return float(value)


def _is_incomplete_pit_store_run(
    *,
    decisions: pd.DataFrame,
    cube: Any | None,
) -> bool:
    return not validate_materializable_pit_store_run(decisions=decisions, cube=cube)


def load_universe_snapshot(
    *,
    as_of: str | date,
    tf: str,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
) -> pd.DataFrame | None:
    """Load selected universe symbols from the latest validated store run."""
    as_of_date = _to_date(as_of)
    run_base = DEFAULT_UNIVERSE_STORE_ROOT / "runs" / f"tf={tf}" / f"as_of={as_of_date.isoformat()}"
    if not run_base.exists():
        return None
    run_dirs = sorted(
        [d for d in run_base.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        return None
    manifest_path = run_dirs[0] / "manifest.parquet"
    if not manifest_path.exists():
        return None
    try:
        manifest_df = pd.read_parquet(manifest_path)
    except Exception:
        return None
    if manifest_df.empty:
        return None
    manifest_row = manifest_df.iloc[0]
    store_run = load_universe_store_run(
        as_of=as_of_date,
        tf=tf,
        config_hash=str(manifest_row["config_hash"]),
        data_manifest_hash=str(manifest_row["data_manifest_hash"]),
        root=DEFAULT_UNIVERSE_STORE_ROOT,
    )
    if store_run is None:
        return None
    manifest, decisions, report, cube = store_run
    if not validate_materializable_pit_store_run(decisions=decisions, cube=cube):
        return None
    _snapshot, selected_frame, _report = materialize_snapshot_from_store(
        manifest=manifest,
        decisions=decisions,
        report=report,
        cube=cube,
    )
    return selected_frame


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
                # G6 continuity fields (compute_continuity_metrics via ledger)
                "n_bar_gaps": int(row.get("n_bar_gaps", 0) or 0),
                "max_gap_bars": int(row.get("max_gap_bars", 0) or 0),
                "frozen_bars": int(row.get("frozen_bars", 0) or 0),
                "n_zero_volume_bars_60d": int(row.get("n_zero_volume_bars_60d", 0) or 0),
                "last_60d_coverage": float(row.get("last_60d_coverage", 1.0) or 1.0),
                "has_nan": bool(row.get("has_nan", False)),
                "has_inf": bool(row.get("has_inf", False)),
                "has_timestamp_issues": bool(row.get("has_timestamp_issues", False)),
                "staleness_bars": 0,  # placeholder: G5 recency not yet wired
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "instrument_id", "symbol", "pair", "quote_asset", "margin_asset",
                "contract_type", "onboard_at", "status", "state_valid_from",
                "available_at", "confidence",
                "n_bar_gaps", "max_gap_bars", "frozen_bars", "n_zero_volume_bars_60d",
                "last_60d_coverage", "has_nan", "has_inf", "has_timestamp_issues",
                "staleness_bars",
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
        empty_cube = UniverseStateCube(
            calendar=pd.DatetimeIndex([]),
            instrument_ids=(),
            eligible=np.empty((0, 0), dtype=bool),
            entry_block=np.empty((0, 0), dtype=bool),
            exit_required=np.empty((0, 0), dtype=bool),
            capacity_usdt=np.empty((0, 0), dtype=np.float64),
            risk_scale=np.empty((0, 0), dtype=np.float64),
            cost_bps=np.empty((0, 0), dtype=np.float64),
        )
        snapshot, selected, report = materialize_snapshot_from_store(
            manifest=manifest,
            decisions=decisions,
            report=report,
            cube=empty_cube,
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
    latest_by_symbol = latest.set_index("symbol", drop=False)
    eligible_syms_all = [
        sym
        for sym in latest["symbol"].tolist()
        if f"binance_usdt_perpetual:{sym}" in eligible_ids
    ]
    # Sort by capacity_usdt descending (helps L2 sizing decisions)
    # Breadth-maximizing: k_max is the only bound (capacity prefix REMOVED per spec C4).
    # capacity_usdt is preserved in state_cube for L2 sizing — NOT used here for prefix cut.
    eligible_syms_all.sort(
        key=lambda s: _cap_lookup.get(f"binance_usdt_perpetual:{s}", 0.0),
        reverse=True,
    )
    _k_max = config.pit_config.k_max  # compute backstop only (now 150, spec C4)
    eligible_syms: list[str] = eligible_syms_all[:_k_max]

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

    selected_meta_items: list[SymbolMeta] = []
    for idx, sym in enumerate(eligible_syms):
        source_row = latest_by_symbol.loc[sym] if sym in latest_by_symbol.index else None
        selected_meta_items.append(
            SymbolMeta(
                symbol=sym,
                role="regular",
                adv_usdt=_metric_from_row(source_row, "adv_usdt_median"),
                execution_cost_bps=_cost_for(sym),
                funding_carry_8h=_metric_from_row(source_row, "funding_rate_8h"),
                beta_vs_market=_metric_from_row(source_row, "beta_vs_market"),
                cluster_id=int(_metric_from_row(source_row, "cluster_id", default=-1.0)),
                tradeable_rank=idx + 1,
                basis_annualized_mean=(
                    None
                    if source_row is None
                    or "basis_annualized_mean" not in source_row.index
                    or pd.isna(source_row.get("basis_annualized_mean"))
                    else float(source_row.get("basis_annualized_mean"))
                ),
                basis_vol=(
                    None
                    if source_row is None
                    or "basis_vol" not in source_row.index
                    or pd.isna(source_row.get("basis_vol"))
                    else float(source_row.get("basis_vol"))
                ),
                capacity_clip_usdt_list=(_capacity_for(sym),),
                cluster_size=_metric_from_row(source_row, "cluster_size", default=1.0),
                anchor_cluster_member=_metric_from_row(source_row, "anchor_cluster_member"),
                vol_30d=_metric_from_row(source_row, "vol_30d"),
                alpha_capacity_score=(
                    _metric_from_row(source_row, "alpha_capacity_score")
                    if source_row is not None and "alpha_capacity_score" in source_row.index
                    else _metric_from_row(source_row, "execution_pool_score")
                ),
                tradeable_score=_metric_from_row(source_row, "tradeable_score"),
            )
        )
    selected_meta = tuple(selected_meta_items)

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
    selected_frame = _selected_meta_to_frame(selected_meta)
    eligibility_report_df = pd.DataFrame(
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
    selection_report_df = pd.DataFrame(
        [
            {
                "symbol": sym,
                "stage": "stage6_selection",
                "passed": True,
                "reason": "selected",
            }
            for sym in eligible_syms
        ]
    )
    non_selected_syms = [
        sym
        for sym in eligible_syms_all
        if sym not in set(eligible_syms)
    ]
    if non_selected_syms:
        selection_report_df = pd.concat(
            [
                selection_report_df,
                pd.DataFrame(
                    [
                        {
                            "symbol": sym,
                            "stage": "stage6_selection",
                            "passed": False,
                            "reason": "not_selected",
                        }
                        for sym in non_selected_syms
                    ]
                ),
            ],
            ignore_index=True,
        )
    report_df = pd.concat(
        [eligibility_report_df, selection_report_df],
        ignore_index=True,
    )
    _log.info(
        "PIT universe built: as_of=%s n_instruments=%d n_eligible=%d",
        as_of_date.isoformat(),
        pit_manifest.n_stage0,
        n_eligible,
    )
    decisions = build_decision_frame(
        manifest=pit_manifest,
        stage5_frame=selected_frame,
        stage6_frame=selected_frame,
        report=report_df,
    )
    _save_snapshot(
        pit_manifest,
        pit_snapshot,
        selected_frame,
        decisions,
        report_df,
        root=snapshot_root,
    )
    return pit_snapshot, selected_frame, report_df


def load_or_build_universe_snapshot(
    *,
    as_of: str | date,
    tf: str,
    cfg: dict[str, Any] | UniverseConfig | None = None,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    previous_selection: tuple[str, ...] | None = None,
) -> tuple[UniverseSnapshot, pd.DataFrame, pd.DataFrame]:
    """Load cached snapshot from store, otherwise build and persist it.

    Store hit returns materialized snapshot with pit_state_cube (if persisted).
    Store miss falls back to build_universe.
    """
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
        manifest, decisions, report, cube = store_run
        if _is_incomplete_pit_store_run(decisions=decisions, cube=cube):
            _log.warning(
                "Rebuilding incomplete PIT store run: as_of=%s tf=%s run_id=%s",
                as_of_date.isoformat(),
                tf,
                manifest.run_id,
            )
        else:
            return materialize_snapshot_from_store(
                manifest=manifest,
                decisions=decisions,
                report=report,
                cube=cube,
            )

    return build_universe(
        as_of=as_of,
        tf=tf,
        cfg=cfg,
        ledger_path=ledger_path,
        snapshot_root=snapshot_root,
        previous_selection=previous_selection,
    )
