"""Strategy runtime bridge with Alpha Foundry L0 gate wiring.

[ADR_20260706_ALPHA_FOUNDRY_MAIN_WIRING][ADR_20260706_ALPHA_FOUNDRY_L0_SIGNAL_RIGOR]
[ADR_20260707_L0_MULTI_TF_GATE_REDESIGN][ADR_20260714_L0_LTF_STREAM_PARALLEL]
"""

from __future__ import annotations

import dataclasses
import gc
import logging
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.domain.futures.strategy.timeframe_contracts import (
    RESAMPLE_METADATA_BOOL_COLS,
    RESAMPLE_METADATA_FLOAT_COLS,
    hours_per_bar,
    infer_source_bar_hours,
)
from src.domain.futures.strategy.timeframe_contracts import (
    select_probe_source_tf as _shared_select_probe_source_tf,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.domain.futures.strategy.candidate_contracts import CandidateSignalPanel
    from src.domain.futures.strategy.common.alignment import AlignedMarketData
    from src.domain.futures.strategy.config import CandidateStrategyConfig, StrategyConfig


_logger = logging.getLogger(__name__)
_run_logger = logging.getLogger("opt_main_futures")


def _get_rss_mb() -> float:
    """Return current process RSS in MB via /proc/self/status (fast, no psutil dep)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        return -1.0
    return -1.0


# ---------------------------------------------------------------------------
# TF Probe helpers
# ---------------------------------------------------------------------------
_HPB_BRIDGE: dict[str, float] = {
    "1m": 1.0 / 60.0,
    "5m": 5.0 / 60.0,
    "15m": 0.25,
    "30m": 0.5,
    "1h": 1.0,
    "2h": 2.0,
    "4h": 4.0,
    "6h": 6.0,
    "8h": 8.0,
    "12h": 12.0,
}


def _fit_table_cell(value: str, width: int) -> str:
    """Fit a cell to width without breaking the ASCII table border."""
    text = value.replace("\n", " ")
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return f"{text[: width - 3]}..."


def _log_ascii_table(
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    widths: Sequence[int],
) -> None:
    """Emit a fixed-width ASCII table using the repo's audit-log style."""
    border = sum(widths) + (3 * len(widths)) + 1
    _logger.info("\n%s", title)
    _logger.info("-" * border)
    _logger.info(
        "| "
        + " | ".join(
            f"{_fit_table_cell(header, width):<{width}}" for header, width in zip(headers, widths, strict=True)
        )
        + " |"
    )
    _logger.info("-" * border)
    for row in rows:
        _logger.info(
            "| "
            + " | ".join(f"{_fit_table_cell(cell, width):<{width}}" for cell, width in zip(row, widths, strict=True))
            + " |"
        )
    _logger.info("-" * border)


def _project_panel_to_base_grid(
    panel: CandidateSignalPanel,
    base_datetimes: NDArray[np.datetime64],
    tf_i: str,
    base_tf: str,
    *,
    ltf_mode: str = "last",
) -> CandidateSignalPanel:
    """Project a non-base TF panel onto the base TF bar grid. Look-ahead safe.

    Args:
        panel: Source CandidateSignalPanel on the tf_i grid [T_i, N].
        base_datetimes: Target base TF datetime array [T_base].
        tf_i: Timeframe string of the source panel (e.g. "1h", "8h").
        base_tf: Timeframe string of the base grid (e.g. "4h").
        ltf_mode: LTF projection mode. "last" (default) picks the last signal
            within each base bar window. "mean" aggregates all signals within
            each window via mean (score, turnover) / any (valid) / mode (side).

    Returns:
        New CandidateSignalPanel projected onto base_datetimes [T_base, N].

    Time Complexity: O(T_i * N) for HTF (project_higher_tf_to_grid per symbol);
                     O(T_base * N) for LTF (cumsum-based window aggregation).
    Space Complexity: O(T_base * N) per output array.
    """
    from src.domain.futures.strategy.candidate_contracts import (  # noqa: N814
        CandidateSignalPanel as _CSP,
    )
    from src.domain.futures.strategy.rule_signals import project_higher_tf_to_grid

    hpb_i = _HPB_BRIDGE.get(tf_i, 4.0)
    hpb_base = _HPB_BRIDGE.get(base_tf, 4.0)
    t_base = len(base_datetimes)
    n_syms = panel.signed_score_2d.shape[1]
    t_i = panel.signed_score_2d.shape[0]

    proj_score = np.zeros((t_base, n_syms), dtype=np.float64)
    proj_valid = np.zeros((t_base, n_syms), dtype=bool)
    proj_side = np.zeros((t_base, n_syms), dtype=np.int8)
    proj_to = np.zeros((t_base, n_syms), dtype=np.float64)

    # Cast datetimes to canonical datetime64 for project_higher_tf_to_grid signature
    panel_dt: NDArray[np.datetime64] = np.asarray(panel.datetimes, dtype="datetime64[ns]")

    if hpb_i >= hpb_base:
        # Slower TF → vectorized backward-asof projection (single call per field)
        raw_score = project_higher_tf_to_grid(
            feature_higher=panel.signed_score_2d,
            dt_higher=panel_dt,
            dt_grid=base_datetimes,
        )
        proj_score = np.where(np.isnan(raw_score), 0.0, raw_score)

        raw_valid = project_higher_tf_to_grid(
            feature_higher=panel.valid_mask_2d.astype(np.float64),
            dt_higher=panel_dt,
            dt_grid=base_datetimes,
        )
        proj_valid = raw_valid > 0.5

        raw_side = project_higher_tf_to_grid(
            feature_higher=panel.side_hint_2d.astype(np.float64),
            dt_higher=panel_dt,
            dt_grid=base_datetimes,
        )
        proj_side = np.where(np.isnan(raw_side), 0, raw_side).astype(np.int8)

        raw_to = project_higher_tf_to_grid(
            feature_higher=panel.turnover_proxy_2d,
            dt_higher=panel_dt,
            dt_grid=base_datetimes,
        )
        proj_to = np.where(np.isnan(raw_to), 0.0, raw_to)
    elif ltf_mode == "last":
        # Faster TF → pick last tf_i signal within each base bar window
        panel_dt_int = np.asarray(panel.datetimes, dtype="datetime64[ns]").view(np.int64)
        base_dt_int = np.asarray(base_datetimes, dtype="datetime64[ns]").view(np.int64)
        idx = np.searchsorted(panel_dt_int, base_dt_int, side="right") - 1
        valid_idx = idx >= 0
        clipped = np.clip(idx, 0, t_i - 1)

        proj_score[valid_idx, :] = panel.signed_score_2d[clipped[valid_idx], :]
        proj_valid[valid_idx, :] = panel.valid_mask_2d[clipped[valid_idx], :]
        proj_side[valid_idx, :] = panel.side_hint_2d[clipped[valid_idx], :]
        proj_to[valid_idx, :] = panel.turnover_proxy_2d[clipped[valid_idx], :]
    elif ltf_mode == "mean":
        # Windowed aggregation: mean(score), any(valid), mode(side), mean(to)
        panel_dt_int = np.asarray(panel.datetimes, dtype="datetime64[ns]").view(np.int64)
        base_dt_int = np.asarray(base_datetimes, dtype="datetime64[ns]").view(np.int64)

        n_base = len(base_dt_int)
        if n_base > 1:
            diffs = np.diff(base_dt_int)
            interval_ns = int(np.median(diffs))
        else:
            interval_ns = int(hpb_base * 3600.0 * 1_000_000_000.0)

        prev_dt = np.empty_like(base_dt_int)
        prev_dt[0] = base_dt_int[0] - interval_ns
        prev_dt[1:] = base_dt_int[:-1]

        starts = np.searchsorted(panel_dt_int, prev_dt, side="right")
        ends = np.searchsorted(panel_dt_int, base_dt_int, side="right")
        valid_windows = ends > starts
        window_cnt = (ends - starts).astype(np.float64)

        for n in range(n_syms):
            # score: windowed mean via cumsum
            col = panel.signed_score_2d[:, n]
            cumsum = np.cumsum(np.insert(col, 0, 0))
            window_sum = cumsum[ends] - cumsum[starts]
            mask = valid_windows & (window_cnt > 0)
            proj_score[mask, n] = window_sum[mask] / window_cnt[mask]

            # valid: any = count > 0
            proj_valid[valid_windows, n] = True

            # side_hint: mode per window via bincount
            for i in np.where(valid_windows)[0]:
                seg = panel.side_hint_2d[starts[i] : ends[i], n]
                proj_side[i, n] = np.bincount(seg.astype(np.int64) + 1).argmax() - 1

            # turnover: windowed mean via cumsum
            col_to = panel.turnover_proxy_2d[:, n]
            cumsum_to = np.cumsum(np.insert(col_to, 0, 0))
            window_sum_to = cumsum_to[ends] - cumsum_to[starts]
            proj_to[mask, n] = window_sum_to[mask] / window_cnt[mask]
    else:
        raise ValueError(f"Unknown ltf_mode: {ltf_mode!r}")

    ratio = hpb_i / hpb_base
    new_hold = max(1, round(panel.expected_holding_bars * ratio))
    new_min_hold = max(1, round(panel.min_holding_bars * ratio))
    new_variant = f"{panel.variant}_{tf_i}"

    return _CSP(
        family=panel.family,
        variant=new_variant,
        params=dict(panel.params),
        datetimes=base_datetimes,
        symbols=panel.symbols,
        signed_score_2d=proj_score,
        side_hint_2d=proj_side,
        expected_holding_bars=new_hold,
        min_holding_bars=new_min_hold,
        stop_atr_mult=panel.stop_atr_mult,
        take_profit_atr_mult=panel.take_profit_atr_mult,
        turnover_proxy_2d=proj_to,
        valid_mask_2d=proj_valid,
        metadata=dict(panel.metadata),
        archetype=panel.archetype,
        allowed_regimes=panel.allowed_regimes,
        exit_policies=panel.exit_policies,
    )


def _select_probe_source_tf(sym_maps: Mapping[str, Any], target_tf: str) -> str | None:
    """Select the finest available source timeframe for a target probe TF."""
    return _shared_select_probe_source_tf(sym_maps, target_tf)


def _resample_probe_source_frame(frame: pd.DataFrame, *, target_tf: str) -> pd.DataFrame:
    """Resample a cached source frame into a virtual probe timeframe.

    [ADR_20260711_L0_HTF_RESAMPLE_ALIGNMENT_FIX]
    Uses open-time labeling (closed="left", label="left") to match native
    exchange candle semantics — verified byte-identical against a live
    Binance 6h kline fetch for BTCUSDT. Completeness of the trailing bin is
    determined by source-row count, not position (see [LIMIT-01]).
    """
    # No copy — all subsequent pandas ops return new DataFrames, original frame unmodified
    if "datetime" not in frame.columns:
        prepared = frame.reset_index()
        if "datetime" not in prepared.columns and len(prepared.columns) > 0:
            prepared = prepared.rename(columns={str(prepared.columns[0]): "datetime"})
    else:
        prepared = frame
    if "datetime" not in prepared.columns:
        raise ValueError("datetime column missing in probe source frame")
    prepared["datetime"] = pd.to_datetime(prepared["datetime"], utc=True, errors="coerce")
    prepared = prepared.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    prepared = prepared.set_index("datetime")
    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "funding_rate_sum" in prepared.columns:
        agg["funding_rate_sum"] = "mean"
    elif "funding_rate" in prepared.columns:
        agg["funding_rate"] = "mean"
    for col in RESAMPLE_METADATA_BOOL_COLS:
        if col in prepared.columns:
            agg[col] = "max"
    for col in RESAMPLE_METADATA_FLOAT_COLS:
        if col in prepared.columns:
            agg[col] = "mean"
    resampler = prepared.resample(target_tf, label="left", closed="left")
    resampled = resampler.agg(agg).dropna(subset=["close"])
    if resampled.empty:
        return resampled.reset_index()
    bin_counts = resampler.size().reindex(resampled.index).fillna(0)
    source_hours = infer_source_bar_hours(prepared.index)
    expected_ratio = max(1, round(hours_per_bar(target_tf) / source_hours))
    resampled = resampled.loc[bin_counts >= expected_ratio]
    return resampled.reset_index()


def _build_virtual_probe_tf_maps(
    data_maps: Mapping[str, Mapping[str, Any]],
    symbols: Sequence[str],
    target_tf: str,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Build wrapper maps {sym: {target_tf: frame}} using direct or resampled sources."""
    virtual_maps: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol in symbols:
        sym_maps = data_maps.get(symbol, {})
        source_tf = _select_probe_source_tf(sym_maps, target_tf)
        if source_tf is None:
            continue
        source_frame = sym_maps.get(source_tf)
        if not isinstance(source_frame, pd.DataFrame) or source_frame.empty:
            continue
        try:
            target_frame = (
                source_frame
                if source_tf == target_tf
                else _resample_probe_source_frame(source_frame, target_tf=target_tf)
            )
        except Exception as exc:
            _logger.warning(
                "[TF-PROBE] source resample failed symbol=%s %s->%s: %s",
                symbol,
                source_tf,
                target_tf,
                exc,
            )
            continue
        if target_frame.empty:
            continue
        virtual_maps[symbol] = {target_tf: target_frame}
    return virtual_maps


def _release_consumed_preloaded_data_maps(data_maps: MutableMapping[str, Any]) -> None:
    """Release raw per-symbol frames once the multi-TF bridge owns dense panels.

    The tiered L0 path only needs the source map until native aligned panels and
    optional LTF panels have been built. Retaining both representations doubles
    the live market-data footprint before the gate can release rejected panels.
    """
    n_symbols = len(data_maps)
    data_maps.clear()
    gc.collect()
    _run_logger.debug(
        "[SYS] stage=bridge_release_preloaded_maps released_symbols=%d rss=%.0fMB",
        n_symbols,
        _get_rss_mb(),
    )


def _build_single_tf_panels(
    data_maps: Mapping[str, Mapping[str, Any]],
    symbols: list[str] | tuple[str, ...],
    cfg: Any,
    tf_i: str,
    base_tf: str,
    family_pool: Callable[[str], tuple[str, ...]],
) -> tuple[str, AlignedMarketData | None, tuple[CandidateSignalPanel, ...]]:
    from src.domain.futures.strategy.common.alignment import align_data_maps
    from src.domain.futures.strategy.rule_signals import (
        _precompute_shared_indicators,
        build_rule_signal_panels,
    )

    tf_maps = _build_virtual_probe_tf_maps(data_maps, symbols, tf_i)
    if not tf_maps:
        return (tf_i, None, ())
    try:
        aligned_i = align_data_maps(tf_maps, symbols, tf_i, cache_result=False)
    except Exception as exc:
        _logger.warning("[MULTI-TF] align_data_maps failed tf=%s: %s", tf_i, exc)
        return (tf_i, None, ())
    del tf_maps
    gc.collect()
    try:
        cfg_i = dataclasses.replace(cfg, timeframe=tf_i)
        try:
            htf_cache = _precompute_shared_indicators(
                aligned=aligned_i,
                cfg=cfg_i,
                normalize_time_horizon=True,
                horizon_base_tf=base_tf,
            )
        except (AttributeError, KeyError, ValueError):
            htf_cache = None
        panels_i = build_rule_signal_panels(
            aligned=aligned_i,
            cfg=cfg_i,
            family_filter=family_pool(tf_i),
            normalize_time_horizon=True,
            horizon_base_tf=base_tf,
            precomputed=htf_cache,
        )
    except Exception as exc:
        _logger.warning("[MULTI-TF] build_rule_signal_panels failed tf=%s: %s", tf_i, exc)
        return (tf_i, None, ())
    del htf_cache
    gc.collect()
    return (tf_i, aligned_i, tuple(panels_i))


def _base_probe_guard_mask(aligned_base: AlignedMarketData) -> NDArray[np.bool_]:
    """Reuse base-grid eligibility masks for projected probe panels.
    Falls back to ones for any missing mask attribute (test compat)."""
    execution_eligibility = getattr(
        aligned_base,
        "execution_eligibility_mask",
        np.ones_like(aligned_base.active_mask, dtype=bool),
    )
    if execution_eligibility is None:
        execution_eligibility = np.ones_like(aligned_base.active_mask, dtype=bool)
    strategy_readiness = getattr(
        aligned_base,
        "strategy_readiness_mask",
        np.ones_like(aligned_base.active_mask, dtype=bool),
    )
    if strategy_readiness is None:
        strategy_readiness = np.ones_like(aligned_base.active_mask, dtype=bool)
    promotion_active = getattr(
        aligned_base,
        "promotion_active_mask",
        np.ones_like(aligned_base.active_mask, dtype=bool),
    )
    if promotion_active is None:
        promotion_active = np.ones_like(aligned_base.active_mask, dtype=bool)
    guard = (
        aligned_base.active_mask
        & aligned_base.warm_mask
        & execution_eligibility
        & strategy_readiness
        & promotion_active
        & ~aligned_base.entry_block_mask
        & ~aligned_base.kill_mask
    )
    return np.asarray(guard, dtype=bool)


def build_native_htf_panels(
    *,
    data_maps: dict[str, Any],
    symbols: list[str],
    aligned_base: AlignedMarketData,
    base_cfg: CandidateStrategyConfig,
    base_tf: str,
    tfs: tuple[str, ...],
    family_pool: Callable[[str], tuple[str, ...]],
    htf_only: bool = True,
) -> dict[str, tuple[AlignedMarketData, tuple[CandidateSignalPanel, ...]]]:
    """Build native timeframe panels with bounded transient concurrency.

    [ADR_20260714_L1_MEMORY_EXECUTION]
    """
    non_base_tfs = {tf for tf in tfs if tf != base_tf}
    if not non_base_tfs:
        return {}

    result: dict[str, tuple[AlignedMarketData, tuple[CandidateSignalPanel, ...]]] = {}

    # Phase 1: collect eligible TFs (HTF filter)
    eligible_tfs: list[str] = []
    for tf_i in non_base_tfs:
        hpb_i = _HPB_BRIDGE.get(tf_i, 4.0)
        hpb_base = _HPB_BRIDGE.get(base_tf, 4.0)
        if htf_only and hpb_i < hpb_base:
            continue
        eligible_tfs.append(tf_i)

    # Phase 2: build via module-level helper
    for tf_i in eligible_tfs:
        _, aligned_i, native_panels = _build_single_tf_panels(
            data_maps=data_maps,
            symbols=symbols,
            cfg=base_cfg,
            tf_i=tf_i,
            base_tf=base_tf,
            family_pool=family_pool,
        )
        if aligned_i is not None and native_panels:
            result[tf_i] = (aligned_i, native_panels)

    return result


def _build_ltf_native_panels_for_l0(
    *,
    data_maps: Mapping[str, Mapping[str, Any]],
    symbols: Sequence[str],
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    runtime_config: Any,
    family_filter: tuple[str, ...] | None = None,
) -> tuple[CandidateSignalPanel, ...]:
    """Build bounded LTF panels without retaining universe-wide 1m frames.

    [ADR_20260713_L1_1M_COVERAGE_WARMUP] Coverage is evaluated only for the
    admitted symbol scope and the configured warmup-to-holdout interval.
    """
    del cfg
    from src.domain.futures.alpha_foundry.entry_timing import resolve_1m_coverage_tier
    from src.domain.futures.alpha_foundry.memory import (
        resolve_effective_memory_budget,
        resolve_ltf_exec_1m_plan,
    )
    from src.domain.futures.optimization.opt_data_utils import load_ltf_exec_1m_frame
    from src.domain.futures.signals.ltf_alpha import build_ltf_native_alpha_panels_streaming

    if not getattr(runtime_config, "ltf_streaming_enabled", True) or not len(aligned.datetimes):
        return ()

    start = pd.Timestamp(aligned.datetimes[0]).tz_localize("UTC")
    end = pd.Timestamp(aligned.datetimes[-1]).tz_localize("UTC")
    data_root = Path("data")
    coverage = resolve_1m_coverage_tier(
        tuple(symbols),
        data_root=data_root / "futures",
        start_date=start.date(),
        end_date=end.date(),
        min_coverage_ratio=float(getattr(runtime_config, "ltf_exec_1m_min_coverage", 0.80)),
    )
    budget = resolve_effective_memory_budget(
        hard_limit_mb=int(getattr(runtime_config, "l0_max_rss_mb", 10_240)),
        fraction_cap=float(getattr(runtime_config, "l0_memory_fraction_cap", 0.60)),
    )
    payload_symbols = frozenset(
        symbol
        for symbol in symbols
        if isinstance(data_maps.get(symbol, {}).get("exec_1m"), pd.DataFrame) and not data_maps[symbol]["exec_1m"].empty
    )
    covered_for_plan = payload_symbols or coverage.covered_symbols
    _logger.info(
        "[LTF_ALPHA] exec_1m payload symbols=%d/%d",
        len(payload_symbols),
        len(symbols),
    )
    plan = resolve_ltf_exec_1m_plan(
        covered_symbols=covered_for_plan,
        valid_symbols=frozenset(symbols),
        max_symbols=int(getattr(runtime_config, "ltf_exec_1m_max_symbols", 64)),
        max_workers=int(getattr(runtime_config, "ltf_exec_1m_max_workers", 1)),
        budget=budget,
    )
    _run_logger.debug(
        "[LTF_ALPHA] stage=l0_ltf_stream rss_mb=%.0f budget_mb=%d symbols_planned=%d "
        "symbols_covered=%d workers=%d skip_reason=%s",
        _get_rss_mb(),
        budget.limit_mb,
        len(plan.symbols),
        len(coverage.covered_symbols),
        plan.max_workers,
        plan.skip_reason,
    )
    if plan.skip_reason is not None:
        return ()

    def _load_frame(symbol: str) -> pd.DataFrame | None:
        payload = data_maps.get(symbol, {}).get("exec_1m")
        if isinstance(payload, pd.DataFrame) and not payload.empty:
            return payload
        return load_ltf_exec_1m_frame(
            symbol=symbol,
            data_root=data_root,
            start_datetime=start,
            end_datetime=end,
        )

    try:
        panels = build_ltf_native_alpha_panels_streaming(
            aligned=aligned,
            plan=plan,
            load_frame=_load_frame,
            budget=budget,
        )
    except Exception as exc:
        _logger.warning("[LTF_ALPHA] streaming build failed: %s", exc)
        return ()
    if family_filter is not None:
        panels = tuple(panel for panel in panels if panel.family in family_filter)
    if not panels:
        _run_logger.info("[LTF_ALPHA] no streaming panels generated for current L0 window")
        return ()

    # [ADR_20260712_L0_EVIDENCE_CONDITIONED_CROSS_TF_ADMISSION] Stamp data-support
    # metadata on LTF panels requiring 1m execution/taker inputs.
    exec_1m_coverage_by_symbol = {symbol: 1.0 if symbol in covered_for_plan else 0.0 for symbol in symbols}

    stamped_panels: list[CandidateSignalPanel] = []
    for p in panels:
        new_meta = dict(p.metadata)
        new_meta["requires_exec_1m"] = True
        new_meta["exec_1m_coverage_by_symbol"] = dict(exec_1m_coverage_by_symbol)
        stamped_panels.append(dataclasses.replace(p, metadata=new_meta))

    return tuple(stamped_panels)


def project_htf_panels_to_base(
    *,
    native_by_tf: Mapping[str, tuple[AlignedMarketData, tuple[CandidateSignalPanel, ...]]],
    aligned_base: AlignedMarketData,
    base_tf: str,
) -> tuple[CandidateSignalPanel, ...]:
    base_datetimes = aligned_base.datetimes
    base_guard = _base_probe_guard_mask(aligned_base)
    extra: list[CandidateSignalPanel] = []
    audit_rows: list[list[str]] = []

    for tf_i, (aligned_i, native_panels) in native_by_tf.items():
        tf_panels: list[CandidateSignalPanel] = []
        for panel in native_panels:
            try:
                projected = _project_panel_to_base_grid(panel, base_datetimes, tf_i, base_tf)
                tagged_metadata: dict[str, Any] = dict(projected.metadata or {})
                tagged_metadata["native_tf"] = tf_i
                tf_panels.append(
                    dataclasses.replace(
                        projected,
                        valid_mask_2d=projected.valid_mask_2d & base_guard,
                        metadata=tagged_metadata,
                    )
                )
            except Exception as exc:
                _logger.warning(
                    "[MULTI-TF] projection failed %s:%s tf=%s: %s",
                    panel.family,
                    panel.variant,
                    tf_i,
                    exc,
                )
        extra.extend(tf_panels)
        audit_rows.append(
            [
                tf_i,
                str(len(aligned_i.symbols) if hasattr(aligned_i, "symbols") else 0),
                "0",
                str(len(tf_panels)),
                "-",
            ]
        )

    if audit_rows:
        import logging

        logger = logging.getLogger("opt_main_futures")
        tf_summaries = [f"[{row[0]}] Proj={row[3]} Syms={row[1]}" for row in audit_rows]
        logger.info(f"\U0001f9ec [L1: MULTI-TF PANEL INJECTION]\n  \u2514\u2500 Active : {' | '.join(tf_summaries)}")

    return tuple(extra)


def build_multi_tf_panels(
    *,
    data_maps: dict[str, Any],
    symbols: list[str],
    aligned_base: AlignedMarketData,
    base_cfg: CandidateStrategyConfig,
    base_tf: str,
    tfs: tuple[str, ...],
    family_pool: Callable[[str], tuple[str, ...]],
    htf_only: bool = True,
) -> tuple[CandidateSignalPanel, ...]:
    """Thin wrapper: build_native_htf_panels -> project_htf_panels_to_base."""
    native_by_tf = build_native_htf_panels(
        data_maps=data_maps,
        symbols=symbols,
        aligned_base=aligned_base,
        base_cfg=base_cfg,
        base_tf=base_tf,
        tfs=tfs,
        family_pool=family_pool,
        htf_only=htf_only,
    )
    return project_htf_panels_to_base(
        native_by_tf=native_by_tf,
        aligned_base=aligned_base,
        base_tf=base_tf,
    )


@dataclass(frozen=True, slots=True)
class _RuntimeBreakdown:
    total: float
    steps: Mapping[str, float]

    @property
    def accounted(self) -> float:
        return float(sum(max(float(value), 0.0) for value in self.steps.values()))

    @property
    def unaccounted(self) -> float:
        return max(float(self.total) - self.accounted, 0.0)


def verify_data_integrity(
    aligned: AlignedMarketData, symbols: list[str], min_length: int = 100
) -> dict[str, dict[str, Any]]:
    """Verify market data integrity per symbol before running the candidate strategy.

    Args:
        aligned: Aligned market data structure containing 2D pricing/volume matrices.
        symbols: List of symbols in the universe.
        min_length: Minimum data length (rows) required for reliable strategy execution.

    Returns:
        A dictionary mapping each symbol to its validation results.
    """
    report: dict[str, dict[str, Any]] = {}
    n_bars = aligned.close_2d.shape[0]

    _logger.debug("[DATA-INTEGRITY] 💠 Starting audit for %d symbols...", len(symbols))

    passed_symbols = []
    failed_symbols_info = []

    for col_idx, sym in enumerate(symbols):
        close = aligned.close_2d[:, col_idx]
        high = aligned.high_2d[:, col_idx]
        low = aligned.low_2d[:, col_idx]
        volume = aligned.volume_2d[:, col_idx]

        nan_count = np.isnan(close).sum() + np.isnan(high).sum() + np.isnan(low).sum() + np.isnan(volume).sum()
        nan_pct = (nan_count / (4 * n_bars)) * 100 if n_bars > 0 else 100.0

        zero_neg_count = (close <= 0).sum() + (high <= 0).sum() + (low <= 0).sum() + (volume < 0).sum()
        zero_neg_pct = (zero_neg_count / (4 * n_bars)) * 100 if n_bars > 0 else 100.0

        close_std = float(np.std(close)) if n_bars > 1 else 0.0
        hi_lo_violation = int((high < low).sum())

        reasons = []
        if n_bars < min_length:
            reasons.append("too_short")
        if nan_pct > 0.0:
            reasons.append("excessive_nan")
        if zero_neg_pct > 1.0:
            reasons.append("invalid_values")
        if close_std < 1e-8:
            reasons.append("stuck_price")
        if hi_lo_violation > 0:
            reasons.append("hi_lo_violation")

        status = "FAIL" if reasons else "PASS"
        status_str = f"FAIL ({','.join(reasons)})" if reasons else "PASS"

        if status == "PASS":
            passed_symbols.append((sym, n_bars))
        else:
            failed_symbols_info.append((sym, n_bars, nan_pct, zero_neg_pct, close_std, hi_lo_violation, status_str))

        report[sym] = {
            "status": status,
            "nan_pct": nan_pct,
            "zero_neg_pct": zero_neg_pct,
            "close_std": close_std,
            "hi_lo_violation": hi_lo_violation,
            "reasons": reasons,
        }

    from src.domain.futures.strategy.tiered_logging import format_data_integrity_summary

    if passed_symbols:
        bar_lengths = sorted({x[1] for x in passed_symbols})
        avg_bars = int(np.mean(bar_lengths)) if bar_lengths else 0
        _logger.info(
            format_data_integrity_summary(
                total=len(symbols), passed=len(passed_symbols), bars=avg_bars, nan_pct=0.0, zero_pct=0.0
            )
        )

    if failed_symbols_info:
        _logger.warning("[DATA-INTEGRITY] FAIL: %d symbols failed integrity check:", len(failed_symbols_info))
        _logger.warning(
            "[DATA-INTEGRITY] %-12s | %-6s | %-6s | %-6s | %-6s | %-6s | %s",
            "Symbol",
            "Bars",
            "NaN%",
            "Zero%",
            "VolStd",
            "Hi>=Lo",
            "Status (Reason)",
        )
        for info in failed_symbols_info:
            sym, n_bars, nan_pct, zero_neg_pct, close_std, hi_lo_violation, status_str = info
            _logger.warning(
                "[DATA-INTEGRITY] %-12s | %-6d | %-5.1f%% | %-5.1f%% | %-6.4f | %-6s | %s",
                sym,
                n_bars,
                nan_pct,
                zero_neg_pct,
                close_std,
                "FAIL" if hi_lo_violation > 0 else "PASS",
                status_str,
            )

    return report


def _candidate_ml_split_indices(
    *,
    n_bars: int,
    fit_fraction: float,
    calibration_fraction: float,
    purge_bars: int,
    embargo_bars: int,
) -> tuple[int, int, int, int, int, int]:
    """Return fit, calibration, and OOS index ranges."""
    fit_start = 0
    fit_end = int(n_bars * fit_fraction)
    calibration_start = fit_end + purge_bars
    calibration_end = int(n_bars * (fit_fraction + calibration_fraction))
    oos_start = calibration_end + embargo_bars
    oos_end = n_bars
    if not (fit_start < fit_end <= n_bars):
        raise ValueError("fit split is empty or invalid")
    if not (calibration_start < calibration_end <= n_bars):
        raise ValueError("calibration split is empty or invalid")
    if not (oos_start < oos_end <= n_bars):
        raise ValueError("oos split is empty or invalid")
    return fit_start, fit_end, calibration_start, calibration_end, oos_start, oos_end


def _finite_summary(values: np.ndarray) -> dict[str, float]:
    """Return finite mean/median/p90/min/max statistics for a numeric array."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _threshold_rate(values: np.ndarray, threshold: float) -> float:
    """Return fraction of finite values above a threshold."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float((finite >= threshold).mean())


def _log_universe_volatility_deciles(
    *,
    events: pd.DataFrame,
    selected: pd.DataFrame,
    mu_net_decision_bps: np.ndarray,
    q10_net_bps: np.ndarray,
) -> None:
    if events.empty or "vol_30d" not in events.columns:
        return
    vol = pd.to_numeric(events["vol_30d"], errors="coerce")
    valid = vol.notna()
    if int(valid.sum()) < 10:
        return
    diag = events.loc[valid].copy()
    diag["_mu_net_decision_bps"] = np.asarray(mu_net_decision_bps, dtype=np.float64)[valid.to_numpy()]
    diag["_q10_net_bps"] = np.asarray(q10_net_bps, dtype=np.float64)[valid.to_numpy()]
    diag["_selected"] = False
    if not selected.empty:
        selected_keys = {
            (
                pd.Timestamp(dt).tz_localize(None) if pd.Timestamp(dt).tzinfo is not None else pd.Timestamp(dt),
                str(sym),
                int(entry_idx),
                str(family),
                str(variant),
            )
            for dt, sym, entry_idx, family, variant in selected.loc[
                :, ["datetime", "symbol", "entry_idx", "family", "variant"]
            ].itertuples(index=False, name=None)
        }
        diag["_selected"] = [
            (
                pd.Timestamp(dt).tz_localize(None) if pd.Timestamp(dt).tzinfo is not None else pd.Timestamp(dt),
                str(sym),
                int(entry_idx),
                str(family),
                str(variant),
            )
            in selected_keys
            for dt, sym, entry_idx, family, variant in diag.loc[
                :, ["datetime", "symbol", "entry_idx", "family", "variant"]
            ].itertuples(index=False, name=None)
        ]
    diag["_vol_decile"] = pd.qcut(vol.loc[valid], q=10, labels=False, duplicates="drop")
    grouped = diag.groupby("_vol_decile", sort=True, dropna=True)
    for decile, group in grouped:
        _logger.debug(
            "[DIAG][VOL_DECILE] decile=%s events=%d mu_mean=%.1f q10_median=%.1f selected_pass_rate=%.3f",
            int(decile) + 1,
            int(group.shape[0]),
            float(pd.to_numeric(group["_mu_net_decision_bps"], errors="coerce").mean()),
            float(pd.to_numeric(group["_q10_net_bps"], errors="coerce").median()),
            float(pd.to_numeric(group["_selected"], errors="coerce").mean()),
        )


def _recommendation_window_indices(
    *,
    fit_start: int,
    fit_end: int,
    calibration_start: int,
    calibration_end: int,
    cfg: Any,
) -> tuple[int, int]:
    """Return the contiguous recommendation window to evaluate for promotion."""
    basis = str(getattr(cfg, "promotion_decision_split", "fit_calibration"))
    if basis == "fit":
        return fit_start, fit_end
    if basis == "calibration":
        return calibration_start, calibration_end
    if basis == "fit_calibration":
        return fit_start, calibration_end
    raise ValueError(f"unsupported promotion_decision_split: {basis}")


def _empty_rule_diagnostics() -> Any:
    """signal_only fast-path용 빈 RuleDiagnosticsResult."""
    empty_df = pd.DataFrame()
    return SimpleNamespace(
        by_family=empty_df,
        by_variant=empty_df,
        by_family_side=empty_df,
        side_flip=empty_df,
        decision={},
        recommended_keep_variants=(),
        recommended_flip_variants=(),
        recommended_keep_signal_cells=(),
        recommended_flip_signal_cells=(),
        recommendation_basis="skipped_signal_only",
        recommendation_split=(0, 0),
        report_split=(0, 0),
        recommendation_failure_report=None,
    )


@dataclass(slots=True)
class CandidatePipelineOutput:
    """Candidate strategy bridge output."""

    alpha_panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    target_weights: np.ndarray | None = None
    rule_report: dict[str, Any] | None = None
    aligned: AlignedMarketData | None = None
    aligned_by_tf: dict[str, AlignedMarketData] | None = None
    labeled_events_by_tf: dict[str, pd.DataFrame] | None = None
    labeled: pd.DataFrame | None = None
    labeled_unfiltered: pd.DataFrame | None = None
    fit_set: Any | None = None
    calibration_set: Any | None = None
    oos_set: Any | None = None
    gate_model: Any | None = None
    edge_models: Any | None = None
    fit_start: int | None = None
    fit_end: int | None = None
    calibration_start: int | None = None
    calibration_end: int | None = None
    oos_start: int | None = None
    oos_end: int | None = None
    fold_oos_boundaries: tuple[tuple[int, int], ...] | None = None
    alpha_foundry_report: Any | None = None
    l0_delivery_manifest: Any | None = None

    def __post_init__(self) -> None:
        if self.aligned is not None and not self.aligned_by_tf:
            _logger.warning(
                "[DATA] CandidatePipelineOutput.aligned is set but aligned_by_tf is "
                "missing/empty -- every TF's L1 walk-forward will silently share one grid"
            )


def run_candidate_strategy_for_universe(
    symbols: list[str],
    tf: str,
    *,
    strategy_cfg: StrategyConfig | None = None,
    preloaded_data_maps: dict[str, dict[str, Any]] | None = None,
    alpha_foundry_config: Any | None = None,
    silent: bool = False,
    state_cube: Any | None = None,
    l0_evidence_end: Any | None = None,
    ltf_sync_selection: Any | None = None,
    derived_store: Any | None = None,
) -> CandidatePipelineOutput:
    """Run candidate strategy pipeline and return candidate output."""
    if strategy_cfg is None or preloaded_data_maps is None:
        return CandidatePipelineOutput()

    import time
    from dataclasses import replace

    from src.domain.futures.alpha_foundry.bridge_helpers import (
        assemble_l0_strategy_delivery_manifest,
        bind_panels_to_alpha_recipes,
        build_causal_l0_panel_views,
        run_alpha_foundry_l0_gate,
        run_alpha_foundry_l0_gate_multi_tf,
    )
    from src.domain.futures.alpha_foundry.recipes import build_alpha_recipe_catalog
    from src.domain.futures.strategy.ablation import apply_variant_promotions
    from src.domain.futures.strategy.candidate_dataset import (
        build_candidate_dataset,
        prepare_labeled_events,
    )
    from src.domain.futures.strategy.candidate_edge import predict_candidate_edges
    from src.domain.futures.strategy.candidate_gate import predict_candidate_gate
    from src.domain.futures.strategy.candidate_labels import (
        _compute_yang_zhang_vol_2d,
        label_candidate_events,
    )
    from src.domain.futures.strategy.candidate_portfolio import (
        build_candidate_alpha_panel,
        build_candidate_target_weights,
        select_candidate_events_for_portfolio,
    )
    from src.domain.futures.strategy.config import resolve_purge_and_embargo_bars, with_max_holding_bars
    from src.domain.futures.strategy.execution_cost import ExecutionCostModel
    from src.domain.futures.strategy.family_lifecycle import RETIRED_FAMILIES, resolve_retired_families_for_tf
    from src.domain.futures.strategy.rule_diagnostics import compute_rule_diagnostics
    from src.domain.futures.strategy.rule_signals import (
        candidate_panels_to_events,
    )
    from src.domain.futures.strategy.walk_forward import build_walk_forward_folds

    bridge_t0 = time.perf_counter()
    alpha_foundry_report: Any | None = None
    l0_delivery_manifest: Any | None = None
    l0_evidence_end_ns: int | None = None
    if l0_evidence_end is not None:
        _cutoff = pd.Timestamp(l0_evidence_end)
        _cutoff = _cutoff.tz_localize("UTC") if _cutoff.tzinfo is None else _cutoff.tz_convert("UTC")
        l0_evidence_end_ns = int(_cutoff.value)
    bridge_prof: dict[str, float] = {
        "align": 0.0,
        "rules": 0.0,
        "events": 0.0,
        "label": 0.0,
        "diagnostics": 0.0,
        "promotions": 0.0,
        "walk_forward": 0.0,
        "post_wf": 0.0,
        "selection": 0.0,
        "weights": 0.0,
        "alpha_panel": 0.0,
    }
    stage_rss_samples: list[tuple[str, float, float]] = []
    rss_baseline = _get_rss_mb()
    wf_fold_times: list[float] = []
    wf_fold_n_events: list[int] = []

    def _sample_rss(stage_name: str) -> None:
        rss = _get_rss_mb()
        delta = rss - rss_baseline if rss >= 0 and rss_baseline >= 0 else -1.0
        stage_rss_samples.append((stage_name, rss, delta))

    def _emit_bridge_profile() -> None:
        total_time = time.perf_counter() - bridge_t0
        breakdown = _RuntimeBreakdown(total=total_time, steps=bridge_prof)

        width = 60
        border = "━" * width
        lines = [
            f"[BRIDGE PERFORMANCE] {border}",
            f"  Total Runtime: {total_time:.2f}s (Accounted: {breakdown.accounted / total_time:.1%})",
            "",
        ]

        # Sort steps by duration for the bar chart
        sorted_steps = sorted(bridge_prof.items(), key=lambda x: x[1], reverse=True)
        max_step_time = max(bridge_prof.values()) if bridge_prof else 1.0

        shown_names: set[str] = set()
        for name, duration in sorted_steps:
            if duration < 0.01 and name != sorted_steps[0][0]:
                continue

            bar_width = int((duration / max_step_time) * 20)
            bar = "█" * bar_width
            pct = (duration / total_time) * 100

            # Add fire emoji for the top bottleneck if it's significant
            suffix = " 🔥" if name == sorted_steps[0][0] and pct > 30 else ""
            skip_label = " (skipped)" if name == "walk_forward" and duration < 0.01 else ""

            label = name.replace("_", " ").title()
            lines.append(f"  {label:<15}: {bar:<20} {duration:>6.2f}s ({pct:>5.1f}%){suffix}{skip_label}")
            shown_names.add(name)

        if "walk_forward" not in shown_names:
            wf_dur = bridge_prof.get("walk_forward", 0.0)
            lines.append(f"  {'Walk Forward':<15}: {'                    '} {wf_dur:>6.2f}s (  0.0%) (skipped)")

        # Memory section
        if stage_rss_samples:
            peak_rss = max(s[1] for s in stage_rss_samples)
            lines.append("")
            lines.append(f"  [MEMORY] Peak RSS: {peak_rss:.0f} MB | Baseline: {rss_baseline:.0f} MB")
            lines.append("  Stage RSS Delta (top 5 by delta):")
            sorted_rss = sorted(
                [(name, delta) for name, rss, delta in stage_rss_samples if delta > 0],
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            for name, delta in sorted_rss:
                label = name.replace("_", " ").title()
                lines.append(f"    {label:<15}: +{delta:>7.1f} MB")

        # WF fold timing section
        if wf_fold_times:
            lines.append("")
            lines.append(f"  [WF FOLDS] n_folds={len(wf_fold_times)}")
            for i, (t_fold, n_ev) in enumerate(zip(wf_fold_times, wf_fold_n_events, strict=False)):
                lines.append(f"    Fold {i:<3}: {t_fold:>6.2f}s  events={n_ev}")

        lines.append(border)
        _logger.debug("\n".join(lines))

    # ── Uniform TF construction (Phase 2: symmetric, no privileged base TF) ──
    from src.domain.futures.strategy.config import resolve_tf_signal_pool

    l1_tfs = getattr(strategy_cfg.candidate, "l1_tfs", (tf,))
    aligned_by_tf: dict[str, Any] = {}
    panels_by_tf: dict[str, Sequence[Any]] = {}
    t_step = time.perf_counter()
    _run_logger.debug(
        "[MEM] stage=bridge_pre_align rss=%.0fMB n_symbols=%d l1_tfs=%s",
        _get_rss_mb(),
        len(symbols),
        l1_tfs,
    )
    for tf_i in l1_tfs:
        try:
            _, aligned_i, panels_i = _build_single_tf_panels(
                data_maps=preloaded_data_maps,
                symbols=symbols,
                cfg=strategy_cfg.candidate,
                tf_i=tf_i,
                base_tf=tf,
                family_pool=lambda t: resolve_tf_signal_pool(strategy_cfg.candidate, t),
            )
        except Exception as _exc:
            _logger.warning("[MULTI-TF] _build_single_tf_panels failed tf=%s: %s", tf_i, _exc)
            aligned_i, panels_i = None, ()
        if aligned_i is not None:
            aligned_by_tf[tf_i] = aligned_i
            if panels_i:
                panels_by_tf[tf_i] = panels_i
            else:
                panels_by_tf[tf_i] = ()
    if tf not in aligned_by_tf:
        _run_logger.error("[DATA] base_tf=%s could not be built via symmetric path", tf)
        return CandidatePipelineOutput()
    aligned = aligned_by_tf[tf]

    # ── Per-TF native labeled events (L1 walk-forward) [LIMIT-01] ──
    # Populated later, after L0 recipe-binding stamps metadata["recipe_id"] onto the
    # admitted panels (pruned_multi_results[tf_k].panels_for_l1) -- using the raw,
    # unstamped panels_by_tf here would produce l0_recipe_id="" for every event,
    # matching no manifest route (confirmed via live re-run: every TF blocked).
    labeled_events_by_tf: dict[str, pd.DataFrame] = {}

    def _label_panels_for_tf_native(tf_i: str, panels_i: Sequence[Any]) -> None:
        if not panels_i:
            return
        try:
            aligned_i = aligned_by_tf[tf_i]
            raw_events_i = candidate_panels_to_events(
                panels_i,
                min_abs_score=strategy_cfg.candidate.min_rule_net_bps * 1e-4,
                side_flip_variants=strategy_cfg.candidate.side_flip_candidate_variants,
                cost_floor_bps=strategy_cfg.candidate.cost_floor_bps,
                execution_cost_bps_2d=aligned_i.execution_cost_bps_2d,
            )
            if raw_events_i.empty:
                return
            labeled_i = label_candidate_events(
                events=raw_events_i,
                aligned=aligned_i,
                cfg=strategy_cfg.candidate,
            )
            labeled_i["native_tf"] = tf_i
            labeled_events_by_tf[tf_i] = labeled_i
        except Exception as exc:
            _logger.warning("[DATA] per-tf native labeling failed tf=%s: %s", tf_i, exc)

    def _build_output(**overrides: Any) -> CandidatePipelineOutput:
        overrides.setdefault("aligned", aligned)
        overrides.setdefault("aligned_by_tf", aligned_by_tf)
        overrides.setdefault("labeled_events_by_tf", labeled_events_by_tf)
        return CandidatePipelineOutput(**overrides)

    panels = tuple(panels_by_tf.get(tf, ()))
    panels = tuple(dataclasses.replace(p, variant=f"{p.variant}_{tf}") for p in panels)
    bridge_prof["align"] = time.perf_counter() - t_step
    _sample_rss("align")
    _run_logger.debug(
        "[MEM] stage=bridge_post_align rss=%.0fMB took=%.4fs n_tfs_built=%d",
        _get_rss_mb(),
        bridge_prof["align"],
        len(aligned_by_tf),
    )
    if aligned is not None:
        n_bars = aligned.close_2d.shape[0]
        _logger.debug(
            "[BRIDGE][INPUT] n_symbols=%d n_bars=%d tf=%s",
            len(symbols),
            n_bars,
            tf,
        )

    t_step = time.perf_counter()
    if alpha_foundry_config is not None and getattr(alpha_foundry_config, "mode", "off") != "off":
        if aligned is not None:
            ltf_panels = _build_ltf_native_panels_for_l0(
                data_maps=preloaded_data_maps,
                symbols=symbols,
                aligned=aligned,
                cfg=strategy_cfg.candidate,
                runtime_config=alpha_foundry_config,
                family_filter=alpha_foundry_config.include_families or None,
            )
            if ltf_panels:
                panels = (*panels, *ltf_panels)
                panels_by_tf[tf] = panels
                _run_logger.debug("[LTF_ALPHA] appended panels=%d base_tf=%s", len(ltf_panels), tf)
            else:
                _run_logger.debug("[LTF_ALPHA] appended panels=0 base_tf=%s", tf)
        else:
            _run_logger.debug("[LTF_ALPHA] skipped (no aligned data for base_tf=%s)", tf)
    bridge_prof["rules"] = time.perf_counter() - t_step
    _sample_rss("rules")
    _run_logger.debug(
        "[MEM] stage=bridge_post_rules rss=%.0fMB took=%.4fs n_panels=%d",
        _get_rss_mb(),
        bridge_prof["rules"],
        len(panels),
    )
    _multi_tf_htf_panels: tuple[Any, ...] | None = None
    if alpha_foundry_config is not None and getattr(alpha_foundry_config, "mode", "off") != "off":
        t_step = time.perf_counter()
        recipes_seq = build_alpha_recipe_catalog(
            timeframe=tf,
            include_families=alpha_foundry_config.include_families,
            exclude_families=tuple(
                set(alpha_foundry_config.exclude_families) | set(resolve_retired_families_for_tf(tf)) | RETIRED_FAMILIES
            ),
            max_recipes_per_family=alpha_foundry_config.max_recipes_per_family,
        )
        recipes = {recipe.recipe_id: recipe for recipe in recipes_seq}
        bindings = bind_panels_to_alpha_recipes(
            panels=panels,
            recipes=recipes,
            timeframe=tf,
            max_recipes_per_family=alpha_foundry_config.max_recipes_per_family,
            include_families=alpha_foundry_config.include_families,
            exclude_families=tuple(
                set(alpha_foundry_config.exclude_families) | set(resolve_retired_families_for_tf(tf)) | RETIRED_FAMILIES
            ),
            enable_synthetic_recipes=alpha_foundry_config.enable_synthetic_recipes,
        )

        _use_multi_tf = getattr(alpha_foundry_config, "use_all_timeframes_in_l0", True) and len(l1_tfs) > 1

        if _use_multi_tf:
            _release_consumed_preloaded_data_maps(preloaded_data_maps)
            # ── Multi-TF L0 gate path (fan-out/fuse/fan-in) ──────────────────
            import time as _time_panel

            _t_panel_construct = _time_panel.perf_counter()

            recipes_by_tf: dict[str, MutableMapping[str, Any]] = {tf: recipes}
            bindings_by_tf: dict[str, Sequence[Any]] = {tf: bindings}

            non_base_tfs = tuple(t for t in l1_tfs if t != tf)
            for htf in non_base_tfs:
                if htf not in aligned_by_tf or htf not in panels_by_tf:
                    panels_by_tf[htf] = []
                    recipes_by_tf[htf] = {}
                    bindings_by_tf[htf] = []
                    aligned_by_tf[htf] = aligned
                    continue
                htf_native_panels = panels_by_tf[htf]
                htf_recipes_seq = build_alpha_recipe_catalog(
                    timeframe=htf,
                    include_families=alpha_foundry_config.include_families,
                    exclude_families=tuple(
                        set(alpha_foundry_config.exclude_families)
                        | set(resolve_retired_families_for_tf(htf))
                        | RETIRED_FAMILIES
                    ),
                    max_recipes_per_family=alpha_foundry_config.max_recipes_per_family,
                )
                htf_recipes = {r.recipe_id: r for r in htf_recipes_seq}
                htf_bindings = bind_panels_to_alpha_recipes(
                    panels=htf_native_panels,
                    recipes=htf_recipes,
                    timeframe=htf,
                    max_recipes_per_family=alpha_foundry_config.max_recipes_per_family,
                    include_families=alpha_foundry_config.include_families,
                    exclude_families=tuple(
                        set(alpha_foundry_config.exclude_families)
                        | set(resolve_retired_families_for_tf(htf))
                        | RETIRED_FAMILIES
                    ),
                    enable_synthetic_recipes=alpha_foundry_config.enable_synthetic_recipes,
                )
                panels_by_tf[htf] = htf_native_panels
                recipes_by_tf[htf] = htf_recipes
                bindings_by_tf[htf] = htf_bindings

            _run_logger.debug(
                "[SYS] stage=panel_construction took=%.4fs rss=%.0fMB n_tfs=%d",
                _time_panel.perf_counter() - _t_panel_construct,
                _get_rss_mb(),
                len(panels_by_tf),
            )
            full_panels_by_tf = dict(panels_by_tf)
            _af_run_id = f"{tf}_{int(_time_panel.time())}"
            _t_l0_gate = _time_panel.perf_counter()
            # The multi-TF panel map is still resident at this point. Forking
            # the canonical L0 gate can dirty those inherited grids in every
            # child and multiply the parent high-water mark beyond the 18 GiB
            # host budget. Keep this gate serial until the two-pass streaming
            # builder owns/release panels one TF at a time.
            _requested_l0_workers = getattr(alpha_foundry_config, "l0_parallel_max_workers", None)
            _workers = 1
            _run_logger.debug(
                "[SYS] stage=l0_gate_worker_plan requested=%s workers=%d reason=all_tf_panels_resident",
                _requested_l0_workers,
                _workers,
            )
            multi_results = run_alpha_foundry_l0_gate_multi_tf(
                panels_by_tf=panels_by_tf,
                bindings_by_tf=bindings_by_tf,
                recipes_by_tf=recipes_by_tf,
                aligned_by_tf=aligned_by_tf,
                cost_model=ExecutionCostModel(),
                runtime_config=alpha_foundry_config,
                run_id_prefix=_af_run_id,
                parallel_max_workers=_workers,
                evidence_end_ns=l0_evidence_end_ns,
            )
            _run_logger.debug(
                "[SYS] stage=l0_gate_multi_tf_wall took=%.4fs rss=%.0fMB parallel_max_workers=%d",
                _time_panel.perf_counter() - _t_l0_gate,
                _get_rss_mb(),
                _workers,
            )

            _run_logger.debug("[SYS] stage=l0_manifest_pre rss=%.0fMB", _get_rss_mb())

            # [ADR_20260712_L0_EVIDENCE_CONDITIONED_CROSS_TF_ADMISSION] Cross-TF
            # evidence-conditioned pruning. canonical_tf is resolved internally
            # from candidates; removed from the call. Added new evidence thresholds.
            pruned_multi_results, _l0_manifest = assemble_l0_strategy_delivery_manifest(
                multi_results=multi_results,
                aligned_by_tf=aligned_by_tf,
                run_id_prefix=_af_run_id,
                enable_audit=getattr(alpha_foundry_config, "enable_cross_tf_diversity_audit", False),
                enable_pruning=getattr(alpha_foundry_config, "enable_cross_tf_pruning", False),
                evidence_end_ns=l0_evidence_end_ns or 0,
                total_l1_verification_budget=getattr(alpha_foundry_config, "total_l1_verification_budget", 30),
                min_survivors_per_archetype=getattr(
                    alpha_foundry_config, "cross_tf_pruning_min_survivors_per_archetype", 1
                ),
                min_survivors_per_tf=getattr(alpha_foundry_config, "cross_tf_pruning_min_survivors_per_tf", 0),
                min_common_active_bars=getattr(alpha_foundry_config, "cross_tf_min_common_active_bars", 480),
                min_directional_entry_jaccard=getattr(
                    alpha_foundry_config, "cross_tf_min_directional_entry_jaccard", 0.50
                ),
                min_shared_directional_entries=getattr(
                    alpha_foundry_config, "cross_tf_min_shared_directional_entries", 12
                ),
            )
            _run_logger.debug("[SYS] stage=l0_manifest_post rss=%.0fMB", _get_rss_mb())
            for _tf_k, _res in pruned_multi_results.items():
                _full_by_recipe = {
                    str(getattr(_binding, "recipe_id", "")): replace(
                        full_panels_by_tf[_tf_k][_binding.panel_index],
                        metadata={
                            **dict(getattr(full_panels_by_tf[_tf_k][_binding.panel_index], "metadata", {}) or {}),
                            "recipe_id": str(getattr(_binding, "recipe_id", "")),
                        },
                    )
                    for _binding in bindings_by_tf.get(_tf_k, ())
                    if 0 <= int(getattr(_binding, "panel_index", -1)) < len(full_panels_by_tf[_tf_k])
                }
                _selected_ids = tuple(
                    str(getattr(_candidate, "recipe_id", "")) for _candidate in getattr(_res, "candidates_for_l1", ())
                )
                pruned_multi_results[_tf_k] = replace(
                    _res,
                    panels_for_l1=tuple(
                        _full_by_recipe[_recipe_id] for _recipe_id in _selected_ids if _recipe_id in _full_by_recipe
                    ),
                )
            _run_logger.debug("[SYS] stage=l0_full_panel_rebind_post rss=%.0fMB", _get_rss_mb())
            l0_delivery_manifest = _l0_manifest
            # [LIMIT-01] Build per-TF native labeled events from the recipe-stamped,
            # L0-admitted panel set (panels_for_l1), now that binding has completed.
            for _tf_k, _res in pruned_multi_results.items():
                _label_panels_for_tf_native(_tf_k, getattr(_res, "panels_for_l1", ()))
            _run_logger.debug("[SYS] stage=l0_native_label_post rss=%.0fMB", _get_rss_mb())
            multi_results = pruned_multi_results
            if _l0_manifest.independence_audit is not None:
                _ia = _l0_manifest.independence_audit
                _run_logger.info(
                    "[EVAL] stage=l0_cross_tf_independence_audit run_id=%s canonical_tf=%s"
                    " n_selected_total=%d n_distinct_thesis_ids=%d n_independent_clusters=%d"
                    " n_demoted=%d pruning_applied=%s",
                    _af_run_id,
                    _ia.canonical_tf,
                    _ia.n_selected_total,
                    _ia.n_distinct_thesis_ids,
                    _ia.n_independent_clusters,
                    len(_ia.demoted_recipe_ids),
                    getattr(alpha_foundry_config, "enable_cross_tf_pruning", False),
                )

            base_result = multi_results[tf]
            panels_before_gate = panels
            panels = base_result.panels_for_l1
            alpha_foundry_report = base_result.report

            # Project gated HTF panels to base grid
            non_base_tfs = tuple(t for t in l1_tfs if t != tf)
            gated_htf: dict[str, tuple[Any, tuple[Any, ...]]] = {}
            for htf in non_base_tfs:
                if htf in multi_results and htf in aligned_by_tf:
                    htf_aligned = aligned_by_tf[htf]
                    if htf_aligned is not None:
                        gated_htf[htf] = (htf_aligned, multi_results[htf].panels_for_l1)
            _multi_tf_htf_panels = (
                project_htf_panels_to_base(
                    native_by_tf=gated_htf,
                    aligned_base=aligned,
                    base_tf=tf,
                )
                if gated_htf
                else ()
            )

            if getattr(alpha_foundry_config, "enable_correlation_audit", False):
                from src.domain.futures.alpha_foundry.diversity import audit_full_family_correlation

                active = aligned.active_mask & aligned.warm_mask & ~aligned.entry_block_mask & ~aligned.kill_mask
                corr_df = audit_full_family_correlation(
                    panels=panels_before_gate,
                    active_mask=active,
                    run_id=_af_run_id,
                    timeframe=tf,
                )
                report_dir = getattr(alpha_foundry_config, "report_dir", Path("logs/futures/alpha_foundry"))
                report_dir = Path(report_dir) if isinstance(report_dir, str) else report_dir
                report_dir.mkdir(parents=True, exist_ok=True)
                corr_df.to_parquet(str(report_dir / f"{_af_run_id}_family_correlation.parquet"))

            panels_by_tf.clear()
            full_panels_by_tf.clear()
            gated_htf.clear()
            multi_results.clear()
            gc.collect()
            _run_logger.debug(
                "[SYS] stage=l0_release_rejected_panel_maps rss=%.0fMB",
                _get_rss_mb(),
            )

            bridge_prof["alpha_foundry"] = time.perf_counter() - t_step
            _sample_rss("alpha_foundry")
        else:
            # ── Single-TF L0 gate path (original) ────────────────────────────
            _af_run_id = f"{tf}_{int(time.time())}"
            _single_l0_panels = (
                build_causal_l0_panel_views(
                    panels=panels,
                    evidence_end_ns=l0_evidence_end_ns,
                )
                if l0_evidence_end_ns is not None
                else panels
            )
            af_result = run_alpha_foundry_l0_gate(
                panels=_single_l0_panels,
                bindings=bindings,
                recipes=recipes,
                aligned=aligned,
                cost_model=ExecutionCostModel(),
                runtime_config=alpha_foundry_config,
                run_id=_af_run_id,
                timeframe=tf,
            )
            panels_before_gate = panels
            _full_by_recipe = {
                str(getattr(_binding, "recipe_id", "")): replace(
                    panels[_binding.panel_index],
                    metadata={
                        **dict(getattr(panels[_binding.panel_index], "metadata", {}) or {}),
                        "recipe_id": str(getattr(_binding, "recipe_id", "")),
                    },
                )
                for _binding in bindings
                if 0 <= int(getattr(_binding, "panel_index", -1)) < len(panels)
            }
            _selected_ids = tuple(
                str(getattr(_candidate, "recipe_id", "")) for _candidate in getattr(af_result, "candidates_for_l1", ())
            )
            panels = tuple(_full_by_recipe[_recipe_id] for _recipe_id in _selected_ids if _recipe_id in _full_by_recipe)
            alpha_foundry_report = af_result.report
            _, l0_delivery_manifest = assemble_l0_strategy_delivery_manifest(
                multi_results={tf: af_result},
                aligned_by_tf={tf: aligned},
                run_id_prefix=_af_run_id,
                enable_audit=False,
                enable_pruning=False,
                evidence_end_ns=l0_evidence_end_ns or 0,
                total_l1_verification_budget=getattr(alpha_foundry_config, "total_l1_verification_budget", 30),
            )

            # [ADR_20260707_L0_ALPHA_EFFECTIVENESS_REDESIGN] opt-in pre-gate family correlation audit
            if getattr(alpha_foundry_config, "enable_correlation_audit", False):
                from src.domain.futures.alpha_foundry.diversity import audit_full_family_correlation

                active = aligned.active_mask & aligned.warm_mask & ~aligned.entry_block_mask & ~aligned.kill_mask
                corr_df = audit_full_family_correlation(
                    panels=panels_before_gate,
                    active_mask=active,
                    run_id=_af_run_id,
                    timeframe=tf,
                )
                report_dir = getattr(alpha_foundry_config, "report_dir", Path("logs/futures/alpha_foundry"))
                report_dir = Path(report_dir) if isinstance(report_dir, str) else report_dir
                report_dir.mkdir(parents=True, exist_ok=True)
                corr_df.to_parquet(str(report_dir / f"{_af_run_id}_family_correlation.parquet"))

            bridge_prof["alpha_foundry"] = time.perf_counter() - t_step
            _sample_rss("alpha_foundry")
    t_step = time.perf_counter()
    raw_events = candidate_panels_to_events(
        panels,
        min_abs_score=strategy_cfg.candidate.min_rule_net_bps * 1e-4,
        side_flip_variants=strategy_cfg.candidate.side_flip_candidate_variants,
        cost_floor_bps=strategy_cfg.candidate.cost_floor_bps,
        execution_cost_bps_2d=aligned.execution_cost_bps_2d,
    )
    bridge_prof["events"] = time.perf_counter() - t_step
    _sample_rss("events")
    max_holding_bars = (
        int(pd.to_numeric(raw_events["expected_holding_bars"], errors="coerce").max())
        if not raw_events.empty and "expected_holding_bars" in raw_events.columns
        else None
    )
    candidate_cfg = with_max_holding_bars(
        strategy_cfg.candidate,
        max_holding_bars=max_holding_bars,
    )
    purge_bars, embargo_bars = resolve_purge_and_embargo_bars(candidate_cfg)

    if raw_events.empty:
        t_step = time.perf_counter()
        alpha_panel = build_candidate_alpha_panel(
            selected_events=raw_events,
            target_weights_2d=np.zeros_like(aligned.close_2d),
            datetimes=aligned.datetimes,
            symbols=tuple(symbols),
            cfg=strategy_cfg.candidate,
        )
        bridge_prof["alpha_panel"] = time.perf_counter() - t_step
        _sample_rss("alpha_panel")

        # [ADR_20260714_L1_ZERO_SIGNAL_REGRESSION] base TF raw_events being
        # empty must not discard already-computed cross-TF HTF panels.
        htf_labeled_all = pd.DataFrame()
        if _multi_tf_htf_panels:
            try:
                htf_raw_events = candidate_panels_to_events(
                    _multi_tf_htf_panels,
                    min_abs_score=candidate_cfg.min_rule_net_bps * 1e-4,
                    side_flip_variants=candidate_cfg.side_flip_candidate_variants,
                    cost_floor_bps=candidate_cfg.cost_floor_bps,
                    execution_cost_bps_2d=aligned.execution_cost_bps_2d,
                )
                if not htf_raw_events.empty:
                    atr_2d_cache = _compute_yang_zhang_vol_2d(aligned)
                    htf_labeled = label_candidate_events(
                        events=htf_raw_events,
                        aligned=aligned,
                        cfg=candidate_cfg,
                        precomputed_atr_2d=atr_2d_cache,
                    )
                    _htf_variant_to_tf = {
                        panel.variant: panel.metadata.get("native_tf", tf) for panel in _multi_tf_htf_panels
                    }
                    htf_labeled["native_tf"] = htf_labeled["variant"].map(_htf_variant_to_tf)
                    htf_labeled_all = htf_labeled
            except Exception as exc:
                _logger.warning("[MULTI-TF] HTF-only labeling failed on empty-base-events path: %s", exc)
                htf_labeled_all = pd.DataFrame()

        _emit_bridge_profile()
        return _build_output(
            alpha_panel=alpha_panel,
            target_weights=np.zeros_like(aligned.close_2d),
            labeled=pd.DataFrame(),
            labeled_unfiltered=htf_labeled_all,
            rule_report={
                "events_total": 0,
                "labeled_total": len(htf_labeled_all),
                "promoted_total": 0,
                "fit_total": 0,
                "calibration_total": 0,
                "oos_total": 0,
                "selected_pre_group": 0,
                "selected_total": 0,
                "eligible": 0,
                "n_keep": 0,
                "policy": candidate_cfg.selection_policy,
                "zero_reason": "no_events" if htf_labeled_all.empty else "base_tf_only_cross_tf_delivered",
                "gate_calibration_used": False,
                "gate_calibration_reason": "no_events",
                "recommended_keep_variants": (),
                "recommended_flip_variants": (),
                "recommended_keep_signal_cells": (),
                "recommended_flip_signal_cells": (),
            },
            alpha_foundry_report=alpha_foundry_report,
            l0_delivery_manifest=l0_delivery_manifest,
        )

    atr_2d_cache = _compute_yang_zhang_vol_2d(aligned)

    t_step = time.perf_counter()
    labeled = label_candidate_events(
        events=raw_events,
        aligned=aligned,
        cfg=candidate_cfg,
        precomputed_atr_2d=atr_2d_cache,
    )
    bridge_prof["label"] = time.perf_counter() - t_step
    _sample_rss("label")
    labeled_all = labeled.assign(native_tf=tf)
    # ── Multi-TF HTF panel generation (Phase B) ──────────────────────────
    htf_tfs = tuple(t for t in getattr(candidate_cfg, "l1_tfs", ()) if t != tf)
    if htf_tfs:
        t_htf = time.perf_counter()
        try:
            from src.domain.futures.strategy.config import resolve_tf_signal_pool

            if _multi_tf_htf_panels is not None:
                htf_panels = _multi_tf_htf_panels
            else:
                htf_panels = build_multi_tf_panels(
                    data_maps=preloaded_data_maps,
                    symbols=symbols,
                    aligned_base=aligned,
                    base_cfg=candidate_cfg,
                    base_tf=tf,
                    tfs=candidate_cfg.l1_tfs,
                    family_pool=lambda t: resolve_tf_signal_pool(candidate_cfg, t),
                    htf_only=False,
                )
            bridge_prof["htf_panels"] = time.perf_counter() - t_htf
            _sample_rss("htf_panels")
            if htf_panels:
                t_htf_events = time.perf_counter()
                htf_raw_events = candidate_panels_to_events(
                    htf_panels,
                    min_abs_score=candidate_cfg.min_rule_net_bps * 1e-4,
                    side_flip_variants=candidate_cfg.side_flip_candidate_variants,
                    cost_floor_bps=candidate_cfg.cost_floor_bps,
                    execution_cost_bps_2d=aligned.execution_cost_bps_2d,
                )
                if not htf_raw_events.empty:
                    t_htf_label = time.perf_counter()
                    htf_labeled = label_candidate_events(
                        events=htf_raw_events,
                        aligned=aligned,
                        cfg=candidate_cfg,
                        precomputed_atr_2d=atr_2d_cache,
                    )
                    bridge_prof["htf_label"] = time.perf_counter() - t_htf_label
                    _sample_rss("htf_label")
                    variant_to_tf: dict[str, str] = {}
                    for panel in htf_panels:
                        variant_to_tf[panel.variant] = panel.metadata.get("native_tf", tf)
                    htf_labeled["native_tf"] = htf_labeled["variant"].map(variant_to_tf)
                    labeled_all = pd.concat([labeled_all, htf_labeled], ignore_index=True)
                bridge_prof["htf_events"] = time.perf_counter() - t_htf_events
                _sample_rss("htf_events")
        except Exception as exc:
            _logger.warning("[MULTI-TF] HTF panel generation failed: %s", exc)
    fit_start, fit_end, calibration_start, calibration_end, oos_start, oos_end = _candidate_ml_split_indices(
        n_bars=n_bars,
        fit_fraction=candidate_cfg.ml_fit_fraction,
        calibration_fraction=candidate_cfg.ml_calibration_fraction,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )
    recommendation_start, recommendation_end = _recommendation_window_indices(
        fit_start=fit_start,
        fit_end=fit_end,
        calibration_start=calibration_start,
        calibration_end=calibration_end,
        cfg=strategy_cfg.candidate,
    )
    if strategy_cfg.candidate.promotion_decision_split == "fit_calibration":
        entry_idx = pd.to_numeric(labeled["entry_idx"], errors="coerce")
        labeled_for_diag = labeled.loc[(entry_idx < fit_end) | (entry_idx >= calibration_start)].copy()
    else:
        labeled_for_diag = labeled

    if strategy_cfg.candidate.signal_only:
        diag = _empty_rule_diagnostics()
        bridge_prof["diagnostics"] = 0.0
        _logger.debug("[BRIDGE] signal_only=True: skipped compute_rule_diagnostics")
    else:
        t_step = time.perf_counter()
        diag = compute_rule_diagnostics(
            labeled_events=labeled_for_diag,
            aligned=aligned,
            cfg=strategy_cfg.candidate,
            min_obs=max(strategy_cfg.candidate.min_candidate_obs, 100),
            silent=silent,
            recommendation_start=recommendation_start,
            recommendation_end=recommendation_end,
            report_start=oos_start,
            report_end=oos_end,
        )
        bridge_prof["diagnostics"] = time.perf_counter() - t_step
    _sample_rss("diagnostics")
    gc.collect()

    if strategy_cfg.candidate.promotion_filter_enabled and not strategy_cfg.candidate.signal_only:
        t_step = time.perf_counter()
        labeled = apply_variant_promotions(
            labeled=labeled,
            keep_variants=diag.recommended_keep_variants,
            flip_variants=diag.recommended_flip_variants,
            keep_signal_cells=diag.recommended_keep_signal_cells,
            flip_signal_cells=diag.recommended_flip_signal_cells,
        )
        bridge_prof["promotions"] = time.perf_counter() - t_step
        _sample_rss("promotions")
        if labeled.empty:
            _logger.debug("[BRIDGE] all candidate variants blocked by promotion filter; producing zero weights")
            t_step = time.perf_counter()
            alpha_panel = build_candidate_alpha_panel(
                selected_events=pd.DataFrame(),
                target_weights_2d=np.zeros_like(aligned.close_2d),
                datetimes=aligned.datetimes,
                symbols=tuple(symbols),
                cfg=strategy_cfg.candidate,
            )
            bridge_prof["alpha_panel"] = time.perf_counter() - t_step
            _sample_rss("alpha_panel")
            _emit_bridge_profile()
            return _build_output(
                alpha_panel=alpha_panel,
                target_weights=np.zeros_like(aligned.close_2d),
                labeled=labeled,
                labeled_unfiltered=labeled_all,
                rule_report={
                    "events_total": len(raw_events),
                    "labeled_total": len(labeled),
                    "promoted_total": 0,
                    "fit_total": 0,
                    "calibration_total": 0,
                    "oos_total": 0,
                    "selected_pre_group": 0,
                    "selected_total": 0,
                    "eligible": 0,
                    "n_keep": 0,
                    "policy": strategy_cfg.candidate.selection_policy,
                    "zero_reason": "promotion_filter_empty",
                    "gate_calibration_used": False,
                    "gate_calibration_reason": "promotion_filter_empty",
                    "recommended_keep_variants": diag.recommended_keep_variants,
                    "recommended_flip_variants": diag.recommended_flip_variants,
                    "recommended_keep_signal_cells": diag.recommended_keep_signal_cells,
                    "recommended_flip_signal_cells": diag.recommended_flip_signal_cells,
                },
                alpha_foundry_report=alpha_foundry_report,
                l0_delivery_manifest=l0_delivery_manifest,
            )
    promoted_total = len(labeled)

    # Compute split indices needed for signal_only + WF (done once for OOS window bounds)
    if strategy_cfg.candidate.wf_enabled and strategy_cfg.candidate.wf_scheme != "single":
        _folds = build_walk_forward_folds(n_bars=n_bars, cfg=candidate_cfg, max_holding_bars=max_holding_bars)
        _oos_start_ref = _folds[0].oos_start if _folds else 0
        _oos_end_ref = _folds[-1].oos_end if _folds else n_bars
    else:
        _s = _candidate_ml_split_indices(
            n_bars=n_bars,
            fit_fraction=candidate_cfg.ml_fit_fraction,
            calibration_fraction=candidate_cfg.ml_calibration_fraction,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )
        _oos_start_ref, _oos_end_ref = _s[4], _s[5]
        _folds = None  # single-fold path handled below

    # signal_only: validate data integrity and skip ML training
    if strategy_cfg.candidate.signal_only:
        integrity_report = verify_data_integrity(aligned=aligned, symbols=symbols)
        any_passes = any(info["status"] == "PASS" for info in integrity_report.values())
        t_step = time.perf_counter()
        alpha_panel_sv = build_candidate_alpha_panel(
            selected_events=pd.DataFrame(),
            target_weights_2d=np.zeros_like(aligned.close_2d),
            datetimes=aligned.datetimes,
            symbols=tuple(symbols),
            cfg=strategy_cfg.candidate,
        )
        bridge_prof["alpha_panel"] = time.perf_counter() - t_step
        _sample_rss("alpha_panel")
        _emit_bridge_profile()
        return _build_output(
            alpha_panel=alpha_panel_sv,
            target_weights=np.zeros_like(aligned.close_2d),
            labeled=labeled,
            labeled_unfiltered=labeled_all,
            oos_start=_oos_start_ref,
            rule_report={
                "events_total": len(raw_events),
                "labeled_total": len(labeled),
                "promoted_total": promoted_total,
                "fit_total": 0,
                "calibration_total": 0,
                "oos_total": 0,
                "selected_pre_group": 0,
                "selected_total": 0,
                "eligible": 0,
                "n_keep": 0,
                "policy": strategy_cfg.candidate.selection_policy,
                "zero_reason": "signal_only_mode",
                "gate_calibration_used": False,
                "gate_calibration_reason": "signal_only_mode",
                "recommended_keep_variants": diag.recommended_keep_variants,
                "recommended_flip_variants": diag.recommended_flip_variants,
                "failure_report": diag.recommendation_failure_report,
                "signal_validation": [
                    {
                        "symbol": sym,
                        "status": info["status"],
                        "nan_pct": info["nan_pct"],
                        "zero_neg_pct": info["zero_neg_pct"],
                        "close_std": info["close_std"],
                        "hi_lo_violation": info["hi_lo_violation"],
                        "fail_reasons": list(info["reasons"]),
                    }
                    for sym, info in integrity_report.items()
                ],
                "signal_validation_pass": any_passes,
            },
            alpha_foundry_report=alpha_foundry_report,
            l0_delivery_manifest=l0_delivery_manifest,
        )

    # Build WF folds (multi-fold or single)
    wf_folds = (
        build_walk_forward_folds(n_bars=n_bars, cfg=candidate_cfg, max_holding_bars=max_holding_bars)
        if _folds is None
        else _folds
    )

    # --- WF fold loop: train per fold using shared workflow ---
    from src.domain.futures.strategy.candidate_workflow import run_candidate_walk_forward

    t_step = time.perf_counter()
    prepared = prepare_labeled_events(
        labeled_events=labeled,
        aligned=aligned,
        cfg=candidate_cfg,
        fit_start_idx=wf_folds[0].fit_start if wf_folds else 0,
        fit_end_idx=wf_folds[-1].fit_end if wf_folds else n_bars,
    )
    wf_outputs = run_candidate_walk_forward(
        labeled_events=prepared,
        aligned=aligned,
        cfg=candidate_cfg,
        folds=wf_folds,
    )
    bridge_prof["walk_forward"] = time.perf_counter() - t_step
    _sample_rss("walk_forward")
    t_post_wf = time.perf_counter()

    fold_p_pass_parts: list[np.ndarray] = []
    fold_mu_parts: list[np.ndarray] = []
    fold_q10_parts: list[np.ndarray] = []
    fold_q90_parts: list[np.ndarray] = []
    fold_utility_parts: list[np.ndarray] = []
    fold_expected_return_r_parts: list[np.ndarray] = []
    fold_q10_return_r_parts: list[np.ndarray] = []
    fold_q90_return_r_parts: list[np.ndarray] = []
    fold_kelly_fraction_parts: list[np.ndarray] = []
    fold_event_parts: list[Any] = []
    fold_gate_model = None  # We don't expose private models directly anymore, fallback used below
    fold_edge_models = None
    fold_calibration_used = False
    fold_calibration_reason = "not_fit"
    total_fit = total_cal = total_oos = 0
    fold_cost_survival: list[bool] = []
    fold_selection_reports: list[dict[str, Any]] = []

    wf_fold_details: list[dict[str, Any]] = []

    # Map validation status from workflow outputs
    for fold_out in wf_outputs:
        fold = wf_folds[fold_out.fold_id]
        ml_out = fold_out.model_output
        selected_fold = fold_out.selected_events
        selection_diag_fold = dict(getattr(selected_fold, "attrs", {}).get("candidate_selection_diagnostics", {}))
        t_fold_start = time.perf_counter()

        if "edge_after_hurdle_bps" in selected_fold.columns:
            realized_edge = pd.to_numeric(selected_fold["edge_after_hurdle_bps"], errors="coerce")
        else:
            realized_edge = pd.Series(dtype="float64")
        selected_count = int(selected_fold.shape[0])
        realized_mean = float(realized_edge.mean()) if realized_edge.notna().any() else float("nan")
        realized_hit_rate = float((realized_edge > 0.0).mean()) if realized_edge.notna().any() else 0.0

        if realized_edge.notna().any():
            log_growth_proxy = float(
                np.mean(np.log1p(np.clip(realized_edge.to_numpy(dtype=np.float64, copy=False) * 1e-4, -0.99, None)))
            )
        else:
            log_growth_proxy = float("-inf")

        # Default values for lift variables (only computed in realized_selected_edge branch)
        ml_lift_bps: float = float("nan")
        pass_lift: bool = False

        survival_metric = strategy_cfg.candidate.fold_survival_metric
        if survival_metric == "predicted_mu_tstat":
            fold_mu_finite = ml_out.mu_net_decision_bps[np.isfinite(ml_out.mu_net_decision_bps)]
            if fold_mu_finite.size >= 10:
                mean_edge = float(np.mean(fold_mu_finite))
                std_edge = float(np.std(fold_mu_finite)) + 1e-12
                n_obs = fold_mu_finite.size
                t_stat = mean_edge / (std_edge / np.sqrt(n_obs))
                pass_survival = bool(mean_edge > 0.0 and t_stat > 1.645)
                survival_reason = "predicted_mu_tstat_pass" if pass_survival else "predicted_mu_tstat_fail"
            else:
                pass_survival = False
                survival_reason = "predicted_mu_tstat_insufficient_obs"
        elif survival_metric == "realized_log_growth":
            pass_survival = (
                selected_count >= strategy_cfg.candidate.min_fold_selected_events
                and np.isfinite(log_growth_proxy)
                and log_growth_proxy >= strategy_cfg.candidate.min_fold_log_growth
            )
            survival_reason = "realized_log_growth_pass" if pass_survival else "realized_log_growth_fail"
        else:
            # Compute ML selection lift: mean(selected_edge) - mean(all_fold_oos_edge)
            fold_oos_events = (
                labeled[(labeled["entry_idx"] >= fold.oos_start) & (labeled["entry_idx"] < fold.oos_end)]
                if "entry_idx" in labeled.columns
                else pd.DataFrame()
            )
            if not fold_oos_events.empty and "edge_after_hurdle_bps" in fold_oos_events.columns:
                baseline_mean = float(pd.to_numeric(fold_oos_events["edge_after_hurdle_bps"], errors="coerce").mean())
            else:
                baseline_mean = float("nan")
            ml_lift_bps = (
                (realized_mean - baseline_mean)
                if np.isfinite(realized_mean) and np.isfinite(baseline_mean)
                else float("nan")
            )
            pass_lift = bool(np.isfinite(ml_lift_bps) and ml_lift_bps > 0.0)
            pass_survival = (
                selected_count >= strategy_cfg.candidate.min_fold_selected_events
                and np.isfinite(realized_mean)
                and realized_mean >= strategy_cfg.candidate.min_fold_realized_edge_bps
                and pass_lift
            )
            survival_reason = "realized_selected_edge_pass" if pass_survival else "realized_selected_edge_fail"

        fold_cost_survival.append(bool(pass_survival))
        fold_selection_reports.append(
            {
                "eligible": int(selection_diag_fold.get("eligible", 0)),
                "selected_total": selected_count,
                "realized_mean_bps": realized_mean,
                "realized_status": "empty" if selected_count == 0 else "observed",
                "log_growth_proxy": log_growth_proxy,
                "waterfall_expected_utility_adj_p90_bps": selection_diag_fold.get(
                    "waterfall_expected_utility_adj_p90_bps",
                    float("nan"),
                ),
                "waterfall_downside_drag_p90_bps": selection_diag_fold.get(
                    "waterfall_downside_drag_p90_bps",
                    float("nan"),
                ),
                "waterfall_breakeven_floor_bps": selection_diag_fold.get(
                    "waterfall_breakeven_floor_bps",
                    float("nan"),
                ),
                "shadow_profile_count": int(selection_diag_fold.get("shadow_profile_count", 0)),
                "shadow_max_selected_total": int(selection_diag_fold.get("shadow_max_selected_total", 0)),
                "shadow_max_eligible": int(selection_diag_fold.get("shadow_max_eligible", 0)),
            }
        )

        # Collect for summary table
        _vdiag = getattr(ml_out, "validation_diagnostics", {}) or {}
        _edge_rep = fold_out.edge_report
        _mode = str(_vdiag.get("prediction_mode", "n/a"))
        _rank_ic_val = (
            _vdiag.get("oos_rank_ic", _edge_rep.prior_rank_ic)
            if _mode == "ensemble_b0"
            else (_edge_rep.residual_rank_ic if _mode == "prior_residual" else _edge_rep.prior_rank_ic)
        )
        wf_fold_details.append(
            {
                "fold_id": len(fold_selection_reports),
                "inference_mode": _mode,
                "rank_ic": float(_rank_ic_val),
                "n_events": int(ml_out.events.shape[0]) if ml_out.events is not None else 0,
                "prior_bps": float(_vdiag.get("prior_component_p90_bps", 0.0)),
                "eu_p90": float(selection_diag_fold.get("waterfall_expected_utility_adj_p90_bps", 0.0)),
                "pass_cost": bool(pass_survival),
                "realized_mean_bps": float(realized_mean),  # actual pass gate: >= min_fold_realized_edge_bps
                "selected_total": int(selected_count),  # actual pass gate: >= min_fold_selected_events
            }
        )

        _logger.debug(
            (
                "[BRIDGE][WF_REALIZED] metric=%s oos=[%d,%d) selected=%d realized_mean=%.3f "
                "status=%s hit_rate=%.3f log_growth=%.6f lift=%.3f pass_lift=%s pass=%s reason=%s"
            ),
            survival_metric,
            fold.oos_start,
            fold.oos_end,
            selected_count,
            realized_mean,
            "empty" if selected_count == 0 else "observed",
            realized_hit_rate,
            log_growth_proxy,
            ml_lift_bps if survival_metric not in ("predicted_mu_tstat", "realized_log_growth") else float("nan"),
            pass_lift if survival_metric not in ("predicted_mu_tstat", "realized_log_growth") else False,
            pass_survival,
            survival_reason,
        )
        _logger.debug(
            (
                "[BRIDGE][WF_DIAG] fold=%d eligible=%s selected=%d eu_p90=%.3f downside_p90=%.3f "
                "breakeven=%.1f shadow_profiles=%d shadow_max_selected=%d shadow_max_eligible=%d"
            ),
            len(fold_selection_reports),
            selection_diag_fold.get("eligible", 0),
            selected_count,
            float(selection_diag_fold.get("waterfall_expected_utility_adj_p90_bps", float("nan"))),
            float(selection_diag_fold.get("waterfall_downside_drag_p90_bps", float("nan"))),
            float(selection_diag_fold.get("waterfall_breakeven_floor_bps", float("nan"))),
            int(selection_diag_fold.get("shadow_profile_count", 0)),
            int(selection_diag_fold.get("shadow_max_selected_total", 0)),
            int(selection_diag_fold.get("shadow_max_eligible", 0)),
        )

        _n_fold = int(ml_out.expected_net_bps.shape[0])
        if pass_survival:
            fold_p_pass_parts.append(ml_out.p_pass)
            fold_mu_parts.append(ml_out.expected_net_bps)
            fold_q10_parts.append(ml_out.q10_net_bps)
            fold_q90_parts.append(ml_out.q90_net_bps)
            fold_utility_parts.append(ml_out.selection_score)
            fold_expected_return_r_parts.append(ml_out.expected_return_r)
            fold_q10_return_r_parts.append(ml_out.q10_return_r)
            fold_q90_return_r_parts.append(ml_out.q90_return_r)
            fold_kelly_fraction_parts.append(ml_out.kelly_fraction)
        else:
            _zeros = np.zeros(_n_fold, dtype=np.float64)
            fold_p_pass_parts.append(_zeros)
            fold_mu_parts.append(_zeros.copy())
            fold_q10_parts.append(_zeros.copy())
            fold_q90_parts.append(_zeros.copy())
            fold_utility_parts.append(_zeros.copy())
            fold_expected_return_r_parts.append(_zeros.copy())
            fold_q10_return_r_parts.append(_zeros.copy())
            fold_q90_return_r_parts.append(_zeros.copy())
            fold_kelly_fraction_parts.append(_zeros.copy())
        fold_event_parts.append(ml_out.events)
        wf_fold_times.append(time.perf_counter() - t_fold_start)
        wf_fold_n_events.append(int(ml_out.events.shape[0]) if ml_out.events is not None else -1)

    # Note: fallback behavior down below is preserved using oos_set_fallback
    # Reconstruct counts from fold outputs
    p_pass = np.concatenate(fold_p_pass_parts) if fold_p_pass_parts else np.array([], dtype=np.float64)
    _combined_mu = np.concatenate(fold_mu_parts) if fold_mu_parts else np.array([], dtype=np.float64)
    _combined_q10 = np.concatenate(fold_q10_parts) if fold_q10_parts else np.array([], dtype=np.float64)
    _combined_q90 = np.concatenate(fold_q90_parts) if fold_q90_parts else np.array([], dtype=np.float64)
    _combined_utility = np.concatenate(fold_utility_parts) if fold_utility_parts else np.array([], dtype=np.float64)
    _combined_expected_return_r = (
        np.concatenate(fold_expected_return_r_parts) if fold_expected_return_r_parts else np.array([], dtype=np.float64)
    )
    _combined_q10_return_r = (
        np.concatenate(fold_q10_return_r_parts) if fold_q10_return_r_parts else np.array([], dtype=np.float64)
    )
    _combined_q90_return_r = (
        np.concatenate(fold_q90_return_r_parts) if fold_q90_return_r_parts else np.array([], dtype=np.float64)
    )
    _combined_kelly_fraction = (
        np.concatenate(fold_kelly_fraction_parts) if fold_kelly_fraction_parts else np.array([], dtype=np.float64)
    )
    # Combine fold OOS outputs (time-ordered concat)
    if fold_event_parts:
        combined_events = (
            pd.concat(fold_event_parts, ignore_index=True) if len(fold_event_parts) > 1 else fold_event_parts[0]
        )
        p_pass = np.concatenate(fold_p_pass_parts) if fold_p_pass_parts else np.array([], dtype=np.float64)
        _combined_mu = np.concatenate(fold_mu_parts) if fold_mu_parts else np.array([], dtype=np.float64)
        _combined_q10 = np.concatenate(fold_q10_parts) if fold_q10_parts else np.array([], dtype=np.float64)
        _combined_q90 = np.concatenate(fold_q90_parts) if fold_q90_parts else np.array([], dtype=np.float64)
        _combined_utility = np.concatenate(fold_utility_parts) if fold_utility_parts else np.array([], dtype=np.float64)
        _combined_expected_return_r = (
            np.concatenate(fold_expected_return_r_parts)
            if fold_expected_return_r_parts
            else np.array([], dtype=np.float64)
        )
        _combined_q10_return_r = (
            np.concatenate(fold_q10_return_r_parts) if fold_q10_return_r_parts else np.array([], dtype=np.float64)
        )
        _combined_q90_return_r = (
            np.concatenate(fold_q90_return_r_parts) if fold_q90_return_r_parts else np.array([], dtype=np.float64)
        )
        _combined_kelly_fraction = (
            np.concatenate(fold_kelly_fraction_parts) if fold_kelly_fraction_parts else np.array([], dtype=np.float64)
        )
    else:
        # Fallback: use last fold's full OOS as single-fold behavior
        oos_set_fallback = build_candidate_dataset(
            labeled_events=labeled,
            aligned=aligned,
            cfg=strategy_cfg.candidate,
            split_start=_oos_start_ref,
            split_end=_oos_end_ref,
        )
        combined_events = oos_set_fallback.event_index
        # Note: We need a trained gate model for fallback prediction. We can extract it
        # from wf_outputs if available. But fold_gate_model is now managed by the workflow outputs.
        # We fallback to the last output fold's models if available.
        # However, if wf_outputs is empty, we create degenerate predictions.
        if len(wf_outputs) > 0:
            # Re-run inference using last fold as fallback reference.
            # (In practice, wf_outputs is rarely empty if n_bars is valid).
            p_pass = np.full(oos_set_fallback.X.shape[0] if oos_set_fallback.X is not None else 0, 0.5)
            _combined_mu = np.zeros_like(p_pass)
            _combined_q10 = np.zeros_like(p_pass)
            _combined_q90 = np.zeros_like(p_pass)
            _combined_utility = np.zeros_like(p_pass)
            _combined_expected_return_r = np.zeros_like(p_pass)
            _combined_q10_return_r = np.zeros_like(p_pass)
            _combined_q90_return_r = np.zeros_like(p_pass)
            _combined_kelly_fraction = np.zeros_like(p_pass)
        else:
            p_pass = np.array([], dtype=np.float64)
            _combined_mu = np.array([], dtype=np.float64)
            _combined_q10 = np.array([], dtype=np.float64)
            _combined_q90 = np.array([], dtype=np.float64)
            _combined_utility = np.array([], dtype=np.float64)
            _combined_expected_return_r = np.array([], dtype=np.float64)
            _combined_q10_return_r = np.array([], dtype=np.float64)
            _combined_q90_return_r = np.array([], dtype=np.float64)
            _combined_kelly_fraction = np.array([], dtype=np.float64)
        total_oos = int(oos_set_fallback.X.shape[0] if oos_set_fallback.X is not None else 0)

    # Use last fold's models for selection_thresholds (best fit available)
    _last_oos_set = build_candidate_dataset(
        labeled_events=labeled,
        aligned=aligned,
        cfg=strategy_cfg.candidate,
        split_start=wf_folds[-1].oos_start,
        split_end=wf_folds[-1].oos_end,
    )

    from src.domain.futures.strategy.candidate_contracts import (
        CandidateModelOutput,
        CandidateWorkflowStatus,
        EdgeSource,
    )

    validation = getattr(fold_gate_model, "validation", None) if fold_gate_model is not None else None
    gate_enabled = validation.enabled if validation is not None else False
    gate_threshold = validation.threshold if validation is not None else 0.5
    edge_source = (
        EdgeSource.PRIOR_RESIDUAL
        if fold_edge_models is not None and fold_edge_models.prediction_mode == "prior_residual"
        else EdgeSource.PRIOR_ONLY
    )

    if fold_gate_model is not None:
        p_pass_ref = predict_candidate_gate(model=fold_gate_model, dataset=_last_oos_set, cfg=strategy_cfg.candidate)
    else:
        p_pass_ref = np.zeros(_last_oos_set.X.shape[0] if _last_oos_set.X is not None else 0)

    _ref_ml = predict_candidate_edges(
        models=fold_edge_models,
        dataset=_last_oos_set,
        p_pass=p_pass_ref,
        cfg=strategy_cfg.candidate,
        gate_enabled=gate_enabled,
        gate_threshold=gate_threshold,
        edge_source=edge_source,
    )
    ml_out = CandidateModelOutput(
        events=combined_events,
        p_pass=p_pass.astype(np.float64, copy=False) if p_pass.size > 0 else p_pass,
        gate_enabled=gate_enabled,
        gate_threshold=gate_threshold,
        edge_source=edge_source,
        expected_return_r=_combined_expected_return_r,
        expected_net_bps=_combined_mu,
        q10_return_r=_combined_q10_return_r,
        q10_net_bps=_combined_q10,
        q90_return_r=_combined_q90_return_r,
        q90_net_bps=_combined_q90,
        selection_score=_combined_utility,
        kelly_fraction=_combined_kelly_fraction,
        validation_diagnostics=_ref_ml.validation_diagnostics,
    )
    ml_out = replace(ml_out, events=combined_events)

    # --- Cross-fold consistency gate (min_wf_fold_pass_ratio) ---
    _n_folds_total = len(fold_cost_survival)
    _n_folds_pass = sum(fold_cost_survival)
    _fold_pass_ratio = _n_folds_pass / max(_n_folds_total, 1)
    _logger.debug(
        "[BRIDGE][WF] fold_cost_survival=%s pass_ratio=%.2f min_required=%.2f",
        fold_cost_survival,
        _fold_pass_ratio,
        strategy_cfg.candidate.min_wf_fold_pass_ratio,
    )
    wf_selected_total = int(sum(int(r.get("selected_total", 0)) for r in fold_selection_reports))
    wf_eligible_total = int(sum(int(r.get("eligible", 0)) for r in fold_selection_reports))
    realized_mean_values = [
        float(r["realized_mean_bps"])
        for r in fold_selection_reports
        if np.isfinite(float(r.get("realized_mean_bps", float("nan"))))
    ]
    log_growth_values = [
        float(r["log_growth_proxy"])
        for r in fold_selection_reports
        if np.isfinite(float(r.get("log_growth_proxy", float("nan"))))
    ]
    wf_fold_realized_mean_bps = float(np.mean(realized_mean_values)) if realized_mean_values else float("nan")
    wf_fold_log_growth_mean = float(np.mean(log_growth_values)) if log_growth_values else float("nan")
    wf_waterfall_expected_utility_p90_bps = (
        float(
            np.mean(
                [
                    float(r["waterfall_expected_utility_adj_p90_bps"])
                    for r in fold_selection_reports
                    if np.isfinite(float(r.get("waterfall_expected_utility_adj_p90_bps", float("nan"))))
                ]
            )
        )
        if any(
            np.isfinite(float(r.get("waterfall_expected_utility_adj_p90_bps", float("nan"))))
            for r in fold_selection_reports
        )
        else float("nan")
    )
    wf_waterfall_downside_drag_p90_bps = (
        float(
            np.mean(
                [
                    float(r["waterfall_downside_drag_p90_bps"])
                    for r in fold_selection_reports
                    if np.isfinite(float(r.get("waterfall_downside_drag_p90_bps", float("nan"))))
                ]
            )
        )
        if any(
            np.isfinite(float(r.get("waterfall_downside_drag_p90_bps", float("nan")))) for r in fold_selection_reports
        )
        else float("nan")
    )
    wf_shadow_profile_count = max((int(r.get("shadow_profile_count", 0)) for r in fold_selection_reports), default=0)
    wf_shadow_max_selected_total = max(
        (int(r.get("shadow_max_selected_total", 0)) for r in fold_selection_reports),
        default=0,
    )
    wf_shadow_max_eligible = max(
        (int(r.get("shadow_max_eligible", 0)) for r in fold_selection_reports),
        default=0,
    )
    _fold_oos_boundaries = tuple((f.oos_start, f.oos_end) for f in wf_folds)
    last_fold_out = wf_outputs[-1] if wf_outputs else None
    if _fold_pass_ratio < strategy_cfg.candidate.min_wf_fold_pass_ratio:
        _logger.debug(
            "[BRIDGE][WF] fold_pass_ratio=%.2f < min_wf_fold_pass_ratio=%.2f → fail-closed",
            _fold_pass_ratio,
            strategy_cfg.candidate.min_wf_fold_pass_ratio,
        )
        bridge_prof["post_wf"] = time.perf_counter() - t_post_wf
        _sample_rss("post_wf")
        t_step = time.perf_counter()
        _wf_fail_panel = build_candidate_alpha_panel(
            selected_events=pd.DataFrame(),
            target_weights_2d=np.zeros_like(aligned.close_2d),
            datetimes=aligned.datetimes,
            symbols=tuple(symbols),
            cfg=strategy_cfg.candidate,
        )
        bridge_prof["alpha_panel"] = time.perf_counter() - t_step
        _sample_rss("alpha_panel")
        out = CandidatePipelineOutput(
            alpha_panel=_wf_fail_panel,
            target_weights=np.zeros_like(aligned.close_2d),
            rule_report={
                "events_total": len(raw_events),
                "labeled_total": len(labeled),
                "promoted_total": promoted_total,
                "fit_total": total_fit,
                "calibration_total": total_cal,
                "oos_total": total_oos,
                "selected_pre_group": 0,
                "selected_total": 0,
                "eligible": 0,
                "n_keep": 0,
                "policy": strategy_cfg.candidate.selection_policy,
                "zero_reason": "wf_fold_pass_ratio_fail",
                "workflow_status": CandidateWorkflowStatus.BLOCKED.value,
                "gate_calibration_used": fold_calibration_used,
                "gate_calibration_reason": fold_calibration_reason,
                "wf_fold_pass_ratio": _fold_pass_ratio,
                "wf_n_folds": _n_folds_total,
                "wf_scheme": strategy_cfg.candidate.wf_scheme,
                "wf_selected_total": wf_selected_total,
                "wf_eligible_total": wf_eligible_total,
                "wf_fold_realized_mean_bps": wf_fold_realized_mean_bps,
                "wf_fold_log_growth_mean": wf_fold_log_growth_mean,
                "wf_shadow_profile_count": wf_shadow_profile_count,
                "wf_shadow_max_selected_total": wf_shadow_max_selected_total,
                "wf_shadow_max_eligible": wf_shadow_max_eligible,
                "wf_waterfall_expected_utility_p90_bps": wf_waterfall_expected_utility_p90_bps,
                "wf_waterfall_downside_drag_p90_bps": wf_waterfall_downside_drag_p90_bps,
                "wf_fold_details": wf_fold_details,
                "recommended_keep_variants": diag.recommended_keep_variants,
                "recommended_flip_variants": diag.recommended_flip_variants,
                "recommended_keep_signal_cells": diag.recommended_keep_signal_cells,
                "recommended_flip_signal_cells": diag.recommended_flip_signal_cells,
                "failure_report": diag.recommendation_failure_report,
                "recommendation_basis": diag.recommendation_basis,
                "recommendation_start": int(diag.recommendation_split[0]),
                "recommendation_end": int(diag.recommendation_split[1]),
                "report_start": int(diag.report_split[0]),
                "report_end": int(diag.report_split[1]),
            },
            aligned=aligned,
            labeled=labeled,
            labeled_unfiltered=labeled_all,
            fit_set=last_fold_out.fit_set if last_fold_out else None,
            calibration_set=last_fold_out.calibration_set if last_fold_out else None,
            oos_set=last_fold_out.oos_set if last_fold_out else None,
            gate_model=last_fold_out.gate_model if last_fold_out else None,
            edge_models=last_fold_out.edge_models if last_fold_out else None,
            fit_start=wf_folds[0].fit_start,
            fit_end=wf_folds[-1].fit_end,
            calibration_start=wf_folds[0].cal_start,
            calibration_end=wf_folds[-1].cal_end,
            oos_start=wf_folds[0].oos_start,
            oos_end=wf_folds[-1].oos_end,
            fold_oos_boundaries=_fold_oos_boundaries,
            alpha_foundry_report=alpha_foundry_report,
            l0_delivery_manifest=l0_delivery_manifest,
        )
        _emit_bridge_profile()
        return out

    # For downstream logging, expose last-fold model state
    gate_model = fold_gate_model
    # Reconstruct oos_set for report counts (use combined)

    _logger.debug(
        ("[DIAG][PIPELINE] raw=%d labeled=%d promoted=%d fit=%d cal=%d oos=%d n_folds=%d wf_scheme=%s"),
        len(raw_events),
        len(labeled),
        promoted_total,
        total_fit,
        total_cal,
        total_oos,
        len(wf_folds),
        strategy_cfg.candidate.wf_scheme,
    )
    gate_summary = _finite_summary(p_pass)
    edge_summary = _finite_summary(ml_out.mu_net_decision_bps)
    q10_summary = _finite_summary(ml_out.q10_net_bps)
    utility_summary = _finite_summary(ml_out.utility_score)
    _logger.debug(
        (
            "[DIAG][PIPELINE_GATE] calibrated=%s reason=%s mean=%.4f median=%.4f p90=%.4f max=%.4f "
            "pct_ge40=%.3f pct_ge45=%.3f pct_ge50=%.3f pct_ge55=%.3f"
        ),
        bool(gate_model.calibration_used) if gate_model is not None else False,
        gate_model.calibration_reason if gate_model is not None else "not_fit",
        gate_summary["mean"],
        gate_summary["median"],
        gate_summary["p90"],
        gate_summary["max"],
        _threshold_rate(p_pass, 0.40),
        _threshold_rate(p_pass, 0.45),
        _threshold_rate(p_pass, 0.50),
        _threshold_rate(p_pass, 0.55),
    )
    _logger.debug(
        (
            "[DIAG][PIPELINE_EDGE] mu_mean=%.1f mu_median=%.1f mu_p90=%.1f mu_max=%.1f "
            "q10_mean=%.1f q10_p10=%.1f q10_median=%.1f q10_min=%.1f "
            "utility_mean=%.3f utility_median=%.3f utility_p90=%.3f utility_max=%.3f"
        ),
        edge_summary["mean"],
        edge_summary["median"],
        edge_summary["p90"],
        edge_summary["max"],
        q10_summary["mean"],
        q10_summary["p10"],
        q10_summary["median"],
        q10_summary["min"],
        utility_summary["mean"],
        utility_summary["median"],
        utility_summary["p90"],
        utility_summary["max"],
    )

    bridge_prof["post_wf"] = time.perf_counter() - t_post_wf
    _sample_rss("post_wf")
    t_step = time.perf_counter()
    selected = select_candidate_events_for_portfolio(model_output=ml_out, cfg=strategy_cfg.candidate)
    bridge_prof["selection"] = time.perf_counter() - t_step
    _sample_rss("selection")
    selection_diag = dict(getattr(selected, "attrs", {}).get("candidate_selection_diagnostics", {}))
    _logger.debug(
        (
            "[DIAG][PIPELINE_SELECT] policy=%s zero_reason=%s eligible=%s selected_pre_group=%s "
            "selected=%s n_keep=%s breakeven_floor=%.1f"
        ),
        selection_diag.get("policy", strategy_cfg.candidate.selection_policy),
        selection_diag.get("zero_reason", "unknown"),
        selection_diag.get("eligible", 0),
        selection_diag.get("selected_pre_group", 0),
        selection_diag.get("selected_total", len(selected)),
        selection_diag.get("n_keep", 0),
        float(selection_diag.get("breakeven_floor_bps", strategy_cfg.candidate.cost_floor_bps)),
    )
    _log_universe_volatility_deciles(
        events=combined_events,
        selected=selected,
        mu_net_decision_bps=ml_out.mu_net_decision_bps,
        q10_net_bps=ml_out.q10_net_bps,
    )
    t_step = time.perf_counter()
    target_weights = build_candidate_target_weights(
        selected_events=selected,
        close_2d=aligned.close_2d,
        symbols=tuple(symbols),
        beta_2d=None,
        sigma_3d=None,
        cfg=strategy_cfg.candidate,
    )
    bridge_prof["weights"] = time.perf_counter() - t_step
    _sample_rss("weights")
    t_step = time.perf_counter()
    alpha_panel = build_candidate_alpha_panel(
        selected_events=selected,
        target_weights_2d=target_weights,
        datetimes=aligned.datetimes,
        symbols=tuple(symbols),
        cfg=strategy_cfg.candidate,
    )
    bridge_prof["alpha_panel"] = time.perf_counter() - t_step
    _sample_rss("alpha_panel")

    if strategy_cfg.candidate.exit_policy_mode == "label_only":
        # label_only: suppress per-event TP/SL; engine uses global ATR_MULT only
        _logger.debug("[BRIDGE] exit_policy_mode=label_only; zeroing per-event TP/SL columns")
        alpha_panel = alpha_panel.copy()
        alpha_panel["candidate_stop_atr_mult"] = 0.0
        alpha_panel["candidate_take_profit_atr_mult"] = 0.0

    _emit_bridge_profile()

    return _build_output(
        alpha_panel=alpha_panel,
        target_weights=target_weights,
        rule_report={
            "events_total": len(raw_events),
            "labeled_total": len(labeled),
            "promoted_total": promoted_total,
            "fit_total": total_fit,
            "calibration_total": total_cal,
            "oos_total": total_oos,
            "fit_start": wf_folds[0].fit_start,
            "fit_end": wf_folds[-1].fit_end,
            "calibration_start": wf_folds[0].cal_start,
            "calibration_end": wf_folds[-1].cal_end,
            "oos_start": wf_folds[0].oos_start,
            "oos_end": wf_folds[-1].oos_end,
            "wf_n_folds": len(wf_folds),
            "wf_scheme": strategy_cfg.candidate.wf_scheme,
            "y_gate_oos_pos_rate": float(np.mean(p_pass > 0.5)) if p_pass.size > 0 else 0.0,
            "gate_calibration_used": fold_calibration_used,
            "gate_calibration_reason": fold_calibration_reason,
            "gate_p_mean": gate_summary["mean"],
            "gate_p_median": gate_summary["median"],
            "gate_p_p90": gate_summary["p90"],
            "gate_p_max": gate_summary["max"],
            "gate_pct_ge40": _threshold_rate(p_pass, 0.40),
            "gate_pct_ge45": _threshold_rate(p_pass, 0.45),
            "gate_pct_ge50": _threshold_rate(p_pass, 0.50),
            "gate_pct_ge55": _threshold_rate(p_pass, 0.55),
            "mu_mean_bps": edge_summary["mean"],
            "mu_median_bps": edge_summary["median"],
            "mu_p90_bps": edge_summary["p90"],
            "mu_max_bps": edge_summary["max"],
            "q10_mean_bps": q10_summary["mean"],
            "q10_p10_bps": q10_summary["p10"],
            "q10_median_bps": q10_summary["median"],
            "q10_min_bps": q10_summary["min"],
            "utility_mean": utility_summary["mean"],
            "utility_median": utility_summary["median"],
            "utility_p90": utility_summary["p90"],
            "utility_max": utility_summary["max"],
            "wf_selected_total": wf_selected_total,
            "wf_eligible_total": wf_eligible_total,
            "wf_fold_realized_mean_bps": wf_fold_realized_mean_bps,
            "wf_fold_log_growth_mean": wf_fold_log_growth_mean,
            "wf_shadow_profile_count": wf_shadow_profile_count,
            "wf_shadow_max_selected_total": wf_shadow_max_selected_total,
            "wf_shadow_max_eligible": wf_shadow_max_eligible,
            "wf_waterfall_expected_utility_p90_bps": wf_waterfall_expected_utility_p90_bps,
            "wf_waterfall_downside_drag_p90_bps": wf_waterfall_downside_drag_p90_bps,
            "wf_fold_details": wf_fold_details,
            "selected_pre_group": int(selection_diag.get("selected_pre_group", len(selected))),
            "selected_total": int(selection_diag.get("selected_total", len(selected))),
            "eligible": int(selection_diag.get("eligible", 0)),
            "n_keep": int(selection_diag.get("n_keep", 0)),
            "policy": str(selection_diag.get("policy", strategy_cfg.candidate.selection_policy)),
            "zero_reason": str(selection_diag.get("zero_reason", "unknown")),
            "workflow_status": (
                CandidateWorkflowStatus.WF_ELIGIBLE.value
                if int(selection_diag.get("selected_total", len(selected))) > 0
                else CandidateWorkflowStatus.BLOCKED.value
            ),
            "breakeven_floor_bps": float(
                selection_diag.get("breakeven_floor_bps", strategy_cfg.candidate.cost_floor_bps)
            ),
            "recommended_keep_variants": diag.recommended_keep_variants,
            "recommended_flip_variants": diag.recommended_flip_variants,
            "recommended_keep_signal_cells": diag.recommended_keep_signal_cells,
            "recommended_flip_signal_cells": diag.recommended_flip_signal_cells,
            "failure_report": diag.recommendation_failure_report,
            "recommendation_basis": diag.recommendation_basis,
            "recommendation_start": int(diag.recommendation_split[0]),
            "recommendation_end": int(diag.recommendation_split[1]),
            "report_start": int(diag.report_split[0]),
            "report_end": int(diag.report_split[1]),
        },
        labeled=labeled,
        labeled_unfiltered=labeled_all,
        fit_set=last_fold_out.fit_set if last_fold_out else None,
        calibration_set=last_fold_out.calibration_set if last_fold_out else None,
        oos_set=last_fold_out.oos_set if last_fold_out else None,
        gate_model=last_fold_out.gate_model if last_fold_out else None,
        edge_models=last_fold_out.edge_models if last_fold_out else None,
        fit_start=wf_folds[0].fit_start,
        fit_end=wf_folds[-1].fit_end,
        calibration_start=wf_folds[0].cal_start,
        calibration_end=wf_folds[-1].cal_end,
        oos_start=wf_folds[0].oos_start,
        oos_end=wf_folds[-1].oos_end,
        fold_oos_boundaries=_fold_oos_boundaries,
        alpha_foundry_report=alpha_foundry_report,
        l0_delivery_manifest=l0_delivery_manifest,
    )


def merge_candidate_output_into_data_maps(
    candidate_out: CandidatePipelineOutput,
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    log_tag: str = "",
) -> None:
    """Merge candidate output payload into data maps."""
    import time

    t_merge_start_all = time.perf_counter()
    panel = getattr(candidate_out, "alpha_panel", None)
    if panel is None or panel.empty:
        return
    required = {
        "alpha_long",
        "alpha_short",
        "target_weight",
        "candidate_family",
        "candidate_variant",
        "p_pass",
        "mu_net_decision_bps",
        "q10_net_bps",
        "utility_score",
        "candidate_stop_atr_mult",
        "candidate_take_profit_atr_mult",
    }
    if not required.issubset(panel.columns):
        _logger.warning("[%s] candidate panel missing required columns; skip merge", log_tag)
        return

    # Hoist pd.to_datetime out of the loop for right dataframe
    panel_df = panel.reset_index() if "symbol" not in panel.columns else panel.copy()
    panel_df["_merge_datetime"] = pd.to_datetime(panel_df["datetime"], utc=True).dt.tz_localize(None)
    by_sym = panel_df.groupby("symbol", sort=False)
    _merge_sym_times: list[float] = []

    for sym in symbols:
        t_sym = time.perf_counter()
        if sym not in data_maps or tf not in data_maps[sym]:
            continue
        df = data_maps[sym][tf]
        for col, default in (
            ("alpha_long", 0.0),
            ("alpha_short", 0.0),
            ("target_weight", 0.0),
            ("p_pass", 0.0),
            ("mu_net_decision_bps", 0.0),
            ("q10_net_bps", 0.0),
            ("utility_score", 0.0),
            ("candidate_stop_atr_mult", 0.0),
            ("candidate_take_profit_atr_mult", 0.0),
        ):
            if col not in df.columns:
                df[col] = np.full(len(df), default, dtype=np.float64)
        if "candidate_family" not in df.columns:
            df["candidate_family"] = np.full(len(df), "", dtype=object)
        if "candidate_variant" not in df.columns:
            df["candidate_variant"] = np.full(len(df), "", dtype=object)

        try:
            sym_rows = by_sym.get_group(sym)
        except KeyError:
            continue

        # Skip pd.to_datetime inside the loop if already datetime64 type
        df_dt = df["datetime"]
        if pd.api.types.is_datetime64_any_dtype(df_dt):
            left_merge_dt = df_dt.dt.tz_convert(None) if isinstance(df_dt.dtype, pd.DatetimeTZDtype) else df_dt
        else:
            left_merge_dt = pd.to_datetime(df_dt, utc=True).dt.tz_localize(None)

        left = pd.DataFrame({"_merge_datetime": left_merge_dt})
        right = sym_rows[["_merge_datetime", *list(required)]]
        merged = left.merge(right, on="_merge_datetime", how="left")

        df["alpha_long"] = merged["alpha_long"].fillna(0.0).to_numpy(dtype=np.float64)
        df["alpha_short"] = merged["alpha_short"].fillna(0.0).to_numpy(dtype=np.float64)
        df["target_weight"] = merged["target_weight"].fillna(0.0).to_numpy(dtype=np.float64)
        df["candidate_family"] = merged["candidate_family"].fillna("").to_numpy(dtype=object)
        df["candidate_variant"] = merged["candidate_variant"].fillna("").to_numpy(dtype=object)
        df["p_pass"] = merged["p_pass"].fillna(0.0).to_numpy(dtype=np.float64)
        df["mu_net_decision_bps"] = merged["mu_net_decision_bps"].fillna(0.0).to_numpy(dtype=np.float64)
        df["q10_net_bps"] = merged["q10_net_bps"].fillna(0.0).to_numpy(dtype=np.float64)
        df["utility_score"] = merged["utility_score"].fillna(0.0).to_numpy(dtype=np.float64)
        df["candidate_stop_atr_mult"] = merged["candidate_stop_atr_mult"].fillna(0.0).to_numpy(dtype=np.float64)
        df["candidate_take_profit_atr_mult"] = (
            merged["candidate_take_profit_atr_mult"].fillna(0.0).to_numpy(dtype=np.float64)
        )
        sym_time = time.perf_counter() - t_sym
        _merge_sym_times.append(sym_time)
    total_merge = time.perf_counter() - t_merge_start_all
    if _merge_sym_times:
        arr = np.array(_merge_sym_times)
        _logger.debug(
            "[PROFILE][MERGE][SUMMARY] tag=%s n_syms=%d total=%.4fs min=%.4f max=%.4f mean=%.4f median=%.4f",
            log_tag,
            len(_merge_sym_times),
            total_merge,
            float(arr.min()),
            float(arr.max()),
            float(arr.mean()),
            float(np.median(arr)),
        )
    else:
        _logger.debug("[PROFILE][MERGE] Total merge %s took %.4fs; no symbols processed", log_tag, total_merge)


def merge_candidate_output_into_is_and_oos(
    candidate_out: CandidatePipelineOutput,
    is_maps: dict[str, dict[str, Any]],
    oos_maps: dict[str, dict[str, Any]],
    valid_symbols: list[str],
    tf: str,
) -> None:
    """Merge candidate output into both IS and OOS maps."""
    merge_candidate_output_into_data_maps(candidate_out, is_maps, valid_symbols, tf, log_tag="is")
    merge_candidate_output_into_data_maps(candidate_out, oos_maps, valid_symbols, tf, log_tag="oos")


def copy_data_maps_tf_clone(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for sym in symbols:
        out[sym] = dict(data_maps.get(sym, {}))
        frame = out[sym].get(tf)
        if isinstance(frame, pd.DataFrame):
            out[sym][tf] = frame.copy()
    return out


def build_probe_prior_map(
    probe_manifest: list[dict[str, Any]],
    boost: float = 0.3,
) -> dict[tuple[str, str, str], float]:
    """Convert probe winning cells to L1 quality weight floor mapping.

    Args:
        probe_manifest: List of TfCellEvidence dicts from probe_timeframe_alpha.
        boost: Quality weight floor for probe-winning signals.

    Returns:
        {(family, variant, symbol): qw_floor}
    """
    prior: dict[tuple[str, str, str], float] = {}
    for cell in probe_manifest:
        if not cell.get("is_winner", False):
            continue
        key = (cell["family"], cell["variant"], cell["symbol"])
        prior[key] = boost
    return prior
