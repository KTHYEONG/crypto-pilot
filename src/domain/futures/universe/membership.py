from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numba
import numpy as np
import pandas as pd


@numba.njit(cache=True)  # type: ignore[untyped-decorator]
def _calculate_warm_ready_numba(active: np.ndarray, warmup_bars_required: int) -> np.ndarray:
    n = len(active)
    run_lens = np.zeros(n, dtype=np.float64)
    current_run = 0.0
    for i in range(n):
        if active[i] > 0.0:
            current_run += 1.0
            run_lens[i] = current_run
        else:
            current_run = 0.0
            run_lens[i] = 0.0

    warm_ready = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if run_lens[i] >= warmup_bars_required:
            warm_ready[i] = 1.0
    return warm_ready


def canonical_symbol(symbol: Any) -> str:
    return str(symbol).strip().upper().replace("/", "").replace("_", "")


def _quarter_start(dt: date) -> date:
    q_month = ((dt.month - 1) // 3) * 3 + 1
    return date(dt.year, q_month, 1)


def _normalize_timeline(
    timeline: Mapping[date, frozenset[str] | set[str]],
) -> dict[date, frozenset[str]]:
    """timeline 매핑을 분기 시작일 키 + canonical frozenset 값으로 정규화한다.

    Args:
        timeline: 임의 날짜 → 심볼 집합 매핑.

    Returns:
        분기 시작일 → canonical frozenset[str] 정규화 딕셔너리.

    Time: O(Q · S_q), Space: O(Q · S_q) — Q=분기 수, S_q=분기당 심볼 수.
    """
    result: dict[date, frozenset[str]] = {}
    for k, syms in timeline.items():
        q_start = _quarter_start(k)
        result[q_start] = frozenset(canonical_symbol(s) for s in syms)
    return result


@dataclass(frozen=True, slots=True)
class MembershipInjectionReport:
    requested_pairs: int
    injected_pairs: int
    missing_pairs: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class PITUniverseAudit:
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    checked_cells: int
    active_cells: int
    missing_pairs: tuple[tuple[str, str], ...]
    parity_mismatches: int
    passed: bool


class PITUniverseContractError(ValueError):
    """Raised when L0/L1 PIT universe masks are incomplete or inconsistent."""


@dataclass(slots=True, frozen=True)
class MembershipMaskBundle:
    """Per-bar membership-derived masks and effective kill signal."""

    universe_active_mask: np.ndarray
    universe_entry_warm_mask: np.ndarray
    membership_kill_signal: np.ndarray
    entry_block_mask: np.ndarray
    kill_signal: np.ndarray
    # C1 inference용 — Stage5 timeline 기반 (inference_timeline 미지정 시 trading mask 복사)
    inference_active_mask: np.ndarray
    inference_entry_warm_mask: np.ndarray


def build_membership_mask_bundle(
    *,
    datetimes: pd.Series,
    symbol: str,
    timeline: Mapping[date, frozenset[str] | set[str]],
    warmup_bars_required: int,
    raw_kill_signal: np.ndarray | None = None,
    inference_timeline: Mapping[date, frozenset[str] | set[str]] | None = None,
    norm_timeline: dict[date, frozenset[str]] | None = None,
    norm_inf_timeline: dict[date, frozenset[str]] | None = None,
) -> MembershipMaskBundle:
    if pd.api.types.is_datetime64_any_dtype(datetimes):
        dt_ser = datetimes
    else:
        dt_ser = pd.to_datetime(datetimes, utc=True, errors="coerce")
    n = len(dt_ser)
    sym_norm = canonical_symbol(symbol)

    # Pre-normalized 전달 시 내부 정규화 스킵; None이면 fallback
    if norm_timeline is None:
        norm_timeline = _normalize_timeline(timeline)

    active_quarters_ts = {pd.Timestamp(q) for q, syms in norm_timeline.items() if sym_norm in syms}
    dti = pd.DatetimeIndex(dt_ser)
    if dti.tz is not None:
        dti = dti.tz_localize(None)
    na_mask = dti.isna()
    has_na = bool(na_mask.any())
    if has_na:
        dti_filled = dti.fillna(pd.Timestamp("1970-01-01"))
        bar_q_starts = dti_filled.to_period("Q").start_time
        active = bar_q_starts.isin(active_quarters_ts).astype(np.float64)
        active[na_mask] = 0.0
    else:
        bar_q_starts = dti.to_period("Q").start_time
        active = bar_q_starts.isin(active_quarters_ts).astype(np.float64)

    active_prev = np.concatenate(([0.0], active[:-1]))
    membership_kill = np.where((active_prev > 0.0) & (active <= 0.0), 1.0, 0.0)

    if warmup_bars_required <= 1:
        warm_ready = active.copy()
    else:
        warm_ready = _calculate_warm_ready_numba(active, warmup_bars_required)

    entry_block_mask = np.where((active > 0.0) & (warm_ready > 0.0), 0.0, 1.0)
    raw_kill = (
        np.zeros(n, dtype=np.float64) if raw_kill_signal is None else np.asarray(raw_kill_signal, dtype=np.float64)
    )
    effective_kill = np.maximum(raw_kill, membership_kill)

    # inference mask 계산 — inference_timeline 미지정 시 trading mask 복사
    if inference_timeline is None and norm_inf_timeline is None:
        inf_active = active.copy()
        inf_warm_ready = warm_ready.copy()
    else:
        # Pre-normalized 전달 시 내부 정규화 스킵; None이면 fallback
        if norm_inf_timeline is None:
            norm_inf_timeline = _normalize_timeline(inference_timeline or {})

        inf_active_quarters_ts = {pd.Timestamp(q) for q, syms in norm_inf_timeline.items() if sym_norm in syms}

        inf_active = bar_q_starts.isin(inf_active_quarters_ts).astype(np.float64)
        if has_na:
            inf_active[na_mask] = 0.0

        if warmup_bars_required <= 1:
            inf_warm_ready = inf_active.copy()
        else:
            inf_warm_ready = _calculate_warm_ready_numba(inf_active, warmup_bars_required)

    return MembershipMaskBundle(
        universe_active_mask=active,
        universe_entry_warm_mask=warm_ready,
        membership_kill_signal=membership_kill,
        entry_block_mask=entry_block_mask,
        kill_signal=effective_kill,
        inference_active_mask=inf_active,
        inference_entry_warm_mask=inf_warm_ready,
    )


def inject_membership_masks_into_maps(
    *,
    data_maps: dict[str, dict[str, Any]],
    oos_data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    timeline: Mapping[date, frozenset[str] | set[str]],
    warmup_bars_required: int,
    inference_timeline: Mapping[date, frozenset[str] | set[str]] | None = None,
) -> None:
    if not timeline:
        return

    # 심볼 루프 진입 전 1회 정규화 — 반복 정규화 O(N_sym x Q·S_q) → O(Q·S_q)
    pre_norm_timeline = _normalize_timeline(timeline)
    pre_norm_inf_timeline: dict[date, frozenset[str]] | None = (
        _normalize_timeline(inference_timeline) if inference_timeline is not None else None
    )

    for sym in symbols:
        for maps in (data_maps, oos_data_maps):
            sym_map = maps.get(sym, {})
            frame = sym_map.get(tf)
            if not isinstance(frame, pd.DataFrame) or frame.empty or "datetime" not in frame.columns:
                continue
            raw_kill = (
                frame["kill_signal"].to_numpy(dtype=np.float64, copy=False) if "kill_signal" in frame.columns else None
            )
            bundle = build_membership_mask_bundle(
                datetimes=frame["datetime"],
                symbol=sym,
                timeline=timeline,
                warmup_bars_required=warmup_bars_required,
                raw_kill_signal=raw_kill,
                inference_timeline=inference_timeline,
                norm_timeline=pre_norm_timeline,
                norm_inf_timeline=pre_norm_inf_timeline,
            )
            frame.loc[:, "universe_active_mask"] = bundle.universe_active_mask
            frame.loc[:, "universe_entry_warm_mask"] = bundle.universe_entry_warm_mask
            frame.loc[:, "membership_kill_signal"] = bundle.membership_kill_signal
            frame.loc[:, "entry_block_mask"] = bundle.entry_block_mask
            frame.loc[:, "kill_signal"] = bundle.kill_signal
            frame.loc[:, "inference_active_mask"] = bundle.inference_active_mask
            frame.loc[:, "inference_entry_warm_mask"] = bundle.inference_entry_warm_mask


def validate_pit_universe_contract(
    *,
    data_maps: Mapping[str, Mapping[str, Any]],
    symbols: Sequence[str],
    timeframes: Sequence[str],
    timeline: Mapping[date, frozenset[str] | set[str]],
    state_cube: Any = None,
) -> PITUniverseAudit:
    required_mask_cols = [
        "universe_active_mask",
        "universe_entry_warm_mask",
        "entry_block_mask",
        "kill_signal",
    ]
    missing_pairs: list[tuple[str, str]] = []
    checked_cells = 0
    active_cells = 0
    parity_mismatches = 0

    for tf in timeframes:
        for sym in symbols:
            checked_cells += 1
            sym_map = data_maps.get(sym, {})
            frame = sym_map.get(tf) if isinstance(sym_map, dict) else None
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                missing_pairs.append((sym, tf))
                continue
            has_all_masks = all(col in frame.columns for col in required_mask_cols)
            if not has_all_masks:
                missing_pairs.append((sym, tf))
                continue
            active_mask = frame["universe_active_mask"].to_numpy(dtype=bool, copy=False)
            active_cells += int(active_mask.any())

    return PITUniverseAudit(
        symbols=tuple(symbols),
        timeframes=tuple(timeframes),
        checked_cells=checked_cells,
        active_cells=active_cells,
        missing_pairs=tuple(missing_pairs),
        parity_mismatches=parity_mismatches,
        passed=len(missing_pairs) == 0,
    )
