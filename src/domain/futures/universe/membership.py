from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd


def canonical_symbol(symbol: Any) -> str:
    return str(symbol).strip().upper().replace("/", "").replace("_", "")


def _quarter_start(dt: date) -> date:
    q_month = ((dt.month - 1) // 3) * 3 + 1
    return date(dt.year, q_month, 1)


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
) -> MembershipMaskBundle:
    if pd.api.types.is_datetime64_any_dtype(datetimes):
        dt_ser = datetimes
    else:
        dt_ser = pd.to_datetime(datetimes, utc=True, errors="coerce")
    n = len(dt_ser)
    sym_norm = canonical_symbol(symbol)
    
    # Normalize timeline keys to their respective quarter starts
    norm_timeline: dict[date, frozenset[str]] = {}
    for k, syms in timeline.items():
        q_start = _quarter_start(k)
        norm_timeline[q_start] = frozenset(canonical_symbol(s) for s in syms)

    active_quarters = {q for q, syms in norm_timeline.items() if sym_norm in syms}
    q_list = np.asarray(list(active_quarters), dtype=object)
    dti = pd.DatetimeIndex(dt_ser)
    if dti.tz is not None:
        dti = dti.tz_localize(None)
    na_mask = dti.isna()
    has_na = bool(na_mask.any())
    if has_na:
        dti_filled = dti.fillna(pd.Timestamp("1970-01-01"))
        bar_q_starts = dti_filled.to_period("Q").start_time.date
        active = np.isin(bar_q_starts, q_list).astype(np.float64)
        active[na_mask] = 0.0
    else:
        bar_q_starts = dti.to_period("Q").start_time.date
        active = np.isin(bar_q_starts, q_list).astype(np.float64)

    active_prev = np.concatenate(([0.0], active[:-1]))
    membership_kill = np.where((active_prev > 0.0) & (active <= 0.0), 1.0, 0.0)

    if warmup_bars_required <= 1:
        warm_ready = active.copy()
    else:
        s = pd.Series(active)
        groups = (s == 0.0).cumsum()
        run_lens = s.groupby(groups, sort=False).cumsum()
        warm_ready = (run_lens >= warmup_bars_required).astype(np.float64).to_numpy()

    entry_block_mask = np.where((active > 0.0) & (warm_ready > 0.0), 0.0, 1.0)
    raw_kill = (
        np.zeros(n, dtype=np.float64)
        if raw_kill_signal is None
        else np.asarray(raw_kill_signal, dtype=np.float64)
    )
    effective_kill = np.maximum(raw_kill, membership_kill)

    # inference mask 계산 — inference_timeline 미지정 시 trading mask 복사
    if inference_timeline is None:
        inf_active = active.copy()
        inf_warm_ready = warm_ready.copy()
    else:
        norm_inf_timeline: dict[date, frozenset[str]] = {}
        for k, syms in inference_timeline.items():
            q_start = _quarter_start(k)
            norm_inf_timeline[q_start] = frozenset(canonical_symbol(s) for s in syms)

        inf_active_quarters = {q for q, syms in norm_inf_timeline.items() if sym_norm in syms}
        inf_q_list = np.asarray(list(inf_active_quarters), dtype=object)

        inf_active = np.isin(bar_q_starts, inf_q_list).astype(np.float64)
        if has_na:
            inf_active[na_mask] = 0.0

        if warmup_bars_required <= 1:
            inf_warm_ready = inf_active.copy()
        else:
            s_inf = pd.Series(inf_active)
            inf_groups = (s_inf == 0.0).cumsum()
            inf_run_lens = s_inf.groupby(inf_groups, sort=False).cumsum()
            inf_warm_ready = (inf_run_lens >= warmup_bars_required).astype(np.float64).to_numpy()

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

    for sym in symbols:
        for maps in (data_maps, oos_data_maps):
            sym_map = maps.get(sym, {})
            frame = sym_map.get(tf)
            if (
                not isinstance(frame, pd.DataFrame)
                or frame.empty
                or "datetime" not in frame.columns
            ):
                continue
            raw_kill = (
                frame["kill_signal"].to_numpy(dtype=np.float64, copy=False)
                if "kill_signal" in frame.columns
                else None
            )
            bundle = build_membership_mask_bundle(
                datetimes=frame["datetime"],
                symbol=sym,
                timeline=timeline,
                warmup_bars_required=warmup_bars_required,
                raw_kill_signal=raw_kill,
                inference_timeline=inference_timeline,
            )
            frame.loc[:, "universe_active_mask"] = bundle.universe_active_mask
            frame.loc[:, "universe_entry_warm_mask"] = bundle.universe_entry_warm_mask
            frame.loc[:, "membership_kill_signal"] = bundle.membership_kill_signal
            frame.loc[:, "entry_block_mask"] = bundle.entry_block_mask
            frame.loc[:, "kill_signal"] = bundle.kill_signal
            frame.loc[:, "inference_active_mask"] = bundle.inference_active_mask
            frame.loc[:, "inference_entry_warm_mask"] = bundle.inference_entry_warm_mask
