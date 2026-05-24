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


def build_membership_mask_bundle(
    *,
    datetimes: pd.Series,
    symbol: str,
    timeline: Mapping[date, frozenset[str] | set[str]],
    warmup_bars_required: int,
    raw_kill_signal: np.ndarray | None = None,
) -> MembershipMaskBundle:
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
    if dti.isna().any():
        na_mask = dti.isna()
        dti_filled = dti.fillna(pd.Timestamp("1970-01-01", tz="UTC"))
        bar_q_starts = dti_filled.to_period("Q").start_time.date
        active = np.isin(bar_q_starts, q_list).astype(np.float64)
        active[na_mask] = 0.0
    else:
        bar_q_starts = dti.to_period("Q").start_time.date
        active = np.isin(bar_q_starts, q_list).astype(np.float64)

    active_prev = np.concatenate(([0.0], active[:-1]))
    active_on = (active_prev <= 0.0) & (active > 0.0)
    membership_kill = np.where((active_prev > 0.0) & (active <= 0.0), 1.0, 0.0)

    warm_ready = np.zeros(n, dtype=np.float64)
    if warmup_bars_required <= 1:
        warm_ready = active.copy()
    else:
        run_len = 0
        for idx in range(n):
            if active[idx] > 0.0:
                run_len = run_len + 1 if not active_on[idx] else 1
                if run_len >= warmup_bars_required:
                    warm_ready[idx] = 1.0
            else:
                run_len = 0

    entry_block_mask = np.where((active > 0.0) & (warm_ready > 0.0), 0.0, 1.0)
    raw_kill = (
        np.zeros(n, dtype=np.float64)
        if raw_kill_signal is None
        else np.asarray(raw_kill_signal, dtype=np.float64)
    )
    effective_kill = np.maximum(raw_kill, membership_kill)
    return MembershipMaskBundle(
        universe_active_mask=active,
        universe_entry_warm_mask=warm_ready,
        membership_kill_signal=membership_kill,
        entry_block_mask=entry_block_mask,
        kill_signal=effective_kill,
    )


def inject_membership_masks_into_maps(
    *,
    data_maps: dict[str, dict[str, Any]],
    oos_data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    timeline: Mapping[date, frozenset[str] | set[str]],
    warmup_bars_required: int,
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
            )
            frame.loc[:, "universe_active_mask"] = bundle.universe_active_mask
            frame.loc[:, "universe_entry_warm_mask"] = bundle.universe_entry_warm_mask
            frame.loc[:, "membership_kill_signal"] = bundle.membership_kill_signal
            frame.loc[:, "entry_block_mask"] = bundle.entry_block_mask
            frame.loc[:, "kill_signal"] = bundle.kill_signal
