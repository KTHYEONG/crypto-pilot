# src/domain/futures/strategy/tiered_workflow/awf_sim.py

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.portfolio.portfolio_constructor import (
    PortfolioCaps,
    diagonal_kelly_weights,
)
from src.domain.futures.portfolio.signal_composer import (
    composer_sigma_lookback_bars,
    hours_per_bar_tf,
    rolling_per_bar_return_std,
)
from src.domain.futures.strategy.candidate_contracts import (
    ValidatedSignalBatch,
    ValidatedSignalEvent,
)
from src.domain.futures.strategy.cs_rank import (
    VOL_FLOOR,
    SymbolSignal,
    rank_and_select,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2AllocationConfig,
    Layer2SignalSchedule,
)

if TYPE_CHECKING:
    from src.domain.futures.strategy.common.alignment import AlignedMarketData
    from src.domain.futures.strategy.tiered_workflow.dataclasses import SymbolRealizedStat
    from src.domain.futures.strategy.walk_forward import WFFold


@dataclass(slots=True)
class _AwfSimResult:
    """run_awf 내부 시뮬레이션 결과 (private)."""

    rets_hybrid: list[float]
    rets_baseline: list[float]
    last_selected: frozenset[str]
    last_w: NDArray[np.float64]
    all_turnovers: list[float]
    friction_pass_total: int
    signal_total: int
    support_leak_count: int
    fold_rets_hybrid: list[list[float]]    # fold별 strategy returns
    fold_rets_baseline: list[list[float]]  # fold별 baseline returns


def _event_strength(event: ValidatedSignalEvent) -> float:
    return (
        abs(float(event.expected_net_bps))
        / max(int(event.expected_holding_bars), 1)
        * max(float(event.quality_weight), 0.0)
    )


def _is_better_event(candidate: ValidatedSignalEvent, incumbent: ValidatedSignalEvent) -> bool:
    if candidate.decision_idx != incumbent.decision_idx:
        return candidate.decision_idx > incumbent.decision_idx
    return _event_strength(candidate) > _event_strength(incumbent)


def build_layer2_signal_schedule(
    *,
    signal_batch: ValidatedSignalBatch,
    start_idx: int,
    end_idx: int,
) -> Layer2SignalSchedule:
    bar_count = max(end_idx - start_idx, 0)
    events_by_bar: list[dict[str, ValidatedSignalEvent]] = [{} for _ in range(bar_count)]
    bounded_events: list[ValidatedSignalEvent] = []
    for event in signal_batch.events:
        active_start = max(start_idx, int(event.decision_idx) + 1)
        active_end = min(
            end_idx,
            int(event.decision_idx) + 1 + max(int(event.expected_holding_bars), 1),
        )
        if active_start >= active_end:
            continue
        bounded_events.append(event)
        for t in range(active_start, active_end):
            slot = events_by_bar[t - start_idx]
            incumbent = slot.get(event.symbol)
            if incumbent is None or _is_better_event(event, incumbent):
                slot[event.symbol] = event
    return Layer2SignalSchedule(
        events=tuple(bounded_events),
        start_idx=start_idx,
        end_idx=end_idx,
        _events_by_bar=tuple(events_by_bar),
    )


def resolve_active_symbol_signals(
    *,
    schedule: Layer2SignalSchedule,
    t: int,
    symbols: tuple[str, ...],
    volatility_1d: NDArray[np.float64],
) -> dict[str, SymbolSignal]:
    if t < schedule.start_idx or t >= schedule.end_idx:
        return {}
    offset = t - schedule.start_idx
    if offset >= len(schedule._events_by_bar):
        return {}
    symbol_lookup = {symbol: idx for idx, symbol in enumerate(symbols)}
    result: dict[str, SymbolSignal] = {}
    for symbol, event in schedule._events_by_bar[offset].items():
        sym_idx = symbol_lookup.get(symbol)
        if sym_idx is None:
            continue
        per_bar_bps = float(event.side) * float(event.expected_net_bps) / max(int(event.expected_holding_bars), 1)
        result[symbol] = SymbolSignal(
            raw_mu=per_bar_bps,
            volatility=float(max(volatility_1d[sym_idx], VOL_FLOOR)),
            n_obs=1,
            t_stat=0.0,
            valid=bool(np.isfinite(per_bar_bps)),
            beta_btc=None,
            quality_weight=float(event.quality_weight),
        )
    return result


def compute_rebalance_cost(
    *,
    previous_weights: NDArray[np.float64],
    target_weights: NDArray[np.float64],
    round_trip_cost_bps: NDArray[np.float64],
) -> float:
    delta = np.abs(np.asarray(target_weights, dtype=np.float64) - np.asarray(previous_weights, dtype=np.float64))
    one_way_cost = np.asarray(round_trip_cost_bps, dtype=np.float64) / 2.0
    return float(np.sum(delta * one_way_cost) * 1e-4)


def compute_futures_bar_return(
    *,
    weights: NDArray[np.float64],
    price_returns: NDArray[np.float64],
    funding_rates: NDArray[np.float64],
) -> float:
    gross_price_return = float(np.dot(weights, price_returns))
    funding_return = -float(np.dot(weights, funding_rates))
    return gross_price_return + funding_return


def _resolve_tradeable_mask(
    *,
    aligned: AlignedMarketData,
    t: int,
    n_sym: int,
) -> NDArray[np.bool_]:
    def _mask_row(name: str, default: bool) -> NDArray[np.bool_]:
        value = getattr(aligned, name, None)
        if isinstance(value, np.ndarray) and value.ndim == 2 and t < value.shape[0] and value.shape[1] == n_sym:
            return np.asarray(value[t], dtype=np.bool_)
        return np.full(n_sym, default, dtype=np.bool_)

    active_mask = _mask_row("active_mask", True)
    warm_mask = _mask_row("warm_mask", True)
    entry_block_mask = _mask_row("entry_block_mask", False)
    kill_mask = _mask_row("kill_mask", False)
    return active_mask & warm_mask & ~entry_block_mask & ~kill_mask


def _resolve_funding_row(
    *,
    aligned: AlignedMarketData,
    t2: int,
    n_sym: int,
) -> NDArray[np.float64]:
    funding = getattr(aligned, "funding_2d", None)
    if isinstance(funding, np.ndarray) and funding.ndim == 2 and t2 < funding.shape[0] and funding.shape[1] == n_sym:
        row = np.asarray(funding[t2], dtype=np.float64)
        return np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
    return np.zeros(n_sym, dtype=np.float64)


def _run_awf_simulation(
    *,
    signal_batch: ValidatedSignalBatch,
    aligned: AlignedMarketData,
    awf_folds: tuple[WFFold, ...],
    config: Layer2AllocationConfig,
    caps: PortfolioCaps,
    tf: str = "4h",
) -> _AwfSimResult:
    """AWF 시뮬레이션 핵심 루프 (L2/L3 공용)."""
    import logging
    import time
    logger = logging.getLogger("src.domain.futures.strategy.tiered_workflow")
    t_start_total = time.perf_counter()
    prof_prep = 0.0
    prof_rank = 0.0
    prof_alloc = 0.0
    prof_eval = 0.0

    k_rank = int(config.k_rank)
    rank_buffer = int(config.rank_buffer)
    kelly_fraction = float(config.kelly_fraction)
    vol_target = config.vol_target
    no_trade_band = float(config.no_trade_band)
    rebalance_bars = int(config.rebalance_bars)
    friction_safety_mult = float(getattr(config, "friction_safety_mult", 1.0))

    symbols = aligned.symbols
    n_sym = len(symbols)
    lookback = composer_sigma_lookback_bars(tf)
    bars_per_year = 24.0 * 365.0 / max(hours_per_bar_tf(tf), 1e-9)
    signal_start = min((fold.oos_start for fold in awf_folds), default=0)
    signal_end = max((fold.oos_end for fold in awf_folds), default=0)
    sym_to_idx = {s: i for i, s in enumerate(symbols)}

    t0_prep = time.perf_counter()
    vol_matrix = np.full_like(aligned.close_2d, VOL_FLOOR)
    for i in range(n_sym):
        close_col = aligned.close_2d[:, i]
        v_std = rolling_per_bar_return_std(close_col, lookback)
        vol_matrix[:, i] = np.maximum(v_std, VOL_FLOOR)
    prof_prep += time.perf_counter() - t0_prep

    all_rets_hybrid: list[float] = []
    all_rets_baseline: list[float] = []
    all_turnovers: list[float] = []
    friction_pass_total = 0
    signal_total = 0
    support_leak_count = 0
    fold_rets_hybrid: list[list[float]] = []
    fold_rets_baseline: list[list[float]] = []

    prev_selection: frozenset[str] = frozenset()
    prev_w: NDArray[np.float64] = np.zeros(n_sym, dtype=np.float64)
    last_selected: frozenset[str] = frozenset()
    last_w: NDArray[np.float64] = np.zeros(n_sym, dtype=np.float64)
    schedule = build_layer2_signal_schedule(
        signal_batch=signal_batch,
        start_idx=signal_start,
        end_idx=signal_end,
    )

    for _fold_idx, fold in enumerate(awf_folds):
        _fold_h: list[float] = []
        _fold_b: list[float] = []
        for t in range(fold.oos_start, fold.oos_end - 1, rebalance_bars):
            t_end = min(t + rebalance_bars, fold.oos_end - 1)

            t0_prep = time.perf_counter()
            tradeable_mask = _resolve_tradeable_mask(
                aligned=aligned,
                t=t,
                n_sym=n_sym,
            )
            valid_signals = resolve_active_symbol_signals(
                schedule=schedule,
                t=t,
                symbols=symbols,
                volatility_1d=vol_matrix[t],
            )
            valid_signals = {
                symbol: signal
                for symbol, signal in valid_signals.items()
                if tradeable_mask[sym_to_idx[symbol]]
            }
            prof_prep += time.perf_counter() - t0_prep

            t0_rank = time.perf_counter()
            selected, _z_scores = rank_and_select(
                valid_signals,
                k_rank=k_rank,
                sector_cap=n_sym,
                prev_selection=prev_selection,
                rank_buffer=rank_buffer,
                min_abs_z=float(config.min_abs_rank_z),
                selection_mode="absolute",
            )
            last_selected = selected
            prof_rank += time.perf_counter() - t0_rank

            t0_alloc = time.perf_counter()
            mu_arr: NDArray[np.float64] = np.zeros(n_sym, dtype=np.float64)
            sig_arr: NDArray[np.float64] = np.full(n_sym, VOL_FLOOR, dtype=np.float64)
            for sym, ss in valid_signals.items():
                if sym in sym_to_idx:
                    i = sym_to_idx[sym]
                    if sym in selected:
                        mu_arr[i] = ss.raw_mu
                    sig_arr[i] = ss.volatility

            if (
                aligned.execution_cost_bps_2d is not None
                and t < aligned.execution_cost_bps_2d.shape[0]
            ):
                hurdle = aligned.execution_cost_bps_2d[t].astype(np.float64)
                hurdle = np.nan_to_num(hurdle, nan=3.8, posinf=3.8, neginf=3.8)
            else:
                hurdle = np.full(n_sym, 3.8, dtype=np.float64)

            btc_beta: NDArray[np.float64] | None = None
            if aligned.beta_vs_market_1d is not None:
                btc_beta = aligned.beta_vs_market_1d.astype(np.float64)
                btc_beta = np.nan_to_num(btc_beta, nan=0.0)

            support_mask = mu_arr != 0.0
            w = diagonal_kelly_weights(
                mu_bps=mu_arr,
                sigma=sig_arr,
                kelly_fraction=kelly_fraction,
                vol_target=vol_target,
                friction_hurdle_bps=hurdle,
                holding_bars=rebalance_bars,
                friction_safety_mult=friction_safety_mult,
                caps=caps,
                prev_w=prev_w,
                no_trade_band=no_trade_band,
                btc_beta=btc_beta,
                bars_per_year=bars_per_year,
                support_mask=support_mask,
            )
            w = np.where(tradeable_mask, w, 0.0)
            last_w = w
            prof_alloc += time.perf_counter() - t0_alloc

            t0_eval = time.perf_counter()
            turnover = float(np.sum(np.abs(w - prev_w))) / 2.0
            all_turnovers.append(turnover)
            support_leak_count += int(np.sum((np.abs(w) > 1e-12) & ~support_mask))
            selected_idxs = [sym_to_idx[s] for s in selected if s in sym_to_idx]
            if selected_idxs:
                sel_idx_arr = np.array(selected_idxs, dtype=np.intp)
                eff_hurdle = hurdle * friction_safety_mult / max(rebalance_bars, 1)
                friction_pass = int(np.sum(np.abs(mu_arr[sel_idx_arr]) >= eff_hurdle[sel_idx_arr]))
            else:
                friction_pass = 0
            friction_pass_total += friction_pass
            signal_total += len(selected)

            # EW Bench: Top-K 선택 심볼만 동일가중 (Kelly 기여만 분리)
            n_sel = max(1, len(selected))
            w_base = np.array(
                [1.0 / n_sel if s in selected else 0.0 for s in symbols],
                dtype=np.float64,
            )
            rebal_cost = compute_rebalance_cost(
                previous_weights=prev_w,
                target_weights=w,
                round_trip_cost_bps=hurdle,
            )

            for t2 in range(t, t_end):
                if t2 + 1 >= aligned.close_2d.shape[0]:
                    break
                c_cur = aligned.close_2d[t2]
                c_nxt = aligned.close_2d[t2 + 1]
                bar_ret = np.where(c_cur > 0, (c_nxt - c_cur) / c_cur, 0.0)
                bar_ret = np.nan_to_num(bar_ret, nan=0.0, posinf=0.0, neginf=0.0)
                funding_rates = _resolve_funding_row(aligned=aligned, t2=t2, n_sym=n_sym)
                gross_ret = compute_futures_bar_return(
                    weights=w,
                    price_returns=bar_ret,
                    funding_rates=funding_rates,
                )
                # 거래비용은 리밸런싱 첫 bar에만 차감
                cost = rebal_cost if t2 == t else 0.0
                r_h = gross_ret - cost
                r_b = compute_futures_bar_return(
                    weights=w_base,
                    price_returns=bar_ret,
                    funding_rates=funding_rates,
                )
                all_rets_hybrid.append(r_h)
                all_rets_baseline.append(r_b)
                _fold_h.append(r_h)
                _fold_b.append(r_b)

            prev_selection = selected
            prev_w = w
            prof_eval += time.perf_counter() - t0_eval

        fold_rets_hybrid.append(_fold_h)
        fold_rets_baseline.append(_fold_b)

    logger.debug(
        "[L2-AWF-PROF] total=%.4fs | prep=%.4fs rank=%.4fs alloc=%.4fs eval=%.4fs",
        time.perf_counter() - t_start_total,
        prof_prep,
        prof_rank,
        prof_alloc,
        prof_eval,
    )

    return _AwfSimResult(
        rets_hybrid=all_rets_hybrid,
        rets_baseline=all_rets_baseline,
        last_selected=last_selected,
        last_w=last_w,
        all_turnovers=all_turnovers,
        friction_pass_total=friction_pass_total,
        signal_total=signal_total,
        support_leak_count=support_leak_count,
        fold_rets_hybrid=fold_rets_hybrid,
        fold_rets_baseline=fold_rets_baseline,
    )


def _stack_oos_signals(
    signals_per_fold: tuple[dict[str, SymbolSignal], ...],
    realized_stats: dict[str, SymbolRealizedStat] | None = None,
) -> dict[str, SymbolSignal]:
    """fold별 SymbolSignal을 per-symbol로 집계 (raw_mu 평균)."""
    sym_mu_lists: dict[str, list[float]] = defaultdict(list)
    sym_vol_lists: dict[str, list[float]] = defaultdict(list)
    sym_beta_lists: dict[str, list[float | None]] = defaultdict(list)

    for fold_sigs in signals_per_fold:
        for sym, sig in fold_sigs.items():
            sym_mu_lists[sym].append(sig.raw_mu)
            sym_vol_lists[sym].append(sig.volatility)
            sym_beta_lists[sym].append(sig.beta_btc)

    oos_stacked: dict[str, SymbolSignal] = {}
    for sym, mus in sym_mu_lists.items():
        real = realized_stats.get(sym) if realized_stats else None
        betas = [b for b in sym_beta_lists[sym] if b is not None]
        avg_beta: float | None = float(np.mean(betas)) if betas else None
        avg_vol = float(np.mean(sym_vol_lists[sym])) if sym_vol_lists[sym] else VOL_FLOOR
        oos_stacked[sym] = SymbolSignal(
            raw_mu=float(np.mean(mus)),
            volatility=avg_vol,
            n_obs=real.n_obs if real is not None else 0,
            t_stat=real.t_stat if real is not None else 0.0,
            valid=real.valid if real is not None else False,
            beta_btc=avg_beta,
        )
    return oos_stacked
