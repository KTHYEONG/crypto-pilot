# src/domain/futures/strategy/tiered_workflow/awf_sim.py

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.portfolio.portfolio_constructor import (
    PortfolioCaps,
    diagonal_kelly_weights,
)
from src.domain.futures.portfolio.signal_composer import (
    composer_sigma_lookback_bars,
    rolling_per_bar_return_std,
)
from src.domain.futures.strategy.cs_rank import (
    VOL_FLOOR,
    SymbolSignal,
    rank_and_select,
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
    fold_rets_hybrid: list[list[float]]    # fold별 strategy returns
    fold_rets_baseline: list[list[float]]  # fold별 baseline returns


def _run_awf_simulation(
    *,
    l1_oos: dict[str, SymbolSignal],
    aligned: AlignedMarketData,
    awf_folds: tuple[WFFold, ...],
    l2_params: dict[str, Any],
    caps: PortfolioCaps,
    tf: str = "4h",
    signals_per_fold: tuple[dict[str, SymbolSignal], ...] | None = None,
    l1_outer_folds: tuple[WFFold, ...] | None = None,
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

    k_rank = int(l2_params.get("K_RANK", 3))
    rank_buffer = int(l2_params.get("rank_buffer", 1))
    kelly_fraction = float(l2_params.get("kelly_fraction", 0.25))
    vol_target: float | None = l2_params.get("vol_target")
    if not isinstance(vol_target, float):
        vol_target = None
    no_trade_band = float(l2_params.get("no_trade_band", 0.01))
    rebalance_bars = int(l2_params.get("REBALANCE_BARS", 3))

    symbols = aligned.symbols
    n_sym = len(symbols)
    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    lookback = composer_sigma_lookback_bars(tf)

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
    fold_rets_hybrid: list[list[float]] = []
    fold_rets_baseline: list[list[float]] = []

    prev_selection: frozenset[str] = frozenset()
    prev_w: NDArray[np.float64] = np.zeros(n_sym, dtype=np.float64)
    last_selected: frozenset[str] = frozenset()
    last_w: NDArray[np.float64] = np.zeros(n_sym, dtype=np.float64)

    for _fold_idx, fold in enumerate(awf_folds):
        _fold_h: list[float] = []
        _fold_b: list[float] = []
        for t in range(fold.oos_start, fold.oos_end - 1, rebalance_bars):
            t_end = min(t + rebalance_bars, fold.oos_end - 1)

            t0_prep = time.perf_counter()
            valid_signals: dict[str, SymbolSignal] = {}
            
            fold_sigs = l1_oos
            if signals_per_fold is not None and l1_outer_folds is not None:
                matched_idx = None
                for f_idx, f in enumerate(l1_outer_folds):
                    if f.oos_start <= t < f.oos_end:
                        matched_idx = f_idx
                        break
                if matched_idx is not None and matched_idx < len(signals_per_fold):
                    fold_sigs = signals_per_fold[matched_idx]
            if not fold_sigs and l1_oos:
                fold_sigs = l1_oos

            for sym, sig in fold_sigs.items():
                if sym not in sym_to_idx:
                    continue
                i = sym_to_idx[sym]
                vol = float(vol_matrix[t, i])
                valid_signals[sym] = SymbolSignal(
                    raw_mu=sig.raw_mu,
                    volatility=vol,
                    n_obs=sig.n_obs,
                    t_stat=sig.t_stat,
                    valid=sig.valid,
                    beta_btc=sig.beta_btc,
                )
            prof_prep += time.perf_counter() - t0_prep

            t0_rank = time.perf_counter()
            selected, _z_scores = rank_and_select(
                valid_signals,
                k_rank=k_rank,
                sector_cap=n_sym,
                prev_selection=prev_selection,
                rank_buffer=rank_buffer,
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

            w = diagonal_kelly_weights(
                mu_bps=mu_arr,
                sigma=sig_arr,
                kelly_fraction=kelly_fraction,
                vol_target=vol_target,
                friction_hurdle_bps=hurdle,
                caps=caps,
                prev_w=prev_w,
                no_trade_band=no_trade_band,
                btc_beta=btc_beta,
            )
            last_w = w
            prof_alloc += time.perf_counter() - t0_alloc

            t0_eval = time.perf_counter()
            turnover = float(np.sum(np.abs(w - prev_w))) / 2.0
            all_turnovers.append(turnover)
            selected_idxs = [sym_to_idx[s] for s in selected if s in sym_to_idx]
            if selected_idxs:
                sel_idx_arr = np.array(selected_idxs, dtype=np.intp)
                friction_pass = int(np.sum(np.abs(mu_arr[sel_idx_arr]) >= hurdle[sel_idx_arr]))
            else:
                friction_pass = 0
            friction_pass_total += friction_pass
            signal_total += max(1, len(selected))

            n_valid_sym = max(1, sum(1 for ss in valid_signals.values() if ss.valid))
            w_base = np.array(
                [
                    1.0 / n_valid_sym
                    if s in valid_signals and valid_signals[s].valid
                    else 0.0
                    for s in symbols
                ],
                dtype=np.float64,
            )

            # 리밸런싱 비용: 편도 회전율 * taker bps (bps -> fraction)
            avg_hurdle = float(np.mean(hurdle)) if hurdle.size > 0 else 3.8
            rebal_cost = turnover * avg_hurdle * 1e-4

            for t2 in range(t, t_end):
                if t2 + 1 >= aligned.close_2d.shape[0]:
                    break
                c_cur = aligned.close_2d[t2]
                c_nxt = aligned.close_2d[t2 + 1]
                bar_ret = np.where(c_cur > 0, (c_nxt - c_cur) / c_cur, 0.0)
                bar_ret = np.nan_to_num(bar_ret, nan=0.0, posinf=0.0, neginf=0.0)
                gross_ret = float(np.dot(w, bar_ret))
                # 거래비용은 리밸런싱 첫 bar에만 차감
                cost = rebal_cost if t2 == t else 0.0
                r_h = gross_ret - cost
                r_b = float(np.dot(w_base, bar_ret))
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
