# src/domain/futures/strategy/tiered_workflow/awf_sim.py

from __future__ import annotations

import re as _re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numba
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.core.utils.utils import PERF
from src.domain.futures.portfolio.portfolio_constructor import (
    PortfolioCaps,
    diagonal_kelly_weights,
    project_all_caps,
)
from src.domain.futures.portfolio.signal_composer import (
    composer_sigma_lookback_bars,
    hours_per_bar_tf,
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
    L2SimulationCache,
    Layer2AllocationConfig,
    Layer2SignalSchedule,
)

if TYPE_CHECKING:
    from src.domain.futures.strategy.common.alignment import AlignedMarketData
    from src.domain.futures.strategy.tiered_workflow.dataclasses import SymbolRealizedStat
    from src.domain.futures.strategy.walk_forward import WFFold

EdgeBasis = Literal["gross", "net"]


@dataclass(slots=True, frozen=True)
class Layer2ExpectedEdge:
    """Layer2 sizing에 사용할 per-bar edge contract."""

    signed_gross_bps_per_bar: float
    signed_net_bps_per_bar: float
    expected_cost_bps_per_bar: float
    basis: EdgeBasis


@dataclass(slots=True, frozen=True)
class Layer2FoldAttribution:
    fold_idx: int
    oos_bars: int
    n_rebal: int
    realized_total: float
    realized_price: float
    realized_funding: float
    realized_cost: float
    expected_net: float
    alpha_gap: float
    mean_gross_exp: float
    mean_net_exp: float
    sleeves_active_mean: float
    friction_pass_ratio: float
    throttle_mult_mean: float
    dropped_below_cost: int
    netting_events: int


def compute_cost_drag_ratio(
    fold_attributions: tuple[Layer2FoldAttribution, ...],
    *,
    eps: float = 1e-9,
) -> float:
    total_cost = sum(a.realized_cost for a in fold_attributions)
    total_price_abs = sum(abs(a.realized_price) for a in fold_attributions)
    ratio = total_cost / max(total_price_abs, eps)
    return min(float(ratio), 100.0)


def _assemble_fold_attribution(
    *,
    fold_idx: int,
    oos_bars: int,
    n_rebal: int,
    realized_price: float,
    realized_funding: float,
    realized_cost: float,
    expected_net: float,
    gross_exps: list[float],
    net_exps: list[float],
    throttle_mults: list[float],
    sleeves_active: list[int],
    friction_pass_total: int,
    signal_total: int,
    dropped_below_cost: int,
    netting_events: int,
) -> Layer2FoldAttribution:
    realized_total = realized_price + realized_funding - realized_cost
    alpha_gap = realized_total - expected_net
    friction_pass_ratio = friction_pass_total / signal_total if signal_total > 0 else 0.0
    throttle_mult_mean = float(np.mean(throttle_mults)) if throttle_mults else 1.0
    mean_gross_exp = float(np.mean(gross_exps)) if gross_exps else 0.0
    mean_net_exp = float(np.mean(net_exps)) if net_exps else 0.0
    sleeves_active_mean = float(np.mean(sleeves_active)) if sleeves_active else 0.0

    def _safe(v: float) -> float:
        return v if np.isfinite(v) else 0.0

    return Layer2FoldAttribution(
        fold_idx=fold_idx,
        oos_bars=oos_bars,
        n_rebal=n_rebal,
        realized_total=_safe(realized_total),
        realized_price=_safe(realized_price),
        realized_funding=_safe(realized_funding),
        realized_cost=_safe(realized_cost),
        expected_net=_safe(expected_net),
        alpha_gap=_safe(alpha_gap),
        mean_gross_exp=_safe(mean_gross_exp),
        mean_net_exp=_safe(mean_net_exp),
        sleeves_active_mean=_safe(sleeves_active_mean),
        friction_pass_ratio=_safe(friction_pass_ratio),
        throttle_mult_mean=_safe(throttle_mult_mean),
        dropped_below_cost=dropped_below_cost,
        netting_events=netting_events,
    )


@dataclass(slots=True)
class _AwfSimResult:
    """run_awf 내부 시뮬레이션 결과 (private)."""

    rets_hybrid: list[float]
    rets_baseline: list[float]
    last_selected: frozenset[str]
    last_w: NDArray[np.float64]
    all_turnovers: list[float]
    all_turnovers_baseline: list[float]
    all_gross_exposures: list[float]
    all_net_exposures: list[float]
    friction_pass_total: int
    signal_total: int
    support_leak_count: int
    total_cost_hybrid: float
    total_cost_baseline: float
    cap_saturation_count: int
    rebalance_count: int
    trade_count: int
    fold_rets_hybrid: list[list[float]]    # fold별 strategy returns
    fold_rets_baseline: list[list[float]]  # fold별 baseline returns
    fold_selected_symbols: tuple[tuple[str, ...], ...]
    block_rets_hybrid: tuple[tuple[float, ...], ...]
    block_rets_baseline: tuple[tuple[float, ...], ...]
    rets_baseline_ew: list[float]          # 순수 1/N EW baseline (uplift 측정 전용)
    fit_rets_hybrid: tuple[float, ...] = ()  # D3: fit-leg 수익률 (look-ahead-free L* calibration용)
    fold_attributions: tuple[Layer2FoldAttribution, ...] = ()


def _book_edge_score(
    w: NDArray[np.float64],
    mu_bps: NDArray[np.float64],  # already net-of-cost (per-bar)
) -> float:
    """사이징된 비중의 gross-weighted 평균 net edge (bps/bar). mu_bps는 이미 net.

    Returns:
        0.0 if no active positions.
    """
    abs_w = np.abs(w)
    den = float(np.sum(abs_w))
    if den < 1e-12:
        return 0.0
    return float(np.dot(abs_w, np.abs(mu_bps)) / den)


def _edge_throttle_multiplier(
    score_bps: float,
    *,
    floor_bps: float,
    ref_bps: float,
    gamma: float,
    min_active_mult: float = 0.0,
) -> float:
    """score_bps → [0, 1] throttle 승수. 선형(gamma=1) 또는 볼록(gamma>1).

    Returns:
        0.0 when score <= floor, 1.0 when score >= ref.
    """
    if not np.isfinite(score_bps):
        return 0.0
    span = max(ref_bps - floor_bps, 1e-9)
    x = float(np.clip((score_bps - floor_bps) / span, 0.0, 1.0))
    raw = float(x ** max(gamma, 1e-9))
    if raw <= 0.0:
        return 0.0
    floor = float(np.clip(min_active_mult, 0.0, 1.0))
    return float(floor + (1.0 - floor) * raw)


def _estimate_annual_vol(
    weights: NDArray[np.float64],
    sigma: NDArray[np.float64],
    bars_per_year: float,
) -> float:
    """Estimate annualized portfolio volatility from diagonal per-bar sigma."""
    w = np.asarray(weights, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64)
    if w.size == 0 or sig.size != w.size:
        return 0.0
    sigma_port_bar = float(np.sqrt(float(np.dot(w**2, sig**2))))
    ann_vol = sigma_port_bar * float(np.sqrt(max(bars_per_year, 1e-12)))
    return float(ann_vol) if np.isfinite(ann_vol) else 0.0


def _apply_risk_budget_floor(
    *,
    weights: NDArray[np.float64],
    sigma: NDArray[np.float64],
    bars_per_year: float,
    vol_target: float | None,
    floor_ratio: float,
    max_scale: float,
    caps: PortfolioCaps,
    btc_beta: NDArray[np.float64] | None,
    support_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Scale an under-deployed book toward a target-vol floor without adding support."""
    w = np.asarray(weights, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64)
    support = np.asarray(support_mask, dtype=np.bool_)
    if (
        vol_target is None
        or floor_ratio <= 0.0
        or max_scale <= 1.0
        or w.size == 0
        or sig.size != w.size
        or support.size != w.size
        or float(np.sum(np.abs(w))) <= 1e-12
    ):
        return w

    sigma_port_bar = float(np.sqrt(float(np.dot(w**2, sig**2))))
    ann_vol = _estimate_annual_vol(w, sig, bars_per_year)
    floor_ann_vol = float(vol_target) * float(floor_ratio)
    if (
        not np.isfinite(ann_vol)
        or not np.isfinite(floor_ann_vol)
        or ann_vol <= 1e-12
        or ann_vol >= floor_ann_vol
    ):
        return w

    scale = min(floor_ann_vol / ann_vol, float(max_scale))
    if not np.isfinite(scale) or scale <= 1.0:
        return w

    beta = (
        np.zeros(w.size, dtype=np.float64)
        if btc_beta is None or btc_beta.size != w.size
        else np.asarray(btc_beta, dtype=np.float64)
    )
    scaled = np.where(support, w * scale, 0.0)
    scaled_sigma_port_bar = sigma_port_bar * scale
    effective_caps = replace(caps, target_ann_vol=float(vol_target))
    projected = project_all_caps(
        scaled,
        beta,
        scaled_sigma_port_bar,
        bars_per_year,
        effective_caps,
        support_mask=support,
        # allow_vol_upscale 기본값 False 유지: scale로 이미 확대 완료, 이중확대 방지
    )
    return np.asarray(np.where(support, projected, 0.0), dtype=np.float64)


def _resolve_adaptive_k_rank(
    *,
    base_k: int,
    n_valid: int,
    prev_weights: NDArray[np.float64],
    sigma: NDArray[np.float64],
    bars_per_year: float,
    vol_target: float | None,
    expand_below_vol_ratio: float,
    max_extra: int,
) -> int:
    """Expand breadth when the prior book used too little of the risk budget."""
    bounded_base = max(0, min(int(base_k), int(n_valid)))
    if (
        bounded_base <= 0
        or n_valid <= 0
        or vol_target is None
        or expand_below_vol_ratio <= 0.0
        or max_extra <= 0
    ):
        return bounded_base
    prev_ann_vol = _estimate_annual_vol(prev_weights, sigma, bars_per_year)
    trigger_vol = float(vol_target) * float(expand_below_vol_ratio)
    if not np.isfinite(prev_ann_vol) or not np.isfinite(trigger_vol) or prev_ann_vol >= trigger_vol:
        return bounded_base
    return min(bounded_base + int(max_extra), int(n_valid))


def _event_strength(event: ValidatedSignalEvent) -> float:
    return (
        abs(float(event.expected_net_bps))
        / max(int(event.expected_holding_bars), 1)
        * max(float(event.quality_weight), 0.0)
    )


def _is_better_event(candidate: ValidatedSignalEvent, incumbent: ValidatedSignalEvent) -> bool:
    if candidate.decision_idx != incumbent.decision_idx:
        return bool(candidate.decision_idx > incumbent.decision_idx)
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


def compute_expected_layer2_edge(
    *,
    side: int,
    expected_gross_bps: float,
    expected_net_bps: float,
    expected_holding_bars: int,
    execution_cost_bps: float,
    edge_basis: EdgeBasis,
    fixed_cost_safety_mult: float,
) -> Layer2ExpectedEdge:
    """Gross/net event prediction을 보수적 per-bar net edge로 정규화한다."""
    if fixed_cost_safety_mult < 1.0:
        raise ValueError("fixed_cost_safety_mult must be >= 1.0")

    direction = 1.0 if side >= 0 else -1.0
    holding_bars = max(int(expected_holding_bars), 1)
    gross_per_bar = direction * float(expected_gross_bps) / float(holding_bars)
    net_per_bar = direction * float(expected_net_bps) / float(holding_bars)
    cost_per_bar = float(execution_cost_bps) * float(fixed_cost_safety_mult) / float(holding_bars)

    if edge_basis == "gross":
        signed_net = np.sign(gross_per_bar) * max(abs(gross_per_bar) - cost_per_bar, 0.0)
        return Layer2ExpectedEdge(
            signed_gross_bps_per_bar=gross_per_bar,
            signed_net_bps_per_bar=float(signed_net),
            expected_cost_bps_per_bar=cost_per_bar,
            basis=edge_basis,
        )

    return Layer2ExpectedEdge(
        signed_gross_bps_per_bar=gross_per_bar,
        signed_net_bps_per_bar=net_per_bar,
        expected_cost_bps_per_bar=cost_per_bar,
        basis=edge_basis,
    )


def build_directional_risk_matched_equal_weight(
    *,
    signed_net_mu_bps: NDArray[np.float64],
    strategy_weights: NDArray[np.float64],
    sigma: NDArray[np.float64],
    btc_beta: NDArray[np.float64],
    caps: PortfolioCaps,
    bars_per_year: float,
) -> NDArray[np.float64]:
    """같은 support/방향에서 전략과 ex-ante risk를 맞추는 directional EW baseline."""
    support = np.abs(np.asarray(strategy_weights, dtype=np.float64)) > 1e-12
    if not np.any(support):
        return np.zeros_like(strategy_weights, dtype=np.float64)

    sigma_arr = np.maximum(np.asarray(sigma, dtype=np.float64), VOL_FLOOR)
    direction = np.sign(np.asarray(signed_net_mu_bps, dtype=np.float64))
    if direction.shape != support.shape:
        direction = np.sign(np.asarray(strategy_weights, dtype=np.float64))
    direction = np.where(support, direction, 0.0)
    if not np.any(direction != 0.0):
        return np.zeros_like(strategy_weights, dtype=np.float64)

    inv_sigma = np.where(support, 1.0 / sigma_arr, 0.0)
    baseline = direction * inv_sigma

    strategy_sigma = float(np.sqrt(np.dot(np.asarray(strategy_weights, dtype=np.float64) ** 2, sigma_arr**2)))
    baseline_sigma = float(np.sqrt(np.dot(baseline**2, sigma_arr**2)))
    strategy_gross = float(np.sum(np.abs(strategy_weights)))
    baseline_gross = float(np.sum(np.abs(baseline)))

    if baseline_sigma > 1e-12 and strategy_sigma > 1e-12:
        baseline = baseline * (strategy_sigma / baseline_sigma)
    elif baseline_gross > 1e-12 and strategy_gross > 1e-12:
        baseline = baseline * (strategy_gross / baseline_gross)
    else:
        return np.zeros_like(strategy_weights, dtype=np.float64)

    sigma_port = float(np.sqrt(np.dot(np.clip(baseline, -caps.per_symbol, caps.per_symbol) ** 2, sigma_arr**2)))
    return project_all_caps(
        baseline,
        np.asarray(btc_beta, dtype=np.float64),
        sigma_port,
        bars_per_year,
        caps,
        support_mask=support,
    )


def build_directional_equal_weight_baseline(
    *,
    signed_net_mu_bps: NDArray[np.float64],
    strategy_weights: NDArray[np.float64],
    sigma: NDArray[np.float64],
    btc_beta: NDArray[np.float64],
    caps: PortfolioCaps,
    bars_per_year: float,
) -> NDArray[np.float64]:
    """Uplift 측정용 순수 1/N EW baseline (sigma-match 미적용).

    Args:
        signed_net_mu_bps: 심볼별 signed net edge (방향 결정용).
        strategy_weights: 전략 비중 (support 마스크 결정용).
        sigma: 심볼별 변동성 (cap 투영에만 사용; sizing 미사용).
        btc_beta: BTC beta 배열 (cap 투영 전달용).
        caps: PortfolioCaps (per_symbol/gross/net/beta cap).
        bars_per_year: 연율화 팩터.

    Returns:
        순수 1/N 방향성 비중 (cap 투영 후). support 없으면 zeros.

    Time Complexity: O(K) — K=support 심볼 수.
    Space Complexity: O(N) — N=전체 심볼 수.
    """
    support = np.abs(np.asarray(strategy_weights, dtype=np.float64)) > 1e-12
    if not np.any(support):
        return np.zeros_like(strategy_weights, dtype=np.float64)

    n_support = int(np.sum(support))
    direction = np.sign(np.asarray(signed_net_mu_bps, dtype=np.float64))
    if direction.shape != support.shape:
        direction = np.sign(np.asarray(strategy_weights, dtype=np.float64))
    direction = np.where(support, direction, 0.0)
    if not np.any(direction != 0.0):
        return np.zeros_like(strategy_weights, dtype=np.float64)

    # 순수 1/N: sigma-match 없음, 단순 방향 균등 비중
    w = direction / float(n_support)

    sigma_arr = np.maximum(np.asarray(sigma, dtype=np.float64), VOL_FLOOR)
    sigma_port = float(
        np.sqrt(np.dot(np.clip(w, -caps.per_symbol, caps.per_symbol) ** 2, sigma_arr**2))
    )
    return project_all_caps(
        w,
        np.asarray(btc_beta, dtype=np.float64),
        sigma_port,
        bars_per_year,
        caps,
        support_mask=support,
    )


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
    execution_eligibility_mask = _mask_row("execution_eligibility_mask", True)
    strategy_readiness_mask = _mask_row("strategy_readiness_mask", True)
    promotion_active_mask = _mask_row("promotion_active_mask", True)
    entry_block_mask = _mask_row("entry_block_mask", False)
    kill_mask = _mask_row("kill_mask", False)
    return (
        active_mask
        & warm_mask
        & execution_eligibility_mask
        & strategy_readiness_mask
        & promotion_active_mask
        & ~entry_block_mask
        & ~kill_mask
    )


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


def _is_cap_saturated(
    *,
    weights: NDArray[np.float64],
    btc_beta: NDArray[np.float64],
    caps: PortfolioCaps,
) -> bool:
    gross = float(np.sum(np.abs(weights)))
    net = float(np.sum(weights))
    beta_exp = float(np.dot(weights, btc_beta))
    per_symbol_hit = bool(np.any(np.isclose(np.abs(weights), caps.per_symbol, atol=1e-9)))
    return bool(
        per_symbol_hit
        or np.isclose(gross, caps.gross, atol=1e-9)
        or np.isclose(abs(net), caps.net, atol=1e-9)
        or np.isclose(abs(beta_exp), caps.beta, atol=1e-9)
    )


@numba.njit(cache=True)  # type: ignore[untyped-decorator]
def _scatter_signals_jit(
    decision_idxs: NDArray[np.int64],
    holding_bars_arr: NDArray[np.int64],
    sleeve_js: NDArray[np.int64],
    gross_vals: NDArray[np.float64],
    net_vals: NDArray[np.float64],
    side_vals: NDArray[np.float64],
    qw_vals: NDArray[np.float64],
    strengths: NDArray[np.float64],
    expected_gross_bps_2d: NDArray[np.float64],
    expected_net_bps_2d: NDArray[np.float64],
    holding_bars_2d: NDArray[np.float64],
    side_2d: NDArray[np.float64],
    quality_weight_2d: NDArray[np.float64],
    event_strength_2d: NDArray[np.float64],
    signal_mask_2d: NDArray[np.bool_],
    t_max: int,
) -> None:
    for e in range(len(decision_idxs)):
        sleeve_j = sleeve_js[e]
        start = decision_idxs[e] + 1
        end = min(t_max, start + holding_bars_arr[e])
        if start >= end:
            continue
        
        g_val = gross_vals[e]
        n_val = net_vals[e]
        h_bars = float(holding_bars_arr[e])
        s_val = side_vals[e]
        q_val = qw_vals[e]
        str_val = strengths[e]
        
        for t in range(start, end):
            if not signal_mask_2d[t, sleeve_j]:
                signal_mask_2d[t, sleeve_j] = True
                expected_gross_bps_2d[t, sleeve_j] = g_val
                expected_net_bps_2d[t, sleeve_j] = n_val
                holding_bars_2d[t, sleeve_j] = h_bars
                side_2d[t, sleeve_j] = s_val
                quality_weight_2d[t, sleeve_j] = q_val
                event_strength_2d[t, sleeve_j] = str_val
            elif str_val > event_strength_2d[t, sleeve_j]:
                expected_gross_bps_2d[t, sleeve_j] = g_val
                expected_net_bps_2d[t, sleeve_j] = n_val
                holding_bars_2d[t, sleeve_j] = h_bars
                side_2d[t, sleeve_j] = s_val
                quality_weight_2d[t, sleeve_j] = q_val
                event_strength_2d[t, sleeve_j] = str_val


def _build_tradeable_mask_vectorized(
    aligned: AlignedMarketData,
    t_max: int,
    n_sym: int,
) -> NDArray[np.bool_]:
    def _mask_2d(name: str, default: bool) -> NDArray[np.bool_]:
        value = getattr(aligned, name, None)
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] >= t_max and value.shape[1] == n_sym:
            return np.asarray(value[:t_max], dtype=np.bool_)
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == n_sym:
            res = np.full((t_max, n_sym), default, dtype=np.bool_)
            valid_len = min(t_max, value.shape[0])
            res[:valid_len] = np.asarray(value[:valid_len], dtype=np.bool_)
            return res
        return np.full((t_max, n_sym), default, dtype=np.bool_)

    active_mask = _mask_2d("active_mask", True)
    warm_mask = _mask_2d("warm_mask", True)
    execution_eligibility_mask = _mask_2d("execution_eligibility_mask", True)
    strategy_readiness_mask = _mask_2d("strategy_readiness_mask", True)
    promotion_active_mask = _mask_2d("promotion_active_mask", True)
    entry_block_mask = _mask_2d("entry_block_mask", False)
    kill_mask = _mask_2d("kill_mask", False)
    return (
        active_mask
        & warm_mask
        & execution_eligibility_mask
        & strategy_readiness_mask
        & promotion_active_mask
        & ~entry_block_mask
        & ~kill_mask
    )


def build_l2_simulation_cache(
    aligned: AlignedMarketData,
    signal_batch: ValidatedSignalBatch,
    tf: str,
) -> L2SimulationCache:
    """L2 시뮬레이션용 사전 계산 행렬 빌드 (Sleeve 차원 도입).

    신호 행렬은 ``[T, S]`` (S = n_sleeves = unique (symbol, strategy_id) 수)로 구성된다.
    같은 symbol의 복수 TF 신호가 각각 독립 sleeve에 보존되어 multi-TF edge collapse 방지.
    심볼 단위 행렬(vol/tradeable/hurdle/funding/beta)은 ``[T, N]``으로 유지.

    ``sleeve_to_sym[j]`` 를 통해 sleeve col j → symbol col 매핑이 제공된다.
    ``_run_awf_simulation``은 이 매핑으로 ``w_sleeve[T,S] → w_sym[T,N]`` netting을 수행한다.

    Args:
        aligned: 공통 base grid AlignedMarketData.
        signal_batch: ValidatedSignalBatch (전 TF 병합 이벤트).
        tf: Annualization/vol lookback 기준 TF 문자열.

    Returns:
        L2SimulationCache: sleeve 차원 행렬 포함.

    Time Complexity: O(T·S + E·H) where E=n_events, H=avg_holding_bars, S=n_sleeves.
    Space Complexity: O(T·(S+N)) where N=n_sym.
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)

    lookback = composer_sigma_lookback_bars(tf)
    n_sym = len(aligned.symbols)
    t_max = aligned.close_2d.shape[0]

    # ── Symbol-level matrices [T, N] ──────────────────────────────────────────
    c_close = np.asarray(aligned.close_2d, dtype=np.float64)
    r_rets = np.zeros((t_max, n_sym), dtype=np.float64)
    if t_max > 1:
        r_rets[1:] = (c_close[1:] - c_close[:-1]) / np.maximum(np.abs(c_close[:-1]), 1e-12)
    rw = max(2, int(lookback))
    s_rolling = pd.DataFrame(r_rets).rolling(rw, min_periods=2).std(ddof=1)
    vol_matrix_2d = s_rolling.to_numpy(dtype=np.float64)
    vol_matrix_2d = np.nan_to_num(vol_matrix_2d, nan=VOL_FLOOR, posinf=VOL_FLOOR, neginf=VOL_FLOOR)
    vol_matrix_2d = np.maximum(vol_matrix_2d, VOL_FLOOR)

    tradeable_mask_2d = _build_tradeable_mask_vectorized(aligned=aligned, t_max=t_max, n_sym=n_sym)

    if aligned.execution_cost_bps_2d is not None:
        hurdle_2d = np.nan_to_num(aligned.execution_cost_bps_2d, nan=3.8, posinf=100.0, neginf=100.0)
    else:
        hurdle_2d = np.full((t_max, n_sym), 3.8, dtype=np.float64)

    if aligned.funding_2d is not None:
        funding_2d = np.nan_to_num(aligned.funding_2d, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        funding_2d = np.zeros((t_max, n_sym), dtype=np.float64)

    if aligned.beta_vs_market_1d is not None:
        beta_1d = np.nan_to_num(aligned.beta_vs_market_1d, nan=0.0).astype(np.float64)
    else:
        beta_1d = np.zeros(n_sym, dtype=np.float64)

    sym_to_idx: dict[str, int] = {s: i for i, s in enumerate(aligned.symbols)}

    # ── Sleeve 인덱싱 ─────────────────────────────────────────────────────────
    # sleeve_ids: 결정적 정렬 — (symbol, strategy_id) unique 집합
    # Shape track: sleeve_ids: [S], sleeve_to_sym: [S], 신호 행렬: [T, S]
    sleeve_id_set: set[tuple[str, str]] = set()
    for event in signal_batch.events:
        if sym_to_idx.get(event.symbol) is not None:
            sleeve_id_set.add((event.symbol, event.strategy_id))

    sleeve_ids_sorted: tuple[tuple[str, str], ...] = tuple(sorted(sleeve_id_set))
    n_sleeve = len(sleeve_ids_sorted)
    sleeve_to_idx: dict[tuple[str, str], int] = {sid: j for j, sid in enumerate(sleeve_ids_sorted)}
    # sleeve_to_sym[j] = symbol col idx (underlying symbol의 vol/beta 참조용)
    sleeve_to_sym_arr: NDArray[np.int64] = np.array(
        [sym_to_idx[sid[0]] for sid in sleeve_ids_sorted],
        dtype=np.int64,
    ) if n_sleeve > 0 else np.empty(0, dtype=np.int64)

    _log.debug(
        "[BUILD-CACHE] n_sym=%d n_sleeve=%d n_events=%d tf=%s",
        n_sym, n_sleeve, len(signal_batch.events), tf,
    )

    if n_sleeve == 0:
        # 빈 배치: 빈 [T, 0] 신호 행렬 반환, crash 없음
        empty_t_s = np.zeros((t_max, 0), dtype=np.float64)
        empty_t_s_bool = np.zeros((t_max, 0), dtype=np.bool_)
        return L2SimulationCache(
            vol_matrix_2d=vol_matrix_2d,
            tradeable_mask_2d=tradeable_mask_2d,
            hurdle_2d=hurdle_2d,
            funding_2d=funding_2d,
            beta_1d=beta_1d,
            expected_gross_bps_2d=empty_t_s,
            expected_net_bps_2d=empty_t_s.copy(),
            holding_bars_2d=empty_t_s.copy(),
            side_2d=empty_t_s.copy(),
            quality_weight_2d=empty_t_s.copy(),
            signal_mask_2d=empty_t_s_bool,
            sleeve_to_sym=sleeve_to_sym_arr,
            sleeve_ids=sleeve_ids_sorted,
            sleeve_to_tf=(),
        )

    # ── Sleeve 신호 행렬 [T, S] ───────────────────────────────────────────────
    # sleeve가 unique (symbol, strategy_id) key이므로 같은 bar·sleeve 충돌은
    # 같은 strategy의 동일 bar 복수 이벤트(rare)만 발생 → strength tie-break.
    expected_gross_bps_2d = np.zeros((t_max, n_sleeve), dtype=np.float64)
    expected_net_bps_2d = np.zeros((t_max, n_sleeve), dtype=np.float64)
    holding_bars_2d = np.ones((t_max, n_sleeve), dtype=np.float64)
    side_2d = np.zeros((t_max, n_sleeve), dtype=np.float64)
    quality_weight_2d = np.zeros((t_max, n_sleeve), dtype=np.float64)
    event_strength_2d = np.zeros((t_max, n_sleeve), dtype=np.float64)
    signal_mask_2d = np.zeros((t_max, n_sleeve), dtype=np.bool_)

    decision_idxs = []
    holding_bars_arr = []
    sleeve_js = []
    gross_vals = []
    net_vals = []
    side_vals = []
    qw_vals = []
    strengths = []
    
    for event in signal_batch.events:
        sleeve_key = (event.symbol, event.strategy_id)
        sleeve_j = sleeve_to_idx.get(sleeve_key)
        if sleeve_j is None:
            continue
        sleeve_js.append(sleeve_j)
        decision_idxs.append(int(event.decision_idx))
        holding_bars_arr.append(max(int(event.expected_holding_bars), 1))
        side_vals.append(float(event.side))
        gross_vals.append(float(event.expected_gross_bps))
        net_vals.append(float(event.expected_net_bps))
        qw_vals.append(float(event.quality_weight))
        strengths.append(float(_event_strength(event)))

    if len(sleeve_js) > 0:
        _scatter_signals_jit(
            np.array(decision_idxs, dtype=np.int64),
            np.array(holding_bars_arr, dtype=np.int64),
            np.array(sleeve_js, dtype=np.int64),
            np.array(gross_vals, dtype=np.float64),
            np.array(net_vals, dtype=np.float64),
            np.array(side_vals, dtype=np.float64),
            np.array(qw_vals, dtype=np.float64),
            np.array(strengths, dtype=np.float64),
            expected_gross_bps_2d,
            expected_net_bps_2d,
            holding_bars_2d,
            side_2d,
            quality_weight_2d,
            event_strength_2d,
            signal_mask_2d,
            t_max,
        )

    # [L2-TFDIAG] TF별 holding/edge/decision_idx 단위 정합성 진단 (multi-TF 불협화음 검출)
    if _log.isEnabledFor(_logging.DEBUG) and len(sleeve_js) > 0:
        _tf_agg: dict[str, dict[str, list[float]]] = {}
        for _ei in range(len(sleeve_js)):
            _sid = sleeve_ids_sorted[sleeve_js[_ei]][1]
            _m = _re.search(r"_(\d+h)\b", _sid) or _re.search(r"(\d+h)$", _sid)
            _tfk = _m.group(1) if _m else "unk"
            _b = _tf_agg.setdefault(_tfk, {"hold": [], "gross": [], "net": [], "didx": []})
            _b["hold"].append(float(holding_bars_arr[_ei]))
            _b["gross"].append(float(gross_vals[_ei]))
            _b["net"].append(float(net_vals[_ei]))
            _b["didx"].append(float(decision_idxs[_ei]))
        for _tfk in sorted(_tf_agg):
            _b = _tf_agg[_tfk]
            _n = len(_b["hold"])
            def _mean(xs: list[float]) -> float:
                return float(sum(xs) / len(xs)) if xs else 0.0
            _log.debug(
                "[L2-TFDIAG] tf=%s n_events=%d mean_hold_bars=%.2f mean_gross_bps=%.1f "
                "mean_net_bps=%.1f mean_per_bar_net=%.2f didx_min=%.0f didx_max=%.0f",
                _tfk, _n, _mean(_b["hold"]), _mean(_b["gross"]), _mean(_b["net"]),
                _mean(_b["net"]) / max(_mean(_b["hold"]), 1.0),
                min(_b["didx"]), max(_b["didx"]),
            )

    return L2SimulationCache(
        vol_matrix_2d=vol_matrix_2d,
        tradeable_mask_2d=tradeable_mask_2d,
        hurdle_2d=hurdle_2d,
        funding_2d=funding_2d,
        beta_1d=beta_1d,
        expected_gross_bps_2d=expected_gross_bps_2d,
        expected_net_bps_2d=expected_net_bps_2d,
        holding_bars_2d=holding_bars_2d,
        side_2d=side_2d,
        quality_weight_2d=quality_weight_2d,
        signal_mask_2d=signal_mask_2d,
        sleeve_to_sym=sleeve_to_sym_arr,
        sleeve_ids=sleeve_ids_sorted,
        sleeve_to_tf=tuple(_parse_tf_from_strategy_id(sid[1]) for sid in sleeve_ids_sorted),
    )

def _parse_tf_from_strategy_id(strategy_id: str) -> str:
    """strategy_id 접미사에서 TF 문자열을 추출한다.

    Args:
        strategy_id: 전략 식별자 (예: ``donchian_72_8h``).

    Returns:
        TF 문자열 (예: ``"8h"``), 미매치 시 ``"unk"``.
    """
    _m = _re.search(r"_(\d+h)\b", strategy_id) or _re.search(r"(\d+h)$", strategy_id)
    return _m.group(1) if _m else "unk"


def compute_per_tf_fit_edge(
    cache: L2SimulationCache,
    aligned: AlignedMarketData,
    fit_start: int,
    fit_end: int,
) -> dict[str, float]:
    """fit-leg 구간 각 TF의 방향 hit 엣지(look-ahead-free).

    ``per_tf_edge[tf] = mean over (active sleeve j of TF, bar t in [fit_start, fit_end))
    of side_j(t) * forward_return(symbol_of_j, t)``

    ``forward_return = (close[t+1]-close[t])/close[t]``. t+1>=T면 skip.

    Args:
        cache: L2SimulationCache (sleeve 행렬 포함).
        aligned: AlignedMarketData (close_2d 필요).
        fit_start: fit-leg 시작 bar index (inclusive).
        fit_end: fit-leg 종료 bar index (exclusive).

    Returns:
        TF → 평균 방향 엣지 dict. 빈 TF는 0.0.
    """
    n_sleeve = cache.signal_mask_2d.shape[1]
    if n_sleeve == 0 or fit_start >= fit_end:
        return {}

    t_max, _ = cache.signal_mask_2d.shape
    if fit_end > t_max:
        fit_end = t_max

    close_2d = np.asarray(aligned.close_2d, dtype=np.float64)
    n_sym = close_2d.shape[1]
    forward_ret_2d = np.zeros((t_max, n_sym), dtype=np.float64)
    if t_max > 1:
        c = close_2d
        forward_ret_2d[1:] = np.where(
            c[:-1] > 0,
            (c[1:] - c[:-1]) / np.maximum(np.abs(c[:-1]), 1e-12),
            0.0,
        )

    sleeve_to_tf_arr: tuple[str, ...] = cache.sleeve_to_tf
    if len(sleeve_to_tf_arr) != n_sleeve:
        return {}

    tf_edges: dict[str, list[float]] = {}
    for t in range(fit_start, fit_end):
        if t + 1 >= t_max:
            break
        mask_row = cache.signal_mask_2d[t]
        active_js = np.where(mask_row)[0]
        if len(active_js) == 0:
            continue
        side_row = cache.side_2d[t]
        for j in active_js:
            tf_key = sleeve_to_tf_arr[int(j)]
            sym_col = int(cache.sleeve_to_sym[int(j)])
            fwd = float(forward_ret_2d[t, sym_col])
            hit = float(side_row[int(j)]) * fwd
            tf_edges.setdefault(tf_key, []).append(hit)

    return {tf: float(np.mean(vals)) if vals else 0.0 for tf, vals in tf_edges.items()}


def compute_per_sleeve_realized_edge(
    cache: L2SimulationCache,
    aligned: AlignedMarketData,
    start: int,
    end: int,
) -> NDArray[np.float64]:
    n_sleeve = cache.signal_mask_2d.shape[1]
    if n_sleeve == 0 or start >= end:
        return np.full(n_sleeve, np.nan, dtype=np.float64)

    t_max, _ = cache.signal_mask_2d.shape
    if end > t_max:
        end = t_max

    close_2d = np.asarray(aligned.close_2d, dtype=np.float64)
    n_sym = close_2d.shape[1]
    forward_ret_2d = np.zeros((t_max, n_sym), dtype=np.float64)
    if t_max > 1:
        c = close_2d
        forward_ret_2d[1:] = np.where(
            c[:-1] > 0,
            (c[1:] - c[:-1]) / np.maximum(np.abs(c[:-1]), 1e-12),
            0.0,
        )

    sleeve_to_sym = cache.sleeve_to_sym
    edges_sum = np.zeros(n_sleeve, dtype=np.float64)
    edges_count = np.zeros(n_sleeve, dtype=np.int64)

    for t in range(start, end):
        if t + 1 >= t_max:
            break
        mask_row = cache.signal_mask_2d[t]
        active_js = np.where(mask_row)[0]
        if len(active_js) == 0:
            continue
        side_row = cache.side_2d[t]
        for j in active_js:
            sym_col = int(sleeve_to_sym[int(j)])
            fwd = float(forward_ret_2d[t, sym_col])
            hit = float(side_row[int(j)]) * fwd
            edges_sum[j] += hit
            edges_count[j] += 1

    result = np.full(n_sleeve, np.nan, dtype=np.float64)
    for j in range(n_sleeve):
        if edges_count[j] > 0:
            result[j] = edges_sum[j] / float(edges_count[j])
    return result


def _resolve_sleeve_signals_at_bar(
    *,
    cache: L2SimulationCache,
    t: int,
    tradeable_mask: NDArray[np.bool_],
    symbols: tuple[str, ...],
    hurdle_row: NDArray[np.float64],
    vol_row: NDArray[np.float64],
    fixed_cost_safety_mult: float,
) -> tuple[
    dict[tuple[str, str], SymbolSignal],
    dict[tuple[str, str], tuple[float, float]],
    int,
]:
    """Bar t에서 활성 sleeve의 SymbolSignal dict를 반환한다.

    sleeve key = (symbol, strategy_id). 기존 symbol 단위 dict 대신
    sleeve 단위로 반환하여 같은 symbol의 복수 TF 신호를 독립 유지한다.
    sizing 시 vol은 underlying symbol의 vol(sleeve_to_sym 참조)을 사용한다.

    Args:
        cache: L2SimulationCache (sleeve 차원 신호 행렬 포함).
        t: 현재 bar index.
        tradeable_mask: [N] bool 배열.
        symbols: 심볼 tuple [N].
        hurdle_row: [N] hurdle bps 배열.
        vol_row: [N] 변동성 배열 (sleeve vol 참조용).
        fixed_cost_safety_mult: 비용 안전 배수.

    Returns:
        tuple: (signals, edges, n_dropped). signals: (symbol, strategy_id) → SymbolSignal.
        edges: (symbol, strategy_id) → (signed_gross_bps_per_bar, expected_cost_bps_per_bar).
        n_dropped: signed_net==0 또는 non-finite로 탈락한 active sleeve 수.

    Time Complexity: O(S) where S = n_sleeves.
    """
    n_sleeve = cache.signal_mask_2d.shape[1]
    if n_sleeve == 0:
        return {}, {}, 0

    sleeve_mask_row = cache.signal_mask_2d[t]  # [S]
    active_sleeves = np.where(sleeve_mask_row)[0]
    if len(active_sleeves) == 0:
        return {}, {}, 0

    result: dict[tuple[str, str], SymbolSignal] = {}
    edges_out: dict[tuple[str, str], tuple[float, float]] = {}
    sleeve_ids = cache.sleeve_ids
    sleeve_to_sym_arr = cache.sleeve_to_sym
    n_dropped = 0

    for j in active_sleeves:
        sym_col = int(sleeve_to_sym_arr[j])
        if not tradeable_mask[sym_col]:
            continue

        sleeve_id = sleeve_ids[j]

        gross_bps = float(cache.expected_gross_bps_2d[t, j])
        net_bps = float(cache.expected_net_bps_2d[t, j])
        holding = int(cache.holding_bars_2d[t, j])
        side = int(cache.side_2d[t, j])
        qw = float(cache.quality_weight_2d[t, j])
        hurdle = float(hurdle_row[sym_col])

        basis: EdgeBasis = "gross" if np.isfinite(gross_bps) and gross_bps != 0.0 else "net"
        edge = compute_expected_layer2_edge(
            side=side,
            expected_gross_bps=gross_bps,
            expected_net_bps=net_bps,
            expected_holding_bars=holding,
            execution_cost_bps=hurdle,
            edge_basis=basis,
            fixed_cost_safety_mult=fixed_cost_safety_mult,
        )

        if not np.isfinite(edge.signed_net_bps_per_bar) or edge.signed_net_bps_per_bar == 0.0:
            n_dropped += 1
            continue

        result[sleeve_id] = SymbolSignal(
            raw_mu=edge.signed_net_bps_per_bar,
            volatility=float(max(vol_row[sym_col], VOL_FLOOR)),
            n_obs=1,
            t_stat=0.0,
            valid=True,
            beta_btc=None,
            quality_weight=qw,
        )
        edges_out[sleeve_id] = (edge.signed_gross_bps_per_bar, edge.expected_cost_bps_per_bar)

    return result, edges_out, n_dropped


def _combine_sleeve_signals_to_symbol(
    sleeve_signals: dict[tuple[str, str], SymbolSignal],
    *,
    method: str = "precision_weighted",
    conviction_cap_mult: float = 1.5,
    sleeve_edges: dict[tuple[str, str], tuple[float, float]] | None = None,
) -> tuple[dict[str, SymbolSignal], dict[str, bool]]:
    """Sleeve 신호를 심볼당 단일 SymbolSignal로 precision-weighted pooling.

    Args:
        sleeve_signals: (symbol, strategy_id) → SymbolSignal dict.
        method: 결합 방식 ('precision_weighted', 'equal', 'max_edge').
        conviction_cap_mult: κ, conviction 상한 배수 [1.0, 3.0].
        sleeve_edges: (symbol, strategy_id) → (signed_gross_pb, cost_pb). 제공 시 friction_by_symbol 산출.

    Returns:
        tuple: (signals_by_symbol, friction_by_symbol). friction_by_symbol은
        sleeve_edges=None 시 빈 dict.

    Time: O(k) per call.
    """
    by_sym: dict[str, list[SymbolSignal]] = {}
    for (sym, _strat), sig in sleeve_signals.items():
        by_sym.setdefault(sym, []).append(sig)

    out: dict[str, SymbolSignal] = {}
    friction_by_sym: dict[str, bool] = {}

    for sym, sigs in by_sym.items():
        mus = np.array([s.raw_mu for s in sigs], dtype=np.float64)
        cs = np.array([max(s.quality_weight, 0.0) for s in sigs], dtype=np.float64)
        vol = sigs[0].volatility
        denom = cs.sum()

        if method == "max_edge":
            j = int(np.argmax(np.abs(mus)))
            mu_s = float(mus[j])
            c_s = float(cs[j])
        elif method == "equal" or denom <= 1e-12:
            mu_s = float(mus.mean())
            c_s = float(min(denom, conviction_cap_mult * cs.max() if cs.max() > 0.0 else 0.0))
        else:
            mu_s = float((cs * mus).sum() / denom)
            c_s = float(min(denom, conviction_cap_mult * cs.max()))

        out[sym] = SymbolSignal(
            raw_mu=mu_s,
            volatility=vol,
            n_obs=len(sigs),
            t_stat=0.0,
            valid=bool(np.isfinite(mu_s)),
            beta_btc=None,
            quality_weight=c_s,
        )

        # friction 판정: sleeve_edges 제공 시 precision-weighted gross vs cost
        if sleeve_edges is not None:
            gross_pb_list = []
            cost_pb_list = []
            for (_sym, _strat) in sleeve_signals:
                if _sym == sym:
                    _e = sleeve_edges.get((_sym, _strat))
                    if _e is not None:
                        gross_pb_list.append(_e[0])
                        cost_pb_list.append(_e[1])
            if gross_pb_list:
                gross_arr = np.array(gross_pb_list, dtype=np.float64)
                cost_arr = np.array(cost_pb_list, dtype=np.float64)
                if denom <= 1e-12:
                    g_bar = float(gross_arr.mean())
                    c_bar = float(cost_arr.mean())
                else:
                    g_bar = float((cs * gross_arr).sum() / denom)
                    c_bar = float((cs * cost_arr).sum() / denom)
                friction_by_sym[sym] = bool(abs(g_bar) >= c_bar)

    return out, friction_by_sym


def _count_netting_symbols(
    sleeve_signals: dict[tuple[str, str], SymbolSignal],
    pooled_signals: dict[str, SymbolSignal],
    *,
    cancel_ratio: float = 0.5,
) -> int:
    """sleeve 신호 부호 상쇄(netting)가 발생한 symbol 수를 반환.

    부호 혼재(양·음 sleeve 공존) & abs(pooled_mu) < cancel_ratio * max(abs(raw_mu_i))
    인 symbol을 netting으로 판정.

    Args:
        sleeve_signals: (symbol, strategy_id) → SymbolSignal (sleeve raw edge).
        pooled_signals: symbol → SymbolSignal (pooled result).
        cancel_ratio: pooled_mu가 raw_mu 최대 대비 얼마나 작으면 netting으로 볼지.

    Returns:
        netting symbol 개수.
    """
    by_sym: dict[str, list[float]] = {}
    for (sym, _strat), sig in sleeve_signals.items():
        by_sym.setdefault(sym, []).append(sig.raw_mu)

    count = 0
    for sym, raw_mus in by_sym.items():
        pos = any(m > 0 for m in raw_mus)
        neg = any(m < 0 for m in raw_mus)
        if not (pos and neg):
            continue
        pooled = pooled_signals.get(sym)
        if pooled is None:
            continue
        max_abs_raw = max(abs(m) for m in raw_mus)
        if max_abs_raw <= 0.0:
            continue
        if abs(pooled.raw_mu) < cancel_ratio * max_abs_raw:
            count += 1
    return count


def _compute_mtf_rebalance_stats(
    sleeve_signals: dict[tuple[str, str], SymbolSignal],
    pooled_signals: dict[str, SymbolSignal],
) -> tuple[int, int, int, float, float]:
    """Multi-TF pooling 효율 진단 통계 (per-rebalance).

    Returns:
        (n_symbols, n_multi, n_conflict, dilution_sum, edge_surrendered_bps)
        - n_symbols: pool 후 고유 symbol 수.
        - n_multi: sleeve >= 2인 symbol 수 (multi-TF 공존).
        - n_conflict: multi symbol 중 부호 혼재(양·음) 수.
        - dilution_sum: Σ_multi |μ_s| / max_i|μ_i| (1.0=무손실, 평균은 caller에서 n_multi로 나눔).
        - edge_surrendered_bps: Σ_multi max(max_i|μ_i| - |μ_s|, 0) — pooling이 버린 per-bar 엣지 bps.
    """
    by_sym: dict[str, list[float]] = {}
    for (sym, _strat), sig in sleeve_signals.items():
        by_sym.setdefault(sym, []).append(sig.raw_mu)

    n_symbols = len(by_sym)
    n_multi = 0
    n_conflict = 0
    dilution_sum = 0.0
    edge_surr = 0.0
    for sym, mus in by_sym.items():
        if len(mus) < 2:
            continue
        n_multi += 1
        max_abs = max(abs(m) for m in mus)
        if any(m > 0 for m in mus) and any(m < 0 for m in mus):
            n_conflict += 1
        pooled = pooled_signals.get(sym)
        if pooled is not None and max_abs > 0.0:
            dilution_sum += abs(pooled.raw_mu) / max_abs
            edge_surr += max(max_abs - abs(pooled.raw_mu), 0.0)
    return n_symbols, n_multi, n_conflict, dilution_sum, edge_surr


def _run_awf_simulation(
    *,
    cache: L2SimulationCache,
    signal_batch: ValidatedSignalBatch,
    aligned: AlignedMarketData,
    awf_folds: tuple[WFFold, ...],
    config: Layer2AllocationConfig,
    caps: PortfolioCaps,
    tf: str = "4h",
    portfolio_nav: float | None = None,
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
    _diag = bool(getattr(config, "l2_diag_attribution_enabled", False))
    _diag_top_k = int(getattr(config, "l2_diag_sleeve_top_k", 15))
    _diag_sample_every = int(getattr(config, "l2_diag_sleeve_sample_every", 0))
    rank_buffer = int(config.rank_buffer)
    kelly_fraction = float(config.kelly_fraction)
    # D2: vol_target=None → unit vol-target(1.0) 강제 — RC-1 cascade 해제.
    # max_ann_vol=None이면 vol-targeting·risk_budget_floor·adaptive_breadth가 전부 사망.
    # 1.0 고정으로 shape 정규화 유지, scale은 D3 closed-form L*가 전담.
    vol_target: float | None = config.max_ann_vol if config.max_ann_vol is not None else 1.0
    no_trade_band = float(config.no_trade_band)
    rebalance_bars = int(config.rebalance_bars)
    fixed_cost_safety_mult = float(getattr(config, "fixed_cost_safety_mult", 1.25))
    edge_throttle_enabled = bool(getattr(config, "edge_throttle_enabled", True))
    edge_floor_bps = float(getattr(config, "edge_floor_bps", 0.0))
    edge_ref_bps = float(getattr(config, "edge_ref_bps", 5.0))
    edge_throttle_gamma = float(getattr(config, "edge_throttle_gamma", 1.0))
    edge_throttle_min_active_mult = float(getattr(config, "edge_throttle_min_active_mult", 0.0))
    risk_budget_floor_ratio = float(getattr(config, "risk_budget_floor_ratio", 0.0))
    risk_budget_max_scale = float(getattr(config, "risk_budget_max_scale", 3.0))
    adaptive_breadth_enabled = bool(getattr(config, "adaptive_breadth_enabled", False))
    adaptive_k_extra = int(getattr(config, "adaptive_k_extra", 0))
    adaptive_expand_below_vol_ratio = float(
        getattr(config, "adaptive_expand_below_vol_ratio", 0.0)
    )

    if _diag and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[L2-ATTR-CFG] k_rank=%d rebalance_bars=%d kelly_fraction=%.3f "
            "fixed_cost_safety_mult=%.3f deploy_cost_safety_mult=%.3f "
            "edge_throttle_enabled=%s edge_floor_bps=%.2f edge_ref_bps=%.2f "
            "edge_throttle_gamma=%.3f risk_budget_floor_ratio=%.3f "
            "risk_budget_max_scale=%.3f l2_sleeve_combine_method=%s",
            config.k_rank, config.rebalance_bars, config.kelly_fraction,
            config.fixed_cost_safety_mult, config.deploy_cost_safety_mult,
            config.edge_throttle_enabled, config.edge_floor_bps, config.edge_ref_bps,
            config.edge_throttle_gamma, config.risk_budget_floor_ratio,
            config.risk_budget_max_scale, config.l2_sleeve_combine_method,
        )

    symbols = aligned.symbols
    n_sym = len(symbols)
    bars_per_year = 24.0 * 365.0 / max(hours_per_bar_tf(tf), 1e-9)
    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    # Phase 3-5: capacity_usdt clip 상수
    # portfolio_nav=None → 단위 NAV(1.0) 기준 시뮬레이션
    _portfolio_nav: float = portfolio_nav if portfolio_nav is not None else 1.0
    _min_order_usdt: float = 5.0
    _capacity_clip_enabled: bool = portfolio_nav is not None  # unit-NAV(1.0)일 때 skip

    vol_matrix = cache.vol_matrix_2d

    all_rets_hybrid: list[float] = []
    all_rets_baseline: list[float] = []
    all_rets_baseline_ew: list[float] = []
    all_turnovers: list[float] = []
    all_turnovers_baseline: list[float] = []
    all_gross_exposures: list[float] = []
    all_net_exposures: list[float] = []
    friction_pass_total = 0
    signal_total = 0
    support_leak_count = 0
    total_cost_hybrid = 0.0
    total_cost_baseline = 0.0
    cap_saturation_count = 0
    rebalance_count = 0
    trade_count = 0
    prev_support: set[int] = set()
    fold_rets_hybrid: list[list[float]] = []
    fold_rets_baseline: list[list[float]] = []
    fold_selected_symbols: list[tuple[str, ...]] = []

    # D3: fit-leg 수익률 수집 (look-ahead-free L* calibration용)
    # fit-leg = fold.fit_start → fold.oos_start (OOS 이전). 독립 상태로 수집.
    # 모든 fold fit-leg 연결 → fit_rets_hybrid (closed-form L* 입력).
    all_fit_rets_hybrid: list[float] = []
    fold_attributions: list[Layer2FoldAttribution] = []
    # persistence 측정용: fold별 fit-leg 수익률 저장 (OOS 루프에서 대응 참조)
    _per_fold_fit_rets: list[list[float]] = []

    prev_selection: frozenset[str] = frozenset()
    prev_w: NDArray[np.float64] = np.zeros(n_sym, dtype=np.float64)
    prev_w_baseline: NDArray[np.float64] = np.zeros(n_sym, dtype=np.float64)
    prev_w_baseline_ew: NDArray[np.float64] = np.zeros(n_sym, dtype=np.float64)
    last_selected: frozenset[str] = frozenset()
    last_w: NDArray[np.float64] = np.zeros(n_sym, dtype=np.float64)


    # C1: fit-leg book 수익률 수집 (RC-2 수정).
    # OOS 루프와 동일한 allocation 체인(rank→diagonal_kelly→throttle→vol_target)을 fit 구간에 적용.
    # 독립 prev_w_fit=0 초기화; signal schedule은 fit 구간 events만 사용 (look-ahead 0).
    # fit_end = fold.oos_start (exclusive) → look-ahead 엄수.
    # Time Complexity: O(F·T_fit·N) where F=n_folds, T_fit=fit bars, N=n_sym.
    for _fit_fold in awf_folds:
        _fit_start = int(_fit_fold.fit_start)
        _fit_end = int(_fit_fold.oos_start)  # exclusive — look-ahead 0
        _this_fold_fit_rets: list[float] = []
        if _fit_start >= _fit_end or _fit_end <= 1:
            _per_fold_fit_rets.append(_this_fold_fit_rets)
            continue

        _prev_w_fit: NDArray[np.float64] = np.zeros(n_sym, dtype=np.float64)
        _prev_selection_fit: frozenset[str] = frozenset()
        for _ft in range(_fit_start, _fit_end - 1, rebalance_bars):
            _ft_end = min(_ft + rebalance_bars, _fit_end - 1)
            if _ft + 1 >= aligned.close_2d.shape[0]:
                break
            _fit_tradeable = cache.tradeable_mask_2d[_ft]
            _fit_hurdle = cache.hurdle_2d[_ft]
            _fit_beta_arr = cache.beta_1d
            _fit_btc_beta = _fit_beta_arr  # Use directly
            
            # Sleeve 기반 signal 읽기 — [S] mask에서 [N] tradeable 참조
            _sleeve_signals, _, _ = _resolve_sleeve_signals_at_bar(
                cache=cache,
                t=_ft,
                tradeable_mask=_fit_tradeable,
                symbols=symbols,
                hurdle_row=_fit_hurdle,
                vol_row=vol_matrix[_ft],
                fixed_cost_safety_mult=fixed_cost_safety_mult,
            )
            # sleeve key → symbol key 변환 (rank_and_select는 str key 요구)
            # 같은 symbol에 복수 sleeve 시: precision-weighted pooling (인플레이션 방지)
            _fit_valid_signals, _ = _combine_sleeve_signals_to_symbol(
                _sleeve_signals,
                method=config.l2_sleeve_combine_method,
                conviction_cap_mult=config.l2_sleeve_conviction_cap_mult,
            )

            if not _fit_valid_signals:
                continue

            _fit_selected, _ = rank_and_select(
                _fit_valid_signals,
                k_rank=k_rank,
                sector_cap=n_sym,
                prev_selection=_prev_selection_fit,
                rank_buffer=rank_buffer,
                min_abs_z=float(config.min_abs_rank_z),
                selection_mode="absolute",
            )
            _prev_selection_fit = _fit_selected

            _fit_mu_arr: NDArray[np.float64] = np.zeros(n_sym, dtype=np.float64)
            _fit_sig_arr: NDArray[np.float64] = np.full(n_sym, VOL_FLOOR, dtype=np.float64)
            for _fs, _fss in _fit_valid_signals.items():
                if _fs in sym_to_idx:
                    _fi = sym_to_idx[_fs]
                    if _fs in _fit_selected:
                        _fit_mu_arr[_fi] = _fss.raw_mu
                    _fit_sig_arr[_fi] = _fss.volatility

            _fit_support_mask = _fit_mu_arr != 0.0
            _fit_w = diagonal_kelly_weights(
                mu_bps=_fit_mu_arr,
                sigma=_fit_sig_arr,
                kelly_fraction=kelly_fraction,
                vol_target=vol_target,
                caps=caps,
                prev_w=_prev_w_fit,
                no_trade_band=no_trade_band,
                btc_beta=_fit_btc_beta,
                bars_per_year=bars_per_year,
                support_mask=_fit_support_mask,
            )
            if edge_throttle_enabled:
                _fit_score = _book_edge_score(_fit_w, _fit_mu_arr)
                _fit_m = _edge_throttle_multiplier(
                    _fit_score,
                    floor_bps=edge_floor_bps,
                    ref_bps=edge_ref_bps,
                    gamma=edge_throttle_gamma,
                    min_active_mult=edge_throttle_min_active_mult,
                )
                _fit_w = _fit_w * _fit_m
            _fit_w = np.where(_fit_tradeable, _fit_w, 0.0)

            # Phase 3-5: capacity_usdt clip (fit-leg)
            # portfolio_nav=None 시 unit-NAV → skip
            _fit_adv = getattr(aligned, "adv_usdt_2d", None)
            if _capacity_clip_enabled and isinstance(_fit_adv, np.ndarray) and _ft < _fit_adv.shape[0]:
                _fit_cap_row = np.nan_to_num(_fit_adv[_ft], nan=0.0, posinf=0.0, neginf=0.0)
                _intended = np.abs(_fit_w) * _portfolio_nav
                _fit_w[_intended < _min_order_usdt] = 0.0
                
                _cap_positive = _fit_cap_row > 0.0
                if np.any(_cap_positive):
                    _max_w = np.where(_cap_positive, _fit_cap_row / max(_portfolio_nav, 1.0), np.inf)
                    _over = np.abs(_fit_w) > _max_w
                    _fit_w[_over] = np.sign(_fit_w[_over]) * _max_w[_over]

            # 리밸런싱 비용
            _fit_rebal_cost = compute_rebalance_cost(
                previous_weights=_prev_w_fit,
                target_weights=_fit_w,
                round_trip_cost_bps=_fit_hurdle,
            )
            _prev_w_fit = _fit_w

            # 비중 보유 기간 내 per-bar book return 수집
            for _ft2 in range(_ft, _ft_end):
                if _ft2 + 1 >= aligned.close_2d.shape[0]:
                    break
                _fc_cur = aligned.close_2d[_ft2]
                _fc_nxt = aligned.close_2d[_ft2 + 1]
                _fbar_ret = np.where(_fc_cur > 0, (_fc_nxt - _fc_cur) / _fc_cur, 0.0)
                _fbar_ret = np.nan_to_num(_fbar_ret, nan=0.0, posinf=0.0, neginf=0.0)
                _ffunding = cache.funding_2d[_ft2]
                _fgross = compute_futures_bar_return(
                    weights=_fit_w,
                    price_returns=_fbar_ret,
                    funding_rates=_ffunding,
                )
                _fcost = _fit_rebal_cost if _ft2 == _ft else 0.0
                _r = _fgross - _fcost
                all_fit_rets_hybrid.append(_r)
                _this_fold_fit_rets.append(_r)
        _per_fold_fit_rets.append(_this_fold_fit_rets)

    # per-fold fit-leg diagnostics (vol-targeting 무결성 확인)
    from src.domain.futures.strategy.tiered_workflow.metrics import _cagr, _mdd
    for _f_idx, _ffit in enumerate(_per_fold_fit_rets):
        if len(_ffit) < 2:
            continue
        _fcagr = _cagr(_ffit, bars_per_year=bars_per_year)
        _fmdd = _mdd(_ffit)
        _farr = np.asarray(_ffit, dtype=np.float64)
        _fann_vol = float(np.std(_farr, ddof=1)) * float(np.sqrt(max(bars_per_year, 1e-12)))
        _fstd = max(float(np.std(_farr, ddof=1)), 1e-12)
        _fsharpe = float(np.mean(_farr) / _fstd) * float(np.sqrt(max(bars_per_year, 1e-12)))
        logger.debug(
            "[L2-FIT-DIAG] fold=%d fit_bars=%d fit_CAGR=%.4f fit_MDD=%.4f "
            "fit_ann_vol=%.4f fit_sharpe=%.4f",
            _f_idx, len(_ffit), _fcagr, _fmdd, _fann_vol, _fsharpe,
        )

    # C4: per-TF fit-leg edge → included_tfs (TF 게이트)
    _tf_inclusion_enabled = bool(getattr(config, "l2_tf_inclusion_enabled", True))
    _tf_min_edge = float(getattr(config, "l2_tf_inclusion_min_edge", 0.0))
    included_tfs_by_fold: list[set[str]] = []
    for _f_idx, _f_fold in enumerate(awf_folds):
        if _tf_inclusion_enabled and int(_f_fold.fit_start) < int(_f_fold.oos_start):
            _per_tf_edge = compute_per_tf_fit_edge(
                cache=cache,
                aligned=aligned,
                fit_start=int(_f_fold.fit_start),
                fit_end=int(_f_fold.oos_start),
            )
            _included = {tf for tf, e in _per_tf_edge.items() if e > _tf_min_edge}
            if not _included:
                logger.debug(
                    "[L2-TFGATE] fold=%d included_tfs=∅ edges=%s → fallback: ALL TFs",
                    _f_idx, _per_tf_edge,
                )
                _included = set(cache.sleeve_to_tf) - {"unk"}
            included_tfs_by_fold.append(_included)
            logger.debug(
                "[L2-TFGATE] fold=%d included=%s edges=%s",
                _f_idx, sorted(_included), _per_tf_edge,
            )
        else:
            included_tfs_by_fold.append(set(cache.sleeve_to_tf) - {"unk"})

    # L2 bucket routing: regime code + fit-leg 버킷 실현엣지 (1회)
    _l2_routing_mode = str(getattr(config, "l2_routing_mode", "bucket"))
    _regime_code_1d: NDArray[np.int8] = np.zeros(aligned.close_2d.shape[0], dtype=np.int8)
    bucket_edges_by_fold: list[dict[tuple[int, str, str], float]] = []
    if _l2_routing_mode == "bucket":
        from src.domain.futures.strategy.market_regime import compute_market_regime_context
        from src.domain.futures.strategy.tiered_workflow.l2_meta import (
            compute_bucket_realized_edges,
        )
        _regime_code_1d = compute_market_regime_context(aligned=aligned).code_1d
        # Step B: per-regime occupancy DEBUG logging
        if logger.isEnabledFor(logging.DEBUG):
            _unique_regimes, _counts_regimes = np.unique(_regime_code_1d, return_counts=True)
            _n_total_regime = int(_regime_code_1d.shape[0])
            for _r, _c in sorted(zip(_unique_regimes.tolist(), _counts_regimes.tolist(), strict=True)):
                _pct = float(_c) / float(_n_total_regime) * 100.0
                logger.debug(
                    "[L2-REGIME-OCC] regime=%d count=%d pct=%.1f%% total=%d",
                    _r, _c, _pct, _n_total_regime,
                )
        for _f_idx, _f_fold in enumerate(awf_folds):
            if int(_f_fold.fit_start) < int(_f_fold.oos_start):
                _be = compute_bucket_realized_edges(
                    cache=cache,
                    aligned=aligned,
                    fit_start=int(_f_fold.fit_start),
                    fit_end=int(_f_fold.oos_start),
                    regime_code_1d=_regime_code_1d,
                    cost_bps=float(getattr(config, "l2_bucket_cost_bps", 6.0)),
                    min_n=int(getattr(config, "l2_bucket_min_n", 15)),
                    shrinkage=float(getattr(config, "l2_bucket_shrinkage", 0.3)),
                )
                bucket_edges_by_fold.append(_be)
            else:
                bucket_edges_by_fold.append({})
    else:
        bucket_edges_by_fold = [{} for _ in awf_folds]

    # Step A: per-fold bucket edge DEBUG logging
    if _l2_routing_mode == "bucket" and logger.isEnabledFor(logging.DEBUG):
        _regime_names_local = ("bull_quiet", "bull_volatile", "bear_quiet", "bear_volatile", "transition", "crash")
        for _bf_idx, _bf_edges in enumerate(bucket_edges_by_fold):
            logger.debug("[L2-BUCKET-MAP] fold=%d n_buckets=%d", _bf_idx, len(_bf_edges))
            for (_br, _bfam, _btf), _bval in sorted(_bf_edges.items(), key=lambda x: -x[1]):
                _rl = _regime_names_local[_br] if 0 <= _br < 6 else f"unknown({_br})"
                logger.debug(
                    "[L2-BUCKET-EDGE] fold=%d regime=%s(%d) family=%s tf=%s edge=%.2f_bps",
                    _bf_idx, _rl, _br, _bfam, _btf, _bval,
                )

    # Step H: per-fold regime distribution stability (fit vs OOS)
    if _l2_routing_mode == "bucket":
        for _fi, _fold in enumerate(awf_folds):
            _fit_slice = _regime_code_1d[int(_fold.fit_start):int(_fold.oos_start)]
            _oos_slice = _regime_code_1d[int(_fold.oos_start):int(_fold.oos_end)]
            if len(_fit_slice) == 0 or len(_oos_slice) == 0:
                continue
            _fit_uniq, _fit_counts = np.unique(_fit_slice, return_counts=True)
            _oos_uniq, _oos_counts = np.unique(_oos_slice, return_counts=True)
            _fit_freq = np.zeros(6, dtype=np.float64)
            _oos_freq = np.zeros(6, dtype=np.float64)
            for _r, _c in zip(_fit_uniq.tolist(), _fit_counts.tolist(), strict=True):
                _fit_freq[int(_r)] = _c / len(_fit_slice)
            for _r, _c in zip(_oos_uniq.tolist(), _oos_counts.tolist(), strict=True):
                _oos_freq[int(_r)] = _c / len(_oos_slice)
            _m = (_fit_freq + _oos_freq) / 2.0
            _js = 0.0
            for _p, _q in zip(_fit_freq, _m, strict=True):
                if _p > 0:
                    _js += _p * np.log2(_p / _q) if _q > 0 else 0.0
            for _p, _q in zip(_oos_freq, _m, strict=True):
                if _p > 0:
                    _js += _p * np.log2(_p / _q) if _q > 0 else 0.0
            _js /= 2.0
            _fit_occ = "|".join(f"{p*100:.0f}" for p in _fit_freq)
            _oos_occ = "|".join(f"{p*100:.0f}" for p in _oos_freq)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[L2-REGIME-SHIFT] fold=%d js_div=%.4f fit=%s oos=%s",
                    _fi, _js, _fit_occ, _oos_occ,
                )
                if _js > 0.15:
                    logger.debug(
                        "[L2-REGIME-SHIFT] fold=%d regime shift detected (fit↔OOS JS=%.4f)",
                        _fi, _js,
                    )

    # Steps G+J: init per-fold bucket hit ratio + OOS realized edge accumulators
    _fold_bucket_hit: list[tuple[int, int]] = [(0, 0) for _ in awf_folds]
    _fold_oos_bucket_sum: list[dict[tuple[int, str, str], float]] = [{} for _ in awf_folds]
    _fold_oos_bucket_cnt: list[dict[tuple[int, str, str], int]] = [{} for _ in awf_folds]

    for _fold_idx, fold in enumerate(awf_folds):
        t_fold_start = time.perf_counter()
        _fold_h: list[float] = []
        _fold_b: list[float] = []
        _fold_selected: set[str] = set()
        _attr_price = 0.0
        _attr_funding = 0.0
        _attr_cost = 0.0
        _fold_rebalance_count = 0
        if _diag:
            _attr_expected = 0.0
            _attr_gross_exps: list[float] = []
            _attr_net_exps: list[float] = []
            _attr_throttle: list[float] = []
            _attr_sleeves_active: list[int] = []
            _attr_friction_pass = 0
            _attr_signal_total = 0
            _attr_dropped = 0
            _attr_netting = 0
            _mtf_n_sym_sum = 0
            _mtf_multi_sum = 0
            _mtf_conflict_sum = 0
            _mtf_dilution_sum = 0.0
            _mtf_edge_surr_sum = 0.0
            _mtf_pooled_sum = 0
            _mtf_selected_sum = 0
        for t in range(fold.oos_start, fold.oos_end - 1, rebalance_bars):
            t_end = min(t + rebalance_bars, fold.oos_end - 1)

            t0_prep = time.perf_counter()
            tradeable_mask = cache.tradeable_mask_2d[t]
            hurdle = cache.hurdle_2d[t]
            beta_arr = cache.beta_1d
            btc_beta = beta_arr  # Use directly

            # Sleeve 기반 signal 읽기 — [S] mask에서 [N] tradeable 참조
            _oos_sleeve_sigs, _oos_sleeve_edges, _oos_dropped = _resolve_sleeve_signals_at_bar(
                cache=cache,
                t=t,
                tradeable_mask=tradeable_mask,
                symbols=symbols,
                hurdle_row=hurdle,
                vol_row=vol_matrix[t],
                fixed_cost_safety_mult=fixed_cost_safety_mult,
            )
            # C4: TF 게이트 필터 — fit-leg에서 edge>min_edge인 TF sleeve만 유지
            if _tf_inclusion_enabled:
                _current_included = included_tfs_by_fold[_fold_idx]
                _oos_sleeve_sigs = {
                    k: v for k, v in _oos_sleeve_sigs.items()
                    if _parse_tf_from_strategy_id(k[1]) in _current_included
                }
                _oos_sleeve_edges = {
                    k: v for k, v in _oos_sleeve_edges.items()
                    if _parse_tf_from_strategy_id(k[1]) in _current_included
                }
            # L2 bucket routing: OOS bar t의 regime 기반 sleeve 필터링
            if _l2_routing_mode == "bucket" and _fold_idx < len(bucket_edges_by_fold):
                _current_bucket_edges = bucket_edges_by_fold[_fold_idx]
                if _current_bucket_edges:
                    from src.domain.futures.strategy.tiered_workflow.l2_meta import (
                        _parse_meta_group_ids,
                        filter_sleeves_by_bucket,
                    )
                    _before_filter_count = len(_oos_sleeve_sigs)
                    _before_sleeve_keys = set(_oos_sleeve_sigs.keys())
                    _regime_now = int(_regime_code_1d[t]) if t < len(_regime_code_1d) else 0
                    # Step J: pre-compute sleeve key → index mapping for OOS edge tracking
                    _sleeve_to_j = {
                        cache.sleeve_ids[_j]: _j for _j in range(cache.signal_mask_2d.shape[1])
                        if cache.signal_mask_2d[t, _j] and cache.sleeve_ids[_j] in _before_sleeve_keys
                    }
                    _oos_sleeve_sigs = filter_sleeves_by_bucket(
                        _oos_sleeve_sigs,
                        _current_bucket_edges,
                        _regime_now,
                        edge_floor_bps=float(
                            getattr(config, "l2_bucket_edge_floor_bps", 0.0)
                        ),
                    )
                    _oos_sleeve_edges = {
                        k: v for k, v in _oos_sleeve_edges.items() if k in _oos_sleeve_sigs
                    }
                    # Step C: per-bar bucket filter stats (every 100 bars)
                    _after_filter_count = len(_oos_sleeve_sigs)
                    _dropped_by_bucket = _before_filter_count - _after_filter_count
                    if _dropped_by_bucket > 0 and logger.isEnabledFor(logging.DEBUG) and t % 100 == 0:
                        _regime_names_local = ("bull_quiet", "bull_volatile", "bear_quiet", "bear_volatile", "transition", "crash")  # noqa: E501
                        _rl = _regime_names_local[_regime_now] if 0 <= _regime_now < 6 else f"unknown({_regime_now})"
                        logger.debug(
                            "[L2-BUCKET-FILTER] t=%d fold=%d regime=%s(%d) "
                            "sleeves_before=%d after=%d dropped=%d",
                            t, _fold_idx, _rl, _regime_now,
                            _before_filter_count, _after_filter_count, _dropped_by_bucket,
                        )
                        # Step D: per-sleeve drop detail (diag only)
                        if _diag and logger.isEnabledFor(logging.DEBUG):
                            _drop_keys = _before_sleeve_keys - set(_oos_sleeve_sigs.keys())
                            for _dk in sorted(_drop_keys):
                                _fam, _tf = _parse_meta_group_ids(_dk[1])
                                _bk = (_regime_now, _fam, _tf)
                                _bedge = _current_bucket_edges.get(_bk, 0.0)
                                logger.debug(
                                    "[L2-BUCKET-DROP] t=%d sym=%s family=%s tf=%s "
                                    "regime=%d edge=%.2f floor=%.2f",
                                    t, _dk[0], _fam, _tf, _regime_now, _bedge,
                                    float(getattr(config, "l2_bucket_edge_floor_bps", 0.0)),
                                )
                    # Step G: bucket hit ratio update
                    if _before_filter_count > 0:
                        _old_hit = _fold_bucket_hit[_fold_idx]
                        _fold_bucket_hit[_fold_idx] = (
                            _old_hit[0] + 1,
                            _old_hit[1] + (1 if _after_filter_count > 0 else 0),
                        )
                    # Step J: OOS realized edge per bucket
                    _oos_cost_bps = float(getattr(config, "l2_bucket_cost_bps", 6.0))
                    _c_row = aligned.close_2d[t]
                    _c_nxt = aligned.close_2d[t + 1] if t + 1 < aligned.close_2d.shape[0] else _c_row
                    for _sk in _oos_sleeve_sigs:
                        _j_idx = _sleeve_to_j.get(_sk)
                        if _j_idx is None:
                            continue
                        _fam_j, _tf_j = _parse_meta_group_ids(_sk[1])
                        _bk_j = (_regime_now, _fam_j, _tf_j)
                        _side_j = float(cache.side_2d[t, _j_idx])
                        _sym_col_j = int(cache.sleeve_to_sym[_j_idx])
                        _denom_j = max(float(abs(_c_row[_sym_col_j])), 1e-12)
                        _fwd_bps_j = (float(_c_nxt[_sym_col_j]) - float(_c_row[_sym_col_j])) / _denom_j * 10000.0
                        _realized_j = _side_j * _fwd_bps_j - _oos_cost_bps
                        _prev_sum = _fold_oos_bucket_sum[_fold_idx].get(_bk_j, 0.0)
                        _fold_oos_bucket_sum[_fold_idx][_bk_j] = _prev_sum + _realized_j
                        _prev_cnt = _fold_oos_bucket_cnt[_fold_idx].get(_bk_j, 0)
                        _fold_oos_bucket_cnt[_fold_idx][_bk_j] = _prev_cnt + 1
            if _diag:
                _attr_dropped += _oos_dropped
                _attr_sleeves_active.append(len(_oos_sleeve_sigs))
            # sleeve key → symbol key 변환: precision-weighted pooling (인플레이션 방지)
            valid_signals, friction_by_symbol = _combine_sleeve_signals_to_symbol(
                _oos_sleeve_sigs,
                method=config.l2_sleeve_combine_method,
                conviction_cap_mult=config.l2_sleeve_conviction_cap_mult,
                sleeve_edges=_oos_sleeve_edges,
            )

            if _oos_sleeve_sigs:
                logger.debug(
                    "[AWF-EDGE] t=%d sleeve_count=%d edge_pass=%d",
                    t, len(_oos_sleeve_sigs), len(valid_signals),
                )
            if _diag:
                _attr_netting += _count_netting_symbols(_oos_sleeve_sigs, valid_signals)
                _m_nsym, _m_multi, _m_conf, _m_dil, _m_surr = _compute_mtf_rebalance_stats(
                    _oos_sleeve_sigs, valid_signals
                )
                _mtf_n_sym_sum += _m_nsym
                _mtf_multi_sum += _m_multi
                _mtf_conflict_sum += _m_conf
                _mtf_dilution_sum += _m_dil
                _mtf_edge_surr_sum += _m_surr
                _mtf_pooled_sum += len(valid_signals)
            prof_prep += time.perf_counter() - t0_prep

            t0_rank = time.perf_counter()
            rank_sig_arr: NDArray[np.float64] = np.full(n_sym, VOL_FLOOR, dtype=np.float64)
            for sym, ss in valid_signals.items():
                idx = sym_to_idx.get(sym)
                if idx is not None:
                    rank_sig_arr[idx] = ss.volatility
            effective_k_rank = k_rank
            if adaptive_breadth_enabled:
                effective_k_rank = _resolve_adaptive_k_rank(
                    base_k=k_rank,
                    n_valid=len(valid_signals),
                    prev_weights=prev_w,
                    sigma=rank_sig_arr,
                    bars_per_year=bars_per_year,
                    vol_target=vol_target,
                    expand_below_vol_ratio=adaptive_expand_below_vol_ratio,
                    max_extra=adaptive_k_extra,
                )
            selected, _z_scores = rank_and_select(
                valid_signals,
                k_rank=effective_k_rank,
                sector_cap=n_sym,
                prev_selection=prev_selection,
                rank_buffer=rank_buffer,
                min_abs_z=float(config.min_abs_rank_z),
                selection_mode="absolute",
            )
            # DEBUG: Z-score distribution diagnostics
            if _z_scores and logger.isEnabledFor(logging.DEBUG):
                _z_vals = [z for z in _z_scores.values() if z > 0.0]
                if _z_vals:
                    _z_arr = np.asarray(_z_vals, dtype=np.float64)
                    logger.debug(
                        "[L2-Z-DIST] t=%d n_pos=%d z_min=%.3f z_max=%.3f z_med=%.3f z_std=%.3f",
                        t, len(_z_vals),
                        float(np.min(_z_arr)), float(np.max(_z_arr)),
                        float(np.median(_z_arr)), float(np.std(_z_arr, ddof=1)),
                    )
            last_selected = selected
            if selected:
                _fold_selected.update(selected)
            if _diag:
                _mtf_selected_sum += len(selected)
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

            support_mask = mu_arr != 0.0
            # CS Score Amplification: _z_scores dict → array
            _z_score_arr: NDArray[np.float64] | None = None
            if config.l2_cs_amp_enabled and _z_scores:
                _z_score_arr = np.zeros(n_sym, dtype=np.float64)
                for sym, z in _z_scores.items():
                    _i = sym_to_idx.get(sym)
                    if _i is not None:
                        _z_score_arr[_i] = z
            w = diagonal_kelly_weights(
                mu_bps=mu_arr,
                sigma=sig_arr,
                kelly_fraction=kelly_fraction,
                vol_target=vol_target,
                caps=caps,
                prev_w=prev_w,
                no_trade_band=no_trade_band,
                btc_beta=btc_beta,
                bars_per_year=bars_per_year,
                support_mask=support_mask,
                z_scores=_z_score_arr,
                cs_amp_alpha=float(config.l2_cs_amp_alpha),
                cs_amp_mode=str(config.l2_cs_amp_mode),
            )
            if edge_throttle_enabled:
                score = _book_edge_score(w, mu_arr)
                m = _edge_throttle_multiplier(
                    score,
                    floor_bps=edge_floor_bps,
                    ref_bps=edge_ref_bps,
                    gamma=edge_throttle_gamma,
                    min_active_mult=edge_throttle_min_active_mult,
                )
                w = w * m
            else:
                m = 1.0
            if _diag:
                _attr_throttle.append(float(m))
            if risk_budget_floor_ratio > 0.0 and vol_target is not None:
                w = _apply_risk_budget_floor(
                    weights=w,
                    sigma=sig_arr,
                    bars_per_year=bars_per_year,
                    vol_target=vol_target,
                    floor_ratio=risk_budget_floor_ratio,
                    max_scale=risk_budget_max_scale,
                    caps=caps,
                    btc_beta=btc_beta,
                    support_mask=support_mask,
                )
            w = np.where(tradeable_mask, w, 0.0)

            # Phase 3-5: capacity_usdt clip (OOS)
            # portfolio_nav=None 시 unit-NAV → skip
            _adv = getattr(aligned, "adv_usdt_2d", None)
            if _capacity_clip_enabled and isinstance(_adv, np.ndarray) and t < _adv.shape[0]:
                _cap_row = np.nan_to_num(_adv[t], nan=0.0, posinf=0.0, neginf=0.0)
                _intended = np.abs(w) * _portfolio_nav
                w[_intended < _min_order_usdt] = 0.0
                
                _cap_positive = _cap_row > 0.0
                if np.any(_cap_positive):
                    _max_w = np.where(_cap_positive, _cap_row / max(_portfolio_nav, 1.0), np.inf)
                    _over = np.abs(w) > _max_w
                    w[_over] = np.sign(w[_over]) * _max_w[_over]

            last_w = w
            if _diag:
                if np.any(np.abs(w) > 1e-12):
                    _attr_expected += float(t_end - t) * float(np.dot(w, mu_arr * 1e-4))
                _attr_gross_exps.append(float(np.sum(np.abs(w))))
                _attr_net_exps.append(float(np.sum(w)))
            _fold_rebalance_count += 1

            w_base = build_directional_risk_matched_equal_weight(
                signed_net_mu_bps=mu_arr,
                strategy_weights=w,
                sigma=sig_arr,
                btc_beta=beta_arr,
                caps=caps,
                bars_per_year=bars_per_year,
            )
            w_base = np.where(tradeable_mask, w_base, 0.0)
            w_base_ew = build_directional_equal_weight_baseline(
                signed_net_mu_bps=mu_arr,
                strategy_weights=w,
                sigma=sig_arr,
                btc_beta=beta_arr,
                caps=caps,
                bars_per_year=bars_per_year,
            )
            w_base_ew = np.where(tradeable_mask, w_base_ew, 0.0)
            prof_alloc += time.perf_counter() - t0_alloc

            t0_eval = time.perf_counter()
            turnover = float(np.sum(np.abs(w - prev_w))) / 2.0
            all_turnovers.append(turnover)
            turnover_baseline = float(np.sum(np.abs(w_base - prev_w_baseline))) / 2.0
            all_turnovers_baseline.append(turnover_baseline)
            all_gross_exposures.append(float(np.sum(np.abs(w))))
            all_net_exposures.append(float(np.sum(w)))
            support_leak_count += int(np.sum((np.abs(w) > 1e-12) & ~support_mask))
            friction_pass = int(sum(1 for s in selected if friction_by_symbol.get(s, False)))
            friction_pass_total += friction_pass
            signal_total += len(selected)
            if _diag:
                _attr_friction_pass += friction_pass
                _attr_signal_total += len(selected)
            if _diag and logger.isEnabledFor(logging.DEBUG):
                _sample_cond = (t == fold.oos_start) or (
                    _diag_sample_every > 0 and rebalance_count > 0
                    and rebalance_count % _diag_sample_every == 0
                )
                if _sample_cond and len(_oos_sleeve_sigs) > 0:
                    _sym_w_pairs: list[tuple[str, float]] = []
                    for _sym_key in _oos_sleeve_sigs:
                        _sym = _sym_key[0]
                        _sym_idx = sym_to_idx.get(_sym)
                        if _sym_idx is not None and abs(w[_sym_idx]) > 1e-12:
                            _sym_w_pairs.append((_sym, float(abs(w[_sym_idx]))))
                    _sym_w_pairs.sort(key=lambda x: -x[1])
                    for _sym, _aw in _sym_w_pairs[:_diag_top_k]:
                        _sym_idx = sym_to_idx.get(_sym, -1)
                        _sleeve_sig = next(
                            (v for k, v in _oos_sleeve_sigs.items() if k[0] == _sym), None
                        )
                        if _sleeve_sig is None:
                            continue
                        _side = 1 if _sleeve_sig.raw_mu > 0 else -1
                        _fpass = friction_by_symbol.get(_sym, False)
                        logger.debug(
                            "[L2-ATTR-SLEEVE] fold=%d t=%d sym=%s side=%d raw_mu_pb=%.4f "
                            "qw=%.3f w=%.4f friction_pass=%s",
                            _fold_idx, t, _sym, _side,
                            _sleeve_sig.raw_mu, _sleeve_sig.quality_weight,
                            float(w[_sym_idx]) if _sym_idx >= 0 else 0.0,
                            _fpass,
                        )
            cap_saturation_count += int(_is_cap_saturated(weights=w, btc_beta=beta_arr, caps=caps))
            rebalance_count += 1
            new_support = set(np.flatnonzero(np.abs(w) > 1e-12).tolist())
            trade_count += len(new_support - prev_support)
            prev_support = new_support

            rebal_cost = compute_rebalance_cost(
                previous_weights=prev_w,
                target_weights=w,
                round_trip_cost_bps=hurdle,
            )
            rebal_cost_baseline = compute_rebalance_cost(
                previous_weights=prev_w_baseline,
                target_weights=w_base,
                round_trip_cost_bps=hurdle,
            )
            rebal_cost_baseline_ew = compute_rebalance_cost(
                previous_weights=prev_w_baseline_ew,
                target_weights=w_base_ew,
                round_trip_cost_bps=hurdle,
            )
            total_cost_hybrid += rebal_cost
            total_cost_baseline += rebal_cost_baseline

            for t2 in range(t, t_end):
                if t2 + 1 >= aligned.close_2d.shape[0]:
                    break
                c_cur = aligned.close_2d[t2]
                c_nxt = aligned.close_2d[t2 + 1]
                bar_ret = np.where(c_cur > 0, (c_nxt - c_cur) / c_cur, 0.0)
                bar_ret = np.nan_to_num(bar_ret, nan=0.0, posinf=0.0, neginf=0.0)
                funding_rates = cache.funding_2d[t2]
                gross_ret = compute_futures_bar_return(
                    weights=w,
                    price_returns=bar_ret,
                    funding_rates=funding_rates,
                )
                # 거래비용은 리밸런싱 첫 bar에만 차감
                cost = rebal_cost if t2 == t else 0.0
                _attr_price += float(np.dot(w, bar_ret))
                _attr_funding += -float(np.dot(w, funding_rates))
                _attr_cost += cost
                cost_baseline = rebal_cost_baseline if t2 == t else 0.0
                cost_baseline_ew = rebal_cost_baseline_ew if t2 == t else 0.0
                r_h = gross_ret - cost
                r_b = compute_futures_bar_return(
                    weights=w_base,
                    price_returns=bar_ret,
                    funding_rates=funding_rates,
                ) - cost_baseline
                r_b_ew = compute_futures_bar_return(
                    weights=w_base_ew,
                    price_returns=bar_ret,
                    funding_rates=funding_rates,
                ) - cost_baseline_ew
                all_rets_hybrid.append(r_h)
                all_rets_baseline.append(r_b)
                all_rets_baseline_ew.append(r_b_ew)
                _fold_h.append(r_h)
                _fold_b.append(r_b)

            prev_selection = selected
            prev_w = w
            prev_w_baseline = w_base
            prev_w_baseline_ew = w_base_ew
            prof_eval += time.perf_counter() - t0_eval

        fold_rets_hybrid.append(_fold_h)
        fold_rets_baseline.append(_fold_b)
        fold_selected_symbols.append(tuple(sorted(_fold_selected)))
        if not _diag:
            _attr_expected = 0.0
            _attr_gross_exps = []
            _attr_net_exps = []
            _attr_throttle = []
            _attr_sleeves_active = []
            _attr_friction_pass = 0
            _attr_signal_total = 0
            _attr_dropped = 0
            _attr_netting = 0
        _attr = _assemble_fold_attribution(
            fold_idx=_fold_idx,
            oos_bars=fold.oos_end - fold.oos_start,
            n_rebal=len(_attr_throttle) or _fold_rebalance_count,
            realized_price=_attr_price,
            realized_funding=_attr_funding,
            realized_cost=_attr_cost,
            expected_net=_attr_expected,
            gross_exps=_attr_gross_exps,
            net_exps=_attr_net_exps,
            throttle_mults=_attr_throttle,
            sleeves_active=_attr_sleeves_active,
            friction_pass_total=_attr_friction_pass,
            signal_total=_attr_signal_total,
            dropped_below_cost=_attr_dropped,
            netting_events=_attr_netting,
        )
        fold_attributions.append(_attr)
        if _diag and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[L2-ATTR] fold=%d oos_bars=%d n_rebal=%d realized_total=%.6f "
                "realized_price=%.6f realized_funding=%.6f realized_cost=%.6f "
                "expected_net=%.6f alpha_gap=%.6f mean_gross_exp=%.4f mean_net_exp=%.4f "
                "sleeves_active_mean=%.1f friction_pass_ratio=%.3f throttle_mult_mean=%.3f "
                "dropped_below_cost=%d netting_events=%d",
                _attr.fold_idx, _attr.oos_bars, _attr.n_rebal, _attr.realized_total,
                _attr.realized_price, _attr.realized_funding, _attr.realized_cost,
                _attr.expected_net, _attr.alpha_gap, _attr.mean_gross_exp, _attr.mean_net_exp,
                _attr.sleeves_active_mean, _attr.friction_pass_ratio, _attr.throttle_mult_mean,
                _attr.dropped_below_cost, _attr.netting_events,
            )
        if _diag and logger.isEnabledFor(logging.DEBUG):
            _nr = max(_fold_rebalance_count, 1)
            _mtf_sym_ratio = _mtf_multi_sum / max(_mtf_n_sym_sum, 1)
            _mtf_dilution = _mtf_dilution_sum / max(_mtf_multi_sum, 1)
            _mtf_conflict = _mtf_conflict_sum / max(_mtf_multi_sum, 1)
            logger.debug(
                "[L2-MTF] fold=%d n_rebal=%d mtf_symbol_ratio=%.3f "
                "pooling_dilution=%.3f conflict_ratio=%.3f "
                "edge_surrendered_bps_per_rebal=%.4f "
                "breadth_funnel=%.1f->%.1f->%.1f",
                _fold_idx, _fold_rebalance_count, _mtf_sym_ratio,
                _mtf_dilution, _mtf_conflict,
                _mtf_edge_surr_sum / _nr,
                (sum(_attr_sleeves_active) / _nr if _attr_sleeves_active else 0.0),
                _mtf_pooled_sum / _nr, _mtf_selected_sum / _nr,
            )
        if logger.isEnabledFor(logging.DEBUG) and _fold_idx < len(_per_fold_fit_rets):
            _ffit = _per_fold_fit_rets[_fold_idx]
            _foos = _fold_h  # current fold OOS rets (still in scope)
            _bars_per_year = bars_per_year
            _bpy = bars_per_year
            def _fold_cagr(rets: list[float], bpy: float = _bpy) -> float:
                if not rets:
                    return float("nan")
                eq = 1.0
                for r in rets:
                    eq *= (1.0 + r)
                years = len(rets) / max(bpy, 1.0)
                return float(eq ** (1.0 / max(years, 1e-9)) - 1.0) if years > 0 else float("nan")
            # sleeve-level L1 예측 일관성(IC): fit-leg vs OOS 기간 평균 expected_net_bps per sleeve
            _n_sl = cache.signal_mask_2d.shape[1]
            if _n_sl > 0:
                _fit_mask = cache.signal_mask_2d[fold.fit_start:fold.oos_start]
                _oos_mask = cache.signal_mask_2d[fold.oos_start:fold.oos_end]
                _fit_net = cache.expected_net_bps_2d[fold.fit_start:fold.oos_start]
                _oos_net = cache.expected_net_bps_2d[fold.oos_start:fold.oos_end]
                _sl_fit_mu = np.nanmean(np.where(_fit_mask, _fit_net, np.nan), axis=0)
                _sl_oos_mu = np.nanmean(np.where(_oos_mask, _oos_net, np.nan), axis=0)
                _valid = np.isfinite(_sl_fit_mu) & np.isfinite(_sl_oos_mu)
                _n_valid = int(_valid.sum())
                if _n_valid >= 5:
                    from scipy.stats import spearmanr as _spearmanr
                    _ic, _pval = _spearmanr(_sl_fit_mu[_valid], _sl_oos_mu[_valid])
                else:
                    _ic, _pval, _n_valid = float("nan"), float("nan"), 0
            else:
                _ic, _pval, _n_valid = float("nan"), float("nan"), 0
            logger.debug(
                "[L2-PERSIST] fold=%d fit_bars=%d fit_cagr=%.4f oos_bars=%d oos_cagr=%.4f "
                "pred_autocorr=%.3f pred_autocorr_pval=%.3f n_sleeves=%d",
                _fold_idx, len(_ffit), _fold_cagr(_ffit),
                len(_foos), _fold_cagr(_foos),
                _ic, _pval, _n_valid,
            )
            # sleeve-level 실현엣지 rank-IC (P2a 게이트)
            _n_sl2 = cache.signal_mask_2d.shape[1]
            if _n_sl2 > 0:
                _e_fit = compute_per_sleeve_realized_edge(cache, aligned, fold.fit_start, fold.oos_start)
                _e_oos = compute_per_sleeve_realized_edge(cache, aligned, fold.oos_start, fold.oos_end)
                _valid2 = np.isfinite(_e_fit) & np.isfinite(_e_oos)
                _n_valid2 = int(_valid2.sum())
                if _n_valid2 >= 5:
                    from scipy.stats import spearmanr as _spearmanr2
                    _ric, _rpval = _spearmanr2(_e_fit[_valid2], _e_oos[_valid2])
                else:
                    _ric, _rpval, _n_valid2 = float("nan"), float("nan"), 0
            else:
                _ric, _rpval, _n_valid2 = float("nan"), float("nan"), 0
            logger.debug(
                "[L2-SLEEVE-IC] fold=%d realized_ic=%.3f pval=%.3f n=%d",
                _fold_idx, _ric, _rpval, _n_valid2,
            )
        logger.log(PERF,
            "[PERF] awf_fold fold=%d/%d oos_bars=%d took=%.4fs",
            _fold_idx + 1, len(awf_folds),
            fold.oos_end - fold.oos_start,
            time.perf_counter() - t_fold_start,
        )

    logger.log(PERF,
        "[PERF] awf_total n_folds=%d n_rebalances=%d "
        "prep=%.4fs rank=%.4fs alloc=%.4fs eval=%.4fs took=%.4fs",
        len(awf_folds), rebalance_count,
        prof_prep, prof_rank, prof_alloc, prof_eval,
        time.perf_counter() - t_start_total,
    )

    # Steps G+J: per-fold bucket hit ratio + OOS vs fit edge comparison
    if _l2_routing_mode == "bucket":
        for _fi, (_active, _hit) in enumerate(_fold_bucket_hit):
            if _active == 0:
                continue
            _hit_pct = _hit / max(_active, 1) * 100.0
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[L2-BUCKET-HIT] fold=%d active_bars=%d bars_with_hit=%d hit_pct=%.1f%%",
                    _fi, _active, _hit, _hit_pct,
                )
                if _hit_pct < 30.0:
                    logger.debug(
                        "[L2-BUCKET-HIT] fold=%d low bucket coverage (%.1f%%)",
                        _fi, _hit_pct,
                    )
    if _l2_routing_mode == "bucket" and logger.isEnabledFor(logging.DEBUG):
        _regime_names_local = ("bull_quiet", "bull_volatile", "bear_quiet", "bear_volatile", "transition", "crash")
        for _fi in range(len(awf_folds)):
            _fit_edges = bucket_edges_by_fold[_fi]
            _oos_sum = _fold_oos_bucket_sum[_fi]
            _oos_cnt = _fold_oos_bucket_cnt[_fi]
            if not _oos_sum:
                continue
            _oos_edges: dict[tuple[int, str, str], float] = {}
            for _bk, _s in _oos_sum.items():
                _oos_edges[_bk] = _s / _oos_cnt[_bk]
            _common = set(_fit_edges) & set(_oos_edges)
            if not _common:
                continue
            _fit_vals = np.array([_fit_edges[_bk] for _bk in _common], dtype=np.float64)
            _oos_vals = np.array([_oos_edges[_bk] for _bk in _common], dtype=np.float64)
            _errors = _oos_vals - _fit_vals
            _rmse = float(np.sqrt(np.mean(_errors ** 2)))
            _mae = float(np.mean(np.abs(_errors)))
            _bias = float(np.mean(_errors))
            _corr = float(np.corrcoef(_fit_vals, _oos_vals)[0, 1]) if len(_common) >= 3 else 0.0
            logger.debug(
                "[L2-BUCKET-OOS] fold=%d n_common=%d rmse=%.2f mae=%.2f bias=%.2f corr=%.3f",
                _fi, len(_common), _rmse, _mae, _bias, _corr,
            )
            _sorted_by_err = sorted(
                _common, key=lambda _bk: abs(_oos_edges[_bk] - _fit_edges[_bk]), reverse=True
            )
            for _bk in _sorted_by_err[:15]:
                _rl = _regime_names_local[_bk[0]] if 0 <= _bk[0] < 6 else f"unknown({_bk[0]})"
                _err = _oos_edges[_bk] - _fit_edges[_bk]
                _n_bucket = _oos_cnt[_bk]
                logger.debug(
                    "[L2-BUCKET-OOS-DETAIL] fold=%d regime=%s(%d) family=%s tf=%s "
                    "fit=%.1fbps oos=%.1fbps err=%.1fbps n=%d",
                    _fi, _rl, _bk[0], _bk[1], _bk[2],
                    _fit_edges[_bk], _oos_edges[_bk], _err, _n_bucket,
                )
            _underfit = sorted(_common, key=lambda _bk: _oos_edges[_bk] - _fit_edges[_bk], reverse=True)[:5]
            _overfit = sorted(_common, key=lambda _bk: _oos_edges[_bk] - _fit_edges[_bk])[:5]
            for _bk in _underfit:
                _rl = _regime_names_local[_bk[0]] if 0 <= _bk[0] < 6 else f"unknown({_bk[0]})"
                _err = _oos_edges[_bk] - _fit_edges[_bk]
                _n_bucket = _oos_cnt[_bk]
                logger.debug(
                    "[L2-BUCKET-UNDERFIT] fold=%d regime=%s(%d) family=%s tf=%s "
                    "fit=%.1fbps oos=%.1fbps surplus=%.1fbps n=%d",
                    _fi, _rl, _bk[0], _bk[1], _bk[2],
                    _fit_edges[_bk], _oos_edges[_bk], _err, _n_bucket,
                )
            for _bk in _overfit:
                _rl = _regime_names_local[_bk[0]] if 0 <= _bk[0] < 6 else f"unknown({_bk[0]})"
                _err = _oos_edges[_bk] - _fit_edges[_bk]
                _n_bucket = _oos_cnt[_bk]
                logger.debug(
                    "[L2-BUCKET-OVERFIT] fold=%d regime=%s(%d) family=%s tf=%s "
                    "fit=%.1fbps oos=%.1fbps deficit=%.1fbps n=%d",
                    _fi, _rl, _bk[0], _bk[1], _bk[2],
                    _fit_edges[_bk], _oos_edges[_bk], _err, _n_bucket,
                )

    return _AwfSimResult(
        rets_hybrid=all_rets_hybrid,
        rets_baseline=all_rets_baseline,
        last_selected=last_selected,
        last_w=last_w,
        all_turnovers=all_turnovers,
        all_turnovers_baseline=all_turnovers_baseline,
        all_gross_exposures=all_gross_exposures,
        all_net_exposures=all_net_exposures,
        friction_pass_total=friction_pass_total,
        signal_total=signal_total,
        support_leak_count=support_leak_count,
        total_cost_hybrid=total_cost_hybrid,
        total_cost_baseline=total_cost_baseline,
        cap_saturation_count=cap_saturation_count,
        rebalance_count=rebalance_count,
        trade_count=trade_count,
        fold_rets_hybrid=fold_rets_hybrid,
        fold_rets_baseline=fold_rets_baseline,
        fold_selected_symbols=tuple(fold_selected_symbols),
        block_rets_hybrid=tuple(tuple(block) for block in fold_rets_hybrid),
        block_rets_baseline=tuple(tuple(block) for block in fold_rets_baseline),
        rets_baseline_ew=all_rets_baseline_ew,
        fit_rets_hybrid=tuple(all_fit_rets_hybrid),
        fold_attributions=tuple(fold_attributions),
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
