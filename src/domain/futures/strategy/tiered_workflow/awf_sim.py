# src/domain/futures/strategy/tiered_workflow/awf_sim.py

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.portfolio.portfolio_constructor import (
    PortfolioCaps,
    diagonal_kelly_weights,
    project_all_caps,
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

EdgeBasis = Literal["gross", "net"]


@dataclass(slots=True, frozen=True)
class Layer2ExpectedEdge:
    """Layer2 sizing에 사용할 per-bar edge contract."""

    signed_gross_bps_per_bar: float
    signed_net_bps_per_bar: float
    expected_cost_bps_per_bar: float
    basis: EdgeBasis


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
    block_rets_hybrid: tuple[tuple[float, ...], ...]
    block_rets_baseline: tuple[tuple[float, ...], ...]
    rets_baseline_ew: list[float]          # 순수 1/N EW baseline (uplift 측정 전용)


def _book_edge_score(
    w: NDArray[np.float64],
    mu_bps: NDArray[np.float64],
    effective_hurdle_bps: NDArray[np.float64],
) -> float:
    """사이징된 비중의 gross-weighted 평균 net-of-cost edge (bps/bar).

    Returns:
        0.0 if no active positions or all edges are non-positive.
    """
    abs_w = np.abs(w)
    den = float(np.sum(abs_w))
    if den < 1e-12:
        return 0.0
    net_edge = np.maximum(np.abs(mu_bps) - effective_hurdle_bps, 0.0)
    return float(np.dot(abs_w, net_edge) / den)


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
    vol_target = config.max_ann_vol
    no_trade_band = float(config.no_trade_band)
    rebalance_bars = int(config.rebalance_bars)
    fixed_cost_safety_mult = float(getattr(config, "fixed_cost_safety_mult", 1.25))
    deploy_cost_safety_mult = float(getattr(config, "deploy_cost_safety_mult", fixed_cost_safety_mult))
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

    prev_selection: frozenset[str] = frozenset()
    prev_w: NDArray[np.float64] = np.zeros(n_sym, dtype=np.float64)
    prev_w_baseline: NDArray[np.float64] = np.zeros(n_sym, dtype=np.float64)
    prev_w_baseline_ew: NDArray[np.float64] = np.zeros(n_sym, dtype=np.float64)
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
            beta_arr = np.zeros(n_sym, dtype=np.float64) if btc_beta is None else btc_beta

            valid_signals: dict[str, SymbolSignal] = {}
            gross_edge_by_symbol: dict[str, float] = {}
            for symbol, event in schedule._events_by_bar[t - schedule.start_idx].items():
                sym_idx = sym_to_idx.get(symbol)
                if sym_idx is None or not tradeable_mask[sym_idx]:
                    continue
                basis: EdgeBasis = (
                    "gross" if np.isfinite(float(event.expected_gross_bps)) and float(event.expected_gross_bps) != 0.0
                    else "net"
                )
                edge = compute_expected_layer2_edge(
                    side=int(event.side),
                    expected_gross_bps=float(event.expected_gross_bps),
                    expected_net_bps=float(event.expected_net_bps),
                    expected_holding_bars=int(event.expected_holding_bars),
                    execution_cost_bps=float(hurdle[sym_idx]),
                    edge_basis=basis,
                    fixed_cost_safety_mult=fixed_cost_safety_mult,
                )
                gross_edge_by_symbol[symbol] = edge.signed_gross_bps_per_bar
                if not np.isfinite(edge.signed_net_bps_per_bar) or edge.signed_net_bps_per_bar == 0.0:
                    continue
                valid_signals[symbol] = SymbolSignal(
                    raw_mu=edge.signed_net_bps_per_bar,
                    volatility=float(max(vol_matrix[t, sym_idx], VOL_FLOOR)),
                    n_obs=1,
                    t_stat=0.0,
                    valid=True,
                    beta_btc=None,
                    quality_weight=float(event.quality_weight),
                )
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

            support_mask = mu_arr != 0.0
            w = diagonal_kelly_weights(
                mu_bps=mu_arr,
                sigma=sig_arr,
                kelly_fraction=kelly_fraction,
                vol_target=vol_target,
                friction_hurdle_bps=hurdle,
                holding_bars=rebalance_bars,
                friction_safety_mult=deploy_cost_safety_mult,
                caps=caps,
                prev_w=prev_w,
                no_trade_band=no_trade_band,
                btc_beta=btc_beta,
                bars_per_year=bars_per_year,
                support_mask=support_mask,
            )
            if edge_throttle_enabled:
                eff_hurdle = hurdle * deploy_cost_safety_mult / max(float(rebalance_bars), 1.0)
                score = _book_edge_score(w, mu_arr, eff_hurdle)
                m = _edge_throttle_multiplier(
                    score,
                    floor_bps=edge_floor_bps,
                    ref_bps=edge_ref_bps,
                    gamma=edge_throttle_gamma,
                    min_active_mult=edge_throttle_min_active_mult,
                )
                w = w * m
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
            last_w = w

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
            selected_idxs = [sym_to_idx[s] for s in selected if s in sym_to_idx]
            if selected_idxs:
                sel_idx_arr = np.array(selected_idxs, dtype=np.intp)
                gross_edges = np.array(
                    [gross_edge_by_symbol.get(symbols[idx], 0.0) for idx in sel_idx_arr],
                    dtype=np.float64,
                )
                expected_cost = hurdle[sel_idx_arr] / max(rebalance_bars, 1)
                friction_pass = int(np.sum(np.abs(gross_edges) >= expected_cost))
            else:
                friction_pass = 0
            friction_pass_total += friction_pass
            signal_total += len(selected)
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
                funding_rates = _resolve_funding_row(aligned=aligned, t2=t2, n_sym=n_sym)
                gross_ret = compute_futures_bar_return(
                    weights=w,
                    price_returns=bar_ret,
                    funding_rates=funding_rates,
                )
                # 거래비용은 리밸런싱 첫 bar에만 차감
                cost = rebal_cost if t2 == t else 0.0
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
        block_rets_hybrid=tuple(tuple(block) for block in fold_rets_hybrid),
        block_rets_baseline=tuple(tuple(block) for block in fold_rets_baseline),
        rets_baseline_ew=all_rets_baseline_ew,
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
