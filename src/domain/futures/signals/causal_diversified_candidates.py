"""liquidity_participation_breakout / btc_neutral_residual_reversal panel generators.

[ADR_20260710_L0_CAUSAL_LIQUIDITY_SIGNAL_DIVERSIFICATION][ADR_20260710_L0_SIGNAL_BREADTH_DIVERSITY_REDESIGN]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig

_EPS = 1e-10


# ── Trailing-exclusive rolling helpers ─────────────────────────────────


def _trailing_max(
    values_2d: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    n = len(values_2d)
    result = np.full_like(values_2d, np.nan)
    for t in range(window, n):
        result[t] = np.max(values_2d[t - window : t], axis=0)
    return result


def _trailing_min(
    values_2d: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    n = len(values_2d)
    result = np.full_like(values_2d, np.nan)
    for t in range(window, n):
        result[t] = np.min(values_2d[t - window : t], axis=0)
    return result


def _trailing_mean(
    values_2d: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    n = len(values_2d)
    result = np.full_like(values_2d, np.nan)
    for t in range(window, n):
        result[t] = np.mean(values_2d[t - window : t], axis=0)
    return result


def _trailing_std(
    values_2d: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    n = len(values_2d)
    result = np.full_like(values_2d, np.nan)
    for t in range(window, n):
        result[t] = np.std(values_2d[t - window : t], axis=0, ddof=0)
    return result


def _trailing_var(
    values_2d: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    n = len(values_2d)
    result = np.full_like(values_2d, np.nan)
    for t in range(window, n):
        result[t] = np.var(values_2d[t - window : t], axis=0, ddof=0)
    return result


def _trailing_cov(
    a_2d: NDArray[np.float64],
    b_2d: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    n = len(a_2d)
    result = np.full_like(a_2d, np.nan)
    for t in range(window, n):
        a_slice = a_2d[t - window : t]
        b_slice = b_2d[t - window : t]
        a_mean = np.mean(a_slice, axis=0)
        b_mean = np.mean(b_slice, axis=0)
        result[t] = np.mean((a_slice - a_mean) * (b_slice - b_mean), axis=0)
    return result


# ── Main generator ─────────────────────────────────────────────────────


def _build_lpb_panels(
    *,
    aligned: AlignedMarketData,
    lpb_cfg: object,
    valid_mask_2d: NDArray[np.bool_],
    atr_2d: NDArray[np.float64],
) -> list[CandidateSignalPanel]:
    from src.domain.futures.strategy.config import LiquidityParticipationBreakoutConfig

    cfg: LiquidityParticipationBreakoutConfig = lpb_cfg  # type: ignore[assignment]
    panels: list[CandidateSignalPanel] = []
    t, n = aligned.close_2d.shape

    high = aligned.high_2d
    low = aligned.low_2d
    close = aligned.close_2d
    volume = aligned.volume_2d
    cost_arr = aligned.execution_cost_bps_2d
    adv_arr = aligned.adv_usdt_2d

    has_liquidity = cost_arr is not None and adv_arr is not None
    cost: NDArray[np.float64] | None = cost_arr
    adv: NDArray[np.float64] | None = adv_arr

    for w in cfg.channel_bars:
        if w < 2 or t <= w:
            continue

        h_max = _trailing_max(high, w)
        l_min = _trailing_min(low, w)

        break_up = close > h_max
        break_down = close < l_min
        has_break = break_up | break_down

        impulse = np.full_like(close, 0.0)
        ref = np.where(break_up, h_max, np.where(break_down, l_min, close))
        impulse = np.abs(close - ref) / np.maximum(atr_2d, _EPS)

        vol_mean = _trailing_mean(volume, w)
        vol_std = _trailing_std(volume, w)
        volume_z = (volume - vol_mean) / np.maximum(vol_std, _EPS)

        active_mask = aligned.active_mask
        liquid = np.zeros_like(close, dtype=np.bool_)  # fail-closed default [LIMIT-06]
        if has_liquidity and cost is not None and adv is not None:
            cost_finite = np.isfinite(cost)
            adv_finite = np.isfinite(adv)
            liquid = cost_finite & adv_finite & active_mask  # [LIMIT-05][LIMIT-07]

        signal = (
            has_break
            & (impulse >= cfg.min_breakout_impulse_atr)
            & (volume_z >= cfg.min_volume_zscore)
            & liquid
            & valid_mask_2d
        )

        side_hint = np.zeros_like(signal, dtype=np.int8)
        side_hint[break_up & signal] = 1
        side_hint[break_down & signal] = -1

        score = np.zeros_like(close)
        raw_score = np.minimum(1.0, impulse / cfg.score_impulse_atr)
        score[side_hint == 1] = raw_score[side_hint == 1]
        score[side_hint == -1] = -raw_score[side_hint == -1]

        holding = w
        min_hold = max(1, holding // 4)

        panel = CandidateSignalPanel(
            family="liquidity_participation_breakout",
            variant=f"lpb_{w}",
            params={"channel": w},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=score,
            side_hint_2d=side_hint,
            expected_holding_bars=holding,
            min_holding_bars=min_hold,
            stop_atr_mult=2.0,
            take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((t, n), dtype=np.float64),
            valid_mask_2d=signal,
            metadata={
                "entry_mode": "sparse",
                "edge_hypothesis": "liquidity_participation_breakout",
                "liquidity_gate": "cost_adv",
            },
        )
        panels.append(panel)

    return panels


def _build_bnrr_panels(
    *,
    aligned: AlignedMarketData,
    bnrr_cfg: object,
    valid_mask_2d: NDArray[np.bool_],
    atr_2d: NDArray[np.float64],
    btc_index: int,
) -> list[CandidateSignalPanel]:
    from src.domain.futures.strategy.config import BtcNeutralResidualReversalConfig

    cfg: BtcNeutralResidualReversalConfig = bnrr_cfg  # type: ignore[assignment]
    panels: list[CandidateSignalPanel] = []
    t, n = aligned.close_2d.shape

    close = aligned.close_2d
    cost_arr = aligned.execution_cost_bps_2d
    adv_arr = aligned.adv_usdt_2d

    log_returns = np.log(close[1:] / close[:-1])
    log_returns = np.vstack([np.zeros((1, n)), log_returns])

    btc_ret = log_returns[:, btc_index:btc_index + 1]

    has_liquidity = cost_arr is not None and adv_arr is not None
    cost: NDArray[np.float64] | None = cost_arr
    adv: NDArray[np.float64] | None = adv_arr

    for w in cfg.lookback_bars:
        if w < 2 or t <= w:
            continue

        btc_tiled = np.tile(btc_ret, (1, n))
        beta = _trailing_cov(log_returns, btc_tiled, w) / np.maximum(
            _trailing_var(btc_tiled, w), _EPS
        )

        u_i = np.zeros_like(close)
        for bar in range(w + 1, t):
            beta_t = beta[bar]
            k_start = bar - w
            ret_window = log_returns[k_start:bar]
            btc_window = btc_ret[k_start:bar]
            res_window = ret_window - beta_t[np.newaxis, :] * btc_window
            u_i[bar] = np.sum(res_window, axis=0)

        active_mask = aligned.active_mask
        eligible = np.zeros_like(close, dtype=np.bool_)  # fail-closed default [LIMIT-06]
        if has_liquidity and cost is not None and adv is not None:
            cost_finite = np.isfinite(cost)
            adv_finite = np.isfinite(adv)
            beta_ok = np.abs(beta) <= cfg.max_abs_btc_beta
            eligible = cost_finite & adv_finite & active_mask & beta_ok  # [LIMIT-05][LIMIT-07]

        eligible = eligible & valid_mask_2d

        side_hint = np.zeros((t, n), dtype=np.int8)
        score = np.zeros((t, n), dtype=np.float64)

        for bar in range(w + 1, t):
            eligible_bar = eligible[bar]
            n_eligible = int(np.sum(eligible_bar))
            if n_eligible < cfg.min_cross_section:
                continue

            res_vals = u_i[bar]
            mask = eligible_bar
            masked_vals = res_vals[mask]
            n_eligible_actual = masked_vals.shape[0]

            if n_eligible_actual < cfg.min_cross_section:
                continue

            ranks = np.argsort(np.argsort(masked_vals))
            pct = ranks.astype(np.float64) / max(n_eligible_actual - 1, 1)
            centered_pct = pct - 0.5

            tail_n = max(1, int(n_eligible_actual * cfg.tail_fraction))
            long_idx = np.argsort(masked_vals)[:tail_n]
            short_idx = np.argsort(masked_vals)[-tail_n:]

            sym_indices = np.where(mask)[0]
            for idx in sym_indices[long_idx]:
                side_hint[bar, idx] = 1
            for idx in sym_indices[short_idx]:
                side_hint[bar, idx] = -1

            for i, idx in enumerate(sym_indices):
                score[bar, idx] = -centered_pct[i]

        holding = w
        min_hold = max(1, holding // 4)

        panel = CandidateSignalPanel(
            family="btc_neutral_residual_reversal",
            variant=f"bnrr_{w}",
            params={"lookback": w, "tail_fraction": cfg.tail_fraction},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=score,
            side_hint_2d=side_hint,
            expected_holding_bars=holding,
            min_holding_bars=min_hold,
            stop_atr_mult=2.0,
            take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((t, n), dtype=np.float64),
            valid_mask_2d=(side_hint != 0),
            metadata={
                "entry_mode": "cross_sectional_rank",
                "edge_hypothesis": "btc_neutral_residual_reversal",
                "liquidity_gate": "cost_adv",
            },
        )
        panels.append(panel)

    return panels


def build_causal_diversified_signal_panels(
    *,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    valid_mask_2d: NDArray[np.bool_],
    atr_2d: NDArray[np.float64],
    btc_index: int,
) -> tuple[CandidateSignalPanel, ...]:
    """Build causal liquidity-qualified breakout and BTC-neutral reversal panels."""
    if btc_index < 0:
        return ()

    panels: list[CandidateSignalPanel] = []

    lpb = _build_lpb_panels(
        aligned=aligned,
        lpb_cfg=cfg.liquidity_participation_breakout,
        valid_mask_2d=valid_mask_2d,
        atr_2d=atr_2d,
    )
    panels.extend(lpb)

    bnrr = _build_bnrr_panels(
        aligned=aligned,
        bnrr_cfg=cfg.btc_neutral_residual_reversal,
        valid_mask_2d=valid_mask_2d,
        atr_2d=atr_2d,
        btc_index=btc_index,
    )
    panels.extend(bnrr)

    return tuple(panels)
