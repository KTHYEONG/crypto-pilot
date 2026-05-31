"""Single SSOT for alpha - cost - hurdle composition (compose_mu)."""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.stats

from src.domain.futures.forecast.contracts import AlphaForecast, CostForecast
from src.domain.futures.strategy.alpha_evaluation import derive_signed_rank_signal
from src.domain.futures.strategy.rank_selection import (
    RankSelectionPolicy,
    apply_rank_selection_policy,
    policy_from_dict,
)


def _rank_weight_1d(ev_row: np.ndarray, *, k: float = 3.0) -> np.ndarray:
    """Cross-sectional rank to continuous weight in (-1, +1) via tanh.

    Args:
        ev_row: 1-D EV signal array for a single bar [N].
        k: Scaling factor controlling weight steepness (default 3.0).

    Returns:
        Weight array of same shape [N] with values in (-1, +1).

    Time: O(N log N), Space: O(N)

    """
    finite = np.isfinite(ev_row)
    out = np.zeros_like(ev_row)
    n = int(finite.sum())
    if n < 2:
        return out
    # rank in [0, 1]
    ranks = scipy.stats.rankdata(ev_row[finite], method="average") / n
    # centered: [-0.5, +0.5], scaled: [-k/2, k/2]
    out[finite] = np.tanh(k * (ranks - 0.5))  # -> (-1, +1)
    return out


def _soft_hurdle(ev: np.ndarray, cost_bps: float, *, steepness: float = 5.0) -> np.ndarray:
    """Sigmoid gate: near-zero EV gets smoothly attenuated near cost threshold.

    Args:
        ev: EV signal array (any shape).
        cost_bps: Cost threshold in basis points.
        steepness: Sigmoid steepness (default 5.0).

    Returns:
        Attenuated EV array of same shape.

    Time: O(N), Space: O(N)

    """
    cost_frac = cost_bps / 1e4
    exponent = np.clip(
        -steepness * (ev - cost_frac) / max(cost_frac, 1e-8), -500.0, 500.0
    )
    gate = 1.0 / (1.0 + np.exp(exponent))
    return ev * gate


def _cs_zscore(score_2d: np.ndarray) -> np.ndarray:
    """Bar-wise cross-sectional z-score (NaN-safe).

    Args:
        score_2d: Input array of shape [T, N].

    Returns:
        Z-scored array of same shape [T, N]. Non-finite positions are set to 0.

    Time: O(T * N), Space: O(T * N)

    """
    out = np.full_like(score_2d, 0.0, dtype=np.float64)
    for t in range(score_2d.shape[0]):
        row = score_2d[t]
        finite = np.isfinite(row)
        if finite.sum() < 3:
            continue
        mu = np.mean(row[finite])
        sd = np.std(row[finite], ddof=1)
        if sd < 1e-12:
            continue
        out[t] = np.where(finite, (row - mu) / sd, 0.0)
    return out


def compose_mu(
    alpha: AlphaForecast,
    cost: CostForecast,
    params: dict[str, Any],
    *,
    holding_bars: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compose expected returns after cost and hurdle gate.

    Args:
        alpha: Typed alpha forecast.
        cost: Typed cost forecast.
        params: Trial/strategy params dict (must contain BETA_ALPHA, EV_HURDLE_BPS).
        holding_bars: Expected holding duration in bars. Only used when
            COST_GATE_AMORTIZE=True to amortize round-trip cost per bar.

    Returns:
        Tuple of (xs_long_2d, xs_short_2d, mu_long_2d, mu_short_2d).

    """
    from src.domain.futures.optimization.opt_config import (
        OPT_FUTURES_CONFIG,
        default_ev_hurdle_bps,
    )

    beta_a = float(
        params.get("BETA_ALPHA", OPT_FUTURES_CONFIG.get("FUTURES_DEFAULT_BETA_ALPHA", 1.0))
    )
    ev_h = float(params.get("EV_HURDLE_BPS", default_ev_hurdle_bps(OPT_FUTURES_CONFIG)))

    cost_frac = np.asarray(cost.execution_cost_fraction_2d, dtype=np.float64)

    # Maker-friendly 복합 실질 비용 모델 반영 (보수적 설정을 위해 Taker 위주 80% 기본값 적용)
    maker_ratio = float(params.get("MAKER_RATIO", 0.20))
    maker_fee = float(params.get("MAKER_FEE_BPS", 2.0))
    taker_fee = float(params.get("TAKER_FEE_BPS", 5.0))
    slippage = float(params.get("SLIPPAGE_BPS", 2.0))

    effective_rt = (
        (maker_ratio * maker_fee) + ((1.0 - maker_ratio) * taker_fee) + slippage
    ) * 2.0 / 10000.0
    baseline_rt = (taker_fee + slippage) * 2.0 / 10000.0
    if baseline_rt > 1e-12:
        cost_frac = cost_frac * (effective_rt / baseline_rt)

    if params.get("COST_GATE_AMORTIZE", False) and holding_bars and int(holding_bars) > 1:
        cost_frac = cost_frac / float(int(holding_bars))

    mu_long = beta_a * np.asarray(alpha.alpha_long_2d, dtype=np.float64) - cost_frac
    mu_short = beta_a * np.asarray(alpha.alpha_short_2d, dtype=np.float64) - cost_frac
    admission_mode = str(params.get("POST_COST_ADMISSION_MODE", "ev_gate"))
    if admission_mode == "rank_then_ev_gate":
        rank_l = getattr(alpha, "rank_score_long_2d", None)
        rank_s = getattr(alpha, "rank_score_short_2d", None)
        if rank_l is not None and rank_s is not None:
            top_k = max(1, int(params.get("RANK_PORTFOLIO_TOP_K", 4)))
            min_spread = float(params.get("RANK_PORTFOLIO_MIN_SCORE_SPREAD_BPS", 0.0)) / 10000.0
            rank_l2d = np.asarray(rank_l, dtype=np.float64)
            rank_s2d = np.asarray(rank_s, dtype=np.float64)
            long_mask = np.zeros_like(mu_long, dtype=bool)
            short_mask = np.zeros_like(mu_short, dtype=bool)
            for t in range(mu_long.shape[0]):
                lrow = rank_l2d[t]
                srow = rank_s2d[t]
                lfinite = np.isfinite(lrow) & (lrow > 0.0)
                sfinite = np.isfinite(srow) & (srow < 0.0)
                if np.count_nonzero(lfinite) > 0:
                    idx = np.flatnonzero(lfinite)
                    l_sorted = idx[np.argsort(lrow[idx])[::-1]]
                    pick = l_sorted[:top_k]
                    if pick.size > 0 and (lrow[pick[0]] - lrow[pick[-1]] >= min_spread):
                        long_mask[t, pick] = True
                if np.count_nonzero(sfinite) > 0:
                    idx = np.flatnonzero(sfinite)
                    s_sorted = idx[np.argsort(srow[idx])]
                    pick = s_sorted[:top_k]
                    if pick.size > 0 and (srow[pick[-1]] - srow[pick[0]] >= min_spread):
                        short_mask[t, pick] = True
            mu_long = np.where(long_mask, mu_long, -np.inf)
            mu_short = np.where(short_mask, mu_short, -np.inf)
    elif admission_mode == "rank_cs_neutral":
        rank_l = getattr(alpha, "rank_score_long_2d", None)
        rank_s = getattr(alpha, "rank_score_short_2d", None)
        if rank_l is not None and rank_s is not None:
            policy_payload = params.get("RANK_SELECTION_POLICY")
            if not isinstance(policy_payload, dict):
                policy_payload = getattr(alpha, "rank_selection_policy", None)
            policy: RankSelectionPolicy | None = None
            if isinstance(policy_payload, dict):
                policy = policy_from_dict(policy_payload)
            elif all(
                key in params
                for key in ("RANK_POLICY_POLARITY", "RANK_POLICY_QUANTILE", "RANK_POLICY_MIN_ABS_Z")
            ):
                policy = RankSelectionPolicy(
                    polarity=1 if int(params["RANK_POLICY_POLARITY"]) >= 0 else -1,
                    quantile=float(params["RANK_POLICY_QUANTILE"]),
                    min_abs_z=float(params["RANK_POLICY_MIN_ABS_Z"]),
                    weighting=str(params.get("RANK_POLICY_WEIGHTING", "tanh")),  # type: ignore[arg-type]
                    weight_k=float(params.get("RANK_POLICY_WEIGHT_K", 3.0)),
                    holding_bars=int(params.get("RANK_POLICY_HOLDING_BARS", 12)),
                    validation_net_lcb_bps=float(params.get("RANK_POLICY_VAL_LCB_BPS", -1.0)),
                    validation_gross_bps=float(params.get("RANK_POLICY_VAL_GROSS_BPS", 0.0)),
                    validation_ir_t=float(params.get("RANK_POLICY_VAL_IR_T", 0.0)),
                    validation_monotonicity=float(params.get("RANK_POLICY_VAL_MONO", 0.0)),
                    n_obs=int(params.get("RANK_POLICY_VAL_N_OBS", 0)),
                )
            if policy is not None:
                signed = derive_signed_rank_signal(
                    np.asarray(rank_l, dtype=np.float64),
                    np.asarray(rank_s, dtype=np.float64),
                )
                policy_long, policy_short = apply_rank_selection_policy(
                    signed_score_2d=signed,
                    eligible_2d=np.isfinite(np.asarray(rank_l)) | np.isfinite(np.asarray(rank_s)),
                    policy=policy,
                )
                mu_long = np.where(policy_long > 0.0, policy_long, -np.inf)
                mu_short = np.where(policy_short > 0.0, policy_short, -np.inf)
            else:
                rank_select_q = float(params.get("RANK_SELECT_QUANTILE", 0.33))
                ic_prior = float(params.get("IC_PRIOR_FOR_GATE", 0.03))
                ev_tilt_w = float(params.get("EV_SECONDARY_TILT_WEIGHT", 0.0))
                rank_l2d = np.asarray(rank_l, dtype=np.float64)
                rank_s2d = np.asarray(rank_s, dtype=np.float64)
                z_long = _cs_zscore(rank_l2d)
                z_short = _cs_zscore(-rank_s2d)   # short: lower rank = better, so negate
                if ev_tilt_w > 1e-9:
                    ev_z_long = _cs_zscore(
                        np.asarray(alpha.alpha_long_2d, dtype=np.float64)
                    )
                    ev_z_short = _cs_zscore(
                        np.asarray(alpha.alpha_short_2d, dtype=np.float64)
                    )
                    z_long = (1.0 - ev_tilt_w) * z_long + ev_tilt_w * ev_z_long
                    z_short = (1.0 - ev_tilt_w) * z_short + ev_tilt_w * ev_z_short

                long_mask = np.zeros_like(mu_long, dtype=bool)
                short_mask = np.zeros_like(mu_short, dtype=bool)
                alpha_long_arr = np.asarray(alpha.alpha_long_2d, dtype=np.float64)
                alpha_short_arr = np.asarray(alpha.alpha_short_2d, dtype=np.float64)
                for t in range(mu_long.shape[0]):
                    lrow = z_long[t]
                    srow = z_short[t]
                    l_finite = np.isfinite(lrow) & np.isfinite(alpha_long_arr[t])
                    s_finite = np.isfinite(srow) & np.isfinite(alpha_short_arr[t])
                    n_l = int(l_finite.sum())
                    n_s = int(s_finite.sum())
                    if n_l > 0:
                        k_l = max(1, int(np.ceil(n_l * rank_select_q)))
                        idx_l = np.flatnonzero(l_finite)
                        top_l = idx_l[np.argsort(lrow[idx_l])[::-1][:k_l]]
                        long_mask[t, top_l] = True
                    if n_s > 0:
                        k_s = max(1, int(np.ceil(n_s * rank_select_q)))
                        idx_s = np.flatnonzero(s_finite)
                        top_s = idx_s[np.argsort(srow[idx_s])[::-1][:k_s]]
                        short_mask[t, top_s] = True
                sigma_r = float(params.get("COMPOSER_SIGMA_BPS", 500.0))
                holding_bars_gate = max(1, int(params.get("REBALANCE_BARS", 1)))
                cost_bps = float(params.get("COST_GATE_BPS", 24.0))
                amortized_cost = cost_bps / holding_bars_gate
                z_long_masked = np.where(long_mask, z_long, 0.0)
                z_short_masked = np.where(short_mask, z_short, 0.0)
                soft_steepness = float(params.get("SOFT_HURDLE_STEEPNESS", 5.0))
                for t in range(mu_long.shape[0]):
                    l_sel = long_mask[t].sum()
                    s_sel = short_mask[t].sum()
                    if l_sel < 1 or s_sel < 1:
                        long_mask[t] = False
                        short_mask[t] = False
                        continue
                    zmean_l = float(np.mean(z_long_masked[t, long_mask[t]]))
                    zmean_s = float(np.mean(z_short_masked[t, short_mask[t]]))
                    gross_spread_bps = ic_prior * sigma_r * (zmean_l + zmean_s)
                    net_edge_arr = np.array([gross_spread_bps - amortized_cost])
                    net_edge_soft = float(
                        _soft_hurdle(net_edge_arr, amortized_cost, steepness=soft_steepness)[0]
                    )
                    if net_edge_soft <= 0.0:
                        long_mask[t] = False
                        short_mask[t] = False
                mu_long = np.where(long_mask, z_long, -np.inf)
                mu_short = np.where(short_mask, z_short, -np.inf)

    if admission_mode == "rank_cs_neutral":
        # hurdle은 rank z-score 맥락에서 의미없음; 선택된 포지션은 모두 통과
        xs_long = np.where(np.isfinite(mu_long), mu_long, 0.0)
        xs_short = np.where(np.isfinite(mu_short), mu_short, 0.0)
    else:
        # rank-sizing: 경질 max(ev,0) 절단 대신 횡단면 rank 기반 연속 가중.
        # C-Clip 수정: 클립이 ranking skill을 파괴하는 것을 방지.
        # 순서: (1) EV 단위 soft-hurdle → (2) rank-sizing.
        # soft-hurdle은 EV(return-fraction)에 적용해야 단위가 일치함.
        # rank-sizing(-1,+1)에 bps 기준 hurdle을 적용하면 단위 불일치로 무력화됨.
        rank_k = float(params.get("RANK_WEIGHT_K", 3.0))
        soft_steepness = float(params.get("SOFT_HURDLE_STEEPNESS", 5.0))
        cost_bps = float(params.get("COST_GATE_BPS", ev_h))
        # Step 1: EV 단위 soft-hurdle (overflow 방지를 위해 exp 인수 클립)
        gated_long = _soft_hurdle(mu_long, cost_bps, steepness=soft_steepness)
        gated_short = _soft_hurdle(mu_short, cost_bps, steepness=soft_steepness)
        # Step 2: 횡단면 rank-sizing — soft-hurdle 후 양수 EV만 대상
        n_bars = mu_long.shape[0]
        rw_long = np.zeros_like(mu_long)
        rw_short = np.zeros_like(mu_short)
        for t in range(n_bars):
            rw_long[t] = _rank_weight_1d(gated_long[t], k=rank_k)
            rw_short[t] = _rank_weight_1d(gated_short[t], k=rank_k)
        xs_long = np.where(rw_long > 0.0, rw_long, 0.0)
        xs_short = np.where(rw_short > 0.0, rw_short, 0.0)
    return xs_long, xs_short, mu_long, mu_short
