from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps, project_all_caps
from src.domain.futures.strategy.cs_rank import VOL_FLOOR

_logger = logging.getLogger(__name__)

AllocationPolicy = Literal[
    "equal_weight",
    "inverse_vol",
    "kelly",
    "l1_confidence_shrinkage",
]

_VALID_POLICIES: frozenset[str] = frozenset({"equal_weight", "inverse_vol", "kelly", "l1_confidence_shrinkage"})


@dataclass(slots=True, frozen=True)
class AllocationPolicyScore:
    policy: AllocationPolicy
    growth_lcb: float
    cagr: float
    mdd: float
    cvar_95: float
    leverage: float
    n_blocks: int
    feasible: bool
    reason: str


@dataclass(slots=True, frozen=True)
class AllocationPolicyDecision:
    selected_policy: AllocationPolicy
    scores: tuple[AllocationPolicyScore, ...]
    fallback_reason: str = ""


def compute_l1_confidence(
    *,
    mu_bps: NDArray[np.float64],
    l1_edge_margin_bps_per_bar: NDArray[np.float64],
    quality_weight: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute bounded L1 evidence confidence. [ADR_20260722_L1_L2_COMPOUNDING_ALIGNMENT]"""
    mu = np.asarray(mu_bps, dtype=np.float64).ravel()
    margin = np.asarray(l1_edge_margin_bps_per_bar, dtype=np.float64).ravel()
    quality = np.asarray(quality_weight, dtype=np.float64).ravel()
    if mu.shape != margin.shape or mu.shape != quality.shape:
        raise ValueError(f"shape mismatch: mu={mu.shape} margin={margin.shape} quality={quality.shape}")
    sign_ok = np.sign(mu) == np.sign(margin)
    sign_ok = sign_ok & (np.abs(mu) > 1e-12)
    r = np.clip(
        np.abs(margin) / np.maximum(np.abs(mu), 1e-12),
        0.0,
        1.0,
    )
    q = np.clip(quality, 0.0, 1.0)
    c = sign_ok.astype(np.float64) * q * r
    return np.where(np.isfinite(c), c, 0.0)


def _l1_normalize(w: NDArray[np.float64], support: NDArray[np.bool_]) -> NDArray[np.float64]:
    gross = float(np.sum(np.abs(w[support])))
    if gross > 1e-12:
        out = w.copy()
        out[support] = w[support] / gross
        return out
    return np.zeros_like(w, dtype=np.float64)


def _build_equal_weight(
    mu_bps: NDArray[np.float64],
    support: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Build one allocation-policy weight vector. [ADR_20260722_L1_L2_COMPOUNDING_ALIGNMENT]"""
    n_support = int(np.sum(support))
    if n_support == 0:
        return np.zeros_like(mu_bps, dtype=np.float64)
    w = np.zeros_like(mu_bps, dtype=np.float64)
    w[support] = np.sign(mu_bps[support]) / n_support
    return w


def _build_inverse_vol(
    mu_bps: NDArray[np.float64],
    sigma: NDArray[np.float64],
    support: NDArray[np.bool_],
) -> NDArray[np.float64]:
    n_support = int(np.sum(support))
    if n_support == 0:
        return np.zeros_like(mu_bps, dtype=np.float64)
    sig_clipped = np.maximum(np.nan_to_num(sigma, nan=VOL_FLOOR, posinf=VOL_FLOOR, neginf=VOL_FLOOR), VOL_FLOOR)
    inv_vol = np.where(support, 1.0 / sig_clipped, 0.0)
    total = float(np.sum(inv_vol))
    if total <= 1e-12:
        return np.zeros_like(mu_bps, dtype=np.float64)
    w = np.where(support, np.sign(mu_bps) * inv_vol / total, 0.0)
    return w


def _build_kelly(
    mu_bps: NDArray[np.float64],
    sigma: NDArray[np.float64],
    support: NDArray[np.bool_],
) -> NDArray[np.float64]:
    n_support = int(np.sum(support))
    if n_support == 0:
        return np.zeros_like(mu_bps, dtype=np.float64)
    sig_clipped = np.maximum(np.nan_to_num(sigma, nan=VOL_FLOOR, posinf=VOL_FLOOR, neginf=VOL_FLOOR), VOL_FLOOR)
    var = sig_clipped ** 2
    mu_ret = mu_bps * 1e-4
    w_raw = np.where(support, mu_ret / var, 0.0)
    gross = float(np.sum(np.abs(w_raw)))
    if gross <= 1e-12:
        return np.zeros_like(mu_bps, dtype=np.float64)
    w = w_raw / gross
    return w


def _build_confidence_shrinkage(
    mu_bps: NDArray[np.float64],
    sigma: NDArray[np.float64],
    l1_edge_margin_bps_per_bar: NDArray[np.float64],
    quality_weight: NDArray[np.float64],
    support: NDArray[np.bool_],
) -> NDArray[np.float64]:
    c = compute_l1_confidence(
        mu_bps=mu_bps,
        l1_edge_margin_bps_per_bar=l1_edge_margin_bps_per_bar,
        quality_weight=quality_weight,
    )
    u_k = _build_kelly(mu_bps, sigma, support)
    u_iv = _build_inverse_vol(mu_bps, sigma, support)
    w_blend = c * u_k + (1.0 - c) * u_iv
    w = _l1_normalize(w_blend, support)
    return w.astype(np.float64)


def build_policy_weights(
    *,
    policy: AllocationPolicy,
    mu_bps: NDArray[np.float64],
    sigma: NDArray[np.float64],
    l1_edge_margin_bps_per_bar: NDArray[np.float64],
    quality_weight: NDArray[np.float64],
    caps: PortfolioCaps,
    prev_w: NDArray[np.float64],
    no_trade_band: float,
    vol_target: float | None,
    btc_beta: NDArray[np.float64] | None,
    bars_per_year: float,
    support_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    if policy not in _VALID_POLICIES:
        raise ValueError(f"unknown allocation policy: {policy!r}")

    mu = np.asarray(mu_bps, dtype=np.float64).ravel()
    sig = np.asarray(sigma, dtype=np.float64).ravel()
    p_w = np.asarray(prev_w, dtype=np.float64).ravel()
    n = mu.size
    beta = np.zeros(n, dtype=np.float64) if btc_beta is None else np.asarray(btc_beta, dtype=np.float64).ravel()

    if mu.size != sig.size or mu.size != p_w.size:
        raise ValueError(f"shape mismatch: mu={mu.shape}, sigma={sig.shape}, prev_w={p_w.shape}")
    l1_margin = np.asarray(l1_edge_margin_bps_per_bar, dtype=np.float64).ravel()
    quality = np.asarray(quality_weight, dtype=np.float64).ravel()
    if l1_margin.size != n or quality.size != n:
        raise ValueError(
            f"shape mismatch: mu={mu.shape}, l1_margin={l1_margin.shape}, quality={quality.shape}"
        )

    support = np.abs(mu) > 1e-12 if support_mask is None else np.asarray(support_mask, dtype=bool).ravel()
    if support.size != n:
        support = np.abs(mu) > 1e-12

    if not np.any(support):
        return np.zeros(n, dtype=np.float64)

    if policy == "equal_weight":
        w_shape = _build_equal_weight(mu, support)
    elif policy == "inverse_vol":
        w_shape = _build_inverse_vol(mu, sig, support)
    elif policy == "kelly":
        w_shape = _build_kelly(mu, sig, support)
    elif policy == "l1_confidence_shrinkage":
        w_shape = _build_confidence_shrinkage(
            mu, sig, l1_margin, quality, support,
        )
    else:
        raise ValueError(f"unknown allocation policy: {policy!r}")

    sigma_port = float(np.sqrt(np.dot(np.clip(w_shape, -caps.per_symbol, caps.per_symbol) ** 2, np.maximum(sig, VOL_FLOOR) ** 2)))

    effective_caps = (
        caps if vol_target is None
        else PortfolioCaps(
            gross=caps.gross,
            per_symbol=caps.per_symbol,
            net=caps.net,
            beta=caps.beta,
            target_ann_vol=vol_target,
        )
    )

    w_capped = project_all_caps(
        w_shape,
        beta,
        sigma_port,
        bars_per_year,
        effective_caps,
        support_mask=support,
        allow_vol_upscale=True,
    )

    delta = np.abs(w_capped - p_w)
    w_final = np.where(delta >= no_trade_band, w_capped, p_w)
    w_final = np.where(support, w_final, 0.0)

    if not np.all(np.isfinite(w_final)):
        return np.zeros(n, dtype=np.float64)

    return w_final.astype(np.float64)


def _block_log_growth(
    returns: NDArray[np.float64],
    block_bars: int,
) -> NDArray[np.float64]:
    n = returns.size
    if n < block_bars:
        return np.array([float(np.sum(returns))], dtype=np.float64)
    n_blocks = n // block_bars
    blocks = returns[:n_blocks * block_bars].reshape(n_blocks, block_bars)
    return np.asarray(np.log1p(blocks).sum(axis=1), dtype=np.float64)


def _growth_lower_confidence_bound(
    block_growth: NDArray[np.float64],
    z: float,
) -> float:
    if block_growth.size < 1:
        return float("-inf")
    mean = float(np.mean(block_growth))
    if block_growth.size < 2:
        return mean
    std = float(np.std(block_growth, ddof=1))
    se = std / math.sqrt(float(block_growth.size))
    return float(mean - float(z) * se)


def _cagr_from_returns(returns: NDArray[np.float64], bars_per_year: float) -> float:
    if returns.size < 2:
        return float("-inf")
    total = float(np.sum(returns))
    n_years = float(returns.size) / max(bars_per_year, 1e-9)
    return total / max(n_years, 1e-9)


def _mdd(returns: NDArray[np.float64]) -> float:
    cum = np.log1p(np.maximum(returns, -1.0 + 1e-9)).cumsum()
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(-np.min(dd)) if dd.size > 0 else 0.0


def _cvar_95(returns: NDArray[np.float64]) -> float:
    if returns.size < 2:
        return 0.0
    sorted_r = np.sort(returns)
    n_var = max(1, int(np.ceil(0.05 * returns.size)))
    return float(-np.mean(sorted_r[:n_var]))


def select_fit_allocation_policy(
    *,
    returns_by_policy: Mapping[AllocationPolicy, NDArray[np.float64]],
    leverage_by_policy: Mapping[AllocationPolicy, float],
    bars_per_year: float,
    block_bars: int,
    growth_lcb_z: float,
    max_mdd: float,
    max_cvar_95: float,
    min_growth_lcb: float,
) -> AllocationPolicyDecision:
    _tie_breaker: tuple[AllocationPolicy, ...] = (
        "inverse_vol",
        "equal_weight",
        "l1_confidence_shrinkage",
        "kelly",
    )
    scores: list[AllocationPolicyScore] = []
    for policy in returns_by_policy:
        if policy not in _VALID_POLICIES:
            raise ValueError(f"unknown policy: {policy!r}")
        rets = np.asarray(returns_by_policy[policy], dtype=np.float64)
        lev = float(leverage_by_policy.get(policy, 1.0))
        deployed = rets * lev
        if not np.all(np.isfinite(deployed)):
            scores.append(AllocationPolicyScore(
                policy=policy, growth_lcb=float("-inf"), cagr=float("-inf"),
                mdd=float("inf"), cvar_95=float("inf"), leverage=lev,
                n_blocks=0, feasible=False, reason="non_finite_returns",
            ))
            continue
        block_g = _block_log_growth(deployed, block_bars)
        g_lcb = _growth_lower_confidence_bound(block_g, growth_lcb_z)
        cagr = _cagr_from_returns(deployed, bars_per_year)
        mdd = _mdd(deployed)
        cv95 = _cvar_95(deployed)
        n_blocks = block_g.size
        if not np.isfinite(g_lcb) or not np.isfinite(cagr):
            scores.append(AllocationPolicyScore(
                policy=policy, growth_lcb=g_lcb, cagr=cagr,
                mdd=mdd, cvar_95=cv95, leverage=lev,
                n_blocks=n_blocks, feasible=False, reason="non_finite_metrics",
            ))
            continue
        feasible = (
            n_blocks >= 3
            and mdd <= max_mdd
            and cv95 <= max_cvar_95
            and g_lcb >= min_growth_lcb
        )
        reason = "" if feasible else f"mdd={mdd:.4f}>={max_mdd:.4f} or cvar={cv95:.4f}>={max_cvar_95:.4f} or g_lcb={g_lcb:.4f}<{min_growth_lcb:.4f}"
        scores.append(AllocationPolicyScore(
            policy=policy, growth_lcb=g_lcb, cagr=cagr,
            mdd=mdd, cvar_95=cv95, leverage=lev,
            n_blocks=n_blocks, feasible=feasible, reason=reason,
        ))

    feasible_scores = [s for s in scores if s.feasible]
    if not feasible_scores:
        _logger.info(
            "[L2-POLICY] no feasible policy, falling back to inverse_vol: %s",
            [(s.policy, s.reason) for s in scores],
        )
        return AllocationPolicyDecision(
            selected_policy="inverse_vol",
            scores=tuple(scores),
            fallback_reason="insufficient_fit_evidence",
        )

    best = max(feasible_scores, key=lambda s: (s.growth_lcb, -_tie_breaker.index(s.policy)))
    for tie_policy in _tie_breaker:
        tied = [s for s in feasible_scores if np.isclose(s.growth_lcb, best.growth_lcb, atol=1e-12) and s.policy == tie_policy]
        if tied:
            best = tied[0]
            break

    _logger.info("[L2-POLICY] selected=%s growth_lcb=%.6f scores=%s", best.policy, best.growth_lcb, [(s.policy, round(s.growth_lcb, 6), s.feasible) for s in scores])
    return AllocationPolicyDecision(
        selected_policy=best.policy,
        scores=tuple(scores),
    )


def choose_deployed_policy(
    *,
    selected: AllocationPolicyScore,
    inverse_vol: AllocationPolicyScore,
) -> tuple[AllocationPolicy | None, bool]:
    sel_ok = selected.feasible and np.isfinite(selected.growth_lcb)
    iv_ok = inverse_vol.feasible and np.isfinite(inverse_vol.growth_lcb)

    if sel_ok and iv_ok:
        if selected.growth_lcb >= inverse_vol.growth_lcb:
            return selected.policy, False
        return inverse_vol.policy, True

    if sel_ok and not iv_ok:
        return selected.policy, False

    if iv_ok and not sel_ok:
        return inverse_vol.policy, True

    return None, False
