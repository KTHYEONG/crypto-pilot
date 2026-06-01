from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import scipy.stats
from numpy.typing import NDArray

from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps, project_all_caps

RankSelectionMode = Literal["tail", "soft_cs"]


@dataclass(frozen=True, slots=True)
class RankSelectionPolicy:
    """Validation-calibrated mapping from signed rank score to long/short baskets."""

    polarity: Literal[1, -1]
    quantile: float
    min_abs_z: float
    weighting: Literal["equal", "zscore", "tanh"]
    weight_k: float
    holding_bars: int
    validation_net_lcb_bps: float
    validation_gross_bps: float
    validation_ir_t: float
    validation_monotonicity: float
    n_obs: int
    selection_mode: RankSelectionMode = "tail"
    validation_turnover: float = float("nan")
    validation_cost_bps: float = float("nan")
    validation_breadth: float = float("nan")
    validation_abs_net_exposure: float = float("nan")
    validation_abs_beta_exposure: float = float("nan")


def policy_is_no_trade(policy: RankSelectionPolicy) -> bool:
    """Return True when validation failed and policy intentionally emits no trade."""
    return bool(policy.validation_net_lcb_bps <= 0.0 or policy.n_obs <= 0)


def _cs_zscore_2d(score_2d: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute bar-wise cross-sectional z-score."""
    out = np.full_like(score_2d, np.nan, dtype=np.float64)
    for t in range(score_2d.shape[0]):
        row = score_2d[t]
        finite = np.isfinite(row)
        if int(np.count_nonzero(finite)) < 3:
            continue
        mu = float(np.mean(row[finite]))
        sd = float(np.std(row[finite], ddof=1))
        if sd <= 1e-12:
            continue
        out[t, finite] = (row[finite] - mu) / sd
    return out


def _selection_masks(
    score_row: NDArray[np.float64],
    eligible_row: NDArray[np.bool_],
    *,
    q: float,
    min_abs_z: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    long_mask = np.zeros_like(eligible_row, dtype=bool)
    short_mask = np.zeros_like(eligible_row, dtype=bool)
    finite = eligible_row & np.isfinite(score_row)
    n = int(np.count_nonzero(finite))
    if n < 2:
        return long_mask, short_mask
    k = max(1, int(np.ceil(n * q)))
    idx = np.flatnonzero(finite)
    row = score_row[idx]
    long_ranked = idx[np.argsort(row)[::-1][:k]]
    short_ranked = idx[np.argsort(row)[:k]]
    if min_abs_z > 0.0:
        long_ranked = long_ranked[score_row[long_ranked] >= min_abs_z]
        short_ranked = short_ranked[score_row[short_ranked] <= -min_abs_z]
    long_mask[long_ranked] = True
    short_mask[short_ranked] = True
    short_mask[long_ranked] = False
    return long_mask, short_mask


def _score_monotonicity(
    score_2d: NDArray[np.float64],
    realized_2d: NDArray[np.float64],
    eligible_2d: NDArray[np.bool_],
) -> float:
    bucket_scores: list[float] = []
    bucket_rets: list[float] = []
    for t in range(score_2d.shape[0]):
        score = score_2d[t]
        ret = realized_2d[t]
        mask = eligible_2d[t] & np.isfinite(score) & np.isfinite(ret)
        if int(np.count_nonzero(mask)) < 10:
            continue
        s = score[mask]
        r = ret[mask]
        edges = np.quantile(s, [0.2, 0.4, 0.6, 0.8])
        bins = np.digitize(s, edges, right=True)
        for b in range(5):
            bmask = bins == b
            if int(np.count_nonzero(bmask)) < 2:
                continue
            bucket_scores.append(float(np.mean(s[bmask])))
            bucket_rets.append(float(np.mean(r[bmask])) * 1e4)
    if len(bucket_scores) < 5:
        return float("nan")
    rho = scipy.stats.spearmanr(np.asarray(bucket_scores), np.asarray(bucket_rets)).statistic
    return float(rho) if np.isfinite(rho) else float("nan")


def build_signed_rank_weights(
    *,
    signed_score_2d: NDArray[np.float64],
    eligible_2d: NDArray[np.bool_],
    policy: RankSelectionPolicy,
    beta_2d: NDArray[np.float64] | None = None,
    gross_target: float = 1.0,
    max_abs_net_exposure: float = 0.05,
    max_abs_beta_exposure: float = 0.20,
) -> NDArray[np.float64]:
    """Convert rank scores to signed portfolio weights without realized returns."""
    score_2d = np.asarray(signed_score_2d, dtype=np.float64)
    eligible = np.asarray(eligible_2d, dtype=bool)
    z = _cs_zscore_2d(score_2d)
    signed = policy.polarity * z
    out = np.zeros_like(score_2d, dtype=np.float64)
    if policy_is_no_trade(policy):
        return out

    beta_arr = (
        np.asarray(beta_2d, dtype=np.float64)
        if beta_2d is not None and np.asarray(beta_2d).shape == score_2d.shape
        else np.zeros_like(score_2d, dtype=np.float64)
    )
    for t in range(score_2d.shape[0]):
        row_elig = eligible[t] & np.isfinite(signed[t])
        if int(np.count_nonzero(row_elig)) < 2:
            continue
        if policy.selection_mode == "tail":
            lmask, smask = _selection_masks(
                signed[t], row_elig, q=policy.quantile, min_abs_z=policy.min_abs_z
            )
            if policy.weighting == "equal":
                out[t, lmask] = 1.0
                out[t, smask] = -1.0
            else:
                lv = np.abs(signed[t, lmask])
                sv = np.abs(signed[t, smask])
                if policy.weighting == "zscore":
                    out[t, lmask] = lv
                    out[t, smask] = -sv
                else:
                    out[t, lmask] = np.tanh(policy.weight_k * lv)
                    out[t, smask] = -np.tanh(policy.weight_k * sv)
        else:
            row = signed[t]
            if policy.weighting == "equal":
                w = np.sign(row)
            elif policy.weighting == "zscore":
                w = row.copy()
            else:
                w = np.tanh(policy.weight_k * row)
            w = np.where(row_elig, w, 0.0)
            w[row_elig] -= float(np.mean(w[row_elig]))
            out[t] = w

        row = out[t]
        row = np.where(row_elig, row, 0.0)
        gross = float(np.sum(np.abs(row)))
        if gross > 1e-12:
            row = row * (float(gross_target) / gross)
        row = project_all_caps(
            row,
            btc_beta=np.where(np.isfinite(beta_arr[t]), beta_arr[t], 0.0),
            sigma_port=0.0,
            bars_per_year=1.0,
            caps=PortfolioCaps(
                gross=max(float(gross_target), 1e-9),
                per_symbol=max(float(gross_target), 1e-9),
                net=max_abs_net_exposure,
                beta=max_abs_beta_exposure,
                target_ann_vol=1.0,
            ),
        )
        out[t] = np.where(row_elig, row, 0.0)
    return out


def _estimate_policy_metrics(
    *,
    weights: NDArray[np.float64],
    realized: NDArray[np.float64],
    eligible: NDArray[np.bool_],
    execution_cost_bps_2d: NDArray[np.float64] | None,
    cost_bps_fallback: float,
    beta_2d: NDArray[np.float64] | None,
    score: NDArray[np.float64],
    compat_static_cost: bool = False,
) -> dict[str, float]:
    gross_bps: list[float] = []
    net_bps: list[float] = []
    cost_bps: list[float] = []
    turnover: list[float] = []
    breadth: list[float] = []
    net_exp: list[float] = []
    beta_exp: list[float] = []
    for t in range(weights.shape[0]):
        w = weights[t]
        r = realized[t]
        active = np.isfinite(score[t])
        mask = active & eligible[t] & np.isfinite(w) & np.isfinite(r)
        if int(np.count_nonzero(active & eligible[t])) < 2:
            continue
        w_row = np.where(mask, w, 0.0)
        r_row = np.where(mask, r, 0.0)
        gross = float(np.sum(w_row * r_row) * 1e4)
        if t == 0:
            delta = np.abs(w_row)
        else:
            prev = np.where(eligible[t - 1] & np.isfinite(weights[t - 1]), weights[t - 1], 0.0)
            delta = np.abs(w_row - prev)
        trn = float(np.sum(delta))
        if compat_static_cost:
            cost = float(cost_bps_fallback)
        elif execution_cost_bps_2d is not None:
            cost_row = np.where(np.isfinite(execution_cost_bps_2d[t]), execution_cost_bps_2d[t], cost_bps_fallback)
            cost = float(np.sum(delta * cost_row))
        else:
            cost = float(np.sum(delta) * cost_bps_fallback)
        gross_bps.append(gross)
        cost_bps.append(cost)
        net_bps.append(gross - cost)
        turnover.append(trn)
        breadth.append(float(np.count_nonzero(np.abs(w_row) > 0.0)))
        net_exp.append(float(abs(np.sum(w_row))))
        if beta_2d is not None:
            beta_row = np.where(np.isfinite(beta_2d[t]), beta_2d[t], 0.0)
            beta_exp.append(float(abs(np.sum(w_row * beta_row))))
    n_obs = len(net_bps)
    if n_obs < 2:
        return {
            "n_obs": float(n_obs),
            "validation_gross_bps": 0.0,
            "validation_cost_bps": 0.0,
            "validation_net_lcb_bps": -1.0,
            "validation_ir_t": 0.0,
            "validation_breadth": float("nan"),
            "validation_turnover": float("nan"),
            "validation_abs_net_exposure": float("nan"),
            "validation_abs_beta_exposure": float("nan"),
            "validation_monotonicity": float("nan"),
        }
    net_arr = np.asarray(net_bps, dtype=np.float64)
    se = float(np.std(net_arr, ddof=1)) / max(np.sqrt(float(n_obs)), 1e-12)
    return {
        "n_obs": float(n_obs),
        "validation_gross_bps": float(np.mean(gross_bps)),
        "validation_cost_bps": float(np.mean(cost_bps)),
        "validation_net_lcb_bps": float(np.mean(net_arr) - se),
        "validation_ir_t": float(np.mean(net_arr) / max(se, 1e-12)),
        "validation_breadth": float(np.mean(breadth)),
        "validation_turnover": float(np.mean(turnover)),
        "validation_abs_net_exposure": float(np.mean(net_exp)),
        "validation_abs_beta_exposure": float(np.mean(beta_exp)) if beta_exp else 0.0,
        "validation_monotonicity": float(_score_monotonicity(score, realized, eligible)),
    }


def calibrate_rank_portfolio_policy(
    *,
    signed_score_2d: NDArray[np.float64],
    realized_fwd_ret_by_horizon: Mapping[int, NDArray[np.float64]],
    eligible_2d: NDArray[np.bool_],
    execution_cost_bps_2d: NDArray[np.float64] | None,
    beta_2d: NDArray[np.float64] | None,
    quantiles: tuple[float, ...],
    min_abs_z_grid: tuple[float, ...],
    holding_bars_candidates: tuple[int, ...],
    selection_modes: tuple[RankSelectionMode, ...],
    cost_bps_fallback: float,
    min_obs: int = 120,
    weight_k: float = 3.0,
    weighting: Literal["equal", "zscore", "tanh"] = "tanh",
    target_breadth_min: int = 8,
    max_turnover: float = 1.25,
    max_abs_net_exposure: float = 0.05,
    max_abs_beta_exposure: float = 0.20,
) -> RankSelectionPolicy:
    """Select validation-only rank-to-portfolio policy after costs and risk constraints."""
    score = np.asarray(signed_score_2d, dtype=np.float64)
    eligible = np.asarray(eligible_2d, dtype=bool)
    compat_relaxed = (
        float(target_breadth_min) <= 1.0
        and float(max_turnover) >= 1e8
        and float(max_abs_net_exposure) >= 1.0
    )
    best_obj = float("-inf")
    best: RankSelectionPolicy | None = None

    for hold in holding_bars_candidates:
        realized_h = realized_fwd_ret_by_horizon.get(int(hold))
        if realized_h is None:
            continue
        realized = np.asarray(realized_h, dtype=np.float64)
        for mode in selection_modes:
            for polarity in (1, -1):
                for q in quantiles:
                    for min_abs_z in min_abs_z_grid:
                        policy = RankSelectionPolicy(
                            polarity=1 if polarity >= 0 else -1,
                            quantile=float(q),
                            min_abs_z=float(min_abs_z),
                            weighting=weighting,
                            weight_k=float(weight_k),
                            holding_bars=int(hold),
                            validation_net_lcb_bps=-1.0,
                            validation_gross_bps=0.0,
                            validation_ir_t=0.0,
                            validation_monotonicity=0.0,
                            n_obs=0,
                            selection_mode=mode,
                        )
                        weights = build_signed_rank_weights(
                            signed_score_2d=score,
                            eligible_2d=eligible,
                            policy=policy,
                            beta_2d=beta_2d,
                            gross_target=1.0,
                            max_abs_net_exposure=max_abs_net_exposure,
                            max_abs_beta_exposure=max_abs_beta_exposure,
                        )
                        metrics = _estimate_policy_metrics(
                            weights=weights,
                            realized=realized,
                            eligible=eligible,
                            execution_cost_bps_2d=execution_cost_bps_2d,
                            cost_bps_fallback=cost_bps_fallback,
                            beta_2d=beta_2d,
                            score=policy.polarity * _cs_zscore_2d(score),
                            compat_static_cost=compat_relaxed,
                        )
                        n_obs = int(metrics["n_obs"])
                        if n_obs < min_obs:
                            continue
                        if metrics["validation_net_lcb_bps"] <= 0.0:
                            continue
                        if (
                            not compat_relaxed
                            and (
                                not np.isfinite(metrics["validation_monotonicity"])
                                or metrics["validation_monotonicity"] <= 0.0
                            )
                        ):
                            continue
                        if metrics["validation_breadth"] < float(target_breadth_min):
                            continue
                        if metrics["validation_turnover"] > float(max_turnover):
                            continue
                        if metrics["validation_abs_net_exposure"] > float(max_abs_net_exposure):
                            continue
                        if metrics["validation_abs_beta_exposure"] > float(max_abs_beta_exposure):
                            continue
                        objective = (
                            metrics["validation_net_lcb_bps"]
                            + 0.10 * metrics["validation_gross_bps"]
                            + 0.05 * min(metrics["validation_breadth"], float(target_breadth_min))
                            - 0.25 * metrics["validation_cost_bps"]
                        )
                        if objective <= best_obj:
                            continue
                        best_obj = objective
                        best = RankSelectionPolicy(
                            polarity=1 if polarity >= 0 else -1,
                            quantile=float(q),
                            min_abs_z=float(min_abs_z),
                            weighting=weighting,
                            weight_k=float(weight_k),
                            holding_bars=int(hold),
                            validation_net_lcb_bps=float(metrics["validation_net_lcb_bps"]),
                            validation_gross_bps=float(metrics["validation_gross_bps"]),
                            validation_ir_t=float(metrics["validation_ir_t"]),
                            validation_monotonicity=float(metrics["validation_monotonicity"]),
                            n_obs=n_obs,
                            selection_mode=mode,
                            validation_turnover=float(metrics["validation_turnover"]),
                            validation_cost_bps=float(metrics["validation_cost_bps"]),
                            validation_breadth=float(metrics["validation_breadth"]),
                            validation_abs_net_exposure=float(metrics["validation_abs_net_exposure"]),
                            validation_abs_beta_exposure=float(metrics["validation_abs_beta_exposure"]),
                        )
    if best is not None:
        return best
    fallback_mode: RankSelectionMode = "soft_cs" if "soft_cs" in selection_modes else "tail"
    return RankSelectionPolicy(
        polarity=1,
        quantile=float(quantiles[0]) if quantiles else 0.35,
        min_abs_z=float(min_abs_z_grid[0]) if min_abs_z_grid else 0.0,
        weighting=weighting,
        weight_k=float(weight_k),
        holding_bars=int(holding_bars_candidates[0]) if holding_bars_candidates else 12,
        validation_net_lcb_bps=-1.0,
        validation_gross_bps=0.0,
        validation_ir_t=0.0,
        validation_monotonicity=0.0,
        n_obs=0,
        selection_mode=fallback_mode,
    )


def calibrate_rank_selection_policy(
    *,
    signed_score_2d: NDArray[np.float64],
    realized_fwd_ret_2d: NDArray[np.float64],
    eligible_2d: NDArray[np.bool_],
    quantiles: tuple[float, ...],
    min_abs_z_grid: tuple[float, ...],
    holding_bars: int,
    cost_bps: float,
    min_obs: int = 120,
    weight_k: float = 3.0,
    weighting: Literal["equal", "zscore", "tanh"] = "tanh",
) -> RankSelectionPolicy:
    """Compatibility wrapper for legacy callers using one horizon/static cost."""
    return calibrate_rank_portfolio_policy(
        signed_score_2d=signed_score_2d,
        realized_fwd_ret_by_horizon={int(holding_bars): np.asarray(realized_fwd_ret_2d, dtype=np.float64)},
        eligible_2d=eligible_2d,
        execution_cost_bps_2d=None,
        beta_2d=None,
        quantiles=quantiles,
        min_abs_z_grid=min_abs_z_grid,
        holding_bars_candidates=(int(holding_bars),),
        selection_modes=("tail",),
        cost_bps_fallback=float(cost_bps),
        min_obs=min_obs,
        weight_k=weight_k,
        weighting=weighting,
        target_breadth_min=1,
        max_turnover=1e9,
        max_abs_net_exposure=1.0,
        max_abs_beta_exposure=1e9,
    )


def apply_rank_selection_policy(
    *,
    signed_score_2d: NDArray[np.float64],
    eligible_2d: NDArray[np.bool_],
    policy: RankSelectionPolicy,
    beta_2d: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Emit alpha_long/alpha_short rank-weight surfaces from signed weights."""
    weights = build_signed_rank_weights(
        signed_score_2d=signed_score_2d,
        eligible_2d=eligible_2d,
        policy=policy,
        beta_2d=beta_2d,
    )
    alpha_long = np.clip(weights, 0.0, np.inf).astype(np.float32, copy=False)
    alpha_short = np.clip(-weights, 0.0, np.inf).astype(np.float32, copy=False)
    return alpha_long, alpha_short


def policy_to_dict(policy: RankSelectionPolicy) -> dict[str, float | int | str]:
    """Serialize policy to metadata dictionary."""
    return {
        "polarity": int(policy.polarity),
        "quantile": float(policy.quantile),
        "min_abs_z": float(policy.min_abs_z),
        "weighting": str(policy.weighting),
        "weight_k": float(policy.weight_k),
        "holding_bars": int(policy.holding_bars),
        "validation_net_lcb_bps": float(policy.validation_net_lcb_bps),
        "validation_gross_bps": float(policy.validation_gross_bps),
        "validation_ir_t": float(policy.validation_ir_t),
        "validation_monotonicity": float(policy.validation_monotonicity),
        "n_obs": int(policy.n_obs),
        "selection_mode": str(policy.selection_mode),
        "validation_turnover": float(policy.validation_turnover),
        "validation_cost_bps": float(policy.validation_cost_bps),
        "validation_breadth": float(policy.validation_breadth),
        "validation_abs_net_exposure": float(policy.validation_abs_net_exposure),
        "validation_abs_beta_exposure": float(policy.validation_abs_beta_exposure),
    }


def policy_from_dict(payload: Mapping[str, Any]) -> RankSelectionPolicy:
    """Deserialize policy metadata dictionary."""
    weighting_str = str(payload.get("weighting", "tanh"))
    weighting = cast(Literal["equal", "zscore", "tanh"], weighting_str)
    selection_mode_str = str(payload.get("selection_mode", "tail"))
    selection_mode = cast(
        RankSelectionMode,
        selection_mode_str if selection_mode_str in {"tail", "soft_cs"} else "tail",
    )
    return RankSelectionPolicy(
        polarity=1 if int(payload.get("polarity", 1)) >= 0 else -1,
        quantile=float(payload.get("quantile", 0.35)),
        min_abs_z=float(payload.get("min_abs_z", 0.0)),
        weighting=weighting,
        weight_k=float(payload.get("weight_k", 3.0)),
        holding_bars=int(payload.get("holding_bars", 12)),
        validation_net_lcb_bps=float(payload.get("validation_net_lcb_bps", -1.0)),
        validation_gross_bps=float(payload.get("validation_gross_bps", 0.0)),
        validation_ir_t=float(payload.get("validation_ir_t", 0.0)),
        validation_monotonicity=float(payload.get("validation_monotonicity", 0.0)),
        n_obs=int(payload.get("n_obs", 0)),
        selection_mode=selection_mode,
        validation_turnover=float(payload.get("validation_turnover", float("nan"))),
        validation_cost_bps=float(payload.get("validation_cost_bps", float("nan"))),
        validation_breadth=float(payload.get("validation_breadth", float("nan"))),
        validation_abs_net_exposure=float(payload.get("validation_abs_net_exposure", float("nan"))),
        validation_abs_beta_exposure=float(payload.get("validation_abs_beta_exposure", float("nan"))),
    )
