from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import scipy.stats
from numpy.typing import NDArray


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
    """Select polarity/quantile/floor using validation-only post-cost spread LCB."""
    z = _cs_zscore_2d(np.asarray(signed_score_2d, dtype=np.float64))
    realized = np.asarray(realized_fwd_ret_2d, dtype=np.float64)
    eligible = np.asarray(eligible_2d, dtype=bool)

    best_obj = float("-inf")
    best: RankSelectionPolicy | None = None

    for polarity in (1, -1):
        score = polarity * z
        for q in quantiles:
            for min_abs_z in min_abs_z_grid:
                spreads_bps: list[float] = []
                for t in range(score.shape[0]):
                    lmask, smask = _selection_masks(
                        score[t], eligible[t], q=float(q), min_abs_z=float(min_abs_z)
                    )
                    if not np.any(lmask) or not np.any(smask):
                        continue
                    lret = realized[t, lmask]
                    sret = realized[t, smask]
                    if lret.size == 0 or sret.size == 0:
                        continue
                    spread_bps = (float(np.mean(lret)) - float(np.mean(sret))) * 1e4
                    if np.isfinite(spread_bps):
                        spreads_bps.append(spread_bps)

                n_obs = len(spreads_bps)
                if n_obs < min_obs:
                    continue
                arr = np.asarray(spreads_bps, dtype=np.float64)
                gross_bps = float(np.mean(arr))
                std = float(np.std(arr, ddof=1)) if n_obs > 1 else 0.0
                se_bps = std / max(np.sqrt(float(n_obs)), 1e-12)
                net_lcb_bps = (gross_bps - se_bps) - float(cost_bps)
                ir_t = gross_bps / max(se_bps, 1e-12)
                mono = _score_monotonicity(score, realized, eligible)

                if net_lcb_bps <= 0.0:
                    continue
                if not np.isfinite(mono) or mono <= 0.0:
                    continue

                obj = net_lcb_bps + 0.25 * gross_bps
                if obj > best_obj:
                    best_obj = obj
                    best = RankSelectionPolicy(
                        polarity=polarity,
                        quantile=float(q),
                        min_abs_z=float(min_abs_z),
                        weighting=weighting,
                        weight_k=float(weight_k),
                        holding_bars=int(holding_bars),
                        validation_net_lcb_bps=net_lcb_bps,
                        validation_gross_bps=gross_bps,
                        validation_ir_t=ir_t,
                        validation_monotonicity=float(mono),
                        n_obs=n_obs,
                    )

    if best is not None:
        return best
    return RankSelectionPolicy(
        polarity=1,
        quantile=float(quantiles[0]) if quantiles else 0.35,
        min_abs_z=float(min_abs_z_grid[0]) if min_abs_z_grid else 0.0,
        weighting=weighting,
        weight_k=float(weight_k),
        holding_bars=int(holding_bars),
        validation_net_lcb_bps=-1.0,
        validation_gross_bps=0.0,
        validation_ir_t=0.0,
        validation_monotonicity=0.0,
        n_obs=0,
    )


def apply_rank_selection_policy(
    *,
    signed_score_2d: NDArray[np.float64],
    eligible_2d: NDArray[np.bool_],
    policy: RankSelectionPolicy,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Emit alpha_long/alpha_short from signed score using a fixed policy."""
    z = _cs_zscore_2d(np.asarray(signed_score_2d, dtype=np.float64))
    score = policy.polarity * z
    eligible = np.asarray(eligible_2d, dtype=bool)
    alpha_long = np.zeros_like(score, dtype=np.float32)
    alpha_short = np.zeros_like(score, dtype=np.float32)
    # Conservative fallback: if validation failed, keep no-trade and let gates fail honestly.
    if policy.validation_net_lcb_bps <= 0.0 or policy.n_obs <= 0:
        return alpha_long, alpha_short

    for t in range(score.shape[0]):
        lmask, smask = _selection_masks(
            score[t], eligible[t], q=policy.quantile, min_abs_z=policy.min_abs_z
        )
        if policy.weighting == "equal":
            alpha_long[t, lmask] = 1.0
            alpha_short[t, smask] = 1.0
            continue
        lz = np.abs(score[t, lmask])
        sz = np.abs(score[t, smask])
        if policy.weighting == "zscore":
            alpha_long[t, lmask] = lz.astype(np.float32, copy=False)
            alpha_short[t, smask] = sz.astype(np.float32, copy=False)
        else:
            alpha_long[t, lmask] = np.tanh(policy.weight_k * lz).astype(np.float32, copy=False)
            alpha_short[t, smask] = np.tanh(policy.weight_k * sz).astype(np.float32, copy=False)

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
    }


def policy_from_dict(payload: Mapping[str, Any]) -> RankSelectionPolicy:
    """Deserialize policy metadata dictionary."""
    weighting_str = str(payload.get("weighting", "tanh"))
    weighting = cast(Literal["equal", "zscore", "tanh"], weighting_str)
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
    )
