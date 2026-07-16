"""Pure L1 evidence policy: fold assessment, strategy admission, symbol posterior, pooled gate.

Must not import from application, runner, scripts, or persistence modules.
Only depends on candidate_contracts + numpy/scipy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import scipy.stats as stats
from numpy.typing import NDArray

from src.domain.futures.strategy.tiered_workflow.metrics import (
    moving_block_bootstrap_mean,
    resolve_lcb_quantile,
    resolve_num_blocks,
)

FoldEvidenceState: TypeAlias = Literal[
    "invalid_contract",
    "insufficient_support",
    "data_eligible",
    "economic_positive",
]


class EvidenceContractError(ValueError):
    """Raised for non-finite or chronologically invalid evidence."""


@dataclass(frozen=True, slots=True)
class StrategyAdmission:
    strategy_id: str
    mean_net_bps: float
    lcb_net_bps: float
    p_value: float
    q_value: float
    effective_n: float
    admitted: bool


@dataclass(frozen=True, slots=True)
class SymbolPosterior:
    symbol: str
    strategy_id: str
    effective_n: float
    posterior_mean_net_bps: float
    posterior_lcb_net_bps: float
    posterior_probability_positive: float
    sign_conflict: bool
    eligible: bool


@dataclass(frozen=True, slots=True)
class FoldEvidenceAssessment:
    fold_id: int
    state: FoldEvidenceState
    net_series_bps: tuple[float, ...]
    net_mean_bps: float
    net_lcb_bps: float | None
    matched_event_count: int
    match_wilson_lcb: float
    decision_count: int
    effective_symbol_count: float
    cost_fallback_ratio: float
    funding_coverage_ratio: float
    blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PooledL1Evidence:
    data_eligible_fold_count: int
    economic_positive_fold_count: int
    pooled_net_mean_bps: float | None
    pooled_net_lcb_bps: float | None
    positive_fold_ratio: float
    structural_passed: bool
    economic_passed: bool
    blockers: tuple[str, ...]


def _wilson_lower_bound(successes: int, n: int, confidence: float = 0.90) -> float:
    if n <= 0:
        return 0.0
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    p_hat = float(successes) / float(n)
    denom = 1.0 + z * z / float(n)
    centre = p_hat + z * z / (2.0 * float(n))
    radius = z * np.sqrt(p_hat * (1.0 - p_hat) / float(n) + z * z / (4.0 * float(n) * float(n)))
    return float((centre - radius) / denom)


def compute_strategy_admissions(
    *,
    strategy_ids: NDArray[np.str_],
    net_returns_bps: NDArray[np.float64],
    uniqueness_weights: NDArray[np.float64],
    decision_indices: NDArray[np.int64],
    block_bars: int,
    n_bootstrap: int,
    fdr_alpha: float,
    seed: int,
) -> tuple[StrategyAdmission, ...]:
    """Apply BH FDR at strategy level across unique strategy hypotheses."""
    unique_sids = np.unique(strategy_ids)
    admissions: list[StrategyAdmission] = []
    p_values: list[float] = []

    for sid in unique_sids:
        mask = strategy_ids == sid
        strat_net = net_returns_bps[mask]
        strat_weights = uniqueness_weights[mask]
        strat_decisions = decision_indices[mask]

        if strat_net.size < 2:
            admissions.append(StrategyAdmission(
                strategy_id=str(sid),
                mean_net_bps=0.0,
                lcb_net_bps=0.0,
                p_value=1.0,
                q_value=1.0,
                effective_n=0.0,
                admitted=False,
            ))
            p_values.append(1.0)
            continue

        weight_sum = float(np.sum(strat_weights))
        if weight_sum > 0:
            mean_net = float(np.average(strat_net, weights=strat_weights))
            denom = float(np.sum(np.square(strat_weights)))
            effective_n = (weight_sum * weight_sum) / denom if denom > 0 else 0.0
        else:
            mean_net = 0.0
            effective_n = 0.0

        boot = moving_block_bootstrap_mean(
            strat_net.astype(np.float64, copy=False),
            strat_decisions,
            block_bars=block_bars,
            n_bootstrap=n_bootstrap,
            seed=seed + hash(str(sid)) % 10_000,
        )
        if boot.size < 2:
            lcb_net = mean_net
            p_val = 1.0
        else:
            lcb_net = float(np.quantile(boot, 0.05))
            se = float(np.std(boot, ddof=1))
            t_stat = mean_net / max(se, 1e-12) if se > 0 else 0.0
            p_val = 1.0 - float(stats.norm.cdf(t_stat))

        p_values.append(p_val)
        admissions.append(StrategyAdmission(
            strategy_id=str(sid),
            mean_net_bps=mean_net,
            lcb_net_bps=lcb_net,
            p_value=p_val,
            q_value=1.0,  # placeholder, updated after BH
            effective_n=effective_n,
            admitted=False,
        ))

    # BH FDR adjustment
    p_arr = np.asarray(p_values, dtype=np.float64)
    n_hypotheses = p_arr.size
    if n_hypotheses > 0:
        order = np.argsort(p_arr)
        q_values = np.full(n_hypotheses, 1.0, dtype=np.float64)
        for i in range(n_hypotheses - 1, -1, -1):
            q_values[order[i]] = min(
                p_arr[order[i]] * n_hypotheses / (i + 1),
                q_values[order[ min(i + 1, n_hypotheses - 1) ]],
            )
        for i, adm in enumerate(admissions):
            admissions[i] = StrategyAdmission(
                strategy_id=adm.strategy_id,
                mean_net_bps=adm.mean_net_bps,
                lcb_net_bps=adm.lcb_net_bps,
                p_value=adm.p_value,
                q_value=float(q_values[i]),
                effective_n=adm.effective_n,
                admitted=bool(q_values[i] <= fdr_alpha),
            )

    return tuple(admissions)


def compute_symbol_posteriors(
    *,
    symbol_ids: NDArray[np.str_],
    strategy_ids: NDArray[np.str_],
    net_returns_bps: NDArray[np.float64],
    uniqueness_weights: NDArray[np.float64],
    fold_ids: NDArray[np.int64],
    admissions: tuple[StrategyAdmission, ...],
    min_effective_n: float,
    min_folds: int,
    min_positive_fold_ratio: float,
    shrinkage_prior_n: float,
    seed: int,
) -> tuple[SymbolPosterior, ...]:
    """Compute per-symbol posterior with strategy-level shrinkage."""
    strategy_map = {a.strategy_id: a for a in admissions}
    post_list: list[SymbolPosterior] = []

    unique_pairs = set(zip(symbol_ids, strategy_ids, strict=False))
    for symbol, sid in unique_pairs:
        mask = (symbol_ids == symbol) & (strategy_ids == sid)
        pair_net = net_returns_bps[mask]
        pair_weights = uniqueness_weights[mask]
        pair_folds = fold_ids[mask]

        if pair_net.size < 2:
            continue

        weight_sum = float(np.sum(pair_weights))
        denom = float(np.sum(np.square(pair_weights)))
        effective_n = (weight_sum * weight_sum) / denom if denom > 0 else 0.0

        n_folds = int(np.unique(pair_folds).size)

        fold_means = {}
        for fold_id in np.unique(pair_folds):
            f_mask = pair_folds == fold_id
            f_net = pair_net[f_mask]
            f_w = pair_weights[f_mask]
            f_ws = float(np.sum(f_w))
            if f_ws > 0:
                fold_means[int(fold_id)] = float(np.average(f_net, weights=f_w))
        positive_fold_ratio = (
            sum(1 for v in fold_means.values() if v > 0) / len(fold_means) if fold_means else 0.0
        )

        admission = strategy_map.get(str(sid))
        mu_strategy = admission.mean_net_bps if (admission and admission.admitted) else 0.0

        obs_mean = float(np.average(pair_net, weights=pair_weights)) if weight_sum > 0 else 0.0

        # Shrinkage: w_s = n_eff / (n_eff + tau); posterior = w * obs + (1-w) * mu_strategy
        w_s = effective_n / max(effective_n + shrinkage_prior_n, 1e-12)
        posterior_mean = w_s * obs_mean + (1.0 - w_s) * mu_strategy

        # Bootstrap posterior LCB
        boot = moving_block_bootstrap_mean(
            pair_net.astype(np.float64, copy=False),
            np.arange(pair_net.size, dtype=np.int64),
            block_bars=max(2, int(shrinkage_prior_n / 10)),
            n_bootstrap=200,
            seed=seed + hash(f"{symbol}:{sid}") % 10_000,
        )
        if boot.size >= 2:
            posterior_lcb = float(np.quantile(boot, 0.05))
            posterior_prob_pos = float(np.mean(boot > 0.0))
        else:
            posterior_lcb = posterior_mean
            posterior_prob_pos = 1.0 if posterior_mean > 0 else 0.0

        sign_conflict = bool(
            admission is not None
            and admission.admitted
            and mu_strategy > 0
            and obs_mean <= 0
            and posterior_lcb <= 0
        )

        eligible = bool(
            effective_n >= min_effective_n
            and n_folds >= min_folds
            and (admission is None or admission.admitted)
            and posterior_prob_pos > 0.50
            and posterior_lcb > 0
            and positive_fold_ratio >= min_positive_fold_ratio
            and not sign_conflict
        )

        post_list.append(SymbolPosterior(
            symbol=str(symbol),
            strategy_id=str(sid),
            effective_n=effective_n,
            posterior_mean_net_bps=posterior_mean,
            posterior_lcb_net_bps=posterior_lcb,
            posterior_probability_positive=posterior_prob_pos,
            sign_conflict=sign_conflict,
            eligible=eligible,
        ))

    return tuple(post_list)


def assess_fold_evidence(
    *,
    fold_id: int,
    gross_series_bps: NDArray[np.float64],
    execution_cost_bps: NDArray[np.float64],
    funding_cost_bps: NDArray[np.float64],
    matched_event_count: int,
    unmatched_event_count: int,
    decision_count: int,
    effective_symbol_count: float,
    cost_observed: NDArray[np.bool_],
    funding_observed: NDArray[np.bool_],
    min_matched_events: int,
    min_match_wilson_lcb: float,
    min_decision_count: int,
    max_cost_fallback_ratio: float,
    min_funding_coverage_ratio: float,
    block_bars: int,
    n_bootstrap: int,
    seed: int,
    lcb_quantile_base: float = 0.05,
    lcb_quantile_relaxed: float = 0.20,
    lcb_quantile_full_conf_blocks: int = 15,
    lcb_quantile_floor_blocks: int = 3,
) -> FoldEvidenceAssessment:
    blockers: list[str] = []

    if gross_series_bps.size == 0:
        return FoldEvidenceAssessment(
            fold_id=fold_id,
            state="invalid_contract",
            net_series_bps=(),
            net_mean_bps=0.0,
            net_lcb_bps=None,
            matched_event_count=0,
            match_wilson_lcb=0.0,
            decision_count=0,
            effective_symbol_count=0.0,
            cost_fallback_ratio=0.0,
            funding_coverage_ratio=0.0,
            blockers=("empty_series",),
        )

    if not np.all(np.isfinite(gross_series_bps)):
        raise ValueError("gross_series_bps contains non-finite values")

    # Compute net series
    net_series = gross_series_bps - execution_cost_bps - funding_cost_bps
    if not np.all(np.isfinite(net_series)):
        raise ValueError("net_series contains non-finite values (check execution_cost_bps/funding_cost_bps)")

    # Cost coverage checks
    n = gross_series_bps.size
    cost_fallback_count = int(n - int(np.sum(cost_observed)))
    cost_fallback_ratio = cost_fallback_count / max(n, 1)
    funding_observed_count = int(np.sum(funding_observed))
    funding_coverage_ratio = funding_observed_count / max(n, 1)

    if cost_fallback_ratio > max_cost_fallback_ratio:
        blockers.append("cost_data_incomplete")
    if funding_coverage_ratio < min_funding_coverage_ratio:
        blockers.append("funding_data_incomplete")

    # Match Wilson LCB
    match_wilson_lcb = _wilson_lower_bound(matched_event_count, matched_event_count + unmatched_event_count)

    if matched_event_count < min_matched_events:
        blockers.append(f"matched_events:{matched_event_count}<{min_matched_events}")
    if match_wilson_lcb < min_match_wilson_lcb:
        blockers.append(f"match_wilson_lcb:{match_wilson_lcb:.4f}<{min_match_wilson_lcb}")
    if decision_count < min_decision_count:
        blockers.append(f"decision_count:{decision_count}<{min_decision_count}")

    if blockers:
        state: FoldEvidenceState = "insufficient_support"
    else:
        state = "data_eligible"

    # Net bootstrap LCB
    net_mean_bps = float(np.mean(net_series))
    net_lcb_bps: float | None
    if n >= 2:
        decision_indices = np.arange(n, dtype=np.int64)
        boot = moving_block_bootstrap_mean(
            net_series.astype(np.float64, copy=False),
            decision_indices,
            block_bars=block_bars,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        if boot.size > 0:
            num_blocks = resolve_num_blocks(n, block_bars)
            q = resolve_lcb_quantile(
                num_blocks,
                base_quantile=lcb_quantile_base,
                relaxed_quantile=lcb_quantile_relaxed,
                full_conf_blocks=lcb_quantile_full_conf_blocks,
                floor_blocks=lcb_quantile_floor_blocks,
            )
            net_lcb_bps = float(np.quantile(boot, q))
        else:
            net_lcb_bps = net_mean_bps
        if state == "data_eligible" and net_lcb_bps > 0:
            state = "economic_positive"
    else:
        net_lcb_bps = net_mean_bps if np.isfinite(net_mean_bps) else None

    return FoldEvidenceAssessment(
        fold_id=fold_id,
        state=state,
        net_series_bps=tuple(float(v) for v in net_series),
        net_mean_bps=net_mean_bps,
        net_lcb_bps=net_lcb_bps,
        matched_event_count=matched_event_count,
        match_wilson_lcb=match_wilson_lcb,
        decision_count=decision_count,
        effective_symbol_count=effective_symbol_count,
        cost_fallback_ratio=cost_fallback_ratio,
        funding_coverage_ratio=funding_coverage_ratio,
        blockers=tuple(blockers),
    )


def pool_l1_evidence(
    *,
    folds: tuple[FoldEvidenceAssessment, ...],
    fold_cov: float,
    effective_symbol_n: float,
    min_fold_cov: float,
    min_data_eligible_folds: int,
    min_effective_symbol_n: float,
    min_positive_fold_ratio: float,
    block_bars: int,
    n_bootstrap: int,
    seed: int,
    lcb_quantile_base: float = 0.05,
    lcb_quantile_relaxed: float = 0.20,
    lcb_quantile_full_conf_blocks: int = 15,
    lcb_quantile_floor_blocks: int = 3,
) -> PooledL1Evidence:
    """Pool fold evidence while preserving negative economic observations.

    [ADR_20260715_L0_L1_NET_EVIDENCE_REPLAY]
    """
    blockers: list[str] = []

    data_eligible_folds = [f for f in folds if f.state in ("data_eligible", "economic_positive")]
    economic_positive_folds = [
        f
        for f in folds
        if f.state == "economic_positive"
        or (f.state == "data_eligible" and f.net_lcb_bps is not None and f.net_lcb_bps > 0.0)
    ]

    data_eligible_fold_count = len(data_eligible_folds)
    economic_positive_fold_count = len(economic_positive_folds)
    positive_fold_ratio = economic_positive_fold_count / max(data_eligible_fold_count, 1)

    structural_passed = True
    if fold_cov < min_fold_cov:
        blockers.append(f"fold_cov:{fold_cov:.3f}<{min_fold_cov}")
        structural_passed = False
    if data_eligible_fold_count < min_data_eligible_folds:
        blockers.append(f"data_eligible_folds:{data_eligible_fold_count}<{min_data_eligible_folds}")
        structural_passed = False
    if effective_symbol_n < min_effective_symbol_n:
        blockers.append(f"effective_symbol_n:{effective_symbol_n:.2f}<{min_effective_symbol_n}")
        structural_passed = False

    # Pool all data-eligible fold net series (including negative folds — P1)
    pooled_net: list[float] = []
    for f in data_eligible_folds:
        pooled_net.extend(f.net_series_bps)

    if not pooled_net:
        return PooledL1Evidence(
            data_eligible_fold_count=0,
            economic_positive_fold_count=0,
            pooled_net_mean_bps=None,
            pooled_net_lcb_bps=None,
            positive_fold_ratio=0.0,
            structural_passed=False,
            economic_passed=False,
            blockers=("no_data_eligible_folds",),
        )

    pooled_arr = np.asarray(pooled_net, dtype=np.float64)
    pooled_mean = float(np.mean(pooled_arr))

    if pooled_arr.size >= 2:
        pooled_boot = moving_block_bootstrap_mean(
            pooled_arr,
            np.arange(pooled_arr.size, dtype=np.int64),
            block_bars=block_bars,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        if pooled_boot.size > 0:
            num_blocks = resolve_num_blocks(pooled_arr.size, block_bars)
            q = resolve_lcb_quantile(
                num_blocks,
                base_quantile=lcb_quantile_base,
                relaxed_quantile=lcb_quantile_relaxed,
                full_conf_blocks=lcb_quantile_full_conf_blocks,
                floor_blocks=lcb_quantile_floor_blocks,
            )
            pooled_lcb = float(np.quantile(pooled_boot, q))
        else:
            pooled_lcb = pooled_mean
    else:
        pooled_lcb = pooled_mean

    economic_passed = bool(
        structural_passed
        and pooled_lcb is not None
        and pooled_lcb > 0
        and positive_fold_ratio >= min_positive_fold_ratio
    )

    return PooledL1Evidence(
        data_eligible_fold_count=data_eligible_fold_count,
        economic_positive_fold_count=economic_positive_fold_count,
        pooled_net_mean_bps=pooled_mean,
        pooled_net_lcb_bps=pooled_lcb,
        positive_fold_ratio=positive_fold_ratio,
        structural_passed=structural_passed,
        economic_passed=economic_passed,
        blockers=tuple(blockers),
    )
