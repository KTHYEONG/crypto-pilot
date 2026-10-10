"""Local strategy evaluation standard: one verdict from historical data only.

Judges the same book that trades. Objective is net log growth; Sharpe enters
only through DSR (multiple-testing control). Never reads paper or live data.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.core.bootstrap import (
    iter_stationary_bootstrap_index_chunks,
    stationary_bootstrap_max_blocks,
)
from src.core.params import REPORT_BOOTSTRAP_PATHS, REPORT_BOOTSTRAP_SEED
from src.evaluation.statistics import deflated_sharpe_ratio, log_growth_lcb, sharpe_sampling_variance
from src.evaluation.trials import TrialPopulation
from src.quant.evaluation.reliability import derive_block_size
from src.strategy.release import EvaluationCriteria, criteria_digest


class Verdict(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class CheckResult:
    code: str
    group: str
    passed: bool | None
    value: float | None
    threshold: str
    reason: str


@dataclass(frozen=True, slots=True)
class StrategyEvaluation:
    strategy_id: str
    spec_digest: str
    criteria_digest: str
    verdict: Verdict
    discovery: tuple[pd.Timestamp, pd.Timestamp]
    holdout: tuple[pd.Timestamp, pd.Timestamp] | None
    n_trials: int
    checks: tuple[CheckResult, ...]

    @property
    def digest(self) -> str:
        payload = {
            "strategy_id": self.strategy_id,
            "spec_digest": self.spec_digest,
            "criteria_digest": self.criteria_digest,
            "verdict": str(self.verdict),
            "discovery": [str(self.discovery[0]), str(self.discovery[1])],
            "holdout": None if self.holdout is None else [str(self.holdout[0]), str(self.holdout[1])],
            "n_trials": self.n_trials,
            "checks": [[c.code, c.group, c.passed, c.value, c.threshold] for c in self.checks],
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EvaluationInputs:
    strategy_id: str
    spec_digest: str
    discovery: tuple[pd.Timestamp, pd.Timestamp]
    holdout: tuple[pd.Timestamp, pd.Timestamp] | None
    design_data_cutoff: pd.Timestamp
    base_returns: pd.Series
    stress_returns: pd.Series
    funding_by_symbol: Mapping[str, float]
    participation: pd.Series
    ledger_certified: bool
    lake_coverage_ok: bool
    causality_ok: bool
    deployed_returns: pd.Series
    deployed_stress_returns: pd.Series
    deployed_liquidated: bool
    deployed_stress_liquidated: bool
    deployed_margin_breaches: int
    deployed_max_leverage: float
    neighbors: tuple[pd.Series, ...]
    trial_population: TrialPopulation
    holdout_returns: pd.Series | None
    funding_income_daily: pd.DataFrame | None = None
    book_identity_ok: bool = True
    withdrawn_seat_fraction: float = 0.0
    participation_scale_to_deployed: float = 1.0


def _require_window(window: tuple[pd.Timestamp, pd.Timestamp], label: str) -> None:
    if not isinstance(window, tuple) or len(window) != 2:
        raise DataIntegrityError(f"{label} must be a (start, end) tuple")
    for tag, ts in (("start", window[0]), ("end", window[1])):
        if not isinstance(ts, pd.Timestamp) or pd.isna(ts):
            raise DataIntegrityError(f"{label} {tag} must be a valid timestamp")
        if ts.tzinfo is None or ts.utcoffset() is None or ts.utcoffset().total_seconds() != 0:
            raise DataIntegrityError(f"{label} {tag} must be timezone-aware UTC")
    if window[1] <= window[0]:
        raise DataIntegrityError(f"{label} end must be after start")


def _require_returns(series: pd.Series, label: str) -> np.ndarray:
    if not isinstance(series, pd.Series) or series.empty:
        raise DataIntegrityError(f"{label} must be a non-empty Series")
    values = np.asarray(series.to_numpy(dtype="float64"), dtype="float64")
    if not bool(np.isfinite(values).all()):
        raise DataIntegrityError(f"{label} must be finite (NaN/inf fail closed)")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise DataIntegrityError(f"{label} must carry a DatetimeIndex")
    index = series.index
    if index.tz is None or str(index.tz) != "UTC" or index.hasnans or not index.is_unique:
        raise DataIntegrityError(f"{label} must have a unique UTC index")
    if not index.is_monotonic_increasing:
        raise DataIntegrityError(f"{label} must be monotonic in time")
    if bool((values < -1.0).any()):
        raise DataIntegrityError(f"{label} cannot fall below -100%")
    return values


def _validate_evidence_metadata(inputs: EvaluationInputs) -> None:
    population = inputs.trial_population
    if not math.isfinite(population.sharpe_variance) or population.sharpe_variance < 0:
        raise DataIntegrityError("trial variance must be finite and non-negative")
    if any(type(flag) is not bool for flag in (inputs.ledger_certified, inputs.lake_coverage_ok, inputs.causality_ok, inputs.book_identity_ok)):
        raise DataIntegrityError("integrity flags must be booleans")
    if inputs.funding_income_daily is not None:
        funding = inputs.funding_income_daily
        if not funding.index.equals(inputs.stress_returns.index) or not np.isfinite(funding.to_numpy(dtype=float)).all():
            raise DataIntegrityError("daily funding must be finite and align with stress returns")
    fraction = inputs.withdrawn_seat_fraction
    if isinstance(fraction, bool) or not math.isfinite(float(fraction)) or float(fraction) < 0.0:
        raise DataIntegrityError("withdrawn_seat_fraction must be finite and >= 0")
    scale = inputs.participation_scale_to_deployed
    if isinstance(scale, bool) or not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise DataIntegrityError("participation_scale_to_deployed must be finite and > 0")


def _validate_inputs(inputs: EvaluationInputs) -> None:
    if not isinstance(inputs.strategy_id, str) or not inputs.strategy_id:
        raise DataIntegrityError("strategy_id must be a non-empty string")
    if not isinstance(inputs.spec_digest, str) or not inputs.spec_digest:
        raise DataIntegrityError("spec_digest must be a non-empty string")
    _validate_evidence_metadata(inputs)
    _require_window(inputs.discovery, "discovery")
    if inputs.holdout is not None:
        _require_window(inputs.holdout, "holdout")
    cutoff = inputs.design_data_cutoff
    if not isinstance(cutoff, pd.Timestamp) or pd.isna(cutoff) or cutoff.tzinfo is None:
        raise DataIntegrityError("design_data_cutoff must be a valid tz-aware timestamp")
    base = _require_returns(inputs.base_returns, "base_returns")
    stress = _require_returns(inputs.stress_returns, "stress_returns")
    if len(base) != len(stress) or not inputs.base_returns.index.equals(inputs.stress_returns.index):
        raise DataIntegrityError("base_returns and stress_returns must share one index")
    _require_returns(inputs.deployed_returns, "deployed_returns")
    _validate_survival(inputs)
    for position, neighbor in enumerate(inputs.neighbors):
        _require_returns(neighbor, f"neighbors[{position}]")
    if inputs.holdout_returns is not None:
        _require_returns(inputs.holdout_returns, "holdout_returns")
    _validate_participation(inputs)
    _validate_funding_and_leverage(inputs)


def _validate_survival(inputs: EvaluationInputs) -> None:
    _require_returns(inputs.deployed_stress_returns, "deployed_stress_returns")
    if not inputs.deployed_returns.index.equals(inputs.deployed_stress_returns.index):
        raise DataIntegrityError("deployed_returns and deployed_stress_returns must share one index")
    if type(inputs.deployed_liquidated) is not bool or type(inputs.deployed_stress_liquidated) is not bool:
        raise DataIntegrityError("liquidation flags must be booleans")
    breaches = inputs.deployed_margin_breaches
    if isinstance(breaches, bool) or not isinstance(breaches, int) or breaches < 0:
        raise DataIntegrityError("deployed_margin_breaches must be a non-negative int")


def _validate_participation(inputs: EvaluationInputs) -> None:
    participation = np.asarray(inputs.participation.to_numpy(dtype="float64"), dtype="float64")
    if participation.size and not inputs.participation.index.equals(inputs.base_returns.index):
        raise DataIntegrityError("participation must align with base_returns")
    if not bool(np.isfinite(participation).all()):
        raise DataIntegrityError("participation must be finite and align with base_returns")
    if bool((participation < 0.0).any()):
        raise DataIntegrityError("participation cannot be negative")


def _validate_funding_and_leverage(inputs: EvaluationInputs) -> None:
    if not isinstance(inputs.funding_by_symbol, Mapping):
        raise DataIntegrityError("funding_by_symbol must be a mapping")
    for symbol, amount in inputs.funding_by_symbol.items():
        if not isinstance(symbol, str) or not symbol:
            raise DataIntegrityError("funding_by_symbol keys must be non-empty strings")
        if isinstance(amount, bool) or not math.isfinite(float(amount)):
            raise DataIntegrityError("funding_by_symbol values must be finite")
    leverage = inputs.deployed_max_leverage
    if isinstance(leverage, bool) or not math.isfinite(float(leverage)) or float(leverage) < 0.0:
        raise DataIntegrityError("deployed_max_leverage must be finite and >= 0")


def _daily_moments(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    std = float(np.std(values, ddof=1)) if values.size >= 2 else float("nan")
    if not np.isfinite(std) or std <= 1e-12:
        return float("nan"), float("nan"), float("nan")
    observed = float(mean / std)
    skew = float(pd.Series(values).skew())
    kurtosis = float(pd.Series(values).kurt() + 3.0)
    return observed, skew, kurtosis


def _log_growth(values: np.ndarray) -> float:
    if bool((values <= -1.0).any()):
        return float("-inf")
    return float(np.log1p(values).sum())


def _integrity_checks(inputs: EvaluationInputs, criteria: EvaluationCriteria) -> list[CheckResult]:
    cutoff = pd.Timestamp(inputs.design_data_cutoff).tz_convert("UTC")
    discovery_end = pd.Timestamp(inputs.discovery[1]).tz_convert("UTC")
    results = [
        CheckResult(
            "I1_LEDGER_CERTIFIED",
            "integrity",
            inputs.ledger_certified is True and inputs.book_identity_ok is True,
            None,
            "replay_ledger_certified",
            "Accounting must be exact before performance means anything.",
        ),
        CheckResult(
            "I2_LAKE_COVERAGE",
            "integrity",
            bool(inputs.lake_coverage_ok and inputs.withdrawn_seat_fraction <= criteria.max_withdrawn_seat_fraction),
            float(inputs.withdrawn_seat_fraction),
            f"<= {criteria.max_withdrawn_seat_fraction}",
            "Source holes that cannot be repaired may withdraw at most the registered fraction of roster seats; each withdrawal is disclosed.",
        ),
        CheckResult(
            "I3_CAUSALITY",
            "integrity",
            bool(inputs.causality_ok),
            None,
            "future-perturbation invariance passes",
            "Zero look-ahead.",
        ),
    ]
    design_ok: bool | None = discovery_end <= cutoff and inputs.base_returns.index[-1] < cutoff
    if inputs.holdout is not None:
        holdout_start = pd.Timestamp(inputs.holdout[0]).tz_convert("UTC")
        design_ok = bool(design_ok) and holdout_start >= cutoff
        if inputs.holdout_returns is not None:
            design_ok = bool(design_ok) and inputs.holdout_returns.index[0] >= cutoff
    results.append(
        CheckResult(
            "I4_DESIGN_CUTOFF",
            "integrity",
            design_ok,
            None,
            "discovery end <= cutoff <= holdout start",
            "Out-of-sample means unseen during design.",
        )
    )
    return results


def _edge_checks(inputs: EvaluationInputs, criteria: EvaluationCriteria) -> list[CheckResult]:
    base = inputs.base_returns.to_numpy(dtype="float64")
    observed, skew, kurtosis = _daily_moments(base)
    population = inputs.trial_population
    if population.n_trials < 1:
        raise DataIntegrityError("trial population must count at least one trial")
    if np.isfinite(observed) and np.isfinite(skew) and np.isfinite(kurtosis):
        dsr = deflated_sharpe_ratio(
            observed,
            max(population.sharpe_variance, sharpe_sampling_variance(observed, int(base.size), skew, kurtosis)),
            population.n_trials,
            int(base.size),
            skew,
            kurtosis,
        )
    else:
        dsr = float("nan")
    dsr_value = float(dsr) if math.isfinite(dsr) else None
    dsr_passed = bool(math.isfinite(dsr) and dsr >= criteria.dsr_min)
    try:
        lcb = log_growth_lcb(
            inputs.stress_returns,
            alpha=criteria.growth_lcb_alpha,
            n_paths=REPORT_BOOTSTRAP_PATHS,
            seed=REPORT_BOOTSTRAP_SEED,
        )
    except DataIntegrityError:
        lcb = float("nan")
    lcb_value = float(lcb) if math.isfinite(lcb) else None
    lcb_passed = bool(math.isfinite(lcb) and lcb > 0.0)
    return [
        CheckResult(
            "E1_DSR",
            "edge",
            dsr_passed,
            dsr_value,
            f">= {criteria.dsr_min}",
            "The best of many tries looks good by luck; DSR removes that.",
        ),
        CheckResult(
            "E2_STRESS_GROWTH_LCB",
            "edge",
            lcb_passed,
            lcb_value,
            "> 0",
            "Growth must survive pessimistic costs with sampling uncertainty.",
        ),
    ]


def _leave_one_year_out_margin(stress: pd.Series) -> tuple[bool | None, float | None]:
    years = sorted({ts.year for ts in stress.index})
    if len(years) < 2:
        return False, None
    log_returns = np.log1p(stress.to_numpy(dtype="float64"))
    positions = {year: np.flatnonzero(stress.index.year == year) for year in years}
    margins = [float(log_returns[np.setdiff1d(np.arange(len(log_returns)), positions[year])].sum()) for year in years]
    worst = min(margins)
    return all(margin > 0.0 for margin in margins), worst


def _ex_funding_growth(inputs: EvaluationInputs, stress_log_growth: float) -> tuple[bool | None, float | None]:
    funding = inputs.funding_income_daily
    if funding is None:
        return None, None
    top = funding.sum().sort_values(ascending=False)
    symbols = top[top > 0].index[:5]
    removed = funding.loc[:, symbols].clip(lower=0).sum(axis=1)
    adjusted = inputs.stress_returns.to_numpy(dtype=float) - removed.to_numpy(dtype=float)
    margin = _log_growth(adjusted)
    return margin > 0.0, margin


def _plateau_result(
    inputs: EvaluationInputs, criteria: EvaluationCriteria, candidate: float
) -> tuple[bool, float | None]:
    if not inputs.neighbors:
        return False, None
    for neighbor in inputs.neighbors:
        growth = _log_growth(neighbor.to_numpy(dtype="float64"))
        if not math.isfinite(growth) or growth <= 0.0:
            return False, growth
    required = criteria.plateau_min_fraction * candidate
    weakest = min(_log_growth(n.to_numpy(dtype="float64")) for n in inputs.neighbors)
    ok = weakest > 0.0 and (weakest >= required if math.isfinite(candidate) and candidate > 0 else False)
    return ok, weakest if math.isfinite(weakest) else None


def _robustness_checks(inputs: EvaluationInputs, criteria: EvaluationCriteria) -> list[CheckResult]:
    stress = inputs.stress_returns.to_numpy(dtype="float64")
    candidate = _log_growth(stress)
    r1_passed, r1_value = _leave_one_year_out_margin(inputs.stress_returns)
    r2_passed, r2_value = _ex_funding_growth(inputs, candidate)
    if not criteria.max_top5_funding_dependence:
        r2_passed, r2_value = True, candidate
    r3_passed, r3_value = _plateau_result(inputs, criteria, candidate)
    participation = inputs.participation.to_numpy(dtype="float64")
    if participation.size:
        p95 = float(np.quantile(participation, 0.95)) * float(inputs.participation_scale_to_deployed)
        r4_passed: bool | None = bool(p95 <= criteria.max_participation_p95)
        r4_reason = "Orders must be a small share of causal ADV at the capital that will trade."
    else:
        p95 = None
        r4_passed = False
        r4_reason = "participation not recorded"
    return [
        CheckResult(
            "R1_LEAVE_ONE_YEAR_OUT",
            "robustness",
            r1_passed,
            r1_value,
            "> 0",
            "No single year carries the result; replaces 'every quarter profitable'.",
        ),
        CheckResult(
            "R2_FUNDING_DEPENDENCE",
            "robustness",
            r2_passed,
            r2_value,
            "> 0",
            "Not reliant on a few funding squeezes (2026H1: LABUSDT).",
        ),
        CheckResult(
            "R3_PLATEAU",
            "robustness",
            r3_passed,
            r3_value,
            f"> 0 and >= {criteria.plateau_min_fraction} x candidate",
            "Knife-edge parameters do not generalize.",
        ),
        CheckResult(
            "R4_CAPACITY",
            "robustness",
            r4_passed,
            p95,
            f"<= {criteria.max_participation_p95}",
            r4_reason,
        ),
    ]


def _bootstrap_paths(values: np.ndarray, path_len: int, n_paths: int, seed: int) -> np.ndarray:
    n = int(values.size)
    mean_block = max(1, int(derive_block_size(values)))
    max_blocks = max(path_len, stationary_bootstrap_max_blocks(path_len, mean_block))
    rng = np.random.default_rng(int(seed))
    out = np.empty((int(n_paths), int(path_len)), dtype="float64")
    filled = 0
    chunk_size = min(int(n_paths), 500)
    for chunk in iter_stationary_bootstrap_index_chunks(
        rng,
        source_len=n,
        path_len=int(path_len),
        n_replicates=int(n_paths),
        mean_block=mean_block,
        chunk_size=chunk_size,
        max_blocks=max_blocks,
    ):
        rows = chunk.indices.shape[0]
        out[filled : filled + rows] = values[chunk.indices]
        filled += rows
    return out


def _annualized_log_growth(paths: np.ndarray, horizon_years: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result: np.ndarray = np.log1p(paths).sum(axis=1) / float(horizon_years)
    return result


def _growth_paths(inputs: EvaluationInputs, criteria: EvaluationCriteria) -> tuple[np.ndarray, float]:
    horizon = float(criteria.growth_horizon_years)
    path_len = max(1, round(horizon * 365.0))
    values = inputs.deployed_stress_returns.to_numpy(dtype="float64")
    paths = _bootstrap_paths(values, path_len, REPORT_BOOTSTRAP_PATHS, REPORT_BOOTSTRAP_SEED)
    return paths, horizon


def _growth_checks(inputs: EvaluationInputs, criteria: EvaluationCriteria) -> list[CheckResult]:
    paths, horizon = _growth_paths(inputs, criteria)
    full = _annualized_log_growth(paths, horizon)
    lcb_full = float(np.quantile(full, float(criteria.growth_lcb_alpha), method="inverted_cdf"))
    scaled = _annualized_log_growth(paths * float(criteria.overbet_probe_scale), horizon)
    lcb_scaled = float(np.quantile(scaled, float(criteria.growth_lcb_alpha), method="inverted_cdf"))
    gap = float(lcb_full - lcb_scaled)
    return [
        CheckResult("G1_GROWTH_LCB", "growth", bool(lcb_full > 0.0), lcb_full, "> 0", "Compound growth must clear zero at the lower quantile."),
        CheckResult("G2_NOT_OVERBET", "growth", bool(gap >= 0.0), gap, ">= 0", "A smaller size must not raise the growth floor."),
    ]


def _survival_check(inputs: EvaluationInputs) -> CheckResult:
    passed = bool(
        inputs.deployed_liquidated is False
        and inputs.deployed_stress_liquidated is False
        and inputs.deployed_margin_breaches == 0
    )
    return CheckResult("S1_SURVIVAL", "survival", passed, float(inputs.deployed_margin_breaches), "== 0", "Liquidation is the only irreversible event.")


def _holdout_checks(inputs: EvaluationInputs, criteria: EvaluationCriteria) -> list[CheckResult]:
    if inputs.holdout_returns is None or inputs.holdout is None:
        skipped = "No holdout observed yet; a short holdout cannot prove edge."
        return [
            CheckResult(
                "H1_HOLDOUT_GROWTH",
                "holdout",
                None,
                None,
                f">= {criteria.holdout_growth_min_quantile} quantile",
                skipped,
            ),
        ]
    holdout = inputs.holdout_returns.to_numpy(dtype="float64")
    stress = inputs.stress_returns.to_numpy(dtype="float64")
    holdout_growth = _log_growth(holdout)
    paths = _bootstrap_paths(stress, len(holdout), REPORT_BOOTSTRAP_PATHS, REPORT_BOOTSTRAP_SEED + 1)
    with np.errstate(divide="ignore"):
        growth_dist = np.log1p(paths).sum(axis=1)
    growth_floor = float(np.quantile(growth_dist, criteria.holdout_growth_min_quantile, method="inverted_cdf"))
    h1 = bool(holdout_growth >= growth_floor)
    return [
        CheckResult(
            "H1_HOLDOUT_GROWTH",
            "holdout",
            h1,
            holdout_growth,
            f">= {growth_floor:.6f}",
            "A short holdout cannot prove edge; it can show the strategy did not break after design.",
        ),
    ]


def _not_evaluated(check: CheckResult, reason: str) -> CheckResult:
    return CheckResult(check.code, check.group, None, None, check.threshold, reason)


def evaluate_strategy(inputs: EvaluationInputs, criteria: EvaluationCriteria) -> StrategyEvaluation:
    """Pure verdict over fixed inputs; deterministic for fixed seeds."""
    _validate_inputs(inputs)
    integrity = _integrity_checks(inputs, criteria)
    if any(check.passed is not True for check in integrity):
        skipped = [
            _not_evaluated(c, "not evaluated: integrity invalid")
            for c in (
                CheckResult("E1_DSR", "edge", None, None, f">= {criteria.dsr_min}", ""),
                CheckResult("E2_STRESS_GROWTH_LCB", "edge", None, None, "> 0", ""),
                CheckResult("R1_LEAVE_ONE_YEAR_OUT", "robustness", None, None, "> 0", ""),
                CheckResult("R2_FUNDING_DEPENDENCE", "robustness", None, None, "> 0", ""),
                CheckResult("R3_PLATEAU", "robustness", None, None, "plateau", ""),
                CheckResult("R4_CAPACITY", "robustness", None, None, f"<= {criteria.max_participation_p95}", ""),
                CheckResult("G1_GROWTH_LCB", "growth", None, None, "> 0", ""),
                CheckResult("G2_NOT_OVERBET", "growth", None, None, ">= 0", ""),
                CheckResult("S1_SURVIVAL", "survival", None, None, "== 0", ""),
                CheckResult("H1_HOLDOUT_GROWTH", "holdout", None, None, "holdout", ""),
            )
        ]
        return StrategyEvaluation(
            strategy_id=inputs.strategy_id,
            spec_digest=inputs.spec_digest,
            criteria_digest=criteria_digest(criteria),
            verdict=Verdict.INVALID,
            discovery=inputs.discovery,
            holdout=inputs.holdout,
            n_trials=inputs.trial_population.n_trials,
            checks=tuple(integrity + skipped),
        )
    edge = _edge_checks(inputs, criteria)
    robustness = _robustness_checks(inputs, criteria)
    growth = _growth_checks(inputs, criteria)
    survival = [_survival_check(inputs)]
    holdout_checks = _holdout_checks(inputs, criteria)
    checks = integrity + edge + robustness + growth + survival + holdout_checks
    discovery_days = int(inputs.base_returns.size)
    holdout_days = 0 if inputs.holdout_returns is None else int(inputs.holdout_returns.size)
    thin_data = (
        discovery_days < criteria.min_discovery_days
        or inputs.holdout_returns is None
        or inputs.holdout is None
        or holdout_days < criteria.min_holdout_days
        or any(check.passed is None for check in checks)
    )
    if thin_data:
        verdict = Verdict.INCONCLUSIVE
    elif any(check.passed is False for check in edge + robustness + growth + survival + holdout_checks):
        verdict = Verdict.REJECT
    else:
        verdict = Verdict.ACCEPT
    return StrategyEvaluation(
        strategy_id=inputs.strategy_id,
        spec_digest=inputs.spec_digest,
        criteria_digest=criteria_digest(criteria),
        verdict=verdict,
        discovery=inputs.discovery,
        holdout=inputs.holdout,
        n_trials=inputs.trial_population.n_trials,
        checks=tuple(checks),
    )
