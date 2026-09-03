from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.application.research.mhs.contracts import MhsFoldReport
from src.application.research.mhs.research_go import GO_REASON_FOLD_GROWTH_CONCENTRATION, GO_REASON_PATH_DIVERGENCE
from src.mhs.calibration import sharpe_lower_confidence_bound
from src.mhs.params import EVIDENCE_GATE_ALPHA, FOLD_BLEND_PARITY_TOLERANCE, FOLD_REALIZED_RISK_PARITY_TOLERANCE


def _pooled_fold_evidence(
    folds: tuple[MhsFoldReport, ...],
    alpha: float = EVIDENCE_GATE_ALPHA,
) -> dict[str, Any]:
    """Pool measurable folds' strict/stress daily ledger returns into level evidence.

    Level-family gates judge the pooled estimates' lower bounds against
    registered absolute economic floors; ``min over folds`` is deliberately
    absent (per-fold minima are noise-dominated). A fold that is invalid or
    missing either replay is recorded under ``unmeasured`` and excluded from
    pooling. Gate adjudication lives in research_go; this function only
    measures. The raw ``pooled_strict_returns``/``pooled_stress_returns``
    arrays are internal-only inputs for null calibration and never persisted.
    """
    strict_parts: list[np.ndarray] = []
    stress_parts: list[np.ndarray] = []
    unmeasured: list[int] = []
    measured_count = 0
    for fold in folds:
        if not fold.primary_valid or fold.strict is None or fold.stress is None:
            unmeasured.append(fold.fold_index)
            continue
        strict_daily = fold.strict.ledger.equity.resample("1D").last().pct_change().dropna()
        stress_daily = fold.stress.ledger.equity.resample("1D").last().pct_change().dropna()
        strict_parts.append(strict_daily.to_numpy(dtype="float64"))
        stress_parts.append(stress_daily.to_numpy(dtype="float64"))
        measured_count += 1
    pooled_strict = (
        np.concatenate(strict_parts) if strict_parts else np.empty(0, dtype="float64")
    )
    pooled_stress = (
        np.concatenate(stress_parts) if stress_parts else np.empty(0, dtype="float64")
    )
    n_pooled_days = int(pooled_strict.size)
    total_log = (
        float(np.log1p(np.clip(pooled_strict, -0.999, None)).sum())
        if n_pooled_days
        else float("-inf")
    )
    if not math.isfinite(total_log):
        total_log = float("-inf")
    annual_log_return = (
        total_log * 365.0 / n_pooled_days if n_pooled_days else float("-inf")
    )
    return {
        "unmeasured": unmeasured,
        "n_measured_folds": measured_count,
        "n_pooled_days": n_pooled_days,
        "pooled_sharpe_lcb": sharpe_lower_confidence_bound(pooled_strict, alpha),
        "pooled_stress_sharpe_lcb": sharpe_lower_confidence_bound(pooled_stress, alpha),
        "pooled_annual_log_return": annual_log_return,
        "pooled_strict_returns": pooled_strict,
        "pooled_stress_returns": pooled_stress,
    }


def _log_growth(daily_returns: np.ndarray) -> float:
    # 일간 수익률의 누적 로그성장(총 로그성장). -1 이하 클립은 방어적 하한.
    return float(np.log1p(np.clip(daily_returns, -0.999, None)).sum())


def _fold_growth_concentration(
    folds: tuple[MhsFoldReport, ...],
    threshold: float,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Check that no single fold dominates total realized log-growth.

    Returns ``(payload, reason_codes)``.  ``threshold`` must be supplied by the
    caller from the registered-alpha null calibration -- no literal default
    lives here. Folds whose ``primary_valid`` is False, whose CAGR is
    non-finite, or whose CAGR is ``<= -1.0`` (total wipeout) are recorded under
    ``payload['unmeasured']`` and excluded from the denominator — mirroring
    ``_fold_blend_parity``'s degenerate-evidence fail-open pattern.
    """
    payload: dict[str, Any] = {
        "folds": {},
        "unmeasured": [],
        "max_fold_share": 0.0,
        "max_share": threshold,
        "threshold": threshold,
    }
    logrets: list[tuple[int, float]] = []
    for fold in folds:
        cagr = fold.primary_geometric_cagr
        if not fold.primary_valid or not math.isfinite(cagr) or cagr <= -1.0:
            payload["unmeasured"].append(fold.fold_index)
            continue
        logret = math.log1p(cagr)
        logrets.append((fold.fold_index, logret))
    if len(logrets) < 2 or sum(lr for _, lr in logrets) <= 0.0:
        return payload, ()
    total = sum(lr for _, lr in logrets)
    max_share_val = 0.0
    for fold_index, logret in logrets:
        share = logret / total
        payload["folds"][fold_index] = {"logret": logret, "share": share}
        max_share_val = max(max_share_val, share)
    payload["max_fold_share"] = max_share_val
    reason_codes = (
        (GO_REASON_FOLD_GROWTH_CONCENTRATION,)
        if max_share_val > threshold
        else ()
    )
    return payload, reason_codes


def _fold_realized_risk_parity(
    folds: tuple[MhsFoldReport, ...],
    tolerance: float = FOLD_REALIZED_RISK_PARITY_TOLERANCE,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Observe realized-risk parity across folds (observation-only diagnostic).

    Returns ``(payload, reason_codes)`` with reason_codes ALWAYS empty -- this
    never alters the Research-GO decision. The reference is the median
    ``realized_annualized_vol`` of measurable folds; each fold contributes its
    ``log_ratio = ln(vol / median)`` and the payload reports
    ``max_abs_log_risk_ratio`` against ``tolerance``. Folds whose vol is
    None/non-finite/<=0 or whose ``primary_valid`` is False are recorded under
    ``payload['unmeasured']`` and excluded from the denominator (the same
    degenerate-evidence fail-open pattern as ``_fold_growth_concentration``).
    """
    payload: dict[str, Any] = {
        "folds": {},
        "unmeasured": [],
        "max_abs_log_risk_ratio": 0.0,
        "tolerance": tolerance,
    }
    measured: list[tuple[int, float]] = []
    for fold in folds:
        vol = fold.realized_annualized_vol
        if not fold.primary_valid or vol is None or not math.isfinite(vol) or vol <= 0.0:
            payload["unmeasured"].append(fold.fold_index)
            continue
        measured.append((fold.fold_index, vol))
    if len(measured) < 2:
        return payload, ()
    median_vol = float(np.median([vol for _, vol in measured]))
    max_abs_log_ratio = 0.0
    for fold_index, vol in measured:
        log_ratio = math.log(vol / median_vol)
        payload["folds"][fold_index] = {
            "realized_annualized_vol": vol,
            "log_ratio": log_ratio,
        }
        max_abs_log_ratio = max(max_abs_log_ratio, abs(log_ratio))
    payload["max_abs_log_risk_ratio"] = max_abs_log_ratio
    return payload, ()


def _fold_blend_parity(
    blend_traces: dict[int, dict[str, float]],
    folds: tuple[MhsFoldReport, ...],
    tolerance: float = FOLD_BLEND_PARITY_TOLERANCE,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Compare each fold's book structure against the blend path's trace.

    Returns ``(payload, reason_codes)``. A fold whose ``book_structure`` is
    None, or whose fold/blend denominator is non-positive, is recorded under
    ``payload['unmeasured']`` and never emits the divergence code itself.
    """
    payload: dict[str, Any] = {
        "folds": {},
        "unmeasured": [],
        "max_abs_log_holdings_ratio": 0.0,
        "max_abs_log_gross_ratio": 0.0,
        "max_abs_log_deployed_gross_ratio": 0.0,
        "tolerance": tolerance,
    }
    max_abs_holdings = 0.0
    max_abs_gross = 0.0
    max_abs_deployed = 0.0
    for fold in folds:
        fold_trace = fold.book_structure
        blend_trace = blend_traces.get(fold.fold_index)
        if fold_trace is None or blend_trace is None:
            payload["unmeasured"].append(fold.fold_index)
            payload["folds"][fold.fold_index] = {
                "holdings_log_ratio": None,
                "gross_log_ratio": None,
                "deployed_gross_log_ratio": None,
                "fold": fold_trace,
                "blend": blend_trace,
            }
            continue
        f_holdings = fold_trace.get("holdings_mean", 0.0)
        b_holdings = blend_trace.get("holdings_mean", 0.0)
        f_gross = fold_trace.get("gross_mean", 0.0)
        b_gross = blend_trace.get("gross_mean", 0.0)
        if f_holdings <= 0.0 or b_holdings <= 0.0 or f_gross <= 0.0 or b_gross <= 0.0:
            payload["unmeasured"].append(fold.fold_index)
            payload["folds"][fold.fold_index] = {
                "holdings_log_ratio": None,
                "gross_log_ratio": None,
                "deployed_gross_log_ratio": None,
                "fold": fold_trace,
                "blend": blend_trace,
            }
            continue
        holdings_log_ratio = float(np.log(f_holdings / b_holdings))
        gross_log_ratio = float(np.log(f_gross / b_gross))
        # Deployed (post-exposure-scale) gross is what actually ships; a trace
        # without exposure_scale_mean stays unmeasured for this ratio -- never
        # silently treated as scale 1.0.
        f_scale = fold_trace.get("exposure_scale_mean")
        b_scale = blend_trace.get("exposure_scale_mean")
        deployed_gross_log_ratio: float | None
        if not isinstance(f_scale, float) or f_scale <= 0.0 or not isinstance(b_scale, float) or b_scale <= 0.0:
            deployed_gross_log_ratio = None
            payload["unmeasured"].append(fold.fold_index)
        else:
            deployed_gross_log_ratio = float(np.log((f_gross * f_scale) / (b_gross * b_scale)))
            max_abs_deployed = max(max_abs_deployed, abs(deployed_gross_log_ratio))
        payload["folds"][fold.fold_index] = {
            "holdings_log_ratio": holdings_log_ratio,
            "gross_log_ratio": gross_log_ratio,
            "deployed_gross_log_ratio": deployed_gross_log_ratio,
            "fold": fold_trace,
            "blend": blend_trace,
        }
        max_abs_holdings = max(max_abs_holdings, abs(holdings_log_ratio))
        max_abs_gross = max(max_abs_gross, abs(gross_log_ratio))
    payload["max_abs_log_holdings_ratio"] = max_abs_holdings
    payload["max_abs_log_gross_ratio"] = max_abs_gross
    payload["max_abs_log_deployed_gross_ratio"] = max_abs_deployed
    reason_codes = (
        (GO_REASON_PATH_DIVERGENCE,)
        if (
            max_abs_holdings > tolerance or max_abs_gross > tolerance
            or max_abs_deployed > tolerance
        )
        else ()
    )
    return payload, reason_codes
















