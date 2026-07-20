from __future__ import annotations

import logging
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)


def _resolve_bars_per_year(obj: Any) -> float | None:
    mtf = getattr(obj, "master_tf", None)
    if isinstance(mtf, str) and mtf:
        from src.domain.futures.strategy.tiered_workflow.metrics import _bars_per_year_for_tf

        return _bars_per_year_for_tf(mtf)
    return None


def _cagr(rets: list[float], bars_per_year: float) -> float:
    """Compound annual growth rate from a return series."""
    arr = np.asarray(rets, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    valid = arr[np.isfinite(arr)]
    if valid.size < 2:
        return 0.0
    total_log = float(np.sum(np.log1p(valid)))
    n_years = len(valid) / max(bars_per_year, 1.0)
    return float(np.expm1(total_log / max(n_years, 1e-12)))


def _recompute_deployed_cagr(obj: Any) -> float | None:
    """Recompute CAGR from deployed_returns_hybrid if present, else fallback."""
    deployed = getattr(obj, "deployed_returns_hybrid", None)
    if deployed and len(deployed) >= 2:
        bars_per_year = _resolve_bars_per_year(obj)
        if bars_per_year is not None:
            return _cagr(list(deployed), bars_per_year)
    rets = getattr(obj, "returns_hybrid", None)
    if rets is None or len(rets) < 2:
        return None
    bars_per_year = _resolve_bars_per_year(obj)
    if bars_per_year is None:
        return None
    l_star = getattr(obj, "deploy_leverage", None)
    if l_star is not None and l_star > 0.0:
        from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
            apply_deployment,
        )
        try:
            _arr = np.asarray(rets, dtype=np.float64)
            return apply_deployment(rets=_arr, leverage=float(l_star), bars_per_year=bars_per_year).cagr
        except Exception:
            return None
    return _cagr(list(rets), bars_per_year)


def assert_selection_replay_parity(
    *,
    replay_evaluation: Any,
    final_evaluation: Any,
    tolerance: float = 1e-8,
    gate: bool = False,
) -> bool:
    """replay/final parity diagnostic. Returns True if within tolerance.

    No longer raises ValueError — logs warning on mismatch and returns False.
    Caller decides whether to gate on parity.

    When gate=True, mismatch causes the caller to block the champion
    via ``blocker_reason="parity_divergence"``.
    """
    metric_names = (
        ("cagr_hybrid", "cagr"),
        ("mdd_hybrid", "mdd"),
        ("fold_pass_ratio", "fold_pass"),
        ("trade_count", "trade_count"),
    )
    mismatches: list[str] = []
    details: list[str] = []
    for attr, label in metric_names:
        replay_value = getattr(replay_evaluation, attr, None)
        final_value = getattr(final_evaluation, attr, None)
        if replay_value is None or final_value is None:
            details.append(f"{label}: missing on {'replay' if replay_value is None else 'final'}")
            continue
        replay_f = float(replay_value)
        final_f = float(final_value)
        delta = abs(replay_f - final_f)
        details.append(f"{label} replay={replay_f:.8f} final={final_f:.8f} delta={delta:.8f}")
        if delta > float(tolerance):
            mismatches.append(f"{label} replay={replay_f:.8f} final={final_f:.8f}")
    for attr, label in (
        ("deploy_leverage", "L*"),
        ("sharpe_hac_hybrid", "sharpe_hac"),
        ("sortino_hybrid", "sortino"),
        ("constraint_values", "constraints"),
    ):
        replay_v = getattr(replay_evaluation, attr, None)
        final_v = getattr(final_evaluation, attr, None)
        if replay_v is not None and final_v is not None:
            details.append(f"{label} replay={replay_v!r} final={final_v!r}")

    for side, obj in (("replay", replay_evaluation), ("final", final_evaluation)):
        _recomputed = _recompute_deployed_cagr(obj)
        _stored = getattr(obj, "cagr_hybrid", None)
        if _stored is not None and _recomputed is not None and abs(float(_stored) - float(_recomputed)) > 1e-6:
            mismatches.append(f"{side}_deployed_cagr stored={_stored:.6f} != recomputed={_recomputed:.6f}")
            details.append(f"{side}: deployed CAGR DECOUPLED")
            _logger.warning(
                "[L2-PARITY-SELFCHECK] side=%s stored=%.6f recomputed=%.6f "
                "-> field/metric DECOUPLED",
                side,
                float(_stored),
                float(_recomputed),
            )

    if mismatches:
        _logger.warning(
            "[L2-PARITY-DIAG] replay/final parity mismatch (tolerance=%s): %s | %s",
            tolerance,
            "; ".join(mismatches),
            " | ".join(details),
        )
        return False
    return True
