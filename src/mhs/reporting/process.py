"""Versioned process proxy reporting evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.mhs.backtest.contracts import (
    PROCESS_CERTIFICATION_LEVEL,
    ProcessBacktestReport,
    ProcessPath,
)

__all__ = [
    "PROCESS_CERTIFICATION_LEVEL",
    "persist_process_report",
    "persist_process_targets",
]


def _tier_payload(path: ProcessPath) -> dict[str, object]:
    """Serialize the exact research path, publication clock and refit-selection audit.
    An hourly proxy may explain decisions but never becomes primary execution proof.
    """
    daily = {ts.isoformat(): float(v) for ts, v in path.daily_returns.items()}
    exposure = {ts.isoformat(): float(v) for ts, v in path.exposure.items()}
    log_growth = (
        float(np.log1p(path.daily_returns.to_numpy(dtype="float64")).mean() * 365.0) if len(path.daily_returns) else 0.0
    )
    exposure_values = path.exposure.to_numpy(dtype="float64")
    zero_share = float((exposure_values == 0.0).mean()) if len(exposure_values) else 0.0
    cap_share = float((exposure_values >= path.leverage_cap).mean()) if len(exposure_values) else 0.0
    if len(path.daily_returns) and len(path.turnover_1h):
        start_label = path.daily_returns.index[0]
        turnover_slice = path.turnover_1h.loc[path.turnover_1h.index >= start_label]
        ann_turnover = float(turnover_slice.sum() * 365.0 / len(path.daily_returns))
    else:
        ann_turnover = 0.0
    mean_unit_gross = float(path.unit_target_weights.abs().sum(axis=1).mean()) if len(path.unit_target_weights) else 0.0
    mean_effective_gross = float(path.target_weights.abs().sum(axis=1).mean()) if len(path.target_weights) else 0.0
    return {
        "one_way_bps": path.one_way_bps,
        "leverage_cap": path.leverage_cap,
        "daily_returns": daily,
        "exposure": exposure,
        "clock_mode": path.clock_mode,
        "signal_available_at": (
            [ts.isoformat() for ts in path.signal_available_at] if path.signal_available_at is not None else None
        ),
        "member_evidence_source": None,
        "refits": [
            {
                "effective_from": r.point.effective_from.isoformat(),
                "effective_to": r.point.effective_to.isoformat(),
                "train_end": r.point.train_end.isoformat(),
                "member_weights": dict(r.member_weights),
                "smoothing_halflife_days": r.smoothing_halflife_days,
            }
            for r in path.refits
        ],
        "refit_audits": [
            {
                "effective_from": r.point.effective_from.isoformat(),
                "effective_to": r.point.effective_to.isoformat(),
                "train_end": r.point.train_end.isoformat(),
                "policy_id": r.policy_id,
                "train_start": None if r.train_start is None else r.train_start.isoformat(),
                "n_train_labels": r.n_train_labels,
            }
            for r in path.refits
        ],
        "policy_choices": [
            {
                "effective_from": c.point.effective_from.isoformat(),
                "effective_to": c.point.effective_to.isoformat(),
                "train_end": c.point.train_end.isoformat(),
                "policy_id": c.policy_id,
                "train_start": None if c.train_start is None else c.train_start.isoformat(),
                "inner_start": None if c.inner_start is None else c.inner_start.isoformat(),
                "inner_end": None if c.inner_end is None else c.inner_end.isoformat(),
                "n_inner_labels": c.n_inner_labels,
                "paired_growth_lcb": c.paired_growth_lcb,
                "reason_codes": list(c.reason_codes),
            }
            for c in path.policy_choices
        ],
        "ann_log_growth": log_growth,
        "exposure_zero_share": zero_share,
        "exposure_cap_share": cap_share,
        "execution_policy": {"tracking_error_threshold": path.execution_policy.tracking_error_threshold},
        "risk_sizing": (
            {
                "annual_volatility_target": path.risk_sizing.annual_volatility_target,
                "ewma_halflife_days": path.risk_sizing.ewma_halflife_days,
                "minimum_observations": path.risk_sizing.minimum_observations,
                "leverage_cap": path.risk_sizing.leverage_cap,
            }
            if path.risk_sizing is not None
            else None
        ),
        "ann_turnover": ann_turnover,
        "mean_unit_gross": mean_unit_gross,
        "mean_effective_gross": mean_effective_gross,
    }


def _resolve_process_report_path(report: ProcessBacktestReport, path: Path | None) -> Path:
    if path is None:
        raise ValueError("process report destination must be an explicit JSON path")
    out = Path(path)
    if out.suffix != ".json":
        raise ValueError(f"destination must be a JSON path, got {path}")
    return out


def persist_process_report(
    report: ProcessBacktestReport,
    path: Path | None = None,
) -> Path:
    """Persist proxy evidence to an explicit destination.

    Args:
        report: The evaluated hourly proxy report.
        path: Explicit JSON artifact path.

    Returns:
        The path of the written JSON evidence.

    Raises:
        ValueError: The destination is missing or not a JSON path.
    """
    out = _resolve_process_report_path(report, path)
    payload = {
        "start": report.start.isoformat(),
        "end": report.end.isoformat(),
        "certification_level": report.certification_level,
        "n_candidates": report.n_candidates,
        "evidence_scope": "retrospective_discovery",
        "multiplicity_adjusted": False,
        "research_diagnostics_only": True,
        "deployment_eligible": False,
        "gate": {
            "go": report.gate.go,
            "reason_codes": list(report.gate.reason_codes),
            "metrics": dict(report.gate.metrics),
        },
        "base": _tier_payload(report.base),
        "stress": _tier_payload(report.stress),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
    return out


def persist_process_targets(path: ProcessPath, output: Path) -> Path:
    """Export exact sized targets without inventing signal availability.

    Args:
        path: Evaluated process path with its sized decision rows.
        output: Explicit parquet destination.

    Returns:
        The written parquet path, retaining float64 values and UTC labels.

    Raises:
        ValueError: The destination is not a parquet path.
    """
    out = Path(output)
    if out.suffix != ".parquet":
        raise ValueError(f"destination must be a parquet path, got {output}")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = path.target_weights.copy().astype("float64")
    frame.to_parquet(out)
    return out
