"""CLI explicit-value adaptation for the MHS request (no second configuration type)."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, cast

from src.mhs.contracts import MhsDiagnosticRequest
from src.mhs.params import COMMITTEE_MEMBER_SET_INERT


def resolve_cli_request(explicit: Mapping[str, Any]) -> MhsDiagnosticRequest:
    """Build the request from explicitly stated CLI values (I-PURE-ADAPTER, I-NO-INERT-FLAGS).

    Unstated fields take the request defaults, except where an opt-out cascades:
    committee capital off disables its dependent features; fixed tranche
    smoothing disables the regime-adaptive tranche; gross opt-out disables
    the funding-carry sleeve; a non two-sided-capable vol-target mode disables
    two-sided scaling. Inactive value-carrying
    dependents take their canonical inert value; an explicitly stated inactive
    dependent is rejected rather than silently discarded.

    Raises:
        ValueError: ``"<flag> is inert unless <requirement>"`` (first rule in
            ``INERT_DEPENDENT_RULES`` order), or any request validation error.
    """
    from src.mhs.validation import INERT_DEPENDENT_RULES

    defaults = {f.name: f.default for f in dataclasses.fields(MhsDiagnosticRequest)}
    unknown = set(explicit) - defaults.keys()
    if unknown:
        raise ValueError(f"unknown CLI request fields: {sorted(unknown)}")
    for name, default in defaults.items():
        if name in explicit and isinstance(default, bool) and not isinstance(explicit[name], bool):
            raise ValueError(f"{name} must be a bool")

    def _get(name: str) -> Any:
        return explicit[name] if name in explicit else defaults[name]

    committee_capital = bool(_get("committee_capital"))
    smoothing = bool(_get("committee_tranche_smoothing"))
    regime_base = bool(_get("committee_regime_adaptive_tranche"))
    committee_regime_adaptive_tranche = committee_capital and regime_base and not smoothing
    funding_carry_sleeve = (
        committee_capital
        and bool(_get("funding_carry_sleeve"))
        and _get("committee_target_gross") is not None
    )
    committee_evidence_weighting = committee_capital and bool(_get("committee_evidence_weighting"))
    committee_kelly_sizing = committee_capital and bool(_get("committee_kelly_sizing"))
    pnl_vol_target_mode = _get("pnl_vol_target_mode")
    two_sided_base = bool(_get("exposure_scale_two_sided"))
    exposure_scale_two_sided = two_sided_base and pnl_vol_target_mode in (
        "exante_target",
        "growth_budget",
        "constant_risk",
    )
    trend_sleeve = bool(_get("trend_sleeve"))

    # CLI activity of each INERT_DEPENDENT_RULES field; a rule missing here fails closed rather
    # than inheriting another rule's gate.
    activity = {
        "committee_member_set": committee_capital,
        "committee_tranche_count": smoothing or committee_regime_adaptive_tranche,
        "committee_target_gross": committee_capital,
        "funding_carry_weight": funding_carry_sleeve,
        "trend_sleeve_gross": trend_sleeve,
    }

    def _active(field: str) -> bool:
        if field not in activity:
            raise RuntimeError(f"no CLI activity condition for inert-dependent field {field!r}")
        return bool(activity[field])

    def _stated(field: str) -> bool:
        return field in explicit and not (
            field == "committee_target_gross" and explicit[field] is None
        )

    for rule in INERT_DEPENDENT_RULES:
        if not _active(rule.field) and _stated(rule.field):
            raise ValueError(f"{rule.flag} is inert unless {rule.requirement}")

    if _active("committee_member_set"):
        member_set = cast(str, _get("committee_member_set"))
    else:
        member_set = COMMITTEE_MEMBER_SET_INERT
    if _active("committee_tranche_count"):
        tranche_count = cast(int, _get("committee_tranche_count"))
    else:
        tranche_count = cast(
            int,
            next(
                rule.canonical
                for rule in INERT_DEPENDENT_RULES
                if rule.field == "committee_tranche_count"
            ),
        )
    target_gross = cast(float | None, _get("committee_target_gross")) if committee_capital else None
    carry_weight = cast(float, _get("funding_carry_weight")) if _active("funding_carry_weight") else 0.0
    sleeve_gross = cast(float, _get("trend_sleeve_gross")) if _active("trend_sleeve_gross") else 0.0

    return MhsDiagnosticRequest(
        start=_get("start"),
        end=_get("end"),
        partition=_get("partition"),
        data_root=_get("data_root"),
        execution_timeframe=_get("execution_timeframe"),
        execution_universe_size=_get("execution_universe_size"),
        max_rss_bytes=_get("max_rss_bytes"),
        log_run=bool(_get("log_run")),
        touch_diagnostic=bool(_get("touch_diagnostic")),
        ladder_diagnostic=bool(_get("ladder_diagnostic")),
        peg_chase_diagnostic=bool(_get("peg_chase_diagnostic")),
        liquidity_cost_model=_get("liquidity_cost_model"),
        passive_timeout_minutes=_get("passive_timeout_minutes"),
        discovery_gate=bool(_get("discovery_gate")),
        trend_sleeve=trend_sleeve,
        trend_sleeve_gross=sleeve_gross,
        multi_feature_book=bool(_get("multi_feature_book")),
        committee_book=bool(_get("committee_book")),
        committee_kelly_sizing=committee_kelly_sizing,
        committee_growth_diagnostic=bool(_get("committee_growth_diagnostic")),
        committee_capital=committee_capital,
        committee_member_set=member_set,  # type: ignore[arg-type]
        committee_tranche_smoothing=smoothing,
        committee_regime_adaptive_tranche=committee_regime_adaptive_tranche,
        committee_tranche_count=tranche_count,
        committee_target_gross=target_gross,
        committee_evidence_weighting=committee_evidence_weighting,
        execution_coverage_gate=bool(_get("execution_coverage_gate")),
        exposure_scale_two_sided=exposure_scale_two_sided,
        exposure_drawdown_brake=bool(_get("exposure_drawdown_brake")),
        name_drift_trim=bool(_get("name_drift_trim")),
        ram_guard=bool(_get("ram_guard")),
        discovery_gate_adjusted_net_t=bool(_get("discovery_gate_adjusted_net_t")),
        discovery_gate_regime_scaled_net_t=bool(_get("discovery_gate_regime_scaled_net_t")),
        fold_safe_horizon_selection=bool(_get("fold_safe_horizon_selection")),
        crash_regime_tilt_alpha=_get("crash_regime_tilt_alpha"),
        slow_book_mode=_get("slow_book_mode"),
        fast_book_mode=_get("fast_book_mode"),
        rebalance_filter=_get("rebalance_filter"),
        beta_neutralize=bool(_get("beta_neutralize")),
        ensemble_signal=_get("ensemble_signal"),
        trend_efficiency_overlay=bool(_get("trend_efficiency_overlay")),
        pnl_vol_target=bool(_get("pnl_vol_target")),
        pnl_vol_target_mode=pnl_vol_target_mode,
        funding_carry_sleeve=funding_carry_sleeve,
        funding_carry_weight=carry_weight,
        growth_envelope=_get("growth_envelope"),
        committee_member_attribution=bool(_get("committee_member_attribution")),
        final_oos_2026h1=bool(_get("final_oos_2026h1")),
        forward_registration_digest=_get("forward_registration_digest"),
        data_policy=_get("data_policy"),
        input_manifest_path=_get("input_manifest_path"),
        forward_execution_quality_dir=_get("forward_execution_quality_dir"),
        forward_strategy_digest=_get("forward_strategy_digest"),
        placebo_diagnostic=bool(_get("placebo_diagnostic")),
        phase_diagnostic=bool(_get("phase_diagnostic")),
        signal_48h_diagnostic=bool(_get("signal_48h_diagnostic")),
        bootstrap_ci_diagnostic=bool(_get("bootstrap_ci_diagnostic")),
        reference_books_diagnostic=bool(_get("reference_books_diagnostic")),
        patient_reference_diagnostic=bool(_get("patient_reference_diagnostic")),
    )
