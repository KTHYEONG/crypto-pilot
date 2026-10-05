"""CLI namespace adaptation for the MHS request (no second configuration type)."""

from __future__ import annotations

import argparse
from typing import cast

from src.mhs.contracts import MhsDiagnosticRequest
from src.mhs.params import COMMITTEE_MEMBER_SET_INERT


def request_from_namespace(args: argparse.Namespace) -> MhsDiagnosticRequest:
    """Sole CLI-to-request adapter; pure and idempotent (part1 contract, unchanged semantics).

    Raises:
        ValueError: an explicit inert flag, or any request validation error.
    """
    import dataclasses

    from src.mhs.validation import INERT_DEPENDENT_RULES

    explicit = {
        dest: getattr(args, dest, None) is not None
        for dest in (
            "committee_member_set",
            "committee_tranche_count",
            "committee_target_gross",
            "funding_carry_weight",
            "trend_sleeve_gross",
        )
    }
    committee_capital = not args.no_committee_capital
    committee_regime_adaptive_tranche = (
        committee_capital
        and not args.no_committee_regime_adaptive_tranche
        and not args.committee_tranche_smoothing
    )
    funding_carry_sleeve = committee_capital and not args.no_funding_carry_sleeve
    committee_evidence_weighting = (
        committee_capital and not args.no_committee_evidence_weighting
    )
    committee_kelly_sizing = committee_capital and not args.no_committee_kelly_sizing
    exposure_scale_two_sided = not args.no_exposure_scale_two_sided and args.pnl_vol_target_mode in (
        "exante_target",
        "growth_budget",
        "constant_risk",
    )
    derived = {
        "committee_capital": committee_capital,
        "committee_tranche_smoothing": bool(args.committee_tranche_smoothing),
        "committee_regime_adaptive_tranche": committee_regime_adaptive_tranche,
        "committee_target_gross_active": committee_capital,
        "funding_carry_sleeve": funding_carry_sleeve,
        "trend_sleeve": bool(args.trend_sleeve),
    }

    def _active(field: str) -> bool:
        if field == "committee_member_set":
            return bool(derived["committee_capital"])
        if field == "committee_tranche_count":
            return bool(
                derived["committee_tranche_smoothing"]
                or derived["committee_regime_adaptive_tranche"]
            )
        if field == "committee_target_gross":
            return bool(derived["committee_target_gross_active"])
        if field == "funding_carry_weight":
            return bool(derived["funding_carry_sleeve"])
        return bool(derived["trend_sleeve"])

    for rule in INERT_DEPENDENT_RULES:
        if not _active(rule.field) and explicit[rule.field]:
            raise ValueError(f"{rule.flag} is inert unless {rule.requirement}")

    defaults = {f.name: f.default for f in dataclasses.fields(MhsDiagnosticRequest)}
    if _active("committee_member_set"):
        member_set = (
            cast(str, args.committee_member_set)
            if explicit["committee_member_set"]
            else cast(str, defaults["committee_member_set"])
        )
    else:
        member_set = COMMITTEE_MEMBER_SET_INERT
    if _active("committee_tranche_count"):
        tranche_count = (
            cast(int, args.committee_tranche_count)
            if explicit["committee_tranche_count"]
            else cast(int, defaults["committee_tranche_count"])
        )
    else:
        tranche_count = cast(
            int,
            next(
                rule.canonical
                for rule in INERT_DEPENDENT_RULES
                if rule.field == "committee_tranche_count"
            ),
        )
    if committee_capital:
        if args.no_committee_target_gross:
            target_gross = None
        elif explicit["committee_target_gross"]:
            target_gross = cast(float | None, args.committee_target_gross)
        else:
            target_gross = cast(float | None, defaults["committee_target_gross"])
    else:
        target_gross = None
    if _active("funding_carry_weight"):
        carry_weight = (
            cast(float, args.funding_carry_weight)
            if explicit["funding_carry_weight"]
            else cast(float, defaults["funding_carry_weight"])
        )
    else:
        carry_weight = 0.0
    if _active("trend_sleeve_gross"):
        sleeve_gross = (
            cast(float, args.trend_sleeve_gross)
            if explicit["trend_sleeve_gross"]
            else cast(float, defaults["trend_sleeve_gross"])
        )
    else:
        sleeve_gross = 0.0

    return MhsDiagnosticRequest(
        start=args.start,
        end=args.end,
        execution_timeframe=args.execution_timeframe,
        execution_universe_size=args.execution_universe_size,
        max_rss_bytes=args.max_rss_bytes,
        log_run=not args.no_log_run,
        touch_diagnostic=args.touch_diagnostic,
        ladder_diagnostic=args.ladder_diagnostic,
        peg_chase_diagnostic=args.peg_chase_diagnostic,
        liquidity_cost_model=args.liquidity_cost_model,
        passive_timeout_minutes=args.passive_timeout_minutes,
        discovery_gate=args.discovery_gate,
        trend_sleeve=args.trend_sleeve,
        trend_sleeve_gross=sleeve_gross,
        multi_feature_book=args.multi_feature_book,
        committee_book=args.committee_book,
        committee_kelly_sizing=committee_kelly_sizing,
        committee_growth_diagnostic=args.committee_growth_diagnostic,
        committee_capital=committee_capital,
        committee_member_set=member_set,  # type: ignore[arg-type]
        committee_tranche_smoothing=args.committee_tranche_smoothing,
        committee_regime_adaptive_tranche=committee_regime_adaptive_tranche,
        committee_tranche_count=tranche_count,
        committee_target_gross=target_gross,
        committee_evidence_weighting=committee_evidence_weighting,
        execution_coverage_gate=args.execution_coverage_gate,
        exposure_scale_two_sided=exposure_scale_two_sided,
        exposure_drawdown_brake=args.exposure_drawdown_brake,
        name_drift_trim=args.name_drift_trim,
        ram_guard=not args.no_ram_guard,
        discovery_gate_adjusted_net_t=args.discovery_gate_adjusted_net_t,
        discovery_gate_regime_scaled_net_t=args.discovery_gate_regime_scaled_net_t,
        fold_safe_horizon_selection=args.fold_safe_horizon,
        crash_regime_tilt_alpha=args.crash_regime_tilt_alpha,
        slow_book_mode=args.slow_book_mode,
        fast_book_mode=args.fast_book_mode,
        rebalance_filter=args.rebalance_filter,
        beta_neutralize=args.beta_neutralize,
        ensemble_signal=args.ensemble_signal,
        trend_efficiency_overlay=args.trend_efficiency_overlay,
        pnl_vol_target=not args.no_pnl_vol_target,
        pnl_vol_target_mode=args.pnl_vol_target_mode,
        funding_carry_sleeve=funding_carry_sleeve,
        funding_carry_weight=carry_weight,
        growth_envelope=args.growth_envelope,
        committee_member_attribution=args.committee_member_attribution,
        final_oos_2026h1=args.final_oos_2026h1,
        forward_registration_digest=args.forward_registration_digest,
        data_policy=args.data_policy,
        input_manifest_path=args.input_manifest_path,
        forward_execution_quality_dir=args.forward_execution_quality_dir,
        forward_strategy_digest=args.forward_strategy_digest,
    )
