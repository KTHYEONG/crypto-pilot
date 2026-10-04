"""MHS run configuration: single source of truth for all defaults.

``MhsRunConfig`` replaces ``MhsDiagnosticRequest`` (FIX D1). The CLI
handler's 25 lines of derived-default logic are absorbed into the
dataclass so that a no-argument CLI invocation and ``MhsRunConfig()``
produce identical ``dataclasses.asdict()`` output.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from src.mhs.data_policy import MHS_DATA_POLICY_DEFAULT
from src.mhs.params import (
    CLI_EXECUTION_UNIVERSE_SIZE_DEFAULT,
    CLI_GROWTH_ENVELOPE_DEFAULT,
    COMMITTEE_MEMBER_SET_INERT,
    COMMITTEE_TARGET_GROSS,
    COMMITTEE_TRANCHE_COUNT,
)


class MemberSet(StrEnum):
    """Registered committee member sets (I_NOVERSION: no _v<N> suffix)."""

    RISK_PREMIA = "risk_premia"
    FLOW_MOMENTUM = "flow_momentum"


@dataclass(frozen=True, slots=True)
class MhsRunConfig:
    """Carry one source-aligned MHS replay configuration across CLI, diagnostic and pipeline entry points. Three-minute OHLCV and funding are the only historical economic feeds."""

    # Time bounds
    start: str | None = None
    end: str | None = None
    partition: Literal["dev", "holdout", "all"] = "dev"
    data_root: str | None = None
    execution_timeframe: Literal["3m"] = "3m"
    execution_universe_size: int = CLI_EXECUTION_UNIVERSE_SIZE_DEFAULT  # was 30 (2026-08-23) per ADR_20260823_MHS_KELLY_TWO_SIDED_SIZING
    max_rss_bytes: int | None = None
    log_run: bool = True

    # Diagnostic opt-ins
    touch_diagnostic: bool = False
    ladder_diagnostic: bool = False
    peg_chase_diagnostic: bool = False
    liquidity_cost_model: Literal["flat", "corwin_schultz"] = "flat"
    # Execution window per intent (S6): exposed so window sweeps need no code edit.
    passive_timeout_minutes: int = 30
    discovery_gate: bool = False
    discovery_gate_adjusted_net_t: bool = False
    discovery_gate_regime_scaled_net_t: bool = False
    fold_safe_horizon_selection: bool = False
    crash_regime_tilt_alpha: float | None = None
    slow_book_mode: Literal["single_horizon", "horizon_ensemble"] = "single_horizon"
    fast_book_mode: Literal["single_horizon", "horizon_ensemble"] = "single_horizon"
    rebalance_filter: Literal["per_symbol_deadband", "portfolio_trigger"] = "per_symbol_deadband"
    beta_neutralize: bool = False
    ensemble_signal: Literal["raw", "vol_normalized"] = "raw"
    trend_efficiency_overlay: bool = False
    pnl_vol_target: bool = True
    pnl_vol_target_mode: Literal["median_relative", "exante_target", "growth_budget", "constant_risk"] = "growth_budget"  # was "median_relative" in MhsDiagnosticRequest -- CLI's real effective default (D1); growth_budget since 2026-08-22
    trend_sleeve: bool = False
    trend_sleeve_gross: float = 0.0
    multi_feature_book: bool = False

    # Committee (FIX D1: defaults absorb CLI derived logic)
    committee_book: bool = False
    committee_kelly_sizing: bool = True  # was False + CLI override True (2026-08-23, ADR_20260823_MHS_KELLY_TWO_SIDED_SIZING treatment B)
    committee_growth_diagnostic: bool = False
    committee_capital: bool = True  # was False + CLI override True
    committee_member_set: MemberSet = MemberSet.FLOW_MOMENTUM  # was "risk_premia_v2" vs params "flow_momentum_v1"
    committee_tranche_smoothing: bool = False
    committee_regime_adaptive_tranche: bool = True  # was False + CLI override
    committee_tranche_count: int = COMMITTEE_TRANCHE_COUNT
    committee_target_gross: float | None = COMMITTEE_TARGET_GROSS  # was _UNSET sentinel
    committee_evidence_weighting: bool = True  # was False + CLI override True (2026-08-22)

    # Funding
    funding_carry_sleeve: bool = True  # was False + CLI override
    funding_carry_weight: float = 0.3  # default when sleeve is on

    # Gates
    execution_coverage_gate: bool = False
    exposure_scale_two_sided: bool = True  # was False; CLI effective default flips like committee_capital/growth_envelope/pnl_vol_target_mode
    exposure_drawdown_brake: bool = False
    name_drift_trim: bool = False
    ram_guard: bool = True

    # Growth envelope & member attribution
    growth_envelope: str = CLI_GROWTH_ENVELOPE_DEFAULT  # was "conservative" (2026-08-22)
    committee_member_attribution: bool = False
    # One-time, narrowly-scoped extension of the sealed evaluation window for a
    # user-authorized final-OOS check (2026-08-25 decision) -- see MHS_FINAL_OOS_CUTOFF_2026H1.
    final_oos_2026h1: bool = False
    forward_registration_digest: str | None = None
    data_policy: Literal['legacy', 'zombie_mask_v1'] = MHS_DATA_POLICY_DEFAULT
    input_manifest_path: str | None = None
    forward_execution_quality_dir: str | None = None
    forward_strategy_digest: str | None = None

    def __post_init__(self) -> None:
        """Validate with the single request validator so no unvalidated configuration can reach the pipeline (I-SINGLE-VALIDATION)."""
        from src.mhs.params import COMMITTEE_TARGET_GROSS_UNSET
        from src.mhs.validation import validate_request

        validate_request(self, COMMITTEE_TARGET_GROSS_UNSET)

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> MhsRunConfig:
        """Sole CLI-to-config adapter; pure (never mutates ``args``) and idempotent.

        Derives the opt-out cascades (committee capital off disables its dependent
        features; fixed tranche smoothing disables the regime-adaptive tranche; a
        non two-sided-capable vol-target mode disables two-sided scaling) and
        resolves every value-carrying dependent field: an active field takes the
        explicit flag value or this class's default; an inactive field takes its
        canonical inert value. An explicitly passed flag whose field is inactive is
        rejected instead of being silently discarded (I-NO-INERT-FLAGS).

        Raises:
            ValueError: ``"<flag> is inert unless <requirement>"`` for the first
                inactive explicit flag in ``INERT_DEPENDENT_RULES`` order; or any
                ``validate_request`` error of the resulting configuration.
        """
        import dataclasses
        from typing import cast

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

        defaults = {f.name: f.default for f in dataclasses.fields(cls)}
        if _active("committee_member_set"):
            member_set = (
                MemberSet(cast(str, args.committee_member_set))
                if explicit["committee_member_set"]
                else cast(MemberSet, defaults["committee_member_set"])
            )
        else:
            member_set = MemberSet(COMMITTEE_MEMBER_SET_INERT)
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

        return cls(
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
            committee_member_set=member_set,
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
