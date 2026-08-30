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

from src.mhs.params import COMMITTEE_TARGET_GROSS

# Main-logic default as of 2026-08-23: selected per ADR_20260823_MHS_KELLY_TWO_SIDED_SIZING
# (pre-registered acceptance, treatment B). The budgeted twin rung keeps the identical
# leverage_ceiling=3.0 -- hence the identical resolved exposure cap and
# deployed exposure -- while its max_drawdown=0.60 sits at the registered
# budget ceiling, making the drawdown risk contract binding instead of
# permanently DRAWDOWN_BUDGET_NON_BINDING. Deliberately decoupled from
# ``src.mhs.params.GROWTH_ENVELOPE_DEFAULT`` ("conservative"), which stays the
# frozen default for ``MhsDiagnosticRequest`` and the golden fixture matrix --
# neither is touched by this change.
CLI_GROWTH_ENVELOPE_DEFAULT = "growth_extreme_budgeted"

# Single owner of the CLI effective breadth default; the contract object
# (MhsDiagnosticRequest) keeps its frozen 30 for bit-exact fixtures.
CLI_EXECUTION_UNIVERSE_SIZE_DEFAULT = 60


class MemberSet(StrEnum):
    """Registered committee member sets (I_NOVERSION: no _v<N> suffix)."""

    RISK_PREMIA = "risk_premia"
    FLOW_MOMENTUM = "flow_momentum"


@dataclass(frozen=True, slots=True)
class MhsRunConfig:
    """Immutable run config; sole owner of effective default values (I-CONFIG).

    CLI is parsing + override only; it never redefines a default.
    ``MhsRunConfig()`` and a no-arg CLI invocation yield the same dict.
    """

    # Time bounds
    start: str | None = None
    end: str | None = None
    partition: Literal["dev", "holdout", "all"] = "dev"
    data_root: str | None = None
    mark_mode: Literal["cache_required", "cache_required_stale_carry", "ohlcv_close_fallback"] = "cache_required"
    execution_timeframe: Literal["1m", "3m", "5m"] = "3m"
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
    committee_target_gross: float | None = COMMITTEE_TARGET_GROSS  # was _UNSET sentinel
    committee_evidence_weighting: bool = True  # was False + CLI override True (2026-08-22)

    # Funding
    funding_carry_sleeve: bool = True  # was False + CLI override
    funding_carry_weight: float = 0.3  # default when sleeve is on

    # Gates
    execution_coverage_gate: bool = False
    fill_mark_parity_gate: bool = True
    exposure_scale_two_sided: bool = True  # was False; CLI effective default flips like committee_capital/growth_envelope/pnl_vol_target_mode
    exposure_drawdown_brake: bool = False
    ram_guard: bool = True

    # Growth envelope & member attribution
    growth_envelope: str = CLI_GROWTH_ENVELOPE_DEFAULT  # was "conservative" (2026-08-22)
    committee_member_attribution: bool = False
    # One-time, narrowly-scoped extension of the sealed evaluation window for a
    # user-authorized final-OOS check (2026-08-25 decision) -- see MHS_FINAL_OOS_CUTOFF_2026H1.
    final_oos_2026h1: bool = False

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> MhsRunConfig:
        """Sole CLI-to-config adapter (FIX D1).

        Mirrors the derivation previously duplicated in the CLI handler
        (``src/cli/commands/research/mhs.py``) field-for-field, including the
        ``--no-*`` negate-flag pattern and the ``fold_safe_horizon`` ->
        ``fold_safe_horizon_selection`` rename. A no-argument CLI invocation
        and ``MhsRunConfig()`` must produce an identical ``dataclasses.asdict``.
        """
        committee_capital = not args.no_committee_capital
        committee_regime_adaptive_tranche = (
            committee_capital
            and not args.no_committee_regime_adaptive_tranche
            and not args.committee_tranche_smoothing
        )
        committee_target_gross = (
            None
            if args.no_committee_target_gross or not committee_capital
            else (
                COMMITTEE_TARGET_GROSS
                if args.committee_target_gross is None
                else args.committee_target_gross
            )
        )
        funding_carry_sleeve = committee_capital and not args.no_funding_carry_sleeve
        committee_evidence_weighting = (
            committee_capital and not args.no_committee_evidence_weighting
        )
        committee_kelly_sizing = committee_capital and not args.no_committee_kelly_sizing
        # two-sided scaling is request-invalid outside exante_target/growth_budget;
        # an explicit median_relative override must opt the default back out.
        if args.pnl_vol_target_mode not in ("exante_target", "growth_budget", "constant_risk"):
            args.no_exposure_scale_two_sided = True

        return cls(
            start=args.start,
            end=args.end,
            mark_mode=args.mark_mode,
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
            trend_sleeve_gross=args.trend_sleeve_gross,
            multi_feature_book=args.multi_feature_book,
            committee_book=args.committee_book,
            committee_kelly_sizing=committee_kelly_sizing,
            committee_growth_diagnostic=args.committee_growth_diagnostic,
            committee_capital=committee_capital,
            committee_member_set=MemberSet(args.committee_member_set),
            committee_tranche_smoothing=args.committee_tranche_smoothing,
            committee_regime_adaptive_tranche=committee_regime_adaptive_tranche,
            committee_target_gross=committee_target_gross,
            committee_evidence_weighting=committee_evidence_weighting,
            execution_coverage_gate=args.execution_coverage_gate,
            fill_mark_parity_gate=not args.no_fill_mark_parity_gate,
            exposure_scale_two_sided=not args.no_exposure_scale_two_sided,
            exposure_drawdown_brake=args.exposure_drawdown_brake,
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
            funding_carry_weight=(args.funding_carry_weight if funding_carry_sleeve else 0.0),
            growth_envelope=args.growth_envelope,
            committee_member_attribution=args.committee_member_attribution,
            final_oos_2026h1=args.final_oos_2026h1,
        )
