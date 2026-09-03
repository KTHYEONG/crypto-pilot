"""S4: Committee evidence weighting + execution book + diagnostics.

Extracted verbatim from ``evaluation.py`` lines 3753-3985 (committee evidence
weighting, ``committee._committee_execution_book`` construction or the fast/slow blend
fallback, trend sleeve overlay, regime cash scale, phase diagnostics, the
48h cross-sectional statistics, the discovery-gate qualification block, and
effective-breadth diagnostics).

All five ``del`` / ``gc.collect`` sites from the source block -- the
``del close, quote_vol, taker_buy_quote`` (3788), ``del vol_mean`` (3824),
``del w_fast_1h, w_slow_1h`` + ``gc.collect()`` (3828-3829), ``del
current_book_for_diagnostic`` (3850), and ``del log_close`` + ``gc.collect()``
(3984-3985) -- are preserved at their exact relative points.
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd

from src.mhs import research_go as _research_go
from src.mhs import scaling as _scaling
from src.mhs import statistics as _statistics
from src.mhs.evaluation import (
    BOOK_BLEND_WEIGHTS,
    CAUSAL_BETA_LOOKBACK_BARS,
    CAUSAL_BETA_MIN_PERIODS,
    COMMITTEE_OOS_START,
    COMMITTEE_REGIME_ADAPTIVE_WINDOW,
    COMMITTEE_TRANCHE_COUNT,
    DISCOVERY_END,
    DISCOVERY_GATE_TRANCHE_COUNT,
    DISCOVERY_MOMENTUM_CANDIDATES,
    DISCOVERY_REVERSAL_CANDIDATES,
    DISCOVERY_START,
    FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS,
    FUNDING_CARRY_SLEEVE_LOOKBACK_HOURS,
    MEASURED_EXECUTION_COST_TIERS_BPS,
    QUALIFICATION_END,
    books,
    causal_market_beta,
    committee,
    diagnostics,
    effective_breadth,
    efficiency_ratio,
    folds,
    funding_carry_execution_book,
    horizon_log_return,
    mhs_ledger_pnl,
    phase_1_anchored_purged_folds,
    realized_vol,
    select_horizon_by_discovery_qualification,
    year_restricted_correlation,
    yearly_net_t_diagnostic,
)
from src.mhs.params import PERIODS_PER_YEAR_1H as _PERIODS_PER_YEAR_1H
from src.mhs.pipeline.context import PipelineContext
from src.mhs.telemetry import StageTelemetry


def build_committee(ctx: PipelineContext, telemetry: StageTelemetry) -> None:
    """Construct the committee execution book and all committee-tier diagnostics."""
    from src.mhs.research_go import _resolved_growth_envelope

    ctx._committee_weights_by_boundary = {}
    ctx._fold_committee_weights = None

    # Build growth envelope payload for the report
    envelope = _resolved_growth_envelope(ctx.config)
    ctx._growth_envelope_payload = {
        "name": envelope.name,
        "max_drawdown": envelope.max_drawdown,
        "max_drawdown_prob": envelope.max_drawdown_prob,
        "ruin_fraction": envelope.ruin_fraction,
        "max_ruin_prob": envelope.max_ruin_prob,
        "horizon_years": envelope.horizon_years,
        "leverage_ceiling": envelope.leverage_ceiling,
    }
    if ctx.config.committee_capital and ctx.config.committee_evidence_weighting:
        _train_ends = {"top_level": COMMITTEE_OOS_START}
        _train_ends.update({
            f"fold_{_i}": _f.train_end
            for _i, _f in enumerate(phase_1_anchored_purged_folds())
        })
        ctx._committee_weights_by_boundary = committee._committee_evidence_weights_by_boundary(
            ctx.close, ctx.quote_vol, ctx.taker_buy_quote, ctx.execution_mask, ctx.slow_grid, ctx.slow.min_symbols, _train_ends,
            members=_research_go._resolved_committee_members(ctx.config),
        )
        # I-SINGLE-CONFIGURATION (mirrors the constant_risk target_vol fix,
        # ADR_20260823_MHS_FOLD_BLEND_PATH_DIVERGENCE_SINGLE_TARGET_VOL): the
        # blend deploys ONE committee member-weight mix, resolved once at the
        # top-level boundary, for the whole run. Anchored-purged folds must
        # validate that SAME deployed mix -- not a per-fold-boundary leak-free
        # refit of it, which drifts the fold's signal blend away from what the
        # blend actually deploys (measured: dominant-member weight drift up to
        # ~0.05, e.g. flow_imb_168h 0.389 top-level vs 0.442 for fold_0).
        ctx._fold_committee_weights = dict.fromkeys(
            range(len(phase_1_anchored_purged_folds())), ctx._committee_weights_by_boundary["top_level"],
        )
    if ctx.config.committee_capital:
        # RC-4: the reported blend is the committee execution book, not the
        # frozen momentum formula. Un-scaled copy feeds the concurrent replay
        # base so regime_scale applies exactly once (matching the fold path).
        ctx.blend_1h = committee._committee_execution_book(
            ctx.close, ctx.quote_vol, ctx.taker_buy_quote, ctx.execution_mask, ctx.slow_grid, ctx.slow.min_symbols,
            COMMITTEE_TRANCHE_COUNT
            if (ctx.config.committee_tranche_smoothing or ctx.config.committee_regime_adaptive_tranche)
            else 1,
            regime_adaptive_window=(
                COMMITTEE_REGIME_ADAPTIVE_WINDOW
                if ctx.config.committee_regime_adaptive_tranche else None
            ),
            target_gross=_research_go._resolved_committee_target_gross(ctx.config),
            member_weights=(ctx._committee_weights_by_boundary.get("top_level") if ctx.config.committee_evidence_weighting else None),
            carry_book=funding_carry_execution_book(ctx.bar_funding, ctx.execution_mask, FUNDING_CARRY_SLEEVE_LOOKBACK_HOURS, ctx.slow_grid, COMMITTEE_TRANCHE_COUNT, ctx.slow.min_symbols) if ctx.config.funding_carry_sleeve else None, carry_weight=ctx.config.funding_carry_weight if ctx.config.funding_carry_sleeve else 0.0,
            members=_research_go._resolved_committee_members(ctx.config),
            coverage_cutoff=COMMITTEE_OOS_START,
            beta=(
                causal_market_beta(
                    ctx.log_close, ctx.eligible,
                    CAUSAL_BETA_LOOKBACK_BARS, CAUSAL_BETA_MIN_PERIODS,
                )
                if ctx.config.beta_neutralize
                else None
            ),
        ).reindex(ctx.grid_1h).ffill().fillna(0.0)
        ctx.committee_execution_book = ctx.blend_1h
        # Build per-member attribution books (I5: observational only)
        if ctx.config.committee_member_attribution:
            ctx.committee_member_books = committee._committee_member_books(
                ctx.close, ctx.quote_vol, ctx.taker_buy_quote, ctx.execution_mask,
                ctx.slow_grid, ctx.slow.min_symbols,
                _research_go._resolved_committee_members(ctx.config),
                _research_go._resolved_committee_target_gross(ctx.config),
            )
            # D6: independent 1h prescreen-proxy Sharpe per member -- the
            # historical selection signal (yearly_net_t_diagnostic /
            # discovery gate all consume mhs_ledger_pnl on the 1h book, never
            # the 3m minute replay). This must stay derived from a source
            # distinct from the 3m ledger replay in replay.py, or
            # proxy_vs_ledger_rank_spearman compares the ledger to itself and
            # can never detect the proxy reversals ADR_20260819/20260817/
            # 20260820 measured.
            ctx.committee_member_proxy_sharpe = {}
            for _member_name, _member_book in ctx.committee_member_books.items():
                _member_net, _ = mhs_ledger_pnl(
                    _member_book.reindex(ctx.grid_1h).ffill().fillna(0.0),
                    ctx.opens, ctx.bar_funding, MEASURED_EXECUTION_COST_TIERS_BPS["base"],
                )
                _member_sd = float(_member_net.std(ddof=1)) if len(_member_net) > 1 else float("nan")
                if np.isfinite(_member_sd) and _member_sd > 0:
                    ctx.committee_member_proxy_sharpe[_member_name] = float(
                        _member_net.mean() / _member_sd * np.sqrt(_PERIODS_PER_YEAR_1H)
                    )
        del ctx.close, ctx.quote_vol, ctx.taker_buy_quote
    else:
        ctx.blend_1h = (
            BOOK_BLEND_WEIGHTS["fast_reversal"] * ctx.w_fast_1h
            + BOOK_BLEND_WEIGHTS["slow_momentum"] * ctx.w_slow_1h
        )
        ctx.committee_execution_book = None
    # Capture the pre-sleeve deployed book for the diagnostic, then add the
    # gross-budget sleeve to the executed blend -- including the committee book
    # passed to the replay -- before the regime cash-scale multiply, so the
    # overlay rides the same de-risking machinery as the deployed book.
    ctx.current_book_for_diagnostic = ctx.blend_1h
    ctx.trend_position = (
        folds._trend_sleeve_position(ctx.log_close, ctx.eligible, ctx.slow_grid)
        if (ctx.config.trend_sleeve and ctx.config.trend_sleeve_gross > 0.0)
        else None
    )
    if ctx.trend_position is not None:
        ctx.blend_1h = folds._apply_trend_sleeve(
            ctx.blend_1h, ctx.trend_position, ctx.execution_mask, ctx.config.trend_sleeve_gross,
        )
        if ctx.committee_execution_book is not None:
            ctx.committee_execution_book = ctx.blend_1h
    ctx.blend_gross = float(ctx.blend_1h.abs().sum(axis=1).mean())
    ctx.blend_cash_fraction = float((1.0 - ctx.blend_1h.abs().sum(axis=1)).mean())
    # R1: apply the same volatility-regime cash scale the fold path applies to
    # its blended targets (_build_fold_target_weights) so top-level prescreen/
    # tail/execution diagnostics are comparable to fold primary evidence
    # (spec §3.2, ``regime_cash_scale``).
    vol_mean = realized_vol(ctx.log_close, 48).where(ctx.execution_mask).reindex(ctx.grid_1h).mean(axis=1)
    ctx.regime_scale = _scaling._regime_cash_scale(vol_mean)
    if ctx.config.trend_efficiency_overlay:
        ctx.regime_scale = ctx.regime_scale.mul(
            _scaling._trend_efficiency_overlay_scale(ctx.log_close, ctx.execution_mask, ctx.fast.horizon_hours, ctx.grid_1h),
        )
    ctx.blend_1h = ctx.blend_1h.mul(ctx.regime_scale, axis=0)
    del vol_mean
    # The 1h book views are only consumed by ``blend_1h`` above.  Releasing
    # them before phase diagnostics and the top-level replays keeps two full
    # multi-year weight matrices out of the replay baseline (spec §3.1).
    del ctx.w_fast_1h, ctx.w_slow_1h
    gc.collect()

    ctx.phase_fast = diagnostics._phase_diagnostics(ctx.log_close, ctx.eligible, ctx.opens, ctx.bar_funding, ctx.grid_1h, ctx.fast)
    ctx.phase_slow = diagnostics._phase_diagnostics(ctx.log_close, ctx.eligible, ctx.opens, ctx.bar_funding, ctx.grid_1h, ctx.slow)
    _blend_spec, _blend_grid = books._active_blend_book_and_grid(ctx.fast, ctx.slow, ctx.fast_grid, ctx.slow_grid)
    del _blend_grid
    ctx.phase_blend = diagnostics._phase_diagnostics(ctx.log_close, ctx.eligible, ctx.opens, ctx.bar_funding, ctx.grid_1h, _blend_spec)

    # R3: the 48h cross-sectional statistics depend only on the 1h panel, not
    # the book replays.  Computing them here -- and computing ``signal_48h``
    # once so the placebo reuses it -- lets ``log_close`` be released before the
    # three top-level replays instead of staying alive throughout them
    # (spec §3.1, ``memory_opt``).
    ctx.signal_48h = horizon_log_return(ctx.log_close, 48)
    ctx.xs_ic = _statistics._xs_rank_ic(ctx.signal_48h, ctx.opens, forward_bars=48)
    ctx.trend_sleeve_diagnostic = diagnostics._trend_sleeve_diagnostic(
        ctx.log_close, ctx.eligible, ctx.opens, ctx.bar_funding, ctx.execution_mask,
        ctx.current_book_for_diagnostic, ctx.config,
    ) if ctx.config.trend_sleeve else None
    # The pre-sleeve book is consumed by the diagnostic; the post-sleeve
    # `blend_1h` alone must survive into the replay.
    del ctx.current_book_for_diagnostic
    # Feature-axis opt-in diagnostics run after fold pool with evicted caches.
    ctx.multi_feature_diagnostic = None
    ctx.committee_diagnostic = None
    ctx.regression = _statistics._date_clustered_ols(ctx.opens, ctx.signal_48h, forward_bars=48)
    ctx.horizon_diagnostics = {
        "realized_vol_48h_mean": float(
            realized_vol(ctx.log_close, 48).mean().mean()
        ),
        "efficiency_ratio_48h_mean": float(
            efficiency_ratio(ctx.log_close, 48).mean().mean()
        ),
    }
    ctx.discovery_qualification = None
    ctx.full_history_yearly_net_t = None
    ctx.funding_carry_worst_year_corr = None
    if ctx.config.discovery_gate:
        assert ctx.candidate_books is not None
        _slow_candidate_weights = ctx.candidate_books["slow"]
        _fast_candidate_weights = ctx.candidate_books["fast"]
        _funding_carry_candidate_weights = {
            1: ctx.candidate_books["funding_long"],
            -1: ctx.candidate_books["funding_short"],
        }
        ctx.discovery_qualification = {
            "reversal": select_horizon_by_discovery_qualification(
                sign=-1, horizon_candidates=DISCOVERY_REVERSAL_CANDIDATES,
                log_close=ctx.log_close, eligible=ctx.eligible, opens=ctx.opens,
                bar_funding=ctx.bar_funding, grid_1h=ctx.grid_1h,
                discovery_start=DISCOVERY_START, discovery_end=DISCOVERY_END,
                qualification_end=QUALIFICATION_END,
                tranche_count=DISCOVERY_GATE_TRANCHE_COUNT,
                precomputed_candidate_weights=_fast_candidate_weights,
                compute_adjusted_net_t=ctx.config.discovery_gate_adjusted_net_t,
                compute_regime_scaled_net_t=ctx.config.discovery_gate_regime_scaled_net_t,
            ),
            "momentum": select_horizon_by_discovery_qualification(
                sign=1, horizon_candidates=DISCOVERY_MOMENTUM_CANDIDATES,
                log_close=ctx.log_close, eligible=ctx.eligible, opens=ctx.opens,
                bar_funding=ctx.bar_funding, grid_1h=ctx.grid_1h,
                discovery_start=DISCOVERY_START, discovery_end=DISCOVERY_END,
                qualification_end=QUALIFICATION_END,
                tranche_count=DISCOVERY_GATE_TRANCHE_COUNT,
                precomputed_candidate_weights=_slow_candidate_weights,
                compute_adjusted_net_t=ctx.config.discovery_gate_adjusted_net_t,
                compute_regime_scaled_net_t=ctx.config.discovery_gate_regime_scaled_net_t,
            ),
            "funding_carry_long": select_horizon_by_discovery_qualification(
                sign=1, horizon_candidates=FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS,
                log_close=ctx.log_close, eligible=ctx.eligible, opens=ctx.opens,
                bar_funding=ctx.bar_funding, grid_1h=ctx.grid_1h,
                discovery_start=DISCOVERY_START, discovery_end=DISCOVERY_END,
                qualification_end=QUALIFICATION_END,
                tranche_count=DISCOVERY_GATE_TRANCHE_COUNT,
                precomputed_candidate_weights=_funding_carry_candidate_weights[1],
                compute_adjusted_net_t=ctx.config.discovery_gate_adjusted_net_t,
                compute_regime_scaled_net_t=ctx.config.discovery_gate_regime_scaled_net_t,
            ),
            "funding_carry_short": select_horizon_by_discovery_qualification(
                sign=-1, horizon_candidates=FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS,
                log_close=ctx.log_close, eligible=ctx.eligible, opens=ctx.opens,
                bar_funding=ctx.bar_funding, grid_1h=ctx.grid_1h,
                discovery_start=DISCOVERY_START, discovery_end=DISCOVERY_END,
                qualification_end=QUALIFICATION_END,
                tranche_count=DISCOVERY_GATE_TRANCHE_COUNT,
                precomputed_candidate_weights=_funding_carry_candidate_weights[-1],
                compute_adjusted_net_t=ctx.config.discovery_gate_adjusted_net_t,
                compute_regime_scaled_net_t=ctx.config.discovery_gate_regime_scaled_net_t,
            ),
        }
        # Full-history (2021-2025) yearly net-t diagnostics (report-only).
        ctx.full_history_yearly_net_t = {
            "slow_momentum": yearly_net_t_diagnostic(
                ctx.w_slow.reindex(ctx.grid_1h).ffill().fillna(0.0), ctx.opens, ctx.bar_funding,
                (2021, 2022, 2023, 2024, 2025),
                MEASURED_EXECUTION_COST_TIERS_BPS["base"], _PERIODS_PER_YEAR_1H,
            ),
            "fast_reversal": yearly_net_t_diagnostic(
                ctx.w_fast.reindex(ctx.grid_1h).ffill().fillna(0.0), ctx.opens, ctx.bar_funding,
                (2021, 2022, 2023, 2024, 2025),
                MEASURED_EXECUTION_COST_TIERS_BPS["base"], _PERIODS_PER_YEAR_1H,
            ),
        }
        _fc_pick = folds._prefer_funding_carry_selection(
            ctx.discovery_qualification["funding_carry_long"],
            ctx.discovery_qualification["funding_carry_short"],
        )
        _fc_lookback, _fc_sign = _fc_pick if _fc_pick is not None else (168, 1)
        _fc_book = _funding_carry_candidate_weights[_fc_sign][_fc_lookback]
        ctx.full_history_yearly_net_t["funding_carry"] = yearly_net_t_diagnostic(
            _fc_book, ctx.opens, ctx.bar_funding, (2021, 2022, 2023, 2024, 2025),
            MEASURED_EXECUTION_COST_TIERS_BPS["base"], _PERIODS_PER_YEAR_1H,
        )
        # Worst-year-restricted correlation: does momentum's weakest calendar
        # year still get funding-carry diversification (spec §2.2)?
        _fc_net, _ = mhs_ledger_pnl(
            _fc_book, ctx.opens, ctx.bar_funding, MEASURED_EXECUTION_COST_TIERS_BPS["base"],
        )
        _fc_daily = (1.0 + _fc_net).resample("1D").apply(lambda s: s.prod() - 1.0)
        _slow_net, _ = mhs_ledger_pnl(
            ctx.w_slow.reindex(ctx.grid_1h).ffill().fillna(0.0), ctx.opens, ctx.bar_funding,
            MEASURED_EXECUTION_COST_TIERS_BPS["base"],
        )
        _momentum_daily = (1.0 + _slow_net).resample("1D").apply(lambda s: s.prod() - 1.0)
        _slow_yearly = ctx.full_history_yearly_net_t["slow_momentum"]
        _finite_years = [y for y, t in _slow_yearly.items() if np.isfinite(t)]
        if _finite_years:
            _worst_year = min(_finite_years, key=lambda y: _slow_yearly[y])
            ctx.funding_carry_worst_year_corr = year_restricted_correlation(
                _fc_daily, _momentum_daily, (_worst_year,),
            )
        # Effective breadth audit across candidate weight books.
        _slow_daily_returns: dict[int, pd.Series] = {}
        _fast_daily_returns: dict[int, pd.Series] = {}
        for _horizon, _book in _slow_candidate_weights.items():
            _net, _ = mhs_ledger_pnl(
                _book, ctx.opens, ctx.bar_funding, MEASURED_EXECUTION_COST_TIERS_BPS["base"],
            )
            _slow_daily_returns[_horizon] = (1.0 + _net).resample("1D").apply(
                lambda s: s.prod() - 1.0
            )
        for _horizon, _book in _fast_candidate_weights.items():
            _net, _ = mhs_ledger_pnl(
                _book, ctx.opens, ctx.bar_funding, MEASURED_EXECUTION_COST_TIERS_BPS["base"],
            )
            _fast_daily_returns[_horizon] = (1.0 + _net).resample("1D").apply(
                lambda s: s.prod() - 1.0
            )
        ctx.horizon_diagnostics["slow_horizon_effective_breadth"], _ = effective_breadth(
            pd.DataFrame(_slow_daily_returns)
        )
        ctx.horizon_diagnostics["fast_horizon_effective_breadth"], _ = effective_breadth(
            pd.DataFrame(_fast_daily_returns)
        )
    del ctx.log_close
    gc.collect()
