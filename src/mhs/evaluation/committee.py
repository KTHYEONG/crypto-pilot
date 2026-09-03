from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.application.research.mhs import statistics as _statistics
from src.application.research.mhs.contracts import MhsBookReport
from src.application.research.mhs.resources import _assert_stage_rss_budget, _StageRecorder
from src.mhs.books import phase_tranche_book, scale_book_to_target_gross
from src.mhs.committee import (
    committee_block_edges_from,
    decompose_cost,
    long_only_equal_risk_weights,
    purged_walk_forward,
    score_weighted_net,
    train_evidence_weights,
    wealth_metrics,
)
from src.mhs.execution import mhs_ledger_pnl_multi_tier
from src.mhs.features import (
    FEATURE_REGISTRY,
    FeatureSpec,
    build_feature_books,
    feature_registry_panel_columns,
    source_coverage_audit,
)
from src.mhs.params import (
    COMMITTEE_GROWTH_BARS_PER_YEAR,
    COMMITTEE_GROWTH_HORIZON_YEARS,
    COMMITTEE_GROWTH_MAX_DRAWDOWN,
    COMMITTEE_GROWTH_MAX_DRAWDOWN_PROB,
    COMMITTEE_GROWTH_MAX_RUIN_PROB,
    COMMITTEE_GROWTH_N_PATHS,
    COMMITTEE_GROWTH_RISK_GRID_MULTIPLIERS,
    COMMITTEE_GROWTH_RUIN_FRACTION,
    COMMITTEE_MEMBERS,
    COMMITTEE_OOS_START,
    COMMITTEE_PURGE_HOURS,
    COMMITTEE_TARGET_VOL,
    FEATURE_MIN_COVERAGE,
    MEASURED_EXECUTION_COST_TIERS_BPS,
    WALK_FORWARD_MIN_TRAIN_BARS,
)
from src.mhs.params import PERIODS_PER_YEAR_1H as _PERIODS_PER_YEAR_1H
from src.mhs.regime import beta_neutralize_weights
from src.research.risk.growth_sizing import GrowthSizingConfig, diagnose_growth_headroom, solve_growth_optimal_risk

from . import diagnostics

_logger = logging.getLogger("MhsHorizonDiagnostic")

def _committee_growth_headroom(
    gross_all: pd.DataFrame,
    tc_all: pd.DataFrame,
    cost_bps: float,
    oos_start: pd.Timestamp = COMMITTEE_OOS_START,
) -> dict[str, Any] | None:
    """Discovery-window-only headroom report via the reused growth_sizing solver.

    Observational only: never feeds back into weights, scales, or replay
    decisions. Fits strictly on bars before ``oos_start``; a degenerate or
    short discovery window returns None instead of raising.
    """
    discovery_mask = gross_all.index < oos_start
    if discovery_mask.sum() < 30:
        return None
    net = gross_all - tc_all * cost_bps
    weights = long_only_equal_risk_weights(net.loc[discovery_mask])
    discovery_net = score_weighted_net(
        weights, gross_all.loc[discovery_mask], tc_all.loc[discovery_mask], cost_bps,
    )
    reference_risk = float(discovery_net.std(ddof=1))
    if not np.isfinite(reference_risk) or reference_risk <= 0:
        return None
    risk_grid = tuple(
        sorted(reference_risk * m for m in COMMITTEE_GROWTH_RISK_GRID_MULTIPLIERS)
    )
    config = GrowthSizingConfig(
        risk_grid=risk_grid,
        reference_risk=reference_risk,
        max_drawdown=COMMITTEE_GROWTH_MAX_DRAWDOWN,
        max_drawdown_prob=COMMITTEE_GROWTH_MAX_DRAWDOWN_PROB,
        ruin_fraction=COMMITTEE_GROWTH_RUIN_FRACTION,
        max_ruin_prob=COMMITTEE_GROWTH_MAX_RUIN_PROB,
        horizon_years=COMMITTEE_GROWTH_HORIZON_YEARS,
        n_paths=COMMITTEE_GROWTH_N_PATHS,
        bars_per_year=COMMITTEE_GROWTH_BARS_PER_YEAR,
    )
    selected = solve_growth_optimal_risk(discovery_net.to_numpy(), config)
    headroom = diagnose_growth_headroom(discovery_net.to_numpy(), config, selected)
    return {
        "reference_risk": reference_risk,
        "selected_risk": (
            _statistics._finite_or_none(selected.selected_risk)
            if selected.selected_risk is not None else None
        ),
        "median_log_growth": _statistics._finite_or_none(selected.median_log_growth),
        "mdd_breach_prob": _statistics._finite_or_none(selected.mdd_breach_prob),
        "ruin_prob": _statistics._finite_or_none(selected.ruin_prob),
        "binding_constraint": selected.binding_constraint,
        "headroom_ratio": _statistics._finite_or_none(headroom.headroom_ratio),
        "risk_constrained": headroom.risk_constrained,
        "discovery_bars": int(discovery_mask.sum()),
    }

def _committee_member_books(
    close: pd.DataFrame,
    quote_vol: pd.DataFrame,
    taker_buy_quote: pd.DataFrame,
    execution_mask: pd.DataFrame,
    decision_grid: pd.DatetimeIndex,
    min_symbols: int,
    members: tuple[str, ...],
    target_gross: float | None,
) -> dict[str, pd.DataFrame]:
    """Build individual execution books for each committee member (I5: observational only).

    Returns ``{member_name: execution_book}`` where each book is a
    single-member ``_committee_execution_book`` call. These books are used
    ONLY for attribution reporting and never enter ``blend_1h``,
    ``committee_execution_book``, ``regime_scale``, any exposure scale, any
    fold report, or any Research-GO reason code.
    """
    from src.mhs.books import scale_book_to_target_gross
    from src.mhs.features import FEATURE_REGISTRY, build_feature_books

    member_specs = [spec for spec in FEATURE_REGISTRY if spec.name in set(members)]
    specs_by_name = {spec.name: spec for spec in member_specs}

    member_books: dict[str, pd.DataFrame] = {}
    for name in members:
        member_spec = specs_by_name.get(name)
        if member_spec is None:
            continue
        single = build_feature_books(
            [member_spec],
            {col: close if col == "close" else (quote_vol if col == "quote_vol" else taker_buy_quote)
             for col in member_spec.required_columns if col in ("close", "quote_vol", "taker_buy_quote")},
            execution_mask,
            decision_grid,
            min_symbols=min_symbols,
        )
        if name not in single:
            continue
        book = single[name]
        if target_gross is not None:
            book = scale_book_to_target_gross(book, target_gross)
        member_books[name] = book
    return member_books


def _committee_member_attribution(
    member_reports: dict[str, MhsBookReport],
    member_proxy_sharpe: dict[str, float],
) -> dict[str, Any]:
    """Compute per-member attribution metrics from their individual replays.

    Returns a dict with per-member metrics (cagr, naive_sharpe, max_drawdown,
    annualized_turnover, net_ann) and the proxy_vs_ledger_rank_spearman
    diagnostic. This is purely observational (I5).
    """
    from scipy.stats import spearmanr

    members_data: dict[str, dict[str, Any]] = {}
    ledger_sharpes: dict[str, float] = {}

    for name, report in member_reports.items():
        if report is None or report.primary is None:
            members_data[name] = {
                "cagr": None,
                "naive_sharpe": None,
                "max_drawdown": None,
                "annualized_turnover": None,
                "net_ann": None,
            }
            continue
        equity = report.primary.ledger.equity
        net_returns = report.primary.ledger.net_returns
        turnover = report.primary.ledger.fill_turnover
        periods_per_year = _PERIODS_PER_YEAR_1H

        equity_1h = equity.resample("1h").last().dropna()
        cagr = float(equity_1h.iloc[-1] ** (periods_per_year / len(equity_1h)) - 1.0) if len(equity_1h) > 0 else None

        mdd = float((equity / equity.cummax() - 1.0).min()) if len(equity) > 0 else None
        sd = float(net_returns.std(ddof=1)) if len(net_returns) > 1 else float("nan")
        sharpe = (
            float(net_returns.mean() / sd * np.sqrt(periods_per_year))
            if np.isfinite(sd) and sd > 0
            else None
        )
        annual_turnover = float(turnover.mean() * periods_per_year) if len(turnover) > 0 else None
        net_ann = float(net_returns.mean() * periods_per_year) if len(net_returns) > 0 else None

        members_data[name] = {
            "cagr": cagr,
            "naive_sharpe": sharpe,
            "max_drawdown": mdd,
            "annualized_turnover": annual_turnover,
            "net_ann": net_ann,
        }
        if sharpe is not None and np.isfinite(sharpe):
            ledger_sharpes[name] = sharpe

    # Proxy vs ledger rank correlation
    shared = sorted(set(ledger_sharpes.keys()) & set(member_proxy_sharpe.keys()))
    proxy_vs_ledger_rank_spearman: float | None = None
    if len(shared) >= 3:
        ledger_vals = [ledger_sharpes[m] for m in shared]
        proxy_vals = [member_proxy_sharpe[m] for m in shared]
        rho, _ = spearmanr(ledger_vals, proxy_vals)
        proxy_vs_ledger_rank_spearman = float(rho) if np.isfinite(rho) else None

    # Daily return correlation matrix
    daily_return_correlation: dict[str, dict[str, float]] = {}
    member_nets: dict[str, pd.Series] = {}
    for name, report in member_reports.items():
        if report is not None and report.primary is not None:
            daily = report.primary.ledger.net_returns.resample("1D").apply(lambda s: (1 + s).prod() - 1.0)
            member_nets[name] = daily
    net_names = sorted(member_nets.keys())
    for i, n1 in enumerate(net_names):
        daily_return_correlation[n1] = {}
        for j, n2 in enumerate(net_names):
            if i == j:
                daily_return_correlation[n1][n2] = 1.0
            elif n2 in daily_return_correlation and n1 in daily_return_correlation[n2]:
                daily_return_correlation[n1][n2] = daily_return_correlation[n2][n1]
            else:
                aligned = pd.concat([member_nets[n1], member_nets[n2]], axis=1).dropna()
                if len(aligned) > 1:
                    rho = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                    daily_return_correlation[n1][n2] = rho if np.isfinite(rho) else 0.0
                else:
                    daily_return_correlation[n1][n2] = 0.0

    return {
        "members": members_data,
        "daily_return_correlation": daily_return_correlation,
        "proxy_vs_ledger_rank_spearman": proxy_vs_ledger_rank_spearman,
    }


def _committee_diagnostic(
    root: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    grid_1h: pd.DatetimeIndex,
    aligned_symbols: list[str],
    execution_mask: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    panels: Mapping[str, pd.DataFrame] | None = None,
    rss_budget_bytes: int | None = None,
    rss_reserve_bytes: int | None = None,
    telemetry: _StageRecorder | None = None,
    sizing_mode: Literal["vol_target", "kelly_blend"] = "vol_target",
    growth_diagnostic: bool = False,
) -> dict[str, Any]:
    """SCENARIO_MHS_COMMITTEE_DIAGNOSTIC_REPORTS_WALK_FORWARD_WEALTH:
    opt-in measurement of the k=5 wealth committee.

    Builds the declared committee members into the dollar-neutral rank books on
    the 24h decision grid, audits the RAW source panels for pre-fillna coverage
    gaps via ``source_coverage_audit`` and fail-closes any member whose required
    source drops below ``FEATURE_MIN_COVERAGE`` in ANY year BEFORE
    ``build_feature_books`` (B3 -- the funding gap the post-fillna feature audit
    cannot see), recovers sign-safe gross and turnover-cost panels from the two
    extreme measured cost tiers via ``decompose_cost``, and runs the purged
    expanding-train walk-forward at every measured cost tier, reporting the
    compounded-growth wealth metrics per tier. The walk-forward block grid is
    anchored at ``COMMITTEE_OOS_START``, and any blocks skipped by the walk-forward
    are reported alongside the edges.

    Memory-optimized streaming: panels are column-pruned to committee requirements
    and processed sequentially to minimize memory residency.
    """
    if panels is None:
        panels = diagnostics._load_feature_panels(
            root, start, end, grid_1h, aligned_symbols,
            columns=feature_registry_panel_columns(
                [
                    spec for spec in FEATURE_REGISTRY
                    if spec.name in set(COMMITTEE_MEMBERS)
                ],
            ),
        )
    _assert_stage_rss_budget("committee_feature_panels", rss_budget_bytes, rss_reserve_bytes)
    member_specs = [
        spec for spec in FEATURE_REGISTRY if spec.name in set(COMMITTEE_MEMBERS)
    ]

    # B3: source-coverage pre-filter. Every required RAW source column present
    # in the panels is audited against the execution mask -- including an
    # all-NaN column, whose per-year coverage is 0.0 (the funding 45/452-symbol
    # gap a post-fillna feature audit cannot see). A member with ANY year below
    # FEATURE_MIN_COVERAGE is dropped from member_specs BEFORE
    # build_feature_books so it never contributes a book, a PnL series, or a
    # weight -- fail closed, mirroring feature_coverage_audit's
    # exclude-not-nan-fill discipline at the source level.
    source_coverage: dict[str, dict[str, dict[int, float]]] = {}
    source_excluded: dict[str, dict[str, Any]] = {}
    source_admissible_specs: list[FeatureSpec] = []
    for spec in member_specs:
        per_source: dict[str, dict[int, float]] = {}
        failing_sources: dict[str, int] = {}
        for column in spec.required_columns:
            if column not in panels:
                continue
            coverage = source_coverage_audit(panels[column], execution_mask)
            per_source[column] = coverage
            for year, cov in coverage.items():
                if cov < FEATURE_MIN_COVERAGE:
                    failing_sources[column] = min(
                        failing_sources.get(column, year), year,
                    )
        source_coverage[spec.name] = per_source
        _logger.debug(
            "[DATA] stage=committee_source_coverage member=%s excluded=%s min_coverage=%.3f",
            spec.name, spec.name in source_excluded,
            min((c for cov in per_source.values() for c in cov.values()), default=1.0),
        )
        if failing_sources:
            failing_source = min(failing_sources, key=lambda c: failing_sources[c])
            source_excluded[spec.name] = {
                "failing_source": failing_source,
                "failing_year": failing_sources[failing_source],
            }
        else:
            source_admissible_specs.append(spec)
    member_specs = source_admissible_specs

    decision_grid = pd.date_range(grid_1h[0], grid_1h[-1], freq="24h", tz="UTC")
    specs_by_name = {spec.name: spec for spec in member_specs}

    bps_low = MEASURED_EXECUTION_COST_TIERS_BPS["optimistic"]
    bps_high = MEASURED_EXECUTION_COST_TIERS_BPS["stress"]

    # Stream one member at a time in COMMITTEE_MEMBERS order (preserving the
    # pre-streaming admitted/net-panel column order), keep only the two cost-tier
    # net series, and drop the book immediately.
    admitted: list[str] = []
    net_low_by_name: dict[str, pd.Series] = {}
    net_high_by_name: dict[str, pd.Series] = {}
    for name in COMMITTEE_MEMBERS:
        member_spec = specs_by_name.get(name)
        if member_spec is None:
            continue
        _assert_stage_rss_budget(
            f"committee_member_{name}", rss_budget_bytes, rss_reserve_bytes,
        )
        single = build_feature_books(
            [member_spec], panels, execution_mask, decision_grid, min_symbols=8,
        )
        if name not in single:
            continue
        book = single[name]
        (net_low, _), (net_high, _) = mhs_ledger_pnl_multi_tier(
            book, opens, bar_funding, [bps_low, bps_high],
        )
        net_low_by_name[name] = net_low
        net_high_by_name[name] = net_high
        admitted.append(name)
        _logger.debug(
            "[ALGO] stage=committee_member member=%s net_low_mean=%.6f net_high_mean=%.6f",
            name, float(net_low.mean()), float(net_high.mean()),
        )
        del single, book

    excluded = [
        {"name": name, "reason": "feature_coverage"}
        for name in COMMITTEE_MEMBERS
        if name not in admitted and name not in source_excluded
    ]
    excluded.extend(
        {
            "name": name,
            "reason": "source_coverage",
            "failing_source": details["failing_source"],
            "failing_year": details["failing_year"],
        }
        for name, details in source_excluded.items()
    )

    gross_all: pd.DataFrame | None = None
    tc_all: pd.DataFrame | None = None
    if admitted:
        net_low_panel = pd.DataFrame(net_low_by_name)
        net_high_panel = pd.DataFrame(net_high_by_name)
        gross_all, tc_all = decompose_cost(
            net_low_panel, net_high_panel, bps_low, bps_high,
        )

    # B1: anchor the OOS block grid at COMMITTEE_OOS_START, never the raw
    # diagnostic start, so min_train_bars (~83 days) can no longer smuggle
    # pre-OOS blocks in as pseudo-OOS.
    edges = committee_block_edges_from(start, COMMITTEE_OOS_START, end)
    purge = pd.Timedelta(hours=COMMITTEE_PURGE_HOURS)

    # B6: re-derive which candidate block edges purged_walk_forward skips
    # (insufficient train rows or no test bars), independently of its internal
    # loop, so a silently-ignored calendar gap in the concatenated wealth
    # series is surfaced to the reader. Report-only, never raises.
    skipped_blocks: list[dict[str, str]] = []
    if gross_all is not None:
        for i, t0 in enumerate(edges):
            next_edge = (
                edges[i + 1]
                if i + 1 < len(edges)
                else gross_all.index[-1] + pd.Timedelta(hours=1)
            )
            train_rows = gross_all.index < (t0 - purge)
            if int(train_rows.sum()) < WALK_FORWARD_MIN_TRAIN_BARS:
                skipped_blocks.append(
                    {"block_start": t0.isoformat(), "reason": "insufficient_train"}
                )
                continue
            test_rows = (gross_all.index >= t0) & (gross_all.index < next_edge)
            if not bool(test_rows.any()):
                skipped_blocks.append(
                    {"block_start": t0.isoformat(), "reason": "no_test_bars"}
                )

    per_tier: dict[str, dict[str, Any]] = {}
    for tier, cost_bps in MEASURED_EXECUTION_COST_TIERS_BPS.items():
        if gross_all is None:
            per_tier[tier] = {
                "net_sharpe": None, "cagr": None, "mdd": None,
                "logret": None, "bars": 0, "blocks": [],
            }
            continue
        wf = purged_walk_forward(
            gross_all, tc_all, cost_bps, edges, purge,
            min_train_bars=WALK_FORWARD_MIN_TRAIN_BARS,
            sizing_mode=sizing_mode,
        )
        if telemetry is not None:
            telemetry.record(f"committee_walk_forward_{tier}")
        metrics = wealth_metrics(wf)
        total_logret = metrics["logret"]
        _logger.debug(
            "[EVAL] stage=committee_tier_summary tier=%s bars=%d sharpe=%s cagr=%s mdd=%s",
            tier, len(wf), metrics["sharpe"], metrics["cagr"], metrics["mdd"],
        )
        blocks: list[dict[str, Any]] = []
        for i, t0 in enumerate(edges):
            next_edge = (
                edges[i + 1] if i + 1 < len(edges)
                else gross_all.index[-1] + pd.Timedelta(hours=1)
            )
            block_wf = wf[(wf.index >= t0) & (wf.index < next_edge)]
            if block_wf.empty:
                continue
            block_metrics = wealth_metrics(block_wf)
            _block_rho1 = (
                block_wf.autocorr(1) if len(block_wf) > 2 else float("nan")
            )
            blocks.append({
                "block_start": t0.isoformat(),
                "bars": len(block_wf),
                "net_sharpe": _statistics._finite_or_none(block_metrics["sharpe"]),
                "cagr": _statistics._finite_or_none(block_metrics["cagr"]),
                "mdd": _statistics._finite_or_none(block_metrics["mdd"]),
                "logret": _statistics._finite_or_none(block_metrics["logret"]),
                "logret_share": (
                    float(block_metrics["logret"] / total_logret)
                    if np.isfinite(total_logret)
                    and total_logret != 0
                    and np.isfinite(block_metrics["logret"])
                    else None
                ),
                "return_autocorr_lag1": (
                    float(_block_rho1) if np.isfinite(_block_rho1) else None
                ),
            })
            _logger.debug(
                "[EVAL] stage=committee_block tier=%s block_start=%s bars=%d sharpe=%s cagr=%s mdd=%s rho1=%s",
                tier, t0.isoformat(), len(block_wf),
                block_metrics["sharpe"], block_metrics["cagr"], block_metrics["mdd"],
                _block_rho1,
            )
        per_tier[tier] = {
            "net_sharpe": _statistics._finite_or_none(metrics["sharpe"]),
            "cagr": _statistics._finite_or_none(metrics["cagr"]),
            "mdd": _statistics._finite_or_none(metrics["mdd"]),
            "logret": _statistics._finite_or_none(metrics["logret"]),
            "bars": len(wf),
            "blocks": blocks,
        }

    growth_headroom = (
        _committee_growth_headroom(
            gross_all, tc_all, MEASURED_EXECUTION_COST_TIERS_BPS["base"],
        )
        if (growth_diagnostic and gross_all is not None)
        else None
    )

    return {
        "evaluation_protocol": "purged_walk_forward_oos",
        "trials_explored": 50,
        "selection_bias_warning": (
            "committee composition (k=5) was chosen after comparing ~50 "
            "feature/combiner/size configurations on this same 2021-2025 panel; "
            "treat OOS Sharpe as an upper bound, not a deflated estimate"
        ),
        "members": list(COMMITTEE_MEMBERS),
        "admitted": admitted,
        "excluded": excluded,
        "source_coverage": {
            name: {
                column: {str(year): float(cov) for year, cov in coverage.items()}
                for column, coverage in sources.items()
            }
            for name, sources in source_coverage.items()
        },
        "walk_forward": {
            "block_edges": [edge.isoformat() for edge in edges],
            "skipped_blocks": skipped_blocks,
            "purge_hours": COMMITTEE_PURGE_HOURS,
            "target_vol": COMMITTEE_TARGET_VOL,
            "sizing_mode": sizing_mode,
            "per_tier": per_tier,
        },
        "growth_headroom": growth_headroom,
    }








def _committee_evidence_weights_by_boundary(
    close: pd.DataFrame,
    quote_vol: pd.DataFrame,
    taker_buy_quote: pd.DataFrame,
    execution_mask: pd.DataFrame,
    decision_grid: pd.DatetimeIndex,
    min_symbols: int,
    train_ends: Mapping[str, pd.Timestamp],
    members: tuple[str, ...] | None = None,
) -> dict[str, dict[str, float]]:
    """Build per-boundary evidence weights for committee members.

    Member books and proxy return series are constructed exactly once
    regardless of ``len(train_ends)`` -- this is what makes fold-level
    evidence weighting possible without loading a second wide panel per fold.
    Each boundary (fold or top-level OOS) then fits its own evidence weights
    from the shared proxy return series, so every fold sees only the training
    data up to its own boundary. Admission is audited only up to
    ``max(train_ends.values())``, so an OOS-only tail beyond every requested
    boundary can never decide a member's availability for any of these fits
    (I-COVERAGE-PIT).
    """
    _resolved = members or COMMITTEE_MEMBERS
    _member_specs = [
        spec for spec in FEATURE_REGISTRY
        if spec.name in set(_resolved)
    ]
    import src.application.research.mhs.evaluation as ev
    _committee_books = ev.build_feature_books(
        _member_specs,
        {"close": close, "quote_vol": quote_vol, "taker_buy_quote": taker_buy_quote},
        execution_mask, decision_grid, min_symbols=min_symbols,
        coverage_cutoff=max(train_ends.values()),
    )
    if not _committee_books:
        return {label: {} for label in train_ends}
    close_grid = close.reindex(decision_grid).ffill()
    fwd_ret = np.log(close_grid).shift(-1) - np.log(close_grid)
    proxies: dict[str, pd.Series] = {}
    for name, book in _committee_books.items():
        book_grid = book.reindex(decision_grid).fillna(0.0)
        proxies[name] = (book_grid * fwd_ret).sum(axis=1)
    result: dict[str, dict[str, float]] = {}
    for label, train_end in train_ends.items():
        train_mask = pd.Series(decision_grid < train_end, index=decision_grid)
        result[label] = train_evidence_weights(proxies, train_mask)
    return result


def _committee_execution_book(
    close: pd.DataFrame,
    quote_vol: pd.DataFrame,
    taker_buy_quote: pd.DataFrame,
    execution_mask: pd.DataFrame,
    decision_grid: pd.DatetimeIndex,
    min_symbols: int,
    tranche_count: int = 1,
    regime_adaptive_window: int | None = None,
    target_gross: float | None = None,
    member_weights: Mapping[str, float] | None = None,
    carry_book: pd.DataFrame | None = None,
    carry_weight: float = 0.0,
    members: tuple[str, ...] | None = None,
    coverage_cutoff: pd.Timestamp | None = None,
    beta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the k=5 committee capital book on the decision grid.

    Shared by the fold path and the top-level blend: filter the registry to
    ``members`` (or ``COMMITTEE_MEMBERS`` when None), build equal-notional
    rank books, average them.  No leg-risk tilt -- tilting the curated committee
    set to equal risk removed the concentration that carries its edge per ADR_20260823_MHS_CONSTANT_RISK_DEPLOYMENT. Fails closed when no member is
    admitted. ``tranche_count`` smooths the decision rows with a staggered tranche
    mean (opt-in, defaults to the identity single-phase book).
    ``regime_adaptive_window`` (opt-in, mutually exclusive with a fixed
    ``tranche_count``-only smooth) selects per-row between the raw book and its
    ``tranche_count``-row smooth using a causal trailing lag-1 autocorrelation of
    the raw book's own proxy return. ``target_gross`` rescales each decision row
    to an explicit gross. ``member_weights`` is an externally-fitted,
    already-normalized-or-not mapping this function applies and renormalizes over
    admitted members. ``coverage_cutoff`` must match the boundary that produced
    ``member_weights`` (I-COVERAGE-PIT) -- otherwise a member available when the
    weights were fit can be silently dropped from the deployed book by a coverage
    gap in a later OOS-only tail the fit itself never saw.
    """
    if tranche_count < 1:
        raise ValueError(f"tranche_count must be >= 1, got {tranche_count}")
    if regime_adaptive_window is not None and regime_adaptive_window < 3:
        raise ValueError(
            f"regime_adaptive_window must be >= 3, got {regime_adaptive_window}"
        )
    _resolved = members or COMMITTEE_MEMBERS
    _member_specs = [
        spec for spec in FEATURE_REGISTRY
        if spec.name in set(_resolved)
    ]
    import src.application.research.mhs.evaluation as ev
    _committee_books = ev.build_feature_books(
        _member_specs,
        {"close": close, "quote_vol": quote_vol, "taker_buy_quote": taker_buy_quote},
        execution_mask, decision_grid, min_symbols=min_symbols,
        coverage_cutoff=coverage_cutoff,
    )
    if not _committee_books:
        raise RuntimeError(
            "committee_capital: no committee member admitted in this fold window"
        )
    if member_weights is not None:
        admitted = {n: max(0.0, member_weights.get(n, 0.0)) for n in _committee_books}
        total = sum(admitted.values())
        if total <= 0.0:
            book = sum(_committee_books.values()) / float(len(_committee_books))
        else:
            book = sum(admitted[n] / total * _committee_books[n] for n in _committee_books)
    else:
        book = sum(_committee_books.values()) / float(len(_committee_books))
    if regime_adaptive_window is not None:
        book_grid = book.reindex(decision_grid).fillna(0.0)
        smoothed_grid = phase_tranche_book(book_grid, tranche_count)
        close_grid = close.reindex(decision_grid).ffill()
        fwd_ret = np.log(close_grid).shift(-1) - np.log(close_grid)
        proxy_return = (book_grid * fwd_ret.reindex(decision_grid)).sum(axis=1)
        trailing_rho1 = (
            proxy_return.rolling(regime_adaptive_window, min_periods=regime_adaptive_window)
            .apply(_statistics._causal_lag1_autocorr, raw=True)
            .shift(1)
        )
        use_smoothed = (trailing_rho1 < 0.0).reindex(decision_grid).fillna(False)
        adaptive_grid = book_grid.mask(use_smoothed, smoothed_grid)
        result = adaptive_grid.reindex(book.index, method="ffill").fillna(0.0)
    elif tranche_count == 1:
        result = book
    else:
        smoothed = phase_tranche_book(book.reindex(decision_grid).fillna(0.0), tranche_count)
        result = smoothed.reindex(book.index, method="ffill").fillna(0.0)
    # Beta-neutralize the pure committee book BEFORE the carry blend (carry's
    # economics must not be distorted) and BEFORE target_gross scaling
    # (renormalize_within_mask resets unit gross, which would silently
    # override the deployed gross contract).
    if beta is not None:
        result = beta_neutralize_weights(
            result,
            beta.reindex(result.index),
            execution_mask.reindex(result.index).fillna(False),
            min_symbols,
        )
    if carry_book is not None and carry_weight > 0.0:
        if target_gross is None:
            raise ValueError(
                "carry_book with carry_weight > 0.0 requires target_gross "
                "to be set (the diluted book has no gross to normalize against)"
            )
        if not (0.0 <= carry_weight < 1.0):
            raise ValueError(f"carry_weight must be in [0.0, 1.0), got {carry_weight}")
        unit_committee = scale_book_to_target_gross(result, 1.0)
        unit_carry = scale_book_to_target_gross(
            carry_book.reindex(result.index).fillna(0.0), 1.0,
        )
        result = (1.0 - carry_weight) * unit_committee + carry_weight * unit_carry
    if target_gross is None:
        return result
    return scale_book_to_target_gross(result, target_gross)