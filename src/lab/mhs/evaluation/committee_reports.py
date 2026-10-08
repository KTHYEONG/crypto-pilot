"""Observational committee reports: growth headroom, member attribution and diagnostic sections.

Nothing here feeds back into weights, scales or replay decisions.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.core.params import (
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
)
from src.lab.mhs import statistics as _statistics
from src.lab.mhs.committee import long_only_equal_risk_weights, score_weighted_net, wealth_metrics
from src.lab.mhs.contracts import MhsBookReport
from src.lab.mhs.params import PERIODS_PER_YEAR_1H as _PERIODS_PER_YEAR_1H
from src.lab.mhs.params import WALK_FORWARD_MIN_TRAIN_BARS
from src.quant.risk.growth_sizing import GrowthSizingConfig, diagnose_growth_headroom, solve_growth_optimal_risk
from src.strategy.features import FeatureSpec, source_coverage_audit

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
        weights,
        gross_all.loc[discovery_mask],
        tc_all.loc[discovery_mask],
        cost_bps,
    )
    reference_risk = float(discovery_net.std(ddof=1))
    if not np.isfinite(reference_risk) or reference_risk <= 0:
        return None
    risk_grid = tuple(sorted(reference_risk * m for m in COMMITTEE_GROWTH_RISK_GRID_MULTIPLIERS))
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
            _statistics._finite_or_none(selected.selected_risk) if selected.selected_risk is not None else None
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
    from src.strategy.books import scale_book_to_target_gross
    from src.strategy.features import FEATURE_REGISTRY, build_feature_books

    member_specs = [spec for spec in FEATURE_REGISTRY if spec.name in set(members)]
    specs_by_name = {spec.name: spec for spec in member_specs}

    member_books: dict[str, pd.DataFrame] = {}
    for name in members:
        member_spec = specs_by_name.get(name)
        if member_spec is None:
            continue
        single = build_feature_books(
            [member_spec],
            {
                col: close if col == "close" else (quote_vol if col == "quote_vol" else taker_buy_quote)
                for col in member_spec.required_columns
                if col in ("close", "quote_vol", "taker_buy_quote")
            },
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
        sharpe = float(net_returns.mean() / sd * np.sqrt(periods_per_year)) if np.isfinite(sd) and sd > 0 else None
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


def _committee_source_admission(
    member_specs: list[FeatureSpec],
    panels: Mapping[str, pd.DataFrame],
    execution_mask: pd.DataFrame,
) -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, dict[str, Any]], list[FeatureSpec]]:
    """B3 source-coverage pre-filter applied before any committee book is built.

    Every required RAW source column present in ``panels`` is audited against
    the execution mask, including an all-NaN column whose per-year coverage is
    0.0 (a gap a post-fillna feature audit cannot see). A member with ANY year
    below ``FEATURE_MIN_COVERAGE`` is excluded so it never contributes a book,
    a PnL series or a weight (fail closed, exclude-not-nan-fill).

    Returns:
        ``(source_coverage, source_excluded, admissible_specs)`` in input order.
    """
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
                    failing_sources[column] = min(failing_sources.get(column, year), year)
        source_coverage[spec.name] = per_source
        _logger.debug(
            "[DATA] stage=committee_source_coverage member=%s excluded=%s min_coverage=%.3f",
            spec.name,
            spec.name in source_excluded,
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
    return source_coverage, source_excluded, source_admissible_specs


def _committee_exclusions(admitted: list[str], source_excluded: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    excluded: list[dict[str, Any]] = [
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
    return excluded


def _block_end(edges: list[pd.Timestamp], i: int, index: pd.DatetimeIndex) -> pd.Timestamp:
    return edges[i + 1] if i + 1 < len(edges) else index[-1] + pd.Timedelta(hours=1)


def _skipped_walk_forward_blocks(
    gross_all: pd.DataFrame, edges: list[pd.Timestamp], purge: pd.Timedelta
) -> list[dict[str, str]]:
    """B6: re-derive which block edges ``purged_walk_forward`` skips, independently of its loop.

    A silently-ignored calendar gap in the concatenated wealth series is
    surfaced to the reader. Report-only, never raises.
    """
    skipped_blocks: list[dict[str, str]] = []
    for i, t0 in enumerate(edges):
        next_edge = _block_end(edges, i, gross_all.index)
        train_rows = gross_all.index < (t0 - purge)
        if int(train_rows.sum()) < WALK_FORWARD_MIN_TRAIN_BARS:
            skipped_blocks.append({"block_start": t0.isoformat(), "reason": "insufficient_train"})
            continue
        test_rows = (gross_all.index >= t0) & (gross_all.index < next_edge)
        if not bool(test_rows.any()):
            skipped_blocks.append({"block_start": t0.isoformat(), "reason": "no_test_bars"})
    return skipped_blocks


def _empty_tier_report() -> dict[str, Any]:
    return {"net_sharpe": None, "cagr": None, "mdd": None, "logret": None, "bars": 0, "blocks": []}


def _committee_block_reports(
    wf: pd.Series,
    edges: list[pd.Timestamp],
    gross_index: pd.DatetimeIndex,
    tier: str,
    total_logret: float,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for i, t0 in enumerate(edges):
        next_edge = _block_end(edges, i, gross_index)
        block_wf = wf[(wf.index >= t0) & (wf.index < next_edge)]
        if block_wf.empty:
            continue
        block_metrics = wealth_metrics(block_wf)
        _block_rho1 = block_wf.autocorr(1) if len(block_wf) > 2 else float("nan")
        blocks.append(
            {
                "block_start": t0.isoformat(),
                "bars": len(block_wf),
                "net_sharpe": _statistics._finite_or_none(block_metrics["sharpe"]),
                "cagr": _statistics._finite_or_none(block_metrics["cagr"]),
                "mdd": _statistics._finite_or_none(block_metrics["mdd"]),
                "logret": _statistics._finite_or_none(block_metrics["logret"]),
                "logret_share": (
                    float(block_metrics["logret"] / total_logret)
                    if np.isfinite(total_logret) and total_logret != 0 and np.isfinite(block_metrics["logret"])
                    else None
                ),
                "return_autocorr_lag1": (float(_block_rho1) if np.isfinite(_block_rho1) else None),
            }
        )
        _logger.debug(
            "[EVAL] stage=committee_block tier=%s block_start=%s bars=%d sharpe=%s cagr=%s mdd=%s rho1=%s",
            tier,
            t0.isoformat(),
            len(block_wf),
            block_metrics["sharpe"],
            block_metrics["cagr"],
            block_metrics["mdd"],
            _block_rho1,
        )
    return blocks


def _committee_tier_report(
    wf: pd.Series, edges: list[pd.Timestamp], gross_index: pd.DatetimeIndex, tier: str
) -> dict[str, Any]:
    metrics = wealth_metrics(wf)
    _logger.debug(
        "[EVAL] stage=committee_tier_summary tier=%s bars=%d sharpe=%s cagr=%s mdd=%s",
        tier,
        len(wf),
        metrics["sharpe"],
        metrics["cagr"],
        metrics["mdd"],
    )
    blocks = _committee_block_reports(wf, edges, gross_index, tier, metrics["logret"])
    return {
        "net_sharpe": _statistics._finite_or_none(metrics["sharpe"]),
        "cagr": _statistics._finite_or_none(metrics["cagr"]),
        "mdd": _statistics._finite_or_none(metrics["mdd"]),
        "logret": _statistics._finite_or_none(metrics["logret"]),
        "bars": len(wf),
        "blocks": blocks,
    }


def _committee_diagnostic_payload(
    admitted: list[str],
    excluded: list[dict[str, Any]],
    source_coverage: Mapping[str, Mapping[str, Mapping[int, float]]],
    edges: list[pd.Timestamp],
    skipped_blocks: list[dict[str, str]],
    sizing_mode: str,
    per_tier: dict[str, dict[str, Any]],
    growth_headroom: dict[str, Any] | None,
) -> dict[str, Any]:
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
