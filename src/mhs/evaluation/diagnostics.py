from __future__ import annotations

import gc
import glob
import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.mhs import statistics as _statistics
from src.mhs.contracts import MhsDiagnosticRequest
from src.mhs.resources import _assert_stage_rss_budget, _StageRecorder
from src.mhs.books import rank_weight_book  # noqa: F401 - re-exported for monkeypatch seams
from src.mhs.discovery import yearly_net_t_diagnostic
from src.mhs.evidence import (
    PhaseDiagnosticResult,
    effective_breadth,
    phase_diagnostic_metrics,  # noqa: F401 - re-exported for monkeypatch seams
    year_restricted_correlation,
)
from src.mhs.execution import (
    mhs_ledger_pnl,
    mhs_ledger_pnl_multi_tier,
)
from src.mhs.features import (
    FEATURE_REGISTRY,
    build_feature_books,
    feature_coverage_audit,
    feature_registry_panel_columns,
)
from src.mhs.horizons import horizon_log_return  # noqa: F401 - re-exported for monkeypatch seams
from src.mhs.panel import load_base_panel
from src.mhs.params import (
    MEASURED_EXECUTION_COST_TIERS_BPS,
)
from src.mhs.params import (
    PERIODS_PER_YEAR_1H as _PERIODS_PER_YEAR_1H,
)
from src.mhs.stability import regime_split_stability
from src.mhs.trend_sleeve import market_basket_log_price, time_series_trend_position, trend_sleeve_weights
from src.mhs.types import TREND_SLEEVE_HORIZONS_HOURS, BookSpec


def _phase_diagnostics(
    log_close: pd.DataFrame,
    eligible: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    grid_1h: pd.DatetimeIndex,
    spec: BookSpec,
) -> PhaseDiagnosticResult:
    import src.mhs.evaluation as ev
    phase_nets: dict[int, pd.Series] = {}
    signal = ev.horizon_log_return(log_close, spec.horizon_hours)
    for offset in range(spec.step_hours):
        phase_grid = grid_1h[offset :: spec.step_hours]
        sig = signal.reindex(phase_grid)
        el = eligible.reindex(phase_grid)
        weights = ev.rank_weight_book(sig, el, spec.band.sign, spec.min_symbols)
        weights_1h = weights.reindex(grid_1h, method="ffill").fillna(0.0)
        net, _turnover = ev.mhs_ledger_pnl(weights_1h, opens, bar_funding, 8.0)
        phase_nets[offset] = net
        # Each phase is independent.  Explicitly dropping its full-grid
        # target/ledger intermediates prevents allocator high-water growth on
        # multi-year, hundreds-of-symbol diagnostics.
        del sig, el, weights, weights_1h, _turnover
        gc.collect()
    return ev.phase_diagnostic_metrics(phase_nets, _PERIODS_PER_YEAR_1H)






def _trend_sleeve_diagnostic(
    log_close: pd.DataFrame,
    eligible: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    execution_mask: pd.DataFrame,
    current_book: pd.DataFrame,
    request: MhsDiagnosticRequest,
) -> dict[str, Any]:
    """SCENARIO_MHS_TREND_SLEEVE_DIAGNOSTIC_POPULATED: report-only measurements
    for the opt-in additive directional trend sleeve.

    Builds the eligible-market basket, the ensemble time-series trend position
    on a 24h decision grid, and the gross-budget-sized sleeve weights, then
    reports the sleeve's standalone net Sharpe per measured cost tier, its
    per-calendar-year net t-stat, its daily-return correlation to the deployed
    book passed in as ``current_book``, and the combined (current book + sleeve)
    book metrics. Every value is finite or an explicit ``None`` -- never NaN
    silently coerced to 0.0. This is a measurement report before configuring
    risk budgets.
    """
    grid_1h = log_close.index
    decision_grid = pd.date_range(grid_1h[0], grid_1h[-1], freq="24h", tz="UTC")
    basket = market_basket_log_price(log_close, eligible)
    position = time_series_trend_position(
        basket, TREND_SLEEVE_HORIZONS_HOURS, decision_grid,
    )
    sleeve = trend_sleeve_weights(position, execution_mask, request.trend_sleeve_gross)

    per_tier: dict[str, float | None] = {}
    combined_per_tier: dict[str, float | None] = {}
    combined = current_book.add(sleeve)
    for tier, cost_bps in MEASURED_EXECUTION_COST_TIERS_BPS.items():
        net, _ = mhs_ledger_pnl(sleeve, opens, bar_funding, cost_bps)
        per_tier[tier] = _statistics._annualized_1h_sharpe(net)
        combined_net, _ = mhs_ledger_pnl(combined, opens, bar_funding, cost_bps)
        combined_per_tier[tier] = _statistics._annualized_1h_sharpe(combined_net)

    yearly = yearly_net_t_diagnostic(
        sleeve, opens, bar_funding, (2021, 2022, 2023, 2024, 2025),
        MEASURED_EXECUTION_COST_TIERS_BPS["base"], _PERIODS_PER_YEAR_1H,
    )
    yearly_net_t = {year: (None if not np.isfinite(v) else float(v)) for year, v in yearly.items()}
    combined_yearly_raw = yearly_net_t_diagnostic(
        combined, opens, bar_funding, (2021, 2022, 2023, 2024, 2025),
        MEASURED_EXECUTION_COST_TIERS_BPS["base"], _PERIODS_PER_YEAR_1H,
    )
    combined_yearly = {
        year: (None if not np.isfinite(v) else float(v))
        for year, v in combined_yearly_raw.items()
    }
    finite_years = [v for v in combined_yearly.values() if v is not None]
    worst_year_net_t = min(finite_years) if finite_years else None

    current_net, _ = mhs_ledger_pnl(
        current_book, opens, bar_funding, MEASURED_EXECUTION_COST_TIERS_BPS["base"],
    )
    sleeve_net, _ = mhs_ledger_pnl(
        sleeve, opens, bar_funding, MEASURED_EXECUTION_COST_TIERS_BPS["base"],
    )
    current_daily = (1.0 + current_net).resample("1D").apply(lambda s: s.prod() - 1.0)
    sleeve_daily = (1.0 + sleeve_net).resample("1D").apply(lambda s: s.prod() - 1.0)
    corr = year_restricted_correlation(
        sleeve_daily, current_daily, (2021, 2022, 2023, 2024, 2025),
    )
    slow_momentum_pnl_corr = float(corr) if np.isfinite(corr) else None

    return {
        "net_sharpe_per_tier": per_tier,
        "yearly_net_t": yearly_net_t,
        "slow_momentum_pnl_corr": slow_momentum_pnl_corr,
        "combined": {
            "net_sharpe_per_tier": combined_per_tier,
            "worst_year_net_t": worst_year_net_t,
        },
    }


# Preregistered regime boundary for the multi-feature stability split.
_MULTI_FEATURE_REGIME_SPLIT = (pd.Timestamp("2024-01-01", tz="UTC"),)

_MULTI_FEATURE_PANEL_COLUMNS = (
    "close", "open", "high", "low", "quote_vol", "taker_buy_quote", "no_trades",
)


def _available_panel_columns(root: str, columns: tuple[str, ...]) -> tuple[str, ...]:
    """Inspect the first 1h parquet schema and return only the columns that exist.

    Avoids ``load_base_panel`` crashing when a column is absent; the downstream
    coverage gate fails it closed.
    """
    paths = sorted(glob.glob(os.path.join(root, "1h", "*.parquet")))
    if not paths:
        return ()
    schema = set(pq.ParquetFile(paths[0]).schema.names)
    return tuple(c for c in columns if c in schema)


def _load_feature_panels(
    root: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    grid_1h: pd.DatetimeIndex,
    aligned_symbols: list[str],
    columns: tuple[str, ...] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load the registry's raw 1h panels, NaN-filling absent columns.

    Present columns come from ``load_base_panel`` (causal survivor discovery);
    a column missing from the store becomes an all-NaN panel aligned to
    ``grid_1h`` x ``aligned_symbols``, which then fails the coverage gate --
    never a silent drop, never a crash. ``columns`` prunes the load to exactly
    the requested raw columns (e.g. ``feature_registry_panel_columns`` for the
    opt-in diagnostics), halving-to-seventhing the resident panels and parquet
    I/O; ``None`` keeps the legacy full ``_MULTI_FEATURE_PANEL_COLUMNS`` set.
    """
    requested = _MULTI_FEATURE_PANEL_COLUMNS if columns is None else columns
    available = _available_panel_columns(root, requested)
    panels: dict[str, pd.DataFrame] = {}
    if available:
        loaded = load_base_panel(
            root, "1h", available, start, end, partition="dev", min_bars=2000,
        )
        for column in available:
            panels[column] = loaded[column].reindex(index=grid_1h, columns=aligned_symbols)
    for column in requested:
        if column not in panels:
            panels[column] = pd.DataFrame(np.nan, index=grid_1h, columns=aligned_symbols)
    return panels


def _multi_feature_diagnostic(
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
) -> dict[str, Any]:
    """SCENARIO_MHS_MULTI_FEATURE_DIAGNOSTIC_REPORTS_COVERAGE_AND_STABILITY:
    report-only measurements for the opt-in multi-feature alpha axis.

    Builds every registry feature from the raw 1h panels, audits per-year
    coverage inside the execution mask (fail-closed exclusion with the failing
    year reported, never a silent drop), converts the admitted features into the
    same dollar-neutral rank books the production stack uses on the 24h decision
    grid, and reports per-admitted-feature regime-split stability, the
    equal-risk combined book's net Sharpe per measured cost tier, and the
    effective breadth of the feature-book PnL panel. Every value is finite or an
    explicit ``None`` -- never NaN silently coerced to 0.0.

    Memory-optimized streaming: the panels are column-pruned to the registry's
    required-column union and built one feature at a time, keeping only the
    small per-feature net series and a single running combined-book accumulator
    instead of every feature book simultaneously.
    """
    if panels is None:
        panels = _load_feature_panels(
            root, start, end, grid_1h, aligned_symbols,
            columns=feature_registry_panel_columns(FEATURE_REGISTRY),
        )
    _assert_stage_rss_budget("multi_feature_feature_panels", rss_budget_bytes, rss_reserve_bytes)
    decision_grid = pd.date_range(grid_1h[0], grid_1h[-1], freq="24h", tz="UTC")

    admitted: dict[str, dict[str, Any]] = {}
    excluded: dict[str, dict[str, Any]] = {}
    # Per-feature streaming state: registry order throughout, matching the
    # pre-streaming dict insertion orders so combined/combined_per_tier/breadth
    # accumulation float order is preserved exactly.
    base_net_by_name: dict[str, pd.Series] = {}
    tier_nets_by_name: dict[str, tuple[pd.Series, pd.Series, pd.Series]] = {}
    combinable_order: list[str] = []
    sd_by_name: dict[str, np.float64] = {}
    combined_acc: pd.DataFrame | None = None
    combined_count = 0

    for spec in FEATURE_REGISTRY:
        feature = spec.builder(panels)
        coverage = feature_coverage_audit(feature, execution_mask)
        failing = [
            year for year, cov in coverage.items() if cov < spec.min_coverage
        ]
        if failing:
            excluded[spec.name] = {"failing_year": min(failing)}
            continue
        single = build_feature_books(
            [spec], panels, execution_mask, decision_grid, min_symbols=8,
        )
        if spec.name not in single:
            continue
        book = single[spec.name]
        _assert_stage_rss_budget(
            f"multi_feature_member_{spec.name}", rss_budget_bytes, rss_reserve_bytes,
        )
        (net_opt, _), (net_base, _), (net_stress, _) = mhs_ledger_pnl_multi_tier(
            book, opens, bar_funding,
            [
                MEASURED_EXECUTION_COST_TIERS_BPS["optimistic"],
                MEASURED_EXECUTION_COST_TIERS_BPS["base"],
                MEASURED_EXECUTION_COST_TIERS_BPS["stress"],
            ],
        )
        stability = regime_split_stability(net_base, _MULTI_FEATURE_REGIME_SPLIT)
        admitted[spec.name] = {
            "coverage": {str(year): float(cov) for year, cov in coverage.items()},
            "regime_split_stability": {
                "window_sharpes": [
                    (label, None if not np.isfinite(value) else float(value))
                    for label, value in stability.window_sharpes
                ],
                "min_window_sharpe": (
                    None if not np.isfinite(stability.min_window_sharpe)
                    else float(stability.min_window_sharpe)
                ),
                "sign_consistent": stability.sign_consistent,
                "decay": (
                    None if not np.isfinite(stability.decay) else float(stability.decay)
                ),
            },
        }
        base_net_by_name[spec.name] = net_base
        tier_nets_by_name[spec.name] = (net_opt, net_base, net_stress)

        # A feature whose realized net PnL has zero or non-finite variance cannot
        # be risk-scaled (equal_risk_combination fails closed on it) -- drop it
        # from the combination, never let one degenerate book crash the whole
        # diagnostic. Accumulate the combined book incrementally in registry
        # order (the exact sequential-add float order of equal_risk_combination).
        cleaned = net_base.dropna()
        sd = cleaned.std(ddof=1) if len(cleaned) > 1 else np.float64(0.0)
        if np.isfinite(sd) and sd > 0:
            sd_by_name[spec.name] = sd
            combinable_order.append(spec.name)
            scaled_book = book / sd
            combined_acc = (
                scaled_book
                if combined_acc is None
                else combined_acc.add(scaled_book)
            )
            combined_count += 1
        del single, book

    # Construct the combined weight book through the equal-risk primitive, but
    # report its net Sharpe per tier from the scaled net-PnL panel: net PnL is
    # linear in the weight book (each bar's return is a weighted sum plus a
    # turnover-proportional cost), so ``mean_i(net_i / sd_i)`` equals the ledger
    # of the combined book without the numerically explosive ~1/sd gross.
    combined = None if combined_acc is None else combined_acc / combined_count
    combined_per_tier: dict[str, float | None] = {}
    if combinable_order:
        tier_index = {"optimistic": 0, "base": 1, "stress": 2}
        for tier in MEASURED_EXECUTION_COST_TIERS_BPS:
            acc: float | pd.Series = 0.0
            for name in combinable_order:
                acc = acc + tier_nets_by_name[name][tier_index[tier]] / sd_by_name[name]
            combined_net = acc / len(combinable_order)
            combined_per_tier[tier] = _statistics._annualized_1h_sharpe(combined_net)
    else:
        combined_per_tier = dict.fromkeys(MEASURED_EXECUTION_COST_TIERS_BPS)

    feature_book_effective_breadth: dict[str, float] | None = None
    if len(base_net_by_name) >= 2:
        n_eff, mean_corr = effective_breadth(pd.DataFrame(base_net_by_name).fillna(0.0))
        feature_book_effective_breadth = {"n_eff": n_eff, "mean_corr": mean_corr}

    return {
        "evaluation_protocol": "in_sample_full_period",
        "trials_explored": len(FEATURE_REGISTRY),
        "admitted": admitted,
        "excluded": excluded,
        "combined": {
            "net_sharpe_per_tier": combined_per_tier,
            "book_mean_gross": (
                None
                if combined is None
                # ``combined`` is a risk-parity blend in raw 1/sd units (sd is a
                # tiny hourly-net-pnl std, so combined's own gross is a
                # meaningless leverage figure, e.g. ~175x). Rescale by
                # n / sum(1/sd_i) so the inverse-vol weights sum to 1 -- the
                # standard risk-parity normalization -- before reporting gross,
                # matching the interpretable ~<=1.0 scale a unit-gross book has.
                else float(
                    (
                        combined
                        * combined_count
                        / sum(1.0 / sd_by_name[name] for name in combinable_order)
                    ).abs().sum(axis=1).mean()
                )
            ),
        },
        "feature_book_effective_breadth": feature_book_effective_breadth,
    }




