from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.core.params import (
    COMMITTEE_MEMBERS,
    COMMITTEE_OOS_START,
    COMMITTEE_PURGE_HOURS,
    MEASURED_EXECUTION_COST_TIERS_BPS,
)
from src.core.resources import _assert_stage_rss_budget, _StageRecorder
from src.engine.execution import mhs_ledger_pnl_multi_tier
from src.lab.mhs import statistics as _statistics
from src.lab.mhs.committee import (
    committee_block_edges_from,
    decompose_cost,
    purged_walk_forward,
    train_evidence_weights,
)
from src.lab.mhs.params import WALK_FORWARD_MIN_TRAIN_BARS
from src.lab.mhs.regime import beta_neutralize_weights
from src.strategy.books import phase_tranche_book, scale_book_to_target_gross
from src.strategy.features import (
    FEATURE_REGISTRY,
    FeatureAdmission,
    FeatureSpec,
    build_admitted_feature_books,
    build_feature_books,
    build_feature_books_by_boundary,
    feature_admission_by_boundary,
    feature_registry_panel_columns,
)

from . import diagnostics
from .committee_reports import (
    _committee_diagnostic_payload,
    _committee_exclusions,
    _committee_growth_headroom,
    _committee_source_admission,
    _committee_tier_report,
    _empty_tier_report,
    _skipped_walk_forward_blocks,
)
from .committee_reports import _committee_member_attribution as _committee_member_attribution
from .committee_reports import _committee_member_books as _committee_member_books

_logger = logging.getLogger("MhsHorizonDiagnostic")


def _stream_committee_member_nets(
    member_specs: list[FeatureSpec],
    panels: Mapping[str, pd.DataFrame],
    execution_mask: pd.DataFrame,
    decision_grid: pd.DatetimeIndex,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    bps_low: float,
    bps_high: float,
    rss_budget_bytes: int | None,
    rss_reserve_bytes: int | None,
) -> tuple[list[str], dict[str, pd.Series], dict[str, pd.Series]]:
    """Stream one member at a time in ``COMMITTEE_MEMBERS`` order, keeping only the two cost-tier nets.

    Order preserves the pre-streaming admitted/net-panel column order; each
    book is dropped immediately to bound memory residency.
    """
    specs_by_name = {spec.name: spec for spec in member_specs}
    admitted: list[str] = []
    net_low_by_name: dict[str, pd.Series] = {}
    net_high_by_name: dict[str, pd.Series] = {}
    for name in COMMITTEE_MEMBERS:
        member_spec = specs_by_name.get(name)
        if member_spec is None:
            continue
        _assert_stage_rss_budget(f"committee_member_{name}", rss_budget_bytes, rss_reserve_bytes)
        single = build_feature_books([member_spec], panels, execution_mask, decision_grid, min_symbols=8)
        if name not in single:
            continue
        book = single[name]
        (net_low, _), (net_high, _) = mhs_ledger_pnl_multi_tier(book, opens, bar_funding, [bps_low, bps_high])
        net_low_by_name[name] = net_low
        net_high_by_name[name] = net_high
        admitted.append(name)
        _logger.debug(
            "[ALGO] stage=committee_member member=%s net_low_mean=%.6f net_high_mean=%.6f",
            name,
            float(net_low.mean()),
            float(net_high.mean()),
        )
        del single, book
    return admitted, net_low_by_name, net_high_by_name


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
            root,
            start,
            end,
            grid_1h,
            aligned_symbols,
            columns=feature_registry_panel_columns(
                [spec for spec in FEATURE_REGISTRY if spec.name in set(COMMITTEE_MEMBERS)],
            ),
        )
    _assert_stage_rss_budget("committee_feature_panels", rss_budget_bytes, rss_reserve_bytes)
    member_specs = [spec for spec in FEATURE_REGISTRY if spec.name in set(COMMITTEE_MEMBERS)]
    source_coverage, source_excluded, member_specs = _committee_source_admission(
        member_specs, panels, execution_mask
    )

    decision_grid = pd.date_range(grid_1h[0], grid_1h[-1], freq="24h", tz="UTC")
    bps_low = MEASURED_EXECUTION_COST_TIERS_BPS["optimistic"]
    bps_high = MEASURED_EXECUTION_COST_TIERS_BPS["stress"]
    admitted, net_low_by_name, net_high_by_name = _stream_committee_member_nets(
        member_specs, panels, execution_mask, decision_grid, opens, bar_funding,
        bps_low, bps_high, rss_budget_bytes, rss_reserve_bytes,
    )
    excluded = _committee_exclusions(admitted, source_excluded)

    gross_all: pd.DataFrame | None = None
    tc_all: pd.DataFrame | None = None
    if admitted:
        gross_all, tc_all = decompose_cost(
            pd.DataFrame(net_low_by_name), pd.DataFrame(net_high_by_name), bps_low, bps_high
        )

    # B1: anchor the OOS block grid at COMMITTEE_OOS_START, never the raw
    # diagnostic start, so min_train_bars (~83 days) can no longer smuggle
    # pre-OOS blocks in as pseudo-OOS.
    edges = committee_block_edges_from(start, COMMITTEE_OOS_START, end)
    purge = pd.Timedelta(hours=COMMITTEE_PURGE_HOURS)
    skipped_blocks = _skipped_walk_forward_blocks(gross_all, edges, purge) if gross_all is not None else []

    per_tier: dict[str, dict[str, Any]] = {}
    for tier, cost_bps in MEASURED_EXECUTION_COST_TIERS_BPS.items():
        if gross_all is None:
            per_tier[tier] = _empty_tier_report()
            continue
        wf = purged_walk_forward(
            gross_all,
            tc_all,
            cost_bps,
            edges,
            purge,
            min_train_bars=WALK_FORWARD_MIN_TRAIN_BARS,
            sizing_mode=sizing_mode,
        )
        if telemetry is not None:
            telemetry.record(f"committee_walk_forward_{tier}")
        per_tier[tier] = _committee_tier_report(wf, edges, gross_all.index, tier)

    growth_headroom = (
        _committee_growth_headroom(gross_all, tc_all, MEASURED_EXECUTION_COST_TIERS_BPS["base"])
        if (growth_diagnostic and gross_all is not None)
        else None
    )
    return _committee_diagnostic_payload(
        admitted, excluded, source_coverage, edges, skipped_blocks, sizing_mode, per_tier, growth_headroom
    )


def _committee_boundary_admission_and_weights(
    close: pd.DataFrame,
    quote_vol: pd.DataFrame,
    taker_buy_quote: pd.DataFrame,
    execution_mask: pd.DataFrame,
    decision_grid: pd.DatetimeIndex,
    min_symbols: int,
    train_ends: Mapping[str, pd.Timestamp],
    members: tuple[str, ...] | None = None,
    *,
    evidence_weighting: bool,
) -> tuple[dict[str, FeatureAdmission], dict[str, dict[str, float]]]:
    """Per-boundary committee admission and (optionally) evidence weights, from one feature build.

    Every boundary admits members on rows strictly before its own train_end
    (INV-WALK-FORWARD-INDEPENDENCE); the returned admission is the only member
    set any book executing that boundary may use (I-FOLD-ADMISSION-PIT). With
    ``evidence_weighting`` the admission is read from the same
    ``build_feature_books_by_boundary`` call that fits the weights, so the two
    can never disagree (I-COVERAGE-PIT); without it only the admission audit
    runs and the weights mapping is empty.

    Returns:
        ``(admission_by_label, weights_by_label)``; ``weights_by_label[label]``
        keys equal ``admission_by_label[label].admitted`` when weighting, else
        ``{}`` overall.
    """
    _resolved = members or COMMITTEE_MEMBERS
    _member_specs = [spec for spec in FEATURE_REGISTRY if spec.name in set(_resolved)]
    _panels = {"close": close, "quote_vol": quote_vol, "taker_buy_quote": taker_buy_quote}
    if not evidence_weighting:
        admission_by_label = feature_admission_by_boundary(
            _member_specs,
            _panels,
            execution_mask,
            train_ends,
        )
        _distinct = {tuple(a.admitted) for a in admission_by_label.values()}
        _counts = [len(a.admitted) for a in admission_by_label.values()]
        _logger.info(
            "[ALGO] committee_admission boundaries=%d evidence_weighting=%s distinct_member_sets=%d min_admitted=%d max_admitted=%d",
            len(admission_by_label),
            bool(evidence_weighting),
            len(_distinct),
            min(_counts, default=0),
            max(_counts, default=0),
        )
        for _label, _adm in admission_by_label.items():
            _excluded = [s.name for s in _member_specs if s.name not in set(_adm.admitted)]
            _logger.debug(
                "[ALGO] committee_admission boundary=%s cutoff=%s admitted=%s excluded=%s",
                _label,
                _adm.cutoff.isoformat(),
                ",".join(_adm.admitted),
                ",".join(_excluded),
            )
        return admission_by_label, {}
    _books_by_boundary = build_feature_books_by_boundary(
        _member_specs,
        _panels,
        execution_mask,
        decision_grid,
        train_ends,
        min_symbols=min_symbols,
    )
    admission_by_label = {
        label: FeatureAdmission(
            cutoff=train_ends[label],
            admitted=tuple(books.keys()),
        )
        for label, books in _books_by_boundary.items()
    }
    close_grid = close.reindex(decision_grid).ffill()
    fwd_ret = np.log(close_grid).shift(-1) - np.log(close_grid)
    weights_by_label: dict[str, dict[str, float]] = {}
    for label, train_end in train_ends.items():
        proxies: dict[str, pd.Series] = {}
        for name, book in _books_by_boundary[label].items():
            book_grid = book.reindex(decision_grid).fillna(0.0)
            proxies[name] = (book_grid * fwd_ret).sum(axis=1)
        # The proxy is observed at the NEXT decision, not its starting row.
        train_mask = pd.Series(decision_grid, index=decision_grid).shift(-1) < train_end
        weights_by_label[label] = train_evidence_weights(proxies, train_mask) if proxies else {}
    _distinct = {tuple(a.admitted) for a in admission_by_label.values()}
    _counts = [len(a.admitted) for a in admission_by_label.values()]
    _logger.info(
        "[ALGO] committee_admission boundaries=%d evidence_weighting=%s distinct_member_sets=%d min_admitted=%d max_admitted=%d",
        len(admission_by_label),
        bool(evidence_weighting),
        len(_distinct),
        min(_counts, default=0),
        max(_counts, default=0),
    )
    for _label, _adm in admission_by_label.items():
        _excluded = [s.name for s in _member_specs if s.name not in set(_adm.admitted)]
        _logger.debug(
            "[ALGO] committee_admission boundary=%s cutoff=%s admitted=%s excluded=%s",
            _label,
            _adm.cutoff.isoformat(),
            ",".join(_adm.admitted),
            ",".join(_excluded),
        )
    return admission_by_label, weights_by_label


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

    Member books are built exactly once per feature; each boundary (fold or
    top-level OOS) then admits members and fits its own evidence weights from
    training data strictly before its own boundary, so no fold sees future
    coverage or future fits (INV-WALK-FORWARD-INDEPENDENCE).
    """
    return _committee_boundary_admission_and_weights(
        close,
        quote_vol,
        taker_buy_quote,
        execution_mask,
        decision_grid,
        min_symbols,
        train_ends,
        members,
        evidence_weighting=True,
    )[1]


def _committee_member_book_set(
    close: pd.DataFrame,
    quote_vol: pd.DataFrame,
    taker_buy_quote: pd.DataFrame,
    execution_mask: pd.DataFrame,
    decision_grid: pd.DatetimeIndex,
    min_symbols: int,
    resolved: tuple[str, ...],
    member_weights: Mapping[str, float] | None,
    coverage_cutoff: pd.Timestamp | None,
    admission: FeatureAdmission | None,
) -> dict[str, pd.DataFrame]:
    panels = {"close": close, "quote_vol": quote_vol, "taker_buy_quote": taker_buy_quote}
    if admission is not None:
        _validate_committee_admission(admission, resolved, member_weights)
        _admission_specs = [spec for spec in FEATURE_REGISTRY if spec.name in set(admission.admitted)]
        books = build_admitted_feature_books(
            _admission_specs, panels, execution_mask, decision_grid, min_symbols=min_symbols
        )
    else:
        _member_specs = [spec for spec in FEATURE_REGISTRY if spec.name in set(resolved)]
        books = build_feature_books(
            _member_specs,
            panels,
            execution_mask,
            decision_grid,
            min_symbols=min_symbols,
            coverage_cutoff=coverage_cutoff,
        )
    if not books:
        raise RuntimeError("committee_capital: no committee member admitted in this fold window")
    return books


def _validate_committee_admission(
    admission: FeatureAdmission, resolved: tuple[str, ...], member_weights: Mapping[str, float] | None
) -> None:
    from .integrity import CommitteeAdmissionIntegrityError

    _resolved_set = set(resolved) & {spec.name for spec in FEATURE_REGISTRY}
    for _name in admission.admitted:
        if _name not in _resolved_set:
            raise CommitteeAdmissionIntegrityError(
                f"committee admission {list(admission.admitted)} outside resolved members {sorted(_resolved_set)}"
            )
    if member_weights is not None and set(member_weights.keys()) != set(admission.admitted):
        raise CommitteeAdmissionIntegrityError(
            f"committee member_weights keys {sorted(member_weights.keys())} != admission {list(admission.admitted)}"
        )
    if len(admission.admitted) == 0:
        raise RuntimeError("committee_capital: no committee member admitted in this fold window")


def _combine_committee_books(
    books: Mapping[str, pd.DataFrame], member_weights: Mapping[str, float] | None
) -> pd.DataFrame:
    if member_weights is not None:
        admitted = {n: max(0.0, member_weights.get(n, 0.0)) for n in books}
        total = sum(admitted.values())
        if total > 0.0:
            return sum(admitted[n] / total * books[n] for n in books)
    return sum(books.values()) / float(len(books))


def _regime_adaptive_book(
    book: pd.DataFrame,
    close: pd.DataFrame,
    decision_grid: pd.DatetimeIndex,
    tranche_count: int,
    window: int,
) -> pd.DataFrame:
    book_grid = book.reindex(decision_grid).fillna(0.0)
    smoothed_grid = phase_tranche_book(book_grid, tranche_count)
    close_grid = close.reindex(decision_grid).ffill()
    fwd_ret = np.log(close_grid).shift(-1) - np.log(close_grid)
    proxy_return = (book_grid * fwd_ret.reindex(decision_grid)).sum(axis=1)
    trailing_rho1 = (
        proxy_return.rolling(window, min_periods=window)
        .apply(_statistics._causal_lag1_autocorr, raw=True)
        .shift(1)
    )
    use_smoothed = (trailing_rho1 < 0.0).reindex(decision_grid).fillna(False)
    adaptive_grid = book_grid.mask(use_smoothed, smoothed_grid)
    return adaptive_grid.reindex(book.index, method="ffill").fillna(0.0)


def _blend_carry_book(
    result: pd.DataFrame, carry_book: pd.DataFrame, carry_weight: float, target_gross: float | None
) -> pd.DataFrame:
    if target_gross is None:
        raise ValueError(
            "carry_book with carry_weight > 0.0 requires target_gross "
            "to be set (the diluted book has no gross to normalize against)"
        )
    if not (0.0 <= carry_weight < 1.0):
        raise ValueError(f"carry_weight must be in [0.0, 1.0), got {carry_weight}")
    unit_committee = scale_book_to_target_gross(result, 1.0)
    unit_carry = scale_book_to_target_gross(carry_book.reindex(result.index).fillna(0.0), 1.0)
    return (1.0 - carry_weight) * unit_committee + carry_weight * unit_carry


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
    *,
    admission: FeatureAdmission | None = None,
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
    admitted members. ``admission`` (production path) is the boundary's fixed member set: books
    are built for exactly those members with no in-window audit, and
    ``member_weights`` (when given) must have been fit on that same boundary --
    its keys must equal ``admission.admitted`` (I-COVERAGE-PIT). Without
    ``admission`` the members are audited in place (``coverage_cutoff``
    restricts that audit); that path is for report-only and legacy callers.

    Raises:
        ValueError: ``admission`` and ``coverage_cutoff`` both given (plus the
            existing tranche/regime/carry errors).
        CommitteeAdmissionIntegrityError: an admitted name outside the resolved
            ``members``, or ``member_weights`` keys != ``admission.admitted``.
        RuntimeError: no member admitted (existing message, unchanged).
    """
    if tranche_count < 1:
        raise ValueError(f"tranche_count must be >= 1, got {tranche_count}")
    if regime_adaptive_window is not None and regime_adaptive_window < 3:
        raise ValueError(f"regime_adaptive_window must be >= 3, got {regime_adaptive_window}")
    _resolved = members or COMMITTEE_MEMBERS
    if admission is not None and coverage_cutoff is not None:
        raise ValueError("admission and coverage_cutoff are mutually exclusive")
    _committee_books = _committee_member_book_set(
        close, quote_vol, taker_buy_quote, execution_mask, decision_grid, min_symbols,
        _resolved, member_weights, coverage_cutoff, admission,
    )
    book = _combine_committee_books(_committee_books, member_weights)
    if regime_adaptive_window is not None:
        result = _regime_adaptive_book(book, close, decision_grid, tranche_count, regime_adaptive_window)
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
        result = _blend_carry_book(result, carry_book, carry_weight, target_gross)
    if target_gross is None:
        return result
    return scale_book_to_target_gross(result, target_gross)
