"""Feature-axis alpha: registry, coverage gate, and equal-risk combination.

This module provides a declared feature registry and builder functions.
``build_feature_books`` converts admitted features into dollar-neutral rank books
on a 24h decision grid, fail-closing any feature whose per-year coverage drops below threshold.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.mhs.books import rank_weight_book
from src.mhs.feature_admission import (
    _admission_row_vectors,
    _coverage_from_vectors,
    _is_admitted,
    feature_admission_coverage,
)
from src.mhs.horizons import horizon_log_return, realized_vol, vol_normalized_horizon_signal
from src.mhs.types import FEATURE_MIN_COVERAGE

MARKET_CLOSE_PANEL: str = "market_close"


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One registered feature: required panel columns, coverage floor, builder, warmup.

    ``builder`` must be causal (look-back only) and return a DataFrame with the
    same index and columns as the ``mask`` it will be audited against. The sign
    of the trade is baked into the builder (e.g. ``rev_24h`` returns the negated
    return), so ``build_feature_books`` always ranks with sign=+1.
    ``warmup_bars`` is the builder's deterministic lookback: the number of rows
    after a symbol's first observed required input during which the builder is
    NaN even on fully populated inputs. Admission audits exclude those cells so
    coverage measures data gaps, not window length; ``0`` (the default) excludes
    nothing.

    Raises:
        ValueError: empty ``name`` or ``required_columns``, ``min_coverage``
            outside ``[0.0, 1.0]``, or ``warmup_bars`` not a non-negative
            ``int`` (``bool`` rejected).
    """

    name: str
    required_columns: tuple[str, ...]
    min_coverage: float
    builder: Callable[[Mapping[str, pd.DataFrame]], pd.DataFrame]
    warmup_bars: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.required_columns:
            raise ValueError("required_columns must not be empty")
        if not (0.0 <= self.min_coverage <= 1.0):
            raise ValueError(f"min_coverage must be in [0.0, 1.0], got {self.min_coverage}")
        if isinstance(self.warmup_bars, bool) or not isinstance(self.warmup_bars, int):
            raise ValueError(f"warmup_bars must be a non-negative int, got {self.warmup_bars!r}")
        if self.warmup_bars < 0:
            raise ValueError(f"warmup_bars must be a non-negative int, got {self.warmup_bars}")


@dataclass(frozen=True, slots=True)
class FeatureAdmission:
    """Point-in-time member admission decided on rows strictly before ``cutoff``.

    The immutable hand-off between the boundary that admits members (and fits
    their weights) and every book that executes that boundary's mix, so that
    execution can never re-decide membership on data the boundary never saw
    (I-COVERAGE-PIT / I-FOLD-ADMISSION-PIT). Picklable for fork-pool transport.

    Attributes:
        cutoff: Exclusive tz-aware UTC audit cutoff (a fold's ``train_end`` or
            ``COMMITTEE_OOS_START`` for the top-level boundary).
        admitted: Admitted feature names in the order of the specs that were
            audited (registry order); may be empty.

    Raises:
        ValueError: ``cutoff`` not a tz-aware UTC ``pd.Timestamp``, ``admitted``
            not a tuple of non-empty unique ``str``.
    """

    cutoff: pd.Timestamp
    admitted: tuple[str, ...]

    def __post_init__(self) -> None:
        cutoff = self.cutoff
        if not isinstance(cutoff, pd.Timestamp):
            raise ValueError(f"cutoff must be a tz-aware UTC pd.Timestamp, got {type(cutoff).__name__}")
        if cutoff.tzinfo is None or str(cutoff.tzinfo) != "UTC":
            raise ValueError(f"cutoff must be a tz-aware UTC pd.Timestamp, got {cutoff}")
        admitted = self.admitted
        if not isinstance(admitted, tuple):
            raise ValueError(f"admitted must be a tuple of str, got {type(admitted).__name__}")
        seen: set[str] = set()
        for name in admitted:
            if not isinstance(name, str) or not name:
                raise ValueError(f"admitted must be non-empty str, got {name!r}")
            if name in seen:
                raise ValueError(f"admitted must contain unique names, duplicate {name!r}")
            seen.add(name)


def _coverage_audit(frame: pd.DataFrame, mask: pd.DataFrame) -> dict[int, float]:
    """Shared per-calendar-year non-null coverage of ``frame`` inside ``mask``.

    For each calendar year in the frame index: the ratio of non-null frame
    cells within the mask to the mask's true cell count. A year with zero mask
    cells maps to ``0.0`` -- never NaN, never a silent drop.
    """
    masked = frame.where(mask)
    years = sorted({ts.year for ts in frame.index})
    out: dict[int, float] = {}
    for year in years:
        year_rows = frame.index.year == year
        mask_cells = int(mask.loc[year_rows].sum().sum())
        if mask_cells == 0:
            out[year] = 0.0
            continue
        covered = int(masked.loc[year_rows].notna().sum().sum())
        out[year] = float(covered / mask_cells)
    return out


def feature_coverage_audit(
    feature: pd.DataFrame,
    mask: pd.DataFrame,
) -> dict[int, float]:
    """Per-calendar-year non-null coverage of ``feature`` inside ``mask``.

    For each calendar year in the feature index: the ratio of non-null feature
    cells within the mask to the mask's true cell count. A year with zero mask
    cells maps to ``0.0`` -- never NaN, never a silent drop. Raises
    ``ValueError`` when ``feature`` and ``mask`` are not identically indexed and
    columned.
    """
    if not feature.index.equals(mask.index) or list(feature.columns) != list(mask.columns):
        raise ValueError("feature and mask must be identically indexed and columned")
    return _coverage_audit(feature, mask)


def source_coverage_audit(
    source: pd.DataFrame,
    mask: pd.DataFrame,
) -> dict[int, float]:
    """Per-calendar-year non-null coverage of a RAW source panel before fillna.

    Mirrors ``feature_coverage_audit`` but audits the panel AT THE SOURCE,
    before any downstream ``fillna``: a column that was zero-filled after
    loading appears fully covered to a post-fillna audit while its raw source is
    mostly NaN. This is the gap the funding panel exposes (only 45 of 452
    symbols carry real funding; the rest are 0-filled and quietly rank at the
    center). Raises ``ValueError`` when ``source`` and ``mask`` are not
    identically indexed and columned.
    """
    if not source.index.equals(mask.index) or list(source.columns) != list(mask.columns):
        raise ValueError("source and mask must be identically indexed and columned")
    return _coverage_audit(source, mask)


def feature_registry_panel_columns(specs: Sequence[FeatureSpec]) -> tuple[str, ...]:
    """Deterministic first-seen union of the specs' required RAW panel columns.

    Prunes ``_load_feature_panels`` to only the columns the given specs'
    builders consume, halving-to-seventhing the resident panel footprint and
    parquet I/O of the opt-in feature diagnostics. For the full registry this is
    ``('close', 'taker_buy_quote', 'quote_vol', 'high', 'low', 'no_trades')`` --
    ``'open'`` is deliberately absent because no builder uses it; for the 6
    committee members it is ``('taker_buy_quote', 'quote_vol', 'close')``.
    """
    columns: list[str] = []
    for spec in specs:
        for column in spec.required_columns:
            if column not in columns:
                columns.append(column)
    return tuple(columns)


def _build_single_book(
    feature: pd.DataFrame,
    mask: pd.DataFrame,
    decision_grid: pd.DatetimeIndex,
    min_symbols: int,
) -> pd.DataFrame:
    step = rank_weight_book(feature, mask, 1, min_symbols)
    sampled = step.reindex(decision_grid)
    return sampled.reindex(step.index, method="ffill").fillna(0.0)


def build_feature_books(
    specs: Sequence[FeatureSpec],
    panels: Mapping[str, pd.DataFrame],
    mask: pd.DataFrame,
    decision_grid: pd.DatetimeIndex,
    min_symbols: int = 8,
    coverage_cutoff: pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    """Build dollar-neutral rank books for every coverage-admitted feature.

    Each admitted feature becomes ``rank_weight_book(feature, mask, +1,
    min_symbols)`` sampled onto ``decision_grid`` and forward-held (the turnover
    discipline the measured cost tiers assume). A feature whose required columns
    are absent from ``panels`` raises ``ValueError``; a feature failing its
    ``min_coverage`` in ANY year is excluded entirely from the returned dict
    (fail closed -- never NaN, never zero-filled, never silently dropped).
    ``coverage_cutoff``, when given, restricts the admission audit to rows
    before it -- a caller fitting or deploying a fixed PIT boundary's weight mix
    must never have a member's availability decided by coverage in years that
    boundary never sees (I-COVERAGE-PIT). Admission uses
    `feature_admission_coverage` (warmup excluded) and fails closed on an empty audit.
    """
    if min_symbols < 2:
        raise ValueError(f"min_symbols must be >= 2, got {min_symbols}")
    books: dict[str, pd.DataFrame] = {}
    for spec in specs:
        missing = [c for c in spec.required_columns if c not in panels]
        if missing:
            raise ValueError(f"spec '{spec.name}' required_columns absent from panels: {missing}")
        feature = spec.builder(panels)
        if not feature.index.equals(mask.index) or list(feature.columns) != list(mask.columns):
            raise ValueError(f"feature '{spec.name}' and mask must be identically indexed and columned")
        for column in spec.required_columns:
            panel = panels[column]
            if not panel.index.equals(mask.index) or list(panel.columns) != list(mask.columns):
                raise ValueError(f"required panel '{column}' and mask must be identically indexed and columned")
        coverage = feature_admission_coverage(spec, feature, panels, mask, coverage_cutoff)
        if not _is_admitted(spec, coverage):
            continue
        books[spec.name] = _build_single_book(feature, mask, decision_grid, min_symbols)
    return books


def feature_admission_by_boundary(
    specs: Sequence[FeatureSpec],
    panels: Mapping[str, pd.DataFrame],
    mask: pd.DataFrame,
    train_ends: Mapping[str, pd.Timestamp],
) -> dict[str, FeatureAdmission]:
    """Admit features per boundary without building rank books.

    Same admission rule as ``build_feature_books_by_boundary`` (identical
    ``admitted`` tuple for identical inputs), for callers that need only the
    member set. Each builder runs exactly once; specs are processed one at a
    time so at most one built feature and its auditable mask are resident.

    Returns:
        ``{label: FeatureAdmission(cutoff=train_ends[label], admitted=...)}``.
    Raises:
        ValueError: as ``feature_admission_coverage``, or a builder output not
            aligned with ``mask``.
    """
    labels = list(train_ends.keys())
    boundary_rows = {label: np.asarray(mask.index < train_ends[label]) for label in labels}
    admitted_per_label: dict[str, list[str]] = {label: [] for label in labels}
    for spec in specs:
        missing = [c for c in spec.required_columns if c not in panels]
        if missing:
            raise ValueError(f"spec '{spec.name}' required_columns absent from panels: {missing}")
        for column in spec.required_columns:
            panel = panels[column]
            if not panel.index.equals(mask.index) or list(panel.columns) != list(mask.columns):
                raise ValueError(f"required panel '{column}' and mask must be identically indexed and columned")
        feature = spec.builder(panels)
        if not feature.index.equals(mask.index) or list(feature.columns) != list(mask.columns):
            raise ValueError(f"feature '{spec.name}' and mask must be identically indexed and columned")
        mask_per_row, auditable_per_row, covered_per_row, years = _admission_row_vectors(
            spec,
            feature,
            panels,
            mask,
        )
        del feature
        for label in labels:
            coverage = _coverage_from_vectors(
                mask_per_row[boundary_rows[label]],
                auditable_per_row[boundary_rows[label]],
                covered_per_row[boundary_rows[label]],
                years[boundary_rows[label]],
            )
            if _is_admitted(spec, coverage):
                admitted_per_label[label].append(spec.name)
    return {
        label: FeatureAdmission(cutoff=train_ends[label], admitted=tuple(admitted_per_label[label])) for label in labels
    }


def build_feature_books_by_boundary(
    specs: Sequence[FeatureSpec],
    panels: Mapping[str, pd.DataFrame],
    mask: pd.DataFrame,
    decision_grid: pd.DatetimeIndex,
    train_ends: Mapping[str, pd.Timestamp],
    min_symbols: int = 8,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Build once, admit per boundary (INV-WALK-FORWARD-INDEPENDENCE).

    Each feature builder runs exactly once over the full panels; admission is
    then audited separately per boundary using only rows strictly before that
    boundary's ``train_end``, so later coverage can never decide an earlier
    boundary's member set. Boundaries admitting an identical feature set share
    the same immutable book objects (no per-fold wide-panel recomputation, no
    duplication: PERF-BOUNDARY-FEATURE-ONCE). Admission uses
    `feature_admission_coverage` (warmup excluded) and fails closed on an empty audit.
    """
    if min_symbols < 2:
        raise ValueError(f"min_symbols must be >= 2, got {min_symbols}")
    labels = list(train_ends.keys())
    boundary_rows = {label: np.asarray(mask.index < train_ends[label]) for label in labels}
    books: dict[str, dict[str, pd.DataFrame]] = {label: {} for label in labels}
    for spec in specs:
        missing = [c for c in spec.required_columns if c not in panels]
        if missing:
            raise ValueError(f"spec '{spec.name}' required_columns absent from panels: {missing}")
        for column in spec.required_columns:
            panel = panels[column]
            if not panel.index.equals(mask.index) or list(panel.columns) != list(mask.columns):
                raise ValueError(f"required panel '{column}' and mask must be identically indexed and columned")
        feature = spec.builder(panels)
        if not feature.index.equals(mask.index) or list(feature.columns) != list(mask.columns):
            raise ValueError(f"feature '{spec.name}' and mask must be identically indexed and columned")
        vectors = _admission_row_vectors(spec, feature, panels, mask)
        admitted_labels = [
            label for label in labels
            if _is_admitted(spec, _coverage_from_vectors(
                *(vector[boundary_rows[label]] for vector in vectors),
            ))
        ]
        if admitted_labels:
            book = _build_single_book(feature, mask, decision_grid, min_symbols)
            for label in admitted_labels:
                books[label][spec.name] = book
        del feature
    admitted: dict[tuple[str, ...], dict[str, pd.DataFrame]] = {}
    for label in labels:
        key = tuple(books[label])
        if key not in admitted:
            admitted[key] = books[label]
        books[label] = admitted[key]
    return books


def build_admitted_feature_books(
    specs: Sequence[FeatureSpec],
    panels: Mapping[str, pd.DataFrame],
    mask: pd.DataFrame,
    decision_grid: pd.DatetimeIndex,
    min_symbols: int = 8,
) -> dict[str, pd.DataFrame]:
    """Build forward-held rank books for features admitted upstream, with no audit.

    The caller owns admission (a ``FeatureAdmission`` decided on rows strictly
    before its boundary); auditing here would re-decide membership on the
    execution window's own data, which is the look-ahead this function exists
    to prevent. Books are produced exactly as ``build_feature_books`` produces
    an admitted feature's book, in ``specs`` order.

    Raises:
        ValueError: ``min_symbols < 2``, a required column absent from
            ``panels``, or a builder output not aligned with ``mask``.
    """
    if min_symbols < 2:
        raise ValueError(f"min_symbols must be >= 2, got {min_symbols}")
    books: dict[str, pd.DataFrame] = {}
    for spec in specs:
        missing = [c for c in spec.required_columns if c not in panels]
        if missing:
            raise ValueError(f"spec '{spec.name}' required_columns absent from panels: {missing}")
        feature = spec.builder(panels)
        if not feature.index.equals(mask.index) or list(feature.columns) != list(mask.columns):
            raise ValueError(f"feature '{spec.name}' and mask must be identically indexed and columned")
        books[spec.name] = _build_single_book(feature, mask, decision_grid, min_symbols)
    return books


def equal_risk_combination(
    books: Mapping[str, pd.DataFrame],
    scale_returns: Mapping[str, pd.Series],
) -> pd.DataFrame:
    """Scale each dollar-neutral book to its own realized risk and average.

    Each book is divided by the standard deviation of its OWN ``scale_returns``
    series (which the caller builds from the training window only -- this
    function never slices by date and never looks at data outside what it is
    handed), then the scaled books are averaged. Scaling and averaging preserve
    dollar neutrality: a scaled dollar-neutral book stays dollar-neutral, and
    the mean of dollar-neutral books is dollar-neutral. Raises ``ValueError`` on
    empty ``books``, a ``books``/``scale_returns`` key mismatch, non-identical
    book index/columns, or a non-positive/non-finite scale standard deviation.
    """
    if not books:
        raise ValueError("books must not be empty")
    if set(books) != set(scale_returns):
        raise ValueError("books and scale_returns keys must match")
    items = list(books.items())
    first_book = items[0][1]
    for _, other in items[1:]:
        if not first_book.index.equals(other.index) or list(first_book.columns) != list(other.columns):
            raise ValueError("all books must share an identical index and column order")
    scaled: dict[str, pd.DataFrame] = {}
    for name, book in items:
        series = scale_returns[name].dropna()
        sd = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        if not (np.isfinite(sd) and sd > 0):
            raise ValueError(f"scale_returns['{name}'] standard deviation must be positive and finite")
        scaled[name] = book / sd
    total = scaled[items[0][0]].copy()
    for name, _ in items[1:]:
        total = total.add(scaled[name])
    return total / len(items)


def _finite(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan)


def _momentum_builder(horizon_bars: int) -> Callable[[Mapping[str, pd.DataFrame]], pd.DataFrame]:
    def _build(panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return _finite(vol_normalized_horizon_signal(np.log(panels["close"]), horizon_bars))

    return _build


def _reversal_24h_builder(panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    return _finite(-horizon_log_return(np.log(panels["close"]), 24))


def _lowvol_168h_builder(panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    return _finite(-realized_vol(np.log(panels["close"]), 168))


def _amihud_builder(panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    log_close = np.log(panels["close"])
    ret1 = log_close.diff()
    dvol = panels["quote_vol"].rolling(24, min_periods=24).mean()
    return _finite(-(ret1.abs().rolling(168, min_periods=168).mean() / dvol.where(dvol > 0)))


def _turnover_chg_builder(panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    qv = panels["quote_vol"]
    return _finite(qv.rolling(24, min_periods=24).mean() / qv.rolling(720, min_periods=720).mean().replace(0, np.nan))


def _avg_trade_size_builder(panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    qv = panels["quote_vol"]
    ntr = panels["no_trades"]
    return _finite(
        qv.rolling(168, min_periods=168).mean() / ntr.rolling(168, min_periods=168).mean().replace(0, np.nan)
    )


def _taker_imbalance_builder(horizon_bars: int) -> Callable[[Mapping[str, pd.DataFrame]], pd.DataFrame]:
    def _build(panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        tbq = panels["taker_buy_quote"]
        qv = panels["quote_vol"]
        return _finite(
            tbq.rolling(horizon_bars, min_periods=horizon_bars).sum()
            / qv.rolling(horizon_bars, min_periods=horizon_bars).sum().replace(0, np.nan)
            - 0.5
        )

    return _build


def _xs_mom_builder(horizon_bars: int) -> Callable[[Mapping[str, pd.DataFrame]], pd.DataFrame]:
    """Cross-sectional momentum: vol-normalized horizon return, row-demeaned.

    Row-demeaning subtracts the cross-sectional mean from each row.  Because
    ``rank_weight_book`` is rank-invariant to constant row shifts, the resulting
    rank book is identical to the raw vol-normalized momentum book (``mom_*``
    family).  ``_xs_idio_mom_builder`` is the builder that actually removes the
    market component via a causal rolling beta residual.
    """

    def _build(panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        signal = vol_normalized_horizon_signal(np.log(panels["close"]), horizon_bars)
        return _finite(signal.sub(signal.mean(axis=1), axis=0))

    return _build


def _xs_idio_mom_builder(
    horizon_bars: int,
    beta_bars: int = 336,
) -> Callable[[Mapping[str, pd.DataFrame]], pd.DataFrame]:
    """Idiosyncratic momentum: market-beta-removed, vol-normalized horizon return.

    Each symbol's horizon return is regressed on the cross-sectional market return via a causal
    rolling beta (rolling moments over ``beta_bars``) and the residual is scaled by its own rolling
    volatility. The market return is the mean horizon return over ``panels[MARKET_CLOSE_PANEL]``
    when supplied, otherwise over the panel's own ``close`` columns. A caller whose ``close``
    columns were chosen with hindsight (e.g. "ever selected over the sample") must supply the
    contemporaneous census plane, because a hindsight-selected cross-section leaks future
    membership into every historical market estimate.
    """

    def _build(panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        log_close = np.log(panels["close"])
        raw = horizon_log_return(log_close, horizon_bars)
        if MARKET_CLOSE_PANEL in panels:
            market_close = panels[MARKET_CLOSE_PANEL]
            if not market_close.index.equals(panels["close"].index):
                raise ValueError("market close panel must share the close panel index exactly")
            market = horizon_log_return(np.log(market_close), horizon_bars).mean(axis=1)
        else:
            market = raw.mean(axis=1)
        mean_r = raw.rolling(beta_bars, min_periods=beta_bars).mean()
        mean_m = market.rolling(beta_bars, min_periods=beta_bars).mean()
        mean_rm = raw.mul(market, axis=0).rolling(beta_bars, min_periods=beta_bars).mean()
        mean_m2 = market.pow(2).rolling(beta_bars, min_periods=beta_bars).mean()
        var_m = mean_m2 - mean_m.pow(2)
        cov_rm = mean_rm - mean_r.mul(mean_m, axis=0)
        beta = cov_rm.div(var_m.replace(0, np.nan), axis=0)
        residual = raw - beta.mul(market, axis=0)
        residual_vol = residual.rolling(horizon_bars, min_periods=horizon_bars).std(ddof=1) * np.sqrt(horizon_bars)
        return _finite(residual.div(residual_vol.replace(0, np.nan)))

    return _build


def _mom3_skew_builder(horizon_bars: int) -> Callable[[Mapping[str, pd.DataFrame]], pd.DataFrame]:
    """Negative-skewness premium: minus the rolling return skewness.

    Investors pay for positive skew (lottery preference), so the measured
    premium is on NEGATIVE skew -- the builder returns ``-skew`` and the rank
    book goes long the most negatively skewed symbols.
    """

    def _build(panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        log_close = np.log(panels["close"])
        ret1 = log_close.diff()
        return _finite(-ret1.rolling(horizon_bars, min_periods=horizon_bars).skew())

    return _build


def _hl_range_168h_builder(panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    high = panels["high"]
    low = panels["low"]
    close = panels["close"]
    return _finite(-((high - low) / close.replace(0, np.nan)).rolling(168, min_periods=168).mean())


# The declared feature registry. Each entry's sign is baked into its builder;
# min_coverage defaults to the frozen FEATURE_MIN_COVERAGE floor.
FEATURE_REGISTRY: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="mom_168h",
        required_columns=("close",),
        min_coverage=FEATURE_MIN_COVERAGE,
        builder=_momentum_builder(168),
        warmup_bars=168,
    ),
    FeatureSpec(
        name="mom_336h",
        required_columns=("close",),
        min_coverage=FEATURE_MIN_COVERAGE,
        builder=_momentum_builder(336),
        warmup_bars=336,
    ),
    FeatureSpec(
        name="rev_24h",
        required_columns=("close",),
        min_coverage=FEATURE_MIN_COVERAGE,
        builder=_reversal_24h_builder,
        warmup_bars=24,
    ),
    FeatureSpec(
        name="taker_imb_168h",
        required_columns=("taker_buy_quote", "quote_vol"),
        min_coverage=FEATURE_MIN_COVERAGE,
        builder=_taker_imbalance_builder(168),
        warmup_bars=168 - 1,
    ),
    FeatureSpec(
        name="taker_imb_24h",
        required_columns=("taker_buy_quote", "quote_vol"),
        min_coverage=FEATURE_MIN_COVERAGE,
        builder=_taker_imbalance_builder(24),
        warmup_bars=24 - 1,
    ),
    FeatureSpec(
        name="amihud",
        required_columns=("close", "quote_vol"),
        min_coverage=FEATURE_MIN_COVERAGE,
        builder=_amihud_builder,
        warmup_bars=168,
    ),
    FeatureSpec(
        name="lowvol_168h",
        required_columns=("close",),
        min_coverage=FEATURE_MIN_COVERAGE,
        builder=_lowvol_168h_builder,
        warmup_bars=168,
    ),
    FeatureSpec(
        name="hl_range_168h",
        required_columns=("high", "low", "close"),
        min_coverage=FEATURE_MIN_COVERAGE,
        builder=_hl_range_168h_builder,
        warmup_bars=168 - 1,
    ),
    FeatureSpec(
        name="turnover_chg",
        required_columns=("quote_vol",),
        min_coverage=FEATURE_MIN_COVERAGE,
        builder=_turnover_chg_builder,
        warmup_bars=720 - 1,
    ),
    FeatureSpec(
        name="avg_trade_size",
        required_columns=("quote_vol", "no_trades"),
        min_coverage=FEATURE_MIN_COVERAGE,
        builder=_avg_trade_size_builder,
        warmup_bars=168 - 1,
    ),
    # Committee-family registry entries (flow imbalance, cross-sectional momentum, skew)
    FeatureSpec(
        name="flow_imb_168h",
        required_columns=("taker_buy_quote", "quote_vol"),
        min_coverage=FEATURE_MIN_COVERAGE,
        builder=_taker_imbalance_builder(168),
        warmup_bars=168 - 1,
    ),
    FeatureSpec(
        name="flow_imb_720h",
        required_columns=("taker_buy_quote", "quote_vol"),
        min_coverage=FEATURE_MIN_COVERAGE,
        builder=_taker_imbalance_builder(720),
        warmup_bars=720 - 1,
    ),
    FeatureSpec(
        name="xs_mom_336h",
        required_columns=("close",),
        min_coverage=FEATURE_MIN_COVERAGE,
        builder=_xs_mom_builder(336),
        warmup_bars=336,
    ),
    FeatureSpec(
        name="xs_mom_720h",
        required_columns=("close",),
        min_coverage=FEATURE_MIN_COVERAGE,
        builder=_xs_mom_builder(720),
        warmup_bars=720,
    ),
    FeatureSpec(
        name="xs_idio_mom_336h",
        required_columns=("close",),
        min_coverage=FEATURE_MIN_COVERAGE,
        builder=_xs_idio_mom_builder(336),
        warmup_bars=336 + 335 + 335,
    ),
    FeatureSpec(
        name="mom3_skew_168h",
        required_columns=("close",),
        min_coverage=FEATURE_MIN_COVERAGE,
        builder=_mom3_skew_builder(168),
        warmup_bars=168,
    ),
)
