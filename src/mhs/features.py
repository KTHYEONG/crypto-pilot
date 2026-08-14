"""Feature-axis alpha: registry, coverage gate, and equal-risk combination.

The architecture restructuring (docs/specs/mhs_multi_feature_alpha_architecture.md
§2) replaces the saturated horizon-axis search with a declared feature registry.
Each ``FeatureSpec`` names the raw panel columns it requires and a causal
builder; ``build_feature_books`` converts the admitted features into the SAME
dollar-neutral rank book the production stack uses, on the 24h decision grid,
and fail-closes any feature whose per-year coverage drops below its floor
inside the execution mask (the ``no_trades`` collapse, spec §0.4). This module
introduces no new weighting scheme for individual books -- every book is the
existing ``rank_weight_book`` primitive.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.mhs.books import rank_weight_book
from src.mhs.contracts import MHS_FEATURE_MIN_COVERAGE
from src.mhs.horizons import horizon_log_return, realized_vol, vol_normalized_horizon_signal


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One registered feature: required panel columns, coverage floor, builder.

    ``builder`` must be causal (look-back only) and return a DataFrame with the
    same index and columns as the ``mask`` it will be audited against. The sign
    of the trade is baked into the builder (e.g. ``rev_24h`` returns the negated
    return), so ``build_feature_books`` always ranks with sign=+1.
    """

    name: str
    required_columns: tuple[str, ...]
    min_coverage: float
    builder: Callable[[Mapping[str, pd.DataFrame]], pd.DataFrame]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.required_columns:
            raise ValueError("required_columns must not be empty")
        if not (0.0 <= self.min_coverage <= 1.0):
            raise ValueError(
                f"min_coverage must be in [0.0, 1.0], got {self.min_coverage}"
            )


def feature_coverage_audit(
    feature: pd.DataFrame, mask: pd.DataFrame,
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
    masked = feature.where(mask)
    years = sorted({ts.year for ts in feature.index})
    out: dict[int, float] = {}
    for year in years:
        year_rows = feature.index.year == year
        mask_cells = int(mask.loc[year_rows].sum().sum())
        if mask_cells == 0:
            out[year] = 0.0
            continue
        covered = int(masked.loc[year_rows].notna().sum().sum())
        out[year] = float(covered / mask_cells)
    return out


def build_feature_books(
    specs: Sequence[FeatureSpec],
    panels: Mapping[str, pd.DataFrame],
    mask: pd.DataFrame,
    decision_grid: pd.DatetimeIndex,
    min_symbols: int = 8,
) -> dict[str, pd.DataFrame]:
    """Build dollar-neutral rank books for every coverage-admitted feature.

    Each admitted feature becomes ``rank_weight_book(feature, mask, +1,
    min_symbols)`` sampled onto ``decision_grid`` and forward-held (the turnover
    discipline the measured cost tiers assume). A feature whose required columns
    are absent from ``panels`` raises ``ValueError``; a feature failing its
    ``min_coverage`` in ANY year is excluded entirely from the returned dict
    (fail closed -- never NaN, never zero-filled, never silently dropped).
    """
    if min_symbols < 2:
        raise ValueError(f"min_symbols must be >= 2, got {min_symbols}")
    books: dict[str, pd.DataFrame] = {}
    for spec in specs:
        missing = [c for c in spec.required_columns if c not in panels]
        if missing:
            raise ValueError(
                f"spec '{spec.name}' required_columns absent from panels: {missing}"
            )
        feature = spec.builder(panels)
        if not feature.index.equals(mask.index) or list(feature.columns) != list(mask.columns):
            raise ValueError(
                f"feature '{spec.name}' and mask must be identically indexed and columned"
            )
        coverage = feature_coverage_audit(feature, mask)
        if any(cov < spec.min_coverage for cov in coverage.values()):
            continue
        step = rank_weight_book(feature, mask, 1, min_symbols)
        sampled = step.reindex(decision_grid)
        book = sampled.reindex(step.index, method="ffill").fillna(0.0)
        books[spec.name] = book
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
            raise ValueError(
                f"scale_returns['{name}'] standard deviation must be positive and finite"
            )
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
    return _finite(
        -(ret1.abs().rolling(168, min_periods=168).mean() / dvol.where(dvol > 0))
    )


def _turnover_chg_builder(panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    qv = panels["quote_vol"]
    return _finite(
        qv.rolling(24, min_periods=24).mean()
        / qv.rolling(720, min_periods=720).mean().replace(0, np.nan)
    )


def _avg_trade_size_builder(panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    qv = panels["quote_vol"]
    ntr = panels["no_trades"]
    return _finite(
        qv.rolling(168, min_periods=168).mean()
        / ntr.rolling(168, min_periods=168).mean().replace(0, np.nan)
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


def _hl_range_168h_builder(panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    high = panels["high"]
    low = panels["low"]
    close = panels["close"]
    return _finite(
        -((high - low) / close.replace(0, np.nan)).rolling(168, min_periods=168).mean()
    )


# The declared feature registry (docs/specs/mhs_multi_feature_alpha_architecture.md
# §2). Each entry's sign is baked into its builder; min_coverage is the frozen
# MHS_FEATURE_MIN_COVERAGE floor. This is a declared registry -- not a
# performance-selected subset -- so the default combination is the no-selection
# full set.
MHS_FEATURE_REGISTRY: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="mom_168h",
        required_columns=("close",),
        min_coverage=MHS_FEATURE_MIN_COVERAGE,
        builder=_momentum_builder(168),
    ),
    FeatureSpec(
        name="mom_336h",
        required_columns=("close",),
        min_coverage=MHS_FEATURE_MIN_COVERAGE,
        builder=_momentum_builder(336),
    ),
    FeatureSpec(
        name="rev_24h",
        required_columns=("close",),
        min_coverage=MHS_FEATURE_MIN_COVERAGE,
        builder=_reversal_24h_builder,
    ),
    FeatureSpec(
        name="taker_imb_168h",
        required_columns=("taker_buy_quote", "quote_vol"),
        min_coverage=MHS_FEATURE_MIN_COVERAGE,
        builder=_taker_imbalance_builder(168),
    ),
    FeatureSpec(
        name="taker_imb_24h",
        required_columns=("taker_buy_quote", "quote_vol"),
        min_coverage=MHS_FEATURE_MIN_COVERAGE,
        builder=_taker_imbalance_builder(24),
    ),
    FeatureSpec(
        name="amihud",
        required_columns=("close", "quote_vol"),
        min_coverage=MHS_FEATURE_MIN_COVERAGE,
        builder=_amihud_builder,
    ),
    FeatureSpec(
        name="lowvol_168h",
        required_columns=("close",),
        min_coverage=MHS_FEATURE_MIN_COVERAGE,
        builder=_lowvol_168h_builder,
    ),
    FeatureSpec(
        name="hl_range_168h",
        required_columns=("high", "low", "close"),
        min_coverage=MHS_FEATURE_MIN_COVERAGE,
        builder=_hl_range_168h_builder,
    ),
    FeatureSpec(
        name="turnover_chg",
        required_columns=("quote_vol",),
        min_coverage=MHS_FEATURE_MIN_COVERAGE,
        builder=_turnover_chg_builder,
    ),
    FeatureSpec(
        name="avg_trade_size",
        required_columns=("quote_vol", "no_trades"),
        min_coverage=MHS_FEATURE_MIN_COVERAGE,
        builder=_avg_trade_size_builder,
    ),
)
