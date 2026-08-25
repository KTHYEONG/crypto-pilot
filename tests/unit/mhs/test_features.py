from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from dataclasses import FrozenInstanceError

from src.mhs.books import rank_weight_book
from src.mhs.features import (
    FEATURE_REGISTRY,
    FeatureSpec,
    build_feature_books,
    equal_risk_combination,
    feature_coverage_audit,
    source_coverage_audit,
)
from src.mhs.horizons import vol_normalized_horizon_signal

_SYMBOLS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")


def _signal_panel(
    n: int = 2000, seed: int = 0, start: str = "2021-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic hourly log-close panel + all-True mask on 10 symbols."""
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    log_close = pd.DataFrame(
        np.cumsum(rng.normal(0.0, 0.005, (n, len(_SYMBOLS))), axis=0),
        index=idx, columns=_SYMBOLS,
    )
    mask = pd.DataFrame(True, index=idx, columns=_SYMBOLS)
    return log_close, mask


def _momentum_spec(min_coverage: float = 0.90) -> FeatureSpec:
    return FeatureSpec(
        name="mom_168h",
        required_columns=("close",),
        min_coverage=min_coverage,
        builder=lambda panels: vol_normalized_horizon_signal(np.log(panels["close"]), 168),
    )


def _gap_spec() -> FeatureSpec:
    """Builder whose feature is NaN in every calendar year after 2021."""
    def _build(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = panels["close"]
        keep = pd.Series(close.index.year == 2021, index=close.index)
        return close.where(keep, axis=0)
    return FeatureSpec(
        name="gap_feature",
        required_columns=("close",),
        min_coverage=0.90,
        builder=_build,
    )


def test_feature_spec_validation() -> None:
    # SCENARIO_FEATURE_SPEC_VALIDATION: FeatureSpec rejects an empty name, an
    # empty required_columns tuple, and a min_coverage outside [0.0, 1.0] with
    # ValueError; a well-formed spec constructs and is frozen (attribute
    # assignment raises).
    with pytest.raises(ValueError, match="name"):
        FeatureSpec(name="", required_columns=("close",), min_coverage=0.9, builder=lambda p: p["close"])
    with pytest.raises(ValueError, match="required_columns"):
        FeatureSpec(name="x", required_columns=(), min_coverage=0.9, builder=lambda p: p["close"])
    with pytest.raises(ValueError, match="min_coverage"):
        FeatureSpec(name="x", required_columns=("close",), min_coverage=-0.1, builder=lambda p: p["close"])
    with pytest.raises(ValueError, match="min_coverage"):
        FeatureSpec(name="x", required_columns=("close",), min_coverage=1.1, builder=lambda p: p["close"])
    spec = FeatureSpec(name="x", required_columns=("close",), min_coverage=0.9, builder=lambda p: p["close"])
    assert spec.name == "x"
    assert spec.required_columns == ("close",)
    assert spec.min_coverage == 0.9
    with pytest.raises(FrozenInstanceError):
        spec.min_coverage = 0.5


def test_coverage_audit_detects_column_collapse() -> None:
    # SCENARIO_COVERAGE_AUDIT_DETECTS_COLUMN_COLLAPSE: a feature fully
    # populated in year 1 and entirely NaN in year 2 within the mask audits to
    # {y1: 1.0, y2: 0.0} -- reproducing the real no_trades collapse. A year
    # whose mask has zero true cells maps to 0.0, never nan. Mismatched
    # index/columns raise ValueError.
    idx = pd.date_range("2021-01-01", periods=2 * 24 * 365, freq="1h", tz="UTC")
    cols = ["A", "B", "C"]
    rng = np.random.default_rng(1)
    feature = pd.DataFrame(rng.normal(0.0, 1.0, (len(idx), len(cols))), index=idx, columns=cols)
    mask = pd.DataFrame(True, index=idx, columns=cols)
    feature.loc[idx[idx.year == 2022], :] = np.nan
    audit = feature_coverage_audit(feature, mask)
    assert audit[2021] == pytest.approx(1.0)
    assert audit[2022] == pytest.approx(0.0)

    # A calendar year with zero mask cells maps to 0.0 (never nan).
    empty_year_mask = mask.copy()
    empty_year_mask.loc[idx[idx.year == 2022], :] = False
    audit2 = feature_coverage_audit(feature, empty_year_mask)
    assert audit2[2022] == pytest.approx(0.0)
    assert np.isfinite(audit2[2022])

    with pytest.raises(ValueError, match="identically indexed"):
        feature_coverage_audit(feature.iloc[1:], mask)
    with pytest.raises(ValueError, match="identically indexed"):
        feature_coverage_audit(feature.rename(columns={"A": "X"}), mask)


def test_build_feature_books_excludes_low_coverage_fail_closed() -> None:
    # SCENARIO_BUILD_FEATURE_BOOKS_EXCLUDES_LOW_COVERAGE_FAIL_CLOSED: given
    # two specs where one has a year below its min_coverage and one is fully
    # covered, build_feature_books returns ONLY the fully covered feature's
    # book; the low-coverage feature is omitted entirely (fail closed). A spec
    # whose required_columns are absent from panels raises ValueError.
    log_close, mask = _signal_panel(n=2 * 24 * 365, start="2021-01-01")
    close = np.exp(log_close)
    decision_grid = pd.date_range(close.index[0], close.index[-1], freq="24h", tz="UTC")
    covered = _momentum_spec()
    gap = _gap_spec()
    books = build_feature_books(
        [gap, covered], {"close": close}, mask, decision_grid, min_symbols=8,
    )
    assert set(books) == {"mom_168h"}
    assert "gap_feature" not in books

    with pytest.raises(ValueError, match="required_columns"):
        build_feature_books(
            [_momentum_spec()], {"open": close}, mask, decision_grid, min_symbols=8,
        )


def test_build_feature_books_coverage_cutoff_ignores_post_cutoff_gap() -> None:
    # SCENARIO_BUILD_FEATURE_BOOKS_COVERAGE_CUTOFF_IGNORES_POST_CUTOFF_GAP:
    # without coverage_cutoff the post-2021 gap still excludes gap_feature
    # (fail-closed regression guard); with coverage_cutoff at the 2022 boundary
    # only year 2021 is audited (fully covered), so gap_feature is admitted --
    # and its book still spans the FULL close.index, never a truncated range.
    log_close, mask = _signal_panel(n=2 * 24 * 365, start="2021-01-01")
    close = np.exp(log_close)
    decision_grid = pd.date_range(close.index[0], close.index[-1], freq="24h", tz="UTC")
    gap = _gap_spec()
    books_no_cutoff = build_feature_books(
        [gap], {"close": close}, mask, decision_grid, min_symbols=8,
    )
    assert "gap_feature" not in books_no_cutoff
    books_with_cutoff = build_feature_books(
        [gap], {"close": close}, mask, decision_grid, min_symbols=8,
        coverage_cutoff=pd.Timestamp("2022-01-01", tz="UTC"),
    )
    assert "gap_feature" in books_with_cutoff
    assert books_with_cutoff["gap_feature"].index.equals(close.index)


def test_build_feature_books_are_dollar_neutral_on_decision_grid() -> None:
    # SCENARIO_BUILD_FEATURE_BOOKS_ARE_DOLLAR_NEUTRAL_ON_DECISION_GRID: every
    # returned book is dollar-neutral per qualifying row with row gross <= 1.0,
    # and is piecewise-constant between consecutive decision_grid stamps
    # (values held, not recomputed every bar).
    log_close, mask = _signal_panel(n=3000, start="2021-01-01")
    close = np.exp(log_close)
    decision_grid = pd.date_range(close.index[0], close.index[-1], freq="24h", tz="UTC")
    books = build_feature_books(
        [_momentum_spec()], {"close": close}, mask, decision_grid, min_symbols=8,
    )
    book = books["mom_168h"]
    assert book.index.equals(close.index)
    assert list(book.columns) == list(_SYMBOLS)
    live = mask.sum(axis=1) >= 8
    live = live & (book.index >= decision_grid[0])
    if not live.all():
        assert book[~live].abs().sum(axis=1).max() == pytest.approx(0.0)
    assert book.where(live).sum(axis=1).abs().max() < 1e-12
    assert book.where(live).abs().sum(axis=1).max() <= 1.0 + 1e-12

    # Piecewise-constant: rows strictly between consecutive decision stamps are
    # identical copies of the preceding stamp's row.
    for a, b in itertools.pairwise(decision_grid):
        between = book.loc[(book.index > a) & (book.index < b)]
        if between.empty:
            continue
        stamp_row = book.loc[a]
        expected = pd.DataFrame(
            np.tile(stamp_row.to_numpy(), (len(between), 1)),
            index=between.index, columns=between.columns,
        )
        pd.testing.assert_frame_equal(between, expected, check_dtype=False)


def _dollar_neutral_book(seed: int) -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=500, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame(rng.normal(0.0, 1.0, (len(idx), 4)), index=idx, columns=list("ABCD"))
    return rank_weight_book(raw, pd.DataFrame(True, index=idx, columns=list("ABCD")), 1, 2)


def test_equal_risk_combination_preserves_dollar_neutrality() -> None:
    # SCENARIO_EQUAL_RISK_COMBINATION_PRESERVES_DOLLAR_NEUTRALITY:
    # equal_risk_combination of two dollar-neutral books is dollar-neutral per
    # row; a book whose scale_returns has double the volatility of the other
    # receives half the weight (the ratio of their contributions equals the
    # inverse ratio of their scale standard deviations); empty books, a
    # books/scale_returns key mismatch, and a zero or non-finite scale standard
    # deviation each raise ValueError.
    book_a = _dollar_neutral_book(0)
    book_b = book_a.copy()
    rng = np.random.default_rng(2)
    scale_a = pd.Series(rng.normal(0.0, 0.01, len(book_a)), index=book_a.index)
    scale_b = pd.Series(rng.normal(0.0, 0.02, len(book_b)), index=book_b.index)
    combined = equal_risk_combination({"a": book_a, "b": book_b}, {"a": scale_a, "b": scale_b})
    assert combined.index.equals(book_a.index)
    assert list(combined.columns) == list(book_a.columns)
    assert combined.sum(axis=1).abs().max() < 1e-12

    # Equal books but double scale volatility => the higher-vol book contributes
    # exactly half the magnitude (inverse-ratio weighting).
    expected = (book_a / scale_a.std(ddof=1)) + (book_b / scale_b.std(ddof=1))
    expected = expected / 2.0
    pd.testing.assert_frame_equal(combined, expected, check_dtype=False)

    with pytest.raises(ValueError, match="must not be empty"):
        equal_risk_combination({}, {})
    with pytest.raises(ValueError, match="keys must match"):
        equal_risk_combination({"a": book_a}, {"b": scale_a})
    zero_std = pd.Series(1.0, index=book_a.index)
    with pytest.raises(ValueError, match="standard deviation"):
        equal_risk_combination({"a": book_a}, {"a": zero_std})
    nan_std = pd.Series(np.nan, index=book_a.index)
    with pytest.raises(ValueError, match="standard deviation"):
        equal_risk_combination({"a": book_a}, {"a": nan_std})
    mismatched_columns = book_b.copy()
    mismatched_columns.columns = ["X", "B", "C", "D"]
    with pytest.raises(ValueError, match="identical index and column"):
        equal_risk_combination(
            {"a": book_a, "b": mismatched_columns}, {"a": scale_a, "b": scale_b},
        )


def test_equal_risk_scale_uses_only_supplied_returns() -> None:
    # SCENARIO_EQUAL_RISK_SCALE_USES_ONLY_SUPPLIED_RETURNS: passing
    # scale_returns truncated to a training window yields weights identical to
    # passing that same truncated series while the books themselves span the
    # full period -- the scaling never reads book data outside what the caller
    # supplied (no look-ahead through the scaling path).
    book_a = _dollar_neutral_book(3)
    book_b = _dollar_neutral_book(4)
    rng = np.random.default_rng(5)
    scale_a = pd.Series(rng.normal(0.0, 0.01, len(book_a)), index=book_a.index)
    scale_b = pd.Series(rng.normal(0.0, 0.03, len(book_b)), index=book_b.index)
    train_end = scale_a.index[len(scale_a) // 2]
    scale_a_train = scale_a[scale_a.index < train_end]
    scale_b_train = scale_b[scale_b.index < train_end]

    truncated = equal_risk_combination(
        {"a": book_a, "b": book_b}, {"a": scale_a_train, "b": scale_b_train},
    )
    expected = (book_a / scale_a_train.std(ddof=1)) + (book_b / scale_b_train.std(ddof=1))
    expected = expected / 2.0
    pd.testing.assert_frame_equal(truncated, expected, check_dtype=False)

    # The scale standard deviation is computed ONLY from the supplied returns:
    # extending the books (but not the scale returns) changes nothing.
    from_full = equal_risk_combination(
        {"a": book_a, "b": book_b}, {"a": scale_a_train, "b": scale_b_train},
    )
    pd.testing.assert_frame_equal(truncated, from_full, check_dtype=False)


def test_new_registry_builders_are_causal_and_finite() -> None:
    # SCENARIO_NEW_REGISTRY_BUILDERS_ARE_CAUSAL_AND_FINITE: the four new
    # builders (flow_imb_720h, xs_mom_720h, xs_idio_mom_336h, mom3_skew_168h)
    # each produce a panel that is finite-or-NaN (never inf), whose leading
    # lookback rows are NaN rather than fabricated, and whose values at bar t
    # are unchanged when the panel is truncated after bar t (causality). Each
    # declares required_columns that exist in the loaded panels.
    new_names = ("flow_imb_720h", "xs_mom_720h", "xs_idio_mom_336h", "mom3_skew_168h")
    leading_nan = {
        "flow_imb_720h": 700,
        "xs_mom_720h": 700,
        "xs_idio_mom_336h": 660,
        "mom3_skew_168h": 150,
    }
    n = 3000
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(7)
    log_close = pd.DataFrame(
        np.cumsum(rng.normal(0.0, 0.005, (n, len(_SYMBOLS))), axis=0),
        index=idx, columns=_SYMBOLS,
    )
    panels = {
        "close": np.exp(log_close),
        "taker_buy_quote": pd.DataFrame(
            rng.uniform(0.4, 0.6, (n, len(_SYMBOLS))), index=idx, columns=_SYMBOLS,
        ),
        "quote_vol": pd.DataFrame(
            rng.uniform(100.0, 200.0, (n, len(_SYMBOLS))), index=idx, columns=_SYMBOLS,
        ),
    }
    for spec in FEATURE_REGISTRY:
        if spec.name not in new_names:
            continue
        assert all(column in panels for column in spec.required_columns)
        feature = spec.builder(panels)
        assert feature.index.equals(idx)
        assert list(feature.columns) == list(_SYMBOLS)
        assert not np.isinf(feature.to_numpy()).any()
        assert feature.iloc[: leading_nan[spec.name]].notna().sum().sum() == 0
        for t in (1000, 1500, 2000):
            truncated = {col: frame.loc[frame.index <= idx[t]] for col, frame in panels.items()}
            rebuilt = spec.builder(truncated)
            pd.testing.assert_series_equal(
                rebuilt.iloc[-1], feature.loc[idx[t]], check_dtype=False,
            )


def test_source_coverage_audit_catches_pre_fillna_gaps() -> None:
    # SCENARIO_SOURCE_COVERAGE_AUDIT_CATCHES_PRE_FILLNA_GAPS: source_coverage_audit
    # reports low coverage for a source panel whose values are missing BEFORE
    # any fillna, on a fixture mirroring the funding case where only a minority
    # of columns carry real data and the rest were zero-filled downstream -- the
    # gap the existing post-fillna feature_coverage_audit cannot see. A fully
    # populated source reports coverage 1.0 for every year.
    idx = pd.date_range("2021-01-01", periods=2 * 24 * 365, freq="1h", tz="UTC")
    cols = ["A", "B", "C"]
    mask = pd.DataFrame(True, index=idx, columns=cols)
    rng = np.random.default_rng(8)
    source = pd.DataFrame(rng.normal(0.0, 1.0, (len(idx), len(cols))), index=idx, columns=cols)
    # funding-style: in 2022 only column A carries real data; the rest were
    # zero-filled downstream before any feature audit ran.
    source.loc[idx[idx.year == 2022], ["B", "C"]] = np.nan
    filled = source.fillna(0.0)

    raw_audit = source_coverage_audit(source, mask)
    filled_audit = feature_coverage_audit(filled, mask)
    assert raw_audit[2021] == pytest.approx(1.0)
    assert raw_audit[2022] == pytest.approx(1.0 / 3.0)
    assert filled_audit[2022] == pytest.approx(1.0)

    full = pd.DataFrame(rng.normal(0.0, 1.0, (len(idx), len(cols))), index=idx, columns=cols)
    for cov in source_coverage_audit(full, mask).values():
        assert cov == pytest.approx(1.0)

    with pytest.raises(ValueError, match="identically indexed"):
        source_coverage_audit(source.iloc[1:], mask)

def test_feature_registry_panel_columns_prunes_to_required_union() -> None:
    # SCENARIOFEATURE_NAME_PANEL_COLUMN_PRUNING: feature_registry_panel_columns
    # returns the deterministic first-seen union of required_columns -- for the
    # full registry 6 columns with NO 'open' (no builder uses it), and for the
    # 6 committee members 3 columns -- so _load_feature_panels can prune its
    # parquet reads and resident panels accordingly.
    from src.mhs.types import COMMITTEE_MEMBERS
    from src.mhs.features import feature_registry_panel_columns

    registry_cols = feature_registry_panel_columns(FEATURE_REGISTRY)
    assert registry_cols == (
        "close", "taker_buy_quote", "quote_vol", "high", "low", "no_trades",
    )
    assert "open" not in registry_cols

    member_specs = [
        spec for spec in FEATURE_REGISTRY if spec.name in set(COMMITTEE_MEMBERS)
    ]
    committee_cols = feature_registry_panel_columns(member_specs)
    # Default flow_momentum: registry order puts flow_imb_168h/flow_imb_720h
    # (taker_buy_quote, quote_vol) before xs_mom_336h/xs_idio_mom_336h/
    # mom3_skew_168h (close,) -> first-seen union is (taker_buy_quote,
    # quote_vol, close).
    assert committee_cols == ("taker_buy_quote", "quote_vol", "close")
    assert all(c in registry_cols for c in committee_cols)


def test_xs_mom_builder_rank_invariance() -> None:
    # SCENARIO_MHS_COMPOUNDING_ALPHA_AXES_02: rank_weight_book of
    # _xs_mom_builder equals rank_weight_book of _momentum_builder cell-for-cell,
    # while the raw signal frames are NOT equal -- locking the measured
    # rank-invariance of the row-demeaning.
    from src.mhs.features import _xs_mom_builder, _momentum_builder

    n, ncols = 400, 12
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(99)
    log_close = pd.DataFrame(
        np.cumsum(rng.normal(0.0, 0.005, (n, ncols)), axis=0),
        index=idx, columns=list("ABCDEFGHIJKL"),
    )
    panels = {"close": np.exp(log_close)}
    mask = pd.DataFrame(True, index=idx, columns=list("ABCDEFGHIJKL"))

    xs_signal = _xs_mom_builder(168)(panels)
    mom_signal = _momentum_builder(168)(panels)

    # Raw signals are NOT equal (row-demeaning changes values)
    assert not xs_signal.equals(mom_signal)

    # But rank books are identical (rank-invariance to row-constant shift)
    xs_book = rank_weight_book(xs_signal, mask, 1, 2)
    mom_book = rank_weight_book(mom_signal, mask, 1, 2)
    assert xs_book.equals(mom_book)
