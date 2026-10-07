from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from dataclasses import FrozenInstanceError

from src.mhs.books import rank_weight_book
from src.mhs.features import (
    FEATURE_REGISTRY,
    FeatureAdmission,
    FeatureSpec,
    build_admitted_feature_books,
    build_feature_books,
    build_feature_books_by_boundary,
    equal_risk_combination,
    feature_admission_by_boundary,
    feature_admission_coverage,
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


def test_boundary_feature_admission_ignores_later_coverage() -> None:
    import pandas as pd
    from src.mhs.features import FeatureSpec, build_feature_books_by_boundary
    idx = pd.date_range('2024-01-01', periods=12, freq='1h', tz='UTC')
    mask = pd.DataFrame({'A': True, 'B': True}, index=idx)
    base = pd.DataFrame({'A': range(12), 'B': range(12)}, index=idx, dtype=float)
    calls = []
    def build(panels):
        calls.append(1)
        return panels['close']
    spec = FeatureSpec('x', ('close',), 1.0, build)
    ends = {'early': idx[6], 'late': idx[-1]+pd.Timedelta(hours=1)}
    first = build_feature_books_by_boundary((spec,), {'close': base}, mask, idx, ends, min_symbols=2)
    assert len(calls) == 1
    changed = base.copy()
    changed.loc[idx[7]:, 'A'] = float('nan')
    second = build_feature_books_by_boundary((spec,), {'close': changed}, mask, idx, ends, min_symbols=2)
    assert len(calls) == 2
    pd.testing.assert_frame_equal(first['early']['x'], second['early']['x'])


def test_boundary_books_reject_bad_inputs() -> None:
    import pandas as pd
    import pytest
    from src.mhs.features import FeatureSpec, build_feature_books_by_boundary
    idx = pd.date_range('2024-01-01', periods=4, freq='1h', tz='UTC')
    mask = pd.DataFrame({'A': True}, index=idx)
    base = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0]}, index=idx)
    spec = FeatureSpec('x', ('close',), 1.0, lambda panels: panels['close'])
    with pytest.raises(ValueError, match='min_symbols'):
        build_feature_books_by_boundary((spec,), {'close': base}, mask, idx, {'b': idx[0]}, min_symbols=1)
    with pytest.raises(ValueError, match='required_columns'):
        build_feature_books_by_boundary((spec,), {}, mask, idx, {'b': idx[0]})
    shifted = base.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match='identically indexed'):
        build_feature_books_by_boundary((spec,), {'close': shifted}, mask, idx, {'b': idx[0]})


def test_boundary_books_rank_each_admitted_feature_once(monkeypatch) -> None:
    import pandas as pd
    import src.mhs.features as features_module
    from src.mhs.books import rank_weight_book
    from src.mhs.features import FeatureSpec, build_feature_books_by_boundary
    idx = pd.date_range('2024-01-01', periods=12, freq='1h', tz='UTC')
    mask = pd.DataFrame({'A': True, 'B': True, 'C': True}, index=idx)
    base = pd.DataFrame({'A': range(12), 'B': range(12, 0, -1), 'C': [1.0, 3.0] * 6}, index=idx, dtype=float)
    calls = []

    def counting(signal, eligible, sign, min_symbols):
        calls.append(sign)
        return rank_weight_book(signal, eligible, sign, min_symbols)

    monkeypatch.setattr(features_module, 'rank_weight_book', counting)
    specs = (
        FeatureSpec('x', ('close',), 0.5, lambda p: p['close']),
        FeatureSpec('y', ('close',), 0.5, lambda p: -p['close']),
    )
    ends = {f'b{i}': idx[3 + i] for i in range(5)}
    # When
    books = build_feature_books_by_boundary(specs, {'close': base}, mask, idx, ends, min_symbols=2)
    # Then: one rank pass per feature, not per boundary x feature
    assert len(calls) == 2
    monkeypatch.setattr(features_module, 'rank_weight_book', rank_weight_book)
    for label in ends:
        for spec in specs:
            step = rank_weight_book(spec.builder({'close': base}), mask, 1, 2)
            expected = step.reindex(idx).reindex(step.index, method='ffill').fillna(0.0)
            pd.testing.assert_frame_equal(books[label][spec.name], expected)


def _idio_close_panel(n: int = 200, seed: int = 11) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    walks = np.cumsum(rng.normal(0.0, 0.005, (n, 6)), axis=0)
    return pd.DataFrame(
        np.exp(walks) * 100.0, index=idx, columns=["A", "B", "C", "D", "E", "F"],
    )


def test_idio_mom_market_proxy_ignores_hindsight_selection() -> None:
    from src.mhs.features import MARKET_CLOSE_PANEL, _xs_idio_mom_builder

    close = _idio_close_panel()
    builder = _xs_idio_mom_builder(5, beta_bars=10)
    narrow = builder({"close": close[["A", "B", "C"]], MARKET_CLOSE_PANEL: close})
    wide = builder({"close": close[["A", "B", "C", "D"]], MARKET_CLOSE_PANEL: close})
    pd.testing.assert_frame_equal(narrow[["A", "B", "C"]], wide[["A", "B", "C"]])


def test_idio_mom_future_perturbation_leaves_past_unchanged() -> None:
    import numpy as np

    from src.mhs.features import MARKET_CLOSE_PANEL, _xs_idio_mom_builder

    close = _idio_close_panel()
    builder = _xs_idio_mom_builder(5, beta_bars=10)
    base = builder({"close": close, MARKET_CLOSE_PANEL: close})
    shocked = close.copy()
    rng = np.random.default_rng(3)
    shocked.iloc[100:] = shocked.iloc[100:] * rng.uniform(0.5, 1.5, shocked.iloc[100:].shape)
    rebuilt = builder({"close": shocked, MARKET_CLOSE_PANEL: shocked})
    pd.testing.assert_frame_equal(rebuilt.iloc[:100], base.iloc[:100])


def test_idio_mom_delisted_names_drop_out_of_market_mean() -> None:
    from src.mhs.features import MARKET_CLOSE_PANEL, _xs_idio_mom_builder

    close = _idio_close_panel()
    builder = _xs_idio_mom_builder(5, beta_bars=10)
    listed = close.copy()
    listed.loc[listed.index[100:], "F"] = float("nan")
    full = builder({"close": close, MARKET_CLOSE_PANEL: listed})
    dropped = builder({"close": close, MARKET_CLOSE_PANEL: listed.drop(columns=["F"])})
    shared = ["A", "B", "C", "D", "E"]
    pd.testing.assert_frame_equal(full[shared].iloc[120:], dropped[shared].iloc[120:])


def test_idio_mom_without_market_plane_keeps_legacy_output() -> None:
    from src.mhs.features import MARKET_CLOSE_PANEL, _xs_idio_mom_builder

    close = _idio_close_panel()
    builder = _xs_idio_mom_builder(5, beta_bars=10)
    legacy = builder({"close": close})
    explicit = builder({"close": close, MARKET_CLOSE_PANEL: close})
    pd.testing.assert_frame_equal(legacy, explicit)


def test_idio_mom_misaligned_market_plane_rejected() -> None:
    import pandas as pd

    from src.mhs.features import MARKET_CLOSE_PANEL, _xs_idio_mom_builder

    close = _idio_close_panel()
    builder = _xs_idio_mom_builder(5, beta_bars=10)
    shifted = close.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="index"):
        builder({"close": close, MARKET_CLOSE_PANEL: shifted})


# --- Spec 28 (I-FOLD-ADMISSION-PIT) invariant scenarios below. ---

def _dense_registry_panels(
    n: int = 3000, n_symbols: int = 8, seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Dense synthetic panels covering every registry required column."""
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    cols = [f"S{i}" for i in range(n_symbols)]
    rng = np.random.default_rng(seed)
    log_close = pd.DataFrame(
        np.cumsum(rng.normal(0.0, 0.005, (n, n_symbols)), axis=0),
        index=idx, columns=cols,
    )
    close = np.exp(log_close) * 100.0
    return {
        "close": close,
        "taker_buy_quote": pd.DataFrame(
            rng.uniform(100.0, 200.0, (n, n_symbols)), index=idx, columns=cols,
        ),
        "quote_vol": pd.DataFrame(
            rng.uniform(1000.0, 2000.0, (n, n_symbols)), index=idx, columns=cols,
        ),
        "high": close * 1.001,
        "low": close * 0.999,
        "no_trades": pd.DataFrame(
            rng.integers(10, 100, (n, n_symbols)).astype(float),
            index=idx, columns=cols,
        ),
    }


def test_registry_warmup_declarations_match_measured_offsets() -> None:
    # SPEC28_REGISTRY_WARMUP_OFFSETS: on a dense panel every registry builder
    # is NaN for rows < warmup_bars on every symbol and non-NaN at row
    # warmup_bars; when symbol 0 lists at row 1500 its first valid row is
    # exactly 1500 + warmup_bars.
    panels = _dense_registry_panels()
    gapped = {col: frame.copy() for col, frame in panels.items()}
    for frame in gapped.values():
        frame.iloc[:1500, 0] = np.nan
    sym0 = panels["close"].columns[0]
    assert len(panels["close"]) >= 3000
    assert len(panels["close"].columns) >= 8
    for spec in FEATURE_REGISTRY:
        warmup = spec.warmup_bars
        feature = spec.builder(panels)
        if warmup > 0:
            assert feature.iloc[:warmup].notna().sum().sum() == 0, spec.name
        assert bool(feature.iloc[warmup].notna().all()), spec.name
        relisted = spec.builder(gapped)
        first_valid = relisted[sym0].first_valid_index()
        assert first_valid is not None, spec.name
        assert relisted.index.get_loc(first_valid) == 1500 + warmup, spec.name


def test_feature_spec_rejects_invalid_warmup() -> None:
    # SPEC28_SPEC_REJECTS_INVALID_WARMUP: warmup_bars of -1, True or 1.5
    # raises ValueError; the default is 0.
    builder = lambda panels: panels["close"]  # noqa: E731
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError, match="warmup_bars"):
            FeatureSpec(
                name="x", required_columns=("close",), min_coverage=0.9,
                builder=builder, warmup_bars=bad,
            )
    assert FeatureSpec(
        name="x", required_columns=("close",), min_coverage=0.9,
        builder=builder,
    ).warmup_bars == 0


def test_feature_admission_rejects_invalid_fields() -> None:
    # SPEC28_ADMISSION_REJECTS_INVALID_FIELDS: a naive cutoff, a non-UTC
    # cutoff, a list instead of a tuple, or duplicate/empty names raise
    # ValueError; a valid instance round-trips through pickle unchanged.
    import pickle
    from datetime import timedelta, timezone

    utc = pd.Timestamp("2023-01-01", tz="UTC")
    with pytest.raises(ValueError, match="cutoff"):
        FeatureAdmission(pd.Timestamp("2023-01-01"), ("a",))
    with pytest.raises(ValueError, match="cutoff"):
        FeatureAdmission(
            pd.Timestamp("2023-01-01", tz=timezone(timedelta(hours=9))), ("a",),
        )
    with pytest.raises(ValueError, match="admitted"):
        FeatureAdmission(utc, ["a"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique"):
        FeatureAdmission(utc, ("a", "a"))
    with pytest.raises(ValueError, match="non-empty"):
        FeatureAdmission(utc, ("a", ""))
    valid = FeatureAdmission(utc, ("a", "b"))
    assert pickle.loads(pickle.dumps(valid)) == valid  # noqa: S301


def test_warmup_cells_excluded_from_admission_coverage() -> None:
    # SPEC28_WARMUP_CELLS_EXCLUDED: a rolling-mean builder with
    # min_periods=W+1 on dense data and an all-true mask audits to
    # {year: 1.0} with warmup W, and to < 1.0 with warmup 0.
    warmup = 24
    n = 500
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    cols = ["A", "B", "C", "D"]
    rng = np.random.default_rng(21)
    close = pd.DataFrame(
        np.exp(np.cumsum(rng.normal(0.0, 0.005, (n, len(cols))), axis=0)) * 100.0,
        index=idx, columns=cols,
    )
    mask = pd.DataFrame(True, index=idx, columns=cols)

    def _build(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return panels["close"].rolling(warmup + 1, min_periods=warmup + 1).mean()

    feature = _build({"close": close})
    year = 2021
    with_warmup = FeatureSpec(
        name="rm", required_columns=("close",), min_coverage=0.9,
        builder=_build, warmup_bars=warmup,
    )
    without_warmup = FeatureSpec(
        name="rm", required_columns=("close",), min_coverage=0.9,
        builder=_build, warmup_bars=0,
    )
    assert feature_admission_coverage(
        with_warmup, feature, {"close": close}, mask,
    ) == {year: 1.0}
    uncovered = feature_admission_coverage(
        without_warmup, feature, {"close": close}, mask,
    )
    assert uncovered[year] == pytest.approx((n - warmup) / n)
    assert uncovered[year] < 1.0


def test_listing_anchor_excludes_only_post_listing_warmup() -> None:
    # SPEC28_LISTING_ANCHOR: symbol B lists at row 10 with W=5 and a NaN gap
    # at rows 20-21 (after anchor+W). The first W rows after listing are not
    # counted; pre-listing mask cells and the injected gap count as missing.
    n, anchor, warmup = 30, 10, 5
    idx = pd.date_range("2021-06-01", periods=n, freq="1h", tz="UTC")
    cols = ["A", "B"]
    rng = np.random.default_rng(22)
    close = pd.DataFrame(
        np.exp(np.cumsum(rng.normal(0.0, 0.005, (n, len(cols))), axis=0)) * 100.0,
        index=idx, columns=cols,
    )
    close.iloc[:anchor, 1] = np.nan
    close.iloc[20:22, 1] = np.nan
    mask = pd.DataFrame(True, index=idx, columns=cols)
    spec = FeatureSpec(
        name="x", required_columns=("close",), min_coverage=0.9,
        builder=lambda panels: panels["close"], warmup_bars=warmup,
    )
    coverage = feature_admission_coverage(
        spec, spec.builder({"close": close}), {"close": close}, mask,
    )
    assert coverage == pytest.approx({2021: (25 + 15 - 2) / (25 + 25)})


def test_late_source_feed_not_hidden_by_anchor() -> None:
    # SPEC28_LATE_SOURCE_FEED: column b starts K=12 rows after column a with
    # W=5. The K-W=7 post-warmup rows with NaN b are counted as missing:
    # coverage is exactly (n-K)/(n-W).
    n, warmup, late = 30, 5, 12
    idx = pd.date_range("2021-06-01", periods=n, freq="1h", tz="UTC")
    cols = ["A", "B"]
    rng = np.random.default_rng(23)
    a = pd.DataFrame(
        np.exp(np.cumsum(rng.normal(0.0, 0.005, (n, len(cols))), axis=0)) * 100.0,
        index=idx, columns=cols,
    )
    b = a.copy()
    b.iloc[:late] = np.nan
    mask = pd.DataFrame(True, index=idx, columns=cols)
    spec = FeatureSpec(
        name="x", required_columns=("a", "b"), min_coverage=0.9,
        builder=lambda panels: panels["b"], warmup_bars=warmup,
    )
    coverage = feature_admission_coverage(
        spec, spec.builder({"a": a, "b": b}), {"a": a, "b": b}, mask,
    )
    assert coverage == pytest.approx({2021: (n - late) / (n - warmup)})


def test_dead_source_symbol_counts_as_missing() -> None:
    # SPEC28_DEAD_SOURCE: symbol B has mask-true cells but all-NaN required
    # inputs, so all of its mask cells are auditable and uncovered:
    # 25 covered of 25 + 30 auditable.
    n, warmup = 30, 5
    idx = pd.date_range("2021-06-01", periods=n, freq="1h", tz="UTC")
    cols = ["A", "B"]
    rng = np.random.default_rng(24)
    close = pd.DataFrame(
        np.exp(np.cumsum(rng.normal(0.0, 0.005, (n, len(cols))), axis=0)) * 100.0,
        index=idx, columns=cols,
    )
    close.iloc[:, 1] = np.nan
    mask = pd.DataFrame(True, index=idx, columns=cols)
    spec = FeatureSpec(
        name="x", required_columns=("close",), min_coverage=0.9,
        builder=lambda panels: panels["close"], warmup_bars=warmup,
    )
    coverage = feature_admission_coverage(
        spec, spec.builder({"close": close}), {"close": close}, mask,
    )
    assert coverage == pytest.approx({2021: 25 / (25 + 30)})


def test_pure_warmup_year_omitted_and_empty_universe_year_stays_zero() -> None:
    # SPEC28_PURE_WARMUP_YEAR_OMITTED: year Y1's mask cells are all inside
    # warmup (omitted) while Y2 has zero mask cells (0.0, never NaN).
    n, warmup = 400, 400
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    assert idx[0].year == 2021
    assert idx[-1].year == 2022
    cols = ["A", "B"]
    rng = np.random.default_rng(25)
    close = pd.DataFrame(
        np.exp(np.cumsum(rng.normal(0.0, 0.005, (n, len(cols))), axis=0)) * 100.0,
        index=idx, columns=cols,
    )
    mask = pd.DataFrame(True, index=idx, columns=cols)
    mask.loc[idx[idx.year == 2022]] = False
    spec = FeatureSpec(
        name="x", required_columns=("close",), min_coverage=0.9,
        builder=lambda panels: panels["close"], warmup_bars=warmup,
    )
    coverage = feature_admission_coverage(
        spec, spec.builder({"close": close}), {"close": close}, mask,
    )
    assert 2021 not in coverage
    assert coverage == {2022: 0.0}


def test_empty_audit_never_admits() -> None:
    # SPEC28_EMPTY_AUDIT_NEVER_ADMITS: a coverage cutoff at or before the
    # first row admits nothing -- build_feature_books returns {} and the
    # boundary admission is empty -- even with min_coverage 0.0.
    n = 20
    idx = pd.date_range("2021-06-01", periods=n, freq="1h", tz="UTC")
    cols = ["A", "B"]
    rng = np.random.default_rng(26)
    close = pd.DataFrame(
        np.exp(np.cumsum(rng.normal(0.0, 0.005, (n, len(cols))), axis=0)) * 100.0,
        index=idx, columns=cols,
    )
    mask = pd.DataFrame(True, index=idx, columns=cols)
    spec = FeatureSpec(
        name="x", required_columns=("close",), min_coverage=0.0,
        builder=lambda panels: panels["close"],
    )
    grid = pd.date_range(idx[0], idx[-1], freq="24h", tz="UTC")
    for cutoff in (idx[0], idx[0] - pd.Timedelta(hours=1)):
        assert build_feature_books(
            [spec], {"close": close}, mask, grid, min_symbols=2,
            coverage_cutoff=cutoff,
        ) == {}
    admission = feature_admission_by_boundary(
        [spec], {"close": close}, mask,
        {"at_first": idx[0], "before_first": idx[0] - pd.Timedelta(hours=1)},
    )
    assert admission["at_first"].admitted == ()
    assert admission["before_first"].admitted == ()


def _rolling_gap_spec(
    name: str, window: int, warmup: int, min_coverage: float = 0.9,
) -> FeatureSpec:
    """Rolling-mean spec whose warmup cells are NaN by construction."""

    def _build(panels: dict[str, pd.DataFrame], _window: int = window) -> pd.DataFrame:
        return panels["close"].rolling(_window, min_periods=_window).mean()

    return FeatureSpec(
        name=name, required_columns=("close",), min_coverage=min_coverage,
        builder=_build, warmup_bars=warmup,
    )


def test_warmup_exclusion_only_widens_admission() -> None:
    # SPEC28_WARMUP_EXCLUSION_MONOTONE: over 5 seeds of random gap patterns
    # (each with audited rows), admission with declared warmups is a superset
    # of admission with warmup 0 -- excluding structural-NaN warmup cells can
    # only raise coverage.
    n = 300
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    cols = ["A", "B", "C", "D"]
    rng_base = np.random.default_rng(27)
    base = pd.DataFrame(
        np.exp(np.cumsum(rng_base.normal(0.0, 0.005, (n, len(cols))), axis=0)) * 100.0,
        index=idx, columns=cols,
    )
    specs = (_rolling_gap_spec("r20", 21, 20), _rolling_gap_spec("r50", 51, 50))
    zero_specs = (
        _rolling_gap_spec("r20", 21, 0), _rolling_gap_spec("r50", 51, 0),
    )
    cutoff = idx[-1] + pd.Timedelta(hours=1)
    assert int((idx < cutoff).sum()) >= 1
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        close = base.where(
            pd.DataFrame(
                rng.random((n, len(cols))) > 0.08, index=idx, columns=cols,
            )
        )
        close.iloc[0] = base.iloc[0]
        mask = pd.DataFrame(True, index=idx, columns=cols)
        assert feature_admission_coverage(
            specs[0], specs[0].builder({"close": close}),
            {"close": close}, mask, cutoff,
        ) != {}
        new = feature_admission_by_boundary(
            specs, {"close": close}, mask, {"b": cutoff},
        )
        old = feature_admission_by_boundary(
            zero_specs, {"close": close}, mask, {"b": cutoff},
        )
        assert set(old["b"].admitted) <= set(new["b"].admitted), seed


def test_admission_by_boundary_equals_book_keys() -> None:
    # SPEC28_ADMISSION_EQUALS_BOOK_KEYS: for every label,
    # admitted == tuple(books[label]) and cutoff == train_ends[label].
    n = 60
    idx = pd.date_range("2021-06-01", periods=n, freq="1h", tz="UTC")
    cols = ["A", "B", "C"]
    rng = np.random.default_rng(28)
    close = pd.DataFrame(
        np.exp(np.cumsum(rng.normal(0.0, 0.005, (n, len(cols))), axis=0)) * 100.0,
        index=idx, columns=cols,
    )
    mask = pd.DataFrame(True, index=idx, columns=cols)
    good = FeatureSpec(
        name="good", required_columns=("close",), min_coverage=0.9,
        builder=lambda panels: panels["close"],
    )
    bad = FeatureSpec(
        name="bad", required_columns=("close",), min_coverage=0.9,
        builder=lambda panels: panels["close"] * np.nan,
    )
    specs = (bad, good)
    grid = pd.date_range(idx[0], idx[-1], freq="24h", tz="UTC")
    ends = {"e1": idx[30], "e2": idx[-1] + pd.Timedelta(hours=1)}
    admission = feature_admission_by_boundary(
        specs, {"close": close}, mask, ends,
    )
    books = build_feature_books_by_boundary(
        specs, {"close": close}, mask, grid, ends, min_symbols=2,
    )
    for label in ends:
        assert admission[label].admitted == tuple(books[label])
        assert admission[label].cutoff == ends[label]


def test_admission_by_boundary_builds_each_feature_once() -> None:
    # SPEC28_ADMISSION_BUILDS_ONCE: with counting builders and 5 boundaries,
    # each builder runs exactly once.
    n = 40
    idx = pd.date_range("2021-06-01", periods=n, freq="1h", tz="UTC")
    cols = ["A", "B", "C"]
    rng = np.random.default_rng(29)
    close = pd.DataFrame(
        np.exp(np.cumsum(rng.normal(0.0, 0.005, (n, len(cols))), axis=0)) * 100.0,
        index=idx, columns=cols,
    )
    mask = pd.DataFrame(True, index=idx, columns=cols)
    calls = {"a": 0, "b": 0}

    def _make(name: str):  # type: ignore[no-untyped-def]
        def _build(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
            calls[name] += 1
            return panels["close"]
        return _build

    specs = (
        FeatureSpec(name="a", required_columns=("close",),
                    min_coverage=0.9, builder=_make("a")),
        FeatureSpec(name="b", required_columns=("close",),
                    min_coverage=0.9, builder=_make("b")),
    )
    ends = {f"b{i}": idx[5 + i] for i in range(5)}
    feature_admission_by_boundary(specs, {"close": close}, mask, ends)
    assert calls == {"a": 1, "b": 1}


def test_admission_by_boundary_ignores_data_at_and_after_boundary() -> None:
    # SPEC28_ADMISSION_IGNORES_POST_BOUNDARY: features that become NaN from
    # boundary b onward leave b's admission equal to the unperturbed result.
    n = 60
    idx = pd.date_range("2021-06-01", periods=n, freq="1h", tz="UTC")
    cols = ["A", "B", "C"]
    rng = np.random.default_rng(30)
    close = pd.DataFrame(
        np.exp(np.cumsum(rng.normal(0.0, 0.005, (n, len(cols))), axis=0)) * 100.0,
        index=idx, columns=cols,
    )
    mask = pd.DataFrame(True, index=idx, columns=cols)
    specs = (
        FeatureSpec(
            name="plain", required_columns=("close",), min_coverage=0.9,
            builder=lambda panels: panels["close"],
        ),
        _rolling_gap_spec("rolled", 11, 10),
    )
    boundary = idx[30]
    ends = {"b": boundary, "late": idx[-1] + pd.Timedelta(hours=1)}
    clean = feature_admission_by_boundary(
        specs, {"close": close}, mask, ends,
    )
    perturbed = close.copy()
    perturbed.loc[perturbed.index >= boundary] = np.nan
    rebuilt = feature_admission_by_boundary(
        specs, {"close": perturbed}, mask, ends,
    )
    assert rebuilt["b"] == clean["b"]


def test_admitted_books_match_audited_books_without_auditing() -> None:
    # SPEC28_ADMITTED_BOOKS_MATCH_AUDITED: for passing specs,
    # build_admitted_feature_books is byte-identical to build_feature_books
    # (check_exact, same key order); a spec that would fail the audit is
    # still built by build_admitted_feature_books.
    n = 200
    idx = pd.date_range("2021-06-01", periods=n, freq="1h", tz="UTC")
    cols = ["A", "B", "C", "D"]
    rng = np.random.default_rng(31)
    close = pd.DataFrame(
        np.exp(np.cumsum(rng.normal(0.0, 0.005, (n, len(cols))), axis=0)) * 100.0,
        index=idx, columns=cols,
    )
    mask = pd.DataFrame(True, index=idx, columns=cols)
    specs = (
        FeatureSpec(
            name="f1", required_columns=("close",), min_coverage=0.9,
            builder=lambda panels: panels["close"],
        ),
        FeatureSpec(
            name="f2", required_columns=("close",), min_coverage=0.9,
            builder=lambda panels: -panels["close"],
        ),
    )
    grid = pd.date_range(idx[0], idx[-1], freq="24h", tz="UTC")
    audited = build_feature_books(
        specs, {"close": close}, mask, grid, min_symbols=2,
    )
    admitted = build_admitted_feature_books(
        specs, {"close": close}, mask, grid, min_symbols=2,
    )
    assert list(admitted) == list(audited)
    for name in audited:
        pd.testing.assert_frame_equal(admitted[name], audited[name], check_exact=True)
    failing = FeatureSpec(
        name="bad", required_columns=("close",), min_coverage=0.9,
        builder=lambda panels: panels["close"] * np.nan,
    )
    assert build_feature_books(
        (failing,), {"close": close}, mask, grid, min_symbols=2,
    ) == {}
    assert list(
        build_admitted_feature_books(
            (failing,), {"close": close}, mask, grid, min_symbols=2,
        )
    ) == ["bad"]




def test_admission_validation_branches_raise() -> None:
    # D5-adjacent guards: every ValueError branch of the admission entry points.
    import pandas as pd
    import pytest

    from src.mhs.features import (
        FeatureAdmission,
        FeatureSpec,
        build_admitted_feature_books,
        build_feature_books,
        feature_admission_by_boundary,
        feature_admission_coverage,
    )

    idx = pd.date_range("2021-01-01", periods=48, freq="1h", tz="UTC")
    mask = pd.DataFrame(True, index=idx, columns=["A", "B"])
    close = pd.DataFrame(1.0, index=idx, columns=["A", "B"])
    spec = FeatureSpec(name="x", required_columns=("close",), min_coverage=0.9,
                       builder=lambda panels: panels["close"])
    feature = spec.builder({"close": close})
    grid = pd.date_range(idx[0], idx[-1], freq="24h", tz="UTC")

    with pytest.raises(ValueError, match="cutoff"):
        FeatureAdmission("2021-01-01", ("a",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="identically indexed"):
        feature_admission_coverage(spec, feature.iloc[1:], {"close": close}, mask)
    with pytest.raises(ValueError, match="required_columns"):
        feature_admission_coverage(spec, feature, {}, mask)
    shifted = close.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="identically indexed"):
        feature_admission_coverage(spec, feature, {"close": shifted}, mask)
    with pytest.raises(ValueError, match="min_symbols"):
        build_feature_books((spec,), {"close": close}, mask, grid, min_symbols=1)
    with pytest.raises(ValueError, match="identically indexed"):
        build_feature_books((spec,), {"close": shifted}, mask, grid)
    with pytest.raises(ValueError, match="identically indexed"):
        build_feature_books(
            (spec,), {"close": close},
            pd.DataFrame(True, index=shifted.index, columns=["A", "B"]), grid,
        )
    with pytest.raises(ValueError, match="required_columns"):
        feature_admission_by_boundary(
            (spec,), {}, mask, {"b": idx[0]},
        )
    with pytest.raises(ValueError, match="identically indexed"):
        feature_admission_by_boundary(
            (spec,), {"close": shifted}, mask, {"b": idx[0]},
        )
    bad_builder = FeatureSpec(name="y", required_columns=("close",), min_coverage=0.9,
                              builder=lambda panels: panels["close"].iloc[1:])
    with pytest.raises(ValueError, match="identically indexed"):
        feature_admission_by_boundary(
            (bad_builder,), {"close": close}, mask, {"b": idx[0]},
        )
    with pytest.raises(ValueError, match="min_symbols"):
        build_admitted_feature_books((spec,), {"close": close}, mask, grid, min_symbols=1)
    with pytest.raises(ValueError, match="required_columns"):
        build_admitted_feature_books((spec,), {}, mask, grid)
    with pytest.raises(ValueError, match="identically indexed"):
        build_admitted_feature_books((bad_builder,), {"close": close}, mask, grid)


def test_admission_vector_branches_dead_source_and_empty_year() -> None:
    # Vector helpers: dead-source anchor-None, zero-mask year, pure-warmup omit.
    import pandas as pd

    from src.mhs.features import (
        FeatureSpec,
        feature_admission_by_boundary,
    )

    idx = pd.date_range("2021-01-01", periods=72, freq="1h", tz="UTC")
    mask = pd.DataFrame(True, index=idx, columns=["A", "B"])
    close = pd.DataFrame(1.0, index=idx, columns=["A", "B"])
    close["B"] = float("nan")  # dead source symbol
    dead = FeatureSpec(name="d", required_columns=("close",), min_coverage=0.0,
                       builder=lambda panels: panels["close"].fillna(0.0),
                       warmup_bars=5)
    out = feature_admission_by_boundary(
        (dead,), {"close": close}, mask, {"b": idx[-1] + pd.Timedelta(hours=1)},
    )
    assert out["b"].admitted == ("d",)

    mask_empty_year = mask.copy()
    mask_empty_year.loc[mask_empty_year.index.year == 2021, :] = False
    warm = FeatureSpec(name="w", required_columns=("close",), min_coverage=0.9,
                       builder=lambda panels: panels["close"], warmup_bars=1000)
    out2 = feature_admission_by_boundary(
        (warm,), {"close": pd.DataFrame(1.0, index=idx, columns=["A", "B"])},
        mask, {"b": idx[-1] + pd.Timedelta(hours=1)},
    )
    assert out2["b"].admitted == ()
    out3 = feature_admission_by_boundary(
        (warm,), {"close": pd.DataFrame(1.0, index=idx, columns=["A", "B"])},
        mask_empty_year, {"b": idx[-1] + pd.Timedelta(hours=1)},
    )
    assert out3["b"].admitted == ()


def test_coverage_symbol_without_year_mask_skipped() -> None:
    # feature_admission_coverage: symbol with no mask cell in the audited year.
    import pandas as pd

    from src.mhs.features import FeatureSpec, feature_admission_coverage

    idx = pd.date_range("2021-06-01", periods=48, freq="1h", tz="UTC")
    mask = pd.DataFrame(True, index=idx, columns=["A", "B"])
    mask["B"] = False
    close = pd.DataFrame(1.0, index=idx, columns=["A", "B"])
    spec = FeatureSpec(name="x", required_columns=("close",), min_coverage=0.0,
                       builder=lambda panels: panels["close"])
    cov = feature_admission_coverage(spec, close, {"close": close}, mask)
    assert cov[2021] == 1.0


def test_book_entry_points_reject_misaligned_panels() -> None:
    # build_feature_books panel-vs-mask check and by_boundary builder-output check.
    import pandas as pd
    import pytest

    from src.mhs.features import FeatureSpec, build_feature_books, build_feature_books_by_boundary

    idx = pd.date_range("2021-01-01", periods=48, freq="1h", tz="UTC")
    mask = pd.DataFrame(True, index=idx, columns=["A", "B"])
    close = pd.DataFrame(1.0, index=idx, columns=["A", "B"])
    shifted = close.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=1)
    spec = FeatureSpec(name="x", required_columns=("close",), min_coverage=0.9,
                       builder=lambda panels: panels["close"])
    grid = pd.date_range(idx[0], idx[-1], freq="24h", tz="UTC")
    with pytest.raises(ValueError, match="identically indexed"):
        build_feature_books((spec,), {"close": shifted}, mask, grid)
    two_col = FeatureSpec(name="z", required_columns=("close", "quote_vol"),
                          min_coverage=0.9, builder=lambda panels: panels["close"])
    with pytest.raises(ValueError, match="identically indexed"):
        build_feature_books(
            (two_col,), {"close": close, "quote_vol": shifted}, mask, grid,
        )
    bad = FeatureSpec(name="y", required_columns=("close",), min_coverage=0.9,
                      builder=lambda panels: panels["close"].iloc[1:])
    with pytest.raises(ValueError, match="identically indexed"):
        build_feature_books_by_boundary(
            (bad,), {"close": close}, mask, grid, {"b": idx[0]},
        )


def test_future_source_start_cannot_change_boundary_admission() -> None:
    from src.mhs.features import (
        build_feature_books_by_boundary,
        feature_admission_by_boundary,
        feature_admission_coverage,
    )

    idx = pd.date_range("2021-01-01", periods=24, freq="h", tz="UTC")
    close = pd.DataFrame({"A": 1.0, "B": np.nan}, index=idx)
    mask = pd.DataFrame(True, index=idx, columns=close.columns)
    spec = FeatureSpec("x", ("close",), 0.9, lambda p: p["close"], 2)
    cutoff = idx[12]
    future_source = close.copy()
    future_source.loc[cutoff:, "B"] = 2.0
    results = []
    for panel in (close, future_source, future_source.loc[future_source.index < cutoff]):
        local_mask = mask.reindex(panel.index)
        panels = {"close": panel}
        coverage = feature_admission_coverage(spec, panel, panels, local_mask, cutoff)
        admission = feature_admission_by_boundary((spec,), panels, local_mask, {"b": cutoff})["b"]
        books = build_feature_books_by_boundary((spec,), panels, local_mask, panel.index, {"b": cutoff}, 2)
        assert tuple(books["b"]) == admission.admitted
        results.append((coverage, admission))
    assert results[0] == results[1] == results[2]
    assert results[0][0] == {2021: 10 / 22}
    assert results[0][1].admitted == ()
