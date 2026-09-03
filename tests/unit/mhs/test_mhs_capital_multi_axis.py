"""Contract coverage for the MHS committee_capital top-level blend wiring (RC-4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.market_data.services.futures_collection as fc
from src.mhs import evaluation as ev
import src.mhs.marks as marks
from src.mhs.evaluation import (
    MhsDiagnosticRequest,
    _committee_execution_book,
)
from src.mhs.types import COMMITTEE_MEMBERS
from src.mhs.evidence import AnchoredPurgedFold
from src.mhs.features import FEATURE_REGISTRY, FeatureSpec, build_feature_books
from src.quant.universe.pit_universe import symbol_partition
from tests.unit.mhs.test_evaluation_appresearch import _write_mhs_market

_START = pd.Timestamp("2021-01-01", tz="UTC")

_FOLD = AnchoredPurgedFold(
    pd.Timestamp("2021-01-01", tz="UTC"),
    pd.Timestamp("2021-01-31", tz="UTC"),
    pd.Timestamp("2021-02-10", tz="UTC"),
    pd.Timestamp("2021-04-19 08:00", tz="UTC"),
    168,
    168,
)


def _committee_member_specs() -> list[FeatureSpec]:
    return [spec for spec in FEATURE_REGISTRY if spec.name in set(COMMITTEE_MEMBERS)]


@pytest.fixture
def mhs_market_with_taker_buy_quote(tmp_path, monkeypatch):
    """Synthetic market with ``taker_buy_quote`` so the committee fold path loads."""
    root = tmp_path / "market_tbq"
    end = _write_mhs_market(root, include_taker_buy_quote=True)
    monkeypatch.setattr(marks, "funding_path", lambda sym: root / "funding" / f"{sym}.parquet")
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet")
    # _get_symbol_mark_frame is a process-global lru_cache keyed on
    # (symbol, timeframe) only; a prior test in the same process/worker using
    # a different root with an overlapping symbol name would otherwise leak
    # stale mark data into this fixture's replay.
    ev._get_symbol_mark_frame.cache_clear()
    return root, end


def _synthetic_panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hourly = pd.date_range(_START, periods=2700, freq="1h", tz="UTC")
    symbols = [f"SYM{i:02d}" for i in range(10)]
    rng = np.random.default_rng(7)
    close = pd.DataFrame(
        {s: 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 2700))) for s in symbols},
        index=hourly,
    )
    quote_vol = pd.DataFrame(1000.0, index=hourly, columns=symbols)
    taker_buy_quote = pd.DataFrame(520.0, index=hourly, columns=symbols)
    return close, quote_vol, taker_buy_quote


def test_committee_execution_book_matches_inline_formula() -> None:
    # SCENARIO_COMMITTEE_EXECUTION_BOOK_FOLD_PATH_UNCHANGED: the extracted
    # helper reproduces the pre-refactor inline computation byte-for-byte --
    # registry filter to the k=6 members, equal-notional books, equal-weight
    # mean -- so the fold path's output cannot drift from the old block.
    close, quote_vol, taker_buy_quote = _synthetic_panels()
    execution_mask = pd.DataFrame(True, index=close.index, columns=close.columns)
    decision_grid = pd.date_range(_START, close.index[-1], freq="24h", tz="UTC")
    books = build_feature_books(
        _committee_member_specs(),
        {"close": close, "quote_vol": quote_vol, "taker_buy_quote": taker_buy_quote},
        execution_mask, decision_grid, min_symbols=8,
    )
    assert books
    inline = sum(books.values()) / float(len(books))
    out = _committee_execution_book(
        close, quote_vol, taker_buy_quote, execution_mask, decision_grid, 8,
    )
    pd.testing.assert_frame_equal(out, inline)
    assert out.index.equals(close.index)
    assert out.shape[1] <= close.shape[1]


def test_committee_execution_book_fails_closed(monkeypatch) -> None:
    # SCENARIO_COMMITTEE_EXECUTION_BOOK_FAILS_CLOSED: a window where every
    # committee member fails its coverage floor raises the existing RuntimeError
    # -- never an empty frame, never a silent fallback to the momentum book.
    close, quote_vol, taker_buy_quote = _synthetic_panels()
    execution_mask = pd.DataFrame(True, index=close.index, columns=close.columns)
    decision_grid = pd.date_range(_START, close.index[-1], freq="24h", tz="UTC")
    monkeypatch.setattr(ev, "build_feature_books", lambda *a, **k: {})
    with pytest.raises(RuntimeError, match="committee_capital: no committee member admitted"):
        _committee_execution_book(
            close, quote_vol, taker_buy_quote, execution_mask, decision_grid, 8,
        )


def test_fold_path_wires_helper_byte_identical(mhs_market_with_taker_buy_quote, monkeypatch) -> None:
    # SCENARIO_COMMITTEE_EXECUTION_BOOK_FOLD_PATH_UNCHANGED: the refactored
    # _build_fold_target_weights calls the extracted helper on the fold's own
    # panels, and that output equals the pre-refactor inline formula computed
    # here from the very same admitted-member books (byte-identical blend_1h
    # implies byte-identical target_weights, since every downstream step is
    # untouched by the refactor).
    root, end = mhs_market_with_taker_buy_quote
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        committee_capital=True,
    )
    orig = ev._committee_execution_book
    captured: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, int] | None = None

    def recording_helper(
        close, quote_vol, taker_buy_quote, execution_mask, decision_grid, min_symbols, *args, **kwargs,
    ):
        nonlocal captured
        captured = (close, quote_vol, taker_buy_quote, execution_mask, decision_grid, min_symbols)
        return orig(close, quote_vol, taker_buy_quote, execution_mask, decision_grid, min_symbols, *args, **kwargs)

    monkeypatch.setattr(ev, "_committee_execution_book", recording_helper)
    target, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request, funding_by_symbol,
    )
    assert captured is not None
    close, quote_vol, taker_buy_quote, execution_mask, decision_grid, min_symbols = captured
    assert min_symbols == ev.BOOK_SPECS["slow_momentum"].min_symbols
    books = build_feature_books(
        _committee_member_specs(),
        {"close": close, "quote_vol": quote_vol, "taker_buy_quote": taker_buy_quote},
        execution_mask, decision_grid, min_symbols=min_symbols,
    )
    assert books
    expected = sum(books.values()) / float(len(books))
    blend_1h = orig(close, quote_vol, taker_buy_quote, execution_mask, decision_grid, min_symbols)
    pd.testing.assert_frame_equal(blend_1h, expected)
    assert np.isfinite(target.to_numpy(dtype="float64")).all()
