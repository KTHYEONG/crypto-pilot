"""Invariant guards for the live frozen book core."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.live.frozen_book import (
    LiveFrozenBook,
    build_live_frozen_book,
    crypto_census,
    extend_unit_history,
    unit_proxy_returns,
)
from src.mhs.books import clip_names_preserving_gross
from src.mhs.frozen_research_candidate import FROZEN_MHS_TOP20_V2, build_frozen_mhs_candidate
from src.mhs.panel import load_base_panel
from src.mhs.params import FROZEN_GROWTH_NAME_CLIP, LIVE_FROZEN_WARMUP_DAYS

_START = pd.Timestamp("2021-01-01", tz="UTC")
_SYMBOLS = tuple(f"SYM{i:02d}USDT" for i in range(10))


def _write_panel(
    root: Path, symbols: tuple[str, ...], *, days: int = 130, seed: int = 7,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    n = days * 24
    grid = pd.date_range(_START, periods=n, freq="1h", tz="UTC")
    out = root / "ohlcv" / "1h"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ms = np.array([int(ts.value // 1_000_000) for ts in grid], dtype="int64")
    for j, sym in enumerate(symbols):
        rets = rng.normal(0, 0.005, size=n)
        close = 100.0 * (1.0 + 0.01 * j) * np.exp(np.cumsum(rets))
        qv = 100_000.0 + rng.uniform(0, 20_000.0, size=n)
        tbq = np.clip(qv * 0.5 * (1.0 + rng.normal(0, 0.02, size=n)), 0.0, None)
        pd.DataFrame(
            {"timestamp": ms, "close": close, "quote_vol": qv, "taker_buy_quote": tbq},
        ).to_parquet(out / f"{sym}.parquet")
    return _START, grid[-1] + pd.Timedelta(hours=1)


def _research_comparison(tmp_path: Path, symbols: tuple[str, ...]) -> None:
    start, end = _write_panel(tmp_path, symbols)
    book = build_live_frozen_book(tmp_path, symbols, panel_start=start, panel_end=end)
    end_inclusive = end - pd.Timedelta(hours=1)
    panel = load_base_panel(
        str(tmp_path / "ohlcv"), "1h", ("close", "quote_vol", "taker_buy_quote"),
        start, end_inclusive, partition="all", selection_mode="causal_history",
    )
    census = list(symbols)
    close_c = panel["close"][census]
    grid_1h = close_c.index
    completed = (grid_1h + pd.Timedelta(hours=1)).to_numpy(dtype="datetime64[ns]")
    avail = pd.DataFrame(
        np.tile(completed[:, None], (1, len(census))), index=grid_1h, columns=census,
    ).apply(lambda col: pd.to_datetime(col).dt.tz_localize("UTC"))
    daily_close = close_c.resample("1D").last().astype("float64")
    daily_qv = panel["quote_vol"][census].resample("1D").sum(min_count=1).astype("float64")
    candidate = build_frozen_mhs_candidate(
        {"close": close_c, "quote_vol": panel["quote_vol"][census],
         "taker_buy_quote": panel["taker_buy_quote"][census]},
        avail, daily_close, daily_qv, tuple(census),
        market_close=close_c, strategy=FROZEN_MHS_TOP20_V2, blocked_decisions=None,
    )
    clipped = clip_names_preserving_gross(candidate.target_weights, FROZEN_GROWTH_NAME_CLIP)
    for day in book.unit_weights.index:
        entry = day + pd.Timedelta(days=1)
        pd.testing.assert_series_equal(
            book.unit_weights.loc[day], clipped.loc[entry], check_names=False,
        )


def _mini_book() -> LiveFrozenBook:
    decisions = pd.DatetimeIndex(["2021-01-01", "2021-01-02", "2021-01-03"], tz="UTC")
    entries = pd.DatetimeIndex(["2021-01-02", "2021-01-03", "2021-01-04"], tz="UTC")
    cols = ["AAAUSDT", "BBBUSDT"]
    unit = pd.DataFrame(
        [[0.5, -0.5], [0.5, -0.5], [0.0, 0.0]], index=decisions, columns=cols, dtype="float64",
    )
    snap = pd.DataFrame(100.0, index=decisions, columns=cols, dtype="float64")
    entry = pd.DataFrame(
        [[100.0, 100.0], [110.0, 100.0], [110.0, 100.0]],
        index=entries, columns=cols, dtype="float64",
    )
    adv = pd.DataFrame(1e6, index=decisions, columns=cols, dtype="float64")
    sigma = pd.DataFrame(0.02, index=decisions, columns=cols, dtype="float64")
    return LiveFrozenBook(
        unit_weights=unit, snapshot_closes=snap, entry_closes=entry,
        adv=adv, daily_sigma=sigma, valid_from=decisions[0],
    )


def test_census_excludes_non_crypto_underlyings() -> None:
    assert crypto_census(["BTCUSDT", "AAPLUSDT", "LUNAUSDT"], frozenset({"AAPLUSDT"})) == (
        "BTCUSDT", "LUNAUSDT",
    )


def test_book_matches_research_builder_on_same_window(tmp_path: Path) -> None:
    _research_comparison(tmp_path, _SYMBOLS)


def test_rows_before_warmup_are_dropped(tmp_path: Path) -> None:
    start, end = _write_panel(tmp_path, _SYMBOLS)
    book = build_live_frozen_book(tmp_path, _SYMBOLS, panel_start=start, panel_end=end)
    assert book.unit_weights.index.min() >= start.normalize() + pd.Timedelta(
        days=int(LIVE_FROZEN_WARMUP_DAYS),
    )


def test_short_window_fails_closed(tmp_path: Path) -> None:
    start = _START
    end = start + pd.Timedelta(days=int(LIVE_FROZEN_WARMUP_DAYS) - 1)
    with pytest.raises(DataIntegrityError):
        build_live_frozen_book(tmp_path, _SYMBOLS, panel_start=start, panel_end=end)


def test_future_bars_after_snapshot_never_change_decision_row(tmp_path: Path) -> None:
    start, end = _write_panel(tmp_path, _SYMBOLS)
    book_a = build_live_frozen_book(tmp_path, _SYMBOLS, panel_start=start, panel_end=end)
    day = book_a.unit_weights.index[1]
    cutoff = day + pd.Timedelta(hours=23)
    for sym in _SYMBOLS:
        path = tmp_path / "ohlcv" / "1h" / f"{sym}.parquet"
        frame = pd.read_parquet(path)
        stamps = pd.to_datetime(pd.to_numeric(frame["timestamp"], errors="coerce"), unit="ms", utc=True)
        frame.loc[stamps > cutoff, "close"] = frame.loc[stamps > cutoff, "close"] * 2.0
        frame.to_parquet(path)
    book_b = build_live_frozen_book(tmp_path, _SYMBOLS, panel_start=start, panel_end=end)
    pd.testing.assert_series_equal(
        book_a.unit_weights.loc[day], book_b.unit_weights.loc[day], check_names=False,
    )


def test_census_column_restriction(tmp_path: Path) -> None:
    start, end = _write_panel(tmp_path, (*_SYMBOLS, "EXTRAUSDT"))
    book = build_live_frozen_book(tmp_path, _SYMBOLS, panel_start=start, panel_end=end)
    assert "EXTRAUSDT" not in book.unit_weights.columns


def test_unit_proxy_prices_entry_to_entry_with_lag_label() -> None:
    book = _mini_book()
    out = unit_proxy_returns(book, {}, cost_bps=0.0)
    assert out.index[0] == pd.Timestamp("2021-01-03", tz="UTC")
    assert out.iloc[0] == pytest.approx(0.05)


def test_longs_pay_positive_funding() -> None:
    book = _mini_book()
    entry = pd.Timestamp("2021-01-02", tz="UTC")
    nxt = pd.Timestamp("2021-01-03", tz="UTC")
    funding = {
        "AAAUSDT": pd.Series([0.001], index=pd.DatetimeIndex([entry + pd.Timedelta(hours=1)], tz="UTC")),
        "BBBUSDT": pd.Series([0.0], index=pd.DatetimeIndex([entry + pd.Timedelta(hours=1)], tz="UTC")),
    }
    base = unit_proxy_returns(book, {}, cost_bps=0.0)
    paid = unit_proxy_returns(book, funding, cost_bps=0.0)
    assert (base.iloc[0] - paid.iloc[0]) == pytest.approx(0.0005)
    assert nxt in paid.index


def test_funding_settlement_cadence_is_summed_exactly() -> None:
    book = _mini_book()
    entry = pd.Timestamp("2021-01-02", tz="UTC")
    nxt = pd.Timestamp("2021-01-03", tz="UTC")
    hourly_idx = pd.DatetimeIndex([entry + pd.Timedelta(hours=h) for h in range(1, 9)], tz="UTC")
    one = pd.DatetimeIndex([entry + pd.Timedelta(hours=8)], tz="UTC")
    eight = {
        "AAAUSDT": pd.Series([0.0001] * 8, index=hourly_idx),
        "BBBUSDT": pd.Series([0.0] * 8, index=hourly_idx),
    }
    single = {
        "AAAUSDT": pd.Series([0.0008], index=one),
        "BBBUSDT": pd.Series([0.0], index=one),
    }
    assert unit_proxy_returns(book, eight, cost_bps=0.0).iloc[0] == pytest.approx(
        unit_proxy_returns(book, single, cost_bps=0.0).iloc[0],
    )


def test_turnover_cost_uses_weight_change() -> None:
    decisions = pd.DatetimeIndex(["2021-01-01", "2021-01-02", "2021-01-03"], tz="UTC")
    entries = pd.DatetimeIndex(["2021-01-02", "2021-01-03", "2021-01-04"], tz="UTC")
    cols = ["AAAUSDT", "BBBUSDT"]
    unit = pd.DataFrame(
        [[0.0, 0.0], [0.2, -0.2], [0.2, -0.2]], index=decisions, columns=cols, dtype="float64",
    )
    flat = pd.DataFrame(100.0, index=entries, columns=cols, dtype="float64")
    snap = pd.DataFrame(100.0, index=decisions, columns=cols, dtype="float64")
    book = LiveFrozenBook(
        unit_weights=unit, snapshot_closes=snap, entry_closes=flat,
        adv=snap.copy(), daily_sigma=snap.copy(), valid_from=decisions[0],
    )
    out = unit_proxy_returns(book, {}, cost_bps=2.0)
    assert out.iloc[1] == pytest.approx(-0.00008)


def test_held_symbol_with_missing_entry_close_fails_closed() -> None:
    book = _mini_book()
    book.entry_closes.iloc[0, 0] = float("nan")
    with pytest.raises(DataIntegrityError):
        unit_proxy_returns(book, {}, cost_bps=0.0)


def test_history_extension_is_contiguous_and_append_only() -> None:
    boot_idx = pd.date_range("2021-06-01", "2021-06-30", freq="1D", tz="UTC")
    fwd_idx = pd.date_range("2021-07-01", "2021-07-10", freq="1D", tz="UTC")
    proxy_idx = pd.date_range("2021-07-05", "2021-07-12", freq="1D", tz="UTC")
    bootstrap = pd.Series(0.001, index=boot_idx, dtype="float64")
    forward = pd.Series(0.002, index=fwd_idx, dtype="float64")
    proxy = pd.Series(0.009, index=proxy_idx, dtype="float64")
    out = extend_unit_history(bootstrap, forward, proxy)
    assert (out.loc["2021-07-05":"2021-07-10"] == 0.002).all()
    assert out.loc["2021-07-11"] == pytest.approx(0.009)
    assert out.loc["2021-07-12"] == pytest.approx(0.009)
    assert (out.index[1:] - out.index[:-1] == pd.Timedelta(days=1)).all()


def test_history_gap_fails_closed() -> None:
    boot_idx = pd.date_range("2021-06-01", "2021-06-30", freq="1D", tz="UTC")
    proxy_idx = pd.date_range("2021-07-03", "2021-07-05", freq="1D", tz="UTC")
    bootstrap = pd.Series(0.001, index=boot_idx, dtype="float64")
    forward = pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))
    proxy = pd.Series(0.001, index=proxy_idx, dtype="float64")
    with pytest.raises(DataIntegrityError):
        extend_unit_history(bootstrap, forward, proxy)


def test_build_rejects_empty_and_duplicate_census(tmp_path: Path) -> None:
    with pytest.raises(DataIntegrityError):
        build_live_frozen_book(tmp_path, (), panel_start=_START, panel_end=_START + pd.Timedelta(days=130))
    with pytest.raises(DataIntegrityError):
        build_live_frozen_book(
            tmp_path, ("AAAUSDT", "AAAUSDT"), panel_start=_START,
            panel_end=_START + pd.Timedelta(days=130),
        )
    with pytest.raises(DataIntegrityError):
        build_live_frozen_book(
            tmp_path, _SYMBOLS, panel_start=_START.tz_localize(None),
            panel_end=_START + pd.Timedelta(days=130),
        )
    with pytest.raises(DataIntegrityError):
        build_live_frozen_book(
            tmp_path, _SYMBOLS, panel_start=_START + pd.Timedelta(days=1),
            panel_end=_START,
        )


def test_census_symbol_without_bars_in_window_is_dropped(tmp_path: Path) -> None:
    # 창 이전에 상장폐지된 심볼은 창 안 봉이 없어 로스터·시장 대용치에 기여할 수 없다.
    start, end = _write_panel(tmp_path, _SYMBOLS)
    dead = tmp_path / "ohlcv" / "1h" / "DEADUSDT.parquet"
    pd.DataFrame({
        "timestamp": np.array([int((start - pd.Timedelta(days=400)).value // 1_000_000)], dtype="int64"),
        "close": [1.0], "quote_vol": [1.0], "taker_buy_quote": [0.5],
    }).to_parquet(dead)
    with_dead = build_live_frozen_book(tmp_path, (*_SYMBOLS, "DEADUSDT"), panel_start=start, panel_end=end)
    without = build_live_frozen_book(tmp_path, _SYMBOLS, panel_start=start, panel_end=end)
    assert "DEADUSDT" not in with_dead.unit_weights.columns
    pd.testing.assert_frame_equal(with_dead.unit_weights, without.unit_weights)


def test_census_with_no_bars_in_window_fails_closed(tmp_path: Path) -> None:
    start, end = _write_panel(tmp_path, _SYMBOLS)
    dead = tmp_path / "ohlcv" / "1h" / "DEADUSDT.parquet"
    pd.DataFrame({
        "timestamp": np.array([int((start - pd.Timedelta(days=400)).value // 1_000_000)], dtype="int64"),
        "close": [1.0], "quote_vol": [1.0], "taker_buy_quote": [0.5],
    }).to_parquet(dead)
    with pytest.raises(DataIntegrityError, match="no census symbol"):
        build_live_frozen_book(tmp_path, ("DEADUSDT",), panel_start=start, panel_end=end)


def test_build_fails_when_source_malformed(tmp_path: Path) -> None:
    start, end = _write_panel(tmp_path, _SYMBOLS)
    other = tmp_path / "other"
    _write_panel(other, _SYMBOLS)
    bad = other / "ohlcv" / "1h" / f"{_SYMBOLS[0]}.parquet"
    frame = pd.read_parquet(bad).drop(columns=["taker_buy_quote"])
    frame.to_parquet(bad)
    with pytest.raises(DataIntegrityError):
        build_live_frozen_book(other, _SYMBOLS, panel_start=start, panel_end=end)
    empty_root = tmp_path / "empty"
    (empty_root / "ohlcv" / "1h").mkdir(parents=True, exist_ok=True)
    with pytest.raises(DataIntegrityError):
        build_live_frozen_book(empty_root, _SYMBOLS, panel_start=start, panel_end=end)


def test_proxy_ignores_zero_weight_nans_and_empty_books() -> None:
    book = _mini_book()
    book.entry_closes.iloc[:, 1] = float("nan")
    book.unit_weights.iloc[:, 1] = 0.0
    book.unit_weights.iloc[1, 1] = 0.0
    out = unit_proxy_returns(book, {}, cost_bps=0.0)
    assert np.isfinite(out.to_numpy()).all()
    empty_idx = pd.DatetimeIndex([], tz="UTC")
    empty = LiveFrozenBook(
        unit_weights=pd.DataFrame(columns=["AAAUSDT"], index=empty_idx, dtype="float64"),
        snapshot_closes=pd.DataFrame(columns=["AAAUSDT"], index=empty_idx, dtype="float64"),
        entry_closes=pd.DataFrame(columns=["AAAUSDT"], index=empty_idx, dtype="float64"),
        adv=pd.DataFrame(columns=["AAAUSDT"], index=empty_idx, dtype="float64"),
        daily_sigma=pd.DataFrame(columns=["AAAUSDT"], index=empty_idx, dtype="float64"),
        valid_from=_START,
    )
    assert len(unit_proxy_returns(empty, {}, cost_bps=0.0)) == 0
    outside = {
        "AAAUSDT": pd.Series([0.01], index=pd.DatetimeIndex(["2021-02-01"], tz="UTC")),
    }
    assert np.isfinite(unit_proxy_returns(book, outside, cost_bps=0.0).to_numpy()).all()
    nan_funding = {
        "AAAUSDT": pd.Series(
            [float("nan")],
            index=pd.DatetimeIndex([pd.Timestamp("2021-01-02 01:00", tz="UTC")], tz="UTC"),
        ),
    }
    with pytest.raises(DataIntegrityError):
        unit_proxy_returns(book, nan_funding, cost_bps=0.0)


def test_assemble_account_inputs_shares_causal_adv_sigma(tmp_path: Path) -> None:
    from src.mhs.account_sources import assemble_account_inputs, causal_adv_sigma
    from src.mhs.frozen_research_candidate import FrozenMhsCandidate
    from src.mhs.frozen_research_run import FrozenSourceContext
    from src.mhs.resources import resolve_mhs_memory_budget

    sym = "AAAUSDT"
    entries = pd.DatetimeIndex(["2021-02-01", "2021-02-02"], tz="UTC")
    weights = pd.DataFrame([[0.5], [-0.5]], index=entries, columns=[sym], dtype="float64")
    candidate = FrozenMhsCandidate(
        target_weights=weights, signal_available_at=entries, strategy=FROZEN_MHS_TOP20_V2,
    )
    daily_idx = pd.date_range("2021-01-01", "2021-02-02", freq="1D", tz="UTC")
    daily_close = pd.DataFrame(100.0, index=daily_idx, columns=[sym], dtype="float64")
    daily_qv = pd.DataFrame(2e6, index=daily_idx, columns=[sym], dtype="float64")
    grid = pd.date_range(entries[0], entries[-1] + pd.Timedelta(days=1), freq="3min", inclusive="left")
    marks_dir = tmp_path / "marks" / "3m"
    marks_dir.mkdir(parents=True, exist_ok=True)
    ms = np.array([int(ts.value // 1_000_000) for ts in grid], dtype="int64")
    pd.DataFrame(
        {"timestamp": ms, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
    ).to_parquet(marks_dir / f"{sym}.parquet")
    context = FrozenSourceContext(
        census=(sym,), funding_by_symbol={}, funding_failures={},
        root=str(tmp_path / "marks"), budget=resolve_mhs_memory_budget(None),
        daily_close=daily_close, daily_quote_volume=daily_qv,
    )
    _, _, _, adv, sigma = assemble_account_inputs(candidate, context)
    expected_adv, expected_sigma = causal_adv_sigma(daily_qv, daily_close)
    pd.testing.assert_frame_equal(adv, expected_adv.reindex(entries))
    pd.testing.assert_frame_equal(sigma, expected_sigma.reindex(entries))


def test_history_validation_rejects_bad_inputs() -> None:
    good_idx = pd.date_range("2021-06-01", "2021-06-05", freq="1D", tz="UTC")
    good = pd.Series(0.001, index=good_idx, dtype="float64")
    with pytest.raises(DataIntegrityError):
        extend_unit_history(
            pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC")), good, good,
        )
    with pytest.raises(DataIntegrityError):
        extend_unit_history(
            pd.Series([0.1], index=pd.Index([0])), good, good,
        )
    with pytest.raises(DataIntegrityError):
        extend_unit_history(
            pd.Series([0.1], index=pd.DatetimeIndex(["2021-06-01"])), good, good,
        )
    with pytest.raises(DataIntegrityError):
        extend_unit_history(
            good,
            pd.Series([0.1, 0.2], index=pd.DatetimeIndex(
                ["2021-07-02", "2021-07-01"], tz="UTC",
            )),
            good,
        )
    bad_vals = good.copy()
    bad_vals.iloc[0] = float("nan")
    with pytest.raises(DataIntegrityError):
        extend_unit_history(bad_vals, good, good)
