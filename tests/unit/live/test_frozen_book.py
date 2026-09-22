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


def _mini_book(panel_last_bar: pd.Timestamp | None = None) -> LiveFrozenBook:
    decisions = pd.DatetimeIndex(["2021-01-01", "2021-01-02", "2021-01-03"], tz="UTC")
    cols = ["AAAUSDT", "BBBUSDT"]
    unit = pd.DataFrame(
        [[0.5, -0.5], [0.5, -0.5], [0.0, 0.0]], index=decisions, columns=cols, dtype="float64",
    )
    snap = pd.DataFrame(
        [[100.0, 100.0], [110.0, 100.0], [110.0, 100.0]],
        index=decisions, columns=cols, dtype="float64",
    )
    adv = pd.DataFrame(1e6, index=decisions, columns=cols, dtype="float64")
    sigma = pd.DataFrame(0.02, index=decisions, columns=cols, dtype="float64")
    return LiveFrozenBook(
        unit_weights=unit, snapshot_closes=snap,
        adv=adv, daily_sigma=sigma, valid_from=decisions[0],
        panel_last_bar=panel_last_bar or pd.Timestamp("2021-01-10", tz="UTC"),
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


def test_panel_last_bar_stops_at_real_data_not_requested_end(tmp_path: Path) -> None:
    # 매일 결정 실행 시각(자정 전)에는 요청한 panel_end 가 항상 실제 수집분보다 앞서 있다
    # -- panel_last_bar 는 요청 경계가 아니라 실제 관측된 마지막 봉이어야 한다.
    start, real_end = _write_panel(tmp_path, _SYMBOLS)
    requested_end = real_end + pd.Timedelta(hours=2)
    book = build_live_frozen_book(tmp_path, _SYMBOLS, panel_start=start, panel_end=requested_end)
    assert book.panel_last_bar == real_end - pd.Timedelta(hours=1)
    # 실제 수집분을 넘어서는 최신 결정일들은 예외 없이 조용히 건너뛴다.
    out = unit_proxy_returns(book, {}, cost_bps=0.0)
    assert np.isfinite(out.to_numpy()).all()
    assert len(out) > 0
    assert out.index.max() <= book.panel_last_bar + pd.Timedelta(hours=1)


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
    cols = ["AAAUSDT", "BBBUSDT"]
    unit = pd.DataFrame(
        [[0.0, 0.0], [0.2, -0.2], [0.2, -0.2]], index=decisions, columns=cols, dtype="float64",
    )
    snap = pd.DataFrame(100.0, index=decisions, columns=cols, dtype="float64")
    book = LiveFrozenBook(
        unit_weights=unit, snapshot_closes=snap,
        adv=snap.copy(), daily_sigma=snap.copy(), valid_from=decisions[0],
        panel_last_bar=pd.Timestamp("2021-01-10", tz="UTC"),
    )
    out = unit_proxy_returns(book, {}, cost_bps=2.0)
    assert out.iloc[1] == pytest.approx(-0.00008)


def test_held_symbol_with_missing_snapshot_close_fails_closed() -> None:
    book = _mini_book()
    book.snapshot_closes.iloc[1, 0] = float("nan")
    with pytest.raises(DataIntegrityError):
        unit_proxy_returns(book, {}, cost_bps=0.0)


def test_forward_bar_past_panel_frontier_is_skipped_not_raised() -> None:
    # day=2021-01-02 는 closing snapshot bar(2021-01-03 22:00)가 아직 관측 안 된 미래이므로
    # 조용히 건너뛴다 -- 매일 자정 직전 실행되는 라이브 사이클의 정상 상황 회귀 가드.
    book = _mini_book(panel_last_bar=pd.Timestamp("2021-01-03 00:00", tz="UTC"))
    out = unit_proxy_returns(book, {}, cost_bps=0.0)
    assert list(out.index) == [pd.Timestamp("2021-01-03", tz="UTC")]


def test_genuine_gap_within_observed_history_still_fails_closed() -> None:
    # panel_last_bar 는 넉넉한데도 특정 보유 심볼 값만 NaN이면 -- 미래가 아니라 진짜 결손
    # -- 여전히 fail-closed 해야 한다(수집 실패를 조용히 넘어가지 않는다).
    book = _mini_book()
    book.snapshot_closes.iloc[1, 0] = float("nan")
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
    book.snapshot_closes.iloc[:, 1] = float("nan")
    book.unit_weights.iloc[:, 1] = 0.0
    book.unit_weights.iloc[1, 1] = 0.0
    out = unit_proxy_returns(book, {}, cost_bps=0.0)
    assert np.isfinite(out.to_numpy()).all()
    empty_idx = pd.DatetimeIndex([], tz="UTC")
    empty = LiveFrozenBook(
        unit_weights=pd.DataFrame(columns=["AAAUSDT"], index=empty_idx, dtype="float64"),
        snapshot_closes=pd.DataFrame(columns=["AAAUSDT"], index=empty_idx, dtype="float64"),
        adv=pd.DataFrame(columns=["AAAUSDT"], index=empty_idx, dtype="float64"),
        daily_sigma=pd.DataFrame(columns=["AAAUSDT"], index=empty_idx, dtype="float64"),
        valid_from=_START, panel_last_bar=_START,
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
    _, _, _, adv, sigma, anchors = assemble_account_inputs(candidate, context)
    expected_adv, expected_sigma = causal_adv_sigma(daily_qv, daily_close)
    decisions = entries - pd.Timedelta(days=1)
    expected_adv = expected_adv.reindex(decisions)
    expected_adv.index = entries
    expected_sigma = expected_sigma.reindex(decisions)
    expected_sigma.index = entries
    pd.testing.assert_frame_equal(adv, expected_adv)
    pd.testing.assert_frame_equal(sigma, expected_sigma)
    pd.testing.assert_index_equal(anchors, pd.DatetimeIndex(entries))


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


def _assemble_context(tmp_path: Path, sym: str, entries: pd.DatetimeIndex, releases: pd.DatetimeIndex, grid: pd.DatetimeIndex, funding_by_symbol: dict | None = None):
    from src.mhs.account_sources import assemble_account_inputs
    from src.mhs.frozen_research_candidate import FrozenMhsCandidate
    from src.mhs.frozen_research_run import FrozenSourceContext
    from src.mhs.resources import resolve_mhs_memory_budget
    import numpy as np

    weights = pd.DataFrame([[0.5]] * len(entries), index=entries, columns=[sym], dtype="float64")
    candidate = FrozenMhsCandidate(
        target_weights=weights, signal_available_at=releases, strategy=FROZEN_MHS_TOP20_V2,
    )
    marks_dir = tmp_path / "marks2" / "3m"
    marks_dir.mkdir(parents=True, exist_ok=True)
    ms = np.array([int(ts.value // 1_000_000) for ts in grid], dtype="int64")
    pd.DataFrame(
        {"timestamp": ms, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
    ).to_parquet(marks_dir / f"{sym}.parquet")
    daily_idx = pd.date_range(pd.Timestamp("2021-01-01", tz="UTC"), entries[-1] + pd.Timedelta(days=1), freq="1D", tz="UTC")
    daily_close = pd.DataFrame(100.0, index=daily_idx, columns=[sym], dtype="float64")
    daily_qv = pd.DataFrame(2e6, index=daily_idx, columns=[sym], dtype="float64")
    context = FrozenSourceContext(
        census=(sym,), funding_by_symbol=dict(funding_by_symbol or {}), funding_failures={},
        root=str(tmp_path / "marks2"), budget=resolve_mhs_memory_budget(None),
        daily_close=daily_close, daily_quote_volume=daily_qv,
    )
    return assemble_account_inputs(candidate, context)


def test_assembled_anchors_follow_release_time(tmp_path: Path) -> None:
    sym = "AAAUSDT"
    entries = pd.DatetimeIndex(["2021-02-01", "2021-02-02"], tz="UTC")
    releases = pd.DatetimeIndex(entries - pd.Timedelta(hours=1))
    grid = pd.date_range(releases[0], entries[-1] + pd.Timedelta(days=1), freq="3min", inclusive="left")
    unit, marks, _, adv, _, anchors = _assemble_context(tmp_path, sym, entries, releases, grid)
    pd.testing.assert_index_equal(anchors, releases)
    assert marks.close.index[0] == releases[0]
    assert len(anchors) == len(unit)


def test_assembled_funding_sampled_at_anchors(tmp_path: Path) -> None:
    sym = "AAAUSDT"
    entries = pd.DatetimeIndex(["2021-02-01", "2021-02-02"], tz="UTC")
    releases = pd.DatetimeIndex(entries - pd.Timedelta(hours=1))
    grid = pd.date_range(releases[0], entries[-1] + pd.Timedelta(days=1), freq="3min", inclusive="left")
    funding = pd.Series([0.001], index=pd.DatetimeIndex([entries[0]], tz="UTC"), dtype="float64")
    _, _, funding_cum, _, _, anchors = _assemble_context(
        tmp_path, sym, entries, releases, grid, funding_by_symbol={sym: funding},
    )
    assert anchors[0] == releases[0]
    assert funding_cum.loc[entries[0], sym] == 0.0
    assert funding_cum.loc[entries[1], sym] == 0.001


def test_assembled_shared_anchor_fails_closed(tmp_path: Path) -> None:
    import numpy as np

    from src.mhs.account_sources import assemble_account_inputs
    from src.mhs.frozen_research_candidate import FrozenMhsCandidate
    from src.mhs.frozen_research_run import FrozenSourceContext
    from src.mhs.resources import resolve_mhs_memory_budget

    sym = "AAAUSDT"
    entries = pd.DatetimeIndex(["2021-02-01", "2021-02-02"], tz="UTC")
    releases = pd.DatetimeIndex([entries[0] - pd.Timedelta(hours=1), entries[0] - pd.Timedelta(minutes=59)])
    weights = pd.DataFrame([[0.5], [0.5]], index=entries, columns=[sym], dtype="float64")
    candidate = FrozenMhsCandidate(
        target_weights=weights, signal_available_at=releases, strategy=FROZEN_MHS_TOP20_V2,
    )
    grid = pd.date_range(releases[0], entries[-1] + pd.Timedelta(days=1), freq="3min", inclusive="left")
    marks_dir = tmp_path / "marks3" / "3m"
    marks_dir.mkdir(parents=True, exist_ok=True)
    ms = np.array([int(ts.value // 1_000_000) for ts in grid], dtype="int64")
    pd.DataFrame(
        {"timestamp": ms, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
    ).to_parquet(marks_dir / f"{sym}.parquet")
    daily_idx = pd.date_range(pd.Timestamp("2021-01-01", tz="UTC"), entries[-1] + pd.Timedelta(days=1), freq="1D", tz="UTC")
    context = FrozenSourceContext(
        census=(sym,), funding_by_symbol={}, funding_failures={},
        root=str(tmp_path / "marks3"), budget=resolve_mhs_memory_budget(None),
        daily_close=pd.DataFrame(100.0, index=daily_idx, columns=[sym], dtype="float64"),
        daily_quote_volume=pd.DataFrame(2e6, index=daily_idx, columns=[sym], dtype="float64"),
    )
    with pytest.raises(DataIntegrityError):
        assemble_account_inputs(candidate, context)


def test_assembled_release_before_first_grid_bar_fails_closed(tmp_path: Path) -> None:
    import numpy as np

    from src.mhs.account_sources import assemble_account_inputs
    from src.mhs.frozen_research_candidate import FrozenMhsCandidate
    from src.mhs.frozen_research_run import FrozenSourceContext
    from src.mhs.resources import resolve_mhs_memory_budget

    sym = "AAAUSDT"
    entries = pd.DatetimeIndex(["2021-02-01", "2021-02-02"], tz="UTC")
    releases = pd.DatetimeIndex(
        [pd.Timestamp("2021-02-01", tz="UTC"), pd.Timestamp("2021-01-15", tz="UTC")]
    )
    weights = pd.DataFrame([[0.5], [0.5]], index=entries, columns=[sym], dtype="float64")
    candidate = FrozenMhsCandidate(
        target_weights=weights, signal_available_at=releases, strategy=FROZEN_MHS_TOP20_V2,
    )
    grid = pd.date_range(releases[0].floor("3min"), entries[-1] + pd.Timedelta(days=1), freq="3min", inclusive="left")
    marks_dir = tmp_path / "marks4" / "3m"
    marks_dir.mkdir(parents=True, exist_ok=True)
    ms = np.array([int(ts.value // 1_000_000) for ts in grid], dtype="int64")
    pd.DataFrame(
        {"timestamp": ms, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
    ).to_parquet(marks_dir / f"{sym}.parquet")
    daily_idx = pd.date_range(pd.Timestamp("2021-01-01", tz="UTC"), entries[-1] + pd.Timedelta(days=1), freq="1D", tz="UTC")
    context = FrozenSourceContext(
        census=(sym,), funding_by_symbol={}, funding_failures={},
        root=str(tmp_path / "marks4"), budget=resolve_mhs_memory_budget(None),
        daily_close=pd.DataFrame(100.0, index=daily_idx, columns=[sym], dtype="float64"),
        daily_quote_volume=pd.DataFrame(2e6, index=daily_idx, columns=[sym], dtype="float64"),
    )
    with pytest.raises(DataIntegrityError):
        assemble_account_inputs(candidate, context)


def _assemble_with_daily(
    tmp_path: Path, name: str, daily_close: pd.DataFrame, daily_qv: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.mhs.account_sources import assemble_account_inputs
    from src.mhs.frozen_research_candidate import FrozenMhsCandidate
    from src.mhs.frozen_research_run import FrozenSourceContext
    from src.mhs.resources import resolve_mhs_memory_budget

    sym = str(daily_close.columns[0])
    entries = pd.DatetimeIndex(["2021-02-01", "2021-02-02"], tz="UTC")
    weights = pd.DataFrame([[0.5], [-0.5]], index=entries, columns=[sym], dtype="float64")
    candidate = FrozenMhsCandidate(
        target_weights=weights, signal_available_at=entries, strategy=FROZEN_MHS_TOP20_V2,
    )
    grid = pd.date_range(entries[0], entries[-1] + pd.Timedelta(days=1), freq="3min", inclusive="left")
    marks_dir = tmp_path / name / "3m"
    marks_dir.mkdir(parents=True, exist_ok=True)
    ms = np.array([int(ts.value // 1_000_000) for ts in grid], dtype="int64")
    pd.DataFrame(
        {"timestamp": ms, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
    ).to_parquet(marks_dir / f"{sym}.parquet")
    context = FrozenSourceContext(
        census=(sym,), funding_by_symbol={}, funding_failures={},
        root=str(tmp_path / name), budget=resolve_mhs_memory_budget(None),
        daily_close=daily_close, daily_quote_volume=daily_qv,
    )
    _, _, _, adv, sigma, _ = assemble_account_inputs(candidate, context)
    return adv, sigma


def test_replay_adv_uses_decision_day(tmp_path: Path) -> None:
    sym = "AAAUSDT"
    entry = pd.Timestamp("2021-02-02", tz="UTC")
    daily_idx = pd.date_range("2021-01-01", "2021-02-02", freq="1D", tz="UTC")
    base_qv = pd.DataFrame(2e6, index=daily_idx, columns=[sym], dtype="float64")
    jumped_qv = base_qv.copy()
    jumped_qv.loc[entry, sym] = 2e8
    daily_close = pd.DataFrame(100.0, index=daily_idx, columns=[sym], dtype="float64")
    adv_base, _ = _assemble_with_daily(tmp_path, "adv_base", daily_close, base_qv)
    adv_jump, _ = _assemble_with_daily(tmp_path, "adv_jump", daily_close, jumped_qv)
    assert adv_base.loc[entry, sym] == pytest.approx(adv_jump.loc[entry, sym])


def test_future_daily_bar_never_moves_replay_adv(tmp_path: Path) -> None:
    sym = "AAAUSDT"
    entry = pd.Timestamp("2021-02-02", tz="UTC")
    daily_idx = pd.date_range("2021-01-01", "2021-02-02", freq="1D", tz="UTC")
    close_a = pd.DataFrame(100.0, index=daily_idx, columns=[sym], dtype="float64")
    close_b = close_a.copy()
    close_b.loc[entry, sym] = 150.0
    daily_qv = pd.DataFrame(2e6, index=daily_idx, columns=[sym], dtype="float64")
    adv_a, sigma_a = _assemble_with_daily(tmp_path, "fut_a", close_a, daily_qv)
    adv_b, sigma_b = _assemble_with_daily(tmp_path, "fut_b", close_b, daily_qv)
    pd.testing.assert_series_equal(adv_a.loc[entry], adv_b.loc[entry], check_names=False)
    pd.testing.assert_series_equal(sigma_a.loc[entry], sigma_b.loc[entry], check_names=False)


def test_proxy_prices_snapshot_to_snapshot() -> None:
    decisions = pd.DatetimeIndex(["2021-01-01", "2021-01-02", "2021-01-03"], tz="UTC")
    cols = ["AAAUSDT"]
    unit = pd.DataFrame([[1.0], [1.0], [1.0]], index=decisions, columns=cols, dtype="float64")
    snap = pd.DataFrame(
        [[100.0], [110.0], [110.0]], index=decisions, columns=cols, dtype="float64",
    )
    book = LiveFrozenBook(
        unit_weights=unit, snapshot_closes=snap,
        adv=pd.DataFrame(1e6, index=decisions, columns=cols, dtype="float64"),
        daily_sigma=pd.DataFrame(0.02, index=decisions, columns=cols, dtype="float64"),
        valid_from=decisions[0], panel_last_bar=pd.Timestamp("2021-01-10", tz="UTC"),
    )
    out = unit_proxy_returns(book, {}, cost_bps=0.0)
    assert out.loc[pd.Timestamp("2021-01-03", tz="UTC")] == pytest.approx(0.10)


def test_proxy_funding_window_is_snapshot_to_snapshot() -> None:
    decisions = pd.DatetimeIndex(["2021-01-01", "2021-01-02", "2021-01-03"], tz="UTC")
    cols = ["AAAUSDT"]
    unit = pd.DataFrame([[1.0], [1.0], [1.0]], index=decisions, columns=cols, dtype="float64")
    snap = pd.DataFrame(100.0, index=decisions, columns=cols, dtype="float64")
    adv = pd.DataFrame(1e6, index=decisions, columns=cols, dtype="float64")
    sigma = pd.DataFrame(0.02, index=decisions, columns=cols, dtype="float64")
    book = LiveFrozenBook(
        unit_weights=unit, snapshot_closes=snap, adv=adv, daily_sigma=sigma,
        valid_from=decisions[0], panel_last_bar=pd.Timestamp("2021-01-10", tz="UTC"),
    )
    day = decisions[0]
    nxt = decisions[1]
    funding = {
        "AAAUSDT": pd.Series(
            [0.01, 0.02, 0.03, 0.04],
            index=pd.DatetimeIndex(
                [day + pd.Timedelta(hours=23), nxt + pd.Timedelta(hours=8),
                 nxt + pd.Timedelta(hours=23), nxt + pd.Timedelta(days=1)],
                tz="UTC",
            ),
            dtype="float64",
        ),
    }
    out = unit_proxy_returns(book, funding, cost_bps=0.0)
    assert out.loc[day + pd.Timedelta(days=2)] == pytest.approx(-(0.02 + 0.03))


def test_unobserved_closing_snapshot_is_skipped() -> None:
    decisions = pd.DatetimeIndex(["2021-01-01", "2021-01-02"], tz="UTC")
    cols = ["AAAUSDT"]
    unit = pd.DataFrame([[1.0], [1.0]], index=decisions, columns=cols, dtype="float64")
    snap = pd.DataFrame([[100.0], [110.0]], index=decisions, columns=cols, dtype="float64")
    adv = pd.DataFrame(1e6, index=decisions, columns=cols, dtype="float64")
    sigma = pd.DataFrame(0.02, index=decisions, columns=cols, dtype="float64")
    book = LiveFrozenBook(
        unit_weights=unit, snapshot_closes=snap, adv=adv, daily_sigma=sigma,
        valid_from=decisions[0],
        panel_last_bar=pd.Timestamp("2021-01-02 12:00", tz="UTC"),
    )
    out = unit_proxy_returns(book, {}, cost_bps=0.0)
    assert len(out) == 0


def test_proxy_matches_replay_anchor_to_anchor() -> None:
    from src.market_data.binance.venue_rules import VenueBracket, VenueRuleSnapshot, VenueSymbolRules

    from src.mhs.account_ledger import AccountMarkPanels, replay_account
    from src.mhs.account_policy import ExposurePolicy

    sym = "AAAUSDT"
    entries = pd.DatetimeIndex(["2021-01-03", "2021-01-04"], tz="UTC")
    releases = entries - pd.Timedelta(hours=1)
    grid = pd.date_range(releases[0], entries[-1] + pd.Timedelta(days=1), freq="3min", inclusive="left")
    closes = np.where(grid >= releases[1], 110.0, 100.0)
    marks = AccountMarkPanels(
        close=pd.DataFrame({sym: closes}, index=grid, dtype="float64"),
        high=pd.DataFrame({sym: closes}, index=grid, dtype="float64"),
        low=pd.DataFrame({sym: closes}, index=grid, dtype="float64"),
    )
    unit_weights = pd.DataFrame([[1.0], [1.0]], index=entries, columns=[sym], dtype="float64")
    funding_cum = pd.DataFrame(0.0, index=entries, columns=[sym], dtype="float64")
    adv = pd.DataFrame(1e12, index=entries, columns=[sym], dtype="float64")
    sigma = pd.DataFrame(0.02, index=entries, columns=[sym], dtype="float64")
    rules = VenueRuleSnapshot(
        captured_at=pd.Timestamp("2021-01-01", tz="UTC"),
        symbols={sym: VenueSymbolRules(
            symbol=sym, brackets=(VenueBracket(0.0, 1e12, 0.0, 0.0, 1000),),
            step_size=1e-9, min_notional=0.0,
        )},
    )
    policy = ExposurePolicy(
        kind="fixed", exposure_max=1.0, exposure_step=0.01, mean_haircut=0.0,
        prior_days=730.0, min_moment_days=30, shock_per_unit=0.0,
        margin_reserve=0.0, initial_margin_cap=10.0, impact_y=0.0,
    )
    result = replay_account(
        unit_weights, marks, funding_cum, adv, sigma, rules, policy,
        anchor_times=releases, capital=1e6, taker_fee_bps=0.0,
        apply_order_filters=False, execution="taker",
    )
    replay_ret = float(result.daily_equity.loc[entries[1]] / result.daily_equity.loc[entries[0]] - 1.0)
    decisions = pd.DatetimeIndex(["2021-01-02", "2021-01-03"], tz="UTC")
    book = LiveFrozenBook(
        unit_weights=pd.DataFrame([[1.0], [1.0]], index=decisions, columns=[sym], dtype="float64"),
        snapshot_closes=pd.DataFrame(
            [[100.0], [110.0]], index=decisions, columns=[sym], dtype="float64",
        ),
        adv=pd.DataFrame(1e6, index=decisions, columns=[sym], dtype="float64"),
        daily_sigma=pd.DataFrame(0.02, index=decisions, columns=[sym], dtype="float64"),
        valid_from=decisions[0], panel_last_bar=pd.Timestamp("2021-01-10", tz="UTC"),
    )
    proxy = unit_proxy_returns(book, {}, cost_bps=0.0)
    assert proxy.loc[entries[1]] == pytest.approx(replay_ret, rel=1e-9)
