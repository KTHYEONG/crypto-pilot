"""Contract coverage for the MHS fast_reversal overlay redesign."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.market_data.services.futures_collection as fc
from src.application.research.mhs import evaluation as ev
import src.application.research.mhs.marks as marks
from src.application.research.mhs.evaluation import MhsDiagnosticRequest
from src.mhs.books import phase_tranche_book, rank_weight_book
from src.mhs.types import BOOK_BLEND_WEIGHTS
from src.mhs.evidence import AnchoredPurgedFold
from src.mhs.horizons import horizon_log_return
from src.research.universe.pit_universe import symbol_partition

_START = pd.Timestamp("2021-01-01", tz="UTC")
_DEV_SYMBOLS = (
    "MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT",
)
_FOLD = AnchoredPurgedFold(
    pd.Timestamp("2021-01-01", tz="UTC"),
    pd.Timestamp("2021-01-31", tz="UTC"),
    pd.Timestamp("2021-02-10", tz="UTC"),
    pd.Timestamp("2021-04-19 08:00", tz="UTC"),
    168,
    168,
)
# Trend-then-choppy fold: trending through hour 1800, then a sustained
# mean-reverting (choppy) stretch through hour 3200. Validation covers the
# choppy stretch, so the overlay must de-risk its early portion.
_CHOPPY_TREND_END_HOUR = 1800
_CHOPPY_FOLD = AnchoredPurgedFold(
    pd.Timestamp("2021-01-01", tz="UTC"),
    pd.Timestamp("2021-02-10", tz="UTC"),
    pd.Timestamp("2021-01-01", tz="UTC") + pd.Timedelta(hours=_CHOPPY_TREND_END_HOUR),
    pd.Timestamp("2021-01-01", tz="UTC") + pd.Timedelta(hours=3200),
    168,
    168,
)


def _write_market(root: Path, n_hours: int, log_price_fn, include_minute: bool = True) -> pd.Timestamp:
    symbols = [s for s in _DEV_SYMBOLS if symbol_partition(s) == "dev"]
    hourly = pd.date_range(_START, periods=n_hours, freq="1h", tz="UTC")
    end = hourly[-1]
    epoch = (hourly - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
    hdir = root / "1h"
    mdir = root / "1m"
    fdir = root / "funding"
    mkdir = root / "markPriceKlines" / "1h"
    for d in (hdir, mdir, fdir, mkdir):
        d.mkdir(parents=True, exist_ok=True)
    minute_idx = pd.date_range(_START, end, freq="1min", tz="UTC")
    minute_epoch = (minute_idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
    for i, sym in enumerate(symbols):
        log_px = log_price_fn(i)
        prices = np.exp(log_px)
        pd.DataFrame(
            {"timestamp": epoch, "open": prices, "high": prices * 1.001,
             "low": prices * 0.999, "close": prices, "quote_vol": [1000.0] * n_hours},
        ).to_parquet(hdir / f"{sym}.parquet")
        pd.DataFrame(
            {"timestamp": epoch, "funding_rate": [0.00005] * n_hours, "datetime": hourly},
        ).to_parquet(fdir / f"{sym}.parquet")
        if include_minute:
            mp = prices.repeat(60)
            fill = len(minute_idx) - len(mp)
            if fill > 0:
                mp = np.concatenate([mp, np.full(fill, mp[-1])])
            mp = mp[: len(minute_idx)]
            pd.DataFrame(
                {"timestamp": minute_epoch, "open": mp, "high": mp * 1.0005,
                 "low": mp * 0.9995, "close": mp, "quote_vol": [1000.0] * len(minute_idx)},
            ).to_parquet(mdir / f"{sym}.parquet")
            mark = pd.Series(mp, index=minute_idx).resample("1h").last().reindex(hourly).to_numpy()
            pd.DataFrame(
                {"timestamp": epoch, "open": mark, "high": mark, "low": mark,
                 "close": mark, "datetime": hourly},
            ).to_parquet(mkdir / f"{sym}.parquet")
    return end


def _random_walk_log_px(n_hours: int):
    def fn(i: int) -> np.ndarray:
        rng = np.random.default_rng(20260807 + i)
        drift = 1e-5 * (i - len([s for s in _DEV_SYMBOLS if symbol_partition(s) == "dev"]) / 2.0)
        return np.cumsum(rng.normal(drift, 0.002, n_hours))
    return fn


def _trend_choppy_log_px(n_hours: int):
    t = np.arange(n_hours, dtype="float64")

    def fn(i: int) -> np.ndarray:
        rng = np.random.default_rng(20260807 + 100 + i)
        drift = 1e-5 * (i - 4.0)
        base = np.zeros(n_hours)
        trending = t < _CHOPPY_TREND_END_HOUR
        base[trending] = 0.001 * t[trending]
        choppy_t = t[~trending] - _CHOPPY_TREND_END_HOUR
        base[~trending] = 0.001 * _CHOPPY_TREND_END_HOUR + (
            0.05 * np.sin(2 * np.pi * choppy_t / 24.0)
            + 0.001 * np.cos(2 * np.pi * choppy_t / 5.0)
            + drift * choppy_t
        )
        return base + rng.normal(0.0, 1e-5, n_hours)
    return fn


@pytest.fixture
def mhs_market(tmp_path, monkeypatch):
    root = tmp_path / "market"
    n_hours = 2700
    end = _write_market(root, n_hours, _random_walk_log_px(n_hours))
    monkeypatch.setattr(marks, "funding_path", lambda sym: root / "funding" / f"{sym}.parquet")
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet")
    # _get_symbol_mark_frame is a process-global lru_cache keyed on
    # (symbol, timeframe) only; a prior test in the same process/worker using
    # a different root with an overlapping symbol name would otherwise leak
    # stale mark data into this fixture's replay.
    ev._get_symbol_mark_frame.cache_clear()
    return root, end


@pytest.fixture
def choppy_market(tmp_path, monkeypatch):
    root = tmp_path / "choppy_market"
    n_hours = 3200
    end = _write_market(root, n_hours, _trend_choppy_log_px(n_hours), include_minute=True)
    monkeypatch.setattr(marks, "funding_path", lambda sym: root / "funding" / f"{sym}.parquet")
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet")
    # _get_symbol_mark_frame is a process-global lru_cache keyed on
    # (symbol, timeframe) only; a prior test in the same process/worker using
    # a different root with an overlapping symbol name would otherwise leak
    # stale mark data into this fixture's replay.
    ev._get_symbol_mark_frame.cache_clear()
    return root, end


def _request(root: Path, end: pd.Timestamp, **overrides) -> MhsDiagnosticRequest:
    kwargs = {
        "start": str(_START), "end": str(end), "data_root": str(root),
        "mark_mode": "cache_required", "execution_timeframe": "1m", "log_run": False,
        "execution_universe_size": 8,
    }
    kwargs.update(overrides)
    return MhsDiagnosticRequest(**kwargs)


def _fold_targets(mhs_market, request: MhsDiagnosticRequest, fold: AnchoredPurgedFold):
    root, end = mhs_market
    symbols = [s for s in _DEV_SYMBOLS if symbol_partition(s) == "dev"]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    return ev._build_fold_target_weights(str(root), fold, request, funding_by_symbol)


class TestSignalEmaSpan:
    """SCENARIO_MHS_EMA_SIGN_AWARE_FAST_UNSMOOTHED_04
    SCENARIO_MHS_EMA_SIGN_AWARE_SLOW_UNCHANGED_05"""

    def test_signal_ema_span_is_sign_aware(self) -> None:
        assert ev._signal_ema_span(-1, 48, 6) is None
        assert ev._signal_ema_span(-1, 168, 24) is None
        assert ev._signal_ema_span(1, 48, 6) == max(1, round(48 / 6 * ev.SIGNAL_EMA_HORIZON_SPAN))
        assert ev._signal_ema_span(1, 168, 24) == 7

    def test_fast_book_unsmoothed_matches_direct_construction(self) -> None:
        idx = pd.date_range(_START, periods=500, freq="1h", tz="UTC")
        rng = np.random.default_rng(20260807)
        log_close = pd.DataFrame(
            {
                sym: np.cumsum(rng.normal(0.0, 0.01, len(idx)))
                for sym in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH")
            },
            index=idx,
        )
        eligible = pd.DataFrame(True, index=idx, columns=log_close.columns)
        fast = ev.BOOK_SPECS["fast_reversal"]
        fast_grid = pd.date_range(_START, idx[-1], freq="6h", tz="UTC")

        w_fast = ev._book_weights(log_close, eligible, fast, fast_grid, ema_span=None)
        sig = horizon_log_return(log_close, fast.horizon_hours)
        sig_step = sig.reindex(fast_grid)
        expected = phase_tranche_book(
            rank_weight_book(sig_step, eligible.reindex(fast_grid), fast.band.sign, fast.min_symbols),
            fast.tranche_count(),
        )
        pd.testing.assert_frame_equal(w_fast, expected)

    def test_slow_ema_span_equals_pre_refactor_inline_formula(self) -> None:
        slow = ev.BOOK_SPECS["slow_momentum"]
        inline = max(1, round(slow.horizon_hours / slow.step_hours * ev.SIGNAL_EMA_HORIZON_SPAN))
        assert ev._signal_ema_span(slow.band.sign, slow.horizon_hours, slow.step_hours) == inline

    def test_fold_targets_byte_identical_when_ema_applied_to_both_bands(self, mhs_market) -> None:
        # The pre-refactor behavior applied the whipsaw EMA to BOTH bands; only
        # the fast-reversal-derived weights may change value. Since fast_reversal
        # carries 0% blend capital, the fold blend (pure slow_momentum) must be
        # byte-identical whether the fast band is smoothed or not.
        root, end = mhs_market
        request = _request(root, end)

        def old_inline_span(band_sign: int, horizon_hours: int, step_hours: int) -> int | None:
            return max(1, round(horizon_hours / step_hours * ev.SIGNAL_EMA_HORIZON_SPAN))

        targets_real, *_ = _fold_targets(mhs_market, request, _FOLD)
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(ev, "_signal_ema_span", old_inline_span)
        try:
            targets_old, *_ = _fold_targets(mhs_market, request, _FOLD)
        finally:
            monkeypatch.undo()
        pd.testing.assert_frame_equal(targets_real, targets_old)


class TestTrendEfficiencyOverlayDefaultOff:
    """SCENARIO_MHS_TREND_EFFICIENCY_OVERLAY_DEFAULT_OFF_06"""

    def test_request_defaults_to_false(self) -> None:
        request = MhsDiagnosticRequest()
        assert request.trend_efficiency_overlay is False
        assert MhsDiagnosticRequest(trend_efficiency_overlay=False) == request

    def test_fold_targets_byte_identical_with_flag_omitted(self, mhs_market) -> None:
        root, end = mhs_market
        default_targets, *_ = _fold_targets(mhs_market, _request(root, end), _FOLD)
        explicit_targets, *_ = _fold_targets(
            mhs_market, _request(root, end, trend_efficiency_overlay=False), _FOLD,
        )
        pd.testing.assert_frame_equal(default_targets, explicit_targets)

    def test_top_level_blend_byte_identical_with_flag_omitted(self, mhs_market, monkeypatch) -> None:
        root, end = mhs_market
        captured: dict[str, pd.DataFrame] = {}

        def _spy_books(*args, **kwargs):
            captured["blend_1h"] = args[20]
            return (None, None, None, {})

        def _spy_post(*args, **kwargs):
            return (None, None, {}, {}, (), None)

        monkeypatch.setattr(ev, "_run_books_concurrent", _spy_books)
        monkeypatch.setattr(ev, "_run_post_book_concurrently", _spy_post)
        ev.run_mhs_horizon_diagnostic(_request(root, end))
        default_blend = captured["blend_1h"].copy()
        ev.run_mhs_horizon_diagnostic(_request(root, end, trend_efficiency_overlay=False))
        explicit_blend = captured["blend_1h"]
        pd.testing.assert_frame_equal(default_blend, explicit_blend)


class TestTrendEfficiencyOverlayScalesSlowOnly:
    """SCENARIO_MHS_TREND_EFFICIENCY_OVERLAY_SCALES_SLOW_ONLY_07"""

    def test_overlay_derisks_choppy_stretch_and_preserves_invariants(self, choppy_market) -> None:
        root, end = choppy_market
        assert BOOK_BLEND_WEIGHTS["fast_reversal"] == 0.0
        # portfolio_trigger is the invariant-preserving fold rebalance mode
        # (same mode the existing fold-invariant test uses): the trigger holds
        # only whole neutral rows and the overlay scales every column by one
        # row scalar, so dollar neutrality holds exactly on every decision.
        base_targets, *_ = _fold_targets(
            choppy_market, _request(root, end, rebalance_filter="portfolio_trigger"), _CHOPPY_FOLD,
        )
        overlay_targets, *_ = _fold_targets(
            choppy_market,
            _request(root, end, rebalance_filter="portfolio_trigger", trend_efficiency_overlay=True),
            _CHOPPY_FOLD,
        )
        base_gross = base_targets.abs().sum(axis=1)
        overlay_gross = overlay_targets.abs().sum(axis=1)
        # Dollar neutrality holds on every row in both modes (scaling every
        # column by the same row scalar cannot break sum(w) == 0).
        assert base_targets.sum(axis=1).abs().max() < 1e-9
        assert overlay_targets.sum(axis=1).abs().max() < 1e-9
        # The overlay is a multiplicative de-risk scalar in [0.5, 1.0]: never
        # levers up, and it bites somewhere in the choppy stretch.
        assert (overlay_gross <= base_gross + 1e-9).all()
        assert overlay_gross.min() < base_gross.min() - 1e-6
        ratio = overlay_gross / base_gross
        assert float(ratio.min()) <= 0.8
        assert float(ratio.max()) <= 1.0 + 1e-9

    def test_overlay_does_not_touch_fast_book(self, choppy_market, monkeypatch) -> None:
        root, end = choppy_market
        recorded: dict[str, int | None] = {}
        original = ev._book_weights

        def _spy_book_weights(log_close, eligible, spec, step_grid, ema_span=None):
            if spec.band.sign == -1:
                recorded["fast_ema_span"] = ema_span
                recorded["fast_calls"] = recorded.get("fast_calls", 0) + 1
            return original(log_close, eligible, spec, step_grid, ema_span=ema_span)

        monkeypatch.setattr(ev, "_book_weights", _spy_book_weights)
        _fold_targets(choppy_market, _request(root, end), _CHOPPY_FOLD)
        n_off = recorded.get("fast_calls", 0)
        ema_off = recorded.get("fast_ema_span")
        _fold_targets(choppy_market, _request(root, end, trend_efficiency_overlay=True), _CHOPPY_FOLD)
        # The overlay flag must not alter the fast band's construction: the
        # same unsmoothed (ema_span=None) fast book is built exactly once per
        # run, unchanged between the flag-off and flag-on runs.
        assert ema_off is None
        assert recorded.get("fast_ema_span") is None
        assert recorded.get("fast_calls", 0) - n_off == 1
