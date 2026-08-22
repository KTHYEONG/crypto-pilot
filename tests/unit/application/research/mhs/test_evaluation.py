"""Contract coverage for the MHS application evaluation resource telemetry."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.application.research.mhs import evaluation as ev
import src.application.research.mhs.scaling as scaling
from src.application.research.mhs.evaluation import (
    MhsDiagnosticRequest,
)
from src.common.errors import DataIntegrityError
from src.mhs.types import BookSpec, ExecutionSpec, HorizonBand
from src.mhs.execution import SimulatedInventoryLedgerResult
from src.mhs.execution import strategy_aware_execution_replay
from src.mhs.evidence import AnchoredPurgedFold
from src.research.universe.pit_universe import symbol_partition

_START = pd.Timestamp("2021-01-01", tz="UTC")


def _write_mhs_market(
    root: Path,
    n_hours: int = 2700,
    include_btc: bool = False,
    funding_cross_sectional: bool = False,
    with_minute: bool = True,
    include_taker_buy_quote: bool = False,
) -> pd.Timestamp:
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    if include_btc:
        symbols = ["BTCUSDT", *symbols]
    hourly = pd.date_range(_START, periods=n_hours, freq="1h", tz="UTC")
    end = hourly[-1]
    rng = np.random.default_rng(20260807)
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
        drift = 1e-5 * (i - len(symbols) / 2.0)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.002, n_hours)))
        # taker_buy_quote (opt-in only) is derived without consuming rng so the
        # existing fixture prices stay byte-identical; a per-symbol constant
        # buy-ratio gives the flow_imb committee members a genuine
        # cross-sectional signal while keeping every row finite (deterministic,
        # no fillna needed). Default False keeps the shared fixture byte-identical
        # so the committee source-coverage gate tests still see the column absent.
        columns = {
            "timestamp": epoch, "open": prices, "high": prices * 1.001,
            "low": prices * 0.999, "close": prices, "quote_vol": [1000.0] * n_hours,
        }
        if include_taker_buy_quote:
            buy_ratio = 0.5 + 0.05 * (i + 1) / len(symbols)
            columns["taker_buy_quote"] = [1000.0 * buy_ratio] * n_hours
        pd.DataFrame(columns).to_parquet(hdir / f"{sym}.parquet")
        if with_minute:
            # 1m fills track the same underlying instrument as the 1h close
            # within a tight intra-hour noise band (mirrors real exchange
            # data, where an instrument's 1h close and its own minute bars
            # are the same market, not independent walks); an unrelated
            # random walk would diverge unboundedly from `prices` over the
            # fixture's multi-month window and spuriously trip
            # fill_mark_parity_mask (I1). Draws the identical
            # rng.normal(..., len(minute_idx)) shape as before so `prices`'s
            # own draws (and every other symbol's downstream draws) stay at
            # the same rng-stream position -- only this local formula changes.
            minute_noise = rng.normal(0.0, 0.0003, len(minute_idx))
            hourly_level = np.repeat(prices, 60)[: len(minute_idx)]
            mp = hourly_level * np.exp(minute_noise)
            pd.DataFrame(
                {"timestamp": minute_epoch, "open": mp, "high": mp * 1.0005,
                 "low": mp * 0.9995, "close": mp, "quote_vol": [1000.0] * len(minute_idx)},
            ).to_parquet(mdir / f"{sym}.parquet")
            mark = pd.Series(mp, index=minute_idx).resample("1h").last().reindex(hourly).to_numpy()
            pd.DataFrame(
                {"timestamp": epoch, "open": mark, "high": mark, "low": mark, "close": mark, "datetime": hourly},
            ).to_parquet(mkdir / f"{sym}.parquet")
        funding_rate = 0.00005 * (1.0 + 0.2 * i) if funding_cross_sectional else 0.00005
        pd.DataFrame(
            {"timestamp": epoch, "funding_rate": [funding_rate] * n_hours, "datetime": hourly},
        ).to_parquet(fdir / f"{sym}.parquet")
    return end


def _write_3m_cache(root: Path) -> None:
    """Derive native 3m execution bars from the fixture's 1m data (identical
    OHLC aggregation to Binance native 3m: open=first/high=max/low=min/close=last)."""
    one_dir = root / "1m"
    three_dir = root / "3m"
    three_dir.mkdir(parents=True, exist_ok=True)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    for path in sorted(one_dir.glob("*.parquet")):
        df = pd.read_parquet(path)
        idx = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        frame = pd.DataFrame(
            {"open": df["open"].to_numpy(), "high": df["high"].to_numpy(),
             "low": df["low"].to_numpy(), "close": df["close"].to_numpy()},
            index=idx,
        )
        if "quote_vol" in df.columns:
            frame["quote_vol"] = df["quote_vol"].to_numpy()
        agg = frame.resample("3min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last",
             "quote_vol": "sum"},
        ).dropna()
        cols = {"timestamp", "open", "high", "low", "close"}
        if "quote_vol" in agg.columns:
            cols.add("quote_vol")
        pd.DataFrame(
            {"timestamp": (agg.index - epoch) // pd.Timedelta("1ms"),
             **{c: agg[c].to_numpy() for c in sorted(cols - {"timestamp"})}},
        ).to_parquet(three_dir / path.name)








def _signal_disagreement_panel():
    """Momentum signal disagreement fixture (horizon=2).

    Cross-sectionally the raw ``horizon_log_return`` ranks NOISY above QUIET
    while the vol-normalized signal ranks QUIET above NOISY, so ``_book_weights``
    / ``_phase_diagnostics`` built from the two signals must differ -- the
    fixture that proves a sign=+1 dispatch to ``vol_normalized_horizon_signal``
    actually took effect.
    """
    idx = pd.date_range("2024-01-01", periods=8, freq="1h", tz="UTC")
    log_close = pd.DataFrame(
        {
            "NOISY": [0.0, 0.5, -0.2, 0.4, 0.8, 0.7, 1.1, 0.9],
            "QUIET": [0.0, 0.01, 0.04, 0.06, 0.07, 0.10, 0.11, 0.14],
        },
        index=idx,
    )
    eligible = pd.DataFrame(True, index=idx, columns=log_close.columns)
    rng = np.random.default_rng(7)
    o2o = pd.DataFrame(rng.normal(0.0, 1e-3, (len(idx), 2)), index=idx, columns=log_close.columns)
    opens = pd.DataFrame(
        100.0 * np.exp(np.cumsum(o2o.to_numpy(), axis=0)), index=idx, columns=log_close.columns,
    )
    bar_funding = pd.DataFrame(0.0, index=idx, columns=log_close.columns)
    return log_close, eligible, opens, bar_funding, idx


def _dispatch_spec(sign: int) -> BookSpec:
    band = HorizonBand(name="test_dispatch", horizons_hours=(2,), sign=sign)
    return BookSpec(band=band, horizon_hours=2, step_hours=1, min_symbols=2)


def _reference_weights(log_close, eligible, step_grid, spec) -> pd.DataFrame:
    sig = ev.horizon_log_return(log_close, spec.horizon_hours).reindex(step_grid)
    return ev.phase_tranche_book(
        ev.rank_weight_book(sig, eligible.reindex(step_grid), spec.band.sign, spec.min_symbols),
        spec.tranche_count(),
    )








def _pnl_vol_spike_returns() -> pd.Series:
    """Calm-then-high-vol daily returns with non-zero vol in each regime."""
    rng = np.random.default_rng(20260807)
    idx = pd.date_range("2024-01-01", periods=200, freq="D", tz="UTC")
    returns = np.concatenate([
        rng.normal(0.001, 0.002, 100),
        rng.normal(0.05, 0.05, 100),
    ])
    return pd.Series(returns, index=idx)









































_FOLD = AnchoredPurgedFold(
    pd.Timestamp("2021-01-01", tz="UTC"),
    pd.Timestamp("2021-01-31", tz="UTC"),
    pd.Timestamp("2021-02-10", tz="UTC"),
    pd.Timestamp("2021-04-19 08:00", tz="UTC"),
    168,
    168,
)






def _build_book_outcome_args(mhs_market) -> dict[str, object]:
    """Replicate the top-level diagnostic setup needed to invoke ``_book_outcome``."""
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ][:8]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
    )
    panel = ev.load_base_panel(
        root, "1h", ("close", "open", "quote_vol"), _START, end,
        partition="dev", min_bars=2000,
    )
    close, opens, quote_vol = panel["close"], panel["open"], panel["quote_vol"]
    grid_1h = close.index
    funded = [s for s in close.columns if s in funding_by_symbol]
    close = close[funded]
    opens = opens[funded]
    quote_vol = quote_vol[funded]
    bar_period = grid_1h[1] - grid_1h[0]
    funding_window = {
        s: funding_by_symbol[s].loc[
            (funding_by_symbol[s].index >= grid_1h[0])
            & (funding_by_symbol[s].index < grid_1h[-1] + bar_period)
        ]
        for s in funded
    }
    bar_funding = ev.bar_funding_panel(funding_window, grid_1h)
    aligned = list(bar_funding.columns)
    close = close[aligned]
    opens = opens[aligned]
    quote_vol = quote_vol[aligned]
    bar_funding = bar_funding[aligned]
    funding_by_symbol = {s: funding_by_symbol[s] for s in aligned}
    eligible = ev.liquid_half_eligibility(quote_vol, lookback_bars=720, min_history_bars=720)
    log_close = np.log(close)
    fast = ev.BOOK_SPECS["fast_reversal"]
    fast_grid = pd.date_range(_START, end, freq="6h", tz="UTC")
    w_fast = ev._book_weights(log_close, eligible, fast, fast_grid)
    phase = ev._phase_diagnostics(log_close, eligible, opens, bar_funding, grid_1h, fast)
    execution_mask = ev._pit_execution_mask(quote_vol, eligible, request.execution_universe_size)
    w_fast_execution = ev.renormalize_within_mask(
        w_fast, execution_mask.reindex(w_fast.index).fillna(False), fast.min_symbols,
    )
    return {
        "name": "fast_reversal",
        "spec": fast,
        "n_symbols": len(aligned),
        "step_grid": fast_grid,
        "weights_step": w_fast,
        "grid_1h": grid_1h,
        "opens": opens,
        "bar_funding": bar_funding,
        "phase": phase,
        "root": str(root),
        "request": request,
        "funding_by_symbol": funding_by_symbol,
        "start": _START,
        "end": end,
        "event_window_bars": fast.horizon_hours,
        "initial_equity": 1.0,
        "replay_weights_step": w_fast_execution,
    }






def _roster_mask_panel_inputs(
    root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    funding_by_symbol: dict[str, pd.Series],
    universe_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Replicate the diagnostic panel prep to independently recompute the
    execution_mask-filtered vol_mean that production must feed _regime_cash_scale."""
    panel = ev.load_base_panel(
        root, "1h", ("close", "open", "quote_vol"), start, end,
        partition="dev", min_bars=2000,
    )
    close, opens, quote_vol = panel["close"], panel["open"], panel["quote_vol"]
    grid_1h = close.index
    funded = [s for s in close.columns if s in funding_by_symbol]
    close = close[funded]
    opens = opens[funded]
    quote_vol = quote_vol[funded]
    bar_period = grid_1h[1] - grid_1h[0]
    funding_window = {
        s: funding_by_symbol[s].loc[
            (funding_by_symbol[s].index >= grid_1h[0])
            & (funding_by_symbol[s].index < grid_1h[-1] + bar_period)
        ]
        for s in funded
    }
    bar_funding = ev.bar_funding_panel(funding_window, grid_1h)
    aligned = list(bar_funding.columns)
    close = close[aligned]
    quote_vol = quote_vol[aligned]
    eligible = ev.liquid_half_eligibility(quote_vol, lookback_bars=720, min_history_bars=720)
    log_close = np.log(close)
    execution_mask = ev._pit_execution_mask(quote_vol, eligible, universe_size)
    return log_close, execution_mask, grid_1h


def _assert_regime_vol_mean_roster_masked(
    captured: dict[str, pd.Series],
    log_close: pd.DataFrame,
    execution_mask: pd.DataFrame,
    grid: pd.DatetimeIndex,
) -> None:
    """Assert the production vol_mean equals the execution_mask-filtered mean and
    genuinely excludes non-roster symbols (masked mean != full-universe mean)."""
    expected = ev.realized_vol(log_close, 48).where(execution_mask).reindex(grid).mean(axis=1)
    all_universe = ev.realized_vol(log_close, 48).reindex(grid).mean(axis=1)
    pd.testing.assert_series_equal(captured["vol_mean"], expected)
    assert int(execution_mask.sum(axis=1).max()) < execution_mask.shape[1]
    assert not expected.equals(all_universe)






















def _perf_opt_placebo_inputs(seed: int) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, BookSpec,
]:
    """Synthetic but structurally faithful placebo inputs.

    Signals are continuous log-price levels, eligibility is a per-symbol
    monotone listing lifecycle (the same shape ``liquid_half_eligibility``
    produces), and opens are NaN before listing so the active-cell ledger guard
    is exercised exactly as in production.
    """
    rng = np.random.default_rng(seed)
    n_hours, n_syms = 600, 10
    grid = pd.date_range("2023-01-01", periods=n_hours, freq="1h", tz="UTC")
    cols = [f"SYM{i}" for i in range(n_syms)]
    base = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.0003, (n_hours, n_syms)), axis=0))
    signal = pd.DataFrame(base, index=grid, columns=cols)
    listing = rng.integers(0, n_hours // 4, size=n_syms)
    elig_raw = np.arange(n_hours)[:, None] >= listing[None, :]
    eligible = pd.DataFrame(elig_raw, index=grid, columns=cols)
    opens = pd.DataFrame(
        base * (1.0 + rng.normal(0.0, 0.0001, (n_hours, n_syms))), index=grid, columns=cols,
    )
    opens = opens.mask(~elig_raw)
    bar_funding = pd.DataFrame(
        rng.normal(0.00005, 0.00001, (n_hours, n_syms)), index=grid, columns=cols,
    )
    return signal, eligible, opens, bar_funding, grid, ev.BOOK_SPECS["fast_reversal"]


def _reference_placebo_percentile(
    signal, eligible, opens, bar_funding, grid_1h, spec, observed_sharpe, n_placebos, seed,
):
    """Original pandas DataFrame-per-iteration placebo loop (baseline)."""
    from src.mhs.books import phase_tranche_book, rank_weight_book
    from src.mhs.execution import mhs_ledger_pnl

    rng = np.random.default_rng(seed)
    ranks = []
    cols = list(signal.columns)
    sig_step = signal.reindex(grid_1h)
    el_step = eligible.reindex(grid_1h)
    for _p in range(n_placebos):
        perm = rng.permutation(len(cols))
        shuffled = sig_step.copy()
        permuted_cols = [cols[i] for i in perm]
        shuffled.columns = permuted_cols
        el_shuffled = el_step.copy()
        el_shuffled.columns = permuted_cols
        weights_p = rank_weight_book(shuffled, el_shuffled, spec.band.sign, spec.min_symbols)
        weights_p = phase_tranche_book(weights_p, spec.tranche_count())
        weights_1h = weights_p.reindex(grid_1h).ffill().fillna(0.0)
        try:
            net, _t = mhs_ledger_pnl(
                weights_1h, opens[permuted_cols], bar_funding[permuted_cols], 8.0,
            )
        except DataIntegrityError:
            continue
        sd = float(net.std(ddof=1)) if len(net) > 1 else 0.0
        if sd > 0:
            ranks.append(float(net.mean() / sd * np.sqrt(ev._PERIODS_PER_YEAR_1H)))
    if not ranks:
        return None
    return float(np.mean([1.0 if observed_sharpe >= r else 0.0 for r in ranks]))




def _write_quote_volume_market(root: Path, symbols: list[str]) -> tuple[pd.DatetimeIndex, int]:
    """Write 1-minute ``quote_vol`` parquet files and return the minute grid."""
    start = pd.Timestamp("2023-02-01", tz="UTC")
    grid = pd.date_range(start, periods=6 * 24, freq="1min", tz="UTC")
    epoch = (grid - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
    base_vol = np.linspace(500.0, 900.0, len(grid))
    (root / "1m").mkdir(parents=True, exist_ok=True)
    for i, sym in enumerate(symbols):
        vol = base_vol * (1.0 + 0.1 * (i + 1)) + np.sin(np.arange(len(grid)) / 12.0) * 50.0
        pd.DataFrame({"timestamp": epoch, "quote_vol": vol}).to_parquet(
            root / "1m" / f"{sym}.parquet", index=False,
        )
    return grid, len(symbols)


def _reference_participation_warnings(replay, root, timeframe, symbols, minute_grid):
    """Original iterrows()/``.loc[t:window_end]`` participation loop (baseline)."""
    if replay.simulated_fills.empty:
        return {}
    fills = replay.simulated_fills
    notional = float((fills["quantity_delta"].abs() * fills["fill_price"]).sum())
    fills_by_symbol = {}
    for _sym, group in fills.groupby("symbol"):
        fills_by_symbol[str(_sym)] = group
    daily_volume = 0.0
    window_totals = {"1m": 0.0, "30m": 0.0}
    window_minutes = (("1m", 1), ("30m", 30))
    for sym in symbols:
        series = ev._load_symbol_quote_volume(
            root, sym, timeframe, minute_grid[0], minute_grid[-1],
        )
        if series is None:
            continue
        daily_volume += float(series.sum())
        group = fills_by_symbol.get(sym)
        if group is None:
            continue
        for _i, row in group.iterrows():
            t = row["timestamp"]
            if t not in series.index:
                continue
            for window_label, minutes in window_minutes:
                window_end = t + pd.Timedelta(minutes=minutes)
                window_totals[window_label] += float(series.loc[t:window_end].sum())
    warnings = {}
    for window_label, _minutes in window_minutes:
        total_volume = window_totals[window_label]
        warnings[f"fill_notional_to_{window_label}_quote_volume"] = (
            notional / total_volume if total_volume > 0 else float("nan")
        )
    warnings["daily_trade_notional_to_daily_quote_volume"] = (
        notional / daily_volume if daily_volume > 0 else float("nan")
    )
    return warnings




def _reference_bootstrap_ci(net, n_replicates, mean_block, seed):
    """Original scalar while-loop block bootstrap (baseline)."""
    rng = np.random.default_rng(seed)
    arr = net.to_numpy(dtype="float64")
    n = len(arr)
    means = []
    p_block = 1.0 / mean_block if mean_block > 0 else 0.0
    for _r in range(n_replicates):
        blocks = []
        while len(blocks) < n:
            start = int(rng.integers(0, n))
            length = 1
            while length < n and rng.random() > p_block:
                length += 1
            length = min(length, n - len(blocks))
            blocks.extend(arr[start : start + length].tolist())
        means.append(float(np.mean(blocks[:n])))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))












def _reference_resolve_ns_scalar(
    spos_all: np.ndarray,
    full_grid_ns: np.ndarray,
    n_grid: int,
    timeout_ns_delta: int,
) -> np.ndarray:
    """The Phase-2 scalar resolve_ns loop (the parity reference for P11)."""
    resolve_ns = np.full(len(spos_all), -1, dtype="int64")
    for i in range(len(spos_all)):
        s = int(spos_all[i])
        if s >= n_grid:
            continue
        tns = full_grid_ns[s] + timeout_ns_delta
        tpos = int(np.searchsorted(full_grid_ns, tns, side="left"))
        if tpos < n_grid and full_grid_ns[tpos] == tns:
            resolve_ns[i] = tns
    return resolve_ns




def _build_books_concurrent_args(
    mhs_market, universe_size: int | None = None,
) -> dict[str, object]:
    """Replicate the top-level diagnostic setup for all three books.

    ``universe_size`` narrows the execution roster (default 30 keeps every
    fixture symbol); a value between ``min_symbols`` and the eligible count
    exercises the renormalization that rescales surviving roster cells.
    """
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        **({"execution_universe_size": universe_size} if universe_size is not None else {}),
    )
    panel = ev.load_base_panel(
        root, "1h", ("close", "open", "quote_vol"), _START, end,
        partition="dev", min_bars=2000,
    )
    close, opens, quote_vol = panel["close"], panel["open"], panel["quote_vol"]
    grid_1h = close.index
    funded = [s for s in close.columns if s in funding_by_symbol]
    close = close[funded]
    opens = opens[funded]
    quote_vol = quote_vol[funded]
    bar_period = grid_1h[1] - grid_1h[0]
    funding_window = {
        s: funding_by_symbol[s].loc[
            (funding_by_symbol[s].index >= grid_1h[0])
            & (funding_by_symbol[s].index < grid_1h[-1] + bar_period)
        ]
        for s in funded
    }
    bar_funding = ev.bar_funding_panel(funding_window, grid_1h)
    aligned = list(bar_funding.columns)
    close = close[aligned]
    opens = opens[aligned]
    quote_vol = quote_vol[aligned]
    bar_funding = bar_funding[aligned]
    funding_by_symbol = {s: funding_by_symbol[s] for s in aligned}
    eligible = ev.liquid_half_eligibility(quote_vol, lookback_bars=720, min_history_bars=720)
    log_close = np.log(close)
    fast = ev.BOOK_SPECS["fast_reversal"]
    slow = ev.BOOK_SPECS["slow_momentum"]
    fast_grid = pd.date_range(_START, end, freq="6h", tz="UTC")
    slow_grid = pd.date_range(_START, end, freq="24h", tz="UTC")
    w_fast = ev._book_weights(log_close, eligible, fast, fast_grid)
    w_slow = ev._book_weights(log_close, eligible, slow, slow_grid)
    phase_fast = ev._phase_diagnostics(log_close, eligible, opens, bar_funding, grid_1h, fast)
    phase_slow = ev._phase_diagnostics(log_close, eligible, opens, bar_funding, grid_1h, slow)
    phase_blend = ev._phase_diagnostics(log_close, eligible, opens, bar_funding, grid_1h, fast)
    execution_mask = ev._pit_execution_mask(quote_vol, eligible, request.execution_universe_size)
    w_fast_execution = ev.renormalize_within_mask(
        w_fast, execution_mask.reindex(w_fast.index).fillna(False), fast.min_symbols,
    )
    w_slow_execution = ev.renormalize_within_mask(
        w_slow, execution_mask.reindex(w_slow.index).fillna(False), slow.min_symbols,
    )
    w_fast_1h = w_fast.reindex(grid_1h).ffill().fillna(0.0)
    w_slow_1h = w_slow.reindex(grid_1h).ffill().fillna(0.0)
    blend_1h = (
        ev.BOOK_BLEND_WEIGHTS["fast_reversal"] * w_fast_1h
        + ev.BOOK_BLEND_WEIGHTS["slow_momentum"] * w_slow_1h
    )
    vol_mean = ev.realized_vol(log_close, 48).where(execution_mask).reindex(grid_1h).mean(axis=1)
    regime_scale = scaling._regime_cash_scale(vol_mean)
    blend_1h = blend_1h.mul(regime_scale, axis=0)
    return {
        "root": str(root),
        "request": request,
        "n_symbols": len(aligned),
        "grid_1h": grid_1h,
        "fast": fast,
        "slow": slow,
        "fast_grid": fast_grid,
        "slow_grid": slow_grid,
        "w_fast": w_fast,
        "w_slow": w_slow,
        "w_fast_execution": w_fast_execution,
        "w_slow_execution": w_slow_execution,
        "opens": opens,
        "bar_funding": bar_funding,
        "phase_fast": phase_fast,
        "phase_slow": phase_slow,
        "phase_blend": phase_blend,
        "start": _START,
        "end": end,
        "funding_by_symbol": funding_by_symbol,
        "blend_1h": blend_1h,
        "execution_mask": execution_mask,
        "initial_equity": 1.0,
    }


def _sequential_book_reports(args: dict[str, object]) -> tuple[object, object, object]:
    fast, slow = args["fast"], args["slow"]
    fast_grid, slow_grid = args["fast_grid"], args["slow_grid"]
    grid_1h = args["grid_1h"]
    fast_rpt, _ = ev._book_outcome(
        "fast_reversal", fast, args["n_symbols"], fast_grid, args["w_fast"], grid_1h,
        args["opens"], args["bar_funding"], args["phase_fast"], args["root"],
        args["request"], args["funding_by_symbol"], args["start"], args["end"],
        fast.horizon_hours, args["initial_equity"], args["w_fast_execution"],
    )
    slow_rpt, _ = ev._book_outcome(
        "slow_momentum", slow, args["n_symbols"], slow_grid, args["w_slow"], grid_1h,
        args["opens"], args["bar_funding"], args["phase_slow"], args["root"],
        args["request"], args["funding_by_symbol"], args["start"], args["end"],
        slow.horizon_hours, args["initial_equity"], args["w_slow_execution"],
    )
    active_spec, active_grid = ev._active_blend_book_and_grid(fast, slow, fast_grid, slow_grid)
    blend_step = args["blend_1h"].reindex(active_grid)
    blend_replay = (
        ev.BOOK_BLEND_WEIGHTS["fast_reversal"] * args["w_fast_execution"].reindex(grid_1h).ffill().fillna(0.0)
        + ev.BOOK_BLEND_WEIGHTS["slow_momentum"] * args["w_slow_execution"].reindex(grid_1h).ffill().fillna(0.0)
    ).reindex(active_grid)
    blend_rpt, _ = ev._book_outcome(
        "blend", active_spec, args["n_symbols"], active_grid, blend_step, grid_1h,
        args["opens"], args["bar_funding"], args["phase_blend"], args["root"],
        args["request"], args["funding_by_symbol"], args["start"], args["end"],
        168, args["initial_equity"], blend_replay,
    )
    return fast_rpt, slow_rpt, blend_rpt


def _assert_books_equal(seq, con, name: str) -> None:
    assert con.name == name
    assert seq.failure is None
    assert con.failure is None
    assert seq.primary is not None
    assert con.primary is not None
    assert seq.stress is not None
    assert con.stress is not None
    assert len(con.primary.simulated_fills) == len(seq.primary.simulated_fills)
    assert con.primary_naive_sharpe == seq.primary_naive_sharpe
    assert con.primary_net_ann == seq.primary_net_ann
    assert con.primary_geometric_cagr == seq.primary_geometric_cagr
    assert con.stress_naive_sharpe == seq.stress_naive_sharpe
    pd.testing.assert_series_equal(
        con.primary.ledger.equity, seq.primary.ledger.equity,
        check_exact=True, rtol=0.0, atol=0.0,
    )
































def _committee_synthetic_panels(
    n_hours: int = 12000, n_symbols: int = 12, seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Synthetic 1h panels that admit all five committee members (per-year raw
    coverage >= 0.90) plus their 24h decision grid for the direct execution-book
    tests. 12000h keeps the 720h rolling builders above the coverage floor."""
    grid = pd.date_range("2021-01-01", periods=n_hours, freq="1h", tz="UTC")
    symbols = [f"S{i:02d}" for i in range(n_symbols)]
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 1e-4, (len(grid), len(symbols))), axis=0)),
        index=grid, columns=symbols,
    )
    quote_vol = pd.DataFrame(
        rng.uniform(900.0, 1100.0, (len(grid), len(symbols))), index=grid, columns=symbols,
    )
    taker_buy_quote = quote_vol * rng.uniform(0.4, 0.6, (len(grid), len(symbols)))
    mask = pd.DataFrame(True, index=grid, columns=symbols)
    decision_grid = pd.date_range(grid[0], grid[-1], freq="24h", tz="UTC")
    return close, quote_vol, taker_buy_quote, mask, decision_grid






































def _build_compact_report() -> ev.MhsHorizonDiagnosticReport:
    """Minimal one-book report with a real small replay for tier tests."""
    idx = pd.date_range("2021-01-01 12:01", periods=4000, freq="1min", tz="UTC")
    px = pd.DataFrame({"A": [100.0] * len(idx)}, index=idx)
    target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 11:00", tz="UTC")])
    signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 12:00", tz="UTC")])
    replay = strategy_aware_execution_replay(
        target, signal_at, px, px, px, px,
        pd.DataFrame(0.0, index=idx, columns=["A"]), 1.0,
        "OHLCV_STRICT_PROXY", ExecutionSpec(),
    )
    book = ev.MhsBookReport(
        name="fast_reversal", band="FAST", horizon_hours=24, step_hours=6,
        tranche_count=1, n_symbols=1,
        phase=ev.PhaseDiagnosticResult(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False),
        prescreen={}, tail=ev.TailSensitivityResult(
            0.0, 0.0, {}, 1, 0, 0.0, 0.0, 0.0, 0.0,
        ),
        primary=replay, stress=None,
        primary_autocorr_sharpe=0.1, primary_naive_sharpe=0.1, primary_net_ann=0.01,
        primary_geometric_cagr=0.01, primary_max_drawdown=-0.01,
        primary_annualized_turnover=1.0, stress_naive_sharpe=None,
    )
    return ev.MhsHorizonDiagnosticReport(
        feature="multi_horizon_market_state", status="COMPLETE", start="2021-01-01",
        end="2021-01-04", resolved_end="2021-01-04", partition="dev",
        execution_tiers_bps=(2.5, 5.0), books={"fast_reversal": book}, blend=None,
        blend_target_gross=0.0, blend_cash_fraction=0.0, eligible_symbols=1,
        trials_attempted=1, deflated_sharpe_ratio=None, xs_rank_ic={},
        date_clustered_regression={}, horizon_diagnostics={}, bootstrap_ci=None,
        placebo_sharpe_percentile=None,
        deployment_readiness=ev.DeploymentReadinessResult(
            0.01, -0.01, 1.0, -0.01, -0.01, -0.01, -0.01, 0, None, 0.5, 0.0, 0.0, {}, {},
            {}, False, False, False, False,
        ),
        synthetic_stress={}, participation_warnings={}, termination_counts={},
        unsupported_assumptions=(), anchored_folds=(), folds=(),
        research_go=ev.MhsResearchGoResult(False, (), 0, 0),
        fill_source="OHLCV_STRICT_PROXY", mark_source="MARK_PRICE",
        execution_timeframe="1m", execution_universe_size=1,
        execution_symbols=("A",), run_elapsed_seconds=0.1,
    )

























def _admitted_selection(selected_horizon: int | None = 360) -> ev.DiscoveryQualificationResult:
    return ev.DiscoveryQualificationResult(
        selected_horizon=selected_horizon,
        admitted=selected_horizon is not None,
        discovery_scores=() if selected_horizon is None else ((selected_horizon, 2.5),),
        discovery_aggregate_net_t=2.5 if selected_horizon is not None else None,
        qualification_net_t=2.3 if selected_horizon is not None else None,
        qualification_sign_consistent=True if selected_horizon is not None else None,
    )





















































def _committee_growth_panels(
    n_days: int = 75, seed: int = 0, discovery_vol_scale: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2022-11-01", periods=n_days, freq="1D", tz="UTC")
    rng = np.random.default_rng(seed)
    columns = list("abcde")
    gross = pd.DataFrame(
        rng.normal(0.0, 0.002, size=(n_days, len(columns))), index=idx, columns=columns,
    )
    discovery = gross.index < ev.COMMITTEE_OOS_START
    gross.loc[discovery] *= discovery_vol_scale
    tc = pd.DataFrame(
        rng.uniform(0.0, 0.005, size=(n_days, len(columns))), index=idx, columns=columns,
    )
    return gross, tc






























































def _deployment_readiness() -> ev.DeploymentReadinessResult:
    """A placeholder deployment readiness result for monkeypatched
    ``_run_post_book_concurrently`` returns (the real folds stream is skipped)."""
    return ev.DeploymentReadinessResult(
        geometric_cagr=0.0, max_drawdown=0.0, calmar=0.0, expected_shortfall=0.0,
        worst_1d=0.0, worst_7d=0.0, worst_event=0.0, time_under_water_bars=0,
        recovery_bars=None, probability_final_wealth_below_initial=0.0,
        probability_mdd_over_20pct=0.0, probability_mdd_over_30pct=0.0,
        leverage_ruin_probabilities={}, concentration={},
        participation_warnings={}, research_go_eligible=False,
        execution_go_eligible=False, pilot_go_eligible=False, scale_go_eligible=False,
    )










def _slow_book_panel_inputs(mhs_market):
    """Panel inputs for the ``_horizon_ensemble_execution_weights`` and fold tests."""
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )
    panel = ev.load_base_panel(
        str(root), "1h", ("close", "open", "quote_vol"), _START, end,
        partition="dev", min_bars=2000,
    )
    close, _opens, quote_vol = panel["close"], panel["open"], panel["quote_vol"]
    grid_1h = close.index
    funded = [s for s in close.columns if s in funding_by_symbol]
    close = close[funded]
    quote_vol = quote_vol[funded]
    eligible = ev.liquid_half_eligibility(quote_vol, lookback_bars=720, min_history_bars=720)
    log_close = np.log(close)
    execution_mask = ev._pit_execution_mask(quote_vol, eligible, request.execution_universe_size)
    return log_close, eligible, execution_mask, request, grid_1h, end


def _pre_change_slow_book(
    log_close, eligible, execution_mask, spec, step_grid, ema_span,
):
    """The frozen production slow-book chain ``_horizon_ensemble_execution_weights``
    must reproduce byte-identically in single_horizon/raw mode."""
    w = ev._book_weights(log_close, eligible, spec, step_grid, ema_span=ema_span)
    w = ev.inverse_realized_vol_tilt(
        w, ev.realized_vol(log_close, spec.horizon_hours).reindex(step_grid),
    )
    return ev.renormalize_within_mask(
        w, execution_mask.reindex(step_grid).fillna(False), spec.min_symbols,
    )












def _synthetic_ledger(
    freq: str,
    n_bars: int,
    mean_ret: float,
    vol_ret: float,
    seed: int,
) -> SimulatedInventoryLedgerResult:
    idx = pd.date_range("2021-01-01", periods=n_bars, freq=freq, tz="UTC")
    rng = np.random.default_rng(seed)
    rets = rng.normal(mean_ret, vol_ret, n_bars)
    equity = pd.Series(np.cumprod(1.0 + rets), index=idx)
    turnover = pd.Series(np.full(n_bars, 0.01), index=idx)
    return SimulatedInventoryLedgerResult(
        equity=equity,
        net_returns=equity.pct_change().dropna(),
        simulated_units=None,
        mark_to_market_pnl=pd.Series(np.zeros(n_bars), index=idx),
        funding_charge=pd.Series(np.zeros(n_bars), index=idx),
        fee_charge=pd.Series(np.zeros(n_bars), index=idx),
        fill_turnover=turnover,
        fill_source="OHLCV_IMMEDIATE_TAKER",
        mark_source="MARK_PRICE",
        primary_valid=True,
        invalid_reasons=(),
    )
















def _passing_fold_report(replay: object) -> ev.MhsFoldReport:
    return ev.MhsFoldReport(
        fold_index=0,
        validation_start="2023-01-08 00:00:00+00:00",
        validation_end="2023-12-31 00:00:00+00:00",
        strict=replay,
        stress=replay,
        primary_valid=True,
        primary_autocorr_sharpe=1.0,
        primary_naive_sharpe=1.0,
        primary_net_ann=0.1,
        primary_geometric_cagr=0.1,
        primary_max_drawdown=-0.02,
        stress_naive_sharpe=0.1,
        decision_intents=1,
        termination_counts={"MISSING_DATA": 0, "UNKNOWN_TERMINATION": 0},
        failures=(),
        strict_elapsed_seconds=0.01,
        stress_elapsed_seconds=0.01,
    )




def _gap_mixed_replay() -> object:
    idx = pd.date_range("2021-01-01 12:01", periods=31, freq="1min", tz="UTC")
    target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 11:00", tz="UTC")])
    signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 12:00", tz="UTC")])
    px = pd.DataFrame({"A": [100.0] * 31}, index=idx)
    return strategy_aware_execution_replay(
        target, signal_at, px, px, px, px,
        pd.DataFrame(0.0, index=idx, columns=["A"]), 1.0,
        "OHLCV_STRICT_PROXY", ExecutionSpec(),
    )



































# ---------------------------------------------------------------------------
# SCENARIO_MHS_EVIDENCE_WEIGHTS_BY_BOUNDARY_BUILDS_BOOKS_ONCE
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# SCENARIO_MHS_EVIDENCE_WEIGHTS_BY_BOUNDARY_DIFFERENTIATES_BY_TRAIN_END
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# SCENARIO_MHS_COMMITTEE_EXECUTION_BOOK_MEMBER_WEIGHTS_NONE_IS_IDENTICAL
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# SCENARIO_MHS_COMMITTEE_EXECUTION_BOOK_APPLIES_MEMBER_WEIGHTS
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# SCENARIO_MHS_COMMITTEE_EXECUTION_BOOK_MEMBER_WEIGHTS_FAIL_CLOSED
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# SCENARIO_MHS_FOLD_TARGET_WEIGHTS_THREADS_COMMITTEE_MEMBER_WEIGHTS
# ---------------------------------------------------------------------------











# ---------------------------------------------------------------------------
# MHS Compounding Growth: scenario references (actual tests in test_compounding_growth.py)
# SCENARIO_EXANTE_SCALE_NEVER_LEVERS_UP
# SCENARIO_EXANTE_SCALE_CAUSAL_SHIFT
# SCENARIO_EXANTE_SCALE_FAILS_CLOSED
# SCENARIO_COMMITTEE_BOOK_CARRY_REQUIRES_TARGET_GROSS
# SCENARIO_DIAGNOSTIC_DEFAULT_REQUEST_BYTE_IDENTICAL
# SCENARIO_DIAGNOSTIC_SLEEVE_WIRED_ON_BOTH_PATHS
# ---------------------------------------------------------------------------


# SCENARIO_MEMBER_ATTRIBUTION_PROXY_RANK_SPEARMAN


