"""Live frozen Top-20 book on a short 1h window, bit-identical to the research builder.

The live daemon recomputes the registered frozen_mhs_top20_v2 book (name clip 0.05) from
the trailing 1h panel instead of loading sealed parameter artifacts, and derives the unit
book's daily return proxy that extends the backtest unit-return bootstrap for Bayesian
Kelly sizing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.live.venue_listing import SettlementEvidence
from src.market_data.services.futures_collection import (
    FUNDING_DEFAULT_INTERVAL_MS,
    FUNDING_TIME_TOLERANCE_MS,
    infer_funding_interval_ms,
)
from src.mhs.account_sources import causal_adv_sigma
from src.mhs.books import clip_names_preserving_gross
from src.mhs.frozen_research_candidate import FROZEN_MHS_TOP20_V2, build_frozen_mhs_candidate
from src.mhs.panel import load_base_panel
from src.mhs.params import FROZEN_GROWTH_NAME_CLIP, LIVE_FROZEN_WARMUP_DAYS

_REQUIRED_COLUMNS = ("close", "quote_vol", "taker_buy_quote")


def crypto_census(disk_symbols: Iterable[str], non_crypto: frozenset[str]) -> tuple[str, ...]:
    """Sorted USDT-perpetual census for the frozen roster, excluding non-crypto underlyings.

    Tokenized equities, commodities and indices share the USDT naming convention but the
    research lake and every frozen backtest exclude them, so admitting them live would
    change the roster.
    """
    excluded = set(non_crypto)
    kept = {str(symbol) for symbol in disk_symbols if str(symbol) not in excluded}
    return tuple(sorted(kept))


@dataclass(frozen=True, slots=True)
class LiveFrozenBook:
    """Frozen unit book and the causal market inputs sizing needs, indexed by decision day."""

    unit_weights: pd.DataFrame
    snapshot_closes: pd.DataFrame
    adv: pd.DataFrame
    daily_sigma: pd.DataFrame
    valid_from: pd.Timestamp
    #: Last hourly bar actually observed on disk across the census (distinct from the
    #: requested ``panel_end``, which reaches past real collection at every nightly cycle
    #: since the decision releases before the next midnight bar closes).
    panel_last_bar: pd.Timestamp


@dataclass(frozen=True, slots=True)
class SnapshotGapReport:
    """Classification of census symbols lacking the decision-day snapshot bar in the raw source."""

    refresh_incomplete: tuple[str, ...]  # raw tail ends before the snapshot bar (not yet refreshed)
    venue_gap: tuple[str, ...]  # snapshot bar absent while a later bar exists (permanent hole)


def _raw_1h_timestamps_ms(data_root: Path, symbol: str) -> set[int]:
    """Open-time millis present in the raw 1h file, reading only the timestamp column.

    Zombie masking is deliberately not applied, so a post-delivery flat bar counts as
    present and settled symbols are never reported as gaps here.
    """
    path = Path(data_root) / "ohlcv" / "1h" / f"{symbol}.parquet"
    try:
        frame = pd.read_parquet(path, columns=["timestamp"])
    except Exception as exc:
        raise DataIntegrityError(f"snapshot gap bars unreadable: {symbol}: {exc}") from exc
    stamps = pd.to_numeric(frame["timestamp"], errors="coerce").dropna()
    return {int(value) for value in stamps.tolist()}


def classify_snapshot_gaps(
    data_root: Path, symbols: Sequence[str], *, snapshot_bar: pd.Timestamp,
) -> SnapshotGapReport:
    """Classify why each symbol lacks the raw 1h bar opening at ``snapshot_bar``.

    Reads only the ``timestamp`` column of raw files (zombie masking is not applied), so a
    post-delivery flat bar counts as present. Settled symbols are therefore never reported here;
    they are priced by the delisting settlement path instead.

    Raises:
        DataIntegrityError: a file is unreadable.
    """
    bar = pd.Timestamp(snapshot_bar)
    bar = bar.tz_localize("UTC") if bar.tzinfo is None else bar.tz_convert("UTC")
    bar_ms = int(bar.value // 1_000_000)
    refresh: list[str] = []
    venue: list[str] = []
    for symbol in symbols:
        name = str(symbol)
        stamps = _raw_1h_timestamps_ms(Path(data_root), name)
        if bar_ms in stamps:
            continue
        if stamps and max(stamps) > bar_ms:
            venue.append(name)
        else:
            refresh.append(name)
    return SnapshotGapReport(
        refresh_incomplete=tuple(sorted(refresh)), venue_gap=tuple(sorted(venue)),
    )


def snapshot_gap_blocked_decisions(
    data_root: Path, decision_index: pd.DatetimeIndex, census: Sequence[str], *, snapshot_hour: int,
) -> pd.DataFrame:
    """Withdraw a symbol's seat on each decision day whose snapshot bar is an evidenced venue gap.

    Mirrors the research source-gap contract: a decision that cannot be anchored on a published
    bar is not traded. Only venue gaps block (a later bar evidences the hole as permanent);
    refresh-incomplete days stay unblocked and are handled by the frozen step's decision-bar
    gate. Snapshot closes are never filled; the gap-blocked seat gives the symbol target
    weight 0 for that decision day. It is composed with the announced-delisting blocks by
    logical OR before being passed to ``build_frozen_mhs_candidate``.
    """
    names = [str(symbol) for symbol in census]
    stamps_by_symbol = {name: _raw_1h_timestamps_ms(Path(data_root), name) for name in names}
    top_by_symbol = {name: (max(stamps) if stamps else None) for name, stamps in stamps_by_symbol.items()}
    days = pd.DatetimeIndex(decision_index)
    values = np.zeros((len(days), len(names)), dtype=bool)
    for row, raw_day in enumerate(days):
        day = pd.Timestamp(raw_day)
        day = day.tz_localize("UTC") if day.tzinfo is None else day.tz_convert("UTC")
        bar_ms = int((day + pd.Timedelta(hours=int(snapshot_hour))).value // 1_000_000)
        for col, name in enumerate(names):
            top = top_by_symbol[name]
            if bar_ms not in stamps_by_symbol[name] and top is not None and top > bar_ms:
                values[row, col] = True
    return pd.DataFrame(values, index=days, columns=names).astype(bool)


def _require_utc(day: pd.Timestamp, label: str) -> pd.Timestamp:
    stamp = pd.Timestamp(day)
    if stamp.tzinfo is None:
        raise DataIntegrityError(f"{label} must be timezone-aware UTC")
    return stamp.tz_convert("UTC")


def build_live_frozen_book(
    data_root: Path, census: tuple[str, ...], *, panel_start: pd.Timestamp, panel_end: pd.Timestamp,
    blocked_decisions: Callable[[pd.DatetimeIndex, tuple[str, ...]], pd.DataFrame] | None = None,
) -> LiveFrozenBook:
    """Rebuild the frozen book from ``data_root/ohlcv/1h`` over ``[panel_start, panel_end)``.

    Uses the research builder and roster unchanged (FROZEN_MHS_TOP20_V2, name clip
    FROZEN_GROWTH_NAME_CLIP, full-census market close for market-relative features). Each
    hourly bar is taken as published one hour after its open, the same availability rule
    the research source loader applies. Decision rows earlier than
    ``panel_start + LIVE_FROZEN_WARMUP_DAYS`` are dropped because their roster and features
    are not yet equal to the full-history result.

    ``blocked_decisions`` builds the research-contract withdrawal frame for the builder's own
    daily decision index and final census order (announced delistings). It is a callable because
    the index and census are only known after the panel is read. Blocking withdraws the roster seat
    and the target for exactly those decision days. This deliberately deviates from the unblocked
    research book, and only for names the venue will stop trading.

    Raises:
        DataIntegrityError: census empty, panel_end - panel_start shorter than the warmup,
            or the 1h source lacks close/quote_vol/taker_buy_quote.
    """
    census_list = list(census)
    if not census_list or any(not isinstance(sym, str) or not sym for sym in census_list):
        raise DataIntegrityError("census must be a non-empty tuple of unique non-empty symbols")
    if len(set(census_list)) != len(census_list):
        raise DataIntegrityError("census must contain unique symbols")
    start = _require_utc(panel_start, "panel_start")
    end = _require_utc(panel_end, "panel_end")
    if end <= start:
        raise DataIntegrityError("panel_end must be after panel_start")
    if end - start < pd.Timedelta(days=int(LIVE_FROZEN_WARMUP_DAYS)):
        raise DataIntegrityError("panel shorter than LIVE_FROZEN_WARMUP_DAYS")
    root = str(Path(data_root) / "ohlcv")
    end_inclusive = end - pd.Timedelta(hours=1)
    try:
        panel = load_base_panel(
            root, "1h", _REQUIRED_COLUMNS, start, end_inclusive,
            partition="all", selection_mode="causal_history",
        )
    except ValueError as exc:
        raise DataIntegrityError(f"1h source unavailable: {exc}") from exc
    # 창 안에 봉이 하나도 없는 심볼(창 이전 상장폐지 등)은 패널에 없다. 로스터에 들 수 없고 시장
    # 대용치에도 기여하지 않으므로 전체 이력 결과와 같게 하려면 창에 존재하는 심볼로 좁힌다.
    census_list = [sym for sym in census_list if sym in panel["close"].columns]
    if not census_list:
        raise DataIntegrityError("no census symbol has 1h bars inside the panel window")
    close_c = panel["close"][census_list].astype("float64")
    observed = close_c.dropna(how="all").index
    if len(observed) == 0:
        raise DataIntegrityError("no observed 1h bar for any census symbol inside the panel window")
    panel_last_bar = pd.Timestamp(observed.max())
    quote_c = panel["quote_vol"][census_list].astype("float64")
    taker_c = panel["taker_buy_quote"][census_list].astype("float64")
    daily_close = close_c.resample("1D").last().astype("float64")
    daily_quote_volume = quote_c.resample("1D").sum(min_count=1).astype("float64")
    grid_1h = close_c.index
    completed = (grid_1h + pd.Timedelta(hours=1)).to_numpy(dtype="datetime64[ns]")
    hourly_available_at = pd.DataFrame(
        np.tile(completed[:, None], (1, len(census_list))),
        index=grid_1h, columns=census_list,
    ).apply(lambda col: pd.to_datetime(col).dt.tz_localize("UTC"))
    strategy = FROZEN_MHS_TOP20_V2
    hourly_panels = {"close": close_c, "quote_vol": quote_c, "taker_buy_quote": taker_c}
    blocked_frame: pd.DataFrame | None = None
    if blocked_decisions is not None:
        # The research builder validates the frame against its own daily index
        # (``daily_close.index``) and census order, so the callable receives exactly those.
        blocked_frame = blocked_decisions(pd.DatetimeIndex(daily_close.index), tuple(census_list))
    candidate = build_frozen_mhs_candidate(
        hourly_panels, hourly_available_at, daily_close, daily_quote_volume,
        tuple(census_list), market_close=close_c,
        strategy=strategy, blocked_decisions=blocked_frame,
    )
    clipped = clip_names_preserving_gross(candidate.target_weights, FROZEN_GROWTH_NAME_CLIP)
    entries = pd.DatetimeIndex(clipped.index).tz_convert("UTC")
    decisions = entries - pd.Timedelta(days=1)
    unit_weights = pd.DataFrame(
        clipped.to_numpy(dtype="float64"), index=decisions, columns=census_list, dtype="float64",
    )
    valid_from = start.normalize() + pd.Timedelta(days=int(LIVE_FROZEN_WARMUP_DAYS))
    keep = decisions >= valid_from
    decisions_kept = decisions[keep]
    unit_weights = unit_weights.loc[decisions_kept]
    snapshot_bars = decisions_kept + pd.Timedelta(hours=int(strategy.snapshot_hour_utc))
    snapshot_closes = pd.DataFrame(
        close_c.reindex(snapshot_bars).to_numpy(dtype="float64"),
        index=decisions_kept, columns=census_list, dtype="float64",
    )
    adv_daily, sigma_daily = causal_adv_sigma(daily_quote_volume, daily_close)
    adv = pd.DataFrame(
        adv_daily.reindex(decisions_kept).to_numpy(dtype="float64"),
        index=decisions_kept, columns=census_list, dtype="float64",
    )
    daily_sigma = pd.DataFrame(
        sigma_daily.reindex(decisions_kept).to_numpy(dtype="float64"),
        index=decisions_kept, columns=census_list, dtype="float64",
    )
    return LiveFrozenBook(
        unit_weights=unit_weights, snapshot_closes=snapshot_closes,
        adv=adv, daily_sigma=daily_sigma,
        valid_from=valid_from, panel_last_bar=panel_last_bar,
    )


def _funding_sum_in_window(series: pd.Series | None, start: pd.Timestamp, stop: pd.Timestamp) -> float:
    if series is None or not len(series):
        return 0.0
    idx = pd.DatetimeIndex(pd.to_datetime(series.index, utc=True))
    vals = series.to_numpy(dtype="float64")
    mask = (idx > start) & (idx <= stop)
    if not bool(mask.any()):
        return 0.0
    window = vals[mask]
    if bool(np.isnan(window).any()):
        return float("nan")
    return float(window.sum())


def _funding_coverage_ms(series: pd.Series | None) -> tuple[int | None, int]:
    """Last settlement millis and inferred interval millis for one funding series.

    A missing or empty series returns ``(None, default)``. A held symbol with no observed funding
    at all makes its window unobserved: scoring it with a 0.0 funding leg would write an
    unobserved value into the append-only proxy history, so the caller defers that day instead.
    """
    if series is None or len(series) == 0:
        return None, FUNDING_DEFAULT_INTERVAL_MS
    idx = pd.DatetimeIndex(pd.to_datetime(series.index, utc=True, errors="coerce"))
    idx = idx[~idx.isna()]
    if len(idx) == 0:
        return None, FUNDING_DEFAULT_INTERVAL_MS
    millis = [int(stamp.value // 1_000_000) for stamp in idx]
    return max(millis), infer_funding_interval_ms(millis)


def unit_proxy_returns(
    book: LiveFrozenBook, funding_by_symbol: Mapping[str, pd.Series], *, cost_bps: float,
    settlements: Mapping[str, SettlementEvidence] | None = None,
) -> pd.Series:
    """Daily unit-book return proxy on the replay ledger's anchor-to-anchor convention.

    The decision-d book is priced from the d snapshot close (the 23:00 release price, the
    replay ledger's submit-bar anchor) to the d+1 snapshot close, minus funding settled in
    (d 23:00, d+1 23:00] paid by longs, minus ``cost_bps`` on the absolute weight change against
    the previous decision. It is labelled d+2, the entry label whose replay return spans the same
    two anchors, so the forward history continues the bootstrap export without a timing seam.
    A day whose closing snapshot bar is not yet observed is skipped and re-scored next cycle.

    A held symbol whose snapshot close is missing at an anchor strictly after its evidenced
    ``delivery_time`` is priced at the settlement price for that anchor. Funding after delivery
    is zero because settlement ends the position. A held symbol missing a close without
    settlement evidence still raises.

    A decision day is scored only when every held symbol's funding series is observed through
    the day's funding window end: the last settlement must reach ``stop - inferred interval -
    FUNDING_TIME_TOLERANCE_MS`` (the interval is inferred from the series itself), unless the
    symbol settled at delivery inside the window. A series with no rows at all keeps the
    legacy 0.0 funding leg. The first unscorable day stops scoring for this call and later
    days are left for a later cycle, so the append-only history never contains a hole or a
    zero-filled funding leg. With complete funding every output is bit-identical to the
    legacy computation.

    Raises:
        DataIntegrityError: a held symbol lacks either snapshot close, or the result is non-finite.
    """
    settled: Mapping[str, SettlementEvidence] = settlements or {}
    decisions = pd.DatetimeIndex(book.unit_weights.index).tz_convert("UTC").sort_values()
    if len(decisions) == 0:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))
    snapshot_hour = int(FROZEN_MHS_TOP20_V2.snapshot_hour_utc)
    snapshot_index = pd.DatetimeIndex(book.snapshot_closes.index).tz_convert("UTC")
    snapshot_lookup = {stamp: pos for pos, stamp in enumerate(snapshot_index)}
    symbols = list(book.unit_weights.columns)
    weights = book.unit_weights.reindex(decisions)
    coverage = {name: _funding_coverage_ms(funding_by_symbol.get(name)) for name in symbols}
    out_idx: list[pd.Timestamp] = []
    out_vals: list[float] = []
    prev = np.zeros(len(symbols), dtype="float64")
    for pos, day in enumerate(decisions):
        nxt_decision = day + pd.Timedelta(days=1)
        label = day + pd.Timedelta(days=2)
        w = weights.to_numpy(dtype="float64")[pos]
        if day not in snapshot_lookup or nxt_decision not in snapshot_lookup:
            prev = w
            continue
        snapshot_bar_next = nxt_decision + pd.Timedelta(hours=snapshot_hour)
        if snapshot_bar_next > book.panel_last_bar:
            prev = w
            continue
        w = np.where(np.isfinite(w), w, 0.0)
        row_s = book.snapshot_closes.loc[day, symbols].to_numpy(dtype="float64").copy()
        row_n = book.snapshot_closes.loc[nxt_decision, symbols].to_numpy(dtype="float64").copy()
        anchor_s = day + pd.Timedelta(hours=snapshot_hour)
        anchor_n = nxt_decision + pd.Timedelta(hours=snapshot_hour)
        for col, name in enumerate(symbols):
            evidence = settled.get(name)
            if evidence is None:
                continue
            settle_price = float(evidence.price)
            if not np.isfinite(row_s[col]) and anchor_s > evidence.delivery_time:
                row_s[col] = settle_price
            if not np.isfinite(row_n[col]) and anchor_n > evidence.delivery_time:
                row_n[col] = settle_price
        held = w != 0.0
        if bool(held.any()) and (
            bool((~np.isfinite(row_s[held])).any()) or bool((~np.isfinite(row_n[held])).any())
        ):
            raise DataIntegrityError(f"snapshot close missing for held symbol at {label}")
        price_ret = 0.0
        funding_pay = 0.0
        start = day + pd.Timedelta(hours=snapshot_hour + 1)
        stop = nxt_decision + pd.Timedelta(hours=snapshot_hour + 1)
        for col, w_s in enumerate(w):
            if w_s == 0.0:
                continue
            c0 = float(row_s[col])
            c1 = float(row_n[col])
            price_ret += float(w_s) * (c1 / c0 - 1.0)
            stop_eff = stop
            evidence = settled.get(symbols[col])
            if evidence is not None and evidence.delivery_time < stop_eff:
                stop_eff = evidence.delivery_time
                if stop_eff <= start:
                    continue
            funding_pay += float(w_s) * _funding_sum_in_window(
                funding_by_symbol.get(symbols[col]), start, stop_eff,
            )
        turnover = float(np.abs(w - prev).sum()) * float(cost_bps) / 10000.0
        value = float(price_ret - funding_pay - turnover)
        if not np.isfinite(value):
            raise DataIntegrityError(f"non-finite unit proxy return at {label}")
        observed = True
        for col, w_s in enumerate(w):
            if w_s == 0.0:
                continue
            last_ms, interval_ms = coverage[symbols[col]]
            if last_ms is None:
                # 보유 종목에 관측된 펀딩이 전혀 없으면 펀딩 0으로 채점하지 않고 이 창의 proxy를 보류한다.
                observed = False
                break
            window_end = stop
            evidence = settled.get(symbols[col])
            if evidence is not None and evidence.delivery_time < window_end:
                window_end = evidence.delivery_time
                if window_end <= start:
                    continue
            if last_ms < int(window_end.value // 1_000_000) - interval_ms - FUNDING_TIME_TOLERANCE_MS:
                observed = False
                break
        if not observed:
            break
        out_idx.append(label)
        out_vals.append(value)
        prev = w
    return pd.Series(out_vals, index=pd.DatetimeIndex(out_idx, tz="UTC"), dtype="float64")


def _require_daily_series(series: pd.Series, label: str) -> pd.DatetimeIndex:
    idx = series.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise DataIntegrityError(f"{label} index must be a UTC daily DatetimeIndex")
    if idx.tz is None or str(idx.tz) != "UTC":
        raise DataIntegrityError(f"{label} index must be UTC")
    if bool(idx.duplicated().any()) or not bool(idx.is_monotonic_increasing):
        raise DataIntegrityError(f"{label} index must be sorted without duplicates")
    vals = series.to_numpy(dtype="float64")
    if len(vals) and not bool(np.isfinite(vals).all()):
        raise DataIntegrityError(f"{label} must contain only finite values")
    return idx


def extend_unit_history(bootstrap: pd.Series, forward: pd.Series, proxy: pd.Series) -> pd.Series:
    """Contiguous daily unit-return history: bootstrap, then persisted forward days, then new proxy days.

    Days already present keep their first recorded value (forward history is append-only),
    so a later panel revision never rewrites the returns sizing already used.

    Raises:
        DataIntegrityError: a missing calendar day between the bootstrap end and the newest
            day, a non-finite value, or a non-UTC/unsorted index.
    """
    if len(bootstrap) == 0:
        raise DataIntegrityError("bootstrap must be non-empty")
    _require_daily_series(bootstrap, "bootstrap")
    if len(forward):
        _require_daily_series(forward, "forward")
    if len(proxy):
        _require_daily_series(proxy, "proxy")
    merged: dict[pd.Timestamp, float] = {}
    for idx, val in bootstrap.items():
        merged[pd.Timestamp(idx).tz_convert("UTC")] = float(val)
    for idx, val in forward.items():
        stamp = pd.Timestamp(idx).tz_convert("UTC")
        if stamp not in merged:
            merged[stamp] = float(val)
    for idx, val in proxy.items():
        stamp = pd.Timestamp(idx).tz_convert("UTC")
        if stamp not in merged:
            merged[stamp] = float(val)
    ordered = sorted(merged)
    full = pd.date_range(ordered[0], ordered[-1], freq="1D", tz="UTC")
    if len(full) != len(ordered) or not bool((pd.DatetimeIndex(ordered) == full).all()):
        raise DataIntegrityError("missing calendar day between bootstrap end and newest day")
    return pd.Series([merged[stamp] for stamp in full], index=full, dtype="float64")
