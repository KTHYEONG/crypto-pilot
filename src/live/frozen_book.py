"""Live frozen Top-20 book on a short 1h window, bit-identical to the research builder.

The live daemon recomputes the registered frozen_mhs_top20_v2 book (name clip 0.05) from
the trailing 1h panel instead of loading sealed parameter artifacts, and derives the unit
book's daily return proxy that extends the backtest unit-return bootstrap for Bayesian
Kelly sizing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
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
    entry_closes: pd.DataFrame
    adv: pd.DataFrame
    daily_sigma: pd.DataFrame
    valid_from: pd.Timestamp
    #: Last hourly bar actually observed on disk across the census (distinct from the
    #: requested ``panel_end``, which reaches past real collection at every nightly cycle
    #: since the decision releases before the next midnight bar closes).
    panel_last_bar: pd.Timestamp


def _require_utc(day: pd.Timestamp, label: str) -> pd.Timestamp:
    stamp = pd.Timestamp(day)
    if stamp.tzinfo is None:
        raise DataIntegrityError(f"{label} must be timezone-aware UTC")
    return stamp.tz_convert("UTC")


def build_live_frozen_book(
    data_root: Path, census: tuple[str, ...], *, panel_start: pd.Timestamp, panel_end: pd.Timestamp,
) -> LiveFrozenBook:
    """Rebuild the frozen book from ``data_root/ohlcv/1h`` over ``[panel_start, panel_end)``.

    Uses the research builder and roster unchanged (FROZEN_MHS_TOP20_V2, name clip
    FROZEN_GROWTH_NAME_CLIP, full-census market close for market-relative features). Each
    hourly bar is taken as published one hour after its open, the same availability rule
    the research source loader applies. Decision rows earlier than
    ``panel_start + LIVE_FROZEN_WARMUP_DAYS`` are dropped because their roster and features
    are not yet equal to the full-history result.

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
    candidate = build_frozen_mhs_candidate(
        hourly_panels, hourly_available_at, daily_close, daily_quote_volume,
        tuple(census_list), market_close=close_c,
        strategy=strategy, blocked_decisions=None,
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
    entries_kept = decisions_kept + pd.Timedelta(days=1)
    entry_bars = entries_kept - pd.Timedelta(hours=1)
    entry_closes = pd.DataFrame(
        close_c.reindex(entry_bars).to_numpy(dtype="float64"),
        index=entries_kept, columns=census_list, dtype="float64",
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
        entry_closes=entry_closes, adv=adv, daily_sigma=daily_sigma,
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


def unit_proxy_returns(
    book: LiveFrozenBook, funding_by_symbol: Mapping[str, pd.Series], *, cost_bps: float,
) -> pd.Series:
    """Daily unit-book return proxy labelled by the settlement day.

    The decision-d book is held from entry d+1 to entry d+2 and labelled d+2: price return
    on entry closes, minus funding settled in (entry, next entry] paid by longs, minus
    ``cost_bps`` on the absolute weight change against the previous decision. It stands in
    for the canonical 3m unit ledger after the bootstrap ends; intraday execution path is
    not modelled.

    Raises:
        DataIntegrityError: non-finite result on a day whose book is non-empty.
    """
    decisions = pd.DatetimeIndex(book.unit_weights.index).tz_convert("UTC").sort_values()
    if len(decisions) == 0:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))
    entry_index = pd.DatetimeIndex(book.entry_closes.index).tz_convert("UTC")
    entry_lookup = {stamp: pos for pos, stamp in enumerate(entry_index)}
    symbols = list(book.unit_weights.columns)
    weights = book.unit_weights.reindex(decisions)
    entries_needed = decisions + pd.Timedelta(days=1)
    out_idx: list[pd.Timestamp] = []
    out_vals: list[float] = []
    prev = np.zeros(len(symbols), dtype="float64")
    for pos, day in enumerate(decisions):
        entry = entries_needed[pos]
        nxt = day + pd.Timedelta(days=2)
        nxt_bar = nxt - pd.Timedelta(hours=1)
        # nxt_bar 는 nxt 라벨이 참조하는 원본 시간 봉(entry_bars 관례와 동일). 아직 관측된
        # 패널 범위를 넘는다면 인과적으로 존재할 수 없는 미래 데이터이지 무결성 결함이 아니다
        # -- 조용히 건너뛰고, 다음 사이클에 그 봉이 관측되면 이 날짜를 다시 채점한다.
        if entry not in entry_lookup or nxt not in entry_lookup or nxt_bar > book.panel_last_bar:
            prev = weights.to_numpy(dtype="float64")[pos]
            continue
        w = weights.to_numpy(dtype="float64")[pos]
        w = np.where(np.isfinite(w), w, 0.0)
        row_e = book.entry_closes.loc[entry, symbols].to_numpy(dtype="float64")
        row_n = book.entry_closes.loc[nxt, symbols].to_numpy(dtype="float64")
        held = w != 0.0
        if bool(held.any()) and (
            bool((~np.isfinite(row_e[held])).any()) or bool((~np.isfinite(row_n[held])).any())
        ):
            raise DataIntegrityError(f"entry close missing for held symbol at {nxt}")
        price_ret = 0.0
        funding_pay = 0.0
        for col, w_s in enumerate(w):
            if w_s == 0.0:
                continue
            c0 = float(row_e[col])
            c1 = float(row_n[col])
            price_ret += float(w_s) * (c1 / c0 - 1.0)
            funding_pay += float(w_s) * _funding_sum_in_window(
                funding_by_symbol.get(symbols[col]), entry, nxt,
            )
        turnover = float(np.abs(w - prev).sum()) * float(cost_bps) / 10000.0
        value = float(price_ret - funding_pay - turnover)
        if not np.isfinite(value):
            raise DataIntegrityError(f"non-finite unit proxy return at {nxt}")
        out_idx.append(nxt)
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
