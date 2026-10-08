"""Golden freeze of live decisions across the research/lab restructure (parts 1-4).

A deterministic synthetic 1h lake (30 seeded symbols, one listed mid-window, one
delisted with a listing-registry fixture) feeds the public live entry point
``build_live_strategy_book`` and the account sizing path used by
``run_strategy_signal_step``. SHA-256 digests of the canonical unit-target and
levered-target frames pin every live decision bit; part 3 may only rename
imports/symbols here, never the digests.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import pytest

from src.live.deployed_weights import load_weights_frame
from src.live.strategy_book import LiveStrategyBook, build_live_strategy_book
from src.live.strategy_signal import run_strategy_signal_step
from src.live.venue_listing import (
    VenueListingEntry,
    VenueListingSnapshot,
    delisting_blocked_decisions,
    load_venue_listing_history,
    write_venue_listing_snapshot,
)
from src.market_data.binance.venue_rules import load_venue_rule_snapshot
from src.strategy.sizing import (
    account_growth_policy,
    bayesian_unit_moments,
    build_venue_ladders,
    choose_exposure,
)
from src.strategy.targets import FLOW_MOM_TOP20
from src.core.params import ACCOUNT_MIN_MOMENT_DAYS, ACCOUNT_PRIOR_DAYS

_START = pd.Timestamp("2021-01-01", tz="UTC")
_DAYS = 150
_SYMS = tuple(f"G{i:02d}USDT" for i in range(30))
_LATE = "G30USDT"
_DLIST = "GDLUSDT"
_CENSUS = tuple(sorted((*_SYMS, _LATE, _DLIST)))
_SETTLE = 0.214
_DELIVERY = _START + pd.Timedelta(days=136)
_FIRST_SEEN = _START + pd.Timedelta(days=130)
_FAR = pd.Timestamp("2049-01-01", tz="UTC")
_SNAPSHOT_HOUR = int(FLOW_MOM_TOP20.snapshot_hour_utc)
_SEED_EQUITY_USDT = 2100.0

_UNIT_BOOK_DIGEST = "d26b96855df02f7f532b7785926b80f679c9e431ea942137b07880122f66c447"
_LEVERED_ROW_DIGEST = "311cc2767be0ecd99788ad84e0eb1ef49c9c8e5cc25ebdb5efd47af5547a758a"
_LIVE_STEP_DIGEST = "e7a9ec1cdf90c4e1510dabae5948cb91af82303814d05e7278a2c81e1f247a73"


class GoldenLayout(TypedDict):
    data: Path
    listing: Path
    venue: Path
    book: LiveStrategyBook
    bootstrap: Path


def _write_panel(root: Path) -> None:
    n = _DAYS * 24
    grid = pd.date_range(_START, periods=n, freq="1h", tz="UTC")
    out = root / "ohlcv" / "1h"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261008)
    ms = np.array([int(ts.value // 1_000_000) for ts in grid], dtype="int64")
    for j, sym in enumerate(_SYMS):
        rets = rng.normal(0, 0.005, size=n)
        close = 100.0 * (1.0 + 0.01 * j) * np.exp(np.cumsum(rets))
        qv = 100_000.0 + rng.uniform(0, 20_000.0, size=n)
        tbq = np.clip(qv * 0.5 * (1.0 + rng.normal(0, 0.02, size=n)), 0.0, None)
        pd.DataFrame(
            {"timestamp": ms, "close": close, "quote_vol": qv, "taker_buy_quote": tbq},
        ).to_parquet(out / f"{sym}.parquet")
    late_from = 30 * 24
    rets = rng.normal(0.006, 0.005, size=n - late_from)
    close = 50.0 * np.exp(np.cumsum(rets))
    qv = 500_000.0 + rng.uniform(0, 50_000.0, size=n - late_from)
    tbq = np.clip(qv * 0.5 * (1.0 + rng.normal(0, 0.02, size=n - late_from)), 0.0, None)
    pd.DataFrame(
        {"timestamp": ms[late_from:], "close": close, "quote_vol": qv, "taker_buy_quote": tbq},
    ).to_parquet(out / f"{_LATE}.parquet")
    delivery_pos = int((_DELIVERY - grid[0]) / pd.Timedelta(hours=1))
    walk = np.exp(np.cumsum(rng.normal(0, 0.005, size=delivery_pos)))
    pre = walk * (2.0 / walk[-1])
    close = np.concatenate([pre, np.full(n - delivery_pos, _SETTLE)])
    volume = np.concatenate([np.full(delivery_pos, 1000.0), np.zeros(n - delivery_pos)])
    qv = np.full(n, 110_000.0)
    tbq = np.full(n, 55_000.0)
    pd.DataFrame(
        {"timestamp": ms, "open": close, "high": close, "low": close,
         "close": close, "volume": volume, "quote_vol": qv, "taker_buy_quote": tbq},
    ).to_parquet(out / f"{_DLIST}.parquet")


def _write_listing(root: Path) -> None:
    for day in range(100, _DAYS + 1):
        slot = _START.normalize() + pd.Timedelta(days=day)
        entries = {
            sym: VenueListingEntry(
                symbol=sym, status="TRADING", contract_type="PERPETUAL",
                underlying_type="COIN", quote_asset="USDT",
                delivery_time=_FAR, announced_delisting=False, delisting_first_seen_at=None,
            )
            for sym in _CENSUS
        }
        entries[_DLIST] = VenueListingEntry(
            symbol=_DLIST,
            status="SETTLING" if slot >= _DELIVERY.normalize() else "TRADING",
            contract_type="PERPETUAL", underlying_type="COIN", quote_asset="USDT",
            delivery_time=_DELIVERY if slot >= _FIRST_SEEN else _FAR,
            announced_delisting=slot >= _FIRST_SEEN,
            delisting_first_seen_at=(
                _FIRST_SEEN + pd.Timedelta(hours=1) if slot >= _FIRST_SEEN else None
            ),
        )
        write_venue_listing_snapshot(
            VenueListingSnapshot(captured_at=slot + pd.Timedelta(hours=1), entries=entries),
            root, slot_day=slot,
        )


def _write_venue(path: Path) -> None:
    payload = {
        "captured_at": "2021-04-30T00:00:00+00:00",
        "symbols": {
            symbol: {
                "brackets": [
                    {"notional_floor": 0.0, "notional_cap": 50000.0,
                     "maint_margin_ratio": 0.004, "maint_amount": 0.0, "initial_leverage": 20},
                    {"notional_floor": 50000.0, "notional_cap": 100000000.0,
                     "maint_margin_ratio": 0.008, "maint_amount": 10.0, "initial_leverage": 20},
                ],
                "step_size": None, "min_notional": None,
            }
            for symbol in _CENSUS
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_bootstrap(path: Path, first_decision: pd.Timestamp) -> None:
    idx = pd.date_range(_START, first_decision - pd.Timedelta(days=1), freq="1D", tz="UTC")
    assert len(idx) >= int(ACCOUNT_MIN_MOMENT_DAYS)
    values = np.random.default_rng(11).normal(0.0008, 0.008, size=len(idx))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(values, index=idx, dtype="float64").to_frame("unit_return").to_parquet(path, index=True)


def _write_funding(root: Path) -> None:
    out = root / "funding"
    out.mkdir()
    grid = pd.date_range(_START, periods=_DAYS * 3, freq="8h", tz="UTC")
    for position, symbol in enumerate(_CENSUS):
        index = grid[grid <= _DELIVERY] if symbol == _DLIST else grid
        if symbol == _LATE:
            index = index[index >= _START + pd.Timedelta(days=30)]
        pd.DataFrame({
            "datetime": index,
            "funding_rate": np.full(len(index), (position % 5 - 2) * 0.00001),
        }).to_parquet(out / f"{symbol}.parquet")


def _canonical_digest(frame: pd.DataFrame) -> str:
    ordered = frame.sort_index(axis=0).sort_index(axis=1)
    lines = ["," + ",".join(str(column) for column in ordered.columns)]
    for stamp, row in ordered.iterrows():
        cells = ",".join(f"{float(value):.17g}" for value in row.to_numpy())
        lines.append(f"{pd.Timestamp(stamp).isoformat()},{cells}")
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def _levered_targets(book: LiveStrategyBook, venue_path: Path, bootstrap_path: Path) -> pd.DataFrame:
    """Account-policy baseline with fixed bootstrap moments and empty inventory."""
    rules = load_venue_rule_snapshot(venue_path)
    census = list(book.unit_weights.columns)
    ladders, _ = build_venue_ladders(census, rules)
    policy = account_growth_policy()
    bootstrap = pd.read_parquet(bootstrap_path)["unit_return"]
    rows = []
    for day in book.unit_weights.index:
        past = bootstrap[bootstrap.index <= day]
        moments = bayesian_unit_moments(
            len(past), float(past.sum()), float((past ** 2).sum()),
            prior_days=ACCOUNT_PRIOR_DAYS, min_moment_days=ACCOUNT_MIN_MOMENT_DAYS,
        )
        unit_values = book.unit_weights.loc[day, census].to_numpy(dtype="float64")
        exposure = choose_exposure(
            unit_values, _SEED_EQUITY_USDT, np.zeros(len(census)),
            book.adv.loc[day, census].to_numpy(dtype="float64"),
            book.daily_sigma.loc[day, census].to_numpy(dtype="float64"),
            ladders, policy, moments,
        )
        rows.append(unit_values * float(exposure))
    return pd.DataFrame(rows, index=book.unit_weights.index, columns=census, dtype="float64")


def _live_step_targets(layout: GoldenLayout, state: Path) -> pd.DataFrame:
    """Exercise production history extension, settlement handling and sizing wiring."""
    weights = state / "deployed_target_weights.parquet"
    for day in layout["book"].unit_weights.index:
        report = run_strategy_signal_step(
            day, now=day + pd.Timedelta(hours=23), data_root=layout["data"],
            weights_path=weights, unit_bootstrap_path=layout["bootstrap"],
            unit_forward_path=state / "forward.parquet", venue_rules_dir=state / "venue",
            fallback_venue_path=layout["venue"], ledger_path=state / "ledger.json",
            seed_equity_usdt=_SEED_EQUITY_USDT, non_crypto=frozenset(),
            listing_root=layout["listing"],
        )
        assert report.written
        assert report.unit_observations >= int(ACCOUNT_MIN_MOMENT_DAYS)
    bootstrap_count = len(pd.read_parquet(layout["bootstrap"]))
    assert report.unit_observations > bootstrap_count
    assert len(pd.read_parquet(state / "forward.parquet")) >= 20
    return load_weights_frame(weights)


@pytest.fixture(scope="module")
def golden_layout(tmp_path_factory: pytest.TempPathFactory) -> GoldenLayout:
    base = tmp_path_factory.mktemp("live_golden")
    data = base / "data"
    listing = base / "listing"
    venue = base / "venue.json"
    _write_panel(data)
    _write_funding(data)
    _write_listing(listing)
    _write_venue(venue)
    history = load_venue_listing_history(listing, through_day=_START + pd.Timedelta(days=_DAYS))

    def _blocked(index: pd.DatetimeIndex, order: tuple[str, ...]) -> pd.DataFrame:
        return delisting_blocked_decisions(
            history, index, order,
            holding_end_offset=pd.Timedelta(days=1, hours=_SNAPSHOT_HOUR),
            lead=pd.Timedelta(hours=48),
        )

    book = build_live_strategy_book(
        data, _CENSUS, panel_start=_START, panel_end=_START + pd.Timedelta(days=_DAYS),
        blocked_decisions=_blocked,
    )
    assert len(book.unit_weights) >= 20
    assert bool((book.unit_weights[_LATE] != 0).any())
    assert bool((book.unit_weights[_DLIST] != 0).any())
    assert bool((book.unit_weights.loc[_DELIVERY:, _DLIST] == 0).all())
    bootstrap = base / "boot.parquet"
    _write_bootstrap(bootstrap, book.unit_weights.index.min())
    return {"data": data, "listing": listing, "venue": venue, "book": book, "bootstrap": bootstrap}


def test_live_unit_book_digest_is_stable(golden_layout: GoldenLayout) -> None:
    assert _canonical_digest(golden_layout["book"].unit_weights) == _UNIT_BOOK_DIGEST


def test_live_levered_row_digest_is_stable(golden_layout: GoldenLayout) -> None:
    levered = _levered_targets(golden_layout["book"], golden_layout["venue"], golden_layout["bootstrap"])
    assert _canonical_digest(levered) == _LEVERED_ROW_DIGEST


def test_production_live_step_digest_is_stable(golden_layout: GoldenLayout, tmp_path: Path) -> None:
    targets = _live_step_targets(golden_layout, tmp_path / "state")
    assert _canonical_digest(targets) == _LIVE_STEP_DIGEST


def test_golden_fixture_is_causal(golden_layout: GoldenLayout, tmp_path: Path) -> None:
    book = golden_layout["book"]
    cutoff = book.unit_weights.index.max() + pd.Timedelta(hours=_SNAPSHOT_HOUR)
    perturbed = tmp_path / "data"
    shutil.copytree(golden_layout["data"], perturbed)
    for sym in _CENSUS:
        path = perturbed / "ohlcv" / "1h" / f"{sym}.parquet"
        frame = pd.read_parquet(path)
        stamps = pd.to_datetime(pd.to_numeric(frame["timestamp"], errors="coerce"), unit="ms", utc=True)
        mask = stamps > cutoff
        assert bool(mask.any())
        for column in ("close", "open", "high", "low", "quote_vol", "taker_buy_quote"):
            if column in frame.columns:
                frame.loc[mask, column] = frame.loc[mask, column] * 3.0
        frame.to_parquet(path)
    history = load_venue_listing_history(
        golden_layout["listing"], through_day=_START + pd.Timedelta(days=_DAYS),
    )

    def _blocked(index: pd.DatetimeIndex, order: tuple[str, ...]) -> pd.DataFrame:
        return delisting_blocked_decisions(
            history, index, order,
            holding_end_offset=pd.Timedelta(days=1, hours=_SNAPSHOT_HOUR),
            lead=pd.Timedelta(hours=48),
        )

    rebuilt = build_live_strategy_book(
        perturbed, _CENSUS, panel_start=_START, panel_end=_START + pd.Timedelta(days=_DAYS),
        blocked_decisions=_blocked,
    )
    assert _canonical_digest(rebuilt.unit_weights) == _UNIT_BOOK_DIGEST
    levered = _levered_targets(rebuilt, golden_layout["venue"], golden_layout["bootstrap"])
    assert _canonical_digest(levered) == _LEVERED_ROW_DIGEST
    perturbed_layout: GoldenLayout = {
        **golden_layout, "data": perturbed, "book": rebuilt,
    }
    targets = _live_step_targets(perturbed_layout, tmp_path / "state")
    assert _canonical_digest(targets) == _LIVE_STEP_DIGEST
