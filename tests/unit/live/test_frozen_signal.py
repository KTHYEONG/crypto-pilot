"""Invariant guards for the live frozen signal step."""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.lib
import pytest

from src.common.errors import DataIntegrityError
from src.live.errors import CausalityViolation
from src.live.frozen_book import build_live_frozen_book
from src.live.frozen_signal import FROZEN_SIGNAL_REPORT_NAME, run_frozen_signal_step
from src.mhs.params import ACCOUNT_EXPOSURE_STEP

_START = pd.Timestamp("2021-01-01", tz="UTC")
_SYMS = (*tuple(f"SYM{i:02d}USDT" for i in range(8)), "BTCUSDT")
_BOOT_END = pd.Timestamp("2021-05-20", tz="UTC")
_DAY = pd.Timestamp("2021-05-25", tz="UTC")


def _write_panel(root: Path, *, days: int = 150, seed: int = 7) -> None:
    n = days * 24
    grid = pd.date_range(_START, periods=n, freq="1h", tz="UTC")
    out = root / "ohlcv" / "1h"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ms = np.array([int(ts.value // 1_000_000) for ts in grid], dtype="int64")
    for j, sym in enumerate(_SYMS):
        if sym == "BTCUSDT":
            close = np.full(n, 60000.0)
            qv = np.full(n, 120_000.0)
            tbq = np.full(n, 60_000.0)
        else:
            rets = rng.normal(0, 0.005, size=n)
            close = 100.0 * (1.0 + 0.01 * j) * np.exp(np.cumsum(rets))
            qv = 100_000.0 + rng.uniform(0, 20_000.0, size=n)
            tbq = np.clip(qv * 0.5 * (1.0 + rng.normal(0, 0.02, size=n)), 0.0, None)
        pd.DataFrame(
            {"timestamp": ms, "close": close, "quote_vol": qv, "taker_buy_quote": tbq},
        ).to_parquet(out / f"{sym}.parquet")
    _write_observed_funding(root, _SYMS, grid)


def _write_observed_funding(
    root: Path, symbols: tuple[str, ...], grid: pd.DatetimeIndex, *, until: pd.Timestamp | None = None,
) -> None:
    """관측된 8h 펀딩(1e-9)을 기록한다: proxy는 보유 종목의 미관측 펀딩 창을 보류하므로 명시 공급한다."""
    funding_dir = root / "funding"
    funding_dir.mkdir(parents=True, exist_ok=True)
    fidx = pd.DatetimeIndex([grid[0] + pd.Timedelta(hours=h) for h in range(0, len(grid), 8)], tz="UTC")
    if until is not None:
        fidx = fidx[fidx <= until]
    for sym in symbols:
        pd.DataFrame(
            {"datetime": fidx, "funding_rate": np.full(len(fidx), 1e-9)},
        ).to_parquet(funding_dir / f"{sym}.parquet")


def _write_unit(path: Path, idx: pd.DatetimeIndex, vals: np.ndarray | float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    series = pd.Series(vals, index=idx, dtype="float64")
    series.to_frame("unit_return").to_parquet(path, index=True)


def _bootstrap_vals(seed: int = 11, mean: float = 0.001) -> np.ndarray:
    n = len(pd.date_range(_START, _BOOT_END, freq="1D", tz="UTC"))
    return np.random.default_rng(seed).normal(mean, 0.01, size=n)


def _write_venue(path: Path, symbols: list[str], *, mmr: float = 0.004, lev: int = 20) -> None:
    payload = {
        "captured_at": "2026-09-21T00:00:00+00:00",
        "symbols": {
            symbol: {
                "brackets": [
                    {
                        "notional_floor": 0.0, "notional_cap": 50000.0,
                        "maint_margin_ratio": mmr, "maint_amount": 0.0,
                        "initial_leverage": lev,
                    },
                    {
                        "notional_floor": 50000.0, "notional_cap": 100000000.0,
                        "maint_margin_ratio": mmr * 2.0, "maint_amount": 10.0,
                        "initial_leverage": lev,
                    },
                ],
                "step_size": None, "min_notional": None,
            }
            for symbol in symbols
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_ledger(path: Path, positions: dict[str, str], cash: str | None) -> None:
    payload: dict[str, object] = {"positions": positions}
    if cash is not None:
        payload["cash_usdt"] = cash
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture(scope="module")
def panel_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("frozen_panel") / "data"
    _write_panel(root)
    return root


@pytest.fixture
def layout(tmp_path: Path, panel_root: Path) -> dict[str, Path]:
    boot_idx = pd.date_range(_START, _BOOT_END, freq="1D", tz="UTC")
    boot = tmp_path / "boot.parquet"
    _write_unit(boot, boot_idx, _bootstrap_vals())
    venue_dir = tmp_path / "venue"
    _write_venue(venue_dir / "20260921.json", [_SYMS[0], _SYMS[1]])
    fallback = tmp_path / "fallback.json"
    _write_venue(fallback, [_SYMS[0], _SYMS[1]])
    return {
        "data": panel_root,
        "weights": tmp_path / "state" / "deployed_target_weights.parquet",
        "boot": boot,
        "forward": tmp_path / "fwd.parquet",
        "venue": venue_dir,
        "fallback": fallback,
        "ledger": tmp_path / "ledger.json",
    }


def _run(day: pd.Timestamp, paths: dict[str, Path], **overrides: object) -> object:
    kwargs: dict[str, object] = {
        "now": day + pd.Timedelta(hours=23),
        "data_root": paths["data"],
        "weights_path": paths["weights"],
        "unit_bootstrap_path": paths["boot"],
        "unit_forward_path": paths["forward"],
        "venue_rules_dir": paths["venue"],
        "fallback_venue_path": paths["fallback"],
        "ledger_path": paths["ledger"],
        "seed_equity_usdt": 2100.0,
        "non_crypto": frozenset(),
    }
    kwargs.update(overrides)
    return run_frozen_signal_step(day, **kwargs)  # type: ignore[arg-type]


def _report_path(paths: dict[str, Path]) -> Path:
    return paths["weights"].parent / FROZEN_SIGNAL_REPORT_NAME


def test_release_hour_guards_causality(layout: dict[str, Path]) -> None:
    with pytest.raises(CausalityViolation):
        _run(_DAY, layout, now=_DAY + pd.Timedelta(hours=22, minutes=59))
    assert not layout["weights"].exists()
    assert not layout["forward"].exists()
    assert not _report_path(layout).exists()


def test_writes_levered_row_equal_to_exposure_times_unit_book(layout: dict[str, Path]) -> None:
    report = _run(_DAY, layout)
    data_start = _START
    data_end = _START + pd.Timedelta(days=150)
    census = tuple(sorted(_SYMS))
    book = build_live_frozen_book(
        layout["data"], census, panel_start=data_start, panel_end=data_end,
    )
    stored = pd.read_parquet(layout["weights"]).loc[_DAY]
    pd.testing.assert_series_equal(
        stored, (book.unit_weights.loc[_DAY] * report.exposure).reindex(stored.index),
        check_names=False,
    )
    closes = pd.read_parquet(
        layout["weights"].parent / "deployed_decision_ohlcv_close.parquet",
    ).loc[_DAY]
    unit_row = book.unit_weights.loc[_DAY]
    assert set(closes.index) == {s for s in census if float(unit_row[s]) != 0.0}
    for symbol in closes.index:
        assert closes[symbol] == pytest.approx(float(book.snapshot_closes.loc[_DAY, symbol]))


def test_plaintext_artifacts_only(layout: dict[str, Path], tmp_path: Path) -> None:
    _run(_DAY, layout)
    assert list(tmp_path.rglob("*.enc")) == []


def test_seals_weight_artifacts_when_artifact_key_provided(
    layout: dict[str, Path], tmp_path: Path,
) -> None:
    from pydantic import SecretStr

    from src.live.crypto import derive_key, open_bytes
    from src.live.errors import ArtifactSealError

    key = SecretStr(base64.b64encode(b"0" * 32).decode("ascii"))
    first = _run(_DAY, layout, artifact_key=key)
    assert first.written is True
    weights_enc = layout["weights"].with_suffix(layout["weights"].suffix + ".enc")
    closes_enc = (layout["weights"].parent / "deployed_decision_ohlcv_close.parquet.enc")
    assert weights_enc.exists()
    assert closes_enc.exists()
    assert not layout["weights"].exists()
    # 평문 parquet 매직바이트가 아니라 봉투 형식이어야 한다.
    with pytest.raises(pyarrow.lib.ArrowInvalid):
        pd.read_parquet(weights_enc)
    open_bytes(weights_enc.read_bytes(), derive_key(key))  # 올바른 키로는 복호화된다.
    with pytest.raises(ArtifactSealError):
        open_bytes(weights_enc.read_bytes(), derive_key(SecretStr(base64.b64encode(b"1" * 32).decode("ascii"))))
    # 재실행 시 봉인된 기존 행을 올바로 읽어 append-only 불변식을 유지한다.
    second = _run(_DAY, layout, artifact_key=key)
    assert second.written is False


def test_seed_equity_on_first_cycle(layout: dict[str, Path]) -> None:
    report = _run(_DAY, layout)
    assert report.equity_usdt == 2100.0


def test_ledger_equity_marks_held_positions_at_snapshot_close(layout: dict[str, Path]) -> None:
    _write_ledger(layout["ledger"], {"BTCUSDT": "0.01"}, "1000")
    report = _run(_DAY, layout)
    assert report.equity_usdt == pytest.approx(1600.0)


def test_held_symbol_without_snapshot_close_fails_closed(
    layout: dict[str, Path], tmp_path: Path,
) -> None:
    patched = tmp_path / "patched"
    shutil.copytree(layout["data"], patched)
    bar = _DAY + pd.Timedelta(hours=22)
    victim = patched / "ohlcv" / "1h" / f"{_SYMS[0]}.parquet"
    frame = pd.read_parquet(victim)
    stamps = pd.to_datetime(pd.to_numeric(frame["timestamp"], errors="coerce"), unit="ms", utc=True)
    frame = frame[stamps != bar]
    frame.to_parquet(victim)
    _write_ledger(layout["ledger"], {_SYMS[0]: "0.01"}, "1000")
    with pytest.raises(DataIntegrityError):
        _run(_DAY, {**layout, "data": patched})


def test_posterior_uses_only_returns_up_to_decision_day(layout: dict[str, Path]) -> None:
    from src.mhs.account_policy import bayesian_unit_moments
    from src.mhs.params import ACCOUNT_MIN_MOMENT_DAYS, ACCOUNT_PRIOR_DAYS

    fwd_idx = pd.date_range(_BOOT_END + pd.Timedelta(days=1), _DAY + pd.Timedelta(days=1), freq="1D", tz="UTC")
    _write_unit(layout["forward"], fwd_idx, 0.5)
    report = _run(_DAY, layout)
    history = pd.read_parquet(layout["forward"])["unit_return"]
    assert history.index.max() == _DAY + pd.Timedelta(days=1)
    boot = pd.read_parquet(layout["boot"])["unit_return"]
    past = pd.concat([boot, history])
    past = past[~past.index.duplicated(keep="first")].sort_index()
    past = past[past.index <= _DAY]
    assert report.unit_observations == len(past) == int((history.index <= _DAY).sum()) + int(
        (boot.index <= _DAY).sum()
    )
    moments = bayesian_unit_moments(
        len(past), float(past.sum()), float((past**2).sum()),
        prior_days=ACCOUNT_PRIOR_DAYS, min_moment_days=ACCOUNT_MIN_MOMENT_DAYS,
    )
    assert moments is not None
    assert report.posterior_mean == pytest.approx(moments.mean)
    assert report.posterior_sigma == pytest.approx(moments.sigma)


def test_rerun_is_idempotent(layout: dict[str, Path]) -> None:
    first = _run(_DAY, layout)
    assert first.written is True
    snap = {
        name: path.read_bytes()
        for name, path in (
            ("w", layout["weights"]),
            ("f", layout["forward"]),
            ("r", _report_path(layout)),
        )
    }
    second = _run(_DAY, layout)
    assert second.written is False
    assert second.exposure == first.exposure
    for name, path in (("w", layout["weights"]), ("f", layout["forward"]), ("r", _report_path(layout))):
        assert path.read_bytes() == snap[name]


def test_symbol_absent_from_todays_census_is_written_as_zero_not_nan(layout: dict[str, Path]) -> None:
    from src.live.deployed_weights import append_weight_row

    # 전날 행에만 있던 심볼(예: 창 밖으로 사라진 상장폐지 종목)이 있어도 오늘 행은 NaN 없이 0이어야 한다.
    append_weight_row(
        layout["weights"], _DAY - pd.Timedelta(days=1),
        pd.Series({"GONEUSDT": 0.1, _SYMS[0]: -0.1}), artifact_key=None,
    )
    _run(_DAY, layout)
    today = pd.read_parquet(layout["weights"]).loc[_DAY]
    assert not today.isna().any()
    assert today["GONEUSDT"] == 0.0


def test_forward_history_is_append_only_across_days(layout: dict[str, Path]) -> None:
    _run(_DAY, layout)
    before = pd.read_parquet(layout["forward"])["unit_return"]
    _run(_DAY + pd.Timedelta(days=1), layout)
    after = pd.read_parquet(layout["forward"])["unit_return"]
    pd.testing.assert_series_equal(after.loc[before.index], before)


def test_cold_start_backfills_from_bootstrap_end(layout: dict[str, Path]) -> None:
    assert not layout["forward"].exists()
    _run(_DAY, layout)
    forward = pd.read_parquet(layout["forward"])["unit_return"]
    assert forward.index.min() == _BOOT_END + pd.Timedelta(days=1)
    assert (forward.index[1:] - forward.index[:-1] == pd.Timedelta(days=1)).all()


def test_venue_fallback_used_and_disclosed(layout: dict[str, Path], tmp_path: Path) -> None:
    empty = tmp_path / "venue_empty"
    empty.mkdir()
    report = _run(_DAY, {**layout, "venue": empty})
    assert report.venue_snapshot == layout["fallback"].name


def test_no_venue_snapshot_fails_closed(layout: dict[str, Path], tmp_path: Path) -> None:
    empty = tmp_path / "venue_empty"
    empty.mkdir()
    with pytest.raises(DataIntegrityError):
        _run(_DAY, {**layout, "venue": empty, "fallback": tmp_path / "missing.json"})


def test_exposure_respects_margin_cap(layout: dict[str, Path], tmp_path: Path) -> None:
    from src.mhs.account_policy import build_venue_ladders, margin_exposure_cap, account_growth_policy
    from src.market_data.binance.venue_rules import load_venue_rule_snapshot

    boot_idx = pd.date_range(_START, _BOOT_END, freq="1D", tz="UTC")
    _write_unit(layout["boot"], boot_idx, _bootstrap_vals(seed=5, mean=-0.01))
    strict = tmp_path / "strict.json"
    _write_venue(strict, [_SYMS[0]], mmr=3.0, lev=1)
    empty = tmp_path / "venue_empty"
    empty.mkdir()
    report = _run(_DAY, {**layout, "venue": empty, "fallback": strict})
    stored = pd.read_parquet(layout["weights"]).loc[_DAY]
    weights = (stored / report.exposure).to_numpy(dtype="float64")
    rules = load_venue_rule_snapshot(strict)
    ladders, _ = build_venue_ladders(sorted(_SYMS), rules)
    cap = margin_exposure_cap(weights, report.equity_usdt, ladders, account_growth_policy())
    assert cap == pytest.approx(ACCOUNT_EXPOSURE_STEP)
    assert report.exposure == pytest.approx(ACCOUNT_EXPOSURE_STEP)


def test_no_book_row_fails_closed(
    layout: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.live.frozen_signal as signal_mod
    from src.live.frozen_book import LiveFrozenBook

    idx = pd.DatetimeIndex(["2021-01-02"], tz="UTC")
    cols = ["AAAUSDT"]
    fake = LiveFrozenBook(
        unit_weights=pd.DataFrame([[0.1]], index=idx, columns=cols, dtype="float64"),
        snapshot_closes=pd.DataFrame([[1.0]], index=idx, columns=cols, dtype="float64"),
        adv=pd.DataFrame([[1.0]], index=idx, columns=cols, dtype="float64"),
        daily_sigma=pd.DataFrame([[0.01]], index=idx, columns=cols, dtype="float64"),
        valid_from=idx[0], panel_last_bar=pd.Timestamp("2021-01-10", tz="UTC"),
    )
    monkeypatch.setattr(signal_mod, "build_live_frozen_book", lambda *args, **kwargs: fake)
    bare = tmp_path / "bare"
    (bare / "ohlcv" / "1h").mkdir(parents=True, exist_ok=True)
    with pytest.raises(DataIntegrityError):
        _run(_DAY, {**layout, "data": bare})


def test_bootstrap_missing_fails_closed(layout: dict[str, Path], tmp_path: Path) -> None:
    missing = tmp_path / "no_boot.parquet"
    with pytest.raises(DataIntegrityError):
        _run(_DAY, {**layout, "boot": missing})
    assert not layout["weights"].exists()
    assert not layout["forward"].exists()
    assert not _report_path(layout).exists()


def test_malformed_bootstrap_fails_closed(layout: dict[str, Path], tmp_path: Path) -> None:
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"not a parquet file")
    with pytest.raises(DataIntegrityError):
        _run(_DAY, {**layout, "boot": bad})


def test_sealed_bootstrap_readable_with_key_and_blocked_without(
    layout: dict[str, Path], tmp_path: Path,
) -> None:
    from pydantic import SecretStr

    from src.live.crypto import derive_key, seal_bytes
    from src.live.errors import ArtifactSealError

    key = SecretStr(base64.b64encode(b"2" * 32).decode("ascii"))
    sealed = tmp_path / "boot.parquet.enc"
    sealed.write_bytes(seal_bytes(layout["boot"].read_bytes(), derive_key(key)))
    # 공개 리포에 커밋되는 건 이 바이트뿐이다 -- 키 없이는 무의미한 암호문이어야 한다.
    with pytest.raises(ArtifactSealError):
        _run(_DAY, {**layout, "boot": sealed})
    report = _run(_DAY, {**layout, "boot": sealed}, artifact_key=key)
    assert report.written is True


def test_empty_forward_file_uses_bootstrap_anchor(layout: dict[str, Path]) -> None:
    pd.DataFrame(
        {"unit_return": pd.Series([], dtype="float64")},
        index=pd.DatetimeIndex([], tz="UTC"),
    ).to_parquet(layout["forward"], index=True)
    _run(_DAY, layout)
    forward = pd.read_parquet(layout["forward"])["unit_return"]
    assert forward.index.min() == _BOOT_END + pd.Timedelta(days=1)


def test_nonfinite_sizing_fails_closed(layout: dict[str, Path]) -> None:
    _write_ledger(layout["ledger"], {_SYMS[0]: "1e308"}, "1e308")
    with pytest.raises(DataIntegrityError):
        _run(_DAY, layout)


def test_venue_prefers_gzip_snapshot_over_older_json(layout: dict[str, Path], tmp_path: Path) -> None:
    import gzip as _gzip

    venue = tmp_path / "venue_mixed"
    venue.mkdir()
    _write_venue(venue / "20260921.json", [_SYMS[0], _SYMS[1]])
    staged = tmp_path / "staged.json"
    _write_venue(staged, [_SYMS[0], _SYMS[1]])
    (venue / "20260922.json.gz").write_bytes(
        _gzip.compress(staged.read_bytes(), compresslevel=9, mtime=0)
    )
    report = _run(_DAY, {**layout, "venue": venue})
    assert report.venue_snapshot == "20260922.json.gz"


def test_live_account_equity_drives_exposure(layout: dict[str, Path], tmp_path: Path) -> None:
    seed_report = _run(_DAY, layout)
    assert seed_report.equity_usdt == 2100.0
    layout2 = {**layout}
    alt = tmp_path / "live_alt"
    alt.mkdir()
    layout2["weights"] = alt / "deployed_target_weights.parquet"
    layout2["forward"] = alt / "fwd.parquet"
    live_report = _run(_DAY, layout2, account_equity_usdt=50_000.0)
    assert live_report.equity_usdt == 50_000.0


def test_invalid_account_equity_fails_closed(layout: dict[str, Path]) -> None:
    import pytest
    from src.common.errors import DataIntegrityError
    with pytest.raises(DataIntegrityError):
        _run(_DAY, layout, account_equity_usdt=0.0)
    with pytest.raises(DataIntegrityError):
        _run(_DAY, layout, account_equity_usdt=float("nan"))
    assert not layout["weights"].exists()
    assert not layout["forward"].exists()


def test_paper_equity_precedence_from_ledger(layout: dict[str, Path]) -> None:
    _write_ledger(layout["ledger"], {"BTCUSDT": "0.01"}, "1000")
    report = _run(_DAY, layout, account_equity_usdt=None)
    assert report.equity_usdt == pytest.approx(1600.0)


_D_SYMS = tuple(f"DR{i:02d}USDT" for i in range(8))
_DLIST = "DLISTUSDT"
_D_SETTLE = 0.214
_D_DELIVERY = pd.Timestamp("2021-05-24 00:00", tz="UTC")
_D_FIRST_SEEN = pd.Timestamp("2021-05-20", tz="UTC")
_D_ALL = (*_D_SYMS, _DLIST)
_D_FAR = pd.Timestamp("2049-01-01", tz="UTC")


def _write_delist_panel(root: Path, *, days: int = 150, seed: int = 7) -> None:
    n = days * 24
    grid = pd.date_range(_START, periods=n, freq="1h", tz="UTC")
    out = root / "ohlcv" / "1h"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ms = np.array([int(ts.value // 1_000_000) for ts in grid], dtype="int64")
    for j, sym in enumerate(_D_SYMS):
        rets = rng.normal(0, 0.005, size=n)
        close = 100.0 * (1.0 + 0.01 * j) * np.exp(np.cumsum(rets))
        qv = 100_000.0 + rng.uniform(0, 20_000.0, size=n)
        tbq = np.clip(qv * 0.5 * (1.0 + rng.normal(0, 0.02, size=n)), 0.0, None)
        pd.DataFrame(
            {
                "timestamp": ms, "open": close, "high": close, "low": close,
                "close": close, "volume": np.full(n, 1000.0),
                "quote_vol": qv, "taker_buy_quote": tbq,
            },
        ).to_parquet(out / f"{sym}.parquet")
    delivery_pos = int((pd.Timestamp(_D_DELIVERY) - grid[0]) / pd.Timedelta(hours=1))
    walk = 1.0 * np.exp(np.cumsum(rng.normal(0, 0.005, size=delivery_pos)))
    scale = 2.0 / walk[-1]
    pre = walk * scale
    flat_n = n - delivery_pos
    close = np.concatenate([pre, np.full(flat_n, _D_SETTLE)])
    volume = np.concatenate([np.full(delivery_pos, 1000.0), np.zeros(flat_n)])
    qv = np.full(n, 110_000.0)
    tbq = np.full(n, 55_000.0)
    pd.DataFrame(
        {
            "timestamp": ms, "open": close, "high": close, "low": close,
            "close": close, "volume": volume,
            "quote_vol": qv, "taker_buy_quote": tbq,
        },
    ).to_parquet(out / f"{_DLIST}.parquet")
    _write_observed_funding(root, _D_SYMS, grid)
    _write_observed_funding(root, (_DLIST,), grid, until=_D_DELIVERY)


def _write_delist_listing(
    root: Path, slots: list[pd.Timestamp], *, status_after_delivery: str = "SETTLING",
) -> None:
    from src.live.venue_listing import (
        VenueListingEntry,
        VenueListingSnapshot,
        write_venue_listing_snapshot,
    )

    for slot in slots:
        status = status_after_delivery if slot >= _D_DELIVERY.normalize() else "TRADING"
        entries = {
            sym: VenueListingEntry(
                symbol=sym, status="TRADING", contract_type="PERPETUAL",
                underlying_type="COIN", quote_asset="USDT",
                delivery_time=_D_FAR, announced_delisting=False, delisting_first_seen_at=None,
            )
            for sym in _D_SYMS
        }
        entries[_DLIST] = VenueListingEntry(
            symbol=_DLIST, status=status, contract_type="PERPETUAL",
            underlying_type="COIN", quote_asset="USDT",
            delivery_time=_D_DELIVERY, announced_delisting=True,
            delisting_first_seen_at=_D_FIRST_SEEN + pd.Timedelta(hours=1),
        )
        snapshot = VenueListingSnapshot(
            captured_at=slot + pd.Timedelta(hours=1), entries=entries,
        )
        write_venue_listing_snapshot(snapshot, root, slot_day=slot)


@pytest.fixture(scope="module")
def delist_panel_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("delist_panel") / "data"
    _write_delist_panel(root)
    return root


def _delist_layout(
    tmp_path: Path, panel: Path, *, day: pd.Timestamp = _DAY,
) -> dict[str, Path]:
    boot_idx = pd.date_range(_START, _BOOT_END, freq="1D", tz="UTC")
    boot = tmp_path / "boot.parquet"
    _write_unit(boot, boot_idx, _bootstrap_vals())
    venue_dir = tmp_path / "venue"
    _write_venue(venue_dir / "20260921.json", [_D_SYMS[0], _D_SYMS[1]])
    fallback = tmp_path / "fallback.json"
    _write_venue(fallback, [_D_SYMS[0], _D_SYMS[1]])
    return {
        "data": panel,
        "weights": tmp_path / "state" / "deployed_target_weights.parquet",
        "boot": boot,
        "forward": tmp_path / "fwd.parquet",
        "venue": venue_dir,
        "fallback": fallback,
        "ledger": tmp_path / "ledger.json",
        "listing": tmp_path / "listing",
    }


def _run_listing(day: pd.Timestamp, paths: dict[str, Path], **overrides: object) -> object:
    kwargs: dict[str, object] = {
        "now": day + pd.Timedelta(hours=23),
        "data_root": paths["data"],
        "weights_path": paths["weights"],
        "unit_bootstrap_path": paths["boot"],
        "unit_forward_path": paths["forward"],
        "venue_rules_dir": paths["venue"],
        "fallback_venue_path": paths["fallback"],
        "ledger_path": paths["ledger"],
        "seed_equity_usdt": 2100.0,
        "non_crypto": frozenset(),
        "listing_root": paths["listing"],
    }
    kwargs.update(overrides)
    return run_frozen_signal_step(day, **kwargs)  # type: ignore[arg-type]


def test_delisted_roster_member_runs_across_delivery_without_halt(
    tmp_path: Path, delist_panel_root: Path,
) -> None:
    paths = _delist_layout(tmp_path, delist_panel_root)
    days = [pd.Timestamp("2021-05-23", tz="UTC"), pd.Timestamp("2021-05-24", tz="UTC"), _DAY]
    _write_delist_listing(
        paths["listing"],
        [pd.Timestamp("2021-05-20", tz="UTC") + pd.Timedelta(days=i) for i in range(6)],
    )
    previous_max: pd.Timestamp | None = None
    for day in days:
        report = _run_listing(day, paths)
        assert report.written is True
        row = pd.read_parquet(paths["weights"]).loc[day]
        assert float(row[_DLIST]) == 0.0
        forward = pd.read_parquet(paths["forward"])["unit_return"]
        assert forward.index.max() == day + pd.Timedelta(days=1)
        if previous_max is not None:
            assert forward.index.max() == previous_max + pd.Timedelta(days=1)
        previous_max = forward.index.max()


def test_held_dust_in_settled_symbol_valued_at_settlement(
    tmp_path: Path, delist_panel_root: Path,
) -> None:
    paths = _delist_layout(tmp_path, delist_panel_root)
    _write_delist_listing(
        paths["listing"],
        [pd.Timestamp("2021-05-20", tz="UTC") + pd.Timedelta(days=i) for i in range(6)],
    )
    _write_ledger(paths["ledger"], {_DLIST: "0.001"}, "1000")
    report = _run_listing(_DAY, paths)
    assert report.written is True
    assert report.equity_usdt == pytest.approx(1000.0 + 0.001 * _D_SETTLE)


def test_settled_holding_without_evidence_fails_closed(
    tmp_path: Path, delist_panel_root: Path,
) -> None:
    import shutil as _shutil

    patched = tmp_path / "patched"
    _shutil.copytree(delist_panel_root, patched)
    victim = patched / "ohlcv" / "1h" / f"{_DLIST}.parquet"
    frame = pd.read_parquet(victim)
    delivery_ms = int(_D_DELIVERY.value // 1_000_000)
    frame = frame[pd.to_numeric(frame["timestamp"]) <= delivery_ms]
    frame.to_parquet(victim)
    paths = _delist_layout(tmp_path, patched)
    _write_delist_listing(
        paths["listing"],
        [pd.Timestamp("2021-05-20", tz="UTC") + pd.Timedelta(days=i) for i in range(6)],
    )
    _write_ledger(paths["ledger"], {_DLIST: "0.001"}, "1000")
    with pytest.raises(DataIntegrityError, match="without settlement evidence"):
        _run_listing(_DAY, paths)


def test_forward_persisted_before_later_venue_halt(
    tmp_path: Path, delist_panel_root: Path,
) -> None:
    paths = _delist_layout(tmp_path, delist_panel_root)
    _write_delist_listing(
        paths["listing"],
        [pd.Timestamp("2021-05-20", tz="UTC") + pd.Timedelta(days=i) for i in range(6)],
    )
    empty = tmp_path / "venue_empty"
    empty.mkdir()
    with pytest.raises(DataIntegrityError):
        _run_listing(_DAY, {**paths, "venue": empty, "fallback": tmp_path / "missing.json"})
    forward = pd.read_parquet(paths["forward"])["unit_return"]
    assert forward.index.max() == _DAY + pd.Timedelta(days=1)
    assert forward.index.min() == _BOOT_END + pd.Timedelta(days=1)
    assert (forward.index[1:] - forward.index[:-1] == pd.Timedelta(days=1)).all()


def test_stale_listing_snapshot_halts(tmp_path: Path, delist_panel_root: Path) -> None:
    from src.live.venue_listing import (
        VenueListingEntry,
        VenueListingSnapshot,
        write_venue_listing_snapshot,
    )

    paths = _delist_layout(tmp_path, delist_panel_root)
    now = _DAY + pd.Timedelta(hours=23)
    captured = now - pd.Timedelta(hours=40)
    entries = {
        sym: VenueListingEntry(
            symbol=sym, status="TRADING", contract_type="PERPETUAL",
            underlying_type="COIN", quote_asset="USDT",
            delivery_time=_D_FAR, announced_delisting=False, delisting_first_seen_at=None,
        )
        for sym in _D_ALL
    }
    write_venue_listing_snapshot(
        VenueListingSnapshot(captured_at=captured, entries=entries),
        paths["listing"], slot_day=captured.normalize(),
    )
    with pytest.raises(DataIntegrityError, match="stale"):
        _run_listing(_DAY, paths, listing_max_age=pd.Timedelta(hours=30))


def test_held_symbol_absent_from_listing_skips_evidence(tmp_path: Path, delist_panel_root: Path) -> None:
    from src.live.venue_listing import (
        VenueListingEntry,
        VenueListingSnapshot,
        write_venue_listing_snapshot,
    )

    paths = _delist_layout(tmp_path, delist_panel_root)
    slot = pd.Timestamp("2021-05-20", tz="UTC")
    entries = {
        _DLIST: VenueListingEntry(
            symbol=_DLIST, status="SETTLING", contract_type="PERPETUAL",
            underlying_type="COIN", quote_asset="USDT",
            delivery_time=_D_DELIVERY, announced_delisting=True,
            delisting_first_seen_at=_D_FIRST_SEEN + pd.Timedelta(hours=1),
        )
    }
    write_venue_listing_snapshot(
        VenueListingSnapshot(captured_at=slot + pd.Timedelta(hours=1), entries=entries),
        paths["listing"],
        slot_day=slot,
    )
    _write_ledger(paths["ledger"], {_D_SYMS[0]: "0.01"}, "1000")
    report = _run_listing(_DAY, paths)
    assert report.written is True
    book = build_live_frozen_book(
        paths["data"], tuple(sorted(_D_ALL)),
        panel_start=_START, panel_end=_START + pd.Timedelta(days=150),
    )
    assert report.equity_usdt == pytest.approx(1000.0 + 0.01 * float(book.snapshot_closes.loc[_DAY, _D_SYMS[0]]))


def test_settled_dust_without_snapshot_bars_valued_at_settlement(
    tmp_path: Path, delist_panel_root: Path,
) -> None:
    import shutil as _shutil

    patched = tmp_path / "patched"
    _shutil.copytree(delist_panel_root, patched)
    stamps = [int((_D_DELIVERY + pd.Timedelta(hours=h)).value // 1_000_000) for h in range(5)]
    pd.DataFrame(
        {
            "timestamp": stamps,
            "open": [_D_SETTLE] * 5,
            "high": [_D_SETTLE] * 5,
            "low": [_D_SETTLE] * 5,
            "close": [_D_SETTLE] * 5,
            "volume": [0.0] * 5,
            "quote_vol": [110_000.0] * 5,
            "taker_buy_quote": [55_000.0] * 5,
        }
    ).to_parquet(patched / "ohlcv" / "1h" / "DUSTUSDT.parquet", index=False)
    paths = _delist_layout(tmp_path, patched)
    _write_delist_listing(
        paths["listing"],
        [pd.Timestamp("2021-05-20", tz="UTC") + pd.Timedelta(days=i) for i in range(6)],
    )
    from src.live.venue_listing import (
        VenueListingEntry,
        VenueListingSnapshot,
        load_venue_listing_history,
        write_venue_listing_snapshot,
    )

    history = load_venue_listing_history(paths["listing"], through_day=_DAY)
    last = history[-1]
    entries = dict(last.entries)
    entries["DUSTUSDT"] = VenueListingEntry(
        symbol="DUSTUSDT", status="SETTLING", contract_type="PERPETUAL",
        underlying_type="COIN", quote_asset="USDT",
        delivery_time=_D_DELIVERY, announced_delisting=True,
        delisting_first_seen_at=_D_FIRST_SEEN + pd.Timedelta(hours=1),
    )
    write_venue_listing_snapshot(
        VenueListingSnapshot(captured_at=last.captured_at, entries=entries),
        paths["listing"],
        slot_day=pd.Timestamp("2021-05-25", tz="UTC"),
    )
    _write_ledger(paths["ledger"], {"DUSTUSDT": "2"}, "1000")
    report = _run_listing(_DAY, paths)
    assert report.written is True
    assert report.equity_usdt == pytest.approx(1000.0 + 2 * _D_SETTLE)


def test_zero_weight_missing_close_fails_closed_without_listing(layout: dict[str, Path]) -> None:
    _write_ledger(layout["ledger"], {"GONEUSDT": "1"}, "1000")
    with pytest.raises(DataIntegrityError, match="snapshot close"):
        _run(_DAY, layout)


_G_SYMS = tuple(f"GX{i:02d}USDT" for i in range(30))


def _write_custom_panel(root: Path, symbols: tuple[str, ...], *, days: int = 150, seed: int = 7) -> None:
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
    _write_observed_funding(root, symbols, grid)


def _custom_layout(tmp_path: Path, symbols: tuple[str, ...]) -> dict[str, Path]:
    data = tmp_path / "data"
    _write_custom_panel(data, symbols)
    boot_idx = pd.date_range(_START, _BOOT_END, freq="1D", tz="UTC")
    boot = tmp_path / "boot.parquet"
    _write_unit(boot, boot_idx, _bootstrap_vals())
    venue_dir = tmp_path / "venue"
    _write_venue(venue_dir / "20260520.json", [symbols[0], symbols[1]])
    fallback = tmp_path / "fallback.json"
    _write_venue(fallback, [symbols[0], symbols[1]])
    return {
        "data": data,
        "weights": tmp_path / "state" / "deployed_target_weights.parquet",
        "boot": boot,
        "forward": tmp_path / "fwd.parquet",
        "venue": venue_dir,
        "fallback": fallback,
        "ledger": tmp_path / "ledger.json",
    }


def _truncate_after(root: Path, symbol: str, cutoff: pd.Timestamp) -> None:
    victim = root / "ohlcv" / "1h" / f"{symbol}.parquet"
    frame = pd.read_parquet(victim)
    cutoff_ms = int(pd.Timestamp(cutoff).tz_convert("UTC").value // 1_000_000)
    frame = frame[pd.to_numeric(frame["timestamp"], errors="coerce") <= cutoff_ms]
    assert len(frame) > 0
    frame.to_parquet(victim)


def _drop_bar(root: Path, symbol: str, bar: pd.Timestamp) -> None:
    victim = root / "ohlcv" / "1h" / f"{symbol}.parquet"
    frame = pd.read_parquet(victim)
    stamps = pd.to_datetime(pd.to_numeric(frame["timestamp"], errors="coerce"), unit="ms", utc=True)
    assert (stamps == pd.Timestamp(bar).tz_convert("UTC")).sum() == 1
    frame = frame[stamps != pd.Timestamp(bar).tz_convert("UTC")]
    frame.to_parquet(victim)


def _write_venue_at(path: Path, symbols: list[str], captured_at: pd.Timestamp) -> None:
    payload = {
        "captured_at": pd.Timestamp(captured_at).tz_convert("UTC").isoformat(),
        "symbols": {
            symbol: {
                "brackets": [
                    {
                        "notional_floor": 0.0, "notional_cap": 50000.0,
                        "maint_margin_ratio": 0.004, "maint_amount": 0.0,
                        "initial_leverage": 20,
                    },
                    {
                        "notional_floor": 50000.0, "notional_cap": 100000000.0,
                        "maint_margin_ratio": 0.008, "maint_amount": 10.0,
                        "initial_leverage": 20,
                    },
                ],
                "step_size": None, "min_notional": None,
            }
            for symbol in symbols
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_required_symbol_without_decision_bar_halts_precisely(
    layout: dict[str, Path], tmp_path: Path,
) -> None:
    patched = tmp_path / "patched"
    shutil.copytree(layout["data"], patched)
    _truncate_after(patched, _SYMS[0], _DAY + pd.Timedelta(hours=21))
    _write_ledger(layout["ledger"], {_SYMS[0]: "0.01"}, "1000")
    with pytest.raises(DataIntegrityError, match="decision_bar_missing required=SYM00USDT"):
        _run(_DAY, {**layout, "data": patched}, decision_bar_max_missing_fraction=0.05)


def test_systemic_incompleteness_halts(tmp_path: Path) -> None:
    paths = _custom_layout(tmp_path, _G_SYMS)
    _truncate_after(paths["data"], _G_SYMS[0], _DAY + pd.Timedelta(hours=21))
    _truncate_after(paths["data"], _G_SYMS[1], _DAY + pd.Timedelta(hours=21))
    with pytest.raises(DataIntegrityError, match=r"decision_bar_missing census=2/30"):
        _run(_DAY, paths, decision_bar_max_missing_fraction=0.05)


def test_sparse_incompleteness_proceeds_and_is_reported(tmp_path: Path) -> None:
    paths = _custom_layout(tmp_path, _G_SYMS)
    # 절단(당일 21:00 꼬리)은 전날 결정행에 영향을 주지 않으므로(인과성), 온전한 북에서
    # 전날 가중치가 0인 비보유 심볼을 골라 절단한다 -- 이후 결정일에서는 NaN 특성으로
    # 리서치 빌더와 동일하게 자동 탈락한다.
    intact = build_live_frozen_book(
        paths["data"], tuple(sorted(_G_SYMS)),
        panel_start=_START, panel_end=_START + pd.Timedelta(days=150),
    )
    prev_day = _DAY - pd.Timedelta(days=1)
    assert prev_day in intact.unit_weights.index
    candidates = [
        symbol for symbol in sorted(_G_SYMS)
        if float(intact.unit_weights.loc[prev_day, symbol]) == 0.0
    ]
    assert candidates
    _truncate_after(paths["data"], candidates[0], _DAY + pd.Timedelta(hours=21))
    report = _run(_DAY, paths, decision_bar_max_missing_fraction=0.05)
    assert report.decision_bar_missing == 1
    assert report.venue_gap_excluded == ()
    assert report.written is True


def test_non_held_venue_gap_symbol_excluded_with_reason(tmp_path: Path) -> None:
    from src.live.frozen_book import build_live_frozen_book, snapshot_gap_blocked_decisions

    paths = _custom_layout(tmp_path, _G_SYMS)
    intact = build_live_frozen_book(
        paths["data"], tuple(sorted(_G_SYMS)),
        panel_start=_START, panel_end=_START + pd.Timedelta(days=150),
    )
    prev_day = _DAY - pd.Timedelta(days=1)
    assert prev_day in intact.unit_weights.index
    # 전날 결정 가중치가 0인 비보유 심볼의 당일 22:00 봉만 제거한다(21:00과 23:00은 존재).
    candidates = [
        symbol for symbol in sorted(_G_SYMS)
        if float(intact.unit_weights.loc[prev_day, symbol]) == 0.0
    ]
    assert candidates
    gap_symbol = candidates[0]
    patched = tmp_path / "patched"
    shutil.copytree(paths["data"], patched)
    _drop_bar(patched, gap_symbol, _DAY + pd.Timedelta(hours=22))
    from src.live.venue_listing import (
        VenueListingEntry as _VEntry,
        VenueListingSnapshot as _VSnapshot,
        write_venue_listing_snapshot as _write_snapshot,
    )

    listing_dir = tmp_path / "listing_gap"
    for slot in (_DAY - pd.Timedelta(days=2), _DAY - pd.Timedelta(days=1), _DAY):
        _write_snapshot(
            _VSnapshot(
                captured_at=slot + pd.Timedelta(hours=1),
                entries={
                    sym: _VEntry(
                        symbol=sym, status="TRADING", contract_type="PERPETUAL",
                        underlying_type="COIN", quote_asset="USDT",
                        delivery_time=None, announced_delisting=False,
                        delisting_first_seen_at=None,
                    )
                    for sym in sorted(_G_SYMS)
                },
            ),
            listing_dir, slot_day=slot,
        )
    report = _run(
        _DAY, {**paths, "data": patched}, decision_bar_max_missing_fraction=0.05,
        listing_root=listing_dir,
    )
    assert report.venue_gap_excluded == (gap_symbol,)
    assert report.decision_bar_missing == 0
    stored = pd.read_parquet(paths["weights"]).loc[_DAY]
    assert float(stored[gap_symbol]) == 0.0
    book = build_live_frozen_book(
        patched, tuple(sorted(_G_SYMS)),
        panel_start=_START, panel_end=_START + pd.Timedelta(days=150),
    )
    # 북은 결손 봉을 어떤 가격으로도 대입하지 않는다 -- 스냅샷 종가는 NaN 그대로다.
    assert not np.isfinite(float(book.snapshot_closes.loc[_DAY, gap_symbol]))
    blocked = snapshot_gap_blocked_decisions(
        patched, book.unit_weights.index, tuple(book.unit_weights.columns), snapshot_hour=22,
    )
    assert bool(blocked.loc[_DAY, gap_symbol]) is True


def test_held_venue_gap_symbol_fails_closed(layout: dict[str, Path], tmp_path: Path) -> None:
    patched = tmp_path / "patched"
    shutil.copytree(layout["data"], patched)
    gap_symbol = _SYMS[1]
    _drop_bar(patched, gap_symbol, _DAY + pd.Timedelta(hours=22))
    _write_ledger(layout["ledger"], {gap_symbol: "0.01"}, "1000")
    with pytest.raises(
        DataIntegrityError,
        match=f"decision_bar_missing required={gap_symbol} kind=venue_gap",
    ):
        _run(_DAY, {**layout, "data": patched}, decision_bar_max_missing_fraction=0.05)


def test_stale_venue_ladder_halts(layout: dict[str, Path], tmp_path: Path) -> None:
    stale_dir = tmp_path / "venue_stale"
    _write_venue_at(stale_dir / "20260516.json", [_SYMS[0], _SYMS[1]], _DAY - pd.Timedelta(days=9))
    with pytest.raises(DataIntegrityError, match="venue snapshot age"):
        _run(
            _DAY, {**layout, "venue": stale_dir},
            venue_max_age=pd.Timedelta(days=7),
        )


def test_stale_fallback_snapshot_halts(layout: dict[str, Path], tmp_path: Path) -> None:
    empty = tmp_path / "venue_empty"
    empty.mkdir()
    old_fallback = tmp_path / "old_fallback.json"
    _write_venue_at(old_fallback, [_SYMS[0], _SYMS[1]], _DAY - pd.Timedelta(days=30))
    with pytest.raises(DataIntegrityError, match="venue snapshot age"):
        _run(
            _DAY, {**layout, "venue": empty, "fallback": old_fallback},
            venue_max_age=pd.Timedelta(days=7),
        )


def test_deployed_nonzero_venue_gap_symbol_fails_closed(tmp_path: Path) -> None:
    from src.live.deployed_weights import append_weight_row
    from src.live.frozen_book import build_live_frozen_book

    paths = _custom_layout(tmp_path, _G_SYMS)
    intact = build_live_frozen_book(
        paths["data"], tuple(sorted(_G_SYMS)),
        panel_start=_START, panel_end=_START + pd.Timedelta(days=150),
    )
    prev_day = _DAY - pd.Timedelta(days=1)
    candidates = [
        symbol for symbol in sorted(_G_SYMS)
        if float(intact.unit_weights.loc[prev_day, symbol]) == 0.0
    ]
    assert candidates
    gap_symbol = candidates[0]
    append_weight_row(
        paths["weights"], prev_day,
        pd.Series({gap_symbol: 0.05}), artifact_key=None,
    )
    from src.live.venue_listing import (
        VenueListingEntry,
        VenueListingSnapshot,
        write_venue_listing_snapshot,
    )

    listing_dir = tmp_path / "listing"
    for slot in (prev_day - pd.Timedelta(days=1), prev_day, _DAY):
        entries = {
            sym: VenueListingEntry(
                symbol=sym, status="TRADING", contract_type="PERPETUAL",
                underlying_type="COIN", quote_asset="USDT",
                delivery_time=None, announced_delisting=False,
                delisting_first_seen_at=None,
            )
            for sym in sorted(_G_SYMS)
        }
        write_venue_listing_snapshot(
            VenueListingSnapshot(captured_at=slot + pd.Timedelta(hours=1), entries=entries),
            listing_dir, slot_day=slot,
        )
    patched = tmp_path / "patched"
    shutil.copytree(paths["data"], patched)
    _drop_bar(patched, gap_symbol, _DAY + pd.Timedelta(hours=22))
    with pytest.raises(
        DataIntegrityError,
        match=f"decision_bar_missing required={gap_symbol} kind=venue_gap",
    ):
        _run(
            _DAY, {**paths, "data": patched}, decision_bar_max_missing_fraction=0.05,
            listing_root=listing_dir,
        )


def test_symbol_absent_from_exchange_info_is_not_counted_as_refresh_incomplete(tmp_path: Path) -> None:
    """exchangeInfo에서 사라진 심볼은 TRADING census가 아니므로 결정봉 결손으로 집계하지 않는다."""
    from src.live.venue_listing import (
        VenueListingEntry as _VEntry,
        VenueListingSnapshot as _VSnapshot,
        write_venue_listing_snapshot as _write_snapshot,
    )

    paths = _custom_layout(tmp_path, _G_SYMS)
    intact = build_live_frozen_book(
        paths["data"], tuple(sorted(_G_SYMS)),
        panel_start=_START, panel_end=_START + pd.Timedelta(days=150),
    )
    prev_day = _DAY - pd.Timedelta(days=1)
    purged = next(
        symbol for symbol in sorted(_G_SYMS)
        if float(intact.unit_weights.loc[prev_day, symbol]) == 0.0
    )
    _truncate_after(paths["data"], purged, _DAY + pd.Timedelta(hours=21))
    listing_dir = tmp_path / "listing_purged"
    for slot in (_DAY - pd.Timedelta(days=2), _DAY - pd.Timedelta(days=1), _DAY):
        _write_snapshot(
            _VSnapshot(
                captured_at=slot + pd.Timedelta(hours=1),
                entries={
                    sym: _VEntry(
                        symbol=sym, status="TRADING", contract_type="PERPETUAL",
                        underlying_type="COIN", quote_asset="USDT",
                        delivery_time=None, announced_delisting=False,
                        delisting_first_seen_at=None,
                    )
                    for sym in sorted(_G_SYMS) if sym != purged
                },
            ),
            listing_dir, slot_day=slot,
        )
    report = _run(_DAY, paths, decision_bar_max_missing_fraction=0.05, listing_root=listing_dir)
    assert report.decision_bar_missing == 0
    assert report.written is True
