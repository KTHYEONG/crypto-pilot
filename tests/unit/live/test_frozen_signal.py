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
    funding_dir = root / "funding"
    funding_dir.mkdir(parents=True, exist_ok=True)
    fidx = pd.DatetimeIndex([grid[0] + pd.Timedelta(hours=h) for h in range(0, n, 8)], tz="UTC")
    pd.DataFrame(
        {"datetime": fidx, "funding_rate": np.full(len(fidx), 1e-9)},
    ).to_parquet(funding_dir / f"{_SYMS[0]}.parquet")


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
        entry_closes=pd.DataFrame(
            [[1.0]], index=pd.DatetimeIndex(["2021-01-03"], tz="UTC"), columns=cols, dtype="float64",
        ),
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
