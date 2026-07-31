from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application import cash_carry_evaluation as app
import src.research.cash_carry.market_data as carry
from src.cli import run_cash_carry_backtest as cli
from src.cli.run_backtest import HOLDOUT_CUTOFF
from src.research.provenance import anti_patterns as rm

_CUTOFF = HOLDOUT_CUTOFF

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _ms(index: pd.DatetimeIndex) -> pd.Series:
    return (index - _EPOCH) // pd.Timedelta("1ms")


def _write_fake_carry_files(root: Path) -> None:
    """Write complete fake spot/perp/funding/borrow parquet inputs.

    Files extend far past the sealed holdout cutoff (through 2026-07-01) so a
    sealing regression would be observable. The borrow export uses the
    canonical ``timestamp``/``borrow_rate``/``accrual_seconds`` columns.
    """
    start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    end = pd.Timestamp("2026-07-01 00:00", tz="UTC")
    hourly = pd.date_range(start, end, freq="1h", inclusive="left")
    n = len(hourly)
    price = 30000.0 + 0.5 * np.arange(n, dtype=np.float64)
    df = pd.DataFrame({
        "timestamp": _ms(hourly),
        "open": price,
        "high": price + 20.0,
        "low": price - 20.0,
        "close": price + 10.0,
        "volume": 100.0,
    })
    spot_dir = root / "spot" / "ohlcv" / "1h"
    perp_dir = root / "futures" / "ohlcv" / "1h"
    fund_dir = root / "futures" / "funding"
    borrow_dir = root / "spot" / "borrow"
    for directory in (spot_dir, perp_dir, fund_dir, borrow_dir):
        directory.mkdir(parents=True, exist_ok=True)
    df.to_parquet(spot_dir / "BTCUSDT.parquet")
    df.to_parquet(perp_dir / "BTCUSDT.parquet")

    eight_h = pd.date_range(start, end, freq="8h", inclusive="left")
    pd.DataFrame({
        "datetime": eight_h.tz_localize(None).strftime("%Y-%m-%d %H:%M:%S"),
        "funding_rate": 0.0002,
    }).to_parquet(fund_dir / "BTCUSDT.parquet")

    four_h = pd.date_range(start, end, freq="4h", inclusive="left")
    pd.DataFrame({
        "timestamp": _ms(four_h),
        "borrow_rate": 0.0,
        "accrual_seconds": 14400.0,
    }).to_parquet(borrow_dir / "BTCUSDT.parquet")


@pytest.fixture
def fake_carry_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    _write_fake_carry_files(tmp_path)
    anti_path = tmp_path / "registry" / "anti_patterns.json"

    def _spot(symbol: str, timeframe: str) -> Path:
        return tmp_path / "spot" / "ohlcv" / timeframe / f"{symbol.replace('/', '_')}.parquet"

    def _perp(symbol: str, timeframe: str) -> Path:
        return tmp_path / "futures" / "ohlcv" / timeframe / f"{symbol.replace('/', '_')}.parquet"

    def _fund(symbol: str) -> Path:
        return tmp_path / "futures" / "funding" / f"{symbol.replace('/', '_')}.parquet"

    def _borrow(symbol: str) -> Path:
        return tmp_path / "spot" / "borrow" / f"{symbol.replace('/', '_')}.parquet"

    for module in (carry,):
        monkeypatch.setattr(module, "spot_ohlcv_path", _spot)
        monkeypatch.setattr(module, "ohlcv_path", _perp)
        monkeypatch.setattr(module, "funding_path", _fund)
        monkeypatch.setattr(module, "borrow_path", _borrow)

    monkeypatch.setattr(app, "load_spot_manifest", lambda: {"schema_version": 1, "datasets": {}})
    monkeypatch.setattr(
        app, "record_rejected_candidate",
        lambda **kwargs: rm.record_rejected_candidate(
            anti_patterns_path=anti_path, **kwargs,
        ),
    )
    return anti_path


@pytest.mark.slow
class TestCashCarryCli:
    def test_run_uses_ephemeral_candidate_identity(
        self,
        fake_carry_env: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(sys, "argv", [
            "run_cash_carry_backtest", "run", "--start", "2024-01-01", "--no-log-run",
        ])
        with caplog.at_level(logging.INFO):
            cli.main()
        assert "candidate identity" in caplog.text
        assert "status=EPHEMERAL" in caplog.text
        assert "carry data status=PASS" in caplog.text

    def test_run_pending_when_borrow_absent(
        self,
        fake_carry_env: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        borrow_file = fake_carry_env.parent.parent / "spot" / "borrow" / "BTCUSDT.parquet"
        borrow_file.unlink()
        monkeypatch.setattr(sys, "argv", [
            "run_cash_carry_backtest", "run", "--start", "2024-01-01", "--no-log-run",
        ])
        with caplog.at_level(logging.INFO):
            cli.main()
        assert "run status=PENDING" in caplog.text
        assert "borrow data missing" in caplog.text

    def test_sealed_run_never_loads_data_after_holdout_cutoff(
        self,
        fake_carry_env: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        ends_seen: list[object] = []
        original_load = carry.load_ohlcv_1h_as_4h

        def spy_load(path, *, start=None, end=None):
            ends_seen.append(end)
            return original_load(path, start=start, end=end)

        funding_max: list[pd.Timestamp] = []
        original_validate = carry.validate_carry_market_data

        def spy_validate(data):
            funding_max.append(data.funding.index.max())
            return original_validate(data)

        monkeypatch.setattr(carry, "load_ohlcv_1h_as_4h", spy_load)
        monkeypatch.setattr(carry, "validate_carry_market_data", spy_validate)
        monkeypatch.setattr(sys, "argv", [
            "run_cash_carry_backtest", "run", "--start", "2024-01-01", "--no-log-run",
        ])
        with caplog.at_level(logging.INFO):
            cli.main()

        assert "carry data status=PASS" in caplog.text
        assert "[EVAL] reliability observation=" in caplog.text
        assert "[EVAL] reliability fold max_period_contribution=" in caplog.text
        assert "[EVAL] reliability stress_test=" in caplog.text
        assert "[EVAL] promotion status=" in caplog.text
        assert "[EVAL] holdout unsealed" not in caplog.text
        assert "[EVAL] reliability holdout=" not in caplog.text
        assert ends_seen, "load_ohlcv_1h_as_4h was never called"
        assert _CUTOFF in ends_seen
        assert all(e is None or e <= _CUTOFF for e in ends_seen)
        assert funding_max, "carry data was never validated"
        assert all(mx <= _CUTOFF for mx in funding_max)

    def test_rejected_run_records_anti_pattern_once(
        self,
        fake_carry_env: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        anti_path = fake_carry_env
        monkeypatch.setattr(sys, "argv", [
            "run_cash_carry_backtest", "run", "--start", "2024-01-01", "--no-log-run",
        ])
        with caplog.at_level(logging.INFO):
            cli.main()
            cli.main()
        anti = json.loads(anti_path.read_text(encoding="utf-8"))
        assert len(anti) == 1
