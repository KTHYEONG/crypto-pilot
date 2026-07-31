from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cli import run_cash_carry_backtest as cli
from src.cli.run_backtest import HOLDOUT_CUTOFF

_CUTOFF = HOLDOUT_CUTOFF


def _write_fake_carry_files(root: Path) -> None:
    """Write complete fake spot/perp/funding/borrow parquet inputs.

    The generated files intentionally extend far past the sealed holdout cutoff
    (through 2026-06-30) so a sealing regression would be observable.
    """
    start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    end = pd.Timestamp("2026-07-01 00:00", tz="UTC")
    hourly = pd.date_range(start, end, freq="1h", inclusive="left")
    n = len(hourly)
    price = 30000.0 + 0.5 * np.arange(n, dtype=np.float64)
    ms = np.array([int(ts.value) // 1_000_000 for ts in hourly], dtype=np.int64)
    df = pd.DataFrame({
        "timestamp": ms,
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
        "datetime": four_h.tz_localize(None).strftime("%Y-%m-%d %H:%M:%S"),
        "borrow_rate": 0.0,
    }).to_parquet(borrow_dir / "BTCUSDT.parquet")


@pytest.fixture
def fake_carry_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _write_fake_carry_files(tmp_path)

    def _spot(symbol: str, timeframe: str) -> Path:
        return tmp_path / "spot" / "ohlcv" / timeframe / f"{symbol.replace('/', '_')}.parquet"

    def _perp(symbol: str, timeframe: str) -> Path:
        return tmp_path / "futures" / "ohlcv" / timeframe / f"{symbol.replace('/', '_')}.parquet"

    def _fund(symbol: str) -> Path:
        return tmp_path / "futures" / "funding" / f"{symbol.replace('/', '_')}.parquet"

    def _borrow(symbol: str) -> Path:
        return tmp_path / "spot" / "borrow" / f"{symbol.replace('/', '_')}.parquet"

    monkeypatch.setattr(cli, "spot_ohlcv_path", _spot)
    monkeypatch.setattr(cli, "ohlcv_path", _perp)
    monkeypatch.setattr(cli, "funding_path", _fund)
    monkeypatch.setattr(cli, "borrow_path", _borrow)
    return tmp_path


@pytest.mark.slow
class TestCashCarryCli:
    def test_sealed_carry_cli_never_loads_data_after_holdout_cutoff(
        self,
        fake_carry_files: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # SC-CARRY-CLI-01: with complete fake inputs and no holdout flag, the
        # sealed run loads nothing after HOLDOUT_CUTOFF yet still composes all
        # canonical gates.
        funding_used_max: list[pd.Timestamp] = []
        original_validate = cli.validate_carry_market_data

        def spy_validate(data):  # noqa: ANN001
            funding_used_max.append(data.funding.index.max())
            return original_validate(data)

        monkeypatch.setattr(cli, "validate_carry_market_data", spy_validate)

        ends_seen: list[pd.Timestamp | None] = []
        original_load = cli.load_ohlcv_4h

        def spy_load(path, *, start=None, end=None):  # noqa: ANN001, ANN201
            ends_seen.append(end)
            return original_load(path, start=start, end=end)

        monkeypatch.setattr(cli, "load_ohlcv_4h", spy_load)

        monkeypatch.setattr(sys, "argv", ["run_cash_carry_backtest", "--no-log-run"])
        with caplog.at_level(logging.INFO):
            cli.main()

        assert "carry data status=PASS" in caplog.text
        assert "[EVAL] reliability observation=" in caplog.text
        assert "[EVAL] reliability fold max_period_contribution=" in caplog.text
        assert "[EVAL] reliability stress_test=" in caplog.text
        assert "[EVAL] promotion status=" in caplog.text
        assert "[EVAL] holdout unsealed" not in caplog.text
        assert "[EVAL] reliability holdout=" not in caplog.text
        assert ends_seen, "load_ohlcv_4h was never called"
        assert _CUTOFF in ends_seen
        assert all(e is None or e <= _CUTOFF for e in ends_seen)
        assert funding_used_max, "carry data was never validated"
        assert all(mx <= _CUTOFF for mx in funding_used_max)

    def test_cli_reports_pending_before_performance_when_inputs_incomplete(
        self,
        fake_carry_files: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # SC-CARRY-DATA-02 at the CLI boundary: a missing spot file makes the
        # run report PENDING and exit before any performance calculation.
        (fake_carry_files / "spot" / "ohlcv" / "1h" / "BTCUSDT.parquet").unlink()
        monkeypatch.setattr(sys, "argv", ["run_cash_carry_backtest", "--no-log-run"])
        with caplog.at_level(logging.INFO):
            cli.main()

        assert "carry data status=PENDING" in caplog.text
        assert "reliability observation=" not in caplog.text
        assert "promotion status=" not in caplog.text
