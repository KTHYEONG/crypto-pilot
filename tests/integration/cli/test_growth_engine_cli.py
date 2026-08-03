from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application.research.growth import evaluation as growth_evaluation_module
from src.cli.main import main as cli_main

_SYMBOLS = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT", "FFFUSDT")
_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _ms(index: pd.DatetimeIndex) -> pd.Series:
    return (index - _EPOCH) // pd.Timedelta("1ms")


def _write_fake_ohlcv_files(root: Path) -> None:
    """Write complete 1h parquet archives with near-flat, noisy prices.

    The price surface is deliberately nearly flat: the pre-registered momentum
    signal cannot produce a significant out-of-sample t-stat on it, so the
    falsification verdict deterministically fails and the engine must hold CASH.
    """
    start = pd.Timestamp("2020-01-01 00:00", tz="UTC")
    end = pd.Timestamp("2023-01-01 00:00", tz="UTC")
    hourly = pd.date_range(start, end, freq="1h", inclusive="left")
    n = len(hourly)
    rng = np.random.default_rng(0)
    directory = root / "futures" / "ohlcv" / "1h"
    directory.mkdir(parents=True, exist_ok=True)
    for i, symbol in enumerate(_SYMBOLS):
        noise = rng.normal(0.0, 0.0001, n)
        price = 100.0 * (1.0 + noise)
        df = pd.DataFrame({
            "timestamp": _ms(hourly),
            "open": price,
            "high": price * 1.001,
            "low": price * 0.999,
            "close": price,
            "volume": 100.0,
            "quote_vol": 1000.0 * (1.0 + i),
        })
        df.to_parquet(directory / f"{symbol}.parquet")


@pytest.fixture
def fake_growth_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    _write_fake_ohlcv_files(tmp_path)

    def _fake_ohlcv_path(symbol: str, timeframe: str) -> Path:
        return tmp_path / "futures" / "ohlcv" / "1h" / f"{symbol}.parquet"

    monkeypatch.setattr(growth_evaluation_module, "ohlcv_path", _fake_ohlcv_path)
    return tmp_path


@pytest.mark.slow
class TestGrowthEngineCli:
    # GEV2-13-NO-ALPHA-IS-CASH
    def test_no_alpha_is_flat_cash_and_cli_exits_cleanly(
        self,
        fake_growth_env: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        captured: list[growth_evaluation_module.GrowthEngineReport] = []
        original = growth_evaluation_module.run_growth_engine_evaluation

        def _spy(request):
            report = original(request)
            captured.append(report)
            return report

        monkeypatch.setattr(
            growth_evaluation_module, "run_growth_engine_evaluation", _spy,
        )

        with caplog.at_level(logging.INFO):
            cli_main([
                "research", "run", "portfolio", "growth",
                "--universe-size", "3",
                "--max-positions", "3",
                "--start", "2020-01-01",
                "--no-log-run",
            ])

        assert len(captured) == 1
        report = captured[0]
        assert report.status == "NO_ADMISSIBLE_ALPHA"
        assert report.start == pd.Timestamp("2021-01-01", tz="UTC")
        assert len(report.trades) == 0
        assert len(report.equity) > 0
        assert float(report.equity.iloc[0]) == 10_000.0
        assert np.allclose(report.equity.to_numpy(), 10_000.0)
        assert report.promotion is None
        assert report.record is None
        assert report.falsification is not None
        assert report.falsification.passed is False

        assert "status=NO_ADMISSIBLE_ALPHA" in caplog.text

    def test_run_logs_growth_engine_status(self, fake_growth_env: Path) -> None:
        # The CLI must exit 0 (no exception) and reach the growth handler.
        cli_main([
            "research", "run", "portfolio", "growth",
            "--universe-size", "3",
            "--max-positions", "3",
            "--start", "2020-01-01",
            "--no-log-run",
        ])
