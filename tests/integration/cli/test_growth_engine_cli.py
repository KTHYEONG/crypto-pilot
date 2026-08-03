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

    def _fake_funding_path(symbol: str) -> Path:
        return tmp_path / "futures" / "funding" / f"{symbol}.parquet"

    monkeypatch.setattr(growth_evaluation_module, "ohlcv_path", _fake_ohlcv_path)
    monkeypatch.setattr(growth_evaluation_module, "funding_path", _fake_funding_path)
    return tmp_path


_TAKER_SYMBOLS = ("BTCUSDT", "SOLUSDT", "XRPUSDT", "ETHUSDT", "BNBUSDT", "AVAXUSDT")
_TAKER_RATIO = {
    "BTCUSDT": 0.90, "SOLUSDT": 0.10, "XRPUSDT": 0.70,
    "ETHUSDT": 0.12, "BNBUSDT": 0.88, "AVAXUSDT": 0.35,
}
# Per-1h drift, divided by 6 inside the loop to express a per-hour rate.  The
# taker-imbalance signal (constant per symbol) separates long from short and is
# stable across every window, so taker_imbalance_v1 dominates discovery, its
# weights hold still (zero turnover after the first rebalance), and every gate
# passes deterministically.
_TAKER_DRIFT = {
    "BTCUSDT": 8e-4, "SOLUSDT": -8e-4, "XRPUSDT": 3e-4,
    "ETHUSDT": -8e-4, "BNBUSDT": 8e-4, "AVAXUSDT": -2e-4,
}


def _write_taker_sorted_ohlcv_files(root: Path) -> None:
    start = pd.Timestamp("2020-01-01 00:00", tz="UTC")
    end = pd.Timestamp("2023-01-01 00:00", tz="UTC")
    hourly = pd.date_range(start, end, freq="1h", inclusive="left")
    n = len(hourly)
    rng = np.random.default_rng(7)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    timestamp = (hourly - epoch) // pd.Timedelta("1ms")
    directory = root / "futures" / "ohlcv" / "1h"
    directory.mkdir(parents=True, exist_ok=True)
    for i, symbol in enumerate(_TAKER_SYMBOLS):
        eps = rng.normal(0.0, 0.00015, n)
        price = 100.0
        prices = np.empty(n)
        for t in range(n):
            price = price * (1.0 + _TAKER_DRIFT[symbol] / 6.0 + eps[t])
            prices[t] = price
        quote_vol = 1000.0 * (1.0 + i)
        df = pd.DataFrame({
            "timestamp": timestamp,
            "open": prices,
            "high": prices * 1.0005,
            "low": prices * 0.9995,
            "close": prices,
            "volume": 100.0,
            "quote_vol": quote_vol,
            "taker_buy_quote": quote_vol * _TAKER_RATIO[symbol],
        })
        df.to_parquet(directory / f"{symbol}.parquet")


@pytest.fixture
def fake_growth_env_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    _write_taker_sorted_ohlcv_files(tmp_path)

    def _fake_ohlcv_path(symbol: str, timeframe: str) -> Path:
        return tmp_path / "futures" / "ohlcv" / "1h" / f"{symbol}.parquet"

    def _fake_funding_path(symbol: str) -> Path:
        return tmp_path / "futures" / "funding" / f"{symbol}.parquet"

    monkeypatch.setattr(growth_evaluation_module, "ohlcv_path", _fake_ohlcv_path)
    monkeypatch.setattr(growth_evaluation_module, "funding_path", _fake_funding_path)
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
        assert report.selected_strategy is not None
        assert report.scorecard is not None
        assert report.scorecard.family_size == 12
        assert len(report.scorecard.entries) == 12

        assert "status=NO_ADMISSIBLE_ALPHA" in caplog.text

    # GSD-07: no passing finalist yields flat CASH plus a deterministic
    # candidate scorecard in the frozen registry order.
    def test_no_passing_finalist_publishes_deterministic_scorecard(
        self,
        fake_growth_env: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        report = growth_evaluation_module.run_growth_engine_evaluation(
            growth_evaluation_module.GrowthEngineEvaluationRequest(
                universe=growth_evaluation_module.PitUniverseSpec(
                    universe_size=3, max_positions=3,
                ),
                construction=growth_evaluation_module.NetConstructionSpec(),
                start="2020-01-01",
                symbol_scope="dev",
                log_run=False,
            )
        )
        assert report.status == "NO_ADMISSIBLE_ALPHA"
        assert report.scorecard is not None
        assert report.scorecard.reason == "no_passing_family"
        order = [
            (entry.strategy_id, entry.parameter) for entry in report.scorecard.entries
        ]
        assert order == [
            ("funding_contrarian_v1", 42),
            ("funding_contrarian_v1", 84),
            ("funding_contrarian_v1", 168),
            ("taker_imbalance_v1", 42),
            ("taker_imbalance_v1", 84),
            ("taker_imbalance_v1", 168),
            ("vol_adjusted_trend_v1", 42),
            ("vol_adjusted_trend_v1", 84),
            ("vol_adjusted_trend_v1", 180),
            ("donchian_channel_position_v1", 42),
            ("donchian_channel_position_v1", 84),
            ("donchian_channel_position_v1", 168),
        ]
        funding_entries = [e for e in report.scorecard.entries if e.strategy_id == "funding_contrarian_v1"]
        assert all(e.status == "DATA_INVALID" for e in funding_entries)
        assert all(e.dev_discovery_score is None for e in funding_entries)
        price_entries = [e for e in report.scorecard.entries if e.strategy_id != "funding_contrarian_v1"]
        assert all(e.status == "SCREENED" for e in price_entries)
        assert all(e.family_passed is False for e in report.scorecard.entries)

    def test_run_logs_growth_engine_status(self, fake_growth_env: Path) -> None:
        # The CLI must exit 0 (no exception) and reach the growth handler.
        cli_main([
            "research", "run", "portfolio", "growth",
            "--universe-size", "3",
            "--max-positions", "3",
            "--start", "2020-01-01",
            "--no-log-run",
        ])

    # GSD-06: family selection is dev-discovery only, passes multiplicity size
    # 12, and inspects the holdout only for the dev-selected finalist.  With a
    # stable cross-sectional taker-imbalance signal the dev finalist passes
    # every unchanged gate and the engine promotes it.
    def test_dev_discovery_selection_passes_and_promotes_finalist(
        self,
        fake_growth_env_pass: Path,
    ) -> None:
        report = growth_evaluation_module.run_growth_engine_evaluation(
            growth_evaluation_module.GrowthEngineEvaluationRequest(
                universe=growth_evaluation_module.PitUniverseSpec(
                    universe_size=6, max_positions=6,
                ),
                construction=growth_evaluation_module.NetConstructionSpec(
                    rebalance_bars=3, no_trade_band=0.0,
                ),
                start="2020-01-01",
                symbol_scope="dev",
                log_run=False,
            )
        )
        assert report.status == "PASS"
        assert report.selected_strategy == "taker_imbalance_v1"
        assert report.promotion is not None
        assert report.falsification is not None
        assert report.falsification.passed is True
        assert report.falsification.binding_constraint == "none"
        assert report.sizing is not None
        assert report.sizing.selected_risk is not None
        assert len(report.trades) == 0
        assert len(report.equity) > 0
        assert report.equity.dropna().iloc[-1] > report.equity.iloc[0]

        # The scorecard is complete (exactly twelve variants), deterministic,
        # and only the selected family was admitted by the discovery plateau.
        assert report.scorecard is not None
        assert report.scorecard.family_size == 12
        assert report.scorecard.reason is None
        assert report.scorecard.selected_strategy_id == "taker_imbalance_v1"
        assert report.scorecard.selected_parameter == 42
        assert len(report.scorecard.entries) == 12
        taker_entries = [
            e for e in report.scorecard.entries if e.strategy_id == "taker_imbalance_v1"
        ]
        assert all(e.family_passed for e in taker_entries)
        assert all(e.status == "SCREENED" for e in taker_entries)

        # Family-size 12 is threaded into the unchanged Bonferroni multiplicity
        # correction used by the falsification verdict.
        from src.research.evaluation.falsification import multiplicity_adjusted_t_floor
        assert abs(
            report.falsification.required_t_floor
            - multiplicity_adjusted_t_floor(12, 2.0)
        ) < 1e-9
