from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application.research.mhs import evaluation as ev
from src.application.research.mhs.evaluation import MhsDiagnosticRequest, run_mhs_horizon_diagnostic
from src.cli.commands.research.mhs import add_mhs_commands
from src.common.errors import DataIntegrityError
from src.research.evaluation.policy import HOLDOUT_CUTOFF
from src.research.universe.pit_universe import symbol_partition

START = pd.Timestamp("2021-01-01", tz="UTC")
N_HOURS = 2000
DEV_SYMBOLS = [
    sym for sym in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
    if symbol_partition(sym) == "dev"
][:8]


def _write_mhs_market(root: Path, symbols: list[str]) -> pd.Timestamp:
    hourly = pd.date_range(START, periods=N_HOURS, freq="1h", tz="UTC")
    end = hourly[-1]
    rng = np.random.default_rng(20260807)
    epoch = (hourly - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")

    hour_dir = root / "1h"
    minute_dir = root / "1m"
    funding_dir = root / "funding"
    mark_dir = root / "markPriceKlines" / "1h"
    hour_dir.mkdir(parents=True, exist_ok=True)
    minute_dir.mkdir(parents=True, exist_ok=True)
    funding_dir.mkdir(parents=True, exist_ok=True)
    mark_dir.mkdir(parents=True, exist_ok=True)

    n = len(hourly)
    minute_idx = pd.date_range(START, end, freq="1min", tz="UTC")
    minute_epoch = (minute_idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
    n_min = len(minute_idx)

    for i, sym in enumerate(symbols):
        drift = 1e-5 * (i - len(symbols) / 2.0)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.002, n)))
        pd.DataFrame(
            {
                "timestamp": epoch,
                "open": prices,
                "high": prices * 1.001,
                "low": prices * 0.999,
                "close": prices,
                "quote_vol": [1000.0] * n,
            },
        ).to_parquet(hour_dir / f"{sym}.parquet")

        minute_prices = 100.0 * np.exp(
            np.cumsum(rng.normal(drift, 0.002, n_min)),
        )
        pd.DataFrame(
            {
                "timestamp": minute_epoch,
                "open": minute_prices,
                "high": minute_prices * 1.0005,
                "low": minute_prices * 0.9995,
                "close": minute_prices,
                "quote_vol": [1000.0] * n_min,
            },
        ).to_parquet(minute_dir / f"{sym}.parquet")

        pd.DataFrame(
            {
                "timestamp": epoch,
                "funding_rate": [0.00005] * n,
                "datetime": hourly,
            },
        ).to_parquet(funding_dir / f"{sym}.parquet")

        mark_hourly = (
            pd.Series(minute_prices, index=minute_idx)
            .resample("1h")
            .last()
            .reindex(hourly)
            .to_numpy()
        )
        pd.DataFrame(
            {
                "timestamp": epoch,
                "open": mark_hourly,
                "high": mark_hourly,
                "low": mark_hourly,
                "close": mark_hourly,
                "datetime": hourly,
            },
        ).to_parquet(mark_dir / f"{sym}.parquet")
    return end


@pytest.fixture(scope="module")
def synthetic_market(tmp_path_factory) -> tuple[Path, pd.Timestamp]:
    import src.market_data.services.futures_collection as fc

    root = tmp_path_factory.mktemp("mhs_market")
    end = _write_mhs_market(root, DEV_SYMBOLS)
    originals = {
        "funding_path": ev.funding_path,
        "mark_price_path": fc._mark_price_path,
        "_BOOTSTRAP_REPLICATES": ev._BOOTSTRAP_REPLICATES,
        "_BOOTSTRAP_MEAN_BLOCK": ev._BOOTSTRAP_MEAN_BLOCK,
        "_BOOTSTRAP_SEED": ev._BOOTSTRAP_SEED,
    }
    ev.funding_path = lambda sym: root / "funding" / f"{sym}.parquet"
    fc._mark_price_path = (
        lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet"
    )
    ev._BOOTSTRAP_REPLICATES = 20
    ev._BOOTSTRAP_MEAN_BLOCK = 24
    ev._BOOTSTRAP_SEED = 20260807
    yield root, end
    for name, value in originals.items():
        if name == "mark_price_path":
            fc._mark_price_path = value
        else:
            setattr(ev, name, value)


@pytest.fixture(scope="module")
def report(synthetic_market):
    root, end = synthetic_market
    return run_mhs_horizon_diagnostic(
        MhsDiagnosticRequest(start=str(START), end=str(end), data_root=str(root), execution_timeframe="1m", log_run=False),
    )


class TestMhsHorizonDiagnostic:
    """MHS-10-DIAGNOSTIC-HOLDOUT-SEALED: dev-only diagnostic on a synthetic panel."""

    def test_produces_frozen_books_and_separate_evidence_paths(self, report) -> None:
        assert report.status == "COMPLETE"
        assert set(report.books) == {"fast_reversal", "slow_momentum"}
        assert report.blend is not None
        assert report.eligible_symbols >= 8
        assert report.trials_attempted > 0
        assert report.execution_tiers_bps == pytest.approx((2.64, 4.18, 6.07))
        assert report.blend_target_gross > 0.0

    def test_mhs_5m_01_default_execution_timeframe(self) -> None:
        """MHS-5M-01-DEFAULT: production requests default to 5m."""
        from src.application.research.mhs.evaluation import MhsDiagnosticRequest

        assert MhsDiagnosticRequest().execution_timeframe == "5m"

    def test_diagnostic_ensemble_separate_from_executable_tranche(self, report) -> None:
        fast = report.books["fast_reversal"]
        assert fast.phase.n_phases > 0
        assert fast.primary.ledger is not None
        assert len(fast.primary.ledger.equity) > 0
        for tier in (2.64, 4.18, 6.07):
            assert tier in fast.prescreen

    def test_holdout_partition_raises(self, synthetic_market) -> None:
        root, end = synthetic_market
        with pytest.raises(RuntimeError):
            run_mhs_horizon_diagnostic(
                MhsDiagnosticRequest(
                    start=str(START), end=str(end), data_root=str(root),
                    partition="holdout", execution_timeframe="1m", log_run=False,
                ),
            )

    def test_end_past_cutoff_raises(self, synthetic_market) -> None:
        root, end = synthetic_market
        with pytest.raises(RuntimeError):
            run_mhs_horizon_diagnostic(
                MhsDiagnosticRequest(
                    start=str(START),
                    end=str(HOLDOUT_CUTOFF + pd.Timedelta(days=1)),
                    data_root=str(root),
                    execution_timeframe="1m", log_run=False,
                ),
            )


class TestStrictSimulatedPrimary:
    """MHS-19-STRICT-SIMULATED-PRIMARY: strict proxy is the only primary evidence."""

    def test_prescreen_and_primary_are_separate(self, report) -> None:
        blend = report.blend
        assert blend is not None
        assert 4.18 in blend.prescreen
        assert 6.07 in blend.prescreen
        assert blend.primary.fill_source == "OHLCV_STRICT_PROXY"
        assert blend.primary.ledger.mark_source == "MARK_PRICE"
        assert blend.primary_naive_sharpe is not None
        assert blend.primary_max_drawdown <= 0.0 or np.isnan(blend.primary_max_drawdown)
        assert blend.stress.fill_source == "OHLCV_IMMEDIATE_TAKER"
        assert report.fill_source == "OHLCV_STRICT_PROXY"
        assert blend.primary.simulated_fills is not None
        # The executable tranche actually traded through the strict proxy.
        assert len(blend.primary.simulated_fills) > 0

    def test_phase_and_tail_diagnostics_reported(self, report) -> None:
        fast = report.books["fast_reversal"]
        assert fast.tail.event_window_bars > 0
        assert set(fast.tail.winsor_curve) == {10, 20, 30, 50}


class TestFreezeBeforeFinalOos:
    """MHS-24-FREEZE-BEFORE-FINAL-OOS: Phase 1 has no unseal path."""

    def test_cli_registers_no_unseal_flag(self) -> None:
        import argparse

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="portfolio_command", required=True)
        add_mhs_commands(sub)
        args = parser.parse_args(["mhs-horizon-diagnostic"])
        assert callable(args.handler)
        assert not hasattr(args, "unseal_holdout")

    def test_forward_observation_is_frozen(self) -> None:
        from src.mhs.execution import ForwardExecutionObservation

        obs = ForwardExecutionObservation(
            symbol="BTCUSDT", signal_time=pd.Timestamp("2026-01-01", tz="UTC"),
            intent_time=pd.Timestamp("2026-01-01", tz="UTC"),
            submit_time=None, fill_time=None, side=1,
            requested_quantity=0.1, filled_quantity=0.0,
            limit_price=None, fill_price=None, best_bid=None, best_ask=None,
            top_n_depth_notional=None, trade_print_notional=None,
            reject_reason=None, cancel_replace_count=0, latency_ms=None,
        )
        with pytest.raises(Exception, match="cannot assign"):
            obs.filled_quantity = 0.1

    def test_mhs_5m_03_signal_preservation(self, report) -> None:
        """MHS-5M-03-SIGNAL-PRESERVATION: signal and replay universes are reported separately."""
        assert isinstance(report.execution_symbols, tuple)
        assert report.execution_symbols


class TestMarkPriceCacheRequired:
    """MHS-MARK-03-CACHE-REQUIRED-INTEGRATION: a complete causal mark cache
    labels every replay MARK_PRICE and the report agrees across fast/slow/blend."""

    def test_fast_slow_blend_and_report_are_mark_price(self, report) -> None:
        assert report.mark_source == "MARK_PRICE"
        for book in (report.books["fast_reversal"], report.books["slow_momentum"], report.blend):
            assert book.primary.ledger.mark_source == "MARK_PRICE"
            assert book.primary.mark_source == "MARK_PRICE"
            assert book.stress.ledger.mark_source == "MARK_PRICE"
            assert book.stress.mark_source == "MARK_PRICE"

    def test_strict_and_stress_share_the_same_mark_source(self, report) -> None:
        assert report.mark_source == report.blend.primary.mark_source
        assert report.blend.primary.mark_source == report.blend.stress.mark_source


class TestNoSilentMarkFallback:
    """MHS-MARK-04-NO-SILENT-FALLBACK: a cache gap is never silently replaced by
    OHLCV closes under cache_required; explicit fallback stays labelled."""

    def _gapped_market(self, root: Path, tmp_path: Path, monkeypatch) -> None:
        import src.market_data.services.futures_collection as fc

        gap_dir = tmp_path / "markPriceKlines" / "1h"
        gap_dir.mkdir(parents=True, exist_ok=True)
        gap_start = pd.Timestamp("2021-02-01", tz="UTC")
        gap_end = pd.Timestamp("2021-02-10", tz="UTC")
        for sym in DEV_SYMBOLS:
            frame = pd.read_parquet(root / "markPriceKlines" / "1h" / f"{sym}.parquet")
            drop = (frame["datetime"] >= gap_start) & (frame["datetime"] < gap_end)
            frame.loc[drop, "close"] = float("nan")
            frame.loc[drop, "high"] = float("nan")
            frame.loc[drop, "low"] = float("nan")
            frame.loc[drop, "open"] = float("nan")
            frame.to_parquet(gap_dir / f"{sym}.parquet")
        monkeypatch.setattr(
            fc,
            "_mark_price_path",
            lambda symbol, timeframe: gap_dir / f"{symbol}.parquet",
        )

    def test_cache_required_raises_on_affected_gap(
        self, synthetic_market, tmp_path, monkeypatch,
    ) -> None:
        root, end = synthetic_market
        self._gapped_market(root, tmp_path, monkeypatch)
        with pytest.raises(DataIntegrityError, match="mark"):
            run_mhs_horizon_diagnostic(
                MhsDiagnosticRequest(
                    start=str(START), end=str(end), data_root=str(root),
                    mark_mode="cache_required", execution_timeframe="1m", log_run=False,
                ),
            )

    def test_explicit_ohlcv_fallback_completes_as_fallback(
        self, synthetic_market, tmp_path, monkeypatch,
    ) -> None:
        root, end = synthetic_market
        self._gapped_market(root, tmp_path, monkeypatch)
        report = run_mhs_horizon_diagnostic(
            MhsDiagnosticRequest(
                start=str(START), end=str(end), data_root=str(root),
                mark_mode="ohlcv_close_fallback", execution_timeframe="1m", log_run=False,
            ),
        )
        assert report.mark_source == "OHLCV_CLOSE_FALLBACK"
        assert report.blend is not None
        assert report.blend.primary.ledger.mark_source == "OHLCV_CLOSE_FALLBACK"


class TestMarkPriceGoValidityIntegration:
    """MHS-MARK-05-GO-VALIDITY: an invalid primary never yields a Research GO."""

    def test_research_go_requires_valid_primary(self) -> None:
        from src.mhs.evaluation import compute_deployment_readiness

        idx = pd.date_range("2025-01-01", periods=5, freq="1h", tz="UTC")
        equity = pd.Series(np.cumprod(1.0 + np.full(5, 0.001)), index=idx)
        invalid = compute_deployment_readiness(equity, 8760.0, primary_valid=False, n_bootstrap=2, mean_block_bars=1)
        assert invalid.research_go_eligible is False
        assert invalid.execution_go_eligible is False
        assert invalid.pilot_go_eligible is False
        assert invalid.scale_go_eligible is False


class TestMarkModeCli:
    """MHS-MARK-06-CLI-MODE: CLI defaults to cache_required, accepts only the
    two named modes, and exposes neither --collect-mark nor --unseal-holdout."""

    def test_cli_defaults_to_cache_required(self) -> None:
        import argparse

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="portfolio_command", required=True)
        add_mhs_commands(sub)
        args = parser.parse_args(["mhs-horizon-diagnostic"])
        assert args.mark_mode == "cache_required"
        assert not hasattr(args, "collect_mark")
        assert not hasattr(args, "unseal_holdout")

    def test_cli_accepts_only_two_named_modes(self) -> None:
        import argparse

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="portfolio_command", required=True)
        add_mhs_commands(sub)
        fallback = parser.parse_args(["mhs-horizon-diagnostic", "--mark-mode", "ohlcv_close_fallback"])
        assert fallback.mark_mode == "ohlcv_close_fallback"
        with pytest.raises(SystemExit):
            parser.parse_args(["mhs-horizon-diagnostic", "--mark-mode", "bogus"])
